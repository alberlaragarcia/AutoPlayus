import ctypes
import logging
import time
from ctypes import wintypes

import cv2
import numpy as np

from autoplayus import config
from autoplayus.device.adb import AdbDevice

log = logging.getLogger(__name__)

_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
_WM_MOUSEMOVE, _WM_LBUTTONDOWN, _WM_LBUTTONUP = 0x0200, 0x0201, 0x0202
_MK_LBUTTON = 0x0001
_PER_MONITOR_AWARE_V2 = -4
_EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _text(getter, hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    getter(hwnd, buf, 256)
    return buf.value


def _find_render_window(title_prefix: str) -> int | None:
    """Ventana hija de Qt ('HD-Player') donde BlueStacks pinta la pantalla del Android."""
    tops: list[int] = []
    _user32.EnumWindows(_EnumProc(lambda h, _: tops.append(h) or True), 0)
    for top in tops:
        if _user32.IsWindowVisible(top) and _text(_user32.GetWindowTextW, top).startswith(title_prefix):
            kids: list[int] = []
            _user32.EnumChildWindows(top, _EnumProc(lambda h, _: kids.append(h) or True), 0)
            for kid in kids:
                if _text(_user32.GetWindowTextW, kid) == "HD-Player":
                    return kid
    return None


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]  # fmt: skip


if _user32 is not None:
    _gdi32 = ctypes.windll.gdi32
    _user32.GetDC.restype = wintypes.HDC
    _user32.GetDC.argtypes = [wintypes.HWND]
    _user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    _gdi32.CreateCompatibleDC.restype = wintypes.HDC
    _gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    _gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    _gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    _gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    _gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]  # fmt: skip
    _gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                 ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]  # fmt: skip
    _gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    _gdi32.DeleteDC.argtypes = [wintypes.HDC]
_SRCCOPY = 0x00CC0020


def _grab_client(hwnd: int) -> np.ndarray | None:
    """Contenido de la ventana (BGR) copiado de su propio contexto: ~3 ms y no depende de lo que la tape.

    None si está minimizada o no se pudo copiar. La imagen tiene el tamaño de la ventana en pantalla.
    """
    previous = _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
    try:
        rect = wintypes.RECT()
        _user32.GetClientRect(hwnd, ctypes.byref(rect))
        w, h = rect.right, rect.bottom
        if w == 0 or h == 0:
            return None
        hdc = _user32.GetDC(hwnd)
        mdc = _gdi32.CreateCompatibleDC(hdc)
        bmp = _gdi32.CreateCompatibleBitmap(hdc, w, h)
        try:
            _gdi32.SelectObject(mdc, bmp)
            _gdi32.BitBlt(mdc, 0, 0, w, h, hdc, 0, 0, _SRCCOPY)
            header = _BitmapInfoHeader(ctypes.sizeof(_BitmapInfoHeader), w, -h, 1, 32, 0)  # alto negativo: de arriba abajo
            buf = ctypes.create_string_buffer(w * h * 4)
            _gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(header), 0)
        finally:
            _gdi32.DeleteObject(bmp)
            _gdi32.DeleteDC(mdc)
            _user32.ReleaseDC(hwnd, hdc)
        return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    finally:
        _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))


class BlueStacksDevice(AdbDevice):
    """Captura y toques directos sobre la ventana de BlueStacks (rápidos); ADB solo como alternativa.

    - Captura: copia el contenido de la ventana (~3 ms) en vez de `adb screencap` (~145-245 ms), y la devuelve
      escalada al tamaño del Android (720x1280 en vertical) para que los agentes no dependan del tamaño de la
      ventana. Si la ventana está minimizada o no se encuentra, usa ADB.
    - Toques y deslizamientos: clics enviados a la ventana (`adb shell input tap` tarda ~0,5 s). No mueven el ratón.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hwnd: int | None = None
        self._warned_capture = False

    @property
    def fast_capture(self) -> bool:
        """True si ahora se puede capturar desde la ventana (los agentes no necesitan capturar en paralelo)."""
        hwnd = self._window() if config.WINDOW_CAPTURE else None
        if hwnd is None:
            return False
        rect = wintypes.RECT()
        _user32.GetClientRect(hwnd, ctypes.byref(rect))
        return rect.right > 0 and rect.bottom > 0

    def screenshot(self) -> np.ndarray:
        if config.WINDOW_CAPTURE and (hwnd := self._window()) is not None:
            raw = _grab_client(hwnd)
            if raw is not None and raw.max() > 0:
                h, w = raw.shape[:2]
                size = (720, 1280) if h >= w else (1280, 720)  # la orientación cambia según la app
                self.last_size = size
                return cv2.resize(raw, size, interpolation=cv2.INTER_LINEAR)
            if not self._warned_capture:
                log.warning("No se pudo capturar la ventana de BlueStacks (¿minimizada?): captura por ADB (lenta)")
                self._warned_capture = True
        return super().screenshot()

    def _window(self) -> int | None:
        if _user32 is None:
            return None
        if self._hwnd is None or not _user32.IsWindow(self._hwnd):
            self._hwnd = _find_render_window(config.BLUESTACKS_WINDOW)
            if self._hwnd is None:
                log.warning("Ventana de BlueStacks no encontrada: toques por ADB (lentos)")
        return self._hwnd

    def _to_client(self, hwnd: int, x: int, y: int) -> int | None:
        """Punto de la pantalla del Android (coordenadas de la captura) -> lparam de la ventana; None si está minimizada."""
        rect = wintypes.RECT()
        _user32.GetClientRect(hwnd, ctypes.byref(rect))
        if rect.right == 0 or rect.bottom == 0:
            return None
        w, h = self.last_size
        cx, cy = int(x / w * rect.right), int(y / h * rect.bottom)
        return (cy << 16) | (cx & 0xFFFF)

    def tap(self, x: int, y: int) -> None:
        hwnd = self._window()
        if hwnd is None:
            return super().tap(x, y)
        previous = _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
        try:
            lparam = self._to_client(hwnd, x, y)
            if lparam is None:  # ventana minimizada
                return super().tap(x, y)
            _user32.PostMessageW(hwnd, _WM_LBUTTONDOWN, _MK_LBUTTON, lparam)
            _user32.PostMessageW(hwnd, _WM_LBUTTONUP, 0, lparam)
        finally:
            _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        """Pulsar, mover por puntos intermedios y soltar. Con `duration_ms` pequeño es un gesto rápido."""
        hwnd = self._window()
        if hwnd is None:
            return super().swipe(x1, y1, x2, y2, duration_ms)
        previous = _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
        try:
            start, end = self._to_client(hwnd, x1, y1), self._to_client(hwnd, x2, y2)
            if start is None or end is None:
                return super().swipe(x1, y1, x2, y2, duration_ms)
            steps = max(2, duration_ms // 8)
            _user32.PostMessageW(hwnd, _WM_LBUTTONDOWN, _MK_LBUTTON, start)
            for i in range(1, steps + 1):
                px = int(x1 + (x2 - x1) * i / steps)
                py = int(y1 + (y2 - y1) * i / steps)
                _user32.PostMessageW(hwnd, _WM_MOUSEMOVE, _MK_LBUTTON, self._to_client(hwnd, px, py))
                time.sleep(duration_ms / 1000 / steps)
            _user32.PostMessageW(hwnd, _WM_LBUTTONUP, 0, end)
        finally:
            _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
