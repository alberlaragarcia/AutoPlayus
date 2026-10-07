import ctypes
import logging
from ctypes import wintypes

from autoplayus import config
from autoplayus.device.adb import AdbDevice

log = logging.getLogger(__name__)

_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
_WM_LBUTTONDOWN, _WM_LBUTTONUP = 0x0201, 0x0202
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


class BlueStacksDevice(AdbDevice):
    """Captura por ADB y toques como clics de Windows sobre la ventana de BlueStacks.

    `adb shell input tap` tarda ~0,5 s; un clic enviado a la ventana es casi instantáneo, no mueve el
    ratón y no necesita que la ventana esté maximizada (sí visible, no minimizada). Si no se encuentra
    la ventana, se usa el toque por ADB.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hwnd: int | None = None

    def _window(self) -> int | None:
        if _user32 is None:
            return None
        if self._hwnd is None or not _user32.IsWindow(self._hwnd):
            self._hwnd = _find_render_window(config.BLUESTACKS_WINDOW)
            if self._hwnd is None:
                log.warning("Ventana de BlueStacks no encontrada: toques por ADB (lentos)")
        return self._hwnd

    def tap(self, x: int, y: int) -> None:
        hwnd = self._window()
        if hwnd is None:
            return super().tap(x, y)
        previous = _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
        try:
            rect = wintypes.RECT()
            _user32.GetClientRect(hwnd, ctypes.byref(rect))
            if rect.right == 0 or rect.bottom == 0:  # ventana minimizada
                return super().tap(x, y)
            w, h = config.SCREEN_SIZE
            cx, cy = int(x / w * rect.right), int(y / h * rect.bottom)
            lparam = (cy << 16) | (cx & 0xFFFF)
            _user32.PostMessageW(hwnd, _WM_LBUTTONDOWN, 1, lparam)
            _user32.PostMessageW(hwnd, _WM_LBUTTONUP, 0, lparam)
        finally:
            _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
