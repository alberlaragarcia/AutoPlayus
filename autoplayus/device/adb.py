import subprocess

import cv2
import numpy as np

from autoplayus import config
from autoplayus.device.base import Device


class AdbDevice(Device):
    def __init__(self, serial: str | None = config.ADB_SERIAL, adb_path: str = config.ADB_PATH):
        self._base = [adb_path] + (["-s", serial] if serial else [])
        self.last_size: tuple[int, int] = config.SCREEN_SIZE  # (ancho, alto) de la última captura
        if serial and ":" in serial:  # emulador por red (BlueStacks): hay que conectarlo
            subprocess.run([adb_path, "connect", serial], capture_output=True)

    def _run(self, *args: str) -> bytes:
        result = subprocess.run([*self._base, *args], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"adb {' '.join(args)} falló: {result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def screenshot(self) -> np.ndarray:
        png = self._run("exec-out", "screencap", "-p")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("No se pudo decodificar la captura de ADB")
        self.last_size = (img.shape[1], img.shape[0])  # la orientación cambia según la app (inicio: horizontal)
        return img

    def screen_size(self) -> tuple[int, int]:
        # `wm size` ignora la rotación actual (BlueStacks lo da como 1280x720 aunque la app va en vertical),
        # así que usamos la captura, que sí refleja las coordenadas que aceptan tap/swipe.
        h, w = self.screenshot().shape[:2]
        return w, h

    def tap(self, x: int, y: int) -> None:
        # Tarda ~0,5 s en aplicarse (arranca Java en Android): los juegos de reflejos deben anticiparlo.
        # `sendevent` sobre el táctil de BlueStacks no tiene efecto, y una shell persistente no lo acelera.
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms))
