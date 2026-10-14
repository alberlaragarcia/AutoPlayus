"""Graba capturas con su hora mientras juegas tú (no pulsa nada).

Uso:  python tools/record_game.py <carpeta_salida> [serial_adb]
(sin serial se usa BlueStacks; con un móvil real, el serial que muestra `adb devices`)
Se detiene cuando se crea el archivo <carpeta_salida>/STOP (o con Ctrl+C).
Genera <carpeta_salida>/NNNN.png y frames.csv (índice, segundos desde el inicio de la captura).
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoplayus import config  # noqa: E402
from autoplayus.device.adb import AdbDevice  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    device = AdbDevice(sys.argv[2], config.PHONE_ADB_PATH) if len(sys.argv) > 2 else AdbDevice()
    t0 = time.time()
    i = 0
    with open(out / "frames.csv", "w") as csv:
        csv.write("idx,t\n")
        while not stop_file.exists():
            t = time.time() - t0
            cv2.imwrite(str(out / f"{i:04d}.png"), device.screenshot())
            csv.write(f"{i},{t:.3f}\n")
            csv.flush()
            i += 1
    print(f"{i} capturas en {out}")


if __name__ == "__main__":
    main()
