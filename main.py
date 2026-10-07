import argparse
import logging

from autoplayus.config import SCREENSHOTS_DIR
from autoplayus.device import create_device


def check_device() -> None:
    device = create_device()
    w, h = device.screen_size()
    print(f"Dispositivo conectado: {w}x{h}")
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOTS_DIR / "check.png"
    import cv2

    cv2.imwrite(str(path), device.screenshot())
    print(f"Captura guardada en {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoPlayus")
    parser.add_argument("--check-device", action="store_true", help="Comprueba ADB y guarda una captura")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.check_device:
        check_device()
    else:
        from autoplayus.app.gui import run_app

        run_app()


if __name__ == "__main__":
    main()
