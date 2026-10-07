"""Ejecuta un agente registrando capturas y gestos con su hora, para analizar después qué pasó.

Uso:  python tools/run_logged.py <id_juego> <carpeta_salida>
Guarda <carpeta_salida>/NNNN.jpg (cada captura que recibió el agente) y events.csv (idx de captura y gestos).
Las capturas se codifican en memoria y se escriben al terminar, para no retrasar el bucle del agente.
"""

import logging
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoplayus.device import create_device  # noqa: E402
from autoplayus.games.registry import load_catalog  # noqa: E402


def main() -> None:
    game_id, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(relativeCreated)7d %(message)s")

    device = create_device()
    frames: list[bytes] = []
    events: list[tuple[float, str, str]] = []
    t0 = time.time()

    shot, swipe, tap = device.screenshot, device.swipe, device.tap

    lock = threading.Lock()  # el agente puede capturar desde varios hilos

    last_saved = [0.0]

    def rec_shot():
        frame = shot()
        now = time.time()
        if now - last_saved[0] >= 0.05:  # con captura por ventana llegan >100/s: guardar solo ~20/s
            last_saved[0] = now
            data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            with lock:
                events.append((now - t0, "frame", str(len(frames))))
                frames.append(data)
        return frame

    def rec_swipe(*a):
        events.append((time.time() - t0, "swipe", str(a)))
        swipe(*a)

    def rec_tap(*a):
        events.append((time.time() - t0, "tap", str(a)))
        tap(*a)

    device.screenshot, device.swipe, device.tap = rec_shot, rec_swipe, rec_tap

    agent = load_catalog()[game_id].build_agent(device)
    result = agent.run(max_steps=3000)
    print("resultado:", result, "en", round(time.time() - t0, 1), "s;", len(frames), "capturas")

    for i, data in enumerate(frames):
        (out / f"{i:04d}.jpg").write_bytes(data)
    with open(out / "events.csv", "w") as f:
        f.write("t,kind,detail\n")
        for t, kind, detail in events:
            f.write(f'{t:.3f},{kind},"{detail}"\n')


if __name__ == "__main__":
    main()
