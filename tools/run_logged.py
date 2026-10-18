"""Ejecuta un agente registrando capturas y gestos con su hora, para analizar después qué pasó.

Uso:  python tools/run_logged.py <id_juego> <carpeta_salida>
Guarda <carpeta_salida>/NNNN.jpg (cada captura que recibió el agente) y events.csv (idx de captura y gestos).
Para pararlo a mano: Ctrl+C o crear el archivo <carpeta_salida>/STOP. En ambos casos se guarda lo grabado.
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

    shot, swipe, tap, drag = device.screenshot, device.swipe, device.tap, device.drag_path

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

    def rec_drag(points, step_ms=60):
        events.append((time.time() - t0, "drag", str(points)))
        drag(points, step_ms)

    device.screenshot, device.swipe, device.tap, device.drag_path = rec_shot, rec_swipe, rec_tap, rec_drag

    agent = load_catalog()[game_id].build_agent(device)
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)
    done = threading.Event()

    def watch_stop():
        while not done.wait(0.2):
            if stop_file.exists():
                print("STOP encontrado: parando el agente")
                agent.stop()
                return

    threading.Thread(target=watch_stop, daemon=True).start()
    try:
        result = agent.run()
    except KeyboardInterrupt:
        agent.stop()
        result = "interrumpido (Ctrl+C)"
    finally:
        done.set()
    print("resultado:", result, "en", round(time.time() - t0, 1), "s;", len(frames), "capturas")

    for i, data in enumerate(frames):
        (out / f"{i:04d}.jpg").write_bytes(data)
    with open(out / "events.csv", "w") as f:
        f.write("t,kind,detail\n")
        for t, kind, detail in events:
            f.write(f'{t:.3f},{kind},"{detail}"\n')


if __name__ == "__main__":
    main()
