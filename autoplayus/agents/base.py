import logging
import threading
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from autoplayus import config
from autoplayus.device.base import Device
from autoplayus.vision.utils import find_template

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ok_template() -> np.ndarray:
    return cv2.imread(str(config.COMMON_TEMPLATES_DIR / "ok_button.png"))


class FrameGrabber:
    """Capturas en paralelo: con N hilos llegan ~N veces más fotogramas por segundo (cada uno con algo más de latencia).

    `next()` devuelve siempre un fotograma más reciente que el anterior y descarta los que llegan desordenados.
    """

    def __init__(self, device: Device, workers: int):
        self._device = device
        self._cond = threading.Condition()
        self._latest: tuple[float, np.ndarray] | None = None  # (instante en que empezó la captura, imagen)
        self._returned = 0.0
        self._error: BaseException | None = None
        self._closed = False
        self._threads = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def _work(self) -> None:
        while not self._closed:
            started = time.time()
            try:
                frame = self._device.screenshot()
            except BaseException as e:  # noqa: BLE001 - se relanza en el hilo principal
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                return
            with self._cond:
                if self._latest is None or started > self._latest[0]:
                    self._latest = (started, frame)
                    self._cond.notify_all()

    def next(self, timeout: float = 5.0) -> tuple[float, np.ndarray]:
        """(instante en que empezó la captura, imagen) del fotograma más reciente aún no entregado."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._error is not None or (self._latest is not None and self._latest[0] > self._returned),
                timeout,
            )
            if self._error is not None:
                raise self._error
            if not ok:
                raise TimeoutError("Sin capturas nuevas")
            self._returned = self._latest[0]
            return self._latest

    def close(self) -> None:
        self._closed = True


class BaseAgent(ABC):
    """Bucle percibir -> decidir -> actuar. Las subclases implementan `step`."""

    def __init__(self, device: Device, params: dict[str, Any] | None = None):
        self.device = device
        self.params = params or {}
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    step_delay = 0.2  # pausa entre pasos; los juegos de reflejos la ponen a 0
    capture_threads = 1  # capturas en paralelo; los juegos de reflejos pueden subirlo (ver FrameGrabber)
    frame_time = 0.0  # instante en que empezó la captura del fotograma que recibe step(); un gesto anterior a él no se ve aún

    def run(self, max_steps: int = 500, step_delay: float | None = None) -> bool:
        """Devuelve True si la partida terminó (pantalla final), False si se paró o agotó los pasos.

        Flujo común: Start -> toque inicial -> bucle step() -> OK en la pantalla final.
        """
        self._stop.clear()
        self.on_start()
        self.start_game()
        # Capturar en paralelo solo compensa con la captura lenta por ADB; desde la ventana basta un hilo.
        workers = 1 if getattr(self.device, "fast_capture", False) else self.capture_threads
        grabber = FrameGrabber(self.device, workers) if workers > 1 else None
        try:
            for i in range(max_steps):
                if self._stop.is_set():
                    log.info("Agente detenido")
                    return False
                if grabber:
                    self.frame_time, frame = grabber.next()
                else:
                    self.frame_time = time.time()
                    frame = self.device.screenshot()
                if self.is_solved(frame):
                    log.info("Partida terminada en %d pasos", i)
                    self.device.tap(*config.UI_OK)
                    return True
                self.step(frame)
                time.sleep(self.step_delay if step_delay is None else step_delay)
            log.warning("Máximo de pasos alcanzado")
            return False
        finally:
            if grabber:
                grabber.close()

    def start_game(self) -> None:
        """Pulsa Start (común a todos los juegos) y el toque que arranca la partida."""
        self.device.tap(*config.UI_START)
        time.sleep(1.0)
        self.device.tap(*config.UI_FIRST_TAP)
        time.sleep(0.5)

    def on_start(self) -> None:
        """Hook opcional: preparar estado antes de pulsar Start (cargar plantillas, leer params)."""

    @abstractmethod
    def step(self, frame: np.ndarray) -> None:
        """Analiza `frame` y ejecuta una o varias acciones en el dispositivo."""

    def is_solved(self, frame: np.ndarray) -> bool:
        """Detecta la pantalla final (botón OK común a todos los juegos)."""
        return find_template(frame, _ok_template(), threshold=0.9) is not None
