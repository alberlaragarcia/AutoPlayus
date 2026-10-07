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


class BaseAgent(ABC):
    """Bucle percibir -> decidir -> actuar. Las subclases implementan `step`."""

    def __init__(self, device: Device, params: dict[str, Any] | None = None):
        self.device = device
        self.params = params or {}
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    step_delay = 0.2  # pausa entre pasos; los juegos de reflejos la ponen a 0

    def run(self, max_steps: int = 500, step_delay: float | None = None) -> bool:
        """Devuelve True si la partida terminó (pantalla final), False si se paró o agotó los pasos.

        Flujo común: Start -> toque inicial -> bucle step() -> OK en la pantalla final.
        """
        self._stop.clear()
        self.on_start()
        self.start_game()
        for i in range(max_steps):
            if self._stop.is_set():
                log.info("Agente detenido")
                return False
            frame = self.device.screenshot()
            if self.is_solved(frame):
                log.info("Partida terminada en %d pasos", i)
                self.device.tap(*config.UI_OK)
                return True
            self.step(frame)
            time.sleep(self.step_delay if step_delay is None else step_delay)
        log.warning("Máximo de pasos alcanzado")
        return False

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
