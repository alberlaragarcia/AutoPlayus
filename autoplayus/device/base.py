from abc import ABC, abstractmethod

import numpy as np


class Device(ABC):
    """Abstracción del móvil: lo único que los agentes conocen del hardware."""

    @abstractmethod
    def screenshot(self) -> np.ndarray:
        """Devuelve la pantalla como imagen BGR (alto x ancho x 3)."""

    @abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """(ancho, alto) en píxeles."""

    @abstractmethod
    def tap(self, x: int, y: int) -> None: ...

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None: ...

    def drag_path(self, points: list[tuple[int, int]], step_ms: int = 60) -> None:
        """Arrastra por una secuencia de puntos (laberintos, conexiones). Por defecto, swipes encadenados."""
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            self.swipe(x1, y1, x2, y2, step_ms)
