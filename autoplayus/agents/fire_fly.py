import logging
import threading
import time

import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.base import BaseAgent
from autoplayus.vision.utils import find_template

log = logging.getLogger(__name__)

# Rangos HSV (OpenCV) del cuerpo de cada luciérnaga.
YELLOW = (np.array([19, 128, 240]), np.array([23, 158, 255]))
BLUE = (np.array([82, 110, 207]), np.array([86, 140, 237]))


class FireFly(BaseAgent):
    """Las luciérnagas bajan por un rayo y en la bifurcación van a la flor del color del rayo.

    Cada toque cambia el rayo. Regla: la luciérnaga más baja que aún no ha llegado a la bifurcación
    debe tener el mismo color que el rayo; si no, se toca. El color del rayo se lee de la estrella
    de la bifurcación (amarillo = flor derecha, azul = flor izquierda).

    Start, toque inicial, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        game_over = config.TEMPLATES_DIR / "FireFly" / "game_over.png"
        self._game_over = cv2.imread(str(game_over)) if game_over.exists() else None
        self._expected: str | None = None  # color del rayo una vez aplicados los toques en vuelo
        self._last_tap = 0.0
        self._speed = float(self.params["speed"])  # px/s, se corrige en cada paso
        self._prev: tuple[float, list[tuple[int, str]]] | None = None

    def is_solved(self, frame: np.ndarray) -> bool:
        if super().is_solved(frame):
            return True
        # Pantalla "JUEGO TERMINADO" (menos de 10 puntos): opcional, solo si existe la plantilla
        return self._game_over is not None and find_template(frame, self._game_over, threshold=0.9) is not None

    def step(self, frame: np.ndarray) -> None:
        # Un toque tarda ~0,5 s en aplicarse, así que se decide con dónde estará cada luciérnaga entonces.
        now = time.time()
        if now - self._last_tap > self.params["settle"]:  # sin toques en vuelo: leer el rayo real
            self._expected = self.beam_color(frame) or self._expected
        if self._expected is None:
            return
        fork_y, margin = self.params["fork_y"], self.params["margin"]
        flies = self.find_fireflies(frame)
        self._update_speed(now, flies)
        reach = self._speed * self.params["lead"]  # px que bajará una luciérnaga hasta que el toque surta efecto
        landing = [(y + reach, color) for y, color in flies]
        if any(abs(y - fork_y) <= margin for y, _ in landing):
            return  # estaría cruzando la bifurcación justo cuando se aplique el toque: no tocar
        pending = [f for f in landing if f[0] < fork_y - margin]  # las que ya habrán cruzado no importan
        if not pending:
            return
        _, color = max(pending)  # la más baja que aún no habrá llegado
        if color != self._expected:
            log.info("Próxima luciérnaga %s con rayo %s -> toque", color, self._expected)
            threading.Thread(target=self.device.tap, args=tuple(self.params["tap"]), daemon=True).start()
            self._expected = color
            self._last_tap = now

    def _update_speed(self, now: float, flies: list[tuple[int, str]]) -> None:
        """La velocidad sube durante la partida (~260 -> ~500 px/s en 40 s): se mide entre capturas."""
        if self._prev is not None:
            prev_t, prev_flies = self._prev
            dt = now - prev_t
            speeds = []
            if 0.05 < dt < 0.6:
                for y0, c0 in prev_flies:
                    steps = [y1 - y0 for y1, c1 in flies if c1 == c0 and 3 < y1 - y0 < 1.5 * self._speed * dt + 20]
                    if steps:
                        speeds.append(min(steps) / dt)
            if speeds:
                self._speed = 0.7 * self._speed + 0.3 * float(np.median(speeds))
        self._prev = (now, flies)

    def find_fireflies(self, frame: np.ndarray) -> list[tuple[int, str]]:
        """Luciérnagas visibles en la franja del rayo como (y, color), ordenadas de arriba abajo."""
        x1, y1, x2, y2 = self.params["strip"]
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        found = []
        for color, (lo, hi) in (("yellow", YELLOW), ("blue", BLUE)):
            n, _, stats, centroids = cv2.connectedComponentsWithStats(cv2.inRange(hsv, lo, hi))
            for k in range(1, n):
                if stats[k][cv2.CC_STAT_AREA] >= self.params["min_area"]:
                    found.append((int(centroids[k][1]) + y1, color))
        return sorted(found)

    def beam_color(self, frame: np.ndarray) -> str | None:
        """Color del rayo según la estrella de la bifurcación ('yellow' = derecha, 'blue' = izquierda)."""
        x, y = self.params["star"]
        patch = cv2.cvtColor(frame[y - 8 : y + 8, x - 8 : x + 8], cv2.COLOR_BGR2HSV)
        hue = float(patch[..., 0].mean())
        if 15 <= hue <= 45:
            return "yellow"
        if 75 <= hue <= 100:
            return "blue"
        return None  # transición / animación
