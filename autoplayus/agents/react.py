import logging
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)


def lit_circles(frame: np.ndarray, params: dict[str, Any]) -> list[int]:
    """Índices (0-8, por filas) de los círculos encendidos. Basta un parche en el centro de cada uno: apagado
    es gris azulado (BGR ~72,55,45) y encendido verde turquesa (~170,212,0).

    Solo se da por buena si los demás círculos se ven apagados: en la pantalla de resumen el texto verde del
    promedio cae sobre los centros de dos círculos y no hay cuadrícula.
    """
    r = params["patch"]
    off = np.array(params["off_bgr"])
    lit, n_off = [], 0
    for i, (x, y) in enumerate(params["circles"]):
        color = frame[y - r : y + r + 1, x - r : x + r + 1].reshape(-1, 3).mean(axis=0)
        if color[1] >= params["green_min"] and color[2] <= params["red_max"]:
            lit.append(i)
        elif np.abs(color - off).max() <= params["off_tol"]:
            n_off += 1
    return lit if n_off >= len(params["circles"]) - params["max_lit"] else []


class React(BaseAgent):
    """9 círculos en 3x3; uno se enciende en verde tras un tiempo aleatorio y hay que tocarlo cuanto antes.
    5 intentos; la puntuación es el tiempo medio de reacción (menos es mejor).

    En cada captura se miran solo los 9 centros y se toca el encendido. Una captura tomada justo antes de que
    el juego apague el círculo tocado aún lo muestra verde: no se vuelve a tocar un círculo hasta haberlo
    visto apagado en una captura posterior al toque (un toque de más podría contar como fallo).

    Start, toque del tutorial (círculo de ejemplo en 360,1030 = UI_FIRST_TAP), detección de fin y OK los hace
    BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._tapped: dict[int, float] = {}  # círculo -> instante del toque, hasta verlo apagado

    def step(self, frame: np.ndarray) -> None:
        lit = lit_circles(frame, self.params)
        for i in list(self._tapped):
            if i not in lit and self.frame_time > self._tapped[i]:
                del self._tapped[i]
        for i in lit:
            if i not in self._tapped:
                self.device.tap(*self.params["circles"][i])
                self._tapped[i] = self.frame_time
                log.info("Círculo %d encendido: toque", i)
