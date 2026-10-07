import logging

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)


class TowerStack(BaseAgent):
    """Start, toque inicial, detección de fin y OK los hace BaseAgent.run(). Aquí solo va la mecánica."""

    def step(self, frame: np.ndarray) -> None:
        # 1. percibir: analizar `frame` (usa vision/utils.py)
        # 2. decidir: ¿está el bloque deslizante alineado con la torre?
        # 3. actuar: self.device.tap(x, y) para soltarlo
        self.device.tap(360, 640)  # PROVISIONAL: toca a ciegas para probar el flujo completo
