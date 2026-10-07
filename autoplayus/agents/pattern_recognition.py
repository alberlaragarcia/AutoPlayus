import numpy as np

from autoplayus.agents.base import BaseAgent


class PatternRecognitionAgent(BaseAgent):
    """Match-3, diferencias, memoria visual: detectar elementos en pantalla y tocar/intercambiar."""

    def step(self, frame: np.ndarray) -> None:
        # TODO: segmentar el tablero según self.params (región, filas, columnas),
        # clasificar celdas (template matching / color) y elegir el mejor movimiento.
        raise NotImplementedError("Implementar por minijuego usando self.params")
