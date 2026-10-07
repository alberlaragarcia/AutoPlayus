import numpy as np

from autoplayus.agents.base import BaseAgent


class PuzzleLogicAgent(BaseAgent):
    """Sudoku, laberintos, conexiones: leer el estado de pantalla a una estructura y resolver con un solver clásico."""

    def step(self, frame: np.ndarray) -> None:
        # TODO: percibir -> estado (matriz/grafo) -> solver (backtracking, BFS/A*) -> acciones (tap / drag_path).
        raise NotImplementedError("Implementar por minijuego usando self.params")
