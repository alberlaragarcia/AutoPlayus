import logging
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

Key = tuple[int, int, int, int]  # (carril, y superior visible, y inferior visible, y superior real estimada)


def find_keys(frame: np.ndarray, params: dict[str, Any]) -> list[Key]:
    """Teclas blancas visibles. Se mira solo una banda en el centro de cada carril: así el dígito de la
    puntuación (blanco, entre carriles) no se confunde con una tecla. El fondo y las teclas ya tocadas
    (que se apagan) tienen poco verde, y las blancas mucho: basta ese canal y unas pocas columnas."""
    play_top, half = params["play_top"], params["band_half"]
    keys: list[Key] = []
    for lane, x in enumerate(params["lanes_x"]):
        white = (frame[play_top:, x - half : x + half : 6, 1] >= params["white_min"]).all(axis=1)
        edges = np.flatnonzero(np.diff(np.concatenate(([False], white, [False])).astype(np.int8)))
        for start, end in zip(edges[::2], edges[1::2]):  # [start, end) en filas de la banda
            top, bottom = play_top + int(start), play_top + int(end) - 1
            if bottom - top + 1 < params["min_len"]:
                continue  # sliver de una tecla que aún está entrando por arriba
            # Recortada por arriba su borde superior no es el real: se deduce del inferior (las teclas son rígidas)
            real_top = bottom - params["key_h"] if top == play_top else top
            keys.append((lane, top, bottom, real_top))
    return keys


class Piano(BaseAgent):
    """4 carriles, una tecla blanca por fila que cae cada vez más rápido. Hay que tocar cada una antes de que
    salga por abajo; tocar el vacío (también una tecla ya tocada) o dejar pasar una tecla termina la partida.

    Se toca siempre la tecla más baja aún no tocada, en el centro de su carril. El juego tarda ~0,12 s en
    apagar una tecla tocada: hasta entonces sigue blanca y no se puede volver a tocar. Para reconocerla se
    sigue cada tecla de un fotograma al siguiente (las teclas solo bajan), sin depender de su velocidad.

    Start, toque inicial (el círculo del tutorial), detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._tracks: list[tuple[int, int, bool]] = []  # (carril, y superior real, ya tocada)

    def step(self, frame: np.ndarray) -> None:
        keys = find_keys(frame, self.params)
        self._tracks = self._follow(keys)
        pending = [(k, i) for i, (k, t) in enumerate(zip(keys, self._tracks)) if not t[2]]
        if not pending:
            return
        (lane, top, bottom, real_top), i = max(pending, key=lambda p: p[0][2])  # la más baja
        self.device.tap(self.params["lanes_x"][lane], (top + bottom) // 2)
        self._tracks[i] = (lane, real_top, True)
        log.info("Toque carril %d (tecla en y=%d..%d)", lane, top, bottom)

    def _follow(self, keys: list[Key]) -> list[tuple[int, int, bool]]:
        """Pistas de este fotograma (mismo orden que `keys`): cada tecla hereda el 'tocada' de la tecla del
        fotograma anterior en su carril que tenía justo por encima; si no hay ninguna, es una tecla nueva."""
        free = list(self._tracks)
        tracks: list[tuple[int, int, bool]] = [(k[0], k[3], False) for k in keys]
        for i in sorted(range(len(keys)), key=lambda i: keys[i][3]):  # de arriba abajo
            lane, real_top = keys[i][0], keys[i][3]
            near = [t for t in free if t[0] == lane and 0 <= real_top - t[1] < self.params["max_move"]]
            if near:
                prev = max(near, key=lambda t: t[1])  # la más cercana por encima
                free.remove(prev)
                tracks[i] = (lane, real_top, prev[2])
        return tracks
