import logging
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

DIRECTIONS = {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}


def read_arrow(frame: np.ndarray, center: tuple[int, int], params: dict[str, Any]) -> str | None:
    """Dirección ("R", "L", "U", "D") de la flecha blanca de una caja, o None si no se ve.

    Si la mancha blanca es más ancha que alta la flecha es horizontal. La punta es el lado donde la mancha
    ocupa más alto (la cabeza triangular) y no el del palo: se compara la extensión en el cuarto de cada extremo.
    """
    cx, cy = center
    h = params["arrow_half"]
    ys, xs = np.nonzero(frame[cy - h:cy + h, cx - h:cx + h].min(axis=2) > params["white_min"])
    if len(xs) < params["arrow_min_px"]:
        return None
    if np.ptp(xs) > np.ptp(ys):
        q = np.ptp(xs) // 4
        start, end = np.ptp(ys[xs <= xs.min() + q]), np.ptp(ys[xs >= xs.max() - q])
        return "R" if end > start else "L"
    q = np.ptp(ys) // 4
    start, end = np.ptp(xs[ys <= ys.min() + q]), np.ptp(xs[ys >= ys.max() - q])
    return "D" if end > start else "U"


def read_bar(frame: np.ndarray, center: tuple[int, int], params: dict[str, Any]) -> int:
    """Largo (px) de la barra de tiempo de una caja (azul, naranja cuando queda poco): ~271 llena."""
    cx, cy = center
    row = frame[cy + params["bar_dy"], cx - params["bar_half"]:cx + params["bar_half"]].astype(np.int16)
    b, g, r = row[:, 0], row[:, 1], row[:, 2]
    return int((((b > 200) & (r < 120)) | ((r > 200) & (g > 100) & (b < 100))).sum())


def find_rows(frame: np.ndarray, params: dict[str, Any]) -> list[int] | None:
    """Desplazamiento vertical (px) de cada caja respecto a `boxes`, o None si no se ven las barras.

    La cabecera cambia según el modo (con la etiqueta "Modo práctica" o sin ella) y el contenido se mueve en
    vertical (en DiceRoll, dados y teclado se movieron distinto). Por cada fila de cajas se busca, cerca de
    donde se espera, la fila de píxeles con más barra encendida en cualquiera de sus cajas."""
    search = range(-params["layout_search_px"], params["layout_search_px"] + 1)
    shifts: dict[int, int] = {}
    for row_y in {cy for _, cy in params["boxes"]}:
        lit = np.array([max(read_bar(frame, (cx, cy + dy), params) for cx, cy in params["boxes"] if cy == row_y)
                        for dy in search])
        if lit.max() < params["layout_bar_px"]:
            return None
        band = np.flatnonzero(lit >= 0.8 * lit.max())  # la barra tiene grosor: se toma el centro de la franja
        shifts[row_y] = int(round(np.mean(np.array(search)[band])))
    return [shifts[cy] for _, cy in params["boxes"]]


class SwipeFast(BaseAgent):
    """4 cajas en 2x2, cada una con una flecha y una barra de tiempo que se vacía; deslizar en la caja en la
    dirección de su flecha da un punto, cambia la flecha y rellena parte de la barra. La partida acaba cuando
    se vacía alguna barra, así que se trata de deslizar lo más deprisa posible en las cuatro.

    En cada captura se leen las 4 flechas y barras. Se desliza en cada caja con una flecha leída igual en
    `stable_frames` capturas seguidas, empezando por la de barra más corta. Tras deslizar, la caja queda
    pendiente hasta ver que contó (otra flecha, o la barra sube `refill_px`): así no se repite el gesto sobre la
    flecha vieja. Si en `pending_s` no se ve nada, se vuelve a deslizar según la flecha que haya (puede salir la
    misma dirección otra vez). En vivo ~9 % de los gestos no contaron, al azar: recuperarse pronto importa.

    Tras un acierto la barra sigue subiendo `refill_anim_s` (animación del relleno): mientras tanto esa subida no
    confirma el gesto siguiente en la caja; después se compara con el mínimo de la barra visto desde entonces.

    Las coordenadas de `boxes` son las de la pantalla con la etiqueta "Modo práctica": hasta ver las barras
    (find_rows) no se hace nada y luego se usan las cajas desplazadas.

    Start, primer toque (tutorial), detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._boxes: list[tuple[int, int]] | None = None
        n = len(self.params["boxes"])
        self._last: list[str | None] = [None] * n  # última lectura de cada caja
        self._seen = [0] * n  # capturas seguidas con esa lectura
        self._pending: list[dict | None] = [None] * n  # gesto sin confirmar: instante, flecha, barra mínima
        self._confirmed_at = [-1e9] * n  # último gesto confirmado de cada caja
        self._swipes = 0

    def step(self, frame: np.ndarray) -> None:
        now = self.frame_time
        if self._boxes is None:
            shifts = find_rows(frame, self.params)
            if shifts is None:
                return
            self._boxes = [(x, y + d) for (x, y), d in zip(self.params["boxes"], shifts)]
            log.info("Cajas desplazadas %s px respecto a las de referencia", shifts)
        ready = []
        for k, center in enumerate(self._boxes):
            bar = read_bar(frame, center, self.params)
            # sin barra no es una caja de juego: en Start y en el tutorial hay texto blanco en la misma zona
            arrow = read_arrow(frame, center, self.params) if bar >= self.params["bar_min_px"] else None
            self._seen[k] = self._seen[k] + 1 if arrow == self._last[k] else 1
            self._last[k] = arrow
            pending = self._pending[k]
            if pending is not None:
                if self._confirmed(k, pending, arrow, bar, now):
                    self._pending[k] = None
                    self._confirmed_at[k] = now
                    self._seen[k] = 1  # la flecha nueva aún tiene que confirmarse
                    continue
                if now - pending["t"] < self.params["pending_s"]:
                    continue
                log.warning("Caja %d: el gesto %s no contó en %.2f s, se repite", k, pending["arrow"], now - pending["t"])
                self._pending[k] = None
            if arrow is not None and self._seen[k] >= self.params["stable_frames"]:
                ready.append((bar, k, arrow))
        for bar, k, arrow in sorted(ready):
            self._swipe(k, arrow)
            animating = now - self._confirmed_at[k] < self.params["refill_anim_s"]
            self._pending[k] = {"t": now, "arrow": arrow, "min_bar": None if animating else bar}

    def _confirmed(self, k: int, pending: dict, arrow: str | None, bar: int, now: float) -> bool:
        """El gesto contó: se ve otra flecha, o la barra sube `refill_px` sobre su mínimo desde el gesto (sin
        contar la subida del relleno anterior, que dura `refill_anim_s`)."""
        if arrow is not None and arrow != pending["arrow"]:
            return True
        if now - self._confirmed_at[k] < self.params["refill_anim_s"]:
            return False
        if pending["min_bar"] is not None and bar >= pending["min_bar"] + self.params["refill_px"]:
            return True
        pending["min_bar"] = bar if pending["min_bar"] is None else min(pending["min_bar"], bar)
        return False

    def _swipe(self, k: int, arrow: str) -> None:
        cx, cy = self._boxes[k]
        dx, dy = DIRECTIONS[arrow]
        half = self.params["swipe_px"] // 2
        self.device.swipe(cx - dx * half, cy - dy * half, cx + dx * half, cy + dy * half, self.params["swipe_ms"])
        self._swipes += 1
        log.info("Caja %d: %s (gesto %d)", k, arrow, self._swipes)
