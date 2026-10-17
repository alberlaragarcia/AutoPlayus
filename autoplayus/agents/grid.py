import logging
from typing import Any

import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

Tile = tuple[int, int, str]  # (x, y, estado): estado "green" (patrón), "yellow" (marcada), "red", "dark"


def find_tiles(frame: np.ndarray, params: dict[str, Any]) -> list[Tile]:
    """Casillas de la cuadrícula con su estado. Son cuadrados que destacan del fondo (BGR ~26,19,18) en la zona
    `grid_rows`; su tamaño cambia con el nivel (la cuadrícula crece), así que se aceptan de `tile_px` a lo
    ancho y alto. El estado se lee en un parche del centro."""
    y0, y1 = params["grid_rows"]
    background = np.array(params["background_bgr"], np.int16)
    mask = (np.abs(frame[y0:y1].astype(np.int16) - background).max(axis=2) > params["background_tol"]).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    lo, hi = params["tile_px"]
    tiles = []
    for (_, _, w, h, _), (cx, cy) in zip(stats[1:], centers[1:]):
        if lo <= w <= hi and lo <= h <= hi and abs(w - h) <= params["tile_square_tol"]:
            x, y = int(cx), int(cy) + y0
            b, g, r = frame[y - 3 : y + 4, x - 3 : x + 4].reshape(-1, 3).mean(axis=0)
            if g > 180 and b > 120 and r < 60:
                state = "green"
            elif r > 230 and g > 180 and b < 140:
                state = "yellow"
            elif r > 200 and g < 120:
                state = "red"
            else:
                state = "dark"
            tiles.append((x, y, state))
    return tiles


def okay_visible(frame: np.ndarray, params: dict[str, Any]) -> bool:
    """El botón "Okay!" (verde ~93,183,97 BGR) aparece cuando se puede responder."""
    x0, y0, x1, y1 = params["okay_box"]
    box = frame[y0:y1, x0:x1].astype(np.int16)
    green = (box[..., 1] > 160) & (box[..., 0] < 120) & (box[..., 2] < 120)
    return float(green.mean()) > 0.3


def match(points: list[tuple[int, int]], others: list[tuple[int, int]], tol: float) -> list[tuple[int, int]]:
    """Puntos de `points` sin ninguno de `others` a menos de `tol` px."""
    return [p for p in points if all(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol for q in others)]


class Grid(BaseAgent):
    """Una cuadrícula de casillas (más grande en cada nivel) enciende en verde un patrón unos 3 s; luego se
    apaga y, con una barra de tiempo, hay que marcar esas casillas (se ponen amarillas) y pulsar "Okay!".
    Fallar termina la partida (el juego enseña en rojo el patrón correcto).

    Mientras se ve el patrón se guarda el conjunto de casillas verdes más grande visto (entran con una pequeña
    animación). En cuanto aparece "Okay!" se tocan una a una (`tap_gap_s` entre toques: el juego no acepta
    toques casi simultáneos), se comprueba que las amarillas sean exactamente esas (si falta o sobra alguna se
    vuelve a tocar: tocar alterna) y se pulsa "Okay!". Al tocar, el juego anima la casilla y durante la
    animación no se ve bien la cuadrícula: solo se comprueba con todas las casillas a la vista y la misma
    lectura en dos capturas seguidas (Grid_run1 desmarcó el patrón entero por comprobar a mitad de animación).

    Al acertar, el juego confirma la respuesta enseñándola en verde en la cuadrícula del nivel acabado y justo
    después muestra la del siguiente con su patrón. Por eso, tras enviar, los verdes contenidos en lo enviado
    se ignoran (son la confirmación) y el nivel nuevo empieza con el primer verde que no estaba en lo enviado.
    Si el patrón nuevo cupiera entero en lo enviado (misma cuadrícula), se usa lo último visto en verde.

    Start, detección de fin y OK los hace BaseAgent.run(); el tutorial ("Memoriza el patrón...") se vuelve a
    tocar si el primer toque no lo cierra.
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._pattern: list[tuple[int, int]] = []
        self._tile_tol = 30.0
        self._state = "watch"  # watch: leyendo el patrón; typed: casillas tocadas; sent: "Okay!" pulsado; between: entre niveles
        self._sent: list[tuple[int, int]] = []  # última respuesta enviada
        self._fallback: list[tuple[int, int]] = []  # verdes vistos entre niveles que caben en lo enviado
        self._queue: list[tuple[int, int]] = []  # casillas por tocar, de una en una
        self._last_tap = -1e9
        self._grid_n = 0  # casillas de la cuadrícula de este nivel
        self._last_read: list[tuple[int, int]] | None = None  # lectura anterior de las marcadas
        self._fixes = 0
        self._playing = False
        self._last_start_tap = -1e9

    def step(self, frame: np.ndarray) -> None:
        now = self.frame_time
        if self._queue:
            if now - self._last_tap >= self.params["tap_gap_s"]:
                self.device.tap(*self._queue.pop(0))
                self._last_tap = now
            return
        tiles = find_tiles(frame, self.params)
        if not tiles:
            self._retap_tutorial(frame)
            return
        self._playing = True
        greens = [(x, y) for x, y, s in tiles if s == "green"]
        okay = okay_visible(frame, self.params)
        if self._state == "sent" and not okay:
            self._state, self._fallback = "between", []
        if greens and not okay:
            if self._state == "between":
                if not match(greens, self._sent, self._tile_tol):  # cabe en lo enviado: es la confirmación
                    if len(greens) >= len(self._fallback):
                        self._fallback = greens
                    return
                self._state, self._pattern, self._fixes = "watch", [], 0  # verde nuevo: empieza otro nivel
            if self._state == "watch" and len(greens) >= len(self._pattern):
                self._pattern, self._grid_n = greens, len(tiles)
                self._tile_tol = self._tile_size(tiles) / 2
            return
        if okay and self._state == "between" and self._fallback:
            log.warning("El patrón nuevo cabe en la respuesta anterior: se usa lo último visto en verde")
            self._state, self._pattern, self._fixes = "watch", self._fallback, 0
        if not okay or not self._pattern:
            return
        if self._state == "watch":
            log.info("Patrón de %d casillas (cuadrícula de %d)", len(self._pattern), self._grid_n)
            self._queue, self._state, self._last_read = list(self._pattern), "typed", None
        elif self._state == "typed" and now - self._last_tap >= self.params["verify_s"]:
            if len(tiles) != self._grid_n:
                return  # alguna casilla animándose: no se ve la cuadrícula entera
            marked = sorted((x, y) for x, y, s in tiles if s == "yellow")
            if marked != self._last_read:
                self._last_read = marked  # esperar a leer lo mismo dos veces
                return
            missing = match(self._pattern, marked, self._tile_tol)
            extra = match(marked, self._pattern, self._tile_tol)
            if not missing and not extra:
                self.device.tap(*self.params["okay"])
                self._state, self._sent = "sent", self._pattern
                log.info("Patrón marcado: Okay")
            elif self._fixes < self.params["max_fixes"]:
                self._fixes += 1
                log.warning("Corrigiendo: faltan %d, sobran %d", len(missing), len(extra))
                self._queue, self._last_read = missing + extra, None

    def _tile_size(self, tiles: list[Tile]) -> float:
        xs = sorted({x for x, _, _ in tiles})
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 20]
        return float(min(gaps)) if gaps else 100.0

    def _retap_tutorial(self, frame: np.ndarray) -> None:
        """Tutorial ("Memoriza el patrón, luego recréalo") aún en pantalla: volver a tocar cada segundo."""
        if self._playing or frame[300, 360].max() > self.params["tutorial_max"]:
            return
        if self.frame_time - self._last_start_tap >= 1.0:
            self.device.tap(*config.UI_FIRST_TAP)
            self._last_start_tap = self.frame_time
