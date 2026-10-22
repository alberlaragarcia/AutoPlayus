import logging
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)


def board_width(frame: np.ndarray, params: dict[str, Any]) -> int | None:
    """Ancho (px) de la parte visible del tablero rojo, o None si no se ve.

    El tablero entra volando por abajo a la izquierda, cruza en perspectiva y sale por arriba a la derecha. El
    juego lo dibuja en una zona más estrecha que la pantalla (`canvas_x`) y desde ~nivel 15 el tablero no cabe
    entero ni en su momento más cercano: no se exige verlo entero (ver count_boxes y decide)."""
    y0, y1 = params["play_rows"]
    sub = frame[y0:y1].astype(np.int16)
    b, g, r = sub[..., 0], sub[..., 1], sub[..., 2]
    xs = np.nonzero((r > 220) & (g < 130) & (b < 110))[1]
    if len(xs) < params["board_min_px"]:
        return None
    return int(xs.max() - xs.min())


def count_boxes(frame: np.ndarray, params: dict[str, Any]) -> int:
    """Cajas que se ven: manchas amarillo claro (la cara de arriba de cada cubo; los lados son más oscuros).

    Los cubos no se tocan (hay hueco entre casillas), así que cada cara es una mancha. Por la perspectiva las de
    delante son mayores: no se divide por un área típica. Se ignoran los trozos menores que `piece_frac` veces
    la mediana (cara tapada en parte por otro cubo), salvo si tocan el borde de la zona de dibujo: es un cubo
    cortado por el borde (en el nivel 18 de la 1.ª partida en vivo, ignorarlo hizo responder 11 en vez de 12)."""
    y0, y1 = params["play_rows"]
    sub = frame[y0:y1].astype(np.int16)
    b, g, r = sub[..., 0], sub[..., 1], sub[..., 2]
    top = ((r > 230) & (g > 195) & (b > 40) & (b < 130)).astype(np.uint8)
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(top)
    stats = stats[1:]
    faces = stats[stats[:, cv2.CC_STAT_AREA] >= params["face_min_px"], cv2.CC_STAT_AREA]
    if not len(faces):
        return 0
    x0, x1 = params["canvas_x"]
    m = params["edge_px"]
    left, width, area = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_AREA]
    whole = (area >= params["face_min_px"]) & (area >= params["piece_frac"] * np.median(faces))
    cut = ((left <= x0 + m) | (left + width >= x1 - m)) & (area >= params["edge_piece_px"])
    return int((whole | cut).sum())


def find_submit(frame: np.ndarray, params: dict[str, Any]) -> tuple[int, int] | None:
    """Centro del botón "Enviar" (morado) si el panel de respuesta está activo; None si no (o ya enviado, gris)."""
    f = frame.astype(np.int16)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    purple = ((b > 200) & (r > 80) & (r < 140) & (g < 120)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(purple)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    lo, hi = params["submit_size"]
    if not (lo[0] <= w <= hi[0] and lo[1] <= h <= hi[1]):
        return None
    return int(cents[i][0]), int(cents[i][1])


def decide(observations: list[tuple[int, int]], params: dict[str, Any]) -> int | None:
    """Cuenta más repetida entre los fotogramas con el tablero casi a su tamaño máximo (más nítido y grande).
    Acierta los 7 niveles de la partida humana y los 18 de la 1.ª en vivo (repasado sobre las grabaciones)."""
    if not observations:
        return None
    widest = max(w for w, _ in observations)
    counts = Counter(c for w, c in observations if w >= params["keep_width"] * widest)
    return counts.most_common(1)[0][0]


class BoxGuesser(BaseAgent):
    """Un tablero con cubos amarillos cruza la pantalla en menos de 1 s; luego hay que decir cuántas cajas había
    con un panel (-, número, +, "Enviar"). Cada nivel tiene más cubos; fallar acaba la partida.

    Mientras se ve el tablero se guarda su ancho visible y las cajas contadas (count_boxes). Al salir el panel
    se decide (decide), se pulsa "+" tantas veces como cajas, esperando a ver cambiar el número tras cada toque,
    y se envía. Las posiciones de "-", "+" y el número se toman respecto al botón "Enviar" encontrado en
    pantalla (la cabecera cambia según el modo y mueve el contenido).

    Start, primer toque, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._obs: list[tuple[int, int]] = []
        self._answered = False

    def step(self, frame: np.ndarray) -> None:
        width = board_width(frame, self.params)
        if width is not None:
            if self._answered:  # tablero de un nivel nuevo
                self._obs, self._answered = [], False
            self._obs.append((width, count_boxes(frame, self.params)))
            return
        submit = find_submit(frame, self.params)
        if submit is None or self._answered:
            return
        answer = decide(self._obs, self.params)
        if answer is None:
            log.warning("Panel de respuesta sin haber visto el tablero: no se responde")
            self._answered = True
            return
        log.info("Cajas: %d (%d fotogramas, cuentas %s)", answer, len(self._obs),
                 dict(Counter(c for _, c in self._obs)))
        self._enter(answer, frame, submit)
        self._answered = True

    def _enter(self, answer: int, frame: np.ndarray, submit: tuple[int, int]) -> None:
        sx, sy = submit
        px, py = sx + self.params["plus_offset"][0], sy + self.params["plus_offset"][1]
        nx, ny = sx + self.params["number_offset"][0], sy + self.params["number_offset"][1]
        h = self.params["number_half"]
        box = lambda f: f[ny - h[1]:ny + h[1], nx - h[0]:nx + h[0]].astype(np.int16)
        shown = box(frame)
        for k in range(answer):
            for attempt in range(self.params["tap_retries"] + 1):
                self.device.tap(px, py)
                deadline = time.time() + self.params["change_s"]
                changed = False
                while time.time() < deadline:
                    now = box(self.device.screenshot())
                    if np.abs(now - shown).max(axis=2).astype(bool).sum() >= self.params["change_px"]:
                        shown, changed = now, True
                        break
                if changed:
                    break
                log.warning("El toque %d en '+' no cambió el número: se repite", k + 1)
        time.sleep(self.params["settle_s"])
        self.device.tap(sx, sy)
