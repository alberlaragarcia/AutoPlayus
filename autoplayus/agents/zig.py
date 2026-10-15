import logging
from collections import deque
from typing import Any

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)


def find_ball(frame: np.ndarray, params: dict[str, Any]) -> tuple[int, int] | None:
    """Centro (x, y) de la bola (negra), o None si no se ve. Se muestrea 1 de cada 2 píxeles. Una mancha
    de tamaño distinto al de la bola es otra cosa: la pantalla oscurecida del tutorial o del final."""
    y0, y1 = params["ball_rows"]
    sub = frame[y0:y1:2, ::2]
    ys, xs = np.nonzero(sub.max(axis=2) < params["ball_dark"])
    if not params["ball_px"][0] <= len(xs) <= params["ball_px"][1]:
        return None
    return int(xs.mean()) * 2, int(ys.mean()) * 2 + y0


def yellow_row(row: np.ndarray) -> np.ndarray:
    """Máscara de la superficie amarilla (cara superior del camino) en una fila o imagen BGR."""
    b, g, r = (row[..., c].astype(np.int16) for c in range(3))
    return (r > 200) & (g > 170) & (b < 150)


def path_center(frame: np.ndarray, y: int, x: int, min_run: int) -> float | None:
    """Centro del tramo amarillo de la fila `y` que contiene a `x` o, si ninguno, el más cercano; None si no hay."""
    mask = yellow_row(frame[y])
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run]
    if not runs:
        return None
    a, b = min(runs, key=lambda r: 0 if r[0] <= x < r[1] else min(abs(r[0] - x), abs(r[1] - x)))
    return (a + b) / 2


class ScrollTracker:
    """Cómo se desplaza el camino de un fotograma al siguiente (correlación de fase sobre la máscara amarilla).

    El camino baja por la pantalla y, según hacia dónde vaya la bola, se desplaza a un lado: el signo de ese
    desplazamiento lateral (suavizado) es la dirección de la bola, y el vertical, la velocidad. Se mira solo
    la zona de juego y 1 de cada 4 píxeles: ~1 ms por fotograma.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self._prev: np.ndarray | None = None
        self._window = cv2.createHanningWindow((180, 225), cv2.CV_32F)
        self._dx: deque[float] = deque(maxlen=params["dir_frames"])
        self._dy: deque[tuple[float, float]] = deque(maxlen=params["dir_frames"])  # (px, s) de cada paso

    def update(self, frame: np.ndarray, t: float) -> None:
        mask = yellow_row(frame[300:1200:4, ::4]).astype(np.float32)
        if self._prev is not None:
            (fx, fy), _ = cv2.phaseCorrelate(self._prev, mask, self._window)
            self._dx.append(fx * 4)
            self._dy.append((fy * 4, t - self._t))
        self._prev, self._t = mask, t

    def direction(self) -> int | None:
        """+1 la bola va hacia la derecha, -1 hacia la izquierda; None si aún no se sabe.

        El camino se desplaza al lado contrario de la bola: si va a la derecha, el camino va a la izquierda."""
        if len(self._dx) < self._dx.maxlen:
            return None
        mean = float(np.mean(self._dx))
        return None if abs(mean) < self.params["dir_min"] else (-1 if mean > 0 else 1)

    def speed(self) -> float:
        """Velocidad de bajada del camino en px/s (0 si aún no se sabe)."""
        px, secs = sum(p for p, _ in self._dy), sum(s for _, s in self._dy)
        return px / secs if secs > 0 else 0.0


def desired_direction(center: float | None, ball_x: int, current: int, margin: float) -> int:
    """Hacia dónde debería ir la bola: hacia el lado donde queda el centro del camino por delante. Dentro del
    margen se mantiene la dirección actual (evita tocar a cada fotograma cuando el camino está recto)."""
    if center is None:
        return current
    if center > ball_x + margin:
        return 1
    if center < ball_x - margin:
        return -1
    return current


class Zig(BaseAgent):
    """Una bola avanza por un camino amarillo en zigzag; cada toque cambia su dirección lateral (derecha <->
    izquierda) y se pierde al salirse del camino. La partida acelera con la distancia.

    Se sigue el centro del camino: a unas filas por delante de la bola (más lejos cuanto más rápido baja el
    camino, para compensar el retraso del toque) se mira dónde está el centro del tramo amarillo. Si queda al
    otro lado de la bola que su dirección actual, se toca. La dirección se lee del desplazamiento del camino
    (ScrollTracker) y, tras un toque, se da por cambiada hasta pasado `block_s`, porque la lectura tarda unos
    fotogramas en reflejarlo.

    Start, primer toque (el círculo del tutorial, en UI_FIRST_TAP), detección de fin y OK los hace
    BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._scroll = ScrollTracker(self.params)
        self._assumed: int | None = None  # dirección tras nuestro último toque
        self._tapped_at = -1e9

    def step(self, frame: np.ndarray) -> None:
        ball = find_ball(frame, self.params)
        if ball is None:
            self._scroll._prev = None  # el desplazamiento entre un fotograma con bola y otro sin ella no vale
            return
        self._scroll.update(frame, self.frame_time)
        measured = self._scroll.direction()
        if self.frame_time - self._tapped_at < self.params["block_s"]:
            return  # el último toque aún no se refleja: ni tocar otra vez ni fiarse de la lectura
        current = measured if measured is not None else self._assumed
        if current is None:
            return  # aún sin dirección
        x, y = ball
        ahead = int(np.clip(self._scroll.speed() * self.params["look_s"], *self.params["look_px"]))
        center = path_center(frame, y - ahead, x, self.params["min_run"])
        want = desired_direction(center, x, current, self.params["margin_px"])
        if want == current:
            return
        self.device.tap(*self.params["tap"])
        self._assumed, self._tapped_at = want, self.frame_time
        log.info("Toque %+d -> %+d (bola x=%d, centro adelante=%s a %d px, bajada %.0f px/s)",
                 current, want, x, None if center is None else round(center), ahead, self._scroll.speed())
