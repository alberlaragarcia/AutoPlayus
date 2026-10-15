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


def heading(direction: int, params: dict[str, Any]) -> tuple[float, float]:
    """Vector unitario (dx, dy) en pantalla del avance de la bola sobre el camino: los bordes del camino, y por
    tanto los dos sentidos de avance, forman ~27° (derecha) y ~142° (izquierda) con la horizontal (no son
    simétricos por el ángulo de la cámara; medido con Hough en la partida humana)."""
    a = np.radians(params["heading_deg"][0 if direction > 0 else 1])
    return float(np.cos(a)), float(-np.sin(a))


def free_distance(frame: np.ndarray, x: int, y: int, direction: int, params: dict[str, Any]) -> float:
    """px de camino amarillo que tiene la bola por delante si sigue en `direction` sin girar.

    Es el mínimo de tres rayos paralelos (el central y uno a cada lado, a `ray_side_px`): así cuenta el ancho
    de la bola, y yendo pegada a un borde la distancia sale corta aunque el centro aún tenga camino. Un rayo
    termina en el primer par de puntos seguidos que no son amarillos (un punto suelto es ruido del borde)."""
    dx, dy = heading(direction, params)
    s = np.arange(params["ray_start_px"], params["ray_max_px"], params["ray_step_px"], dtype=np.float32)
    shortest = float(params["ray_max_px"])
    for side in (-params["ray_side_px"], 0, params["ray_side_px"]):
        px = np.round(x - dy * side + s * dx).astype(int)  # (-dy, dx) es la perpendicular al avance
        py = np.round(y + dx * side + s * dy).astype(int)
        inside = (px >= 0) & (px < frame.shape[1]) & (py >= 0) & (py < frame.shape[0])
        on_path = np.zeros(len(s), bool)
        on_path[inside] = yellow_row(frame[py[inside], px[inside]])
        off = np.flatnonzero(~on_path[:-1] & ~on_path[1:])
        if len(off):
            shortest = min(shortest, float(s[off[0]]))
    return shortest


class ScrollTracker:
    """Cómo se desplaza el camino de un fotograma al siguiente (correlación de fase sobre la máscara amarilla).

    El camino baja por la pantalla y, según hacia dónde vaya la bola, se desplaza a un lado: el signo de ese
    desplazamiento lateral es la dirección de la bola, y el vertical, la velocidad. Todo en px/s sobre los
    últimos `scroll_window_s` s: por fotograma depende de cuántas capturas por segundo lleguen (el juego va a
    ~60 fps y la captura por ventana a ~200, así que muchas capturas repiten imagen). Se mira solo la zona de
    juego y 1 de cada 4 píxeles: ~1 ms por fotograma.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self._prev: np.ndarray | None = None
        self._window = cv2.createHanningWindow((180, 225), cv2.CV_32F)
        self._steps: deque[tuple[float, float, float, float]] = deque()  # (t, dx, dy, dt) de cada paso

    def reset(self) -> None:
        """El desplazamiento entre un fotograma con bola y otro sin ella no vale."""
        self._prev = None

    def update(self, frame: np.ndarray, t: float) -> None:
        mask = yellow_row(frame[300:1200:4, ::4]).astype(np.float32)
        if self._prev is not None and t > self._t:
            (fx, fy), _ = cv2.phaseCorrelate(self._prev, mask, self._window)
            self._steps.append((t, fx * 4, fy * 4, t - self._t))
            while self._steps and self._steps[0][0] < t - self.params["scroll_window_s"]:
                self._steps.popleft()
        self._prev, self._t = mask, t

    def _rate(self, k: int) -> float | None:
        secs = sum(s[3] for s in self._steps)
        if secs < self.params["scroll_window_s"] / 2:
            return None
        return sum(s[k] for s in self._steps) / secs

    def direction(self) -> int | None:
        """+1 la bola va hacia la derecha, -1 hacia la izquierda; None si aún no se sabe.

        La cámara no sigue a la bola de lado al instante: se desplaza con retraso hacia donde va, y con ella todo
        el camino se mueve en pantalla hacia ese mismo lado (medido en la partida humana: mismo signo que la bola
        en el 96 % de los fotogramas). Por ese retraso esta lectura llega tarde tras cada giro."""
        vx = self._rate(1)
        if vx is None or abs(vx) < self.params["dir_min_pxs"]:
            return None
        return 1 if vx > 0 else -1

    def speed(self) -> float:
        """Velocidad de bajada del camino en px/s (0 si aún no se sabe)."""
        return self._rate(2) or 0.0


def should_turn(ahead: float, other: float, trigger: float) -> bool:
    """Girar cuando en la dirección actual queda menos camino que `trigger` (lo que la bola recorre mientras
    el toque surte efecto, con margen) y en la otra queda más. Si en la otra tampoco hay, girar no arregla
    nada: se sigue (en un tramo recto el camino de lado siempre es corto)."""
    return ahead <= trigger and other > ahead


class Zig(BaseAgent):
    """Una bola avanza por un camino amarillo en zigzag; cada toque cambia su dirección lateral (derecha <->
    izquierda) y se pierde al salirse del camino. La partida acelera con la distancia.

    En cada captura se mide cuánto camino queda por delante de la bola en sus dos sentidos posibles
    (free_distance, con rayos sobre la propia captura) y se gira cuando el actual se acaba y el otro tiene más
    (should_turn). La distancia a la que se gira es lo que la bola recorre en `look_s` s (retraso del toque,
    ~0,06-0,1 s medidos en vivo, más margen): su velocidad sobre el camino se deduce de lo rápido que baja el
    camino en pantalla. Vale igual para la plataforma ancha del principio (se va recto hasta su borde) que
    para entrar en el camino y seguirlo; yendo pegada a un borde, el rayo lateral la hace corregir.

    La dirección la lleva el agente: empieza en `start_dir` (la bola sale siempre hacia la izquierda) y cada
    toque la invierte. Leerla del desplazamiento del camino (ScrollTracker) va con retraso: fiarse de ella
    justo después de un toque hacía repetir el toque y deshacer el giro. Solo se corrige con la lectura si
    discrepa de forma sostenida (`resync_s`), por si el juego no hizo caso a algún toque.

    Start, primer toque (el círculo del tutorial, en UI_FIRST_TAP), detección de fin y OK los hace
    BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._scroll = ScrollTracker(self.params)
        self._dir = self.params["start_dir"]  # dirección de la bola según los toques dados
        self._tapped_at = -1e9
        self._disagree_since: float | None = None  # desde cuándo la lectura contradice a _dir

    def step(self, frame: np.ndarray) -> None:
        ball = find_ball(frame, self.params)
        if ball is None:
            self._scroll.reset()
            return
        self._scroll.update(frame, self.frame_time)
        if self.frame_time - self._tapped_at < self.params["block_s"]:
            return  # el último toque aún no se refleja: ni tocar otra vez ni fiarse de la lectura
        self._resync(self._scroll.direction())
        current = self._dir
        x, y = ball
        ahead = free_distance(frame, x, y, current, self.params)
        other = free_distance(frame, x, y, -current, self.params)
        trigger = self.ball_speed(current) * self.params["look_s"] + self.params["margin_px"]
        if not should_turn(ahead, other, trigger):
            return
        self.device.tap(*self.params["tap"])
        self._dir, self._tapped_at, self._disagree_since = -current, self.frame_time, None
        log.info("Toque %+d -> %+d (bola x=%d, camino delante %.0f px / al otro lado %.0f px, giro a %.0f px)",
                 current, -current, x, ahead, other, trigger)

    def ball_speed(self, direction: int) -> float:
        """px/s de la bola sobre el camino: la cámara la mantiene a la misma altura, así que el camino baja en
        pantalla lo que ella sube; dividiendo por la componente vertical de su avance sale la velocidad."""
        down = max(self._scroll.speed(), self.params["min_scroll_pxs"])
        return down / abs(heading(direction, self.params)[1])

    def _resync(self, measured: int | None) -> None:
        """Si la dirección leída contradice a la de los toques durante `resync_s`, manda la leída."""
        if measured is None or measured == self._dir:
            self._disagree_since = None
        elif self._disagree_since is None:
            self._disagree_since = self.frame_time
        elif self.frame_time - self._disagree_since >= self.params["resync_s"]:
            log.warning("La bola va hacia %+d y no hacia %+d: se corrige la dirección", measured, self._dir)
            self._dir, self._disagree_since = measured, None
