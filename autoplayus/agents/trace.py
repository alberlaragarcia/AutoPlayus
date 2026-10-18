import logging
from typing import Any

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

Point = tuple[int, int]


def detect_dots(frame: np.ndarray, params: dict[str, Any]) -> list[Point]:
    """Puntos de la retícula visibles (círculos algo más claros que el panel azul, que tiene degradado: se
    buscan por contraste local en el canal rojo). Los tapados por la ruta no salen."""
    y0, y1 = params["panel_rows"]
    red = frame[y0:y1, :, 2].astype(np.float32)
    contrast = red - cv2.GaussianBlur(red, (0, 0), 15)
    _, _, stats, centers = cv2.connectedComponentsWithStats((contrast > params["dot_contrast"]).astype(np.uint8))
    lo, hi = params["dot_area"]
    return [
        (int(cx), int(cy) + y0)
        for (_, _, w, h, area), (cx, cy) in zip(stats[1:], centers[1:])
        if lo <= area <= hi and 0.6 < w / h < 1.6
    ]


def build_lattice(dots: list[Point], params: dict[str, Any]) -> list[Point]:
    """Todos los nodos de la retícula hexagonal (centrada en `center`) a partir de los puntos visibles: la
    separación es la distancia típica entre vecinos y el radio, cuántas filas hay a cada lado del centro. La
    retícula crece con las rondas (7 nodos al principio, 19 después...)."""
    if len(dots) < 3:
        return []
    pts = np.array(dots, float)
    dist = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(dist, np.inf)
    spacing = float(np.median(dist.min(axis=1)))
    cx, cy = params["center"]
    row_h = spacing * np.sqrt(3) / 2
    radius = int(round(np.abs(pts[:, 1] - cy).max() / row_h))
    nodes = []
    for k in range(-radius, radius + 1):
        count = 2 * radius + 1 - abs(k)
        for j in range(count):
            nodes.append((int(round(cx + (j - (count - 1) / 2) * spacing)), int(round(cy + k * row_h))))
    return nodes


def node_state(frame: np.ndarray, x: int, y: int) -> str:
    """Color del centro de un nodo: "yellow" (ruta que se enseña), "teal" (inicio marcado o trazo propio),
    "red" (ruta revelada al fallar) u "other"."""
    b, g, r = frame[y - 3 : y + 4, x - 3 : x + 4].reshape(-1, 3).mean(axis=0)
    if r > 220 and g > 190 and b < 150:
        return "yellow"
    if g > 200 and b > 180 and r < 150:
        return "teal"
    if r > 200 and g < 130 and b < 160:
        return "red"
    return "other"


def has_yellow(frame: np.ndarray, params: dict[str, Any]) -> bool:
    y0, y1 = params["panel_rows"]
    p = frame[y0:y1:4, ::4].astype(np.int16)
    return int(((p[..., 2] > 220) & (p[..., 1] > 190) & (p[..., 0] < 150)).sum()) >= params["yellow_px"]


def timer_visible(frame: np.ndarray, params: dict[str, Any]) -> bool:
    """Barra de tiempo amarilla bajo la puntuación: aparece cuando se puede trazar."""
    x0, y0, x1, y1 = params["timer_box"]
    p = frame[y0:y1, x0:x1].astype(np.int16)
    return float(((p[..., 2] > 220) & (p[..., 1] > 190) & (p[..., 0] < 150)).mean()) > 0.3


class Trace(BaseAgent):
    """Sobre una retícula hexagonal de puntos se dibuja en amarillo, tramo a tramo, una ruta que luego
    desaparece; con el inicio marcado en turquesa y una barra de tiempo, hay que arrastrar sin levantar el dedo
    desde el inicio siguiendo la ruta exacta. Cada ronda es más larga; un giro equivocado termina la partida.

    Mientras se dibuja la ruta se anota el orden en que cada nodo se vuelve amarillo (la ruta se revela del
    inicio al final). Al desaparecer y aparecer la barra, se comprueba que el inicio marcado sea el primero y
    se arrastra por todos los nodos en ese orden (Device.drag_path, continuo en BlueStacks).

    Start, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._state = "wait"  # wait: entre rondas; reveal: anotando la ruta; dragged: ya trazada
        self._nodes: list[Point] = []
        self._order: list[Point] = []

    def step(self, frame: np.ndarray) -> None:
        if has_yellow(frame, self.params):
            if self._state != "reveal":
                self._state, self._nodes, self._order = "reveal", [], []
            if not self._nodes:
                self._nodes = build_lattice(detect_dots(frame, self.params), self.params)
            self._track(frame)
            return
        if self._state != "reveal" or len(self._order) < 2 or not timer_visible(frame, self.params):
            return
        start = [n for n in self._nodes if node_state(frame, *n) == "teal"]
        path = self._order
        if start and start[0] != path[0] and start[0] == path[-1]:
            log.warning("El inicio marcado es el último nodo anotado: se traza al revés")
            path = path[::-1]
        elif start and start[0] != path[0]:
            log.warning("El inicio marcado %s no es el primero anotado %s", start[0], path[0])
        log.info("Ruta de %d nodos: %s", len(path), path)
        self.device.drag_path(path, self.params["segment_ms"])
        self._state = "dragged"

    def _track(self, frame: np.ndarray) -> None:
        """Añadir los nodos que se acaban de volver amarillos. Si salen varios a la vez (captura lenta), se
        encadenan por cercanía al último: la ruta va de nodo vecino en nodo vecino."""
        new = [n for n in self._nodes if n not in self._order and node_state(frame, *n) == "yellow"]
        while new:
            last = self._order[-1] if self._order else None
            nxt = min(new, key=lambda n: 0 if last is None else (n[0] - last[0]) ** 2 + (n[1] - last[1]) ** 2)
            self._order.append(nxt)
            new.remove(nxt)
