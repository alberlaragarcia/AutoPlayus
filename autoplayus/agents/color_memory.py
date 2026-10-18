import colorsys
import logging
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

SLIDERS = ("hue", "sat", "val")


def background_bgr(frame: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Color del fondo (media de un parche fuera de los controles)."""
    x0, y0, x1, y1 = params["sample_box"]
    return frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def uniform_screen(frame: np.ndarray, params: dict[str, Any]) -> bool:
    """La pantalla de "Memoriza" es un único color: varios puntos alejados se ven iguales."""
    colors = np.array([frame[y, x] for x, y in params["uniform_points"]], float)
    return float(np.abs(colors - colors.mean(axis=0)).max()) <= params["uniform_tol"]


def to_hsv(bgr: np.ndarray) -> tuple[float, float, float]:
    """(tono en grados, saturación 0-1, brillo 0-1), como los deslizadores del juego."""
    b, g, r = (float(c) / 255 for c in bgr)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def find_knob(frame: np.ndarray, y: int, params: dict[str, Any]) -> float | None:
    """x del botón blanco del deslizador de la fila `y` (None si no se ve: aún no es la fase de recrear)."""
    x0, x1 = params["track_x"]
    row = frame[y - 3 : y + 4, x0:x1].min(axis=2).min(axis=0)
    xs = np.flatnonzero(row > params["knob_white"])
    return float(xs.mean()) + x0 if params["knob_px"][0] <= len(xs) <= params["knob_px"][1] else None


def unwrap_hue(measured: float, near: float) -> float:
    """El tono es circular (0 = 360, rojo) pero el deslizador es lineal, con rojo en los dos extremos: se toma
    la lectura (o la misma más/menos 360) más cercana a `near`, dentro de 0-360."""
    options = [h for h in (measured - 360, measured, measured + 360) if -5 <= h <= 365]
    return min(options, key=lambda h: abs(h - near))


class ColorMemory(BaseAgent):
    """Un color llena la pantalla unos segundos ("Memoriza"); luego, con tres deslizadores (tono, saturación y
    brillo) hay que recrearlo y pulsar "FIJAR". La puntuación es el porcentaje de acierto.

    El fondo de la pantalla de recrear muestra en vivo el color de los deslizadores, así que se ajusta en bucle
    cerrado: se lee el objetivo durante "Memoriza" y, deslizador a deslizador, se arrastra el botón a donde
    debería estar según una calibración lineal (medida en la partida humana), se mide el color resultante y se
    corrige el error restante, actualizando la pendiente con cada movimiento, hasta quedar dentro de
    `tolerance`. Luego otra pasada para afinar y "FIJAR".

    Arrastres: Android no da por arrastre un movimiento de menos de ~10-15 px, así que en ColorMemory_run1
    todas las correcciones finas se perdieron (el tono no se movió nada; 98,34 % con 10° de error). Ahora se
    pulsa en el botón, se aleja `overshoot_px` para superar ese umbral y se vuelve al punto exacto antes de
    soltar, sin levantar el dedo (Device.drag_path). Dónde queda el botón respecto a dónde se soltó se mide y
    se compensa en los siguientes arrastres; si un arrastre no lo mueve, se aleja el doble la próxima vez.

    Start, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._target: tuple[float, float, float] | None = None
        self._slider = 0
        self._passes = 0
        self._moves = 0
        self._moved_at = -1e9
        self._last: tuple[float, float] | None = None  # (x, valor) antes del último arrastre
        self._slope = {name: 1 / self.params["slope_px"][name] for name in SLIDERS}
        self._offset = {name: 0.0 for name in SLIDERS}  # dónde queda el botón menos dónde se soltó (px)
        self._requested: float | None = None  # x a la que se soltó el último arrastre
        self._overshoot = float(self.params["overshoot_px"])
        self._done = False

    def step(self, frame: np.ndarray) -> None:
        if self._done:
            return
        knobs = [find_knob(frame, y, self.params) for y in self.params["rows"]]
        if None in knobs:
            if uniform_screen(frame, self.params):
                self._target = to_hsv(background_bgr(frame, self.params))  # se queda con el último visto
            return
        if self._target is None or self.frame_time - self._moved_at < self.params["settle_s"]:
            return
        name, y, x = SLIDERS[self._slider], self.params["rows"][self._slider], knobs[self._slider]
        current = to_hsv(background_bgr(frame, self.params))
        value, goal = current[self._slider], self._target[self._slider]
        if name == "hue":  # rama del tono coherente con dónde está el botón, y extremo rojo más cercano
            x0, x1 = self.params["track_x"]
            value = unwrap_hue(value, (x - x0) / (x1 - x0) * 360)
            goal = unwrap_hue(goal, value)
        self._learn_from_last_drag(name, x)
        if self._last is not None and abs(x - self._last[0]) > 3:
            measured = (value - self._last[1]) / (x - self._last[0])  # cambio de valor por px
            if measured * self._slope[name] > 0:  # mismo signo: es una medida válida de la pendiente
                self._slope[name] = measured
        err = goal - value
        tolerance = self.params["tolerance"][name]
        if abs(err) <= tolerance or self._moves >= self.params["max_moves"]:
            self._next_slider()
            return
        lo, hi = self.params["track_x"]
        new_x = float(np.clip(x + err / self._slope[name], lo + 8, hi - 8))
        release = float(np.clip(new_x - self._offset[name], lo + 8, hi - 8))
        self._drag(x, y, release)
        self._last, self._moved_at, self._moves = (x, value), self.frame_time, self._moves + 1
        log.info("%s: %.3f -> objetivo %.3f (error %.3f): botón %.0f -> %.0f", name, value, goal, err, x, new_x)

    def _drag(self, x: float, y: int, release: float) -> None:
        """Pulsar en el botón, alejarse `self._overshoot` px (para que cuente como arrastre) y soltar en
        `release`. El alejamiento va hacia el lado de `release`, salvo que no quepa en la barra."""
        lo, hi = self.params["track_x"]
        side = 1.0 if release >= x else -1.0
        away = x + side * self._overshoot
        if not lo + 8 <= away <= hi - 8:
            away = x - side * self._overshoot
        points = [(int(round(x)), y), (int(round(away)), y), (int(round(release)), y)]
        self.device.drag_path(points, self.params["drag_ms"])
        self._requested = release

    def _learn_from_last_drag(self, name: str, x: float) -> None:
        """Comparar dónde quedó el botón con dónde se soltó: compensar la diferencia en los siguientes
        arrastres, o alejarse más si ni se movió."""
        if self._requested is None or self._last is None:
            return
        requested, before = self._requested, self._last[0]
        self._requested = None
        if abs(x - before) < 1.5 and abs(requested - before) >= self.params["stuck_min_px"]:
            self._overshoot = min(self._overshoot * 2, self.params["overshoot_max_px"])
            log.warning("%s: el arrastre no movió el botón; alejamiento %.0f px", name, self._overshoot)
            return
        self._offset[name] = 0.5 * self._offset[name] + 0.5 * (x - requested)

    def _next_slider(self) -> None:
        self._slider, self._moves, self._last = self._slider + 1, 0, None
        if self._slider < len(SLIDERS):
            return
        self._passes += 1
        if self._passes < self.params["passes"]:
            self._slider = 0
            return
        log.info("Color ajustado: FIJAR")
        self.device.tap(*self.params["submit"])
        self._done = True
