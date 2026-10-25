import logging
import sys
import time
from typing import Any

import numpy as np

from autoplayus import config
from autoplayus.agents.base import BaseAgent, _start_template
from autoplayus.vision.utils import find_template

log = logging.getLogger(__name__)

Ring = tuple[float, float, float]  # centro x, centro y, radio (mitad del grosor del anillo)


def _ring_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = frame.astype(np.int16)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    green = (g > 200) & (b > 140) & (r < 120)
    grey = (np.abs(r - g) < 15) & (np.abs(g - b) < 20) & (r > 45) & (r < 110)
    return green, grey


def find_ring(frame: np.ndarray, params: dict[str, Any]) -> Ring | None:
    """Centro y radio del anillo del cronómetro (parte verde recorrida + parte gris por recorrer), por ajuste de
    círculo. La cabecera cambia según el modo y mueve el contenido: se mide en pantalla en vez de fijarlo."""
    green, grey = _ring_masks(frame)
    mask = green | grey
    y0, y1 = params["ring_rows"]
    mask[:y0] = False
    mask[y1:] = False
    if green[y0:y1].sum() < params["green_min_px"]:
        return None  # sin arco aún (en el aviso de "Toca para empezar" el anillo sale atenuado)
    ys, xs = np.nonzero(mask[::2, ::2])
    if len(xs) < params["ring_min_px"]:
        return None
    xs, ys = xs * 2.0, ys * 2.0
    a = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    cx, cy, c = np.linalg.lstsq(a, xs ** 2 + ys ** 2, rcond=None)[0]
    radius = float(np.sqrt(c + cx ** 2 + cy ** 2))
    lo, hi = params["ring_radius"]
    return (float(cx), float(cy), radius) if lo <= radius <= hi else None


def arc_angle(frame: np.ndarray, ring: Ring) -> float | None:
    """Grados recorridos por el arco verde, desde arriba en sentido horario; None si no hay arco que empiece
    arriba (otra pantalla). El arco crece a 360° cada 7 s: es el cronómetro, con más resolución que los dígitos."""
    cx, cy, radius = ring
    steps = np.radians(np.arange(0, 360, 0.1))
    px = np.clip((cx + radius * np.sin(steps)).round().astype(int), 0, frame.shape[1] - 1)
    py = np.clip((cy - radius * np.cos(steps)).round().astype(int), 0, frame.shape[0] - 1)
    s = frame[py, px].astype(np.int16)
    on = np.flatnonzero((s[:, 1] > 200) & (s[:, 0] > 140) & (s[:, 2] < 120))
    if len(on) < 5 or on[0] > 50:
        return None
    gaps = np.flatnonzero(np.diff(on) > 5)
    return float((on[gaps[0]] if len(gaps) else on[-1]) * 0.1)


class OnTime(BaseAgent):
    """Un toque arranca un cronómetro y otro lo para: gana quien más se acerque a 7 s exactos.

    El toque que arranca lo da el agente (no el genérico de BaseAgent.start_game, que en este juego ya
    arrancaría el cronómetro sin saber cuándo). Pero no cuenta cuando se envía sino cuando le llega al juego, con
    un retraso que varía de un toque a otro. Por eso no se para a ciegas 7 s después: en cada captura el arco
    verde dice cuánto tiempo lleva el cronómetro (360° = 7 s). Con todas las capturas se ajusta una recta
    ángulo-hora de captura y se calcula a qué hora se verá el arco completo; sin suponer la velocidad, porque en
    la partida humana el arco iba un 0,23 % más rápido que el reloj del PC (16 ms en 7 s). El toque que para
    sale entonces - `see_to_effect_s` (lo que tarda en contar un toque respecto a lo que se ve en pantalla; aquí
    casi 0, no los 0,033 s de CenterHit). Así solo queda la variación del retraso del segundo toque.

    Si no se llega a ver el arco, se para a ciegas `target_s` después de enviar el primer toque.
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._started_at: float | None = None  # cuándo se envió el toque que arranca
        self._ring: Ring | None = None
        self._readings: list[tuple[float, float]] = []  # (hora de captura, grados de arco)
        self._stopped = False

    def start_game(self) -> None:
        """Start, como en BaseAgent, pero sin el toque siguiente: aquí ese toque arranca el cronómetro."""
        for attempt in range(3):
            self.device.tap(*config.UI_START)
            time.sleep(self.params["after_start_s"])
            if find_template(self.device.screenshot(), _start_template(), threshold=0.9) is None:
                return
            log.warning("El botón Start sigue visible: reintento %d", attempt + 1)

    def step(self, frame: np.ndarray) -> None:
        if self._stopped:
            return
        if self._started_at is None:
            self.device.tap(*self.params["tap"])
            self._started_at = time.time()
            log.info("Toque de arranque enviado")
            return
        self._observe(frame)
        target = self._stop_time()
        if time.time() < target - self.params["schedule_ahead_s"]:
            return
        # Espera activa (sleep() en Windows tiene ~15 ms de resolución). Desde main.py el agente va en un hilo
        # aparte de la ventana: sin subir el intervalo de cambio de hilo, el de la ventana podría quitarle el
        # turno (GIL) justo antes del toque y retrasarlo unos ms. Con run_logged no hay ventana.
        switch = sys.getswitchinterval()
        sys.setswitchinterval(1.0)
        try:
            while time.time() < target:
                pass
            self.device.tap(*self.params["tap"])
            sent = time.time()
        finally:
            sys.setswitchinterval(switch)
        self._stopped = True
        fit = self._fit()
        if fit is not None:
            rate, seen = fit
            log.info("Parado: arranque visto %.1f ms tras enviar el primer toque, arco a %.3f°/s (%+.2f %% sobre "
                     "360/%g), %d capturas; segundo toque a %.4f s del arranque visto (%+.1f ms sobre lo previsto)",
                     (seen - self._started_at) * 1000, rate, (rate / (360 / self.params["target_s"]) - 1) * 100,
                     self.params["target_s"], len(self._readings), sent - seen, (sent - target) * 1000)
        else:
            log.warning("Sin ver el arco: parado a ciegas %.4f s tras el primer toque", sent - self._started_at)

    def _observe(self, frame: np.ndarray) -> None:
        if self._ring is None:
            self._ring = find_ring(frame, self.params)
            if self._ring is None:
                return
            log.info("Anillo en (%.1f, %.1f), radio %.1f", *self._ring)
        angle = arc_angle(frame, self._ring)
        lo, hi = self.params["angle_range"]
        if angle is not None and lo <= angle <= hi:
            self._readings.append((self.frame_time, angle))

    def _fit(self) -> tuple[float, float] | None:
        """(grados por segundo, hora de captura en que el arco marcaba 0) ajustando una recta a las lecturas;
        se repite sin las que se alejan más de `outlier_deg` (capturas a medio pintar)."""
        if len(self._readings) < self.params["min_readings"]:
            return None
        t, a = np.array(self._readings).T
        t0 = t[0]
        keep = np.ones(len(t), bool)
        for _ in range(2):
            k, c = np.polyfit(t[keep] - t0, a[keep], 1)
            keep = np.abs(a - (k * (t - t0) + c)) <= self.params["outlier_deg"]
            if keep.sum() < self.params["min_readings"]:
                return None
        return float(k), float(t0 - c / k)

    def _stop_time(self) -> float:
        p = self.params
        fit = self._fit()
        if fit is not None:
            rate, start = fit
            return start + 360 / rate - p["see_to_effect_s"]  # cuando se vería el arco completo
        return self._started_at + p["target_s"]  # a ciegas: misma latencia en los dos toques, de media
