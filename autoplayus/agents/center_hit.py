import logging
import time
from collections import deque
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)


def find_pointer(frame: np.ndarray, params: dict[str, Any]) -> float | None:
    """x del puntero (línea blanca vertical), o None si no se ve. Se mira la franja justo debajo de la barra:
    el puntero sobresale por abajo y ahí no hay nada más blanco (la línea fija del centro acaba en la barra)."""
    y0, y1 = params["pointer_rows"]
    band = frame[y0:y1].min(axis=(0, 2))
    xs = np.flatnonzero(band > params["white_min"])
    lo, hi = params["pointer_px"]
    return float(xs.mean()) if lo <= len(xs) <= hi else None


def score(x: float, params: dict[str, Any]) -> float:
    """Nota (%) de un toque con el puntero en x: 100 en el centro y baja 0,35 puntos por px (0 en el borde).
    Deducida de la partida humana: el % que muestra el juego es la media acumulada de los toques."""
    return max(0.0, 100.0 - params["pct_per_px"] * abs(x - params["center_x"]))


def next_crossing(obs: list[tuple[float, float]], params: dict[str, Any]) -> tuple[float, float] | None:
    """(instante en que el puntero pasará por el centro en la pasada actual, velocidad en px/s), o None.

    El puntero va de un extremo a otro a velocidad constante y rebota: con las posiciones de la pasada actual
    (desde el último cambio de sentido) basta una recta. Solo si va hacia el centro y la recta tiene datos
    suficientes (`fit_min_pts` posiciones en `fit_min_span_s` s)."""
    if len(obs) < params["fit_min_pts"]:
        return None
    t, x = np.array(obs).T
    if t[-1] - t[0] < params["fit_min_span_s"]:
        return None
    v, x0 = np.polyfit(t - t[-1], x, 1)
    c = params["center_x"]
    if abs(v) < params["min_speed"] or (c - x0) * v <= 0:
        return None  # parado o alejándose del centro
    return float(t[-1] + (c - x0) / v), float(abs(v))


class CenterHit(BaseAgent):
    """Una línea blanca va y viene por una barra horizontal; hay que tocar cuando pase por el centro. 5 toques y
    tras cada uno la línea acelera (552 -> 830 -> 1100 -> 1380 -> 1740 px/s); la nota es la precisión media.

    Truco: 5 toques a la vez en el primer intento cuentan los 5 a la velocidad inicial. El agente sigue el
    puntero (find_pointer), predice cuándo cruzará el centro en la pasada actual (next_crossing) y, a
    `tap_latency_s` de ese instante, espera sin dormir hasta el momento exacto y envía `burst` toques seguidos.
    Si la partida no termina (los toques no contaron todos), sigue con un toque en cada cruce por el centro.

    Start, primer toque (tutorial), detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._sweep: deque[tuple[float, float]] = deque(maxlen=200)  # (t, x) de la pasada actual
        self._dir = 0
        self._last_tap = -1e9
        self._bursts = 0
        self._aim: tuple[float, float, float] | None = None  # (instante del toque, cruce previsto, velocidad con signo)
        self._reported = False

    def step(self, frame: np.ndarray) -> None:
        now = self.frame_time
        x = find_pointer(frame, self.params)
        if x is None:
            return
        if self._report_stop(x):
            return
        self._track(now, x)
        if now - self._last_tap < self.params["cooldown_s"]:
            return
        pred = next_crossing(list(self._sweep), self.params)
        if pred is None:
            return
        t_cross, speed = pred
        t_tap = t_cross - self.params["tap_latency_s"]
        wait = t_tap - time.time()
        if wait > self.params["schedule_ahead_s"]:
            return  # aún falta: se decide con capturas posteriores (más datos)
        if wait < -self.params["late_ok_s"]:
            return  # demasiado tarde para este cruce
        while time.time() < t_tap:
            pass  # espera activa: sleep() en Windows tiene ~15 ms de resolución
        n = self.params["burst"] if self._bursts == 0 else 1
        for _ in range(n):
            self.device.tap(*self.params["tap"])
        self._bursts += 1
        self._last_tap = time.time()
        self._aim = (self._last_tap, t_cross, speed if self._dir > 0 else -speed)
        log.info("%d toque(s) con %+.1f ms de retraso sobre lo previsto: puntero visto en x=%.1f a %.0f px/s, "
                 "cruce por el centro en %.0f ms", n, (self._last_tap - t_tap) * 1000, x, speed,
                 (t_cross - now) * 1000)

    def _report_stop(self, x: float) -> bool:
        """Tras la ráfaga el puntero se queda quieto donde contó el toque: se registra el desvío respecto al
        centro y el `tap_latency_s` con el que se habría acertado (para afinarlo entre partidas)."""
        if self._aim is None or self._reported or not self._sweep or abs(x - self._sweep[-1][1]) >= 0.3:
            return False
        if time.time() - self._aim[0] < self.params["stop_check_s"]:
            return False  # aún se mueve o la captura es de antes del toque
        _, _, v = self._aim
        early_s = (self.params["center_x"] - x) / v  # > 0: contó antes de llegar al centro
        log.info("Puntero parado en x=%.1f (%+.1f px del centro, nota %.2f %%): el toque contó %.0f ms %s; "
                 "tap_latency_s acertado: %.3f", x, x - self.params["center_x"], score(x, self.params),
                 abs(early_s) * 1000, "antes" if early_s > 0 else "después",
                 self.params["tap_latency_s"] - early_s)
        self._reported = True
        return True

    def _track(self, t: float, x: float) -> None:
        """Guarda la posición en la pasada actual; al cambiar de sentido empieza una pasada nueva."""
        if self._sweep and abs(x - self._sweep[-1][1]) < 0.3:
            return  # captura repetida
        if self._sweep:
            d = 1 if x > self._sweep[-1][1] else -1
            if self._dir and d != self._dir:
                self._sweep.clear()
            self._dir = d
        self._sweep.append((t, x))
