import logging
from collections import deque
from typing import Any

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

Spans = list[tuple[int, int]]  # (y arriba, y abajo) de la base de cada pincho de una pared


def find_box(frame: np.ndarray, params: dict[str, Any]) -> tuple[float, float] | None:
    """Centro (x, y) de la caja blanca, o None si no se ve (pantalla de inicio o final).

    Se mira la zona de juego a mitad de resolución. La estela y los dígitos del marcador también son blancos,
    pero son trazos más finos que la caja (lado ~71 px): una apertura morfológica los borra y deja solo la caja."""
    x0, y0, x1, y1 = params["arena"]
    sub = frame[y0:y1:2, x0:x1:2]
    white = (sub.min(axis=2) > params["white_min"]).astype(np.uint8)
    k = params["open_px"] // 2
    core = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(core)
    if n < 2:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lo, hi = params["box_area_px"]
    if not lo <= stats[j, cv2.CC_STAT_AREA] * 4 <= hi:
        return None
    return float(cents[j][0] * 2 + x0), float(cents[j][1] * 2 + y0)


def read_spikes(frame: np.ndarray, side: int, params: dict[str, Any]) -> Spans | None:
    """Pinchos de una pared (side -1 izquierda, +1 derecha) leídos en una columna pegada a ella: cada pincho es
    un triángulo rojo con la base en la pared, de ~70 px de alto. None si alguno está a medias (al rebotar en
    una pared sus pinchos se esconden y salen otros nuevos durante ~0,25 s)."""
    x = params["spike_col"][0 if side < 0 else 1]
    _, y0, _, y1 = params["arena"]
    col = frame[y0:y1, x - 2:x + 2].astype(np.int16)
    b, g, r = col[..., 0], col[..., 1], col[..., 2]
    red = ((r > 180) & (g < 110) & (b < 110)).any(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], red.astype(np.int8), [0]))))
    spans = [(int(a) + y0, int(e) - 1 + y0) for a, e in zip(edges[::2], edges[1::2])]
    lo, hi = params["spike_len"]
    spans = [s for s in spans if s[1] - s[0] >= params["spike_min_len"]]
    if any(not lo <= s[1] - s[0] <= hi for s in spans):
        return None
    return spans


def box_corners(x: float | np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Esquinas de la caja respecto a su centro cuando está en x: (4, 2), o (len(x), 4, 2) para varias x.

    El giro solo depende de x: `rot_deg_per_px` grados por px de avance, recta (0°, módulo 90) en x = `rot_x0`,
    así que llega recta a las dos paredes (178° por cruce). Medido en vivo con error de ~2°."""
    a = np.radians(params["rot_deg_per_px"] * (np.asarray(x, float) - params["rot_x0"]))
    h = params["box_half"]
    c, s = np.cos(a)[..., None], np.sin(a)[..., None]
    bx = np.array([-h, h, h, -h])
    by = np.array([-h, -h, h, h])
    return np.stack((bx * c - by * s, bx * s + by * c), axis=-1)


def spike_triangle(side: int, span: tuple[int, int], params: dict[str, Any]) -> np.ndarray:
    """Vértices (3x2) de un pincho: base `span` sobre la pared y punta `spike_depth` px hacia dentro."""
    wall = params["wall_x"][0 if side < 0 else 1]
    a, b = span
    return np.array([[wall, a], [wall, b], [wall - side * params["spike_depth"], (a + b) / 2]], float)


def spike_clearance(x: float, y: np.ndarray, side: int, spans: Spans, params: dict[str, Any]) -> np.ndarray:
    """Distancia (px) entre la caja (cuadrado girado, ver box_corners) con centro en (x, y[i]) y el pincho más
    cercano de una pared, menos `hit_margin`; negativa si se solapan (cuánto).

    El juego choca con el cuadrado girado: un círculo de radio media diagonal daba por imposibles huecos de 85 px
    entre pinchos por los que la caja, recta al llegar a la pared (71 px), pasa. Distancia exacta entre dos
    polígonos convexos: si el teorema del eje separador encuentra un eje con hueco, la distancia es la menor
    entre vértices de uno y lados del otro; si no, se solapan y vale la menor penetración."""
    return _wall_clearance(np.array([float(x)]), np.asarray(y, float), side, spans, params)[0]


def _wall_clearance(xs: np.ndarray, ys: np.ndarray, side: int, spans: Spans, params: dict[str, Any]) -> np.ndarray:
    """spike_clearance en la rejilla xs x ys de una vez (len(xs), len(ys)). Para cada pincho solo se calculan
    las filas y donde la holgura puede quedar por debajo del tope `clear_cap`; el resto queda en infinito."""
    corners = box_corners(xs, params)  # (R, 4, 2)
    reach = params["box_half"] * np.sqrt(2) + params["clear_cap"] + params["hit_margin"]
    out = np.full((len(xs), len(ys)), np.inf)
    for span in spans:
        tri = spike_triangle(side, span, params)
        cols = np.flatnonzero((ys >= tri[:, 1].min() - reach) & (ys <= tri[:, 1].max() + reach))
        if len(cols):
            gap = _polygon_gap(corners, xs[:, None], ys[None, cols], tri)
            out[:, cols] = np.minimum(out[:, cols], gap)
    return out - params["hit_margin"]


def _polygon_gap(corners: np.ndarray, x: np.ndarray, y: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distancia (o -penetración) entre la caja de esquinas `corners` (R, 4, 2) centrada en (x[r], y[c]) y el
    triángulo `tri` (3, 2); resultado (R, C)."""
    sep = np.full(np.broadcast_shapes(x.shape, y.shape), -np.inf)
    edges = np.roll(corners, -1, axis=1) - corners
    box_n = np.stack((-edges[:, :2, 1], edges[:, :2, 0]), axis=-1)  # (R, 2, 2): los dos ejes del cuadrado
    box_n /= np.linalg.norm(box_n, axis=-1, keepdims=True)
    tri_e = np.roll(tri, -1, axis=0) - tri
    tri_n = np.stack((-tri_e[:, 1], tri_e[:, 0]), axis=1)
    tri_n /= np.linalg.norm(tri_n, axis=1, keepdims=True)
    for k in range(2):
        n = box_n[:, k, :]  # (R, 2)
        pa = np.einsum("rvd,rd->rv", corners, n)
        pb = tri @ n.T  # (3, R)
        c = x * n[:, :1] + y * n[:, 1:]
        sep = np.maximum(sep, np.maximum(pb.min(0)[:, None] - (c + pa.max(1)[:, None]),
                                         (c + pa.min(1)[:, None]) - pb.max(0)[:, None]))
    for n in tri_n:
        pa = corners @ n  # (R, 4)
        pb = tri @ n
        c = x * n[0] + y * n[1]
        sep = np.maximum(sep, np.maximum(pb.min() - (c + pa.max(1)[:, None]), (c + pa.min(1)[:, None]) - pb.max()))
    dist = np.full(sep.shape, np.inf)
    for k in range(4):  # vértices de la caja contra lados del pincho
        px, py = x + corners[:, k, :1], y + corners[:, k, 1:]
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            dist = np.minimum(dist, _seg_dist(px, py, a[0], a[1], b[0], b[1]))
    for v in tri:  # vértices del pincho contra lados de la caja
        for k in range(4):
            a, b = corners[:, k, :], corners[:, (k + 1) % 4, :]
            dist = np.minimum(dist, _seg_dist(v[0], v[1], x + a[:, :1], y + a[:, 1:], x + b[:, :1], y + b[:, 1:]))
    return np.where(sep > 0, dist, sep)


def _seg_dist(px, py, ax, ay, bx, by) -> np.ndarray:
    dx, dy = bx - ax, by - ay
    u = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - ax - u * dx, py - ay - u * dy)


def clearance_map(walls: dict[int, Spans | None], params: dict[str, Any]) -> np.ndarray:
    """Holgura (px) en cada punto de la zona de juego, cada `map_px` px: la menor entre techo y suelo (con el
    alto de la caja girada en esa x, menos `floor_margin`) y los pinchos conocidos (spike_clearance), con tope
    `clear_cap`. Como el giro solo depende de x, el mapa (x, y) es exacto. Los pinchos solo cambian al rebotar,
    así que cada pared se calcula una vez por combinación y el planificador solo consulta el mapa."""
    key = (tuple(walls.get(-1) or ()), tuple(walls.get(1) or ()))
    cached = _maps.get(key)
    if cached is not None:
        return cached
    x0, y0, x1, y1 = params["arena"]
    step = params["map_px"]
    ys = np.arange(y0, y1 + step, step, dtype=float)
    xs = np.arange(x0, x1 + step, step, dtype=float)
    reach = np.abs(box_corners(xs, params)[:, :, 1]).max(axis=1)[:, None]  # medio alto de la caja girada
    floor = np.minimum(ys - y0, y1 - ys)[None, :] - reach - params["floor_margin"]
    out = np.minimum(floor, params["clear_cap"])
    for side in (-1, 1):
        spans = walls.get(side)
        if spans:
            out = np.minimum(out, _wall_map(side, tuple(spans), xs, ys, params))
    if len(_maps) > 16:
        _maps.clear()
    _maps[key] = out
    return out


def _wall_map(side: int, spans: tuple, xs: np.ndarray, ys: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    key = (side, spans)
    cached = _wall_maps.get(key)
    if cached is not None:
        return cached
    wall = params["wall_x"][0 if side < 0 else 1]
    out = np.full((len(xs), len(ys)), np.inf)
    rows = np.flatnonzero(np.abs(xs - wall) <= params["near_wall_px"])
    out[rows] = _wall_clearance(xs[rows], ys, side, list(spans), params)
    if len(_wall_maps) > 16:
        _wall_maps.clear()
    _wall_maps[key] = out
    return out


_wall_maps: dict[tuple, np.ndarray] = {}
_maps: dict[tuple, np.ndarray] = {}


def physics(score: int, params: dict[str, Any]) -> tuple[float, float, float]:
    """(vx, g, v_salto) esperados con `score` puntos: el juego acelera en cada rebote hasta los 25 puntos y
    luego ya no (medido en vivo hasta 37). Ajuste lineal: vx 250 -> 558 px/s, g 860 -> 1845 px/s², salto
    -468 -> -775 px/s."""
    s = min(score, params["max_score_fit"])
    vx = params["vx"][0] + params["vx"][1] * s
    g = params["gravity"][0] + params["gravity"][1] * s
    jump = params["jump"][0] + params["jump"][1] * s
    return vx, g, jump


def plan_tap(x: float, y: float, vy: float, direction: int, vx: float, g: float, jump: float,
             walls: dict[int, Spans | None], params: dict[str, Any]) -> tuple[bool, float]:
    """¿Tocar ahora? Busca, en pasos de `plan_dt` s, la secuencia de toques que deja más holgura mínima.

    El estado (x, y, vy) es el del instante en que surtiría efecto un toque dado ahora. x no depende de los
    toques: avanza a `vx` y rebota en las paredes. y sigue la gravedad y cada toque fija vy = `jump`. Así
    que basta con programación dinámica sobre (y, vy): en cada paso cada estado se abre en "toca" y "no toca",
    los estados casi iguales se funden quedándose con el mejor, y la nota de un camino es su peor holgura
    (pinchos, techo y suelo; ver clearance_map, con tope para no preferir sin motivo alejarse más de lo
    necesario).

    El horizonte llega hasta el choque con la pared a la que va la caja; si ya se conocen los pinchos de la
    otra, hasta el choque con ella; si no, `after_contact_s` más. Devuelve (tocar, holgura del mejor plan)."""
    dt = params["plan_dt"]
    contact = params["contact_x"]

    # trayectoria horizontal y fin del horizonte
    xs = []
    cx, d, t, bounces = x, direction, 0.0, 0
    end_after: float | None = None
    while t < params["max_horizon_s"]:
        cx += d * vx * dt
        t += dt
        target = contact[1] if d > 0 else contact[0]
        if (cx - target) * d >= 0:
            cx = 2 * target - cx
            bounces += 1
            if bounces == 1 and walls.get(-d) is None:
                end_after = t + params["after_contact_s"]
            d = -d
            if bounces == 2:
                end_after = t + params["after_contact_s"]
        xs.append(cx)
        if end_after is not None and t >= end_after:
            break

    cmap = clearance_map(walls, params)
    step = params["map_px"]
    x0, y0 = params["arena"][0], params["arena"][1]
    last_row = cmap.shape[1] - 1
    ys = np.array([y])
    vys = np.array([vy])
    value = np.array([np.inf])
    first = np.array([-1])
    tap_k = params["tap_bins"]
    for k, cx in enumerate(xs):
        # cada estado: no tocar / tocar al empezar el paso. Más allá del primero solo se puede tocar cada
        # `tap_every` pasos: el plan se rehace en cada captura, así que solo el primer toque necesita precisión
        if k == 0 or k % params["tap_every"] == 0:
            vy0 = np.concatenate((vys, np.full(vys.shape, jump)))
            ys2 = np.concatenate((ys, ys))
            nfirst = np.concatenate((first, first)) if k else np.array([0, 1])
            prev = np.concatenate((value, value))
        else:
            vy0, ys2, nfirst, prev = vys, ys, first, value
        ny = ys2 + vy0 * dt + 0.5 * g * dt * dt
        nvy = vy0 + g * dt
        row = cmap[min(max(int(round((cx - x0) / step)), 0), cmap.shape[0] - 1)]
        cell = np.round((ny - y0) / step).astype(np.int64)
        clear = np.where((cell < 0) | (cell > last_row), -np.inf, row[np.clip(cell, 0, last_row)])
        nvalue = np.minimum(prev, clear)
        # fundir estados casi iguales (mejor nota primero) y descartar los que ya chocan sin remedio
        key = np.round(ny / tap_k[0]).astype(np.int64) * 100003 + np.round(nvy / tap_k[1]).astype(np.int64)
        order = np.lexsort((-nvalue, key))
        keep = order[np.concatenate(([True], key[order][1:] != key[order][:-1]))]
        best = nvalue[keep].max()
        keep = keep[nvalue[keep] >= min(best, params["dead_clear"])]
        if len(keep) > params["max_states"]:
            keep = keep[np.argsort(-nvalue[keep])[:params["max_states"]]]
        ys, vys, value, first = ny[keep], nvy[keep], nvalue[keep], nfirst[keep]
    if not len(value):
        return False, -np.inf
    best = value.max()
    # entre planes igual de buenos, no tocar (el siguiente fotograma vuelve a decidir con más datos)
    tap_best = value[first == 1].max(initial=-np.inf)
    wait_best = value[first == 0].max(initial=-np.inf)
    return bool(tap_best > wait_best + params["tap_gain_px"]), float(best)


class WallTracker:
    """Pinchos conocidos de cada pared. Al rebotar en una pared sus pinchos cambian: se olvidan y se vuelven a
    leer cuando la lectura está completa y se repite `stable_frames` fotogramas (sin pinchos solo se acepta
    pasados `empty_after_s`, porque al cambiar hay un momento en que no se ve ninguno)."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.known: dict[int, Spans | None] = {-1: None, 1: None}
        self._seen: dict[int, tuple[Spans, int]] = {}
        self._reset_at = {-1: -1e9, 1: -1e9}

    def forget(self, side: int, t: float) -> None:
        self.known[side] = None
        self._seen.pop(side, None)
        self._reset_at[side] = t

    def update(self, frame: np.ndarray, t: float) -> None:
        for side in (-1, 1):
            if self.known[side]:
                continue  # una pared sin pinchos se sigue leyendo: al empezar la partida aún no han salido
            spans = read_spikes(frame, side, self.params)
            if spans is None or (not spans and t - self._reset_at[side] < self.params["empty_after_s"]):
                self._seen.pop(side, None)
                continue
            prev = self._seen.get(side)
            same = prev is not None and len(prev[0]) == len(spans) and all(
                abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2 for a, b in zip(prev[0], spans))
            count = prev[1] + 1 if same else 1
            self._seen[side] = (spans, count)
            if count >= self.params["stable_frames"] and t - self._reset_at[side] >= self.params["regen_s"]:
                self.known[side] = spans
                log.info("Pinchos %s: %s", "izq" if side < 0 else "der", [(a + b) // 2 for a, b in spans])


class DangerWall(BaseAgent):
    """Una caja cruza de pared a pared rebotando; cada toque la hace saltar (fija su velocidad vertical) y la
    gravedad la baja. Chocar con un pincho de las paredes, o con el techo o el suelo, acaba la partida. Cada
    rebote es un punto, cambia los pinchos de la pared tocada y acelera el juego.

    En cada captura: se localiza la caja (find_box), se estima su velocidad vertical con las posiciones desde
    el último toque y la gravedad esperada, se lleva el estado hasta el instante en que surtiría efecto un
    toque dado ahora (`tap_latency_s`, con los toques ya enviados que aún no se ven) y plan_tap decide si
    tocar. La puntuación (y con ella vx, g y el salto, ver physics) se cuenta por rebotes; vx se mide además
    en cada tramo, porque es lo que más fácil se ve.

    Start, primer toque, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0

    def on_start(self) -> None:
        self._walls = WallTracker(self.params)
        self._obs: deque[tuple[float, float, float]] = deque(maxlen=200)  # (t, x, y) de capturas distintas
        self._taps: list[float] = []  # instante en que surte efecto cada toque enviado
        self._dir = 0
        self._score = 0
        self._leg_start = 0.0
        self._last_plan_log = -1e9
        self._tap_scores: list[int] = []  # puntos al dar cada toque
        self._calibrated = 0  # toques cuyo tramo siguiente ya se usó para calibrar
        self._g_ratio: deque[float] = deque(maxlen=self.params["calib_segments"])
        self._jump_ratio: deque[float] = deque(maxlen=self.params["calib_segments"])

    def step(self, frame: np.ndarray) -> None:
        now = self.frame_time
        box = find_box(frame, self.params)
        if box is None:
            return
        x, y = box
        if self._obs and abs(x - self._obs[-1][1]) < 0.5 and abs(y - self._obs[-1][2]) < 0.5:
            pass  # captura repetida: el juego aún no ha pintado otro fotograma
        else:
            self._obs.append((now, x, y))
            self._track_direction(now)
        self._walls.update(frame, now)
        if self._dir == 0:
            return

        vx_prior, g, jump = physics(self._score, self.params)
        self._calibrate(now)
        g *= self._ratio(self._g_ratio)
        jump *= self._ratio(self._jump_ratio)
        vx = self._measured_vx(now) or vx_prior
        lat = self.params["tap_latency_s"]
        t_eff = now + lat
        state = self._state_at(t_eff, g, jump)
        if state is None:
            return
        sy, svy = state
        sx, sdir = self._advance_x(x, now, t_eff, vx)
        tap, clear = plan_tap(sx, sy, svy, sdir, vx, g, jump, self._walls.known, self.params)
        if tap:
            self.device.tap(*self.params["tap"])
            self._taps.append(t_eff)
            self._tap_scores.append(self._score)
            log.info("Toque: caja (%.0f, %.0f) vy %.0f -> efecto en (%.0f, %.0f) vy %.0f, holgura %.0f, puntos %d",
                     x, y, self._vy_now(now, g, jump) or 0, sx, sy, svy, clear, self._score)
        elif now - self._last_plan_log > 0.5:
            self._last_plan_log = now
            log.info("Caja (%.0f, %.0f) dir %+d vx %.0f, holgura %.0f, puntos %d", x, y, self._dir, vx, clear,
                     self._score)

    # --- estimación del estado ---

    def _track_direction(self, now: float) -> None:
        """Dirección horizontal por el avance en los últimos `dir_window_s` s (entre dos capturas seguidas el
        giro de la caja mueve el centro unos px); al invertirse hay rebote: un punto más y los pinchos de esa
        pared cambian."""
        past = [x for t, x, _ in self._obs if t >= now - self.params["dir_window_s"]]
        if len(past) < 2:
            return
        dx = past[-1] - past[0]
        if abs(dx) < self.params["dir_min_dx"]:
            return
        d = 1 if dx > 0 else -1
        if self._dir and d != self._dir:
            self._score += 1
            self._walls.forget(self._dir, now)
            self._leg_start = now
            log.info("Rebote en la pared %s: %d puntos", "der" if self._dir > 0 else "izq", self._score)
        self._dir = d

    def _calibrate(self, now: float) -> None:
        """Corrige g y el salto del ajuste con lo observado: el tramo entre dos toques ya terminado es una
        parábola cuya curvatura da g y cuya pendiente al surtir efecto el primero, el salto. Se guarda el cociente
        frente a physics() con los puntos de entonces."""
        p = self.params
        while self._calibrated + 1 < len(self._taps) and self._taps[self._calibrated + 1] <= now:
            k = self._calibrated
            self._calibrated += 1
            start, end = self._taps[k], self._taps[k + 1]
            pts = [(t, y) for t, _, y in self._obs if start + p["fit_after_tap_s"] <= t <= end - p["calib_margin_s"]]
            if len(pts) < p["calib_min_pts"] or pts[-1][0] - pts[0][0] < p["calib_min_span_s"]:
                continue
            t, y = np.array(pts).T
            a, b, _ = np.polyfit(t - start, y, 2)
            _, g0, jump0 = physics(self._tap_scores[k], p)
            self._g_ratio.append(2 * a / g0)
            self._jump_ratio.append(b / jump0)
            log.info("Calibración: g %.0f (ajuste %.0f), salto %.0f (ajuste %.0f)", 2 * a, g0, b, jump0)

    def _ratio(self, ratios: deque[float]) -> float:
        lo, hi = self.params["calib_clip"]
        return float(np.clip(np.median(ratios), lo, hi)) if ratios else 1.0

    def _measured_vx(self, now: float) -> float | None:
        pts = [(t, x) for t, x, _ in self._obs if t > self._leg_start + 0.05]
        if len(pts) < 6 or pts[-1][0] - pts[0][0] < self.params["vx_min_span_s"]:
            return None
        t, x = np.array(pts).T
        return float(abs(np.polyfit(t, x, 1)[0]))

    def _advance_x(self, x: float, t0: float, t1: float, vx: float) -> tuple[float, int]:
        lo, hi = self.params["contact_x"]
        nx, d = x + self._dir * vx * (t1 - t0), self._dir
        if nx > hi:
            nx, d = 2 * hi - nx, -1
        elif nx < lo:
            nx, d = 2 * lo - nx, 1
        return nx, d

    def _vy_now(self, now: float, g: float, jump: float) -> float | None:
        """vy en `now`: ajuste con la gravedad conocida de las posiciones vistas desde que surtió efecto el último
        toque (con margen); con pocas, la del modelo desde ese toque."""
        last = max((t for t in self._taps if t <= now), default=-1e9)
        since = last + self.params["fit_after_tap_s"]
        pts = [(t, y) for t, _, y in self._obs if t >= since and t >= now - self.params["vy_window_s"]]
        if len(pts) >= self.params["vy_min_pts"]:
            t, y = np.array(pts).T
            tau = t - now
            a, b = np.polyfit(tau, y - 0.5 * g * tau * tau, 1)
            return float(a)
        if last > -1e9:
            return jump + g * (now - last)
        return None

    def _state_at(self, t1: float, g: float, jump: float) -> tuple[float, float] | None:
        """(y, vy) previstos en t1 partiendo de la última captura y aplicando los toques pendientes."""
        t0, _, y = self._obs[-1]
        vy = self._vy_now(t0, g, jump)
        if vy is None:
            return None
        t = t0
        for tap in sorted(tp for tp in self._taps if t0 < tp <= t1):
            y += vy * (tap - t) + 0.5 * g * (tap - t) ** 2
            vy, t = jump, tap
        y += vy * (t1 - t) + 0.5 * g * (t1 - t) ** 2
        vy += g * (t1 - t)
        return y, vy
