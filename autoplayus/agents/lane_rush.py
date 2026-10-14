import logging
import time
from typing import Any

import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

LANES = 3
CENTER = 1


Geometry = tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]


def corridors(params: dict[str, Any]) -> Geometry:
    """Filas de la carretera y, por carril, los píxeles de la franja central de cada fila.

    La carretera está en perspectiva y curvada: el semiancho de carril (distancia del centro de la pantalla a
    cada línea divisoria) se interpola de la tabla `half_lane` medida sobre las capturas. El carril central
    va en x=360 y los laterales a dos semianchos; la franja abarca `corridor` semianchos y no toca las líneas.
    Por carril se guardan (filas, y de cada píxel, x de cada píxel, índice del primer píxel de cada fila) para
    leer la franja entera de una vez.
    """
    table = np.array(params["half_lane"], float)
    ys = np.arange(params["horizon_y"], 1280, params["row_step"])
    half = np.interp(ys, table[:, 0], table[:, 1])
    x_min, x_max = params["play_x"]
    lanes = []
    for side in (-2, 0, 2):
        x0 = np.clip((360 + side * half - params["corridor"] * half).astype(int), x_min, x_max)
        x1 = np.clip((360 + side * half + params["corridor"] * half).astype(int), x_min, x_max)
        keep = x1 - x0 >= 4  # al fondo los carriles laterales se salen de la pantalla
        rows, x0, x1 = ys[keep], x0[keep], x1[keep]
        lengths = x1 - x0
        px_y = np.repeat(rows, lengths)
        px_x = np.concatenate([np.arange(a, b) for a, b in zip(x0, x1)])
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        lanes.append((rows, px_y, px_x, starts))
    return ys, lanes


def find_player(frame: np.ndarray, params: dict[str, Any]) -> int | None:
    """x del coche del jugador (verde azulado), o None si no se ve (tutorial, choque)."""
    y0, y1 = params["player_rows"]
    b, g, r = (frame[y0:y1:4, :, c].astype(np.int16) for c in range(3))
    xs = np.nonzero((g > 140) & (g - r > 60) & (b > 100) & (g - b > 10))[1]
    return int(np.median(xs)) if len(xs) >= params["player_min_px"] else None


def lane_dangers(frame: np.ndarray, params: dict[str, Any], player_x: int, geometry: Geometry) -> list[int]:
    """Por carril, y inferior del coche rival más cercano (más abajo = más cerca); 0 si el carril está libre.

    Una fila de la franja está ocupada si gran parte de sus píxeles no son asfalto. En los carriles que pisa el
    coche del jugador (uno si está quieto, dos mientras cambia) solo se mira por encima de él; en los demás
    hasta abajo, porque un coche a su altura impide pasar.
    """
    road = np.array(params["road_bgr"], np.int16)
    dangers = []
    for lane, (rows, px_y, px_x, starts) in enumerate(geometry[1]):
        not_road = (np.abs(frame[px_y, px_x].astype(np.int16) - road).max(axis=1) > params["road_tol"]).astype(np.int32)
        fill = np.add.reduceat(not_road, starts) / np.diff(np.append(starts, len(px_y)))
        occupied = rows[fill > params["min_fill"]]
        if abs(params["player_lanes_x"][lane] - player_x) < params["player_reach_px"]:
            occupied = occupied[occupied < params["own_lane_max_y"]]
        dangers.append(int(occupied.max()) if len(occupied) else 0)
    return dangers


def time_to_reach(y: int, params: dict[str, Any], speed: float) -> float:
    """s hasta que un coche rival con el borde inferior en `y` llegue a la altura del jugador (negativo si ya
    está a su lado). `approach_s` da ese tiempo a la velocidad de referencia (la del principio de la partida);
    todos los coches van a la misma velocidad, así que a otra velocidad basta dividir por `speed`."""
    table = np.array(params["approach_s"], float)
    return float(np.interp(y, table[:, 0], table[:, 1])) / speed


def choose_lane(
    lane: int, dangers: list[int], times: list[float], params: dict[str, Any], wave_in: float = float("inf")
) -> int:
    """Carril al que ir (el actual si no hay que moverse).

    Los coches llegan en oleadas de uno o dos (nunca tres) y, a velocidad alta, un cambio de carril dura casi
    todo el tiempo que tarda un coche en llegar: solo da tiempo a esquivar un carril. Desde el centro cualquier
    oleada se esquiva con un cambio; desde un lateral, una pareja en ese lateral y el central obliga a cruzar
    dos. Pero moverse también cuesta: mientras dura el cambio no se puede reaccionar a una oleada nueva.

    - Esquivar: si viene un coche por el carril actual se elige el carril cuyo coche
      más cercano esté más lejos (o libre) y más lejos que el del actual, siempre que en él y en los que hay
      que cruzar ningún coche llegue a la altura del jugador antes de `min_ttc` s (lo que dura el cambio con
      margen). A igualdad, el central (mejor posición para la oleada siguiente) y luego el más cercano. Si
      ninguno cumple, quedarse es chocar seguro: se intenta igualmente con el margen mínimo `escape_ttc`.
      Desde el centro se espera a que el coche baje hasta `decide_y`, porque a veces viene otro a la par por un
      lateral que asoma un instante después y eso decide el lado. Desde un lateral no: el primer paso es hacia
      el centro en cualquier caso (si luego resulta ser una pareja, LaneRush.step amplía el cambio).
    - Recolocarse en el centro (solo con `return_to_center`) si está libre y el cambio da tiempo a hacerse
      antes de la oleada siguiente (`wave_in` s como pronto, ver WaveClock): si no, es mejor quedarse y
      conservar la reacción. En la práctica, moverse sin necesidad ha costado más partidas de las que ha salvado.
    """
    if dangers[lane] >= params["decide_y"] or (lane != CENTER and dangers[lane]):
        order = sorted(range(LANES), key=lambda t: (dangers[t], t != CENTER, abs(t - lane)))
        for margin in (params["min_ttc"], params["escape_ttc"]):
            for target in order:
                if target == lane or dangers[target] >= dangers[lane]:
                    continue
                step = 1 if target > lane else -1
                if all(times[p] > margin for p in range(lane + step, target + step, step)):
                    return target
        return lane
    if params["return_to_center"] and lane != CENTER and dangers[lane] == 0 and dangers[CENTER] == 0 and wave_in > params["reposition_s"]:
        return CENTER
    return lane


class WaveClock:
    """Ritmo de las oleadas de coches rivales, para saber cuándo puede asomar la siguiente.

    Una oleada empieza cuando asoma un coche (aparece en un carril, por encima de `appear_y`); los que asoman
    menos de `wave_join_s` después son de la misma oleada (la pareja). El intervalo entre oleadas se acorta
    con la velocidad y varía algo: la siguiente se espera, como pronto, `wave_gap_factor` veces el menor de los
    tres últimos intervalos después de la última.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self._prev = [0] * LANES
        self._last: float | None = None
        self._gaps: list[float] = []

    def update(self, t: float, dangers: list[int]) -> None:
        for lane, y in enumerate(dangers):
            prev = self._prev[lane]
            if 0 < y <= self.params["appear_y"] and (prev == 0 or y < prev - 40):
                if self._last is None or t - self._last > self.params["wave_join_s"]:
                    if self._last is not None:
                        self._gaps.append(t - self._last)
                    self._last = t
        self._prev = list(dangers)

    def time_to_next(self, t: float) -> float:
        """s hasta que, como pronto, asome la oleada siguiente (0 si ya puede asomar)."""
        if self._last is None or not self._gaps:
            return float("inf")
        return max(0.0, self._last + self.params["wave_gap_factor"] * min(self._gaps[-3:]) - t)


class SpeedMeter:
    """Velocidad del juego respecto a la de referencia de `approach_s`, medida siguiendo a los coches rivales.

    Por carril se guarda un ancla (instante, y del coche más cercano); pasados `speed_window` s se compara cuánto
    tiempo de aproximación ha recorrido el coche con el tiempo real transcurrido. Si el coche más cercano
    cambia (sube la y), se reinicia el ancla. El juego solo acelera: se suaviza con una media exponencial y
    nunca baja de la de referencia.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.speed = 1.0
        self._anchors: list[tuple[float, int] | None] = [None] * LANES

    def update(self, t: float, dangers: list[int]) -> None:
        for lane, y in enumerate(dangers):
            anchor = self._anchors[lane]
            if not y or y >= self.params["player_rows"][0]:
                self._anchors[lane] = None
            elif anchor is None or y < anchor[1]:
                self._anchors[lane] = (t, y)
            elif t - anchor[0] >= self.params["speed_window"]:
                ran = time_to_reach(anchor[1], self.params, 1.0) - time_to_reach(y, self.params, 1.0)
                sample = ran / (t - anchor[0])
                self.speed = max(1.0, 0.7 * self.speed + 0.3 * sample)
                self._anchors[lane] = (t, y)


class LaneRush(BaseAgent):
    """3 carriles; el coche del jugador solo cambia de carril y por ellos vienen coches de frente cada vez más
    rápido. Chocar con uno termina la partida.

    Se mide en cada carril lo cerca que está el coche rival más próximo. Si viene uno por el carril del jugador,
    se toca el lado de la pantalla hacia el carril más despejado (un carril por toque). Para cruzar dos, el
    segundo toque sale `chain_gap_s` después del primero, sin esperar a ver el coche a mitad de camino: el
    juego acepta toques durante la animación y a velocidad alta cada milisegundo cuenta. Un coche rival tarda ~0,4 s del horizonte al jugador al
    principio y ~0,23 s a x1,8; como el juego acelera, para saber si da tiempo a entrar en un carril se mide la
    velocidad siguiendo a los coches (SpeedMeter) en vez de usar una altura fija, y para saber si da tiempo a
    recolocarse se mide el ritmo de las oleadas (WaveClock). La decisión está en choose_lane().

    Start, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._geometry = corridors(self.params)
        self._speed = SpeedMeter(self.params)
        self._waves = WaveClock(self.params)
        self._target: int | None = None  # carril al que se está cambiando
        self._origin = 0  # carril de partida de ese cambio
        self._move_deadline = 0.0
        self._taps_left = 0  # toques pendientes para cruzar dos carriles
        self._next_tap_at = 0.0
        self._tap_dir = 0
        self._last_start_tap = 0.0
        self._playing = False  # ya se vio el coche del jugador: la pantalla oscura no es el tutorial sino el final

    def step(self, frame: np.ndarray) -> None:
        now = time.time()
        if not self._playing and frame[300, 100].max() < self.params["dim_max"]:
            # Tutorial ("Desliza o toca los lados..."): la pantalla está oscurecida hasta el primer gesto.
            if now - self._last_start_tap > 1.0:
                self.device.tap(*self.params["tap_right"])
                self._last_start_tap = now
            return
        x = find_player(frame, self.params)
        if x is None:
            return
        self._playing = True
        dists = [abs(x - c) for c in self.params["player_lanes_x"]]
        lane = int(np.argmin(dists))
        settled = dists[lane] <= self.params["settle_px"]
        dangers = lane_dangers(frame, self.params, x, self._geometry)
        self._speed.update(self.frame_time, dangers)
        self._waves.update(self.frame_time, dangers)
        if self._target is not None:
            if self._taps_left and now >= self._next_tap_at:
                self._tap(self._tap_dir)
                self._taps_left -= 1
                self._move_deadline = now + self.params["move_timeout"]
            elif self._should_extend(dangers):
                # Se salía de un lateral hacia el centro y por el centro asoma otro coche: seguir hasta el otro lado.
                self._target += self._tap_dir
                self._tap(self._tap_dir)
                self._move_deadline = now + self.params["move_timeout"]
                log.info("Pareja: se amplía el cambio hasta el carril %d (rivales y=%s)", self._target, dangers)
                return
            # Hasta pasar la mitad del cambio pedido no se vuelve a decidir; pasada, ya se puede encadenar otro.
            # Si el coche se queda quieto en un carril intermedio (el juego no hizo caso a un toque), también.
            arrived = lane == self._target and dists[lane] <= self.params["arrive_px"]
            stuck = settled and lane != self._origin
            if self._taps_left or (now < self._move_deadline and not arrived and not stuck):
                return
            self._target = None
        elif not settled:
            return
        times = [time_to_reach(y, self.params, self._speed.speed) if y else float("inf") for y in dangers]
        wave_in = self._waves.time_to_next(self.frame_time)
        target = choose_lane(lane, dangers, times, self.params, wave_in)
        if target == lane:
            return
        self._tap_dir = 1 if target > lane else -1
        self._tap(self._tap_dir)
        self._target, self._origin, self._move_deadline = target, lane, now + self.params["move_timeout"]
        self._taps_left, self._next_tap_at = abs(target - lane) - 1, now + self.params["chain_gap_s"]
        log.info(
            "Carril %d -> %d (rivales y=%s, velocidad x%.2f, oleada en %.2f s)",
            lane, target, dangers, self._speed.speed, wave_in,
        )

    def _tap(self, direction: int) -> None:
        self.device.tap(*self.params["tap_right" if direction > 0 else "tap_left"])

    def _should_extend(self, dangers: list[int]) -> bool:
        """Durante un cambio de un lateral al centro, ¿hay que seguir hasta el otro lateral? Sí si por el
        centro viene un coche (a la par del que se esquiva o antes) y el otro lateral está libre."""
        if self._target != CENTER or self._origin == CENTER or self._taps_left:
            return False
        far = CENTER + self._tap_dir
        return 0 < dangers[CENTER] < self.params["own_lane_max_y"] and dangers[far] == 0
