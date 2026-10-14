import cv2

from autoplayus import config
from autoplayus.agents.lane_rush import LaneRush, SpeedMeter, WaveClock, choose_lane, corridors, find_player, lane_dangers, time_to_reach
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = load_catalog()["LaneRush"].params
    agent = LaneRush(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "LaneRush" / f"{name}.png"))


def test_perceives_player_and_rivals_on_real_capture():
    _, params = _agent()
    frame = _capture("two_cars")  # jugador en el centro; vienen coches por el izquierdo y el central
    assert abs(find_player(frame, params) - 360) < 10
    dangers = lane_dangers(frame, params, 360, corridors(params))
    assert dangers[2] == 0 and 820 < dangers[0] < 850 and 820 < dangers[1] < 850


def test_ignores_rival_in_other_lane_and_own_car():
    agent, params = _agent()
    frame = _capture("car_left")  # jugador en el centro, coche lejano por el izquierdo
    assert lane_dangers(frame, params, 360, corridors(params))[1] == 0
    agent.step(frame)
    assert agent.device.taps == []


def test_dodges_to_the_free_lane_once():
    agent, params = _agent()
    frame = _capture("two_cars")
    agent.step(frame)
    assert agent.device.taps == [tuple(params["tap_right"])]
    agent.step(frame)  # el coche aún no se ha movido: no repetir el toque
    assert len(agent.device.taps) == 1


def test_starts_the_game_from_the_tutorial():
    agent, params = _agent()
    agent.step(_capture("tutorial"))
    assert agent.device.taps == [tuple(params["tap_right"])]


def test_choose_lane():
    params = load_catalog()["LaneRush"].params
    free = float("inf")

    def choose(lane, dangers, times, wave_in=free, to_center=False):
        return choose_lane(lane, dangers, times, {**params, "return_to_center": to_center}, wave_in)

    assert choose(1, [0, 0, 0], [free] * 3) == 1  # nada en el carril: quedarse
    assert choose(1, [819, 819, 0], [0.4, 0.4, free]) == 1  # el rival acaba de asomar: esperar a ver si viene otro a la par
    assert choose(1, [0, 850, 0], [free, 0.25, free]) == 0  # ambos libres: a igualdad, cualquiera adyacente
    assert choose(1, [840, 850, 0], [0.28, 0.25, free]) == 2  # el coche del izquierdo está más cerca
    assert choose(0, [850, 850, 0], [0.25, 0.25, free]) == 2  # al derecho, cruzando el central (su coche tarda aún)
    assert choose(0, [850, 880, 0], [0.25, 0.15, free]) == 2  # quedarse es chocar: cruzar aunque vaya justo
    assert choose(0, [850, 950, 0], [0.25, 0.05, free]) == 0  # el del central ya está encima: no hay salida
    assert choose(0, [0, 0, 0], [free] * 3) == 0  # sin peligro: no moverse
    assert choose(0, [0, 0, 0], [free] * 3, to_center=True) == 1  # con return_to_center y tiempo de sobra: al centro
    assert choose(0, [0, 0, 0], [free] * 3, wave_in=0.05, to_center=True) == 0  # la oleada siguiente está al caer: no moverse
    assert choose(2, [0, 0, 820], [free, free, 0.4]) == 1  # asoma un rival por el propio lateral: hacia el centro ya
    assert choose(1, [0, 820, 0], [free, 0.4, free]) == 1  # en el centro sí se espera a ver si viene otro a la par
    assert choose(0, [0, 1200, 0], [free, -0.2, free]) == 0  # coche al lado en el central: esperar
    assert choose(0, [850, 0, 0], [0.25, free, free]) == 1  # dos libres: el central
    assert choose(0, [818, 0, 0], [0.4, free, free]) == 1  # desde un lateral no se espera: hacia el centro ya
    assert choose(2, [0, 0, 850], [free, free, 0.25]) == 1
    assert choose(2, [0, 1100, 850], [free, -0.1, 0.25]) == 2  # coche al lado en el central: no se puede salir


def test_speed_meter_measures_how_much_faster_cars_approach():
    params = load_catalog()["LaneRush"].params
    meter = SpeedMeter(params)
    # Un rival que recorre el tiempo de aproximación de referencia el doble de rápido
    for i in range(10):
        t = i * 0.05
        ys = [y for y in range(818, 1000) if time_to_reach(y, params, 1.0) <= 0.41 - 2 * t]
        meter.update(t, [0, ys[0] if ys else 0, 0])
    assert 1.7 < meter.speed < 2.1


def test_wave_clock_predicts_next_wave_from_recent_gaps():
    params = load_catalog()["LaneRush"].params
    clock = WaveClock(params)
    assert clock.time_to_next(0.0) == float("inf")
    for t, dangers in [(0.0, [820, 0, 0]), (0.03, [826, 822, 0]), (0.2, [900, 900, 0]), (0.5, [0, 0, 820])]:
        clock.update(t, dangers)  # pareja a t=0 (el segundo asoma después) y otra oleada a t=0,5
    assert abs(clock.time_to_next(0.5) - 0.8 * 0.5) < 1e-6
    assert clock.time_to_next(1.0) == 0.0


def test_crosses_two_lanes_with_two_quick_taps():
    agent, params = _agent()
    agent._geometry = None  # no se usa: se sustituye la percepción
    import autoplayus.agents.lane_rush as lr

    real = lr.lane_dangers, lr.find_player
    try:
        lr.find_player = lambda frame, params: 138  # jugador en el carril izquierdo
        lr.lane_dangers = lambda *a: [850, 850, 0]  # pareja por el izquierdo y el central: salir por el derecho
        frame = _capture("two_cars")
        agent.step(frame)
        assert agent.device.taps == [tuple(params["tap_right"])]
        agent.step(frame)  # aún no ha pasado chain_gap_s
        assert len(agent.device.taps) == 1
        agent._next_tap_at = 0.0
        agent.step(frame)
        agent.step(frame)
        assert agent.device.taps == [tuple(params["tap_right"])] * 2
    finally:
        lr.lane_dangers, lr.find_player = real


def test_extends_the_move_when_a_pair_shows_up_in_the_center():
    agent, params = _agent()
    import autoplayus.agents.lane_rush as lr

    real = lr.lane_dangers, lr.find_player
    try:
        frame = _capture("two_cars")
        lr.find_player = lambda frame, params: 138
        lr.lane_dangers = lambda *a: [818, 0, 0]  # asoma un coche por el izquierdo: hacia el centro sin esperar
        agent.step(frame)
        assert agent.device.taps == [tuple(params["tap_right"])]
        lr.find_player = lambda frame, params: 200  # cambiando de carril
        lr.lane_dangers = lambda *a: [836, 830, 0]  # era una pareja: seguir hasta el derecho
        agent.step(frame)
        assert agent.device.taps == [tuple(params["tap_right"])] * 2
        agent.step(frame)
        assert len(agent.device.taps) == 2
    finally:
        lr.lane_dangers, lr.find_player = real
