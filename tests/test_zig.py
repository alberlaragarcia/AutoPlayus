import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.zig import ScrollTracker, Zig, find_ball, free_distance, heading, should_turn, yellow_row
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = load_catalog()["Zig"].params
    agent = Zig(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "Zig" / f"{name}.png"))


def test_finds_the_ball_only_during_play():
    _, params = _agent()
    x, y = find_ball(_capture("mid_game"), params)
    assert abs(x - 508) < 6 and abs(y - 822) < 6
    assert find_ball(_capture("tutorial"), params) is None  # pantalla oscurecida: no es la bola
    assert find_ball(_capture("end"), params) is None


def test_free_distance_along_the_path():
    _, params = _agent()
    frame = _capture("mid_game")
    x, y = find_ball(frame, params)
    right, left = free_distance(frame, x, y, 1, params), free_distance(frame, x, y, -1, params)
    assert right > 80 and left < 40  # el tramo sigue hacia la derecha; a la izquierda está el borde


def test_should_turn_only_when_the_other_way_has_more_room():
    assert should_turn(ahead=30, other=150, trigger=60)
    assert not should_turn(ahead=100, other=150, trigger=60)  # aún queda camino delante
    assert not should_turn(ahead=30, other=20, trigger=60)  # girar no mejora nada


def _moving_path(shift_x, shift_y, n=10):
    """Fotogramas sintéticos con un tramo amarillo que se desplaza (shift_x, shift_y) px por fotograma."""
    frames = []
    for i in range(n):
        f = np.full((1280, 720, 3), (140, 80, 100), np.uint8)
        x0, y0 = 250 + i * shift_x, 500 + i * shift_y
        f[y0 : y0 + 500, x0 : x0 + 160] = (90, 225, 255)
        frames.append(f)
    return frames


def test_scroll_tracker_reads_direction_and_speed():
    params = load_catalog()["Zig"].params
    for shift_x, expected in ((+4, +1), (-4, -1)):  # el camino se desplaza hacia el mismo lado que la bola
        tracker = ScrollTracker(params)
        for i, f in enumerate(_moving_path(shift_x, 8)):
            tracker.update(f, i * 0.02)
        assert tracker.direction() == expected
        assert 300 < tracker.speed() < 500  # 8 px cada 0,02 s


def test_scroll_tracker_does_not_depend_on_capture_rate():
    """Con capturas el triple de rápidas (y el juego repitiendo imagen) la lectura en px/s es la misma."""
    params = load_catalog()["Zig"].params
    tracker = ScrollTracker(params)
    for i, f in enumerate(_moving_path(2, 4, n=12)):
        for k in range(3):  # cada imagen del juego llega en 3 capturas
            tracker.update(f, i * 0.02 + k * 0.0066)
    assert tracker.direction() == 1  # 100 px/s de desplazamiento lateral


def test_taps_when_the_path_ahead_is_on_the_other_side_and_not_twice():
    params = dict(load_catalog()["Zig"].params, min_scroll_pxs=220)  # bajada medida a mitad de partida
    agent = Zig(FakeDevice(), params)
    agent.on_start()
    frame = _capture("mid_game")
    assert agent._dir == -1  # al empezar la bola va a la izquierda; el camino por delante queda a la derecha
    agent.frame_time = 10.0
    agent.step(frame)
    assert agent.device.taps == [tuple(params["tap"])]
    agent.frame_time = 10.05
    agent.step(frame)  # el toque aún no se refleja: no repetir
    assert len(agent.device.taps) == 1


def test_does_not_tap_without_seeing_the_ball():
    agent, _ = _agent()
    agent.frame_time = 1.0
    agent.step(_capture("tutorial"))
    agent.step(_capture("end"))
    assert agent.device.taps == []


def test_corrects_its_direction_only_after_a_sustained_disagreement():
    agent, params = _agent()
    agent.frame_time = 0.0
    agent._resync(+1)
    assert agent._dir == -1  # una lectura suelta no basta
    agent.frame_time = params["resync_s"]
    agent._resync(+1)
    assert agent._dir == 1
    agent.frame_time += 0.01
    agent._resync(None)  # sin lectura fiable no cambia nada
    assert agent._dir == 1


def test_goes_straight_while_the_start_platform_has_room():
    agent, _ = _agent()
    agent.frame_time = 1.0
    agent.step(_capture("platform_start"))
    assert agent.device.taps == []


def _simulate(frame, direction, speed, params, latency=0.08, dt=1 / 240):
    """Bola virtual sobre una captura real (el mundo quieto en sus coordenadas): avanza a `speed` px/s y cada
    decisión del agente cambia su sentido `latency` s después. Devuelve dónde se sale del amarillo (o se acaba
    el tiempo): sobre una captura, el camino visible termina a cierta altura y ahí acaba siempre."""
    frame = frame.copy()
    x, y = find_ball(frame, params)
    frame[y - 20 : y + 21, x - 20 : x + 21][yellow_row(frame[y - 20 : y + 21, x - 20 : x + 21]) == 0] = (90, 225, 255)
    bx, by, d, pending, last, t = float(x), float(y), direction, [], -1e9, 0.0
    while t < 6:
        if pending and t >= pending[0][0]:
            d = pending.pop(0)[1]
        intended = pending[-1][1] if pending else d
        if t - last >= params["block_s"]:
            ahead = free_distance(frame, int(bx), int(by), intended, params)
            other = free_distance(frame, int(bx), int(by), -intended, params)
            if should_turn(ahead, other, speed * params["look_s"] + params["margin_px"]):
                pending.append((t + latency, -intended))
                last = t
        hx, hy = heading(d, params)
        bx, by = bx + hx * speed * dt, by + hy * speed * dt
        if not yellow_row(frame[int(by), int(bx)]):
            break
        t += dt
    return int(bx), int(by)


def test_simulated_ball_crosses_the_start_platform_into_the_path():
    """Regresión del fallo en vivo: desde la plataforma inicial, la bola debe llegar al final del camino visible
    (y~575 en esta captura), no caerse por el borde de la plataforma (y~800)."""
    _, params = _agent()
    for speed in (100, 150):
        _, y = _simulate(_capture("platform_start"), -1, speed, params)
        assert y < 600


def test_reads_the_real_direction_on_the_start_platform():
    """Fotogramas reales (0,05 s entre sí) con la bola yendo a la izquierda por la plataforma inicial."""
    params = load_catalog()["Zig"].params
    tracker = ScrollTracker(params)
    for k, name in enumerate(("platform_1", "platform_2", "platform")):
        tracker.update(_capture(name), k * 0.05)
    assert tracker.direction() == -1
