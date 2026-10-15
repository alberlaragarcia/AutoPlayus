import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.zig import ScrollTracker, Zig, desired_direction, find_ball, path_center
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


def test_path_center_ahead_of_the_ball():
    _, params = _agent()
    frame = _capture("mid_game")
    x, y = find_ball(frame, params)
    assert abs(path_center(frame, y - 30, x, params["min_run"]) - 569.5) < 8  # el camino sigue hacia la derecha
    assert path_center(frame, 400, x, params["min_run"]) is None  # por encima del camino solo hay cielo


def test_desired_direction_has_a_dead_zone():
    assert desired_direction(300, 360, 1, 6) == -1
    assert desired_direction(420, 360, -1, 6) == 1
    assert desired_direction(363, 360, -1, 6) == -1  # dentro del margen: se mantiene
    assert desired_direction(None, 360, 1, 6) == 1  # sin camino a la vista: se mantiene


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
    for shift_x, expected in ((+4, -1), (-4, +1)):  # el camino va al lado contrario de la bola
        tracker = ScrollTracker(params)
        for i, f in enumerate(_moving_path(shift_x, 8)):
            tracker.update(f, i * 0.02)
        assert tracker.direction() == expected
        assert 300 < tracker.speed() < 500  # 8 px cada 0,02 s


def test_taps_when_the_path_ahead_is_on_the_other_side_and_not_twice():
    agent, params = _agent()
    frame = _capture("mid_game")
    agent._assumed = -1  # la bola va a la izquierda, el camino por delante tuerce a la derecha
    agent.frame_time = 10.0
    agent.step(frame)
    assert agent.device.taps == [tuple(params["tap"])]
    agent.frame_time = 10.05
    agent.step(frame)  # el toque aún no se refleja: no repetir
    assert len(agent.device.taps) == 1


def test_does_not_tap_without_knowing_the_direction_or_seeing_the_ball():
    agent, _ = _agent()
    agent.frame_time = 1.0
    agent.step(_capture("mid_game"))  # sin dirección conocida
    agent.step(_capture("tutorial"))  # sin bola
    assert agent.device.taps == []
