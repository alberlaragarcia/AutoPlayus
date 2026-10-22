import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.swipe_fast import SwipeFast, read_arrow, read_bar
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.swipes = []

    def swipe(self, x1, y1, x2, y2, duration_ms=200):
        self.swipes.append((x1, y1, x2, y2))


def _params():
    return load_catalog()["SwipeFast"].params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "SwipeFast" / f"{name}.png"))


def test_reads_the_four_arrows_and_bars():
    params = _params()
    frame = _capture("mid_game")
    assert [read_arrow(frame, c, params) for c in params["boxes"]] == ["R", "L", "D", "D"]
    bars = [read_bar(frame, c, params) for c in params["boxes"]]
    assert all(100 < b <= 275 for b in bars)


def test_no_swipes_outside_the_game():
    params = _params()
    for name in ("start", "tutorial", "end"):
        device = FakeDevice()
        agent = SwipeFast(device, params)
        agent.on_start()
        for t in (0.0, 0.01, 0.02):
            agent.frame_time = t
            agent.step(_capture(name))
        assert device.swipes == [], name


def test_swipes_each_box_once_in_its_arrow_direction():
    params = _params()
    device = FakeDevice()
    agent = SwipeFast(device, params)
    agent.on_start()
    frame = _capture("mid_game")
    for t in (0.0, 0.01, 0.02, 0.03):
        agent.frame_time = t
        agent.step(frame)
    assert len(device.swipes) == 4  # una vez por caja: luego quedan pendientes hasta ver la flecha nueva
    for (x1, y1, x2, y2), (cx, cy), arrow in zip(sorted(device.swipes, key=lambda s: ((s[1] + s[3]) // 2, (s[0] + s[2]) // 2)),
                                                 params["boxes"], ["R", "L", "D", "D"]):
        assert ((x1 + x2) / 2, (y1 + y2) / 2) == (cx, cy)
        dx, dy = np.sign(x2 - x1), np.sign(y2 - y1)
        assert (dx, dy) == {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}[arrow]
    # sin confirmación, se repite pasado pending_s
    agent.frame_time = params["pending_s"] + 0.05
    agent.step(frame)
    assert len(device.swipes) == 8


def test_most_urgent_box_first():
    params = _params()
    device = FakeDevice()
    agent = SwipeFast(device, params)
    agent.on_start()
    frame = _capture("mid_game")
    for t in (0.0, 0.01):
        agent.frame_time = t
        agent.step(frame)
    bars = [read_bar(frame, c, params) for c in params["boxes"]]
    first = device.swipes[0]
    centre = ((first[0] + first[2]) // 2, (first[1] + first[3]) // 2)
    assert list(map(tuple, params["boxes"])).index(centre) == int(np.argmin(bars))


def test_follows_the_boxes_when_the_screen_shifts():
    params = _params()
    frame = _capture("mid_game")
    shifted = np.vstack([frame[25:], np.repeat(frame[-1:], 25, axis=0)])  # sin la etiqueta todo sube
    device = FakeDevice()
    agent = SwipeFast(device, params)
    agent.on_start()
    for t in (0.0, 0.01, 0.02):
        agent.frame_time = t
        agent.step(shifted)
    centres = sorted(((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in device.swipes)
    assert len(centres) == 4
    for (x, y), (bx, by) in zip(centres, sorted(map(tuple, params["boxes"]))):
        assert x == bx and abs(y - (by - 25)) <= 2


def test_confirmation_ignores_blank_readings_and_the_previous_refill():
    params = _params()
    agent = SwipeFast(FakeDevice(), params)
    agent.on_start()
    anim = params["refill_anim_s"]
    # una lectura vacía (None) no es una flecha nueva
    pending = {"t": 0.0, "arrow": "L", "min_bar": 150}
    assert not agent._confirmed(0, pending, None, 150, 0.05)
    assert agent._confirmed(0, pending, "U", 150, 0.06)
    # justo tras un acierto la barra aún sube por su relleno: no confirma el gesto siguiente
    agent._confirmed_at[0] = 1.0
    pending = {"t": 1.02, "arrow": "L", "min_bar": None}
    assert not agent._confirmed(0, pending, "L", 200, 1.05)
    # acabada la animación, una subida sobre el mínimo visto sí confirma (misma dirección otra vez)
    assert not agent._confirmed(0, pending, "L", 190, 1.0 + anim + 0.01)
    assert agent._confirmed(0, pending, "L", 190 + params["refill_px"], 1.0 + anim + 0.05)
