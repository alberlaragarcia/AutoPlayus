import time

import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.center_hit import CenterHit, find_pointer, next_crossing, score
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((time.time(), x, y))


def _params():
    return load_catalog()["CenterHit"].params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "CenterHit" / f"{name}.png"))


def test_finds_the_pointer_only_during_play():
    params = _params()
    assert abs(find_pointer(_capture("mid_game"), params) - 614.5) < 1
    assert find_pointer(_capture("tutorial"), params) is None
    assert find_pointer(_capture("end"), params) is None


def test_score_matches_the_human_run():
    params = _params()
    # notas por toque deducidas de la media que mostró el juego, con la marca de cada toque en la barra
    for x, pct in ((385, 91.23), (362, 99.23), (403, 84.82), (278, 72.0), (388, 90.0)):
        assert abs(score(x, params) - pct) < 1.5


def test_next_crossing_only_when_heading_to_the_centre():
    params = _params()
    t = np.arange(0, 0.2, 0.01)
    towards = [(ti, 100 + 552.5 * ti) for ti in t]
    tc, v = next_crossing(towards, params)
    assert abs(tc - (params["center_x"] - 100) / 552.5) < 1e-6 and abs(v - 552.5) < 1e-6
    away = [(ti, 400 + 552.5 * ti) for ti in t]
    assert next_crossing(away, params) is None
    assert next_crossing(towards[:3], params) is None  # pocos datos


def _frame(x):
    img = np.full((1280, 720, 3), (52, 28, 25), np.uint8)
    cv2.rectangle(img, (int(x) - 3, 700), (int(x) + 3, 824), (255, 255, 255), -1)
    return img


def test_agent_bursts_five_taps_at_the_centre_crossing():
    params = dict(_params(), tap_latency_s=0.0)
    device = FakeDevice()
    agent = CenterHit(device, params)
    agent.on_start()
    t0 = time.time()
    x_at = lambda t: 71 + 552.5 * (t - t0)  # va hacia la derecha: cruza el centro a los ~0,52 s
    while time.time() - t0 < 0.8 and not device.taps:
        agent.frame_time = time.time()
        agent.step(_frame(x_at(agent.frame_time)))
        time.sleep(0.004)
    assert len(device.taps) == 5
    assert abs(x_at(device.taps[0][0]) - params["center_x"]) < 6  # ~10 ms a 552 px/s
