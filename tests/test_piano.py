import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.piano import Piano, find_keys
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = load_catalog()["Piano"].params
    agent = Piano(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _frame(params, keys):
    """Fotograma sintético 720x1280: fondo morado y teclas blancas [(carril, y superior)]."""
    frame = np.full((1280, 720, 3), (58, 10, 30), np.uint8)
    for lane, top in keys:
        x = params["lanes_x"][lane]
        frame[max(top, params["play_top"]) : top + params["key_h"], x - 80 : x + 81] = (240, 238, 238)
    return frame


def test_find_keys_on_real_capture_ignores_score_digit_and_entering_sliver():
    _, params = _agent()
    frame = cv2.imread(str(config.TEMPLATES_DIR / "Piano" / "in_game.png"))
    keys = find_keys(frame, params)
    assert sorted((lane, top, bottom) for lane, top, bottom, _ in keys) == [
        (0, 230, 502),
        (0, 785, 1055),
        (2, 508, 778),
        (2, 1062, 1279),
    ]


def test_taps_each_key_once_while_it_is_still_white_and_falling():
    agent, params = _agent()
    frame = cv2.imread(str(config.TEMPLATES_DIR / "Piano" / "in_game.png"))
    agent.step(frame)
    assert agent.device.taps == [(444, (1062 + 1279) // 2)]
    agent.step(frame)  # el juego aún no ha apagado la tecla: se toca la siguiente, no la misma
    assert agent.device.taps[1] == (111, (785 + 1055) // 2)


def test_does_not_retap_a_fast_falling_key_and_taps_new_ones():
    agent, params = _agent()
    agent.step(_frame(params, [(2, 400)]))
    assert agent.device.taps == [(444, (400 + 400 + 271) // 2)]
    # 0,12 s después a ~1300 px/s la misma tecla ha bajado ~160 px y sigue blanca; arriba entra otra en el carril 0
    agent.step(_frame(params, [(2, 560), (0, 560 - 277)]))
    assert len(agent.device.taps) == 2 and agent.device.taps[1][0] == 111
    # la primera ya se apagó (no se ve) y la segunda sigue blanca más abajo: nada que tocar
    agent.step(_frame(params, [(0, 283 + 160)]))
    assert len(agent.device.taps) == 2


def test_two_keys_in_the_same_row_are_tapped_one_after_the_other():
    agent, params = _agent()
    agent.step(_frame(params, [(1, 500), (2, 500)]))
    agent.step(_frame(params, [(1, 520), (2, 520)]))
    agent.step(_frame(params, [(1, 540), (2, 540)]))
    assert sorted(x for x, _ in agent.device.taps) == [277, 444]
