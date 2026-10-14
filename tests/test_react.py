import cv2

from autoplayus import config
from autoplayus.agents.react import React, lit_circles
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = load_catalog()["React"].params
    agent = React(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "React" / f"{name}.png"))


def test_detects_only_the_lit_circle():
    _, params = _agent()
    assert lit_circles(_capture("lit"), params) == [2]
    assert lit_circles(_capture("idle"), params) == []
    assert lit_circles(_capture("tutorial"), params) == []  # el círculo blanco del tutorial no cuenta


def test_taps_once_until_the_circle_is_seen_off():
    agent, params = _agent()
    lit, idle = _capture("lit"), _capture("idle")
    agent.frame_time = 1.0
    agent.step(lit)
    assert agent.device.taps == [tuple(params["circles"][2])]
    agent.frame_time = 1.01
    agent.step(lit)  # captura anterior a que el juego apague el círculo: no repetir
    assert len(agent.device.taps) == 1
    agent.frame_time = 1.02
    agent.step(idle)
    agent.frame_time = 2.0
    agent.step(lit)  # se vuelve a encender (otra ronda): nuevo toque
    assert len(agent.device.taps) == 2
