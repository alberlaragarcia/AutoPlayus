import cv2

from autoplayus import config
from autoplayus.agents.grid import Grid, find_tiles, match, okay_visible
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = dict(load_catalog()["Grid"].params, tap_gap_s=0.0)
    agent = Grid(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "Grid" / f"{name}.png"))


def _count(frame, params, state):
    return sum(s == state for *_, s in find_tiles(frame, params))


def test_reads_the_grid_and_the_pattern():
    _, params = _agent()
    pattern = _capture("pattern5")
    assert len(find_tiles(pattern, params)) == 25 and _count(pattern, params, "green") == 11
    assert len(find_tiles(_capture("pattern8"), params)) == 42 and _count(_capture("pattern8"), params, "green") == 17
    assert find_tiles(_capture("tutorial"), params) == []


def test_okay_button_only_while_answering():
    _, params = _agent()
    assert not okay_visible(_capture("pattern5"), params)
    assert okay_visible(_capture("input5"), params)


def _tiles(frame, params, state):
    return [(x, y) for x, y, s in find_tiles(frame, params) if s == state]


def test_taps_the_pattern_then_okay_once_it_is_marked():
    agent, params = _agent()
    pattern = _tiles(_capture("pattern4"), params, "green")
    agent.frame_time = 1.0
    agent.step(_capture("pattern4"))
    agent.frame_time = 2.0
    agent.step(_capture("done4"))  # aparece "Okay!": se tocan las 7 casillas
    assert sorted(agent.device.taps) == sorted(pattern)
    agent.frame_time = 2.0 + params["verify_s"]
    agent.step(_capture("done4"))  # las amarillas son el patrón: Okay
    assert agent.device.taps[-1] == tuple(params["okay"])
    assert agent._state == "sent"


def test_fixes_missing_cells_before_sending():
    agent, params = _agent()
    agent.frame_time = 1.0
    agent.step(_capture("pattern5"))
    agent.frame_time = 2.0
    agent.step(_capture("input5"))
    n = len(agent.device.taps)
    assert n == 11
    agent.frame_time = 2.0 + params["verify_s"]
    agent.step(_capture("partial5"))  # solo 5 marcadas (y ninguna sobra): se tocan las 6 que faltan
    fixes = agent.device.taps[n:]
    assert len(fixes) == 6 and tuple(params["okay"]) not in fixes
    pattern = _tiles(_capture("pattern5"), params, "green")
    marked = _tiles(_capture("partial5"), params, "yellow")
    assert not match(fixes, pattern, 30) and not match(marked, pattern, 30)


def test_ignores_the_green_confirmation_of_the_previous_level():
    """Al acertar, el juego enseña en verde la respuesta del nivel acabado (19 casillas en 6x6) y luego el patrón
    del siguiente (17 en 7x6). Guardar el mayor conjunto verde mezclaba ambos."""
    agent, params = _agent()
    confirm = _tiles(_capture("confirm7"), params, "green")
    agent._state, agent._sent, agent._tile_tol = "between", confirm, 30.0
    agent.frame_time = 1.0
    agent.step(_capture("confirm7"))
    assert agent._state == "between"  # la confirmación no es un patrón nuevo
    agent.step(_capture("pattern8"))
    assert agent._state == "watch" and len(agent._pattern) == 17


def test_retaps_the_tutorial():
    agent, _ = _agent()
    agent.frame_time = 1.0
    agent.step(_capture("tutorial"))
    agent.frame_time = 2.1
    agent.step(_capture("tutorial"))
    assert len(agent.device.taps) == 2
