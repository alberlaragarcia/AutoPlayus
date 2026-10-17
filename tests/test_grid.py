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
    params = load_catalog()["Grid"].params
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


def _run(agent, frame, start, steps, dt):
    """Pasa `frame` al agente `steps` veces cada `dt` s; devuelve el instante siguiente."""
    for k in range(steps):
        agent.frame_time = start + k * dt
        agent.step(frame)
    return start + steps * dt


def test_taps_the_pattern_one_by_one_then_okay_once_it_is_marked():
    agent, params = _agent()
    pattern = _tiles(_capture("pattern4"), params, "green")
    gap = params["tap_gap_s"] * 1.05  # algo más que el intervalo (en vivo el tiempo es continuo)
    t = _run(agent, _capture("pattern4"), 1.0, 1, gap)
    t = _run(agent, _capture("done4"), t, 1, gap)  # aparece "Okay!": se encolan las 7 casillas
    assert agent.device.taps == []
    t = _run(agent, _capture("done4"), t, 2, gap / 2)  # dos capturas en un mismo intervalo: un solo toque
    assert len(agent.device.taps) == 1
    t = _run(agent, _capture("done4"), t, 8, gap)
    assert sorted(agent.device.taps) == sorted(pattern)
    _run(agent, _capture("done4"), t + params["verify_s"], 2, 0.01)  # misma lectura dos veces: Okay
    assert agent.device.taps[-1] == tuple(params["okay"])
    assert agent._state == "sent"


def test_does_not_check_while_a_tile_is_animating():
    """Grid_run1: comprobar con casillas sin ver (animándose tras el toque) desmarcó el patrón entero."""
    agent, params = _agent()
    gap = params["tap_gap_s"] * 1.05  # algo más que el intervalo (en vivo el tiempo es continuo)
    t = _run(agent, _capture("pattern4"), 1.0, 1, gap)
    t = _run(agent, _capture("done4"), t, 9, gap)
    n = len(agent.device.taps)
    animating = _capture("done4").copy()
    x, y = _tiles(animating, params, "yellow")[0]
    animating[y - 70 : y + 70, x - 70 : x + 70] = params["background_bgr"]  # una casilla que no se ve
    _run(agent, animating, t + params["verify_s"], 5, 0.05)
    assert len(agent.device.taps) == n  # ni corrige ni envía


def test_fixes_missing_cells_before_sending():
    agent, params = _agent()
    gap = params["tap_gap_s"] * 1.05  # algo más que el intervalo (en vivo el tiempo es continuo)
    t = _run(agent, _capture("pattern5"), 1.0, 1, gap)
    t = _run(agent, _capture("input5"), t, 13, gap)
    n = len(agent.device.taps)
    assert n == 11
    t = _run(agent, _capture("partial5"), t + params["verify_s"], 2, 0.01)  # 5 marcadas, ninguna sobra
    t = _run(agent, _capture("partial5"), t, 8, gap)
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


def test_reads_marked_cells_even_when_adjacent_ones_merge():
    """Grid_run2: marcadas se ven más grandes y las adyacentes se juntan en una mancha; el agente solo encontraba
    9 de 12 casillas y no enviaba. Leyendo el centro de las posiciones guardadas se ve el patrón completo."""
    agent, params = _agent()
    gap = params["tap_gap_s"] * 1.05
    assert len(find_tiles(_capture("done2_merged"), params)) < 12  # la búsqueda de casillas falla aquí
    t = _run(agent, _capture("pattern2"), 1.0, 1, gap)
    t = _run(agent, _capture("done2_merged"), t, 10, gap)  # tocar las 8 (la captura ya las muestra marcadas)
    _run(agent, _capture("done2_merged"), t + params["verify_s"], 2, 0.01)
    assert agent.device.taps[-1] == tuple(params["okay"])


def test_recovers_when_the_game_submits_on_timeout():
    """Grid_run2: sin pulsar "Okay!", el juego envió lo marcado al acabarse el tiempo, enseñó la confirmación y
    pasó al nivel 3. El agente seguía esperando para comprobar el nivel 2 y no jugó el 3."""
    agent, params = _agent()
    gap = params["tap_gap_s"] * 1.05
    t = _run(agent, _capture("pattern2"), 1.0, 1, gap)
    t = _run(agent, _capture("input5"), t, 10, gap)  # "Okay!" visible: marca las 8 (estado "typed")
    assert agent._state == "typed" and not agent._queue
    t = _run(agent, _capture("confirm2"), t, 2, gap)  # confirmación en verde del nivel 2
    assert agent._state == "between"
    _run(agent, _capture("pattern3"), t, 2, gap)
    assert agent._state == "watch" and len(agent._pattern) == 10
