import cv2

from autoplayus import config
from autoplayus.agents.trace import Trace, build_lattice, detect_dots, node_state, timer_visible
from autoplayus.games.registry import load_catalog

ROUTE3 = [(146, 949), (288, 949), (359, 826), (430, 703), (288, 703), (217, 580)]


class FakeDevice:
    def __init__(self):
        self.drags = []

    def drag_path(self, points, step_ms):
        self.drags.append((points, step_ms))

    def tap(self, x, y):
        pass


def _agent():
    params = load_catalog()["Trace"].params
    agent = Trace(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "Trace" / f"{name}.png"))


def test_lattice_grows_with_the_rounds():
    _, params = _agent()
    small = build_lattice(detect_dots(_capture("r1_reveal"), params), params)
    big = build_lattice(detect_dots(_capture("r3_reveal0"), params), params)
    assert len(small) == 7 and len(big) == 19
    assert (359, 826) in small and (217, 580) in big


def test_reads_route_and_start_marker_colours():
    _, params = _agent()
    assert node_state(_capture("r3_reveal5"), 217, 580) == "yellow"  # último nodo de la ruta revelada
    assert node_state(_capture("r3_start"), 146, 949) == "teal"  # inicio marcado
    assert timer_visible(_capture("r3_start"), params) and not timer_visible(_capture("r3_reveal5"), params)


def _replay_round3(agent):
    for k in range(6):
        agent.frame_time = 1.0 + k * 0.3
        agent.step(_capture(f"r3_reveal{k}"))
    agent.frame_time = 3.0
    agent.step(_capture("r3_start"))


def test_traces_the_route_in_the_order_it_was_revealed():
    agent, params = _agent()
    _replay_round3(agent)
    assert agent.device.drags == [(ROUTE3, params["segment_ms"])]
    agent.frame_time = 3.1
    agent.step(_capture("r3_start"))  # ya trazada: no repetir
    assert len(agent.device.drags) == 1


def test_reverses_if_the_marked_start_is_the_last_node():
    agent, _ = _agent()
    for k in range(6):
        agent.frame_time = 1.0 + k * 0.3
        agent.step(_capture(f"r3_reveal{k}"))
    agent._order = agent._order[::-1]  # como si se hubiera anotado al revés
    agent.frame_time = 3.0
    agent.step(_capture("r3_start"))
    assert agent.device.drags[0][0] == ROUTE3


class _FakeUser32:
    def __init__(self):
        self.msgs = []

    def PostMessageW(self, hwnd, msg, wparam, lparam):
        self.msgs.append(msg)

    def SetThreadDpiAwarenessContext(self, ctx):
        return getattr(ctx, "value", ctx) or 0


def test_bluestacks_drag_path_keeps_the_finger_down(monkeypatch):
    """Un solo pulsar y un solo soltar para toda la ruta (drag_path de la base encadenaba swipes y levantaba
    el dedo en cada nodo)."""
    from autoplayus.device import bluestacks

    fake = _FakeUser32()
    monkeypatch.setattr(bluestacks, "_user32", fake)
    device = bluestacks.BlueStacksDevice.__new__(bluestacks.BlueStacksDevice)
    monkeypatch.setattr(device, "_window", lambda: 1, raising=False)
    monkeypatch.setattr(device, "_to_client", lambda hwnd, x, y: (y << 16) | x, raising=False)
    device.drag_path([(100, 100), (200, 100), (200, 200)], step_ms=16)
    assert fake.msgs.count(bluestacks._WM_LBUTTONDOWN) == 1
    assert fake.msgs.count(bluestacks._WM_LBUTTONUP) == 1
    assert fake.msgs.count(bluestacks._WM_MOUSEMOVE) == 4  # 2 tramos x 2 pasos
