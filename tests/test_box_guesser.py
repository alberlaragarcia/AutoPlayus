import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.box_guesser import BoxGuesser, board_width, count_boxes, decide, find_submit
from autoplayus.games.registry import load_catalog


class FakeDevice:
    """Cada toque en "+" sube el número que se ve en el panel."""

    def __init__(self, panel, params):
        self.panel = panel.copy()
        self.params = params
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))
        sx, sy = find_submit(self.panel, self.params) or (0, 0)
        if (x, y) == (sx + self.params["plus_offset"][0], sy + self.params["plus_offset"][1]):
            nx, ny = sx + self.params["number_offset"][0], sy + self.params["number_offset"][1]
            self.panel[ny - 20:ny + 20, nx - 30:nx + 30] = (len(self.taps) * 37) % 255

    def screenshot(self):
        return self.panel


def _params():
    return load_catalog()["BoxGuesser"].params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "BoxGuesser" / f"{name}.png"))


def test_counts_the_boxes_on_a_whole_board():
    params = _params()
    assert count_boxes(_capture("board_l7"), params) == 9
    assert count_boxes(_capture("board_l2"), params) == 5
    assert board_width(_capture("board_l7"), params) > 500
    # aún entrando se ve menos tablero: decide() se queda con los fotogramas anchos
    assert board_width(_capture("board_entering"), params) < 0.9 * board_width(_capture("board_l7"), params)
    assert board_width(_capture("panel"), params) is None


def test_counts_boxes_cut_by_the_edge_of_the_drawing_area():
    params = _params()
    # nivel 18 de la 1.ª partida en vivo, tablero en su punto más grande: no cabe y hay cubos cortados por el
    # borde de la zona de dibujo. Sin contar esos trozos salían 11 y se falló el nivel
    frame = _capture("board_l18_cut")
    assert count_boxes(frame, params) == 12
    assert count_boxes(frame, dict(params, edge_piece_px=10**9)) == 11


def test_finds_the_active_submit_button_only():
    params = _params()
    assert find_submit(_capture("panel"), params) == (359, 826)
    assert find_submit(_capture("panel_sent"), params) is None  # ya enviado (gris)
    assert find_submit(_capture("board_l7"), params) is None
    assert find_submit(_capture("end"), params) is None


def test_decides_with_the_largest_views():
    params = _params()
    # tablero pequeño (lejos, borroso) cuenta mal; cerca del ancho máximo, bien
    obs = [(200, 2), (250, 3), (300, 3), (540, 9), (560, 9), (550, 8), (520, 9)]
    assert decide(obs, params) == 9
    assert decide([], params) is None


def test_enters_the_count_and_submits():
    params = dict(_params(), settle_s=0.0, change_s=0.05)
    panel = _capture("panel")
    panel[680:730, 320:400] = 40  # número a cero (zona lisa)
    device = FakeDevice(panel, params)
    agent = BoxGuesser(device, params)
    agent.on_start()
    agent.frame_time = 0.0
    agent.step(_capture("board_l2"))  # ve el tablero: 5 cajas
    agent.step(device.screenshot())  # sale el panel
    sx, sy = find_submit(panel, params)
    plus = (sx + params["plus_offset"][0], sy + params["plus_offset"][1])
    assert device.taps == [plus] * 5 + [(sx, sy)]
    agent.step(device.screenshot())  # no vuelve a responder
    assert len(device.taps) == 6
