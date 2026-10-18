import colorsys

import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.color_memory import ColorMemory, background_bgr, find_knob, to_hsv, uniform_screen
from autoplayus.games.registry import load_catalog


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "ColorMemory" / f"{name}.png"))


def test_reads_the_target_and_the_sliders():
    params = load_catalog()["ColorMemory"].params
    memorize, recreate = _capture("memorize"), _capture("recreate")
    assert uniform_screen(memorize, params) and not uniform_screen(recreate, params)
    h, s, v = to_hsv(background_bgr(memorize, params))
    assert abs(h - 151) < 2 and abs(s - 0.83) < 0.02 and abs(v - 0.66) < 0.02
    assert [round(find_knob(recreate, y, params)) for y in params["rows"]] == [360, 452, 360]
    assert find_knob(memorize, params["rows"][0], params) is None


class FakeGame:
    """Juego sintético: el fondo es el color de tres deslizadores cuyos botones mueve drag_path(). `mapping` pasa
    de posición x del botón al valor (tono en grados, saturación y brillo 0-1). Como en el juego real
    (ColorMemory_run1), un arrastre que no se aleja `slop` px del punto inicial no mueve nada, y el botón queda
    `landing` px desviado de donde se suelta."""

    def __init__(self, mapping, knobs=(360, 452, 360), slop=12, landing=5):
        self.mapping, self.knobs, self.taps = mapping, list(knobs), []
        self.slop, self.landing, self.drags = slop, landing, 0

    def drag_path(self, points, step_ms=60):
        (x0, y0), (x_end, _) = points[0], points[-1]
        row = load_catalog()["ColorMemory"].params["rows"].index(y0)
        assert abs(self.knobs[row] - x0) < 2  # el arrastre empieza sobre el botón
        self.drags += 1
        if max(abs(x - x0) for x, _ in points) >= self.slop:
            self.knobs[row] = min(max(x_end + self.landing, 100), 620)

    def tap(self, x, y):
        self.taps.append((x, y))

    def frame(self):
        params = load_catalog()["ColorMemory"].params
        h, s, v = self.mapping(*self.knobs)
        r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, s, v)
        f = np.zeros((1280, 720, 3), np.uint8)
        f[:] = (round(b * 255), round(g * 255), round(r * 255))
        for x, y in zip(self.knobs, params["rows"]):
            f[y - 28 : y + 29, int(x) - 25 : int(x) + 26] = 255
        return f


def _game_mapping(xh, xs, xv):
    """Escala medida en la partida humana: tono ~0,84°/px; saturación y brillo de 0,1 a 1 entre x=100 y 620."""
    return (xh - 145.4) / 428.3 * 360, 0.1 + 0.9 * (xs - 100) / 520, 0.1 + 0.9 * (xv - 100) / 520


def _plain_mapping(xh, xs, xv):
    """Otra escala distinta (0-360 y 0-1 a lo largo de 100-620): el bucle cerrado no debe depender de ella."""
    return (xh - 100) / 520 * 360, (xs - 100) / 520, (xv - 100) / 520


def _play(mapping, target_hsv, **game_kwargs):
    params = load_catalog()["ColorMemory"].params
    game = FakeGame(mapping, **game_kwargs)
    agent = ColorMemory(game, params)
    agent.on_start()
    r, g, b = colorsys.hsv_to_rgb(target_hsv[0] / 360, *target_hsv[1:])
    memorize = np.zeros((1280, 720, 3), np.uint8)
    memorize[:] = (round(b * 255), round(g * 255), round(r * 255))
    agent.frame_time = 0.0
    agent.step(memorize)
    t = 1.0
    while not game.taps and t < 60:
        agent.frame_time = t
        agent.step(game.frame())
        t += params["settle_s"] + 0.01
    return game, to_hsv(background_bgr(game.frame(), params)), params


def test_converges_on_the_real_scale_and_presses_fijar():
    game, (h, s, v), params = _play(_game_mapping, (151.1, 0.827, 0.659))
    assert game.taps == [tuple(params["submit"])]
    assert abs(h - 151.1) < 1.5 and abs(s - 0.827) < 0.01 and abs(v - 0.659) < 0.01


def test_converges_even_if_the_scale_is_different():
    game, (h, s, v), _ = _play(_plain_mapping, (300.0, 0.4, 0.9))
    assert game.taps
    assert abs(h - 300) < 1.5 and abs(s - 0.4) < 0.01 and abs(v - 0.9) < 0.01


def test_reaches_red_at_either_end_of_the_hue_slider():
    for target in (3.0, 357.0):
        game, (h, s, v), _ = _play(_game_mapping, (target, 0.7, 0.8))
        assert game.taps
        assert min(abs(h - target), 360 - abs(h - target)) < 1.5, (target, h)


def test_fine_corrections_beat_the_drag_threshold_and_landing_offset():
    """ColorMemory_run1: los arrastres de menos de ~12 px no hacían nada y el tono se quedó 10° desviado."""
    game, (h, s, v), _ = _play(_game_mapping, (188.3, 0.243, 0.580), slop=12, landing=5)
    assert game.taps
    assert abs(h - 188.3) < 1.5 and abs(s - 0.243) < 0.01 and abs(v - 0.580) < 0.01


def test_moves_further_away_if_the_drags_do_not_register():
    """Con un umbral mayor que el alejamiento inicial (40 px), el agente detecta que el botón no se mueve y se
    aleja más."""
    game, (h, s, v), _ = _play(_game_mapping, (188.3, 0.243, 0.580), slop=60, landing=0)
    assert game.taps
    assert abs(h - 188.3) < 1.5 and abs(s - 0.243) < 0.01 and abs(v - 0.580) < 0.01
