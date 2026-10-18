import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.danger_wall import (DangerWall, WallTracker, clearance_map, find_box, physics, plan_tap,
                                           read_spikes, spike_clearance)
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _params():
    return load_catalog()["DangerWall"].params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "DangerWall" / f"{name}.png"))


def test_finds_the_box_only_during_play():
    params = _params()
    x, y = find_box(_capture("mid_game"), params)
    assert abs(x - 293) < 4 and abs(y - 954) < 4  # con la estela detrás y pasando cerca del marcador
    assert find_box(_capture("tutorial"), params) is None
    assert find_box(_capture("end"), params) is None


def test_reads_spikes_and_rejects_half_drawn_ones():
    params = _params()
    frame = _capture("mid_game")
    assert read_spikes(frame, -1, params) == [(392, 463), (553, 623), (688, 759)]
    assert read_spikes(frame, 1, params) == [(294, 362), (590, 658), (893, 962)]
    regen = _capture("regen")
    assert read_spikes(regen, -1, params) is None  # la izquierda está sacando pinchos nuevos
    assert read_spikes(regen, 1, params) == [(474, 543)]


def test_wall_tracker_waits_for_new_spikes_after_a_bounce():
    params = _params()
    tracker = WallTracker(params)
    frame = _capture("mid_game")
    tracker.update(frame, 0.0)
    tracker.update(frame, 0.02)
    assert tracker.known[-1] == [(392, 463), (553, 623), (688, 759)]
    tracker.forget(-1, 1.0)
    tracker.update(frame, 1.05)
    tracker.update(frame, 1.07)
    assert tracker.known[-1] is None  # demasiado pronto: podrían ser los pinchos viejos escondiéndose
    tracker.update(frame, 1.25)
    assert tracker.known[-1] == [(392, 463), (553, 623), (688, 759)]


def test_spike_clearance_uses_the_rotated_box():
    params = _params()
    spans = [(600, 670)]
    # en la pared la caja llega recta: 71 px de alto, así que cabe en un hueco de 85 px entre dos pinchos
    gap = [(500, 570), (655, 725)]
    c = spike_clearance(params["rot_x0"], np.array([612.5]), -1, gap, params)
    assert c[0] > 0
    ys = np.array([635.0, 600 - params["box_half"] - 5, 400.0])
    c = spike_clearance(params["rot_x0"], ys, -1, spans, params)
    assert c[0] < -30  # dentro del pincho
    assert abs(c[1] - (5 - params["hit_margin"])) < 0.5  # el lado de abajo de la caja a 5 px de la base
    assert c[2] > 150
    # girada 45° (a ~148 px de la pared) la esquina llega más lejos, pero el pincho ya no alcanza
    far = spike_clearance(300, ys, -1, spans, params)
    assert far.min() > 150


def test_clearance_map_includes_floor_and_ceiling():
    params = _params()
    cmap = clearance_map({-1: None, 1: None}, params)
    step = params["map_px"]
    assert cmap[:, 0].max() < 0  # pegado al techo
    assert cmap[100, (750 - params["arena"][1]) // step] == params["clear_cap"]


def test_planner_jumps_over_a_spike_in_front():
    params = _params()
    vx, g, jump = physics(5, params)
    # cayendo hacia un pincho a la altura de la caja en la pared izquierda: hay que saltar ya
    spans = [(700, 770)]
    tap, clear = plan_tap(200, 700, 50, -1, vx, g, jump, {-1: spans, 1: None}, params)
    assert clear > 0
    # sin pinchos ni riesgo inmediato no toca
    tap_free, _ = plan_tap(360, 700, 0, -1, vx, g, jump, {-1: [], 1: []}, params)
    assert not tap_free


def test_planner_taps_before_hitting_the_floor():
    params = _params()
    vx, g, jump = physics(0, params)
    tap, _ = plan_tap(360, 1190, 300, 1, vx, g, jump, {-1: [], 1: []}, params)
    assert tap
    # más arriba esperar un fotograma más no quita holgura (tope clear_cap): se espera
    wait, _ = plan_tap(360, 1100, 300, 1, vx, g, jump, {-1: [], 1: []}, params)
    assert not wait


def test_agent_counts_bounces_and_taps_when_falling():
    params = _params()
    device = FakeDevice()
    agent = DangerWall(device, params)
    agent.on_start()
    frame = np.full((1280, 720, 3), (74, 46, 43), np.uint8)
    t, x, y, vy, d = 0.0, 560.0, 1000.0, 400.0, 1
    vx, g, jump = physics(0, params)
    applied = 0
    for _ in range(90):
        img = frame.copy()
        cv2.rectangle(img, (int(x) - 35, int(y) - 35), (int(x) + 35, int(y) + 35), (255, 255, 255), -1)
        agent.frame_time = t
        agent.step(img)
        dt = 1 / 60
        # cada toque surte efecto cuando el agente lo espera (tap_latency_s después)
        while applied < len(agent._taps) and agent._taps[applied] <= t:
            vy, applied = jump, applied + 1
        y += vy * dt + 0.5 * g * dt * dt
        vy += g * dt
        x += d * vx * dt
        if x >= params["contact_x"][1]:
            x, d = 2 * params["contact_x"][1] - x, -1
        t += dt
    assert agent._score == 1
    assert 300 < y < 1200  # no se dejó caer al suelo
