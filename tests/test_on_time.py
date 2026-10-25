import time

import cv2
import numpy as np

from autoplayus import config
from autoplayus.agents.on_time import OnTime, arc_angle, find_ring
from autoplayus.games.registry import load_catalog


def _params():
    return load_catalog()["OnTime"].params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "OnTime" / f"{name}.png"))


def _ring_frame(angle, centre=(359, 691), radius=241):
    """Pantalla del cronómetro con el arco verde recorrido `angle` grados desde arriba en sentido horario."""
    bg = (20, 10, 8)
    img = np.full((1280, 720, 3), bg, np.uint8)
    cv2.circle(img, centre, radius + 21, (70, 62, 60), -1)  # parte gris por recorrer
    if angle > 0:  # sector relleno, para que el arco acabe recto como en el juego (un trazo grueso lo redondea)
        cv2.ellipse(img, centre, (radius + 21, radius + 21), -90, 0, angle, (190, 235, 60), -1)
    cv2.circle(img, centre, radius - 21, bg, -1)  # hueco del anillo
    return img


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append(time.time())


def test_reads_the_ring_and_its_arc():
    params = _params()
    frame = _capture("counting")  # el juego marcaba 2.91 s
    ring = find_ring(frame, params)
    assert ring is not None and abs(ring[0] - 359) < 3 and abs(ring[1] - 691) < 3
    assert abs(arc_angle(frame, ring) / 360 * 7 - 2.91) < 0.03
    assert find_ring(_capture("tap_to_start"), params) is None  # aviso: anillo atenuado, sin arco
    assert find_ring(_capture("end"), params) is None


def test_follows_the_ring_when_the_screen_shifts():
    params = _params()
    frame = _ring_frame(100, centre=(359, 665))
    ring = find_ring(frame, params)
    assert abs(ring[1] - 665) < 3
    assert abs(arc_angle(frame, ring) - 100) < 1.5


def test_stops_when_the_seen_clock_reaches_the_target():
    target = 0.6  # en lugar de 7 s, para que el test sea corto
    params = dict(_params(), target_s=target, see_to_effect_s=0.0, min_readings=10)
    device = FakeDevice()
    agent = OnTime(device, params)
    agent.on_start()
    agent.frame_time = time.time()
    agent.step(None)  # envía el toque de arranque
    seen_start = device.taps[0] + 0.05  # el juego arranca 50 ms después (retraso del toque)
    while len(device.taps) < 2 and time.time() - seen_start < 1.5:
        agent.frame_time = time.time()
        elapsed = max(0.0, agent.frame_time - seen_start)
        agent.step(_ring_frame(min(359.9, 360 * elapsed / target)))
        time.sleep(0.005)
    assert len(device.taps) == 2
    # se para cuando el arco visto completa la vuelta, no 0,6 s después de enviar el primer toque
    assert abs(device.taps[1] - (seen_start + target)) < 0.01
