import cv2

from autoplayus import config
from autoplayus.agents.dice_roll import DiceRoll, answer_found, read_dice, read_landing
from autoplayus.games.registry import load_catalog


class FakeDevice:
    def __init__(self):
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))


def _agent():
    params = dict(load_catalog()["DiceRoll"].params, digit_gap_s=0.0)
    agent = DiceRoll(FakeDevice(), params)
    agent.on_start()
    return agent, params


def _capture(name):
    return cv2.imread(str(config.TEMPLATES_DIR / "DiceRoll" / f"{name}.png"))


def test_reads_the_settled_dice_of_each_round():
    _, params = _agent()
    assert read_dice(_capture("round1"), params) == (5, 5, 2, 1)
    assert read_dice(_capture("round2"), params) == (3, 6, 6, 6)
    assert read_dice(_capture("round3"), params) == (1, 1, 2, 2)


def test_does_not_read_rolling_dice_or_other_screens():
    _, params = _agent()
    for name in ("rolling", "tutorial", "end"):
        assert read_dice(_capture(name), params) is None, name


def test_detects_the_green_correct_answer():
    _, params = _agent()
    assert answer_found(_capture("correct"), params)
    assert not answer_found(_capture("round1"), params)


def test_keys_land_on_the_keypad():
    """Cada tecla cae sobre su botón (más claro que el fondo) en una captura con el teclado activo."""
    _, params = _agent()
    frame = _capture("correct")
    background = frame[1250, 20].astype(int)
    for name, (x, y) in {**params["keys"], "backspace": params["backspace"]}.items():
        assert abs(frame[y, x - 55].astype(int) - background).max() > 8, name  # fuera del dígito, dentro de la tecla


def _feed(agent, frame, start, steps=8, dt=0.01):
    for k in range(steps):
        agent.frame_time = start + k * dt
        agent.step(frame)
    return start + steps * dt


def test_types_each_digit_once_when_the_dice_are_still():
    agent, params = _agent()
    _feed(agent, _capture("round1"), 1.0)
    assert agent.device.taps == [tuple(params["keys"]["1"]), tuple(params["keys"]["3"])]  # suma 13


def test_waits_while_dice_roll_and_for_stable_readings():
    agent, params = _agent()
    t = _feed(agent, _capture("rolling"), 1.0)
    assert agent.device.taps == []
    agent.frame_time = t
    agent.step(_capture("round2"))  # primera lectura: aún no ha pasado stable_s
    assert agent.device.taps == []
    agent.frame_time = t + params["stable_s"] + 0.001
    agent.step(_capture("round2"))
    assert agent.device.taps == [tuple(params["keys"]["2"]), tuple(params["keys"]["1"])]  # suma 21


def test_moves_on_to_the_next_round_after_the_answer_is_accepted():
    agent, params = _agent()
    t = _feed(agent, _capture("round1"), 1.0)
    t = _feed(agent, _capture("correct"), t)  # número verde: acierto, dados iguales
    t = _feed(agent, _capture("rolling"), t)  # dados girando: ronda nueva
    assert len(agent.device.taps) == 2
    _feed(agent, _capture("round3"), t + 0.5)
    assert agent.device.taps[2:] == [tuple(params["keys"]["6"])]  # suma 6, un solo dígito


def test_clears_and_retypes_when_the_answer_is_not_accepted():
    agent, params = _agent()
    t = _feed(agent, _capture("round1"), 1.0)
    agent.frame_time = t + params["check_s"] + 0.01  # sigue sin verde
    agent.step(_capture("round1"))
    assert agent.device.taps[2:] == [tuple(params["backspace"])] * 2 + [tuple(params["keys"]["1"]), tuple(params["keys"]["3"])]


def test_reads_the_final_faces_while_landing():
    _, params = _agent()
    frame = _capture("landing2")  # 0,08 s antes de asentarse: cara final, dado más grande
    assert read_dice(frame, params) is None
    assert read_landing(frame, params) == (3, 6, 6, 6)
    assert read_landing(_capture("rolling"), params) is None


def _answered_round1(agent):
    """Ronda 1 tecleada y aceptada; los dados empiezan a rodar en t0 (devuelto)."""
    t = _feed(agent, _capture("round1"), 1.0)
    t = _feed(agent, _capture("correct"), t)
    agent.frame_time = t
    agent.step(_capture("rolling"))  # deja de leerse: empieza el giro de la ronda 2
    return t


def test_types_ahead_of_the_predicted_settle_in_later_rounds():
    agent, params = _agent()
    t0 = _answered_round1(agent)
    taps_before = len(agent.device.taps)
    expected_at = t0 + params["roll_s"] - params["lead_s"]
    _feed(agent, _capture("landing2"), t0 + 1.0, steps=40)  # aterrizando de t0+1,0 a t0+1,4
    assert agent.device.taps[taps_before:] == [tuple(params["keys"]["2"]), tuple(params["keys"]["1"])]
    assert expected_at <= agent._typed_at < expected_at + 0.02


def test_falls_back_when_early_keys_were_ignored():
    agent, params = _agent()
    t0 = _answered_round1(agent)
    t = _feed(agent, _capture("landing2"), t0 + 1.0, steps=40)
    n = len(agent.device.taps)
    _feed(agent, _capture("round2"), t, steps=20)  # asentado y sin nada tecleado en pantalla
    assert agent.device.taps[n:] == [tuple(params["keys"]["2"]), tuple(params["keys"]["1"])]
    assert not agent._early_ok  # sin adelanto el resto de la partida


def test_corrects_a_wrong_landing_guess():
    agent, params = _agent()
    t0 = _answered_round1(agent)
    t = _feed(agent, _capture("landing2"), t0 + 1.0, steps=40)  # tecleó 21 por adelantado
    n = len(agent.device.taps)
    agent.frame_time = t
    agent.step(_capture("round1"))  # pero se asentó en otra cosa (13)
    assert agent.device.taps[n:] == [tuple(params["backspace"])] * 2 + [tuple(params["keys"]["1"]), tuple(params["keys"]["3"])]
