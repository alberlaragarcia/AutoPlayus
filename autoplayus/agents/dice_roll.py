import logging
import time
from typing import Any

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

Dice = tuple[int, int, int, int]


def read_dice(frame: np.ndarray, params: dict[str, Any]) -> Dice | None:
    """Puntos de los 4 dados (cuadrícula 2x2, por filas) o None si no se leen con seguridad.

    En cada dado se miran los píxeles de dentro de la cara: los puntos son manchas oscuras y, con la cara
    plana, el blanco restante es 1 - `pip_frac` por cada punto. Se acepta la lectura solo si el número de
    manchas y esa fracción coinciden en los 4 dados (de 1 a 6): mientras los dados ruedan, las caras se ven
    inclinadas y ambas cuentas discrepan casi siempre.
    """
    r = params["die_half"]
    counts = []
    for x, y in params["dice"]:
        roi = frame[y - r : y + r, x - r : x + r]
        white = float((roi.min(axis=2) > params["white_min"]).mean())
        dark = (roi.max(axis=2) < params["pip_max"]).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        blobs = int((stats[1:, cv2.CC_STAT_AREA] >= params["pip_min_px"]).sum())
        if blobs != round((1 - white) / params["pip_frac"]) or not 1 <= blobs <= 6:
            return None
        counts.append(blobs)
    return tuple(counts)  # type: ignore[return-value]


def read_landing(frame: np.ndarray, params: dict[str, Any]) -> Dice | None:
    """Puntos de los 4 dados durante el aterrizaje (o None).

    Unos 0,17-0,31 s antes de asentarse, los dados ya muestran su cara final pero ~10 % más grandes y con el
    canto en 3D, así que read_dice no los acepta. Aquí se cuentan solo manchas redondas (proporción y relleno
    de círculo) que no toquen el borde de la zona, en una zona algo mayor. Mientras ruedan también salen
    lecturas plausibles pero falsas (hasta ~0,1 s seguidas en la partida humana): hay que exigir estabilidad.
    """
    r = params["landing_half"]
    counts = []
    for x, y in params["dice"]:
        dark = (frame[y - r : y + r, x - r : x + r].max(axis=2) < params["pip_max"]).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        n = 0
        for left, top, w, h, area in stats[1:]:
            round_blob = 0.7 < w / h < 1.4 and area / (w * h) > 0.6 and 150 <= area < 1500
            inside = left > 0 and top > 0 and left + w < 2 * r and top + h < 2 * r
            n += int(round_blob and inside)
        if not 1 <= n <= 6:
            return None
        counts.append(n)
    return tuple(counts)  # type: ignore[return-value]


def typed_visible(frame: np.ndarray, params: dict[str, Any]) -> bool:
    """Hay algún número tecleado en pantalla (blanco mientras no es la suma, verde si lo es)."""
    y0, y1, x0, x1 = params["typed_box"]
    box = frame[y0:y1, x0:x1]
    b, g, r = (box[..., c].astype(np.int16) for c in range(3))
    marked = (box.min(axis=2) > 225) | ((g > 180) & (r < 140) & (b < 170))
    return int(marked.sum()) >= params["typed_px"]


def answer_found(frame: np.ndarray, params: dict[str, Any]) -> bool:
    """El número tecleado se pone verde cuando es la suma correcta (no hay botón de enviar)."""
    y0, y1, x0, x1 = params["typed_box"]
    b, g, r = (frame[y0:y1, x0:x1, c].astype(np.int16) for c in range(3))
    return int(((g > 180) & (r < 140) & (b < 170)).sum()) >= params["green_px"]


class DiceRoll(BaseAgent):
    """Cuatro dados ruedan y se asientan; hay que teclear la suma en un teclado numérico (3 rondas, gana el
    menor tiempo total). El reloj solo corre con los dados quietos, así que cada milisegundo entre que se
    asientan y se teclea la suma cuenta.

    Se lee en cada captura; cuando la misma lectura coherente se mantiene `stable_s` s (los dados ya no giran)
    se teclean las cifras de la suma. Acertar pone el número en verde y arranca la siguiente ronda sin más
    gestos. Si pasado `check_s` el número no está verde (una tecla perdida o mal leída) se borra y se vuelve
    a teclear, hasta `max_retries` veces.

    Adelanto (`early_mode`): el valor final se puede leer ya durante el aterrizaje (read_landing), y desde que
    los dados empiezan a rodar hasta que se asientan pasan siempre ~1,33 s (rondas 2 y 3; la 1 no es fiable).
    - "predict": en las rondas 2 y 3 los toques salen `lead_s` antes del asentamiento previsto, para que
      lleguen al juego (que tarda ~0,07 s en registrarlos) justo cuando arranca el reloj. En la 1 se teclea en
      el primer fotograma asentado, sin esperar `stable_s`, si coincide con lo leído en el aterrizaje.
    - "asap": teclear en cuanto la lectura de aterrizaje sea estable, sin esperar al asentamiento.
    - "off": solo con los dados asentados.
    Si al asentarse no hay nada tecleado (el juego no aceptó teclas antes de tiempo) se teclea al momento y
    se desactiva el adelanto el resto de la partida; si lo leído en el aterrizaje era otra cosa, se borra y se
    teclea lo correcto.

    Start, toque del tutorial, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._seen: Dice | None = None  # última lectura coherente
        self._seen_since = 0.0
        self._guess: Dice | None = None  # última lectura de aterrizaje
        self._guess_since = 0.0
        self._answered: Dice | None = None  # dados cuya suma ya se tecleó
        self._typed_at = 0.0
        self._retries = 0
        self._early = False  # la suma de esta ronda se tecleó antes de asentarse los dados
        self._settled_at: float | None = None  # cuándo se asentaron tras teclear antes de tiempo
        self._early_ok = self.params["early_mode"] != "off"
        self._roll_start: float | None = None  # cuándo empezaron a rodar los dados de esta ronda

    def step(self, frame: np.ndarray) -> None:
        dice = read_dice(frame, self.params)
        now = self.frame_time
        if self._answered is not None:
            if self._early:
                self._check_early(frame, dice, now)
            else:
                self._wait_for_result(frame, dice, now)
            return
        if dice is None:
            self._seen = None
            self._track_guess(frame, now)
            if self._may_type_early(now):
                self._type_sum(self._stable_guess(), now, early=True)  # type: ignore[arg-type]
            return
        if self._early_ok and dice == self._stable_guess():
            self._type_sum(dice, now)  # ya se leyó durante el aterrizaje: no hace falta esperar
            return
        if dice != self._seen:
            self._seen, self._seen_since = dice, now
            return
        if now - self._seen_since < self.params["stable_s"]:
            return
        self._type_sum(dice, now)

    def _track_guess(self, frame: np.ndarray, now: float) -> None:
        guess = read_landing(frame, self.params) if self._early_ok else None
        if guess != self._guess:
            self._guess, self._guess_since = guess, now
        self._now = now

    def _stable_guess(self) -> Dice | None:
        if self._guess is None or self._now - self._guess_since < self.params["early_stable_s"]:
            return None
        return self._guess

    def _may_type_early(self, now: float) -> bool:
        if not self._early_ok or self._stable_guess() is None:
            return False
        if self.params["early_mode"] == "asap":
            return True
        return self._roll_start is not None and now >= self._roll_start + self.params["roll_s"] - self.params["lead_s"]

    def _check_early(self, frame: np.ndarray, dice: Dice | None, now: float) -> None:
        """Tras teclear antes del asentamiento: comprobar al asentarse que se aceptó y que era lo correcto."""
        if dice is None:
            return  # aún aterrizando
        if self._settled_at is None:
            self._settled_at = now
        if dice != self._answered:
            log.warning("Lo leído al aterrizar %s no era lo final %s: se corrige", self._answered, dice)
            self._early = False
            self._tap_all([self.params["backspace"]] * 2)
            self._type_sum(dice, now)
        elif answer_found(frame, self.params):
            self._early = False  # aceptado: ya solo queda esperar a la ronda siguiente
        elif now - self._settled_at >= self.params["fallback_s"] and not typed_visible(frame, self.params):
            log.warning("El juego no aceptó teclas antes de asentarse los dados: sin adelanto el resto de la partida")
            self._early = self._early_ok = False
            self._type_sum(dice, now)
        elif now - self._settled_at >= self.params["check_s"]:
            self._early = False  # tecleado pero no aceptado: lo resuelve el reintento normal

    def _wait_for_result(self, frame: np.ndarray, dice: Dice | None, now: float) -> None:
        if dice != self._answered:
            if dice is None or answer_found(frame, self.params) or now - self._typed_at > self.params["check_s"]:
                # Los dados cambiaron: empezó otra ronda (o terminó la partida).
                self._answered, self._seen, self._retries = None, None, 0
                self._guess, self._settled_at = None, None
                self._roll_start = now if dice is None else None
            return
        if answer_found(frame, self.params):
            return  # acierto: esperar a que arranque la ronda siguiente
        if now - self._typed_at >= self.params["check_s"]:
            if self._retries >= self.params["max_retries"]:
                return
            self._retries += 1
            log.warning("La suma %d no se aceptó: se borra y se repite (intento %d)", sum(self._answered), self._retries)
            self._tap_all([self.params["backspace"]] * 2)
            self._type_sum(self._answered, now)

    def _type_sum(self, dice: Dice, now: float, early: bool = False) -> None:
        total = sum(dice)
        self._tap_all([self.params["keys"][d] for d in str(total)])
        self._answered, self._typed_at, self._early = dice, now, early
        log.info("Dados %s: suma %d%s", dice, total, " (antes de asentarse)" if early else "")

    def _tap_all(self, points: list[tuple[int, int]]) -> None:
        for k, point in enumerate(points):
            if k:
                time.sleep(self.params["digit_gap_s"])
            self.device.tap(*point)
