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

    Start, toque del tutorial, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.0  # juego de reflejos: no esperar entre capturas

    def on_start(self) -> None:
        self._seen: Dice | None = None  # última lectura coherente
        self._seen_since = 0.0
        self._answered: Dice | None = None  # dados cuya suma ya se tecleó
        self._typed_at = 0.0
        self._retries = 0

    def step(self, frame: np.ndarray) -> None:
        dice = read_dice(frame, self.params)
        now = self.frame_time
        if self._answered is not None:
            self._wait_for_result(frame, dice, now)
            return
        if dice is None:
            self._seen = None
            return
        if dice != self._seen:
            self._seen, self._seen_since = dice, now
            return
        if now - self._seen_since < self.params["stable_s"]:
            return
        self._type_sum(dice, now)

    def _wait_for_result(self, frame: np.ndarray, dice: Dice | None, now: float) -> None:
        if dice != self._answered:
            if dice is None or answer_found(frame, self.params) or now - self._typed_at > self.params["check_s"]:
                # Los dados cambiaron: empezó otra ronda (o terminó la partida).
                self._answered, self._seen, self._retries = None, None, 0
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

    def _type_sum(self, dice: Dice, now: float) -> None:
        total = sum(dice)
        self._tap_all([self.params["keys"][d] for d in str(total)])
        self._answered, self._typed_at = dice, now
        log.info("Dados %s: suma %d", dice, total)

    def _tap_all(self, points: list[tuple[int, int]]) -> None:
        for k, point in enumerate(points):
            if k:
                time.sleep(self.params["digit_gap_s"])
            self.device.tap(*point)
