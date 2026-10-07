import logging
import time

import cv2
import numpy as np

from autoplayus.agents.base import BaseAgent

log = logging.getLogger(__name__)

# Rangos HSV (OpenCV: H 0-179) del color de cada caja y gesto que le corresponde.
COLORS = {
    "red": [((0, 120, 120), (12, 255, 255)), ((170, 120, 120), (179, 255, 255))],
    "yellow": [((18, 120, 150), (35, 255, 255))],
    "blue": [((95, 120, 120), (112, 255, 255))],
}


class ParcelPanic(BaseAgent):
    """Cada caja se desliza según su color: roja <-, amarilla ^, azul ->. Un fallo termina la partida.

    El juego rodea con un recuadro blanco la caja a clasificar (la primera en salir). Se localiza el
    recuadro, se clasifica el color de la caja que contiene y se desliza una sola vez por caja.

    Start, toque inicial, detección de fin y OK los hace BaseAgent.run().
    """

    step_delay = 0.005  # juego de reflejos: apenas esperar entre capturas (la captura por ventana tarda ~3 ms)
    capture_threads = 3  # solo si hay que capturar por ADB: ~18 capturas/s en vez de ~4-7

    def on_start(self) -> None:
        self._last_swipe = 0.0
        self._last_y: int | None = None  # altura del recuadro en el último fotograma con objetivo
        self._frame_lost = True  # el recuadro desapareció desde el último deslizamiento

    def step(self, frame: np.ndarray) -> None:
        if self.frame_time < self._last_swipe + self.params["settle_s"]:
            # Fotograma capturado antes de que el juego aplicara el último deslizamiento: aún muestra la caja
            # ya clasificada. Hacerle caso provoca un segundo deslizamiento sobre la caja siguiente (fallo).
            return
        target = self.find_target(frame)
        if target is None:
            # Tras clasificar una caja, el recuadro desaparece hasta que aparece la siguiente.
            self._frame_lost = True
            return
        y = target[1]
        color = self.box_color(frame, target)
        if color is None:
            return  # color dudoso (p. ej. otra caja volando por encima): esperar a un fotograma claro
        now = time.time()
        # Es una caja nueva si el recuadro desapareció, saltó hacia arriba (la siguiente va detrás), o si
        # pasó tanto tiempo que el deslizamiento anterior debió de fallar. Las cajas bajan, así que el
        # recuadro de una misma caja solo desciende.
        moved_up = self._last_y is not None and y < self._last_y - self.params["new_box_dy"]
        timed_out = now - self._last_swipe > self.params["retry_s"]
        self._last_y = y
        if now - self._last_swipe < self.params["min_gap_s"]:
            return
        if self._frame_lost or moved_up or timed_out:
            self.device.swipe(*self.params["swipes"][color], self.params["swipe_ms"])
            log.info("caja %s (y=%d) -> deslizar", color, y)
            self._last_swipe = now
            self._frame_lost = False

    def find_target(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Recuadro blanco que marca la caja objetivo: (x, y, ancho, alto) o None."""
        x1, y1, x2, y2 = self.params["belt"]
        region = frame[y1:y2, x1:x2]
        white = cv2.inRange(region, (self.params["white_min"],) * 3, (255, 255, 255))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wmin, wmax = self.params["frame_w"]
        hmin, hmax = self.params["frame_h"]
        best = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if wmin <= w <= wmax and hmin <= h <= hmax:
                # un recuadro está vacío por dentro: poco blanco respecto a su área
                if cv2.countNonZero(white[y : y + h, x : x + w]) < 0.25 * w * h:
                    if best is None or y > best[1]:  # el más bajo = el más antiguo
                        best = (x + x1, y + y1, w, h)
        return best

    def box_color(self, frame: np.ndarray, target: tuple[int, int, int, int]) -> str | None:
        """Color predominante (red/yellow/blue) de la caja dentro del recuadro."""
        x, y, w, h = target
        pad = 4  # excluir el propio borde blanco
        hsv = cv2.cvtColor(frame[y + pad : y + h - pad, x + pad : x + w - pad], cv2.COLOR_BGR2HSV)
        counts = {}
        for name, ranges in COLORS.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            counts[name] = cv2.countNonZero(mask)
        best, second = sorted(counts, key=counts.get, reverse=True)[:2]
        if counts[best] < self.params["min_color_px"]:
            return None
        if counts[second] > self.params["ambiguity"] * counts[best]:
            return None  # dos colores en el recuadro: no arriesgar, un fallo termina la partida
        return best
