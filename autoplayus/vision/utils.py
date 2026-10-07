import cv2
import numpy as np


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y : y + h, x : x + w]


def find_template(img: np.ndarray, template: np.ndarray, threshold: float = 0.85) -> tuple[int, int] | None:
    """Centro de la mejor coincidencia de `template` en `img`, o None si no supera el umbral."""
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < threshold:
        return None
    th, tw = template.shape[:2]
    return loc[0] + tw // 2, loc[1] + th // 2


def find_all_templates(
    img: np.ndarray, template: np.ndarray, threshold: float = 0.85
) -> list[tuple[int, int]]:
    """Centros de todas las coincidencias (sin supresión de solapes fina; suficiente para empezar)."""
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    th, tw = template.shape[:2]
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        cx, cy = int(x) + tw // 2, int(y) + th // 2
        if all(abs(cx - px) > tw // 2 or abs(cy - py) > th // 2 for px, py in points):
            points.append((cx, cy))
    return points
