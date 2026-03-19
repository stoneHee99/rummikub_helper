"""Debug visualization utilities."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from models.tile import Tile


# Color map for drawing (BGR)
COLOR_MAP = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'black': (50, 50, 50),
    'orange': (0, 140, 255),
    None: (128, 128, 128),
}


def draw_tile_boxes(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image."""
    result = image.copy()
    for x, y, w, h in bboxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def draw_recognized_tiles(
    image: np.ndarray,
    tiles: List[Tile],
) -> np.ndarray:
    """Draw recognized tiles with color-coded boxes and labels."""
    result = image.copy()

    for tile in tiles:
        x, y, w, h = tile.bbox
        color = COLOR_MAP.get(tile.color, (128, 128, 128))

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        if tile.is_joker:
            label = "JKR"
        elif tile.number is not None:
            label = str(tile.number)
        else:
            label = "?"

        label += f" {tile.confidence:.1%}"

        cv2.putText(
            result, label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    return result


def show_image(title: str, image: np.ndarray, max_width: int = 1200) -> None:
    """Display image in a window, resized if too large."""
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        image = cv2.resize(image, (max_width, int(h * ratio)))

    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_debug_image(path: str, image: np.ndarray) -> None:
    """Save debug visualization to file."""
    cv2.imwrite(path, image)
