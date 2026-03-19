"""Classify the color of a Rummikub tile."""

from typing import Optional

import cv2
import numpy as np

from utils.image_utils import to_hsv


def classify_color(tile_image: np.ndarray) -> Optional[str]:
    """Determine the color of a tile from its cropped BGR image.

    Uses adaptive ink extraction and HSV-based color matching.
    Calibrated from actual Rummikub app tile measurements:
      - Red:    H~2,   S~187, V~178
      - Blue:   H~115, S~205, V~238
      - Orange: H~16,  S~97,  V~161
      - Black:  S<80,  V<80

    Args:
        tile_image: Cropped BGR image of a single tile.

    Returns:
        Color name ('red', 'blue', 'black', 'orange') or None.
    """
    h, w = tile_image.shape[:2]

    # Focus on center where the number is drawn
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.15)
    center = tile_image[margin_y:h - margin_y, margin_x:w - margin_x]

    if center.size == 0:
        return None

    center_hsv = to_hsv(center)
    center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    # Adaptive ink extraction: use Otsu's threshold to separate
    # ink from background, regardless of ink brightness
    _, ink_mask = cv2.threshold(center_gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If the tile background is dark (unlikely), invert
    white_ratio = np.sum(ink_mask == 0) / ink_mask.size
    if white_ratio < 0.3:
        ink_mask = cv2.bitwise_not(ink_mask)

    ink_pixels = center_hsv[ink_mask > 0]

    if len(ink_pixels) < 5:
        return None

    mean_h = float(np.mean(ink_pixels[:, 0]))
    mean_s = float(np.mean(ink_pixels[:, 1]))
    mean_v = float(np.mean(ink_pixels[:, 2]))

    # Black: low saturation AND low value
    if mean_s < 80 and mean_v < 80:
        return 'black'

    # Blue: high hue (~115), high saturation, high value
    if 100 <= mean_h <= 130 and mean_s > 150:
        return 'blue'

    # Red vs Orange: both have low hue, distinguish by H value
    # Red:    H ~2,  S ~187, V ~178 (deep red ink)
    # Orange: H ~13, S ~155, V ~225 (bright orange ink)
    if mean_h < 30 and mean_s > 60:
        if mean_h < 7:
            return 'red'
        return 'orange'

    if mean_h < 30:
        # Low saturation with low value -> black
        if mean_v < 100:
            return 'black'

    # High hue but not blue range -> might be edge case
    if mean_h >= 30 and mean_s < 80 and mean_v < 80:
        return 'black'

    return None
