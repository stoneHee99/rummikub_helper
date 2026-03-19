"""Detect joker tiles in Rummikub."""

import cv2
import numpy as np

from utils.image_utils import to_hsv


def is_joker(tile_image: np.ndarray, number_confidence: float) -> bool:
    """Determine if a tile is a joker.

    Jokers are identified by:
    1. Low number recognition confidence (doesn't match any number)
    2. High color variance in the center (multi-colored smiley face)
    3. Multiple distinct color clusters (not just one ink color)

    Args:
        tile_image: Cropped BGR image of a single tile.
        number_confidence: Confidence score from number recognition.

    Returns:
        True if the tile is likely a joker.
    """
    # If number recognition was confident, it's not a joker
    if number_confidence > 0.5:
        return False

    h, w = tile_image.shape[:2]
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.15)
    center = tile_image[margin_y:h - margin_y, margin_x:w - margin_x]

    if center.size == 0:
        return False

    hsv = to_hsv(center)
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

    # Get ink pixels only (not background)
    _, ink_mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_pixels = hsv[ink_mask > 0]

    if len(ink_pixels) < 10:
        return False

    # Joker criteria: ink has multiple distinct hue clusters
    # Regular tiles have uniform hue across all ink pixels
    hue_std = np.std(ink_pixels[:, 0].astype(float))

    # Count distinct hue ranges present in ink
    hue_bins = np.histogram(ink_pixels[:, 0], bins=18, range=(0, 180))[0]
    significant_bins = np.sum(hue_bins > len(ink_pixels) * 0.05)

    # Jokers have 3+ distinct color clusters with high hue spread
    # Regular tiles (even "11") have 1-2 bins with low std
    return hue_std > 40 and significant_bins >= 3
