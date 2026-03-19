"""Rack tile detection using brightness-transition boundary finding."""

from typing import List, Tuple

import cv2
import numpy as np


# Brightness threshold for tile background (cream/white)
_BRIGHT_THRESH = 175
# Minimum tile width to accept
_MIN_TILE_W = 30
# Maximum tile width to accept
_MAX_TILE_W = 100


def detect_rack_tiles_grid(
    rack_region: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> List[Tuple[int, int, int, int]]:
    """Detect rack tiles using brightness-transition edge finding.

    The app UI tiles have bright cream backgrounds. By scanning
    thin horizontal strips near the top/bottom edges of tiles
    (where there is no ink from numbers), we can find precise
    tile left/right boundaries from dark-to-bright transitions.

    Args:
        rack_region: BGR image of the rack area.
        tile_width: Expected tile width from board detection.
        tile_height: Expected tile height from board detection.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    h, w = rack_region.shape[:2]
    gray = cv2.cvtColor(rack_region, cv2.COLOR_BGR2GRAY)

    # Find tile rows
    tile_rows = _find_tile_rows(gray, tile_height)

    all_tiles = []
    for row_y, row_h in tile_rows:
        row_tiles = _detect_row_by_brightness(
            rack_region, gray, row_y, row_h, tile_width, tile_height, w
        )
        all_tiles.extend(row_tiles)

    return all_tiles


def _find_tile_rows(gray: np.ndarray, tile_height: int) -> List[Tuple[int, int]]:
    """Find horizontal rows containing tiles."""
    h, w = gray.shape
    bright_counts = np.sum(gray > 160, axis=1)
    bright_threshold = w * 0.04

    bands = []
    in_band = False
    band_start = 0

    for y in range(h):
        if bright_counts[y] > bright_threshold and not in_band:
            band_start = y
            in_band = True
        elif (bright_counts[y] <= bright_threshold or y == h - 1) and in_band:
            band_h = y - band_start
            if band_h >= tile_height * 0.5:
                bands.append((band_start, band_h))
            in_band = False

    # Split tall bands into upper/lower rows
    result = []
    for by, bh in bands:
        if bh > tile_height * 1.6:
            mid = by + bh // 2
            result.append((by, mid - by))
            result.append((mid, by + bh - mid))
        else:
            result.append((by, bh))

    return result


def _detect_row_by_brightness(
    region: np.ndarray,
    gray: np.ndarray,
    row_y: int,
    row_h: int,
    tile_w: int,
    tile_h: int,
    img_w: int,
) -> List[Tuple[int, int, int, int]]:
    """Detect tiles in a row using brightness profile edge detection.

    Scans thin strips at the top and bottom edges of tiles (avoiding
    the center where number ink reduces brightness) and finds
    dark-to-bright / bright-to-dark transitions that mark tile edges.
    """
    y1 = max(0, row_y)
    y2 = min(gray.shape[0], row_y + row_h)

    if y2 - y1 < 10:
        return []

    # Sample strips near top and bottom of the tile row (ink-free zones)
    margin = max(2, int((y2 - y1) * 0.05))
    strip_h = max(3, int((y2 - y1) * 0.1))

    top_y1 = y1 + margin
    top_y2 = min(y2, top_y1 + strip_h)
    bot_y2 = y2 - margin
    bot_y1 = max(y1, bot_y2 - strip_h)

    strip_top = gray[top_y1:top_y2, :]
    strip_bot = gray[bot_y1:bot_y2, :]

    # Column-wise mean brightness, taking max of top/bottom
    col_top = np.mean(strip_top, axis=0) if strip_top.size > 0 else np.zeros(img_w)
    col_bot = np.mean(strip_bot, axis=0) if strip_bot.size > 0 else np.zeros(img_w)
    col_profile = np.maximum(col_top, col_bot)

    # Smooth slightly to remove noise
    k = 3
    col_smooth = np.convolve(col_profile, np.ones(k) / k, mode='same')

    min_w = max(_MIN_TILE_W, int(tile_w * 0.7))

    # Adaptive threshold: try multiple brightness thresholds and pick
    # the one that finds the most valid tiles
    best_tiles = []
    for thresh in range(170, 200, 5):
        bright = col_smooth > thresh
        edges = np.diff(bright.astype(int))

        left_edges = np.where(edges == 1)[0] + 1
        right_edges = np.where(edges == -1)[0]

        # Pair left/right edges into segments
        segments = []
        for le in left_edges:
            candidates = right_edges[right_edges > le]
            if len(candidates) == 0:
                continue
            re = int(candidates[0])
            segments.append((int(le), re))

        # Merge segments that are close together (ink dip splitting a tile)
        # Gap must be very small (< 6px) to avoid merging adjacent tiles
        merged = _merge_close_segments(segments, max_gap=5)

        candidate_tiles = []
        for seg_left, seg_right in merged:
            width = seg_right - seg_left

            # If segment is narrower than expected tile, try expanding
            # (brightness dip from ink may have shrunk it)
            if width < min_w and width > min_w * 0.5:
                expand = (tile_w - width) // 2
                seg_left = max(0, seg_left - expand)
                seg_right = min(img_w, seg_right + expand)
                width = seg_right - seg_left

            if min_w < width < _MAX_TILE_W:
                candidate = region[y1:y2, seg_left:seg_right]
                if _validate_tile(candidate):
                    candidate_tiles.append((seg_left, y1, width, y2 - y1))
            elif width >= _MAX_TILE_W:
                # Oversized segment: try extracting tile from the right end
                # (left side often merges with UI elements)
                tile_right = seg_right
                tile_left = max(seg_left, tile_right - tile_w)
                tw = tile_right - tile_left
                if min_w < tw < _MAX_TILE_W:
                    candidate = region[y1:y2, tile_left:tile_right]
                    if _validate_tile(candidate):
                        candidate_tiles.append(
                            (tile_left, y1, tw, y2 - y1))

        if len(candidate_tiles) > len(best_tiles):
            best_tiles = candidate_tiles

    tiles = best_tiles

    # If brightness method found too few tiles, fall back to scoring
    if len(tiles) < 2:
        tiles = _fallback_scoring(region, gray, y1, y2, tile_w, img_w)

    return tiles


def _fallback_scoring(
    region: np.ndarray,
    gray: np.ndarray,
    y1: int,
    y2: int,
    tile_w: int,
    img_w: int,
) -> List[Tuple[int, int, int, int]]:
    """Fallback tile detection using sliding window scoring.

    Used when brightness transition method doesn't find enough tiles
    (e.g., when tiles are sparse or have unusual backgrounds).
    """
    row_gray = gray[y1:y2, :]
    win_w = int(tile_w * 0.85)
    min_spacing = int(tile_w * 0.8)

    scores = np.zeros(img_w, dtype=np.float32)
    for x in range(0, img_w - win_w + 1):
        window = row_gray[:, x:x + win_w]
        total = window.size
        bright = np.sum(window > 180) / total
        if bright < 0.25:
            continue
        std = np.std(window.astype(np.float32))
        if std < 15:
            continue
        _, binary = cv2.threshold(window, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark = np.sum(binary > 0) / total
        if dark < 0.03 or dark > 0.6:
            continue
        scores[x] = (
            bright * 0.4 +
            min(std / 50, 1.0) * 0.3 +
            max(0.0, 1.0 - abs(dark - 0.15) * 3) * 0.3
        )

    # Greedy NMS
    used = np.zeros_like(scores, dtype=bool)
    tiles = []
    for _ in range(20):
        masked = scores.copy()
        masked[used] = 0
        if masked.max() < 0.3:
            break
        peak = int(np.argmax(masked))
        x = max(0, peak)
        w = min(tile_w, img_w - x)
        candidate = region[y1:y2, x:x + w]
        if _validate_tile(candidate):
            tiles.append((x, y1, w, y2 - y1))
        s = max(0, peak - min_spacing)
        e = min(len(used), peak + min_spacing)
        used[s:e] = True

    return tiles


def _merge_close_segments(
    segments: List[Tuple[int, int]],
    max_gap: int = 15,
) -> List[Tuple[int, int]]:
    """Merge segments that are separated by a small gap.

    When a number's ink creates a brightness dip, the tile can be
    split into two segments. Merge them if the gap is small enough.
    """
    if not segments:
        return []

    segments.sort()
    merged = [segments[0]]

    for left, right in segments[1:]:
        prev_left, prev_right = merged[-1]
        gap = left - prev_right
        if gap <= max_gap:
            # Merge: extend the previous segment
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))

    return merged


def _validate_tile(candidate: np.ndarray) -> bool:
    """Validate that a candidate region is a face-up tile with a number."""
    if candidate.size == 0:
        return False

    gray = (cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            if len(candidate.shape) == 3 else candidate)
    h, w = gray.shape

    if w < 30 or h < 10:
        return False

    total = gray.size

    # Must have bright background
    bright_ratio = np.sum(gray > 180) / total
    if bright_ratio < 0.2:
        return False

    # Must have ink (via Otsu)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = np.sum(binary > 0) / total
    if dark_ratio < 0.03 or dark_ratio > 0.6:
        return False

    # Must have contrast
    if np.std(gray.astype(float)) < 20:
        return False

    # Center must have ink present — check using both grayscale
    # contrast AND color saturation (orange ink has low grayscale
    # contrast but high saturation)
    ch, cw = h // 4, w // 4
    center_gray = gray[ch:h - ch, cw:w - cw]
    if center_gray.size == 0:
        return False

    center_std = np.std(center_gray.astype(float))

    # Also check color saturation if we have BGR input
    has_color_ink = False
    if len(candidate.shape) == 3:
        center_bgr = candidate[ch:h - ch, cw:w - cw]
        center_hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV)
        sat_std = np.std(center_hsv[:, :, 1].astype(float))
        max_sat = np.max(center_hsv[:, :, 1])
        # Colored ink (red/blue/orange) has high saturation pixels
        if max_sat > 100 and sat_std > 20:
            has_color_ink = True

    if center_std < 35 and not has_color_ink:
        return False

    return True
