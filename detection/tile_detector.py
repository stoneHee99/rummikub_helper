"""Detect individual tile locations in a Rummikub board image."""

from typing import List, Tuple

import cv2
import numpy as np

import config


def detect_tiles(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find all tile bounding boxes in the image.

    Uses HSV color masking to find light-colored tile rectangles,
    with separate strategies for board and rack areas.

    Args:
        image: BGR image of the game board.

    Returns:
        List of (x, y, w, h) bounding boxes sorted left-to-right, top-to-bottom.
    """
    h, w = image.shape[:2]

    # Auto-detect the shelf (rack) position for classification
    shelf_top = _find_shelf_top(image)

    # Detect board tiles in the area above the shelf (with margin)
    board_bottom = min(h, shelf_top + 100)
    board_img = image[:board_bottom]
    board_bboxes = _detect_board_tiles(board_img)

    # Filter out non-tile detections (e.g., player avatars on the left)
    board_bboxes = _filter_non_tiles(board_bboxes, board_img)

    # Estimate tile size from board detections
    tile_w, tile_h = _estimate_tile_size(board_bboxes)

    # Detect rack tiles: brightness-based is primary (precise boundaries),
    # contour-based supplements where brightness method misses
    rack_start = max(0, shelf_top - 10)
    rack_img = image[rack_start:]

    from detection.rack_detector import detect_rack_tiles_grid
    grid_bboxes = detect_rack_tiles_grid(rack_img, tile_w, tile_h)

    contour_bboxes = _detect_rack_tiles(rack_img)

    # Merge brightness and contour results:
    # - For overlapping detections, keep the wider bbox (more context for OCR)
    # - For non-overlapping, add if reasonable size
    rack_bboxes = list(grid_bboxes)

    for cb in contour_bboxes:
        cx, cy, cw, ch = cb
        # Skip fragments with wrong dimensions
        if cw < tile_w * 0.7:
            continue
        if ch > tile_h * 1.3:
            continue

        # Check overlap with existing bboxes
        overlap_indices = []
        for i, gb in enumerate(rack_bboxes):
            if _iou(cb, gb) > 0.15:
                overlap_indices.append(i)

        if len(overlap_indices) == 1:
            # Single overlap: replace if contour is moderately wider
            idx = overlap_indices[0]
            gb = rack_bboxes[idx]
            if cw > gb[2] and cw < tile_w * 1.4:
                rack_bboxes[idx] = cb
        elif len(overlap_indices) == 0:
            rack_bboxes.append(cb)
        # If overlaps with 2+, contour likely merged tiles — skip

    # Offset rack bboxes to full image coordinates
    rack_bboxes = [(x, y + rack_start, bw, bh) for x, y, bw, bh in rack_bboxes]

    # Merge and deduplicate
    all_bboxes = board_bboxes + rack_bboxes
    all_bboxes = _merge_overlapping(all_bboxes)

    # Post-merge: remove bboxes with abnormal height
    # (merge can create tall bboxes from close tiles in different rows)
    if all_bboxes:
        med_h = sorted(b[3] for b in all_bboxes)[len(all_bboxes) // 2]
        all_bboxes = [(x, y, w, h) for x, y, w, h in all_bboxes
                      if h <= med_h * 1.3]

    # Sort: top-to-bottom first, then left-to-right
    all_bboxes.sort(key=lambda b: (b[1] // 40, b[0]))

    # Store the detected shelf position for separate_regions
    detect_tiles._shelf_top = shelf_top

    return all_bboxes


def _find_shelf_top(image: np.ndarray) -> int:
    """Auto-detect the top edge of the wooden shelf (rack area).

    The shelf is a horizontal band of tan/wood color. We find it by
    looking for rows with high concentration of tan-colored pixels.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Shelf is tan/beige: H~10-30, S~30-150, V~120-230
    shelf_mask = cv2.inRange(hsv, np.array([10, 30, 120]), np.array([30, 150, 230]))
    row_sums = np.sum(shelf_mask > 0, axis=1)

    # Find the shelf: a wide continuous band in the bottom half
    # Threshold for shelf detection
    shelf_threshold = w * 0.20
    # Only search the bottom 30% of image for the shelf
    bottom_half_start = int(h * 0.70)

    # Search from bottom up for first row above threshold
    shelf_start = None
    for y in range(h - 1, bottom_half_start, -1):
        if row_sums[y] > shelf_threshold:
            shelf_start = y
        elif shelf_start is not None:
            # We've found the top edge of the shelf band
            break

    if shelf_start is not None:
        # The wooden shelf top is where rack tiles sit
        # Tiles on the shelf extend ~80px above the shelf surface
        # But we need to avoid including board tiles that might be nearby

        # Find the actual shelf top: where row_sum jumps above threshold
        # (not just any row, but the start of the dense shelf band)
        jump_threshold = w * 0.25
        actual_shelf_top = shelf_start
        for y in range(shelf_start, bottom_half_start, -1):
            if row_sums[y] < jump_threshold:
                actual_shelf_top = y + 1
                break

        # Rack tiles sit on the shelf, extending ~80px above it
        return max(0, actual_shelf_top - 80)

    # Fallback to configured ratio
    return int(h * config.BOARD_REGION[3])


def _estimate_tile_size(
    bboxes: List[Tuple[int, int, int, int]],
) -> Tuple[int, int]:
    """Estimate tile dimensions from detected board tiles."""
    if not bboxes:
        return (40, 60)  # fallback defaults

    widths = [b[2] for b in bboxes]
    heights = [b[3] for b in bboxes]

    # Use median to be robust against outliers
    median_w = sorted(widths)[len(widths) // 2]
    median_h = sorted(heights)[len(heights) // 2]

    return (median_w, median_h)


def _detect_board_tiles(region: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect tiles on the board using combined HSV masks."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, config.TILE_HSV_LOWER, config.TILE_HSV_UPPER)
    mask2 = cv2.inRange(hsv, config.TILE_HSV_LOWER_ALT, config.TILE_HSV_UPPER_ALT)
    tile_mask = cv2.bitwise_or(mask1, mask2)

    # Use a vertical erode to separate horizontally adjacent tiles
    # without shrinking tiles too much vertically
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    tile_mask = cv2.erode(tile_mask, kernel_v, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return _extract_tile_bboxes(tile_mask, region.shape)


def _detect_rack_tiles(region: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect tiles in the rack/hand area.

    Uses the same broad HSV mask as board detection, but subtracts
    the wooden shelf background to isolate tiles.
    """
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Use the same tile mask as board (catches tiles in various lighting)
    mask1 = cv2.inRange(hsv, config.TILE_HSV_LOWER, config.TILE_HSV_UPPER)
    mask2 = cv2.inRange(hsv, config.TILE_HSV_LOWER_ALT, config.TILE_HSV_UPPER_ALT)
    tile_mask = cv2.bitwise_or(mask1, mask2)

    # Subtract the wooden shelf background to prevent merging
    shelf_mask = cv2.inRange(hsv, np.array([10, 30, 120]), np.array([30, 150, 230]))
    tile_mask = cv2.bitwise_and(tile_mask, cv2.bitwise_not(shelf_mask))

    # Erode to separate closely packed rack tiles
    # Vertical erode separates horizontal neighbors
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    tile_mask = cv2.erode(tile_mask, kernel_v, iterations=3)
    # Horizontal erode separates vertical neighbors (upper/lower row)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    tile_mask = cv2.erode(tile_mask, kernel_h, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return _extract_tile_bboxes(tile_mask, region.shape)


def _filter_non_tiles(
    bboxes: List[Tuple[int, int, int, int]],
    region: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Filter out non-tile detections like player avatars.

    Checks that each bbox contains a tile-like image (light background
    with darker number ink), not a photo/avatar.
    """
    if not bboxes:
        return []

    # Compute median dimensions for aspect ratio filtering
    if len(bboxes) > 3:
        med_h = sorted(b[3] for b in bboxes)[len(bboxes) // 2]
    else:
        med_h = None

    result = []
    for x, y, w, h in bboxes:
        # Skip bboxes with abnormal height (e.g., merged multi-row contours)
        if med_h and h > med_h * 1.4:
            continue

        tile = region[y:y + h, x:x + w]
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        # Tiles have a very light, uniform background
        bright_ratio = np.sum(gray > 200) / gray.size
        if bright_ratio < 0.5:
            continue

        # Tiles have low color variance on the edges (uniform cream background)
        # Avatars/photos have high variance
        hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        edge_top = hsv_tile[:max(1, h // 5), :, :]
        edge_bot = hsv_tile[h - max(1, h // 5):, :, :]
        edges = np.vstack([edge_top, edge_bot])
        hue_std = np.std(edges[:, :, 0].astype(float))
        sat_std = np.std(edges[:, :, 1].astype(float))
        if hue_std > 30 or sat_std > 40:
            # High color variance on edges -> not a tile
            continue

        result.append((x, y, w, h))

    return result


def _extract_tile_bboxes(
    tile_mask: np.ndarray,
    region_shape: Tuple[int, ...],
) -> List[Tuple[int, int, int, int]]:
    """Extract and validate tile bounding boxes from a binary mask."""
    contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tile_bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.TILE_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # If the contour is too large, try splitting
        if area > config.TILE_MAX_AREA:
            # Estimate single tile area from median of valid detections
            if tile_bboxes:
                med_w = sorted(b[2] for b in tile_bboxes)[len(tile_bboxes)//2]
                med_h = sorted(b[3] for b in tile_bboxes)[len(tile_bboxes)//2]
            else:
                med_w, med_h = 50, 70

            # Split vertically if too tall
            if h > med_h * 1.5:
                n_rows = round(h / med_h)
                row_step = h // n_rows
                for r in range(n_rows):
                    sy = y + r * row_step
                    sh = row_step if r < n_rows - 1 else (y + h - sy)
                    # Split horizontally if too wide
                    if w > med_w * 1.5:
                        n_cols = round(w / med_w)
                        col_step = w // n_cols
                        for c in range(n_cols):
                            sx = x + c * col_step
                            sw = col_step if c < n_cols - 1 else (x + w - sx)
                            tile_bboxes.append((sx, sy, sw, sh))
                    else:
                        tile_bboxes.append((x, sy, w, sh))
            elif w > med_w * 1.5:
                n_cols = round(w / med_w)
                col_step = w // n_cols
                for c in range(n_cols):
                    sx = x + c * col_step
                    sw = col_step if c < n_cols - 1 else (x + w - sx)
                    tile_bboxes.append((sx, y, sw, h))
            continue

        aspect = w / h if h > 0 else 0

        if config.TILE_ASPECT_MIN <= aspect <= config.TILE_ASPECT_MAX:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, config.TILE_APPROX_EPSILON * peri, True)
            if 4 <= len(approx) <= 8:
                tile_bboxes.append((x, y, w, h))

    tile_bboxes.sort(key=lambda b: (b[1] // 40, b[0]))
    tile_bboxes = _merge_overlapping(tile_bboxes)
    tile_bboxes = _split_wide_tiles(tile_bboxes)

    return tile_bboxes


def _split_wide_tiles(
    bboxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Split bounding boxes that are too wide (likely two tiles merged)."""
    if not bboxes:
        return []

    widths = [b[2] for b in bboxes]
    median_w = sorted(widths)[len(widths) // 2]

    result = []
    for x, y, w, h in bboxes:
        if w > median_w * 1.5 and w < median_w * 2.5:
            half = w // 2
            result.append((x, y, half, h))
            result.append((x + half, y, w - half, h))
        elif w > median_w * 2.5:
            # Very wide: possibly 3+ tiles merged, try splitting by median
            n = round(w / median_w)
            step = w // n
            for k in range(n):
                sx = x + k * step
                sw = step if k < n - 1 else (x + w - sx)
                result.append((sx, y, sw, h))
        else:
            result.append((x, y, w, h))

    result.sort(key=lambda b: (b[1] // 40, b[0]))
    return result


def _merge_overlapping(
    bboxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """Remove overlapping bounding boxes, keeping the larger one."""
    if not bboxes:
        return []

    result = []
    used = set()

    for i, box_a in enumerate(bboxes):
        if i in used:
            continue
        merged = box_a
        for j, box_b in enumerate(bboxes):
            if j <= i or j in used:
                continue
            if _iou(merged, box_b) > iou_threshold:
                if _area(box_b) > _area(merged):
                    merged = box_b
                used.add(j)
        result.append(merged)

    return result


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _area(bbox: Tuple[int, int, int, int]) -> int:
    return bbox[2] * bbox[3]


def separate_regions(
    bboxes: List[Tuple[int, int, int, int]],
    image_height: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Separate tile bboxes into board tiles and rack tiles.

    Uses the shelf position detected during detect_tiles().
    """
    # Use auto-detected shelf position if available
    shelf_top = getattr(detect_tiles, '_shelf_top', int(image_height * config.BOARD_REGION[3]))

    board_tiles = []
    rack_tiles = []

    for bbox in bboxes:
        _, y, _, h = bbox
        center_y = y + h // 2
        if center_y < shelf_top:
            board_tiles.append(bbox)
        else:
            rack_tiles.append(bbox)

    return board_tiles, rack_tiles
