"""Recognize the number (1-13) on a Rummikub tile using OCR + contour analysis."""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

import config
from utils.image_utils import to_grayscale, resize, normalize


class NumberRecognizer:
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or config.TEMPLATE_DIR
        self.templates: List[Tuple[int, np.ndarray]] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """Load number template images from disk."""
        if not os.path.isdir(self.template_dir):
            return
        for filename in sorted(os.listdir(self.template_dir)):
            if not filename.endswith(('.png', '.jpg')):
                continue
            name = os.path.splitext(filename)[0]
            try:
                number = int(name)
            except ValueError:
                continue
            path = os.path.join(self.template_dir, filename)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                template = cv2.resize(template, config.TEMPLATE_SIZE)
                template = normalize(template).astype(np.uint8)
                self.templates.append((number, template))

    def recognize(self, tile_image: np.ndarray) -> Tuple[Optional[int], float]:
        """Recognize the number on a tile image.

        Uses OCR as primary method, with contour-based fallback for
        numbers that OCR struggles with (e.g., italic "11").
        """
        # Check ink width ratio to estimate digit count
        expected_double = self._is_likely_double_digit(tile_image)

        # Primary: OCR
        number = self._ocr_recognize(tile_image)
        if number is not None:
            # Special case: OCR returns 1, but tile actually contains "11"
            # Check with _looks_like_eleven which is robust against bbox variations
            if number == 1:
                if self._check_eleven_pattern(tile_image):
                    return 11, 0.8
            # Cross-check only when ink ratio clearly suggests double digit
            elif expected_double and number < 10:
                contour_num = self._contour_recognize(tile_image)
                if contour_num is not None and contour_num >= 10:
                    return contour_num, 0.8
            return number, 0.9

        # Fallback: contour-based digit count + OCR per digit
        contour_num = self._contour_recognize(tile_image)
        if contour_num is not None:
            return contour_num, 0.7

        # Last resort: template matching
        if self.templates:
            return self._template_recognize(tile_image)

        return None, 0.0

    def _check_eleven_pattern(self, tile_image: np.ndarray) -> bool:
        """Check if a tile that OCR reads as '1' is actually '11'.

        Uses a focused analysis on the center ink region only,
        ignoring bbox edge noise.
        """
        gray = to_grayscale(tile_image)
        h, w = gray.shape

        # Crop to center region to avoid edge noise
        mx, my = int(w * 0.15), int(h * 0.1)
        center = gray[my:h - my, mx:w - mx]

        big = cv2.resize(center, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(big, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(binary)
        if coords is None:
            return False

        ix, iy, iw, ih = cv2.boundingRect(coords)

        # Check ink ratio on the cropped center
        ink_ratio = iw / big.shape[1]
        if ink_ratio < 0.35 or ink_ratio > 0.80:
            return False

        ink_crop = binary[iy:iy + ih, ix:ix + iw]
        return self._looks_like_eleven(ink_crop, iw, ih)

    def _is_likely_double_digit(self, tile_image: np.ndarray) -> bool:
        """Check if the tile likely contains a double-digit number (10-13).

        Uses the ink width ratio: single digits occupy ~22-42% of tile width,
        double digits occupy ~46-68%.
        """
        gray = to_grayscale(tile_image)
        big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(big, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return False
        _, _, iw, _ = cv2.boundingRect(coords)
        ink_ratio = iw / big.shape[1]
        # Double digit: ink takes 42-70% of tile width
        # Higher than 0.70 usually means threshold noise (shadow/shelf)
        return 0.42 < ink_ratio < 0.70

    def _ocr_recognize(self, tile_image: np.ndarray) -> Optional[int]:
        """Use pytesseract OCR to read the number."""
        gray = to_grayscale(tile_image)
        h, w = gray.shape

        mx, my = int(w * 0.1), int(h * 0.08)
        center = gray[my:h - my, mx:w - mx]

        # Scale up for better OCR
        scale = max(2, 100 // max(center.shape))
        center = cv2.resize(center, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(center, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio < 0.3:
            binary = cv2.bitwise_not(binary)

        border = 15
        padded = cv2.copyMakeBorder(binary, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=255)

        # Try multiple PSM modes for robustness
        for psm in [7, 8, 6]:
            cfg = f'--psm {psm} -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(padded, config=cfg).strip()
            num = self._parse_number(text)
            if num is not None:
                return num

        return None

    def _contour_recognize(self, tile_image: np.ndarray) -> Optional[int]:
        """Fallback: use contour analysis to estimate the number.

        Splits the ink region into individual digit contours,
        then uses the shape of each digit for recognition.
        """
        gray = to_grayscale(tile_image)
        h, w = gray.shape

        # Scale up for better analysis
        big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bh, bw = big.shape

        _, binary = cv2.threshold(big, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all ink pixels' bounding box
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None

        ix, iy, iw, ih = cv2.boundingRect(coords)
        ink_ratio = iw / bw

        # If ink covers nearly the entire width, Otsu likely captured
        # background noise (e.g., shelf color on narrow bbox).
        # Retry with center-cropped region to exclude shelf edges.
        if ink_ratio > 0.85:
            cx = bw // 5
            cropped = big[:, cx:bw - cx]
            _, crop_bin = cv2.threshold(cropped, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            crop_coords = cv2.findNonZero(crop_bin)
            if crop_coords is None:
                return None
            cix, ciy, ciw, cih = cv2.boundingRect(crop_coords)
            crop_ratio = ciw / cropped.shape[1]
            if crop_ratio > 0.85:
                return None
            # Use cropped version
            binary = np.zeros_like(big, dtype=np.uint8)
            binary[:, cx:bw - cx] = crop_bin
            ix, iy, iw, ih = cx + cix, ciy, ciw, cih
            ink_ratio = iw / bw

        # Estimate digit count from ink width
        # Single digits: ink_ratio ~0.22-0.42
        # Double digits: ink_ratio ~0.46-0.68
        is_double = ink_ratio > 0.44

        if not is_double:
            # Single digit: try OCR on just the ink region
            ink_region = big[max(0, iy-5):iy+ih+5, max(0, ix-5):ix+iw+5]
            _, ink_binary = cv2.threshold(ink_region, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_r = np.sum(ink_binary == 255) / ink_binary.size
            if white_r < 0.3:
                ink_binary = cv2.bitwise_not(ink_binary)

            padded = cv2.copyMakeBorder(ink_binary, 15, 15, 15, 15,
                                        cv2.BORDER_CONSTANT, value=255)
            cfg = '--psm 10 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(padded, config=cfg).strip()
            return self._parse_number(text)

        # Double digit: try to split vertically
        ink_crop = binary[iy:iy+ih, ix:ix+iw]
        col_sums = np.sum(ink_crop > 0, axis=0)

        # Look for a gap/minimum in the middle region
        third = iw // 3
        mid_region = col_sums[third:2*third]

        if len(mid_region) > 0:
            split_offset = third + np.argmin(mid_region)

            # Extract left and right digit regions
            left = big[max(0, iy-5):iy+ih+5, max(0, ix-5):ix+split_offset+5]
            right = big[max(0, iy-5):iy+ih+5, ix+split_offset-5:ix+iw+5]

            left_num = self._recognize_single_digit(left)
            right_num = self._recognize_single_digit(right)

            if left_num is not None and right_num is not None:
                result = left_num * 10 + right_num
                if 10 <= result <= 13:
                    return result

        # If split failed, try shape-based "11" detection
        # "11" has two narrow, tall ink regions separated by a gap
        if self._looks_like_eleven(ink_crop, iw, ih):
            return 11

        # Last try: OCR on the whole ink region
        ink_region = big[max(0, iy-5):iy+ih+5, max(0, ix-10):ix+iw+10]
        _, ink_binary = cv2.threshold(ink_region, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_r = np.sum(ink_binary == 255) / ink_binary.size
        if white_r < 0.3:
            ink_binary = cv2.bitwise_not(ink_binary)

        ink_binary = cv2.resize(ink_binary, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST)
        padded = cv2.copyMakeBorder(ink_binary, 20, 20, 20, 20,
                                    cv2.BORDER_CONSTANT, value=255)
        cfg = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(padded, config=cfg).strip()
        return self._parse_number(text)

    def _looks_like_eleven(self, ink_crop: np.ndarray, iw: int, ih: int) -> bool:
        """Check if the ink pattern looks like "11".

        "11" in the Rummikub italic font has two narrow vertical strokes
        with a gap between them. The column density shows a clear dip.
        """
        col_sums = np.sum(ink_crop > 0, axis=0)
        if len(col_sums) < 5:
            return False

        # Normalize column sums
        max_sum = col_sums.max()
        if max_sum == 0:
            return False
        density = col_sums / max_sum

        # Reject if ink covers nearly the entire width (threshold noise)
        # Use 0.3 threshold to ignore low-density noise pixels
        ink_coverage = np.sum(density > 0.3) / len(density)
        if ink_coverage > 0.85:
            return False

        # "11" pattern: two peaks with a valley in the middle
        # Find columns that are dense (part of a stroke) vs sparse (gap)
        dense_threshold = 0.3
        dense_cols = density > dense_threshold

        # Count transitions from dense->sparse->dense
        transitions = 0
        in_dense = False
        found_gap = False
        for d in dense_cols:
            if d and not in_dense:
                if found_gap:
                    transitions += 1
                in_dense = True
            elif not d and in_dense:
                in_dense = False
                found_gap = True

        # "11" should have exactly 1 gap between 2 strokes
        if transitions != 1:
            return False

        # Both strokes should be narrow (each "1" is thin)
        # The total ink width for "11" is moderate (ratio 0.44-0.65)
        # Each stroke should take < 40% of the total width
        dense_sections = []
        current_start = None
        for i, d in enumerate(dense_cols):
            if d and current_start is None:
                current_start = i
            elif not d and current_start is not None:
                dense_sections.append((current_start, i))
                current_start = None
        if current_start is not None:
            dense_sections.append((current_start, len(dense_cols)))

        if len(dense_sections) != 2:
            return False

        widths = [(end - start) for start, end in dense_sections]

        # Each "1" stroke should be narrow relative to total width
        for w in widths:
            stroke_ratio = w / iw
            if stroke_ratio > 0.45:
                return False

        # Both strokes should be similar width (both are "1")
        # Allow up to 2x difference
        if max(widths) > min(widths) * 2.5:
            return False

        # The gap between strokes should be significant
        gap = dense_sections[1][0] - dense_sections[0][1]
        gap_ratio = gap / iw
        if gap_ratio < 0.1:
            return False

        return True

    def _recognize_single_digit(self, digit_image: np.ndarray) -> Optional[int]:
        """Recognize a single digit using OCR + shape analysis."""
        # First try OCR
        num = self._ocr_single_digit(digit_image)
        if num is not None:
            return num

        # Shape-based fallback: check if it looks like a "1"
        gray = to_grayscale(digit_image) if len(digit_image.shape) == 3 else digit_image
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None

        _, _, dw, dh = cv2.boundingRect(coords)
        # "1" is very narrow and tall: width/height < 0.4
        if dh > 0 and dw / dh < 0.4:
            return 1

        return None

    def _ocr_single_digit(self, digit_image: np.ndarray) -> Optional[int]:
        """OCR a single digit image."""
        _, binary = cv2.threshold(digit_image, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_r = np.sum(binary == 255) / binary.size
        if white_r < 0.3:
            binary = cv2.bitwise_not(binary)

        binary = cv2.resize(binary, None, fx=2, fy=2,
                            interpolation=cv2.INTER_NEAREST)
        padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                    cv2.BORDER_CONSTANT, value=255)
        cfg = '--psm 10 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(padded, config=cfg).strip()
        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            return int(digits[0])
        return None

    def _parse_number(self, text: str) -> Optional[int]:
        """Parse OCR text into a valid Rummikub number (1-13)."""
        if not text:
            return None
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('l', '1').replace('I', '1')
        text = text.replace('S', '5').replace('s', '5')
        text = text.replace('B', '8')
        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            try:
                num = int(digits)
                if 1 <= num <= 13:
                    return num
            except ValueError:
                pass
        return None

    def _template_recognize(self, tile_image: np.ndarray) -> Tuple[Optional[int], float]:
        """Fallback: recognize using template matching."""
        gray = to_grayscale(tile_image)
        h, w = gray.shape
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
        center_resized = cv2.resize(center, config.TEMPLATE_SIZE)
        center_resized = normalize(center_resized).astype(np.uint8)

        best_number = None
        best_score = 0.0
        for number, template in self.templates:
            result = cv2.matchTemplate(center_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_number = number

        if best_score < config.TEMPLATE_MATCH_THRESHOLD:
            return None, best_score
        return best_number, best_score

    @property
    def is_ready(self) -> bool:
        return True


def extract_templates_from_tiles(
    image: np.ndarray,
    tile_bboxes: List[Tuple[int, int, int, int]],
    output_dir: str,
) -> None:
    """Helper to extract and save tile center regions as templates."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (x, y, w, h) in enumerate(tile_bboxes):
        tile = image[y:y+h, x:x+w]
        gray = to_grayscale(tile)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.10)
        center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
        center_resized = cv2.resize(center, config.TEMPLATE_SIZE)
        path = os.path.join(output_dir, f"tile_{i:03d}.png")
        cv2.imwrite(path, center_resized)
        print(f"Saved template: {path}")
