"""Image preprocessing utilities."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for tile edge detection."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 5
    )


def canny_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Apply Canny edge detection."""
    return cv2.Canny(gray, low, high)


def resize(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    if width and height:
        return cv2.resize(image, (width, height))
    if width:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h))
    if height:
        ratio = height / h
        new_w = int(w * ratio)
        return cv2.resize(image, (new_w, height))
    return image


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a rectangular region from image."""
    return image[y:y+h, x:x+w].copy()


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to 0-255 range."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
