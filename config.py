"""Central configuration for Rummikub tile recognition."""

import numpy as np

# --- Tile Colors (HSV ranges) ---
# OpenCV uses H: 0-179, S: 0-255, V: 0-255
COLOR_RANGES = {
    'red': [
        {'lower': np.array([0, 100, 80]), 'upper': np.array([10, 255, 255])},
        {'lower': np.array([170, 100, 80]), 'upper': np.array([179, 255, 255])},
    ],
    'blue': [
        {'lower': np.array([100, 80, 60]), 'upper': np.array([130, 255, 255])},
    ],
    'orange': [
        {'lower': np.array([10, 100, 100]), 'upper': np.array([25, 255, 255])},
    ],
    'black': [
        {'lower': np.array([0, 0, 0]), 'upper': np.array([179, 80, 80])},
    ],
}

# --- Tile Detection ---
TILE_MIN_AREA = 800        # Minimum contour area for a tile (pixels^2)
TILE_MAX_AREA = 15000      # Maximum contour area for a tile
TILE_ASPECT_MIN = 0.4      # Minimum width/height ratio
TILE_ASPECT_MAX = 1.6      # Maximum width/height ratio (allow double-digit tiles)
TILE_APPROX_EPSILON = 0.04  # approxPolyDP epsilon factor

# --- HSV-based Tile Mask ---
# Tiles are light/cream colored rectangles
# Board tiles: very light, low saturation
TILE_HSV_LOWER = np.array([0, 0, 180])
TILE_HSV_UPPER = np.array([40, 80, 255])
# Rack tiles may have slightly different lighting
TILE_HSV_LOWER_ALT = np.array([0, 0, 155])
TILE_HSV_UPPER_ALT = np.array([25, 130, 255])

# --- Template Matching ---
TEMPLATE_SIZE = (30, 45)           # Standard size for number templates (w, h)
TEMPLATE_MATCH_THRESHOLD = 0.6     # Minimum confidence for a match
TEMPLATE_DIR = 'templates/numbers'

# --- Screen Capture ---
CAPTURE_FPS = 5  # Frames per second for real-time capture

# --- Region of Interest ---
# These will be calibrated per setup; defaults assume a common layout
# Format: (x, y, width, height) as fractions of the app window
BOARD_REGION = (0.0, 0.0, 1.0, 0.72)   # Top 72% of play area
RACK_REGION = (0.0, 0.72, 1.0, 0.28)   # Bottom 28% of play area

# --- Debug ---
DEBUG_MODE = True  # Set True to show detection visualizations
