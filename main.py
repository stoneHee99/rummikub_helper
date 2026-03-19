"""Rummikub tile recognition pipeline."""

import argparse
import json
import sys
from typing import List

import cv2
import numpy as np

import config
from capture.screen_capture import ScreenCapture
from detection.tile_detector import detect_tiles, separate_regions
from detection.color_classifier import classify_color
from detection.number_recognizer import NumberRecognizer, extract_templates_from_tiles
from detection.joker_detector import is_joker
from models.tile import Tile
from utils.debug import draw_tile_boxes, draw_recognized_tiles, show_image, save_debug_image
from utils.image_utils import crop


def recognize_tiles(image: np.ndarray, recognizer: NumberRecognizer) -> List[Tile]:
    """Run the full recognition pipeline on an image.

    Args:
        image: BGR image of the game screen.
        recognizer: NumberRecognizer instance.

    Returns:
        List of recognized Tile objects.
    """
    # Step 1: Detect tile bounding boxes
    bboxes = detect_tiles(image)

    # Step 2: Separate board vs rack
    h = image.shape[0]
    board_bboxes, rack_bboxes = separate_regions(bboxes, h)

    tiles = []

    for bbox, region in [(board_bboxes, 'board'), (rack_bboxes, 'rack')]:
        for x, y, w, bh in bbox:
            tile_img = crop(image, x, y, w, bh)

            # Step 3: Recognize number
            number, confidence = recognizer.recognize(tile_img)

            # Step 4: Check if joker
            if is_joker(tile_img, confidence):
                tile = Tile(
                    is_joker=True,
                    position=(x + w // 2, y + bh // 2),
                    bbox=(x, y, w, bh),
                    confidence=confidence,
                    region=region,
                )
            else:
                # Step 5: Classify color
                color = classify_color(tile_img)
                tile = Tile(
                    number=number,
                    color=color,
                    position=(x + w // 2, y + bh // 2),
                    bbox=(x, y, w, bh),
                    confidence=confidence,
                    region=region,
                )
            tiles.append(tile)

    return tiles


def cmd_analyze(args):
    """Analyze a screenshot image file."""
    capture = ScreenCapture()
    image = capture.load_image(args.image)

    recognizer = NumberRecognizer()
    if not recognizer.is_ready:
        print("Warning: No number templates loaded. Run --extract-templates first.")
        print("Color detection and tile detection will still work.\n")

    tiles = recognize_tiles(image, recognizer)

    # Print results
    print(f"Detected {len(tiles)} tiles:\n")
    board_tiles = [t for t in tiles if t.region == 'board']
    rack_tiles = [t for t in tiles if t.region == 'rack']

    if board_tiles:
        print(f"Board ({len(board_tiles)} tiles):")
        for t in board_tiles:
            print(f"  {t}")

    if rack_tiles:
        print(f"\nRack ({len(rack_tiles)} tiles):")
        for t in rack_tiles:
            print(f"  {t}")

    # JSON output
    if args.json:
        output = {
            'board': [t.to_dict() for t in board_tiles],
            'rack': [t.to_dict() for t in rack_tiles],
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to: {args.json}")

    # Debug visualization
    if args.debug:
        bboxes = [t.bbox for t in tiles]
        debug_img = draw_tile_boxes(image, bboxes)
        debug_img = draw_recognized_tiles(image, tiles)
        save_debug_image(args.debug, debug_img)
        print(f"Debug image saved to: {args.debug}")

    if args.show:
        debug_img = draw_recognized_tiles(image, tiles)
        show_image("Rummikub Tiles", debug_img)


def cmd_extract(args):
    """Extract number templates from a screenshot."""
    capture = ScreenCapture()
    image = capture.load_image(args.image)
    bboxes = detect_tiles(image)

    print(f"Detected {len(bboxes)} tiles. Extracting templates...")

    # Show detected tiles for verification
    debug_img = draw_tile_boxes(image, bboxes)
    if args.show:
        show_image("Detected Tiles", debug_img)

    extract_templates_from_tiles(image, bboxes, args.output)
    print(f"\nTemplates saved to: {args.output}")
    print("Rename each file to its number (e.g., tile_000.png -> 7.png)")


def cmd_capture(args):
    """Capture a screenshot of the current screen."""
    capture = ScreenCapture()
    image = capture.capture_full()
    capture.save_image(args.output, image)
    print(f"Screenshot saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Rummikub Tile Recognition')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # analyze command
    p_analyze = subparsers.add_parser('analyze', help='Analyze a screenshot')
    p_analyze.add_argument('image', help='Path to screenshot image')
    p_analyze.add_argument('--json', help='Save results as JSON file')
    p_analyze.add_argument('--debug', help='Save debug visualization image')
    p_analyze.add_argument('--show', action='store_true', help='Show debug window')
    p_analyze.set_defaults(func=cmd_analyze)

    # extract-templates command
    p_extract = subparsers.add_parser('extract-templates', help='Extract number templates')
    p_extract.add_argument('image', help='Path to screenshot image')
    p_extract.add_argument('--output', default='templates/numbers', help='Output directory')
    p_extract.add_argument('--show', action='store_true', help='Show detected tiles')
    p_extract.set_defaults(func=cmd_extract)

    # capture command
    p_capture = subparsers.add_parser('capture', help='Capture screenshot')
    p_capture.add_argument('--output', default='samples/capture.png', help='Output path')
    p_capture.set_defaults(func=cmd_capture)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
