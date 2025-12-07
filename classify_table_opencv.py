#!/usr/bin/env python
"""Detect whether an image contains a table using OpenCV.

Algorithm (robust and conservative):
- Convert to grayscale and adaptive threshold to increase contrast.
- Extract long horizontal and vertical lines with morphological operations.
- Compute intersections between horizontal and vertical masks.
- Compute coverage metrics and estimate row/column counts from intersection clusters.
- Conservative decision: classify as table if intersections >= min_intersections OR
  horizontal_coverage >= coverage_threshold OR vertical_coverage >= coverage_threshold.

Usage examples:
  python detect_table_opencv.py --input inputs\image.png --verbose
  python detect_table_opencv.py --input inputs --output results.csv

This script requires `opencv-python` (and `numpy`).
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np


def load_image(path: str) -> Tuple[np.ndarray, int, int]:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return img, w, h


def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding for variable lighting
    th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, blockSize=15, C=-2)
    return th


def extract_lines(binary: np.ndarray, axis: str, img_dim: int) -> np.ndarray:
    # axis: 'horizontal' or 'vertical'
    if axis == 'horizontal':
        size = max(1, img_dim // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    else:
        size = max(1, img_dim // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))

    # Use morphology to extract lines
    extracted = cv2.erode(binary, kernel, iterations=1)
    extracted = cv2.dilate(extracted, kernel, iterations=1)
    return extracted


def count_intersections(horiz: np.ndarray, vert: np.ndarray) -> int:
    inter = cv2.bitwise_and(horiz, vert)
    # Count connected components (each component is an intersection cluster)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inter, connectivity=8)
    # connectedComponents includes background as label 0
    return max(0, num_labels - 1)


def filter_long_components(mask: np.ndarray, axis: str, min_length_px: int) -> np.ndarray:
    """Keep only connected components whose bbox length along axis >= min_length_px."""
    out = np.zeros_like(mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        length = w if axis == 'horizontal' else h
        if length >= min_length_px:
            out[labels == lbl] = 255
    return out


def coverage_fraction(mask: np.ndarray, axis: str) -> float:
    # axis coverage: for horizontal -> fraction of columns that contain at least one line pixel
    h, w = mask.shape[:2]
    if axis == 'horizontal':
        vertical_lines_with = np.count_nonzero(np.any(mask > 0, axis=0))
        return vertical_lines_with / float(w)
    else:
        horizontal_lines_with = np.count_nonzero(np.any(mask > 0, axis=1))
        return horizontal_lines_with / float(h)


def estimate_grid_dimensions(inter_mask: np.ndarray, min_sep: int = 10) -> Tuple[int, int]:
    # Find centroids of intersection components and cluster x/y coordinates
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inter_mask, connectivity=8)
    if num_labels <= 1:
        return 0, 0
    pts = centroids[1:]  # skip background
    xs = np.sort(pts[:, 0])
    ys = np.sort(pts[:, 1])

    def cluster_coords(coords: np.ndarray) -> int:
        if len(coords) == 0:
            return 0
        groups = 1
        last = coords[0]
        for v in coords[1:]:
            if v - last > min_sep:
                groups += 1
                last = v
        return groups

    vertical_lines = cluster_coords(xs)
    horizontal_lines = cluster_coords(ys)
    return horizontal_lines, vertical_lines


def detect_table(path: str, min_intersections: int = 10, coverage_threshold: float = 0.25,
                 min_line_length_ratio: float = 0.4, min_rows_cols: int = 2, verbose: bool = False) -> Dict[str, Any]:
    img, w, h = load_image(path)
    th = preprocess_gray(img)

    # Extract horizontal and vertical lines
    horiz = extract_lines(th, 'horizontal', w)
    vert = extract_lines(th, 'vertical', h)

    # Optionally filter by line length: remove short segments using Hough or morphological opening
    # For simplicity we keep morphological result and compute coverage metrics

    # --- Raw metrics (for context/debugging) ---
    horiz_cov = coverage_fraction(horiz, 'horizontal')
    vert_cov = coverage_fraction(vert, 'vertical')

    # --- Robust metrics based on LONG lines only ---
    min_h_len = max(4, int(min_line_length_ratio * w))
    min_v_len = max(4, int(min_line_length_ratio * h))
    horiz_long = filter_long_components(horiz, 'horizontal', min_h_len)
    vert_long = filter_long_components(vert, 'vertical', min_v_len)

    horiz_long_cov = coverage_fraction(horiz_long, 'horizontal')
    vert_long_cov = coverage_fraction(vert_long, 'vertical')

    # Base all decisions on the intersection of LONG lines
    inter_long_mask = cv2.bitwise_and(horiz_long, vert_long)
    
    # Count intersections of long lines
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(inter_long_mask, connectivity=8)
    intersections = max(0, num_labels - 1)

    # Estimate grid from long lines
    rows, cols = estimate_grid_dimensions(inter_long_mask, min_sep=max(8, min(h, w)//100))

    # Stricter decision logic, now based on long-line metrics
    cond_strong_grid = (intersections >= min_intersections and
                        rows >= min_rows_cols and
                        cols >= min_rows_cols)
    cond_strong_grid = False


    cond_both_long_lines = (intersections >= max(4, min_intersections // 2) and
                            horiz_long_cov >= coverage_threshold and
                            vert_long_cov >= coverage_threshold)

    is_chart =  bool(cond_strong_grid or cond_both_long_lines)
    is_table = not is_chart

    result = {
        'path': path,
        'is_table': bool(is_table),
        'intersections': int(intersections),
        'horiz_coverage': float(round(horiz_cov, 4)),
        'vert_coverage': float(round(vert_cov, 4)),
        'horiz_long_coverage': float(round(horiz_long_cov, 4)),
        'vert_long_coverage': float(round(vert_long_cov, 4)),
        'rows_est': int(rows),
        'cols_est': int(cols),
        'image_width': int(w),
        'image_height': int(h)
    }

    if verbose:
        print(json.dumps(result, indent=2))

    return result


def process_inputs(inputs: str, output: str = None, **kwargs):
    # inputs may be a file or directory
    paths: List[str] = []
    if os.path.isfile(inputs):
        paths = [inputs]
    else:
        for name in sorted(os.listdir(inputs)):
            if name.startswith('.'):
                continue
            full = os.path.join(inputs, name)
            if os.path.isfile(full):
                # accept common image extensions
                if any(name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']):
                    paths.append(full)

    results = []
    for p in paths:
        try:
            res = detect_table(p, **kwargs)
            results.append(res)
        except Exception as e:
            results.append({'path': p, 'error': str(e)})

    if output:
        import csv
        with open(output, 'w', newline='', encoding='utf-8') as fd:
            fieldnames = ['path', 'is_table', 'intersections', 'horiz_coverage', 'vert_coverage', 'rows_est', 'cols_est', 'image_width', 'image_height']
            writer = csv.DictWriter(fd, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(row)
        print(f'Wrote results to {output}')
    else:
        for r in results:
            print(json.dumps(r))


def parse_args():
    p = argparse.ArgumentParser(description='OpenCV table detector')
    p.add_argument('--input', '-i', required=True, help='Input file or directory')
    p.add_argument('--output', '-o', help='CSV output path')
    p.add_argument('--min-intersections', type=int, default=10, help='Minimum intersection count to decide table')
    p.add_argument('--coverage-threshold', type=float, default=0.25, help='Coverage fraction threshold (0-1)')
    p.add_argument('--min-line-length-ratio', type=float, default=0.4, help='Minimum line length as a ratio of image dimension for filtering long lines.')
    p.add_argument('--min-rows-cols', type=int, default=2, help='Minimum estimated rows and columns required to consider grid')
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_inputs(args.input, output=args.output, min_intersections=args.min_intersections,
                   coverage_threshold=args.coverage_threshold, min_line_length_ratio=args.min_line_length_ratio,
                   verbose=args.verbose)
