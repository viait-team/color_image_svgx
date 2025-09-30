
from __future__ import annotations
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
        cols_with = np.count_nonzero(np.any(mask > 0, axis=0))
        return cols_with / float(w)
    else:
        rows_with = np.count_nonzero(np.any(mask > 0, axis=1))
        return rows_with / float(h)


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

    cols = cluster_coords(xs)
    rows = cluster_coords(ys)
    return rows, cols


def is_table_image(path: str, min_intersections: int = 10, coverage_threshold: float = 0.25,
                   min_line_length_ratio: float = 0.4, min_rows_cols: int = 2) -> bool:
    """
    Detects if an image is a table based on line structure.
    This function replicates the logic from classify_table_opencv.py to ensure identical results.
    """
    img, w, h = load_image(path)
    th = preprocess_gray(img)

    # Extract horizontal and vertical lines
    horiz = extract_lines(th, 'horizontal', w)
    vert = extract_lines(th, 'vertical', h)

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

    return is_table
