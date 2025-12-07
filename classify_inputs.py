#!/usr/bin/env python
"""Classify images in `inputs/` as 'table' or 'chart'.

Heuristic (Pillow-only, no NumPy required):
- detect edges with `ImageFilter.FIND_EDGES`
- compute row/column edge projection peaks (tables have many straight grid lines)
- count distinct colors (charts tend to have more colors)

Usage:
  python classify_inputs.py --inputs inputs --output results.csv

This script is intentionally small and conservative; tune thresholds for your dataset.
"""

from __future__ import annotations
import os
import sys
import argparse
import csv
from PIL import Image, ImageFilter

SUPPORTED_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def load_image(path, max_dim=1000):
    img = Image.open(path)
    img = img.convert('RGB')
    # Resize for speed if very large while preserving aspect
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img, scale


def edge_projections(gray_img, threshold=60):
    """Return (row_sums, col_sums) of edge pixels after FIND_EDGES and thresholding."""
    edges = gray_img.filter(ImageFilter.FIND_EDGES).convert('L')
    # Binarize
    bw = edges.point(lambda p: 255 if p > threshold else 0)
    width, height = bw.size
    data = list(bw.getdata())

    row_sums = [0] * height
    col_sums = [0] * width
    for y in range(height):
        row_off = y * width
        cnt = 0
        for x in range(width):
            v = data[row_off + x]
            if v:
                cnt += 1
                col_sums[x] += 1
        row_sums[y] = cnt

    return row_sums, col_sums


def count_colors(img, maxcolors=1000000):
    # getcolors can return None if there are more than maxcolors unique colors
    palette = img.convert('RGB').getcolors(maxcolors)
    if palette is None:
        return maxcolors + 1
    return len(palette)


def classify_image(path):
    img, scale = load_image(path, max_dim=1000)
    gray = img.convert('L')

    row_sums, col_sums = edge_projections(gray, threshold=60)
    w, h = gray.size

    max_row = max(row_sums) if row_sums else 0
    max_col = max(col_sums) if col_sums else 0

    # normalized mean densities
    avg_row_density = (sum(row_sums) / float(w * max(1, h))) if h else 0.0
    avg_col_density = (sum(col_sums) / float(h * max(1, w))) if w else 0.0

    # count strong peaks (rows/cols with high edge counts relative to max)
    def count_peaks(sums, maxval, rel=0.45):
        if maxval <= 0:
            return 0
        return sum(1 for s in sums if s >= rel * maxval)

    row_peaks = count_peaks(row_sums, max_row, rel=0.45)
    col_peaks = count_peaks(col_sums, max_col, rel=0.45)

    # colorfulness proxy: distinct color count
    color_count = count_colors(img, maxcolors=1000000)

    # Heuristic scoring
    table_score = 0.0
    # Row/col peaks and density favor tables (regular grid lines)
    peak_factor = (row_peaks + col_peaks) / max(1.0, (h + w) * 0.5)
    density_factor = (avg_row_density + avg_col_density) * 0.5
    table_score = 0.6 * peak_factor + 0.4 * density_factor

    # Colorfulness reduces table likelihood (tables often monochrome or low colors)
    color_factor = min(color_count / 64.0, 1.0)
    table_score = table_score * (1.0 - 0.6 * color_factor)

    # Clip and normalize to [0,1]
    conf = max(0.0, min(1.0, table_score))

    # Decision threshold (tunable); conservative default: require both row & col peaks
    is_table = (row_peaks >= 3 and col_peaks >= 3 and conf >= 0.12)

    label = 'table' if is_table else 'chart'

    reason = {
        'row_peaks': row_peaks,
        'col_peaks': col_peaks,
        'avg_row_density': round(avg_row_density, 4),
        'avg_col_density': round(avg_col_density, 4),
        'color_count': color_count,
        'confidence': round(conf, 3)
    }

    return label, conf, reason


def find_input_images(inputs_dir):
    if os.path.isfile(inputs_dir):
        return [inputs_dir]
    files = []
    for name in sorted(os.listdir(inputs_dir)):
        if name.startswith('.'):
            continue
        if name.lower().endswith(SUPPORTED_EXT):
            files.append(os.path.join(inputs_dir, name))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description='Classify images in an inputs folder as table or chart.')
    parser.add_argument('--inputs', '-i', default='inputs', help='Input file or directory (default: inputs)')
    parser.add_argument('--output', '-o', help='CSV output path (optional). If omitted results are printed to stdout.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose progress output')
    args = parser.parse_args(argv)

    if not os.path.exists(args.inputs):
        print(f"Inputs path not found: {args.inputs}")
        sys.exit(2)

    images = find_input_images(args.inputs)
    if not images:
        print(f"No supported images found in: {args.inputs}")
        return

    results = []
    for path in images:
        try:
            label, conf, reason = classify_image(path)
            results.append((path, label, conf, reason))
            if args.verbose:
                print(f"{os.path.basename(path)} -> {label} (conf={conf:.3f}) {reason}")
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)

    if args.output:
        with open(args.output, 'w', newline='', encoding='utf-8') as fd:
            writer = csv.writer(fd)
            writer.writerow(['path', 'label', 'confidence', 'details'])
            for path, label, conf, reason in results:
                writer.writerow([path, label, f"{conf:.3f}", str(reason)])
        print(f"Wrote results to {args.output}")
    else:
        for path, label, conf, reason in results:
            print(f"{path}\t{label}\t{conf:.3f}\t{reason}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        sys.exit(1)
