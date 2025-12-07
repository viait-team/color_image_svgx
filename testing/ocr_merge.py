#!/usr/bin/env python3
"""
OCR Merge Tool - Test merging PSM 6 and PSM 11 results
Run Tesseract with both PSM modes and intelligently merge the detections.
"""
import sys
import subprocess
import csv
import os

def run_ocr(image_file, psm_mode):
    """Run Tesseract with specified PSM mode."""
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_stem = f"{base_name}_psm{psm_mode}"
    tsv_file = f"{output_stem}.tsv"
    
    print(f"Running Tesseract with --psm {psm_mode}...")
    cmd = ["tesseract", image_file, output_stem, "-l", "eng", "--psm", str(psm_mode), "tsv"]
    subprocess.run(cmd, check=True)
    print(f"  Created: {tsv_file}")
    return tsv_file

def load_tsv(tsv_file):
    """Load TSV file and return rows as list of dicts."""
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)

def is_duplicate(row1, row2, distance_threshold=20, text_threshold=0.8):
    """
    Check if two detections are duplicates based on:
    1. Spatial proximity (coordinates within threshold)
    2. Text similarity
    """
    try:
        # Get coordinates
        x1, y1 = int(row1['left']), int(row1['top'])
        x2, y2 = int(row2['left']), int(row2['top'])
        
        # Check spatial distance
        distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        if distance > distance_threshold:
            return False
        
        # Check text similarity
        text1 = row1.get('text', '').strip()
        text2 = row2.get('text', '').strip()
        
        if not text1 or not text2:
            return False
        
        # Exact match
        if text1 == text2:
            return True
        
        # Fuzzy match (one is substring of other)
        if text1 in text2 or text2 in text1:
            return True
        
        return False
        
    except (ValueError, KeyError):
        return False

def merge_tsvs(tsv1_rows, tsv2_rows):
    """
    Merge two TSV row lists, removing duplicates.
    Priority: Keep detection with higher confidence.
    """
    merged = []
    psm2_unique = []
    
    # Start with all rows from PSM1
    merged.extend(tsv1_rows)
    
    # Add rows from PSM2 that are not duplicates
    for row2 in tsv2_rows:
        is_dup = False
        for i, row1 in enumerate(merged):
            if is_duplicate(row1, row2):
                is_dup = True
                # Keep the one with higher confidence
                try:
                    conf1 = float(row1.get('conf', -1))
                    conf2 = float(row2.get('conf', -1))
                    if conf2 > conf1:
                        merged[i] = row2  # Replace with higher confidence
                        print(f"  Replaced duplicate: '{row1.get('text', '')}' (conf {conf1:.1f}) → '{row2.get('text', '')}' (conf {conf2:.1f})")
                except ValueError:
                    pass
                break
        
        if not is_dup:
            psm2_unique.append(row2)
            merged.append(row2)
    
    return merged, psm2_unique

def save_merged_tsv(rows, output_file, fieldnames):
    """Save merged rows to TSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved merged TSV: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ocr_merge.py <image_file>")
        sys.exit(1)
    
    image_file = sys.argv[1]
    if not os.path.exists(image_file):
        print(f"Error: File not found: {image_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("OCR Merge Tool")
    print("=" * 60)
    
    # Step 1: Run OCR with PSM 6
    tsv6_file = run_ocr(image_file, 6)
    
    # Step 2: Run OCR with PSM 11
    tsv11_file = run_ocr(image_file, 11)
    
    print("\n" + "=" * 60)
    print("Merging Results")
    print("=" * 60)
    
    # Step 3: Load both TSVs
    tsv6_rows = load_tsv(tsv6_file)
    tsv11_rows = load_tsv(tsv11_file)
    
    print(f"PSM 6 rows:  {len(tsv6_rows)}")
    print(f"PSM 11 rows: {len(tsv11_rows)}")
    
    # Step 4: Merge
    merged_rows, psm11_unique = merge_tsvs(tsv6_rows, tsv11_rows)
    
    print(f"Merged rows: {len(merged_rows)}")
    print(f"Unique from PSM 11: {len(psm11_unique)}")
    
    # Step 5: Show unique detections from PSM 11
    if psm11_unique:
        print("\n" + "=" * 60)
        print("Unique Detections from PSM 11 (not in PSM 6):")
        print("=" * 60)
        for row in psm11_unique:
            text = row.get('text', '').strip()
            conf = row.get('conf', '-1')
            if text and float(conf) > 0:
                print(f"  '{text}' (conf: {conf})")
    
    # Step 6: Save merged TSV
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    merged_file = f"{base_name}_merged.tsv"
    
    fieldnames = list(tsv6_rows[0].keys()) if tsv6_rows else []
    save_merged_tsv(merged_rows, merged_file, fieldnames)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PSM 6 TSV:   {tsv6_file}")
    print(f"PSM 11 TSV:  {tsv11_file}")
    print(f"Merged TSV:  {merged_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
