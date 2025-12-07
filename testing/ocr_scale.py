#!/usr/bin/env python3
import cv2
import sys
import subprocess
import os
import csv

def get_dpi(image_path):
    """Get DPI using ImageMagick. Defaults to 72 if undefined."""
    try:
        cmd = ["magick", "identify", "-format", "%x x %y", image_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        parts = output.split('x')
        if len(parts) == 2:
            x_dpi = float(parts[0].strip())
            y_dpi = float(parts[1].strip())
            
            if x_dpi == 0: x_dpi = 72.0
            if y_dpi == 0: y_dpi = 72.0
            
            return x_dpi, y_dpi
            
    except Exception as e:
        print(f"Warning: Could not determine DPI: {e}. Defaulting to 72.")
    
    return 72.0, 72.0

def scale_and_ocr(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        sys.exit(1)

    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ===== STEP 1: OCR Original Image =====
    print("=" * 60)
    print("STEP 1: OCR Original Image")
    print("=" * 60)
    original_tsv_file = f"{base_name}.tsv"
    print(f"Running Tesseract on original: {image_path}...")
    cmd = ["tesseract", image_path, 
           os.path.splitext(original_tsv_file)[0],
           "-l", "eng", "--psm", "6", "tsv"]
    subprocess.run(cmd, check=True)
    print(f"Saved original TSV: {original_tsv_file}\n")

    # ===== STEP 2: Calculate Scale Factor =====
    print("=" * 60)
    print("STEP 2: Calculate Scale Factor")
    print("=" * 60)
    dpi_x, dpi_y = get_dpi(image_path)
    print(f"Detected DPI: {dpi_x}x{dpi_y}")

    target_dpi = 300.0
    raw_scale = target_dpi / dpi_x
    scale_factor = round(raw_scale)
    
    # If smaller than 2, no scale
    if scale_factor < 2:
        scale_factor = 1
        
    # Clamp to max 5
    if scale_factor > 5:
        scale_factor = 5

    print(f"Calculated Scale Factor: {scale_factor} (Raw: {raw_scale:.2f})\n")

    # ===== STEP 3: Scale Image =====
    print("=" * 60)
    print("STEP 3: Scale Image")
    print("=" * 60)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Failed to load image with OpenCV.")
        sys.exit(1)

    if scale_factor > 1:
        height, width = img.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"Upscaling from {width}x{height} to {new_width}x{new_height}...")
        scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        scaled_image_file = f"{base_name}_scaled_{scale_factor}x.png"
        cv2.imwrite(scaled_image_file, scaled_img)
        print(f"Saved scaled image: {scaled_image_file}\n")
        ocr_input_file = scaled_image_file
    else:
        print("No scaling needed (scale factor = 1).")
        print("Skipping remaining steps.\n")
        return

    # ===== STEP 4: OCR Scaled Image =====
    print("=" * 60)
    print("STEP 4: OCR Scaled Image")
    print("=" * 60)
    scaled_tsv_file = f"{base_name}_scaled_{scale_factor}x.tsv"
    print(f"Running Tesseract on scaled: {ocr_input_file}...")
    cmd = ["tesseract", ocr_input_file, 
           os.path.splitext(scaled_tsv_file)[0],
           "-l", "eng", "--psm", "6", "tsv"]
    subprocess.run(cmd, check=True)
    print(f"Saved scaled TSV: {scaled_tsv_file}\n")

    # ===== STEP 5: Rescale TSV Coordinates =====
    print("=" * 60)
    print("STEP 5: Rescale TSV Coordinates")
    print("=" * 60)
    rescaled_tsv_file = f"{base_name}_rescaled.tsv"
    print(f"Rescaling coordinates back to original dimensions...")
    
    with open(scaled_tsv_file, 'r', encoding='utf-8') as f_in, \
         open(rescaled_tsv_file, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in, delimiter='\t')
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for row in reader:
            # Rescale ALL rows to preserve structure
            try:
                x = int(row['left'])
                y = int(row['top'])
                w = int(row['width'])
                h = int(row['height'])
                
                # Scale back to original dimensions
                row['left'] = str(int(x / scale_factor))
                row['top'] = str(int(y / scale_factor))
                row['width'] = str(int(w / scale_factor))
                row['height'] = str(int(h / scale_factor))
            except (ValueError, KeyError):
                pass
            
            writer.writerow(row)
    
    print(f"Saved rescaled TSV: {rescaled_tsv_file}\n")
    
    # ===== Summary =====
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original TSV:  {original_tsv_file}")
    print(f"Scaled Image:  {scaled_image_file}")
    print(f"Scaled TSV:    {scaled_tsv_file}")
    print(f"Rescaled TSV:  {rescaled_tsv_file}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ocr_scale.py <image_file>")
        sys.exit(1)
    
    scale_and_ocr(sys.argv[1])
