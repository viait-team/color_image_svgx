
import os
import subprocess
import argparse
import sys
import tempfile
from datetime import datetime

def log(msg):
    """Prints a log message with a timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [LOG] {msg}")

def fail(msg):
    """Prints an error message and exits the script."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] ❌ {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    """Main function to perform batch OCR on images."""
    parser = argparse.ArgumentParser(description="Performs OCR on all PNG images in a directory using Tesseract.")
    parser.add_argument(
        "--inputs-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs"),
        help="The directory containing the input PNG files."
    )
    parser.add_argument(
        "--outputs-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
        help="The directory where the OCR output files (.txt, .tsv) will be saved."
    )
    parser.add_argument(
        "--tesseract-path",
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="The full path to the Tesseract executable."
    )
    parser.add_argument(
        "--magick-path",
        default=r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe",
        help="The full path to the ImageMagick 'magick' executable."
    )
    args = parser.parse_args()

    print("--- SCRIPT STARTED ---")

    # --- 1. VALIDATION ---
    log("Validating environment...")

    if not os.path.exists(args.tesseract_path):
        fail(f"Tesseract executable not found at '{args.tesseract_path}'. Please verify the path.")
    if not os.path.isdir(args.inputs_dir):
        fail(f"Inputs directory not found: {args.inputs_dir}")

    if not os.path.isdir(args.outputs_dir):
        log(f"Output directory not found. Creating '{args.outputs_dir}'...")
        os.makedirs(args.outputs_dir)

    image_files = [f for f in os.listdir(args.inputs_dir) if f.lower().endswith(".png")]

    if not image_files:
        log(f"No PNG images found in '{args.inputs_dir}'. Exiting.")
        sys.exit(0)

    log(f"Found {len(image_files)} image(s). Starting OCR...")

    # --- 2. PROCESSING LOOP ---
    for image_file_name in image_files:
        image_full_path = os.path.join(args.inputs_dir, image_file_name)
        log(f"⚙️ Processing '{image_full_path}'...")

        base_name = os.path.splitext(image_file_name)[0]
        output_stem = os.path.join(args.outputs_dir, base_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Intermediate file paths
            reduce_color_image_file_path = os.path.join(tmpdir, f"{base_name}_reduced.png")
            bilevel_image_file_path = os.path.join(tmpdir, f"{base_name}_bilevel.png")

            # ImageMagick commands
            try:
                # Get DPI
                dpi_cmd = [args.magick_path, "identify", "-format", "%x %y %U", image_full_path]
                dpi_result = subprocess.run(dpi_cmd, capture_output=True, text=True, check=True)
                log(f"DPI for '{image_file_name}': {dpi_result.stdout.strip()}")

                # Reduce color and create bilevel image
                reduce_cmd = [args.magick_path, image_full_path, "-fuzz", "20%", "-fill", "white", "-opaque", "red", reduce_color_image_file_path]
                subprocess.run(reduce_cmd, check=True, capture_output=True)

                bilevel_cmd = [args.magick_path, reduce_color_image_file_path, "-type", "Bilevel", "-morphology", "Erode", "Diamond", bilevel_image_file_path]
                subprocess.run(bilevel_cmd, check=True, capture_output=True)

            except subprocess.CalledProcessError as e:
                fail(f"ImageMagick command failed for {image_file_name}: {e.stderr}")


            # Tesseract OCR commands
            try:
                # TXT output
                txt_cmd = [args.tesseract_path, bilevel_image_file_path, output_stem, "-l", "eng"]
                subprocess.run(txt_cmd, check=True, capture_output=True)

                # TSV output
                tsv_cmd = [args.tesseract_path, bilevel_image_file_path, output_stem, "-l", "eng", "tsv"]
                subprocess.run(tsv_cmd, check=True, capture_output=True)

            except subprocess.CalledProcessError as e:
                fail(f"Tesseract command failed for {image_file_name}: {e.stderr}")

        log(f"✅ OCR complete for '{image_full_path}'.")

    # --- 3. COMPLETION ---
    log("🎉 Batch OCR processing finished.")

if __name__ == "__main__":
    main()
