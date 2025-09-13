#!/usr/bin/env python
"""trace a color image with potrace and extract its text using OCR"""

# color_image_svg.py
# Original script by ukurereh, May 20, 2012
# Modified by VIAIT team for Windows compatibility, September 12, 2025
# Refactored for sequential processing and modern Windows 10 compatibility.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


# External program commands.
# On modern Windows with ImageMagick 7+, the 'convert' subcommand is deprecated.
# Use 'magick' for conversion and 'magick identify' for identification.
PNGQUANT_PATH               = 'pngquant'
PNGNQ_PATH                  = 'pngnq'
IMAGEMAGICK_CONVERT_PATH    = 'magick'
IMAGEMAGICK_IDENTIFY_PATH   = 'magick identify'
POTRACE_PATH                = 'potrace'
TESSERACT_PATH              = 'tesseract'

POTRACE_DPI = 90.0 # potrace docs say it's 72, but this seems to work best
COMMAND_LEN_NEAR_MAX = 1900 # a low approximate (but not maximum) limit for
                            # very large command-line commands
VERBOSITY_LEVEL = 0 # not just a constant, also affected by -v/--verbose option

VERSION = '3.00'

import os
import sys
import shutil
import subprocess
import argparse
from glob import iglob
import functools
import tempfile
import csv
import xml.etree.ElementTree as ET
import re


#
def verbose(*args, level=1):
    if VERBOSITY_LEVEL >= level:
        # Use a single print call with space separation
        print(' '.join(map(str, args)))

def process_command(command, stdinput=None, stdout_=False, stderr_=False):
    """run command, return stdout and/or stderr as specified"""
    verbose("Executing command:", command)

    # Popen expects a sequence of program arguments, but shell=True allows a string.
    # It's generally safer to pass a string with shell=True on Windows.
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdinput is not None else None,
            stdout=subprocess.PIPE if stdout_ else None,
            stderr=subprocess.PIPE,
            shell=True
        )

        stdoutput, stderror = process.communicate(input=stdinput)

        if process.returncode != 0:
            # Provide a more informative error message
            raise subprocess.CalledProcessError(process.returncode, command, output=stdoutput, stderr=stderror)

        if stdout_ and not stderr_:
            return stdoutput
        elif stderr_ and not stdout_:
            return stderror
        elif stdout_ and stderr_:
            return (stdoutput, stderror)
        else:
            return None

    except FileNotFoundError:
        # This error occurs if the command itself (e.g., 'magick') isn't found.
        print(f"Error: Command not found: '{command.split()[0]}'. Is it installed and in your system's PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr.decode(errors='ignore').strip()}", file=sys.stderr)
        sys.exit(1)


def rescale(src, destscale, scale, filter='lanczos'):
    """rescale src image to scale, save to destscale"""
    if scale == 1.0:
        shutil.copyfile(src, destscale)
    else:
        command = f'{IMAGEMAGICK_CONVERT_PATH} "{src}" -filter {filter} -resize {scale*100}% "{destscale}"'
        process_command(command)


def quantize(src, destquant, colors, algorithm='mc', dither=None):
    """quantize src image to colors, save to destquant"""
    if colors == 0:
        shutil.copyfile(src, destquant)
        return

    if algorithm == 'mc':
        ditheropt = '--nofs' if dither is None else ''
        command = f'"{PNGQUANT_PATH}" {ditheropt} --force --output "{destquant}" {colors} "{src}"'
        process_command(command)

    elif algorithm == 'as':
        ditheropt = 'None' if dither is None else dither
        command = f'{IMAGEMAGICK_CONVERT_PATH} "{src}" -dither {ditheropt} -colors {colors} "{destquant}"'
        process_command(command)

    elif algorithm == 'nq':
        ext = "-quant.png"
        destdir = os.path.dirname(destquant)
        base_name = os.path.splitext(os.path.basename(src))[0]
        expected_output = os.path.join(destdir, f"{base_name}{ext}")

        ditheropt = '' if dither is None else '-Q f'
        command = f'"{PNGNQ_PATH}" -f {ditheropt} -d "{destdir}" -n {colors} -e {ext} "{src}"'
        process_command(command)
        if os.path.exists(expected_output):
             os.rename(expected_output, destquant)
        else:
             print(f"Warning: pngnq did not produce expected output file: {expected_output}", file=sys.stderr)

    else:
        raise NotImplementedError(f'Unknown quantization algorithm "{algorithm}"')


def palette_remap(src, destremap, paletteimg, dither=None):
    """remap src to paletteimage's colors, save to destremap"""
    if not os.path.exists(paletteimg):
        raise IOError(f"Remapping palette image not found: {paletteimg}")

    ditheropt = 'None' if dither is None else dither
    command = f'{IMAGEMAGICK_CONVERT_PATH} "{src}" -dither {ditheropt} -remap "{paletteimg}" "{destremap}"'
    process_command(command)


def make_palette(srcimage):
    """get unique colors from srcimage, return #rrggbb hex color strings"""
    command = f'{IMAGEMAGICK_CONVERT_PATH} "{srcimage}" -unique-colors -compress none ppm:-'
    stdoutput = process_command(command, stdout_=True)

    lines = stdoutput.decode(errors='ignore').splitlines()
    # Skip PPM header lines (P3, dimensions, max color value)
    color_lines = [line for line in lines if not line.startswith('#') and line.strip() != ''][3:]

    # Combine all color values into a single list of numbers
    colorvals = []
    for line in color_lines:
        colorvals.extend(int(s) for s in line.split())

    # Create hex color strings
    hex_colors = []
    for i in range(0, len(colorvals), 3):
        rgb = colorvals[i:i+3]
        hex_colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    hex_colors.reverse()
    return hex_colors


def get_nonpalette_color(palette, start_black=True, additional=None):
    """return a color hex string not listed in palette"""
    palette_ = set(p.lower() for p in palette)
    if additional:
        palette_.update(a.lower() for a in additional)

    color_range = range(int('ffffff', 16) + 1) if start_black else range(int('ffffff', 16), -1, -1)

    for i in color_range:
        color = f"#{i:06x}"
        if color not in palette_:
            return color
    raise Exception("All colors exhausted, could not find a nonpalette color")


def isolate_color(src, destlayer, target_color, palette, stack=False):
    """fills the specified color of src with black, all else is white"""
    coloridx = palette.index(target_color)

    bg_white = "#FFFFFF"
    fg_black = "#000000"
    bg_almost_white = get_nonpalette_color(palette, False, (bg_white, fg_black))
    fg_almost_black = get_nonpalette_color(palette, True, (bg_almost_white, bg_white, fg_black))

    command_mid = ''
    for i, col in enumerate(palette):
        fill = fg_almost_black if i == coloridx or (i > coloridx and stack) else bg_almost_white
        command_mid += f' -fill "{fill}" -opaque "{col}"'

    # Combine all fills into one command, then finalize colors.
    command = (f'{IMAGEMAGICK_CONVERT_PATH} "{src}"{command_mid} '
               f'-fill "{bg_white}" -opaque "{bg_almost_white}" '
               f'-fill "{fg_black}" -opaque "{fg_almost_black}" "{destlayer}"')
    process_command(command)


def get_width(src):
    """return width of src image in pixels"""
    command = f'{IMAGEMAGICK_IDENTIFY_PATH} -ping -format "%w" "{src}"'
    stdoutput = process_command(command, stdout_=True)
    return int(stdoutput)


def trace(src, desttrace, outcolor, despeckle=2, smoothcorners=1.0, optimizepaths=0.2, width=None):
    """runs potrace with specified color and options"""
    width_arg = f'--width {width/POTRACE_DPI}' if width is not None else ''
    command = (f'"{POTRACE_PATH}" "{src}" --svg --output "{desttrace}" '
               f'--color "{outcolor}" --turdsize {despeckle} '
               f'--alphamax {smoothcorners} --opttolerance {optimizepaths} {width_arg}')
    process_command(command)



def run_ocr_replace_text_path(image_file, svg_file, output_stem):
    """Runs Tesseract OCR, scales coordinates, and replaces text paths in svg."""
    verbose(f"Running OCR on '{image_file}' and replacing text paths in '{svg_file}'...")
    # Tesseract automatically adds the .tsv extension to the output stem
    command = f'"{TESSERACT_PATH}" "{image_file}" "{output_stem}" -l eng tsv'
    process_command(command)
    tsv_file = f"{output_stem}.tsv"
    verbose(f"Successfully created '{tsv_file}'")

    # --- Helper functions for SVG transformation parsing ---

    def combine_transforms(t1, t2):
        """Combines two transformation matrices (t1 * t2)."""
        a1, b1, c1, d1, e1, f1 = t1
        a2, b2, c2, d2, e2, f2 = t2
        return [
            a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1
        ]

    def parse_transform(transform_string):
        """Parses a complete SVG transform string into a single transformation matrix."""
        transform_regex = re.compile(r'(\w+)\(([^)]+)\)')
        final_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if not transform_string:
            return final_matrix
        for func, params_str in transform_regex.findall(transform_string):
            params = [float(p) for p in re.split(r'[,\s]+', params_str.strip()) if p]
            current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            if func == 'matrix':
                current_matrix = params
            elif func == 'translate':
                tx = params[0]
                ty = params[1] if len(params) > 1 else 0
                current_matrix = [1, 0, 0, 1, tx, ty]
            elif func == 'scale':
                sx = params[0]
                sy = params[1] if len(params) > 1 else sx
                current_matrix = [sx, 0, 0, sy, 0, 0]
            final_matrix = combine_transforms(final_matrix, current_matrix)
        return final_matrix

    def transform_point(point, matrix):
        """Applies a transformation matrix to a point."""
        x, y = point
        a, b, c, d, e, f = matrix
        return (a * x + c * y + e, b * x + d * y + f)

    def get_first_point(path_data):
        """Extracts the first point from an SVG path 'd' attribute."""
        match = re.search(r'[Mm]\s*([-\d.]+)[,\s]+([-\d.]+)', path_data)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None

    # --- Main Logic ---

    try:
        # Step 1: Parse all words from the TSV file
        words = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader) # Store header for later
            for row in reader:
                # A valid word row has 12 columns and text in the last one
                if len(row) == 12 and row[11].strip():
                    words.append({
                        "text": row[11],
                        "left": int(row[6]), "top": int(row[7]),
                        "width": int(row[8]), "height": int(row[9])
                    })
        
        if not words:
            verbose("OCR found no words, no paths will be removed from the SVG.")
            return
            
        # Step 2: Calculate the scaling factor and scale the OCR word coordinates
        scale_factor = 72.0 / POTRACE_DPI
        scaled_words = []
        for word in words:
            scaled_words.append({
                "text": word["text"],
                "left": word["left"] * scale_factor,
                "top": word["top"] * scale_factor,
                "width": word["width"] * scale_factor,
                "height": word["height"] * scale_factor
            })
        verbose(f"Calculated coordinate scale factor: {scale_factor}. Scaled {len(words)} OCR boxes.")

        # Step 3: Parse SVG and find paths to remove
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_file)
        root = tree.getroot()
        parent_map = {c: p for p in tree.iter() for c in p}
        paths_to_remove = []
        svg_ns = "{http://www.w3.org/2000/svg}"

        def traverse_and_find(element, parent_transform):
            local_transform_str = element.get('transform', '')
            local_transform = parse_transform(local_transform_str)
            current_transform = combine_transforms(parent_transform, local_transform)

            for child in element:
                tag = child.tag
                if tag == f'{svg_ns}g':
                    traverse_and_find(child, current_transform)
                elif tag == f'{svg_ns}path':
                    path_data = child.get('d')
                    if not path_data:
                        continue
                    
                    first_point = get_first_point(path_data)
                    if first_point:
                        global_point = transform_point(first_point, current_transform)
                        gx, gy = global_point

                        # Use the SCALED word boxes for the comparison
                        for word in scaled_words:
                            if (gx >= word["left"] and gx <= (word["left"] + word["width"]) and
                                gy >= word["top"] and gy <= (word["top"] + word["height"])):
                                paths_to_remove.append(child)
                                break 
        
        identity_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        traverse_and_find(root, identity_matrix)

        # Step 4: Remove the identified paths
        if paths_to_remove:
            verbose(f"Removing {len(paths_to_remove)} vectorized text paths from SVG...")
            for path in paths_to_remove:
                if path in parent_map:
                    parent = parent_map[path]
                    parent.remove(path)
        
        # Step 5: Save the modified SVG
        tree.write(svg_file, encoding='utf-8', xml_declaration=True)
        verbose(f"Successfully modified '{svg_file}' with text paths removed.")

    except (FileNotFoundError, IndexError, ValueError, ET.ParseError) as e:
        print(f"An error occurred during verification and text replacement for '{image_file}': {e}", file=sys.stderr)
        return




def check_range(min_val, max_val, typefunc, typename, strval):
    """for argparse type functions, checks the range of a value"""
    try:
        val = typefunc(strval)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be {typename}")
    if (max_val is not None and not min_val <= val <= max_val):
        raise argparse.ArgumentTypeError(f"must be between {min_val} and {max_val}")
    elif not min_val <= val:
        raise argparse.ArgumentTypeError(f"must be {min_val} or greater")
    return val


def get_args(cmdargs=None):
    """return parser and namespace of parsed command-line arguments"""
    parser = argparse.ArgumentParser(description="Trace a color image with Potrace, outputting a color SVG file.", add_help=False)
    parser.add_argument('-h', '--help', action='help', help="Show this help message and exit")

    # File IO arguments
    parser.add_argument('-i', '--input', metavar='src', nargs='+', required=True, help="Path of input image(s) to trace, supports wildcards.")
    parser.add_argument('-o', '--output', metavar='dest', help="Path of output image to save to, supports * as a wildcard for the input name.")
    parser.add_argument('-d', '--directory', metavar='destdir', help="Directory to save output files in.")

    # Color/Palette options
    color_palette_group = parser.add_mutually_exclusive_group(required=True)
    color_palette_group.add_argument('-c', '--colors', metavar='N', type=functools.partial(check_range, 0, 256, int, "an integer"), help="Number of colors to reduce image to (0-256). 0 skips reduction.")
    color_palette_group.add_argument('-r', '--remap', metavar='paletteimg', help="Use a custom palette image for color reduction.")

    # Quantization and Dithering
    parser.add_argument('-q', '--quantization', metavar='algorithm', choices=('mc', 'as', 'nq'), default='mc', help="Color quantization algorithm: 'mc' (pngquant, default), 'as' (ImageMagick), or 'nq' (pngnq).")
    dither_group = parser.add_mutually_exclusive_group()
    dither_group.add_argument('-fs', '--floydsteinberg', action='store_true', help="Enable Floyd-Steinberg dithering.")
    dither_group.add_argument('-ri', '--riemersma', action='store_true', help="Enable Riemersma dithering (only for 'as' quantization or --remap).")

    # Image options
    parser.add_argument('-s', '--stack', action='store_true', help="Stack color traces for more accurate output.")
    parser.add_argument('-p', '--prescale', metavar='size', type=functools.partial(check_range, 0.1, None, float, "a number"), default=1.0, help="Scale image by this factor before tracing for greater detail (default: 1.0).")

    # Potrace options
    parser.add_argument('-D', '--despeckle', metavar='size', type=functools.partial(check_range, 0, None, int, "an integer"), default=2, help='Suppress speckles of this many pixels (potrace --turdsize, default: 2).')
    parser.add_argument('-S', '--smoothcorners', metavar='threshold', type=functools.partial(check_range, 0, 1.334, float, "a number"), default=1.0, help="Set corner smoothing (potrace --alphamax, 0-1.334, default: 1.0).")
    parser.add_argument('-O', '--optimizepaths', metavar='tolerance', type=functools.partial(check_range, 0, 5, float, "a number"), default=0.2, help="Set Bezier curve optimization (potrace --opttolerance, 0-5, default: 0.2).")
    parser.add_argument('-bg', '--background', metavar='color', help="Specify a background color (e.g., '#FFFFFF') to be ignored during tracing.")

    # Other options
    parser.add_argument('-v', '--verbose', action='store_true', help="Print details about commands executed by this script.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    args = parser.parse_args(cmdargs)

    # Validation
    if len(args.input) > 1 and args.output and '*' not in args.output:
        parser.error("argument -o/--output: must contain '*' wildcard when using multiple input files.")
    if args.riemersma and args.quantization != 'as' and not args.remap:
        parser.error("argument -ri/--riemersma: only allowed with 'as' quantization or --remap.")

    return args


def get_inputs_outputs(arg_inputs, output_pattern):
    """Yields (input, output) pairs, expanding shell wildcards."""
    # Using a set to avoid processing the same file twice if wildcards overlap
    processed_inputs = set()
    for arg_input in arg_inputs:
        # glob handles wildcards like * and ?
        for input_path in iglob(arg_input):
            # Use absolute path to have a unique identifier
            abs_input_path = os.path.abspath(input_path)
            if abs_input_path not in processed_inputs:
                basename = os.path.splitext(os.path.basename(input_path))[0]
                output_path = output_pattern.format(basename)
                yield input_path, output_path
                processed_inputs.add(abs_input_path)

def color_trace_sequential(args):
    """Main sequential processing loop."""
    # Set output filename pattern
    if args.output is None:
        output_pattern = "{0}.svg"
    elif '*' in args.output:
        output_pattern = args.output.replace('*', "{0}")
    else:
        output_pattern = args.output

    if args.directory:
        # Ensure the output directory exists
        os.makedirs(args.directory, exist_ok=True)
        output_pattern = os.path.join(args.directory, os.path.basename(output_pattern))

    if args.floydsteinberg:
        dither = 'floydsteinberg'
    elif args.riemersma:
        dither = 'riemersma'
    else:
        dither = None

    # Process each file
    for input_file, output_file in get_inputs_outputs(args.input, output_pattern):
        verbose(f"\nProcessing '{input_file}' -> '{output_file}'")
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            continue

        # Use a temporary directory for intermediate files for this image
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # 1. Rescale
                scaled_path = os.path.join(tmpdir, "scaled.png")
                filter_ = 'point' if args.colors == 0 else 'lanczos'
                rescale(input_file, scaled_path, args.prescale, filter=filter_)

                # 2. Quantize or Remap
                reduced_path = os.path.join(tmpdir, "reduced.png")
                if args.colors is not None:
                    quantize(scaled_path, reduced_path, args.colors, algorithm=args.quantization, dither=dither)
                elif args.remap:
                    palette_remap(scaled_path, reduced_path, args.remap, dither=dither)

                # 3. Get Palette and original width
                palette = make_palette(reduced_path)
                width = get_width(input_file)
                
                if args.background:
                    bg_color_lower = args.background.lower()
                    palette = [p for p in palette if p.lower() != bg_color_lower]
                    verbose(f"Ignoring background color {args.background}. Tracing {len(palette)} colors.")


                # 4. Isolate and Trace each color, collecting the output paths
                trace_paths = []
                for i, color in enumerate(palette):
                    verbose(f"  - Tracing color {i+1}/{len(palette)}: {color}")
                    isolated_path = os.path.join(tmpdir, f"isolated_{i}.bmp")
                    trace_path = os.path.join(tmpdir, f"trace_{i}.svg")
                    trace_paths.append(trace_path)

                    isolate_color(reduced_path, isolated_path, color, palette, stack=args.stack)
                    trace(isolated_path, trace_path, color, args.despeckle, args.smoothcorners, args.optimizepaths, width)

                # 5. Merge SVG layers using the PowerShell text-based logic
                verbose("Merging SVG layers...")
                
                # Concatenate all temporary SVG file contents
                merged_content_parts = []
                for path in trace_paths:
                    with open(path, 'r', encoding='utf-8') as f:
                        merged_content_parts.append(f.read())
                merged_content = "\n".join(merged_content_parts)

                # Find the very first SVG tag from the merged content and keep it
                first_svg_tag_match = re.search(r'<svg[^>]*>', merged_content, flags=re.IGNORECASE)
                if not first_svg_tag_match:
                    raise Exception("Could not find a valid <svg> tag in the generated files.")
                first_svg_tag = first_svg_tag_match.group(0)

                # Post-merge cleanup with regex
                clean_content = re.sub(r'<\?xml.*?\?>', '', merged_content, flags=re.IGNORECASE)
                clean_content = re.sub(r'<!DOCTYPE[^>]*>', '', clean_content, flags=re.IGNORECASE)
                clean_content = re.sub(r'<metadata>.*?</metadata>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove all SVG start and end tags from the body of the content
                clean_content = re.sub(r'<svg[^>]*>', '', clean_content, flags=re.IGNORECASE)
                clean_content = re.sub(r'</svg>', '', clean_content, flags=re.IGNORECASE)
                
                # Reconstruct the final SVG
                final_svg_content = f"{first_svg_tag}\n{clean_content.strip()}\n</svg>"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_svg_content)
                verbose(f"Successfully created '{output_file}'")

            except Exception as e:
                print(f"\nAn error occurred while processing '{input_file}': {e}", file=sys.stderr)
                # Continue to the next file
                continue

      # 6. Run OCR and replace text paths after SVG is successfully created
        if os.path.exists(output_file):
            try:
                # The output stem is the full path to the output file, without its extension
                output_stem = os.path.splitext(output_file)[0]
                run_ocr_replace_text_path(input_file, output_file, output_stem)
            except Exception as e:
                print(f"\nAn error occurred during OCR and SVG text path replacement for '{input_file}': {e}", file=sys.stderr)
                continue

def main():
    """Main entry point."""
    args = get_args()

    global VERBOSITY_LEVEL
    if args.verbose:
        VERBOSITY_LEVEL = 1

    color_trace_sequential(args)
    print("\nProcessing complete.")


if __name__ == '__main__':
    main()