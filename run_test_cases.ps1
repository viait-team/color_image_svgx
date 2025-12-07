# This script runs the color_image_svgx.py script in batch mode.
# It first activates the Python virtual environment in 'myenv'.
# It then processes all .png files in the 'inputs' directory and saves the resulting .svg files to the 'outputs' directory.

# Activate the virtual environment
.\myenv\Scripts\Activate.ps1

# Run the main script
python color_image_svgx.py --input "test_cases\*.png" --directory "test_cases" --colors 2 --verbose
