# This script runs the color_image_svgx.py script in batch mode.
# It first activates the Python virtual environment in 'myenv'.
# It then processes all .png files in the 'inputs' directory and saves the resulting .svg files to the 'outputs' directory.

# Activate the virtual environment
.\myenv\Scripts\Activate.ps1

# Run the main script
python color_image_svgx.py --input "inputs\*.png" --directory "outputs" --colors 16 --verbose

# python color_image_svgx.py --input "inputs_2024\*.jpg" --directory "outputs_2024" --colors 8 --verbose