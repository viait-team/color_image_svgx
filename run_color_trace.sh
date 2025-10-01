#!/bin/bash

# === Configuration ===
INPUT_DIR="inputs"
OUTPUT_DIR="outputs"
COLOR_COUNT=16
SCRIPT="color_trace_multi.py"

# === Check Python ===
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3."
    exit 1
fi

# === Check script exists ===
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Script '$SCRIPT' not found in current directory."
    exit 1
fi

# === Create output directory if missing ===
mkdir -p "$OUTPUT_DIR"

# === Run the command ===
echo "🚀 Running color_trace_multi.py with $COLOR_COUNT colors..."
python "$SCRIPT" --input "$INPUT_DIR"/*.png --directory "$OUTPUT_DIR" --colors "$COLOR_COUNT" --verbose