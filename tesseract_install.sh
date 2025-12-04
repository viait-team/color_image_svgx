#!/bin/bash
# tesseract_install.sh
# Script to build Tesseract OCR from source and install language data
# Target: Ubuntu 22.04+
# Version: Tesseract 5.5.0 (or latest)

set -e  # Exit on error

# Configuration
TESSERACT_VERSION="5.5.0"
TESSDATA_REPO="https://github.com/tesseract-ocr/tessdata_best/raw/main"
INSTALL_PREFIX="/usr/local"
TESSDATA_DIR="$INSTALL_PREFIX/share/tessdata"

echo ">>> Starting Tesseract $TESSERACT_VERSION Installation <<<"

# 1. Install Dependencies
echo ">>> Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libpango1.0-dev libcairo2-dev libpangoft2-1.0-0 libpangocairo-1.0-0 \
    libfontconfig1-dev libicu-dev libleptonica-dev \
    git cmake automake libtool pkg-config

# 2. Clone and Build Tesseract
echo ">>> Cloning Tesseract..."
WORKDIR=$(mktemp -d)
cd "$WORKDIR"

git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
# Checkout specific version if needed, or use default (main/latest)
# git checkout "$TESSERACT_VERSION" 2>/dev/null || echo "Version tag not found, using main branch"

echo ">>> Building Tesseract..."
./autogen.sh
./configure --prefix="$INSTALL_PREFIX"
make -j"$(nproc)"
sudo make install
sudo ldconfig

# 3. Install Language Data
echo ">>> Installing Language Data (eng, osd)..."
sudo mkdir -p "$TESSDATA_DIR"

# Function to download if missing
download_lang() {
    local lang=$1
    local url="$TESSDATA_REPO/$lang.traineddata"
    local dest="$TESSDATA_DIR/$lang.traineddata"
    
    if [ ! -f "$dest" ]; then
        echo "Downloading $lang.traineddata..."
        sudo wget -O "$dest" "$url"
        sudo chmod 644 "$dest"
    else
        echo "$lang.traineddata already exists."
    fi
}

download_lang "eng"
download_lang "osd"

# 4. Environment Setup
echo ">>> Configuring Environment..."
SHELL_RC="$HOME/.bashrc"

if ! grep -q "TESSDATA_PREFIX" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# Tesseract OCR Data Path" >> "$SHELL_RC"
    echo "export TESSDATA_PREFIX=$TESSDATA_DIR" >> "$SHELL_RC"
    echo "Added TESSDATA_PREFIX to $SHELL_RC"
else
    echo "TESSDATA_PREFIX already set in $SHELL_RC"
fi

# Cleanup
rm -rf "$WORKDIR"

echo ">>> Installation Complete! <<<"
echo "Please run: source ~/.bashrc"
echo "Verify with: tesseract --version"
