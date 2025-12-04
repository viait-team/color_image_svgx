#!/bin/bash
# tools_install.sh
# Script to install ImageMagick 7, pngquant, and potrace
# Target: Ubuntu 22.04+

set -e  # Exit on error

INSTALL_PREFIX="/usr/local"
WORKDIR=$(mktemp -d)

echo ">>> Starting Tools Installation <<<"

# 1. Update Repositories
sudo apt-get update

# 2. Install Potrace (Available in apt)
echo ">>> Installing potrace..."
if sudo apt-get install -y potrace; then
    echo "potrace installed: $(potrace --version | head -n 1)"
else
    echo "Failed to install potrace via apt."
    exit 1
fi

# 3. Install pngquant (Available in apt)
echo ">>> Installing pngquant..."
if sudo apt-get install -y pngquant; then
    echo "pngquant installed: $(pngquant --version)"
else
    echo "Failed to install pngquant via apt."
    exit 1
fi

# 4. Install ImageMagick 7 (Source Build required for v7)
# Ubuntu 22.04 only provides ImageMagick 6 via apt.
# We need v7 for the 'magick' command and better OCR preprocessing.
echo ">>> Installing ImageMagick 7 (Source Build)..."

# Install build dependencies
sudo apt-get install -y \
    build-essential git curl wget \
    libjpeg-dev libpng-dev libtiff-dev libgif-dev \
    libfreetype6-dev liblcms2-dev libxml2-dev \
    libfontconfig1-dev libx11-dev libxext-dev libxt-dev \
    liblzma-dev zlib1g-dev checkinstall ghostscript

# Remove existing ImageMagick 6 if present
sudo apt-get purge -y imagemagick imagemagick-6-common || true

# Clone Latest Source (matches user's setup)
cd "$WORKDIR"
echo "Cloning ImageMagick..."
git clone https://github.com/ImageMagick/ImageMagick.git
cd ImageMagick

# Build and Install using checkinstall (creates a clean .deb package)
echo "Configuring ImageMagick..."
./configure --prefix="$INSTALL_PREFIX" --with-modules

echo "Compiling ImageMagick (this may take a while)..."
make -j"$(nproc)"

echo "Installing ImageMagick via checkinstall..."
# --default accepts default answers
# --pkgname sets the package name
# --pkgversion sets the version (we extract it or use date)
PKG_VERSION=$(./util/ImageMagick-config --version | cut -d ' ' -f 1)
sudo checkinstall --default --pkgname=imagemagick-source --pkgversion="$PKG_VERSION" --nodoc

sudo ldconfig

# Verify ImageMagick
if command -v magick >/dev/null; then
    echo "ImageMagick installed: $(magick --version | head -n 1)"
else
    echo "Error: ImageMagick installation failed."
    exit 1
fi

# Cleanup
rm -rf "$WORKDIR"

echo ">>> Tools Installation Complete! <<<"
echo "Installed: potrace, pngquant, magick (ImageMagick 7)"
