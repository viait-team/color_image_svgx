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

# 2. Install Potrace
echo ">>> Installing potrace..."
if sudo apt-get install -y potrace; then
    echo "potrace installed: $(potrace --version | head -n 1)"
else
    echo "Failed to install potrace via apt."
    exit 1
fi

# 3. Install pngquant
echo ">>> Installing pngquant..."
if sudo apt-get install -y pngquant; then
    echo "pngquant installed: $(pngquant --version)"
else
    echo "Failed to install pngquant via apt."
    exit 1
fi

# 4. Install ImageMagick 7 (Source Build)
echo ">>> Installing ImageMagick 7 (Source Build)..."

# Install build dependencies
# ADDED: libwebp-dev, libheif-dev, libopenjp2-7-dev, libdjvulibre-dev, libopenexr-dev
# These prevent "missing delegate" errors for modern formats.
sudo apt-get install -y \
    build-essential git curl wget \
    libjpeg-dev libpng-dev libtiff-dev libgif-dev \
    libfreetype6-dev liblcms2-dev libxml2-dev \
    libfontconfig1-dev libx11-dev libxext-dev libxt-dev \
    liblzma-dev zlib1g-dev checkinstall ghostscript \
    libwebp-dev libheif-dev libopenjp2-7-dev libdjvulibre-dev libopenexr-dev

# Remove existing ImageMagick 6 if present
sudo apt-get purge -y imagemagick imagemagick-6-common || true

# Clone Latest Source
cd "$WORKDIR"
echo "Cloning ImageMagick..."
git clone https://github.com/ImageMagick/ImageMagick.git
cd ImageMagick

# Build and Install
echo "Configuring ImageMagick..."
./configure --prefix="$INSTALL_PREFIX" --with-modules

echo "Compiling ImageMagick (this may take a while)..."
make -j"$(nproc)"

echo "Installing ImageMagick via checkinstall..."
PKG_VERSION=$(./util/ImageMagick-config --version | cut -d ' ' -f 1)
sudo checkinstall --default --pkgname=imagemagick-source --pkgversion="$PKG_VERSION" --nodoc

# --- FIX FOR MISSING LIB ERROR ---
echo ">>> Configuring Dynamic Linker Run-time Bindings..."

# 1. Ensure /usr/local/lib is in the search path
if [ ! -f /etc/ld.so.conf.d/libc.conf ]; then
    echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf
else
    # Verify it exists in the default configs, if not append it
    if ! grep -q "/usr/local/lib" /etc/ld.so.conf.d/*.conf; then
        echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/imagemagick-local.conf
    fi
fi

# 2. Update the cache immediately
sudo ldconfig

# Verify ImageMagick
if command -v magick >/dev/null; then
    echo "ImageMagick installed successfully: $(magick --version | head -n 1)"
else
    echo "Error: ImageMagick installation failed."
    exit 1
fi

# Cleanup
rm -rf "$WORKDIR"

echo ">>> Tools Installation Complete! <<<"
echo "--------------------------------------------------------"
echo "If you still see 'error while loading shared libraries':"
echo "Run this command or add it to your ~/.bashrc:"
echo ""
echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
echo "--------------------------------------------------------"