#!/bin/bash
# quick_install_blender.sh - Quick Blender installation (for environments without sudo privileges)

set -e

echo "🔧 Quick Blender Installation for Cluster"

# Try using the pre-compiled portable version
BLENDER_VERSION="4.0.2"
INSTALL_DIR="${HOME}/.local/blender"
BLENDER_DIR="${INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="${BLENDER_DIR}/blender"

# Check if already installed in system PATH
if command -v blender &> /dev/null; then
    echo "✓ Blender already available in PATH"
    export BLENDER_EXECUTABLE=$(which blender)
    blender --version
    exit 0
fi

# Check if local installation already exists
if [ -d "${BLENDER_DIR}" ]; then
    echo "✓ Blender directory found at ${BLENDER_DIR}"
    if [ -f "${BLENDER_BIN}" ]; then
        echo "✓ Blender binary found at ${BLENDER_BIN}"
        chmod +x "${BLENDER_BIN}"
        export BLENDER_EXECUTABLE="${BLENDER_BIN}"
        export PATH="${BLENDER_DIR}:${PATH}"
        echo "✓ Blender ready: ${BLENDER_EXECUTABLE}"
        ${BLENDER_EXECUTABLE} --version
        exit 0
    else
        echo "⚠️  Blender directory exists but binary not found, re-installing..."
        rm -rf "${BLENDER_DIR}"
    fi
fi

# Download and install
echo "Downloading Blender ${BLENDER_VERSION} (portable version)..."
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

BLENDER_ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
DOWNLOAD_URL="https://download.blender.org/release/Blender4.0/${BLENDER_ARCHIVE}"

# Clean up possibly existing old download files
rm -f "${BLENDER_ARCHIVE}"

# Download using wget or curl
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget -q --show-progress "${DOWNLOAD_URL}" || {
        echo "❌ Download failed with wget"
        exit 1
    }
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o "${BLENDER_ARCHIVE}" "${DOWNLOAD_URL}" || {
        echo "❌ Download failed with curl"
        exit 1
    }
else
    echo "❌ Neither wget nor curl available!"
    exit 1
fi

# Check the downloaded file
if [ ! -f "${BLENDER_ARCHIVE}" ]; then
    echo "❌ Download failed: ${BLENDER_ARCHIVE} not found"
    exit 1
fi

echo "File downloaded: $(ls -lh ${BLENDER_ARCHIVE})"
echo "Extracting..."
tar -xf "${BLENDER_ARCHIVE}" || {
    echo "❌ Extraction failed"
    rm -f "${BLENDER_ARCHIVE}"
    exit 1
}

# Clean up the downloaded archive
rm -f "${BLENDER_ARCHIVE}"

# Check extraction results
echo "Checking extracted files..."
ls -la "${INSTALL_DIR}/"

if [ ! -f "${BLENDER_BIN}" ]; then
    echo "❌ Blender binary not found after extraction: ${BLENDER_BIN}"
    echo "Contents of ${INSTALL_DIR}:"
    find "${INSTALL_DIR}" -type f -name "blender" 2>/dev/null || echo "No blender binary found"
    exit 1
fi

# Ensure the binary has execute permission
chmod +x "${BLENDER_BIN}"

# Set environment variables
export BLENDER_EXECUTABLE="${BLENDER_BIN}"
export PATH="${BLENDER_DIR}:${PATH}"

echo "✓ Blender installed successfully!"
echo "BLENDER_EXECUTABLE=${BLENDER_EXECUTABLE}"

# Verify Blender (using --background to avoid GUI dependency)
if [ -x "${BLENDER_EXECUTABLE}" ]; then
    echo "Testing Blender (background mode)..."
    # Note: LD_LIBRARY_PATH must include the necessary libraries
    # This script assumes the caller has already set the correct environment variables
    if "${BLENDER_EXECUTABLE}" --background --version >/dev/null 2>&1; then
        echo "✓ Blender is ready and working!"
        "${BLENDER_EXECUTABLE}" --background --version 2>&1 | head -n 3
    else
        echo "⚠️  Blender binary installed but may need LD_LIBRARY_PATH set"
        echo "Checking for missing libraries..."
        ldd "${BLENDER_EXECUTABLE}" | grep "not found" || echo "Library check complete"
        echo ""
        echo "Note: If you see missing libraries above, make sure to:"
        echo "  export LD_LIBRARY_PATH=\"\$HOME/miniconda/lib:\$LD_LIBRARY_PATH\""
        echo "  before running Blender"
        echo ""
        echo "Installation completed, verification will be done by the calling script"
    fi
else
    echo "❌ Blender binary not executable!"
    ls -la "${BLENDER_BIN}"
    exit 1
fi
