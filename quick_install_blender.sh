#!/bin/bash
# quick_install_blender.sh - 快速安装 Blender（适用于无 sudo 权限的环境）

set -e

echo "🔧 Quick Blender Installation for Cluster"

# 尝试使用预编译的便携版
BLENDER_VERSION="4.0.2"
INSTALL_DIR="${HOME}/.local/blender"
BLENDER_DIR="${INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="${BLENDER_DIR}/blender"

# 检查是否已安装在系统PATH中
if command -v blender &> /dev/null; then
    echo "✓ Blender already available in PATH"
    export BLENDER_EXECUTABLE=$(which blender)
    blender --version
    exit 0
fi

# 检查是否已经存在本地安装
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

# 下载并安装
echo "Downloading Blender ${BLENDER_VERSION} (portable version)..."
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

BLENDER_ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
DOWNLOAD_URL="https://download.blender.org/release/Blender4.0/${BLENDER_ARCHIVE}"

# 清理可能存在的旧下载文件
rm -f "${BLENDER_ARCHIVE}"

# 使用 wget 或 curl 下载
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

# 检查下载的文件
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

# 清理下载的压缩包
rm -f "${BLENDER_ARCHIVE}"

# 检查解压结果
echo "Checking extracted files..."
ls -la "${INSTALL_DIR}/"

if [ ! -f "${BLENDER_BIN}" ]; then
    echo "❌ Blender binary not found after extraction: ${BLENDER_BIN}"
    echo "Contents of ${INSTALL_DIR}:"
    find "${INSTALL_DIR}" -type f -name "blender" 2>/dev/null || echo "No blender binary found"
    exit 1
fi

# 确保二进制文件有执行权限
chmod +x "${BLENDER_BIN}"

# 设置环境变量
export BLENDER_EXECUTABLE="${BLENDER_BIN}"
export PATH="${BLENDER_DIR}:${PATH}"

echo "✓ Blender installed successfully!"
echo "BLENDER_EXECUTABLE=${BLENDER_EXECUTABLE}"

# 验证 Blender（使用 --background 避免 GUI 依赖）
if [ -x "${BLENDER_EXECUTABLE}" ]; then
    echo "Testing Blender (background mode)..."
    # 注意：需要 LD_LIBRARY_PATH 包含必要的库
    # 这个脚本假设调用者已经设置了正确的环境变量
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
