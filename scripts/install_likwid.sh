#!/bin/bash
# Script to install Likwid from source
# Usage: ./install_likwid.sh

set -e

LIKWID_VERSION="5.2.2"
INSTALL_PREFIX="/usr/local"
TEMP_DIR="/tmp/likwid_install"

echo "=== Installing Likwid $LIKWID_VERSION from source ==="

# Check for sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo to install to $INSTALL_PREFIX"
  echo "Try: sudo ./install_likwid.sh"
  exit 1
fi

# Install build dependencies if missing (redundant but safe)
apt-get update
apt-get install -y git build-essential

# Prepare temp directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download Likwid
echo "Downloading Likwid..."
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid
git checkout "v$LIKWID_VERSION"

# Configure (Modify config.mk if needed, default is usually /usr/local)
# We ensure the access daemon is built
sed -i 's#ACCESSMODE = access_daemon#ACCESSMODE = access_daemon#g' config.mk
sed -i "s#PREFIX = /usr/local#PREFIX = $INSTALL_PREFIX#g" config.mk

# Detect Architecture and set compiler
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Detected ARM64 architecture. Configuring for GCCARMv8..."
    sed -i 's#^COMPILER .*#COMPILER = GCCARMv8#g' config.mk
fi

# Build
echo "Building Likwid..."
make

# Install
echo "Installing Likwid..."
make install

# Configure access daemon permissions (if installed)
if [ -f "$INSTALL_PREFIX/sbin/likwid-accessD" ]; then
    echo "Configuring access daemon..."
    chown root:root $INSTALL_PREFIX/sbin/likwid-accessD
    chmod u+s $INSTALL_PREFIX/sbin/likwid-accessD
else
    echo "likwid-accessD not found (normal for perf_event mode on ARM64)."
fi

# Clean up
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "=== Likwid Installation Complete ==="
echo "You may need to run 'sudo modprobe msr' to load the MSR module."
echo "Verify installation with: likwid-topology"
