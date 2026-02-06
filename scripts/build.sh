#!/bin/bash
# Build script for N-Body simulation with different configurations

set -e

BUILD_DIR="build"
BUILD_TYPE="${1:-Release}"

echo "=== N-Body Simulation Build Script ==="
echo "Build type: $BUILD_TYPE"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Choose build configuration
case "$BUILD_TYPE" in
    "Release")
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DUSE_OPENMP=ON \
                 -DUSE_MPI=ON \
                 -DUSE_CUDA=ON
        ;;
    "Debug")
        cmake .. -DCMAKE_BUILD_TYPE=Debug \
                 -DUSE_OPENMP=ON \
                 -DUSE_MPI=ON \
                 -DUSE_CUDA=ON
        ;;
    "gprof")
        cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                 -DUSE_OPENMP=ON \
                 -DUSE_MPI=OFF \
                 -DUSE_CUDA=OFF \
                 -DENABLE_GPROF=ON
        ;;
    "gcov")
        cmake .. -DCMAKE_BUILD_TYPE=Debug \
                 -DUSE_OPENMP=ON \
                 -DUSE_MPI=OFF \
                 -DUSE_CUDA=OFF \
                 -DENABLE_GCOV=ON
        ;;
    "likwid")
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DUSE_OPENMP=ON \
                 -DUSE_MPI=OFF \
                 -DUSE_CUDA=OFF \
                 -DENABLE_LIKWID=ON
        ;;
    "serial-only")
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DUSE_OPENMP=OFF \
                 -DUSE_MPI=OFF \
                 -DUSE_CUDA=OFF
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        echo "Options: Release, Debug, gprof, gcov, likwid, serial-only"
        exit 1
        ;;
esac

# Build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete! Executables in $BUILD_DIR/"
echo ""
