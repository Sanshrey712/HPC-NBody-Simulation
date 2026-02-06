#!/bin/bash
# LIKWID performance counter profiling script
# Usage: ./run_likwid.sh [particles] [steps] [group]
# Groups: FLOPS_DP, MEM, L2CACHE, L3CACHE, BRANCH, CPI, ENERGY

set -e

PARTICLES=${1:-10000}
STEPS=${2:-50}
GROUP=${3:-"FLOPS_DP"}
THREADS=${4:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
PROFILE_DIR="$PROJECT_DIR/profiling/likwid"

echo "=== LIKWID Performance Counter Profiling ==="
echo "Particles: $PARTICLES"
echo "Steps: $STEPS"  
echo "Performance Group: $GROUP"
echo "Threads: $THREADS"
echo ""

# Check LIKWID availability
if ! command -v likwid-perfctr &> /dev/null; then
    echo "Error: LIKWID not found. Please install LIKWID:"
    echo "  Ubuntu/Debian: sudo apt install likwid"
    echo "  macOS: brew install likwid (limited support)"
    echo "  Manual: https://github.com/RRZE-HPC/likwid"
    exit 1
fi

# Create output directory
mkdir -p "$PROFILE_DIR"

# Build with LIKWID support
if [ ! -f "$BUILD_DIR/nbody" ]; then
    echo "Building with LIKWID support..."
    cd "$PROJECT_DIR"
    ./scripts/build.sh likwid || ./scripts/build.sh Release
fi

cd "$BUILD_DIR"

# Define CPU affinity
CPUS="0-$((THREADS-1))"

# Available performance groups
GROUPS=("FLOPS_DP" "MEM" "L2CACHE" "L3CACHE" "BRANCH" "CPI")

# Run with specified group
OUTPUT_FILE="$PROFILE_DIR/likwid_${GROUP}_${PARTICLES}p_${STEPS}s_${THREADS}t.txt"

echo "Running LIKWID with group: $GROUP"
echo "CPU affinity: $CPUS"
echo ""

likwid-perfctr -C $CPUS -g $GROUP -m ./nbody \
    --particles $PARTICLES \
    --steps $STEPS \
    --mode openmp \
    --threads $THREADS \
    --barnes-hut \
    > "$OUTPUT_FILE" 2>&1

echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "=== LIKWID Results Summary ==="
grep -A 100 "Region" "$OUTPUT_FILE" | head -50

# Generate comparison across groups
if [ "$5" == "all" ]; then
    echo ""
    echo "Running all performance groups..."
    
    for g in "${GROUPS[@]}"; do
        echo "  Running $g..."
        OUTPUT="$PROFILE_DIR/likwid_${g}_${PARTICLES}p_${STEPS}s_${THREADS}t.txt"
        
        likwid-perfctr -C $CPUS -g $g -m ./nbody \
            --particles $PARTICLES \
            --steps $STEPS \
            --mode openmp \
            --threads $THREADS \
            --barnes-hut \
            > "$OUTPUT" 2>&1
    done
    
    echo ""
    echo "All group results saved to: $PROFILE_DIR/"
fi

# Summary report
echo ""
echo "=== Performance Analysis ==="

if [ -f "$PROFILE_DIR/likwid_FLOPS_DP_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" ]; then
    echo "FLOPS:"
    grep -i "flops" "$PROFILE_DIR/likwid_FLOPS_DP_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" | head -5
fi

if [ -f "$PROFILE_DIR/likwid_MEM_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" ]; then
    echo ""
    echo "Memory Bandwidth:"
    grep -i "bandwidth" "$PROFILE_DIR/likwid_MEM_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" | head -5
fi

if [ -f "$PROFILE_DIR/likwid_L3CACHE_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" ]; then
    echo ""
    echo "L3 Cache:"
    grep -i "miss\|hit\|ratio" "$PROFILE_DIR/likwid_L3CACHE_${PARTICLES}p_${STEPS}s_${THREADS}t.txt" | head -5
fi
