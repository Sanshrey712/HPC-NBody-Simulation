#!/bin/bash
# gprof profiling script
# Usage: ./run_gprof.sh [particles] [steps] [mode]

set -e

PARTICLES=${1:-10000}
STEPS=${2:-50}
MODE=${3:-"serial"}
METHOD=${4:-"barnes-hut"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
PROFILE_DIR="$PROJECT_DIR/profiling/gprof"

echo "=== gprof Profiling ==="
echo "Particles: $PARTICLES"
echo "Steps: $STEPS"
echo "Mode: $MODE"
echo "Method: $METHOD"
echo ""

# Create output directory
mkdir -p "$PROFILE_DIR"

# Ensure gprof build exists
if [ ! -f "$BUILD_DIR/nbody_gprof" ]; then
    echo "Building gprof version..."
    cd "$PROJECT_DIR"
    ./scripts/build.sh gprof
fi

cd "$BUILD_DIR"

# Run simulation
echo "Running simulation..."
./nbody_gprof --particles $PARTICLES --steps $STEPS --mode $MODE --$METHOD

# Generate gprof report
if [ -f gmon.out ]; then
    echo ""
    echo "Generating gprof report..."
    
    # Flat profile
    gprof --flat-profile ./nbody_gprof gmon.out > "$PROFILE_DIR/flat_profile_${PARTICLES}p_${STEPS}s_${MODE}.txt"
    
    # Call graph
    gprof --graph ./nbody_gprof gmon.out > "$PROFILE_DIR/call_graph_${PARTICLES}p_${STEPS}s_${MODE}.txt"
    
    # Annotated source (if available)
    gprof --annotated-source ./nbody_gprof gmon.out > "$PROFILE_DIR/annotated_${PARTICLES}p_${STEPS}s_${MODE}.txt" 2>/dev/null || true
    
    # Summary
    echo ""
    echo "=== Top 10 Functions by Time ==="
    head -30 "$PROFILE_DIR/flat_profile_${PARTICLES}p_${STEPS}s_${MODE}.txt"
    
    echo ""
    echo "Full reports saved to: $PROFILE_DIR/"
    echo "  - flat_profile_${PARTICLES}p_${STEPS}s_${MODE}.txt"
    echo "  - call_graph_${PARTICLES}p_${STEPS}s_${MODE}.txt"
    
    # Clean up
    rm -f gmon.out
else
    echo "Error: gmon.out not generated"
    exit 1
fi
