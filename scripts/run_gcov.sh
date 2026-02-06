#!/bin/bash
# gcov code coverage script
# Usage: ./run_gcov.sh [particles] [steps]

set -e

PARTICLES=${1:-1000}
STEPS=${2:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
PROFILE_DIR="$PROJECT_DIR/profiling/gcov"

echo "=== gcov Code Coverage ==="
echo "Particles: $PARTICLES"
echo "Steps: $STEPS"
echo ""

# Create output directory
mkdir -p "$PROFILE_DIR"

# Ensure gcov build exists
if [ ! -f "$BUILD_DIR/nbody_gcov" ]; then
    echo "Building gcov version..."
    cd "$PROJECT_DIR"
    ./scripts/build.sh gcov
fi

cd "$BUILD_DIR"

# Clean previous coverage data
find . -name "*.gcda" -delete 2>/dev/null || true

# Run simulation (serial with both methods)
echo "Running direct method..."
./nbody_gcov --particles $PARTICLES --steps $STEPS --mode serial --direct

echo "Running Barnes-Hut method..."
./nbody_gcov --particles $PARTICLES --steps $STEPS --mode serial --barnes-hut

# Run OpenMP if available
if ./nbody_gcov --help 2>&1 | grep -q "openmp"; then
    echo "Running OpenMP mode..."
    ./nbody_gcov --particles $PARTICLES --steps $STEPS --mode openmp --threads 4
fi

# Generate coverage reports
echo ""
echo "Generating coverage reports..."

# Process each source file
for gcda in $(find . -name "*.gcda"); do
    gcov "$gcda" > /dev/null 2>&1 || true
done

# Move gcov files to profile directory
mv *.gcov "$PROFILE_DIR/" 2>/dev/null || true

# Generate summary
echo ""
echo "=== Coverage Summary ==="
for gcov_file in "$PROFILE_DIR"/*.gcov; do
    if [ -f "$gcov_file" ]; then
        filename=$(basename "$gcov_file" .gcov)
        lines_executed=$(grep -c "^[^-]" "$gcov_file" 2>/dev/null | tr -cd '0-9')
        [ -z "$lines_executed" ] && lines_executed=0
        
        lines_not_executed=$(grep -c "^####" "$gcov_file" 2>/dev/null | tr -cd '0-9')
        [ -z "$lines_not_executed" ] && lines_not_executed=0
        
        total=$((lines_executed + lines_not_executed))
        if [ $total -gt 0 ]; then
            coverage=$((lines_executed * 100 / total))
            echo "$filename: $coverage% ($lines_executed/$total lines)"
        fi
    fi
done

# Create HTML report if lcov is available
if command -v lcov &> /dev/null && command -v genhtml &> /dev/null; then
    echo ""
    echo "Generating HTML report..."
    
    lcov --capture --directory . --output-file "$PROFILE_DIR/coverage.info" --ignore-errors source 2>/dev/null
    lcov --remove "$PROFILE_DIR/coverage.info" '/usr/*' --output-file "$PROFILE_DIR/coverage.info" 2>/dev/null
    genhtml "$PROFILE_DIR/coverage.info" --output-directory "$PROFILE_DIR/html" 2>/dev/null
    
    echo "HTML report: $PROFILE_DIR/html/index.html"
fi

# Detailed branch analysis
echo ""
echo "=== Branch Coverage Analysis ==="
for gcov_file in "$PROFILE_DIR"/*.gcov; do
        if [ -f "$gcov_file" ]; then
        filename=$(basename "$gcov_file" .gcov)
        
        # Sanitize grep output: keep only numbers, remove whitespace, default to 0
        branches=$(grep -c "branch" "$gcov_file" 2>/dev/null | tr -cd '0-9')
        [ -z "$branches" ] && branches=0
        
        taken=$(grep "branch.*taken" "$gcov_file" 2>/dev/null | grep -v "never" | wc -l | tr -cd '0-9')
        [ -z "$taken" ] && taken=0
        
        if [ "$branches" -gt 0 ]; then
            echo "$filename: $taken/$branches branches taken"
        fi
    fi
done

echo ""
echo "Coverage files saved to: $PROFILE_DIR/"
