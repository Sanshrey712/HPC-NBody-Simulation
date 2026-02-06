#!/bin/bash
# Scaling experiment script for performance analysis
# Usage: ./run_scaling.sh [max_threads] [particles]

set -e

MAX_THREADS=${1:-8}
PARTICLES=${2:-100000}
STEPS=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/results"

echo "=== Scaling Experiments ==="
echo "Max threads: $MAX_THREADS"
echo "Particles: $PARTICLES"
echo "Steps: $STEPS"
echo ""

mkdir -p "$RESULTS_DIR"

# Build if needed
if [ ! -f "$BUILD_DIR/nbody" ]; then
    echo "Building..."
    cd "$PROJECT_DIR"
    ./scripts/build.sh Release
fi

cd "$BUILD_DIR"

# ========================================
# Strong Scaling (fixed N, increasing threads)
# ========================================
echo ""
echo "=== Strong Scaling ==="
STRONG_FILE="$RESULTS_DIR/strong_scaling_${PARTICLES}p.csv"
echo "threads,time_ms,speedup,efficiency" > "$STRONG_FILE"

# Serial baseline
echo "Running serial baseline..."
SERIAL_TIME=$(./nbody --particles $PARTICLES --steps $STEPS --mode serial --barnes-hut --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
echo "Serial time: ${SERIAL_TIME}ms"
echo "1,$SERIAL_TIME,1.0,100.0" >> "$STRONG_FILE"

# OpenMP scaling
for t in 2 4 8 16; do
    if [ $t -le $MAX_THREADS ]; then
        echo "Running with $t threads..."
        TIME=$(./nbody --particles $PARTICLES --steps $STEPS --mode openmp --threads $t --barnes-hut --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
        SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $TIME" | bc)
        EFFICIENCY=$(echo "scale=1; $SPEEDUP / $t * 100" | bc)
        echo "$t,$TIME,$SPEEDUP,$EFFICIENCY" >> "$STRONG_FILE"
        echo "  $t threads: ${TIME}ms (speedup: ${SPEEDUP}x, efficiency: ${EFFICIENCY}%)"
    fi
done

# ========================================
# Weak Scaling (N proportional to threads)
# ========================================
echo ""
echo "=== Weak Scaling ==="
BASE_N=$((PARTICLES / MAX_THREADS))
WEAK_FILE="$RESULTS_DIR/weak_scaling_${BASE_N}p_per_thread.csv"
echo "threads,particles,time_ms,efficiency" > "$WEAK_FILE"

for t in 1 2 4 8 16; do
    if [ $t -le $MAX_THREADS ]; then
        N=$((BASE_N * t))
        echo "Running $t threads with $N particles..."
        
        if [ $t -eq 1 ]; then
            TIME=$(./nbody --particles $N --steps $STEPS --mode serial --barnes-hut --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
            BASELINE=$TIME
        else
            TIME=$(./nbody --particles $N --steps $STEPS --mode openmp --threads $t --barnes-hut --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
        fi
        
        EFFICIENCY=$(echo "scale=1; $BASELINE / $TIME * 100" | bc)
        echo "$t,$N,$TIME,$EFFICIENCY" >> "$WEAK_FILE"
        echo "  $t threads, $N particles: ${TIME}ms (efficiency: ${EFFICIENCY}%)"
    fi
done

# ========================================
# Algorithm Comparison (Direct vs Barnes-Hut)
# ========================================
echo ""
echo "=== Algorithm Comparison ==="
ALGO_FILE="$RESULTS_DIR/algorithm_comparison.csv"
echo "particles,direct_ms,barnes_hut_ms,speedup" > "$ALGO_FILE"

for N in 1000 2000 5000 10000; do
    echo "N=$N..."
    DIRECT=$(./nbody --particles $N --steps 10 --mode openmp --threads $MAX_THREADS --direct --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
    BH=$(./nbody --particles $N --steps 10 --mode openmp --threads $MAX_THREADS --barnes-hut --no-energy 2>&1 | grep "Total time" | awk '{print $3}')
    SPEEDUP=$(echo "scale=2; $DIRECT / $BH" | bc)
    echo "$N,$DIRECT,$BH,$SPEEDUP" >> "$ALGO_FILE"
    echo "  Direct: ${DIRECT}ms, Barnes-Hut: ${BH}ms (speedup: ${SPEEDUP}x)"
done

# ========================================
# Generate Summary
# ========================================
echo ""
echo "=== Results Summary ==="
echo "Strong scaling results: $STRONG_FILE"
echo "Weak scaling results: $WEAK_FILE"
echo "Algorithm comparison: $ALGO_FILE"

# Create Gnuplot script for visualization
cat > "$RESULTS_DIR/plot_scaling.gp" << 'EOF'
set terminal png size 1200,400
set output 'scaling_plots.png'
set multiplot layout 1,3

# Strong Scaling
set title "Strong Scaling"
set xlabel "Threads"
set ylabel "Speedup"
set key left top
plot 'strong_scaling_*.csv' using 1:3 with linespoints title "Actual", \
     x title "Ideal" with lines

# Weak Scaling  
set title "Weak Scaling"
set xlabel "Threads"
set ylabel "Efficiency (%)"
plot 'weak_scaling_*.csv' using 1:4 with linespoints title "Efficiency", \
     100 title "Ideal" with lines

# Algorithm Comparison
set title "Algorithm Comparison"
set xlabel "Particles"
set ylabel "Time (ms)"
set logscale x
set key right top
plot 'algorithm_comparison.csv' using 1:2 with linespoints title "Direct O(N²)", \
     'algorithm_comparison.csv' using 1:3 with linespoints title "Barnes-Hut O(N log N)"

unset multiplot
EOF

if command -v gnuplot &> /dev/null; then
    cd "$RESULTS_DIR"
    gnuplot plot_scaling.gp
    echo "Plots generated: $RESULTS_DIR/scaling_plots.png"
fi

echo ""
echo "Done!"
