# Parallel N-Body Simulation with Barnes-Hut Optimization

High-performance N-body gravitational simulation implementing **Direct ($O(N^2)$)** and **Barnes-Hut ($O(N \log N)$)** algorithms. The project leverages **OpenMP** for shared-memory parallelism, **MPI** for distributed-memory systems, and **CUDA** for GPU acceleration.

![Barnes-Hut Visualization](src/assets/BarnesHut_MPI.png)

## Features

- **Algorithms**:
    - **Direct Method**: Brute-force calculation of all particle pairs. $O(N^2)$ complexity. Accurate but slow for large $N$.
    - **Barnes-Hut**: Tree-based approximation using an Octree. $O(N \log N)$ complexity. Significant speedup for large datasets ($N > 10,000$).
- **Parallel Implementations**:
    - **OpenMP**: Multi-threaded execution with dynamic scheduling to handle load usage.
    - **MPI**: Domain decomposition with halo exchange for multi-node clusters.
    - **CUDA**: Extremely fast GPU implementation using shared memory tiling.
- **Profiling**: Integrated support for `gprof`, `gcov`, and `LIKWID` hardware counters.

## Performance Results

We conducted extensive benchmarking to compare the implementations. Detailed PDF reports are available:

*   **[OpenMP Report](OpenMP_REPORT.pdf)**: Shared-memory analysis.
*   **[MPI Report](MPI_Report.pdf)**: Distributed-memory analysis.

### Summary of Results

#### OpenMP (4 Threads, 100,000 Particles)
The Barnes-Hut algorithm demonstrates a clear advantage at this scale, providing a **1.5x speedup** over the Direct method despite the overhead of rebuilding the Octree at every step.

| Method | Time Complexity | Total Time | Interaction Rate |
| :--- | :--- | :--- | :--- |
| **Direct** | $O(N^2)$ | 950.56 s | ~5.26 M/s |
| **Barnes-Hut** | $O(N \log N)$ | 641.60 s | ~968 M/s (Effective) |

#### MPI (4 Ranks, 10,000 Particles)
MPI scales the problem across strictly distinct memory spaces. Barnes-Hut again proves superior, achieving a **1.53x speedup** with significantly reduced computational load.

| Method | Total Time | Interactions | Speedup |
| :--- | :--- | :--- | :--- |
| **Direct** | 61.93 s | 8.37 x $10^7$ | 1.0x |
| **Barnes-Hut** | 40.52 s | 2.09 x $10^7$ | **1.53x** |

## Quick Start

### Build
```bash
./scripts/build.sh Release
```

### Run Simulation

**OpenMP (Shared Memory)**
```bash
# Barnes-Hut (O(N log N)) - Best for Single Node
./build/nbody --particles 100000 --steps 100 --mode openmp --threads 4 --barnes-hut

# Direct Method (O(N^2))
./build/nbody --particles 10000 --steps 100 --mode openmp --threads 4 --direct
```

**MPI (Distributed Memory)**
```bash
# Barnes-Hut (O(N log N)) - Best for Clusters
mpirun -np 4 ./build/nbody --particles 10000 --steps 100 --mode mpi --barnes-hut

# Direct Method (O(N^2))
mpirun -np 4 ./build/nbody --particles 10000 --steps 100 --mode mpi --direct
```

## Build Options

```bash
./scripts/build.sh Release       # Optimized build
./scripts/build.sh Debug         # Debug symbols
./scripts/build.sh gprof         # gprof profiling
./scripts/build.sh gcov          # Code coverage
./scripts/build.sh likwid        # LIKWID counters
```

## Profiling & Analysis Tools

The project includes scripts to automate profiling:

### gprof (Function Profiling)
Identifies "hot" functions where the CPU spends most of its time.
```bash
./scripts/run_gprof.sh 10000 50 serial
# Output: profiling/gprof/flat_profile_*.txt
```

### gcov (Code Coverage)
Verifies which lines of code are executed during a run.
```bash
./scripts/run_gcov.sh 1000 10
# Output: profiling/gcov/*.gcov
```

### LIKWID (Hardware Counters)
Measures hardware metrics like FLOPS, Cache Misses, and Memory Bandwidth.
```bash
./scripts/run_likwid.sh 10000 50 FLOPS_DP 4
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-n, --particles N` | Number of particles |
| `-s, --steps N` | Timesteps |
| `--mode MODE` | serial, openmp, mpi, cuda |
| `-t, --threads N` | OpenMP threads |
| `--theta T` | Barnes-Hut opening angle (default: 0.5) |
| `--direct` | Use O(N²) method |
| `--barnes-hut` | Use O(N log N) method |

## Project Structure
```
├── src/
│   ├── common/      # Core data structures (Particle, Vector3D)
│   ├── serial/      # Serial logic & Octree implementation
│   ├── openmp/      # OpenMP implementation (Pro-active parallelization)
│   ├── mpi/         # MPI implementation (Domain Decomposition)
│   ├── cuda/        # CUDA GPU implementation (Kernel launch)
│   └── assets/      # Visualization & Report assets
├── scripts/         # Build and profiling shell scripts
├── profiling/       # Verification and profiling output directories
├── OpenMP_REPORT.pdf # Detailed OpenMP Performance Analysis
└── MPI_Report.pdf    # Detailed MPI Performance Analysis
```
