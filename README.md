# Parallel N-Body Simulation with Barnes-Hut Optimization

# Parallel N-Body Simulation with Barnes-Hut Optimization

## Introduction

The N-Body problem is a classic problem in physics and computer science that involves simulating the motion of a system of particles under the influence of physical forces, such as gravity. For a system of $N$ particles, a direct brute-force approach requires computing $O(N^2)$ interactions per time step, which becomes computationally prohibitive as $N$ grows.

This project implements a high-performance N-Body simulation to explore and compare different parallel computing paradigms. It features two core algorithms:
1.  **Direct Method ($O(N^2)$)**: Computes exact forces between all particle pairs. Highly accurate but computationally expensive.
2.  **Barnes-Hut Algorithm ($O(N \log N)$)**: Uses an Octree spatial data structure to approximate forces from distant particles, significantly reducing computational complexity for large datasets.

The simulation is implemented using three major parallel programming models:
-   **OpenMP**: Shared-memory parallelization for multi-core CPUs.
-   **MPI**: Distributed-memory parallelization for clusters and multi-node systems, employing domain decomposition and halo exchange.
-   **CUDA**: Massively parallel acceleration on NVIDIA GPUs.

This codebase serves as a comprehensive benchmark for analyzing the trade-offs between accuracy, speed, and implementation complexity across these different architectures.

## Features

- **Scalable Algorithms**: Implements both Direct and Barnes-Hut methods to handle small to large-scale simulations.
- **Multi-Paradigm Support**:
    - **OpenMP**: Dynamic scheduling to effectively load-balance the irregular tree traversal of Barnes-Hut.
    - **MPI**: Robust domain decomposition handling particle migration and boundary data exchange.
    - **CUDA**: Optimized kernels using shared memory tiling to maximize GPU throughput.
- **Comprehensive Profiling**: Integrated support for `gprof`, `gcov`, and `LIKWID` to analyze CPU usage, code coverage, and hardware performance counters (FLOPS, cache hits).

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

## Key Takeaways

1.  **Algorithmic Superiority**: The **Barnes-Hut algorithm ($O(N \log N)$)** is essential for large-scale N-Body simulations, providing order-of-magnitude speedups over the Direct method ($O(N^2)$) once $N > 10,000$.
2.  **Parallel Scaling**:
    -   **OpenMP** effectively utilizes shared memory for simpler implementation and good performance on a single node.
    -   **MPI** successfully scales the problem across distributed memory, enabling the simulation of much larger systems than possible on a single machine.
3.  **Optimization Trade-offs**: While Barnes-Hut is faster, it introduces complexity in memory access (pointer chasing) and parallel load balancing (irregular tree structure), which must be carefully managed.

## References

1.  J. Barnes and P. Hut, "A hierarchical O(N log N) force-calculation algorithm," *Nature*, vol. 324, no. 4, pp. 446-449, 1986.
2.  L. Greengard and V. Rokhlin, "A fast algorithm for particle simulations," *Journal of Computational Physics*, vol. 73, no. 2, pp. 325-348, 1987.

## Author

**Sanshrey**\
Roll No.: CS23B2014\
Department of Computer Science and Engineering
