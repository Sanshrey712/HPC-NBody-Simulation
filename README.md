# Parallel N-Body Simulation with Barnes-Hut Optimization

High-performance N-body gravitational simulation using OpenMP, MPI, and CUDA with comprehensive profiling support.

## Features

- **Serial baseline**: Direct O(N²) and Barnes-Hut O(N log N) algorithms
- **OpenMP**: Shared-memory parallelization with dynamic scheduling
- **MPI**: Distributed-memory with domain decomposition and halo exchange
- **CUDA**: GPU acceleration with shared memory tiling
- **Profiling**: gprof, gcov, LIKWID integration

## Performance Reports

Comprehensive performance analysis reports are available:
- **[OpenMP Report](OpenMP_REPORT.pdf)**: Analysis of shared-memory performance (Direct vs. Barnes-Hut).
- **[MPI Report](MPI_Report.pdf)**: Analysis of distributed-memory performance on multi-node systems.

## Quick Start

### Build
```bash
./scripts/build.sh Release
```

### Run Simulation

**OpenMP (Shared Memory)**
```bash
# Barnes-Hut (O(N log N))
./build/nbody --particles 100000 --steps 100 --mode openmp --threads 4 --barnes-hut

# Direct Method (O(N^2))
./build/nbody --particles 10000 --steps 100 --mode openmp --threads 4 --direct
```

**MPI (Distributed Memory)**
```bash
# Barnes-Hut (O(N log N)) - Recommended for large N
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

## Profiling & Analysis

### gprof (Function Profiling)
```bash
./scripts/run_gprof.sh 10000 50 serial
# Output: profiling/gprof/flat_profile_*.txt
```

### gcov (Code Coverage)
```bash
./scripts/run_gcov.sh 1000 10
# Output: profiling/gcov/*.gcov
```

### LIKWID (Hardware Counters)
```bash
./scripts/run_likwid.sh 10000 50 FLOPS_DP 4
# Groups: FLOPS_DP, MEM, L2CACHE, L3CACHE, BRANCH, CPI
```

## command Line Options

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
│   ├── common/      # Data structures (Particle, Vector3D)
│   ├── serial/      # Serial logic & Octree implementation
│   ├── openmp/      # OpenMP implementation
│   ├── mpi/         # MPI implementation (Domain Decomposition)
│   ├── cuda/        # CUDA GPU implementation
│   └── assets/      # Visualization & Report assets
├── scripts/         # Build and profiling scripts
├── profiling/       # Profiling output
├── OpenMP_REPORT.pdf
└── MPI_Report.pdf
```
