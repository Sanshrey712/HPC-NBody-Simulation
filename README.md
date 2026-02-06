# Parallel N-Body Simulation with Barnes-Hut Optimization

High-performance N-body gravitational simulation using OpenMP, MPI, and CUDA with comprehensive profiling support.

## Features

- **Serial baseline**: Direct O(N²) and Barnes-Hut O(N log N) algorithms
- **OpenMP**: Shared-memory parallelization with dynamic scheduling
- **MPI**: Distributed-memory with domain decomposition and halo exchange
- **CUDA**: GPU acceleration with shared memory tiling
- **Profiling**: gprof, gcov, LIKWID integration

## Quick Start

```bash
# Build
./scripts/build.sh Release

# Run simulation
./build/nbody --particles 10000 --steps 100 --mode openmp --threads 4 --barnes-hut

# See all options
./build/nbody --help
```

## Build Options

```bash
./scripts/build.sh Release       # Optimized build
./scripts/build.sh Debug         # Debug symbols
./scripts/build.sh gprof         # gprof profiling
./scripts/build.sh gcov          # Code coverage
./scripts/build.sh likwid        # LIKWID counters
```

## Profiling

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

### Scaling Experiments
```bash
./scripts/run_scaling.sh 8 100000
# Output: results/strong_scaling*.csv, results/weak_scaling*.csv
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-n, --particles N` | Number of particles |
| `-s, --steps N` | Timesteps |
| `--mode MODE` | serial, openmp, mpi, cuda |
| `-t, --threads N` | OpenMP threads |
| `--theta T` | Barnes-Hut opening angle |
| `--direct` | Use O(N²) method |
| `--barnes-hut` | Use O(N log N) method |

## Requirements

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- OpenMP (optional)
- MPI (optional, OpenMPI recommended)
- CUDA Toolkit 11+ (optional)
- LIKWID (optional)

## Project Structure

```
├── src/
│   ├── common/      # Data structures
│   ├── serial/      # Serial implementations
│   ├── openmp/      # OpenMP parallel
│   ├── mpi/         # MPI distributed
│   └── cuda/        # CUDA GPU
├── scripts/         # Profiling scripts
├── tests/           # Validation tests
└── profiling/       # Profiling output
```
