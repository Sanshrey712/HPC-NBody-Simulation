# Parallel N-Body Simulation with Barnes–Hut Optimization using OpenMP, MPI, CUDA & Profiling

## 1. Project Title

**Design and Performance Evaluation of a Hybrid Parallel N-Body Simulation Framework using OpenMP, MPI, and CUDA**

## 2. Project Abstract

N-body simulation is a fundamental computational problem in physics, astrophysics, molecular dynamics, and scientific computing, where the motion of particles is governed by pairwise interactions. Direct computation of interactions scales as O(N²), making it computationally infeasible for large systems.

This project aims to design and implement a high-performance N-body simulation framework using:

- Serial baseline implementation
- Shared-memory parallelization using OpenMP
- Distributed-memory parallelization using MPI
- GPU acceleration using CUDA

To improve computational efficiency, the **Barnes–Hut algorithm** (O(N log N)) will be implemented.

The project emphasizes performance engineering and profiling, using:

- `gprof` for function-level profiling
- `gcov` for code coverage and branch analysis
- `LIKWID` for hardware performance counter measurement

The final system will enable detailed performance analysis, scalability evaluation, and optimization, making this a top-tier undergraduate HPC academic project.

## 3. Learning & Engineering Objectives

### Technical Objectives

- Implement N-body simulation using classical Newtonian gravity
- Implement Barnes–Hut hierarchical tree algorithm
- Develop four execution models:
  - Serial
  - OpenMP
  - MPI
  - CUDA
- Perform deep profiling and performance analysis

### HPC Objectives

Study shared-memory vs distributed-memory vs GPU parallelism.

Analyze:
- Cache behavior
- Memory bandwidth
- Communication overhead
- GPU occupancy & efficiency

### Academic Objectives

- Perform strong & weak scaling analysis
- Build roofline performance models
- Produce publication-quality performance graphs

## 4. Problem Definition

Given N particles, each with mass, position, and velocity, simulate their motion under gravitational interaction:

```
F_i = Σ(j≠i) G · (m_i · m_j) / |r_i - r_j|²
```

### Computational Challenge:

- Naive algorithm → O(N²)
- For N = 10⁶ → 10¹² interactions per timestep
- This is computationally infeasible

## 5. Algorithm Design

### 5.1 Baseline Algorithm – Direct N² Simulation

**Steps:**
1. For each particle i
2. For each particle j
3. Compute gravitational force
4. Update velocity and position

**Complexity:**
- Time: O(N²)
- Memory: O(N)

**Used only for:**
- Baseline comparison
- Validation
- Profiling reference

### 5.2 Optimized Algorithm – Barnes–Hut Algorithm

**Core Idea:**

Approximate distant particle clusters as single mass points using a hierarchical spatial tree.

**Data Structure:** Quad-tree (2D) / Octree (3D)

Each node stores:
- Center of mass
- Total mass
- Spatial boundary
- Child pointers

**Algorithm Steps:**
1. Build spatial tree
2. For each particle:
   - Traverse tree
   - If distant node: approximate
   - Else: recursively traverse children
3. Compute force
4. Integrate motion

**Complexity:**
- Time: O(N log N)
- Memory: O(N)

## 6. System Architecture

```
                     Input Generator
                            |
                            v
                 +---------------------+
                 |   Simulation Core   |
                 |---------------------|
                 | Tree Construction   |
                 | Force Computation   |
                 | Time Integration    |
                 +---------------------+
                            |
        --------------------------------------------------
        |                     |                        |
        v                     v                        v
    OpenMP Engine         MPI Engine               CUDA Engine
        |                     |                        |
        +---------------------+------------------------+
                            |
                            v
                     Profiling Engine
                 (gprof + gcov + LIKWID)
```

## 7. Software Stack

| Component | Technology |
|-----------|------------|
| Language | C / C++ |
| Shared Memory | OpenMP |
| Distributed Memory | MPI |
| GPU Acceleration | CUDA |
| Profiling | gprof, gcov, LIKWID |
| Build | CMake + Make |

## 8. Implementation Plan (PHASE-WISE)

### PHASE 1 — Serial Baseline Implementation

**Goals:**
- Correctness
- Reference performance
- Profiling baseline

**Modules to Build:**

1. **Particle Generator**
   - Uniform random
   - Gaussian clusters
   - Galaxy-style spiral distribution

2. **Direct N² Solver**
   - Pairwise force calculation
   - Euler / Leapfrog integration

3. **Barnes–Hut Tree Construction**

4. **Barnes–Hut Force Computation**

5. **Simulation Controller**
   - Timestep loop
   - Position updates
   - Output generation

**Deliverables:**
- Serial N² solver
- Serial Barnes–Hut solver
- Validation test suite
- Baseline timing

### PHASE 2 — OpenMP Parallel Implementation

**Parallelization Strategy:**

| Component | Parallelization |
|-----------|----------------|
| Tree construction | Parallel particle insertion |
| Force computation | Parallel loop over particles |
| Integration | Parallel update |

**OpenMP Techniques:**
- `#pragma omp parallel for`
- Dynamic scheduling
- Reduction operations
- Thread pinning

**Goals:**
- Shared-memory scaling study
- Cache behavior analysis
- NUMA impact study

### PHASE 3 — MPI Distributed Implementation

**Domain Decomposition Strategy:**

**Spatial Partitioning**

Each MPI process handles:
- A subregion of space
- Local particles
- Local tree

**Communication:**
- Boundary particle exchange
- Center-of-mass summaries
- MPI_Allreduce for global statistics

**MPI Communication Pattern:**
```
Local Tree → Boundary Exchange → Force Compute → Position Update → Sync
```

**MPI Primitives Used:**
- MPI_Isend
- MPI_Irecv
- MPI_Allreduce
- MPI_Barrier

**Goals:**
- Communication vs computation study
- Strong & weak scaling

### PHASE 4 — CUDA GPU Implementation

**GPU Parallelization Strategy:**

| Component | GPU Strategy |
|-----------|-------------|
| Direct N² | Massive kernel launch |
| Tree traversal | Warp-level traversal |
| Force compute | Thread per particle |

**CUDA Optimization Techniques:**
- Shared memory caching
- Coalesced memory access
- Kernel fusion
- Warp divergence minimization
- Occupancy tuning

**GPU Kernels:**
- Force computation kernel
- Tree traversal kernel
- Integration kernel

## 9. Profiling & Performance Analysis Plan

### 9.1 gprof – Function Profiling

**Used to identify:**

Time spent in:
- Tree construction
- Force computation
- MPI communication
- CUDA kernels

### 9.2 gcov – Code Coverage & Branch Profiling

**Used to analyze:**
- Branch divergence in tree traversal
- Loop execution frequency
- Dead code detection

### 9.3 LIKWID – Hardware Performance Counters

**Measured metrics:**

| Metric | Purpose |
|--------|---------|
| L1/L2/L3 cache misses | Memory locality |
| Memory bandwidth | Bottleneck analysis |
| CPI | Pipeline efficiency |
| NUMA traffic | Memory locality |
| Vectorization | SIMD utilization |

## 10. Performance Experiments

### 10.1 Strong Scaling

Fix N = 10⁶

Increase:
- Threads (OpenMP)
- Processes (MPI)
- GPU blocks

### 10.2 Weak Scaling

Increase N proportionally with:
- Threads / Processes

### 10.3 CPU vs GPU Comparison

Compare:
- Serial vs OpenMP vs MPI vs CUDA

### 10.4 Roofline Modeling

Plot:
- FLOPS vs Arithmetic Intensity

## 11. Input Sizes & Benchmarks

| N | Use Case |
|---|----------|
| 10⁴ | Debug |
| 10⁵ | Profiling |
| 10⁶ | Strong scaling |
| 10⁷ | GPU stress |

## 12. Expected Performance

| Platform | Expected Speedup |
|----------|-----------------|
| OpenMP | 8× – 20× |
| MPI | 20× – 80× |
| CUDA | 40× – 200× |

## 13. Project Timeline (10–12 Weeks)

| Week | Task |
|------|------|
| 1–2 | Serial N² + BH |
| 3–4 | OpenMP |
| 5–6 | MPI |
| 7–8 | CUDA |
| 9 | Profiling |
| 10 | Optimization |
| 11 | Experiments |
| 12 | Report |

## 14. Final Report Structure (ACADEMIC GRADE)

1. Introduction
2. Mathematical Model
3. Algorithm Design
4. Serial Implementation
5. OpenMP Parallelization
6. MPI Distributed Framework
7. CUDA GPU Acceleration
8. Profiling Methodology
9. Experimental Results
10. Performance Modeling
11. Optimization Analysis
12. Conclusion

## 15. Final Deliverables


1. Complete Source Code Repository

   - Serial N-Body simulation (O(N²))
   - Barnes–Hut optimized implementation
   - OpenMP-based shared-memory parallel version
   - MPI-based distributed-memory version
   - CUDA-based GPU-accelerated version
   - Build system using CMake/Make

2. Profiling and Performance Analysis Reports

   - Function-level profiling results (gprof)
   - Branch and code coverage analysis (gcov)
   - Hardware performance counter analysis (LIKWID)
   - GPU kernel profiling results (CUDA profiling tools)

3. Scalability and Benchmarking Results
   - Strong scaling graphs
   - Weak scaling graphs
   - CPU vs GPU performance comparison
   - Communication vs computation analysis (MPI)

4. Performance Modeling Artifacts
   - Roofline performance models
   - Cache and memory bandwidth analysis
   - Arithmetic intensity calculations

5. Validation and Correctness Evidence
   - Numerical accuracy comparison between serial and parallel versions
   - Conservation checks (energy / momentum)
   - Test cases for small and large N

6. Final Technical Report
   - Detailed explanation of algorithms
   - Parallelization strategies
   - Profiling methodology
   - Experimental setup and results
   - Optimization insights and conclusions

7. Optional Supplementary Materials
   - Execution scripts for benchmarking
   - Visualization of particle trajectories
   - Documentation for running on multi-core, multi-node, and GPU systems