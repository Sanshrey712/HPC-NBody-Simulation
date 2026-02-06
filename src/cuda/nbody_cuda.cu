#include "nbody_cuda.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Tile size for shared memory optimization
#define TILE_SIZE 256

// ============================================================================
// CUDA Kernels
// ============================================================================

// Direct N² force computation kernel with shared memory tiling
__global__ void compute_forces_kernel(
    const double *__restrict__ pos_x, const double *__restrict__ pos_y,
    const double *__restrict__ pos_z, const double *__restrict__ mass,
    double *__restrict__ acc_x, double *__restrict__ acc_y,
    double *__restrict__ acc_z, int n, double G, double softening_sq) {
  // Shared memory for tile of particles
  __shared__ double s_pos_x[TILE_SIZE];
  __shared__ double s_pos_y[TILE_SIZE];
  __shared__ double s_pos_z[TILE_SIZE];
  __shared__ double s_mass[TILE_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double ax = 0.0, ay = 0.0, az = 0.0;
  double px, py, pz;

  if (i < n) {
    px = pos_x[i];
    py = pos_y[i];
    pz = pos_z[i];
  }

  // Loop over tiles
  for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    int j = tile * TILE_SIZE + threadIdx.x;

    // Load tile into shared memory
    if (j < n) {
      s_pos_x[threadIdx.x] = pos_x[j];
      s_pos_y[threadIdx.x] = pos_y[j];
      s_pos_z[threadIdx.x] = pos_z[j];
      s_mass[threadIdx.x] = mass[j];
    } else {
      s_pos_x[threadIdx.x] = 0.0;
      s_pos_y[threadIdx.x] = 0.0;
      s_pos_z[threadIdx.x] = 0.0;
      s_mass[threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Compute interactions with tile
    if (i < n) {
#pragma unroll 8
      for (int k = 0; k < TILE_SIZE; ++k) {
        int global_j = tile * TILE_SIZE + k;
        if (global_j < n && global_j != i) {
          double dx = s_pos_x[k] - px;
          double dy = s_pos_y[k] - py;
          double dz = s_pos_z[k] - pz;

          double dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
          double inv_dist = rsqrt(dist_sq);
          double inv_dist_cubed = inv_dist * inv_dist * inv_dist;

          double force_factor = G * s_mass[k] * inv_dist_cubed;

          ax += dx * force_factor;
          ay += dy * force_factor;
          az += dz * force_factor;
        }
      }
    }

    __syncthreads();
  }

  if (i < n) {
    acc_x[i] = ax;
    acc_y[i] = ay;
    acc_z[i] = az;
  }
}

// Leapfrog integration kernel (kick-drift)
__global__ void integrate_kick_drift_kernel(
    double *__restrict__ pos_x, double *__restrict__ pos_y,
    double *__restrict__ pos_z, double *__restrict__ vel_x,
    double *__restrict__ vel_y, double *__restrict__ vel_z,
    const double *__restrict__ acc_x, const double *__restrict__ acc_y,
    const double *__restrict__ acc_z, int n, double half_dt, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    // Kick
    vel_x[i] += acc_x[i] * half_dt;
    vel_y[i] += acc_y[i] * half_dt;
    vel_z[i] += acc_z[i] * half_dt;

    // Drift
    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
  }
}

// Leapfrog integration kernel (final kick)
__global__ void integrate_kick_kernel(double *__restrict__ vel_x,
                                      double *__restrict__ vel_y,
                                      double *__restrict__ vel_z,
                                      const double *__restrict__ acc_x,
                                      const double *__restrict__ acc_y,
                                      const double *__restrict__ acc_z, int n,
                                      double half_dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    vel_x[i] += acc_x[i] * half_dt;
    vel_y[i] += acc_y[i] * half_dt;
    vel_z[i] += acc_z[i] * half_dt;
  }
}

// ============================================================================
// Host Implementation
// ============================================================================

NBodyCUDA::NBodyCUDA(double G, double softening)
    : G_(G), softening_(softening), softening_sq_(softening * softening),
      num_particles_(0), block_size_(TILE_SIZE), interaction_count_(0),
      kernel_time_(0.0), d_pos_x_(nullptr), d_pos_y_(nullptr),
      d_pos_z_(nullptr), d_vel_x_(nullptr), d_vel_y_(nullptr),
      d_vel_z_(nullptr), d_acc_x_(nullptr), d_acc_y_(nullptr),
      d_acc_z_(nullptr), d_mass_(nullptr) {}

NBodyCUDA::~NBodyCUDA() { free_device_memory(); }

bool NBodyCUDA::is_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}

void NBodyCUDA::print_device_info() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  printf("\n=== CUDA Device Information ===\n");
  printf("Number of CUDA devices: %d\n", device_count);

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    printf("\nDevice %d: %s\n", i, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared memory per block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
  }
  printf("================================\n\n");
}

void NBodyCUDA::allocate_device_memory(int n) {
  size_t size = n * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_pos_x_, size));
  CUDA_CHECK(cudaMalloc(&d_pos_y_, size));
  CUDA_CHECK(cudaMalloc(&d_pos_z_, size));
  CUDA_CHECK(cudaMalloc(&d_vel_x_, size));
  CUDA_CHECK(cudaMalloc(&d_vel_y_, size));
  CUDA_CHECK(cudaMalloc(&d_vel_z_, size));
  CUDA_CHECK(cudaMalloc(&d_acc_x_, size));
  CUDA_CHECK(cudaMalloc(&d_acc_y_, size));
  CUDA_CHECK(cudaMalloc(&d_acc_z_, size));
  CUDA_CHECK(cudaMalloc(&d_mass_, size));
}

void NBodyCUDA::free_device_memory() {
  if (d_pos_x_)
    cudaFree(d_pos_x_);
  if (d_pos_y_)
    cudaFree(d_pos_y_);
  if (d_pos_z_)
    cudaFree(d_pos_z_);
  if (d_vel_x_)
    cudaFree(d_vel_x_);
  if (d_vel_y_)
    cudaFree(d_vel_y_);
  if (d_vel_z_)
    cudaFree(d_vel_z_);
  if (d_acc_x_)
    cudaFree(d_acc_x_);
  if (d_acc_y_)
    cudaFree(d_acc_y_);
  if (d_acc_z_)
    cudaFree(d_acc_z_);
  if (d_mass_)
    cudaFree(d_mass_);

  d_pos_x_ = d_pos_y_ = d_pos_z_ = nullptr;
  d_vel_x_ = d_vel_y_ = d_vel_z_ = nullptr;
  d_acc_x_ = d_acc_y_ = d_acc_z_ = nullptr;
  d_mass_ = nullptr;
}

void NBodyCUDA::copy_to_device(const std::vector<Particle> &particles) {
  int n = static_cast<int>(particles.size());

  // Convert AOS to SOA and copy
  std::vector<double> pos_x(n), pos_y(n), pos_z(n);
  std::vector<double> vel_x(n), vel_y(n), vel_z(n);
  std::vector<double> mass(n);

  for (int i = 0; i < n; ++i) {
    pos_x[i] = particles[i].position.x;
    pos_y[i] = particles[i].position.y;
    pos_z[i] = particles[i].position.z;
    vel_x[i] = particles[i].velocity.x;
    vel_y[i] = particles[i].velocity.y;
    vel_z[i] = particles[i].velocity.z;
    mass[i] = particles[i].mass;
  }

  size_t size = n * sizeof(double);
  CUDA_CHECK(cudaMemcpy(d_pos_x_, pos_x.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos_y_, pos_y.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pos_z_, pos_z.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vel_x_, vel_x.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vel_y_, vel_y.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vel_z_, vel_z.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mass_, mass.data(), size, cudaMemcpyHostToDevice));
}

void NBodyCUDA::copy_from_device(std::vector<Particle> &particles) {
  int n = num_particles_;
  particles.resize(n);

  std::vector<double> pos_x(n), pos_y(n), pos_z(n);
  std::vector<double> vel_x(n), vel_y(n), vel_z(n);
  std::vector<double> acc_x(n), acc_y(n), acc_z(n);

  size_t size = n * sizeof(double);
  CUDA_CHECK(cudaMemcpy(pos_x.data(), d_pos_x_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pos_y.data(), d_pos_y_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pos_z.data(), d_pos_z_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vel_x.data(), d_vel_x_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vel_y.data(), d_vel_y_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vel_z.data(), d_vel_z_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(acc_x.data(), d_acc_x_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(acc_y.data(), d_acc_y_, size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(acc_z.data(), d_acc_z_, size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    particles[i].position = Vector3D(pos_x[i], pos_y[i], pos_z[i]);
    particles[i].velocity = Vector3D(vel_x[i], vel_y[i], vel_z[i]);
    particles[i].acceleration = Vector3D(acc_x[i], acc_y[i], acc_z[i]);
    particles[i].id = i;
  }
}

void NBodyCUDA::initialize(const std::vector<Particle> &particles) {
  num_particles_ = static_cast<int>(particles.size());

  free_device_memory();
  allocate_device_memory(num_particles_);
  copy_to_device(particles);
}

void NBodyCUDA::compute_forces_direct() {
  int grid_size = (num_particles_ + block_size_ - 1) / block_size_;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  compute_forces_kernel<<<grid_size, block_size_>>>(
      d_pos_x_, d_pos_y_, d_pos_z_, d_mass_, d_acc_x_, d_acc_y_, d_acc_z_,
      num_particles_, G_, softening_sq_);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  kernel_time_ = milliseconds;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  interaction_count_ =
      static_cast<long long>(num_particles_) * (num_particles_ - 1) / 2;
}

void NBodyCUDA::integrate(double dt) {
  int grid_size = (num_particles_ + block_size_ - 1) / block_size_;
  double half_dt = 0.5 * dt;

  // Kick-Drift
  integrate_kick_drift_kernel<<<grid_size, block_size_>>>(
      d_pos_x_, d_pos_y_, d_pos_z_, d_vel_x_, d_vel_y_, d_vel_z_, d_acc_x_,
      d_acc_y_, d_acc_z_, num_particles_, half_dt, dt);

  // Compute new forces
  compute_forces_direct();

  // Final kick
  integrate_kick_kernel<<<grid_size, block_size_>>>(
      d_vel_x_, d_vel_y_, d_vel_z_, d_acc_x_, d_acc_y_, d_acc_z_,
      num_particles_, half_dt);

  CUDA_CHECK(cudaDeviceSynchronize());
}

void NBodyCUDA::step(double dt) { integrate(dt); }

void NBodyCUDA::synchronize(std::vector<Particle> &particles) {
  copy_from_device(particles);
}

double NBodyCUDA::compute_total_energy(std::vector<Particle> &particles) {
  synchronize(particles);

  int n = static_cast<int>(particles.size());
  double KE = 0.0, PE = 0.0;

  for (int i = 0; i < n; ++i) {
    KE += particles[i].kinetic_energy();

    for (int j = i + 1; j < n; ++j) {
      Vector3D r = particles[j].position - particles[i].position;
      double dist = sqrt(r.magnitude_sq() + softening_sq_);
      PE -= G_ * particles[i].mass * particles[j].mass / dist;
    }
  }

  return KE + PE;
}
