#ifndef NBODY_CUDA_H
#define NBODY_CUDA_H

#include "../common/config.h"
#include "../common/particle.h"
#include <vector>

// CUDA-accelerated N-Body solver
class NBodyCUDA {
public:
  NBodyCUDA(double G = Constants::G_NORMALIZED,
            double softening = Constants::SOFTENING);
  ~NBodyCUDA();

  // Initialize with particles (copies to GPU)
  void initialize(const std::vector<Particle> &particles);

  // Compute forces using direct O(N²) method on GPU
  void compute_forces_direct();

  // Integration on GPU
  void integrate(double dt);

  // Full simulation step
  void step(double dt);

  // Copy results back to host
  void synchronize(std::vector<Particle> &particles);

  // Compute total energy (requires sync)
  double compute_total_energy(std::vector<Particle> &particles);

  // Get statistics
  long long get_interaction_count() const { return interaction_count_; }
  double get_kernel_time() const { return kernel_time_; }

  // Set parameters
  void set_block_size(int size) { block_size_ = size; }
  int get_block_size() const { return block_size_; }

  // Check CUDA availability
  static bool is_available();
  static void print_device_info();

private:
  double G_;
  double softening_;
  double softening_sq_;
  int num_particles_;
  int block_size_;
  long long interaction_count_;
  double kernel_time_;

  // Device memory pointers
  double *d_pos_x_;
  double *d_pos_y_;
  double *d_pos_z_;
  double *d_vel_x_;
  double *d_vel_y_;
  double *d_vel_z_;
  double *d_acc_x_;
  double *d_acc_y_;
  double *d_acc_z_;
  double *d_mass_;

  // Allocate device memory
  void allocate_device_memory(int n);

  // Free device memory
  void free_device_memory();

  // Copy to device
  void copy_to_device(const std::vector<Particle> &particles);

  // Copy from device
  void copy_from_device(std::vector<Particle> &particles);
};

#endif // NBODY_CUDA_H
