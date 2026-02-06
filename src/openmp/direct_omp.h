#ifndef DIRECT_OMP_H
#define DIRECT_OMP_H

#include "../common/config.h"
#include "../common/particle.h"
#include <vector>

// OpenMP-parallel direct O(N²) N-Body force computation
class DirectNBodyOMP {
public:
  DirectNBodyOMP(double G = Constants::G_NORMALIZED,
                 double softening = Constants::SOFTENING);

  // Compute all pairwise forces with OpenMP parallelization
  void compute_forces(std::vector<Particle> &particles);

  // Single timestep using Leapfrog integration (parallel)
  void integrate(std::vector<Particle> &particles, double dt);

  // Full simulation step
  void step(std::vector<Particle> &particles, double dt);

  // Compute total energy (kinetic + potential) - parallel
  double compute_total_energy(const std::vector<Particle> &particles) const;

  // Compute kinetic energy (parallel)
  double compute_kinetic_energy(const std::vector<Particle> &particles) const;

  // Compute potential energy (parallel)
  double compute_potential_energy(const std::vector<Particle> &particles) const;

  // Get interaction count
  long long get_interaction_count() const { return interaction_count_; }

  // Set number of threads
  void set_num_threads(int n);
  int get_num_threads() const { return num_threads_; }

private:
  double G_;
  double softening_;
  double softening_sq_;
  long long interaction_count_;
  int num_threads_;
};

#endif // DIRECT_OMP_H
