#ifndef DIRECT_NBODY_H
#define DIRECT_NBODY_H

#include "../common/config.h"
#include "../common/particle.h"
#include <vector>

// Direct O(N²) N-Body force computation (serial)
class DirectNBody {
public:
  DirectNBody(double G = Constants::G_NORMALIZED,
              double softening = Constants::SOFTENING);

  // Compute all pairwise forces and update accelerations
  void compute_forces(std::vector<Particle> &particles);

  // Single timestep using Leapfrog integration
  void integrate(std::vector<Particle> &particles, double dt);

  // Full simulation step
  void step(std::vector<Particle> &particles, double dt);

  // Compute total energy (kinetic + potential)
  double compute_total_energy(const std::vector<Particle> &particles) const;

  // Compute total kinetic energy
  double compute_kinetic_energy(const std::vector<Particle> &particles) const;

  // Compute total potential energy
  double compute_potential_energy(const std::vector<Particle> &particles) const;

  // Compute center of mass
  Vector3D compute_center_of_mass(const std::vector<Particle> &particles) const;

  // Compute total momentum
  Vector3D compute_total_momentum(const std::vector<Particle> &particles) const;

  // Get interaction count for last force computation
  long long get_interaction_count() const { return interaction_count_; }

private:
  double G_;         // Gravitational constant
  double softening_; // Softening parameter
  double softening_sq_;
  long long interaction_count_;
};

#endif // DIRECT_NBODY_H
