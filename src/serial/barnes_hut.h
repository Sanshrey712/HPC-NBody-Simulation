#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include "../common/config.h"
#include "../common/particle.h"
#include "octree.h"
#include <vector>

// Barnes-Hut O(N log N) N-Body solver (serial)
class BarnesHutSolver {
public:
  BarnesHutSolver(double G = Constants::G_NORMALIZED,
                  double theta = BarnesHut::THETA,
                  double softening = Constants::SOFTENING);

  // Build octree from particles
  void build_tree(const std::vector<Particle> &particles);

  // Compute forces using Barnes-Hut algorithm
  void compute_forces(std::vector<Particle> &particles);

  // Single timestep using Leapfrog integration
  void integrate(std::vector<Particle> &particles, double dt);

  // Full simulation step
  void step(std::vector<Particle> &particles, double dt);

  // Compute total energy
  double compute_total_energy(const std::vector<Particle> &particles);

  // Compute kinetic energy
  double compute_kinetic_energy(const std::vector<Particle> &particles) const;

  // Compute potential energy (approximate using tree)
  double compute_potential_energy(const std::vector<Particle> &particles);

  // Get tree statistics
  int get_node_count() const { return tree_.get_node_count(); }
  int get_max_depth() const { return tree_.get_max_depth(); }
  long long get_interaction_count() const { return interaction_count_; }

  // Set parameters
  void set_theta(double theta) {
    theta_ = theta;
    theta_sq_ = theta * theta;
  }

private:
  double G_;
  double theta_;
  double theta_sq_;
  double softening_;
  double softening_sq_;
  Octree tree_;
  long long interaction_count_;

  // Recursive force computation on a single particle
  void compute_force_on_particle(Particle &particle, const OctreeNode *node);
};

#endif // BARNES_HUT_H
