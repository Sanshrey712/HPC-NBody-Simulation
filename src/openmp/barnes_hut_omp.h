#ifndef BARNES_HUT_OMP_H
#define BARNES_HUT_OMP_H

#include "../common/config.h"
#include "../common/particle.h"
#include "../serial/octree.h"
#include <vector>

// OpenMP-parallel Barnes-Hut O(N log N) N-Body solver
class BarnesHutOMP {
public:
  BarnesHutOMP(double G = Constants::G_NORMALIZED,
               double theta = BarnesHut::THETA,
               double softening = Constants::SOFTENING);

  // Build octree (serial - tree construction is harder to parallelize)
  void build_tree(const std::vector<Particle> &particles);

  // Compute forces using Barnes-Hut with OpenMP
  void compute_forces(std::vector<Particle> &particles);

  // Single timestep using Leapfrog integration (parallel)
  void integrate(std::vector<Particle> &particles, double dt);

  // Full simulation step
  void step(std::vector<Particle> &particles, double dt);

  // Compute total energy
  double compute_total_energy(std::vector<Particle> &particles);

  // Compute kinetic energy (parallel)
  double compute_kinetic_energy(const std::vector<Particle> &particles) const;

  // Get statistics
  int get_node_count() const { return tree_.get_node_count(); }
  int get_max_depth() const { return tree_.get_max_depth(); }
  long long get_interaction_count() const { return interaction_count_; }

  // Set parameters
  void set_theta(double theta) {
    theta_ = theta;
    theta_sq_ = theta * theta;
  }
  void set_num_threads(int n);
  int get_num_threads() const { return num_threads_; }

private:
  double G_;
  double theta_;
  double theta_sq_;
  double softening_;
  double softening_sq_;
  Octree tree_;
  long long interaction_count_;
  int num_threads_;

  // Recursive force computation (called per particle, thread-safe for reading
  // tree)
  void compute_force_on_particle(Particle &particle, const OctreeNode *node,
                                 long long &local_count);
};

#endif // BARNES_HUT_OMP_H
