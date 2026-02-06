#ifndef NBODY_MPI_H
#define NBODY_MPI_H

#include "../common/config.h"
#include "../common/particle.h"
#include "../serial/octree.h"
#include "domain.h"
#include <mpi.h>
#include <vector>

// MPI-distributed N-Body solver
class NBodyMPI {
public:
  NBodyMPI(double G = Constants::G_NORMALIZED, double theta = BarnesHut::THETA,
           double softening = Constants::SOFTENING,
           MPI_Comm comm = MPI_COMM_WORLD);
  ~NBodyMPI() = default;

  // Initialize with particles (only rank 0 should have all particles)
  void initialize(const std::vector<Particle> &all_particles,
                  double domain_size);

  // Build local tree with halo particles
  void build_local_tree();

  // Compute forces on local particles
  void compute_forces();

  // Integrate and redistribute
  void integrate(double dt);

  // Full simulation step
  void step(double dt);

  // Gather all particles to rank 0
  std::vector<Particle> gather_all();

  // Compute total energy (requires gather)
  double compute_total_energy();

  // Get local particles
  const std::vector<Particle> &get_local_particles() const {
    return local_particles_;
  }
  std::vector<Particle> &get_local_particles() { return local_particles_; }

  // Statistics
  long long get_interaction_count() const { return interaction_count_; }
  double get_communication_time() const {
    return domain_.get_communication_time();
  }
  int get_rank() const { return domain_.get_rank(); }
  int get_size() const { return domain_.get_size(); }

  // Set parameters
  void set_theta(double theta) {
    theta_ = theta;
    theta_sq_ = theta * theta;
  }
  void set_use_barnes_hut(bool use_bh) { use_barnes_hut_ = use_bh; }

private:
  double G_;
  double theta_;
  double theta_sq_;
  double softening_;
  double softening_sq_;
  bool use_barnes_hut_;

  DomainDecomposition domain_;
  std::vector<Particle> local_particles_;
  std::vector<Particle> halo_particles_;
  Octree local_tree_;

  long long interaction_count_;
  double halo_width_;

  MPI_Comm comm_;

  // Force computation helpers
  void compute_force_direct();
  void compute_force_barnes_hut();
  void compute_force_on_particle(Particle &p, const OctreeNode *node,
                                 long long &count);
};

#endif // NBODY_MPI_H
