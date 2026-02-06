#ifndef DOMAIN_H
#define DOMAIN_H

#include "../common/config.h"
#include "../common/particle.h"
#include <mpi.h>
#include <vector>

// Domain decomposition for distributed N-Body simulation
class DomainDecomposition {
public:
  DomainDecomposition(MPI_Comm comm = MPI_COMM_WORLD);
  ~DomainDecomposition();

  // Initialize domain decomposition
  void initialize(int num_particles, double domain_size);

  // Distribute particles across ranks
  void distribute_particles(std::vector<Particle> &local_particles,
                            const std::vector<Particle> &all_particles);

  // Gather all particles to rank 0
  void gather_particles(std::vector<Particle> &all_particles,
                        const std::vector<Particle> &local_particles);

  // Exchange boundary particles with neighbors
  void exchange_halo(std::vector<Particle> &local_particles,
                     std::vector<Particle> &halo_particles, double halo_width);

  // Redistribute particles after position update (load balancing)
  void redistribute(std::vector<Particle> &local_particles);

  // Get local domain bounds
  void get_local_bounds(Vector3D &min, Vector3D &max) const;

  // Check if particle belongs to this rank
  bool owns_particle(const Vector3D &position) const;

  // Getters
  int get_rank() const { return rank_; }
  int get_size() const { return size_; }
  int get_local_count() const { return local_count_; }
  int get_global_count() const { return global_count_; }

  // MPI statistics
  double get_communication_time() const { return comm_time_; }
  void reset_communication_time() { comm_time_ = 0.0; }

private:
  MPI_Comm comm_;
  int rank_;
  int size_;
  int local_count_;
  int global_count_;

  // Domain bounds for this rank
  Vector3D local_min_;
  Vector3D local_max_;
  double domain_size_;

  // 3D Cartesian decomposition
  int dims_[3];
  int coords_[3];
  MPI_Comm cart_comm_;

  // Communication timing
  double comm_time_;

  // Determine which rank owns a position
  int get_owner_rank(const Vector3D &position) const;

  // Create derived MPI datatype for Particle
  MPI_Datatype particle_type_;
  void create_mpi_particle_type();
};

#endif // DOMAIN_H
