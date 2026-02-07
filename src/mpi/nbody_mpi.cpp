#include "nbody_mpi.h"
#include <algorithm>
#include <cmath>

NBodyMPI::NBodyMPI(double G, double theta, double softening, MPI_Comm comm)
    : G_(G), theta_(theta), theta_sq_(theta * theta), softening_(softening),
      softening_sq_(softening * softening), use_barnes_hut_(true),
      domain_(comm), interaction_count_(0), halo_width_(10.0), comm_(comm) {}

void NBodyMPI::initialize(const std::vector<Particle> &all_particles,
                          double domain_size) {
  int n = static_cast<int>(all_particles.size());
  domain_.initialize(n, domain_size);

  // Distribute particles
  domain_.distribute_particles(local_particles_, all_particles);

  // Set halo width based on domain size and theta
  halo_width_ = domain_size / 10.0;
}

void NBodyMPI::build_local_tree() {
  // Get halo particles from neighbors
  domain_.exchange_halo(local_particles_, halo_particles_, halo_width_);

  // Combine local and halo particles for tree building
  // Combine local and halo particles for tree building
  all_local_particles_.clear();
  all_local_particles_.reserve(local_particles_.size() +
                               halo_particles_.size());
  all_local_particles_.insert(all_local_particles_.end(),
                              local_particles_.begin(), local_particles_.end());
  all_local_particles_.insert(all_local_particles_.end(),
                              halo_particles_.begin(), halo_particles_.end());

  // Build tree
  if (!all_local_particles_.empty()) {
    local_tree_.build(all_local_particles_);
  }
}

void NBodyMPI::compute_force_on_particle(Particle &p, const OctreeNode *node,
                                         long long &count) {
  if (!node || node->get_type() == OctreeNode::Type::EMPTY) {
    return;
  }

  if (node->get_type() == OctreeNode::Type::LEAF) {
    const Particle *other = node->get_particle();
    if (other->id == p.id)
      return;

    Vector3D r = other->position - p.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * other->mass / dist_cubed;
    p.acceleration += r * force_factor;
    count++;
    return;
  }

  if (node->can_approximate(p.position, theta_)) {
    Vector3D r = node->get_center_of_mass() - p.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * node->get_total_mass() / dist_cubed;
    p.acceleration += r * force_factor;
    count++;
  } else {
    for (int i = 0; i < 8; ++i) {
      compute_force_on_particle(p, node->get_child(i), count);
    }
  }
}

void NBodyMPI::compute_force_direct() {
  // Combine local and halo for direct computation
  std::vector<Particle> all_local;
  all_local.reserve(local_particles_.size() + halo_particles_.size());
  all_local.insert(all_local.end(), local_particles_.begin(),
                   local_particles_.end());
  all_local.insert(all_local.end(), halo_particles_.begin(),
                   halo_particles_.end());

  int n_local = static_cast<int>(local_particles_.size());
  int n_all = static_cast<int>(all_local.size());

  interaction_count_ = 0;

  for (int i = 0; i < n_local; ++i) {
    local_particles_[i].reset_acceleration();

    for (int j = 0; j < n_all; ++j) {
      if (all_local[j].id == local_particles_[i].id)
        continue;

      Vector3D r = all_local[j].position - local_particles_[i].position;
      double dist_sq = r.magnitude_sq() + softening_sq_;
      double dist = sqrt(dist_sq);
      double dist_cubed = dist_sq * dist;

      double force_factor = G_ * all_local[j].mass / dist_cubed;
      local_particles_[i].acceleration += r * force_factor;
      interaction_count_++;
    }
  }
}

void NBodyMPI::compute_force_barnes_hut() {
  interaction_count_ = 0;

  for (auto &p : local_particles_) {
    p.reset_acceleration();
    long long count = 0;
    compute_force_on_particle(p, local_tree_.get_root(), count);
    interaction_count_ += count;
  }
}

void NBodyMPI::compute_forces() {
  if (use_barnes_hut_) {
    build_local_tree();
    compute_force_barnes_hut();
  } else {
    domain_.exchange_halo(local_particles_, halo_particles_, halo_width_);
    compute_force_direct();
  }
}

void NBodyMPI::integrate(double dt) {
  double half_dt = 0.5 * dt;

  // Kick-Drift
  for (auto &p : local_particles_) {
    p.velocity += p.acceleration * half_dt;
    p.position += p.velocity * dt;
  }

  // Redistribute particles that moved to other domains
  domain_.redistribute(local_particles_);

  // Compute new forces
  compute_forces();

  // Final kick
  for (auto &p : local_particles_) {
    p.velocity += p.acceleration * half_dt;
  }
}

void NBodyMPI::step(double dt) { integrate(dt); }

std::vector<Particle> NBodyMPI::gather_all() {
  std::vector<Particle> all_particles;
  domain_.gather_particles(all_particles, local_particles_);
  return all_particles;
}

double NBodyMPI::compute_total_energy() {
  // Compute local kinetic energy
  double local_KE = 0.0;
  for (const auto &p : local_particles_) {
    local_KE += p.kinetic_energy();
  }

  // Sum across all ranks
  double global_KE;
  MPI_Allreduce(&local_KE, &global_KE, 1, MPI_DOUBLE, MPI_SUM, comm_);

  // Potential energy requires gathering (expensive but accurate)
  auto all = gather_all();
  double PE = 0.0;

  if (domain_.get_rank() == 0) {
    int n = static_cast<int>(all.size());
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        Vector3D r = all[j].position - all[i].position;
        double dist = sqrt(r.magnitude_sq() + softening_sq_);
        PE -= G_ * all[i].mass * all[j].mass / dist;
      }
    }
  }

  MPI_Bcast(&PE, 1, MPI_DOUBLE, 0, comm_);

  return global_KE + PE;
}
