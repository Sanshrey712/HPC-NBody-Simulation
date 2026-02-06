#include "barnes_hut_omp.h"
#include <cmath>
#include <omp.h>
#include <functional>

BarnesHutOMP::BarnesHutOMP(double G, double theta, double softening)
    : G_(G), theta_(theta), theta_sq_(theta * theta), softening_(softening),
      softening_sq_(softening * softening), interaction_count_(0),
      num_threads_(omp_get_max_threads()) {}

void BarnesHutOMP::set_num_threads(int n) {
  num_threads_ = n;
  omp_set_num_threads(n);
}

void BarnesHutOMP::build_tree(const std::vector<Particle> &particles) {
  // Tree construction is serial (parallelizing octree build is complex)
  // The tree is read-only during force computation, so this is safe
  tree_.build(particles);
}

void BarnesHutOMP::compute_force_on_particle(Particle &particle,
                                             const OctreeNode *node,
                                             long long &local_count) {
  if (!node || node->get_type() == OctreeNode::Type::EMPTY) {
    return;
  }

  if (node->get_type() == OctreeNode::Type::LEAF) {
    const Particle *other = node->get_particle();
    if (other->id == particle.id) {
      return;
    }

    Vector3D r = other->position - particle.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * other->mass / dist_cubed;
    particle.acceleration += r * force_factor;

    local_count++;
    return;
  }

  // Internal node - check MAC criterion
  if (node->can_approximate(particle.position, theta_)) {
    Vector3D r = node->get_center_of_mass() - particle.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * node->get_total_mass() / dist_cubed;
    particle.acceleration += r * force_factor;

    local_count++;
  } else {
    for (int i = 0; i < 8; ++i) {
      compute_force_on_particle(particle, node->get_child(i), local_count);
    }
  }
}

void BarnesHutOMP::compute_forces(std::vector<Particle> &particles) {
  int n = static_cast<int>(particles.size());
  long long total_interactions = 0;

// Reset accelerations in parallel
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].reset_acceleration();
  }

// Parallel force computation - each particle traverses tree independently
// Tree is read-only, so this is safe
#pragma omp parallel reduction(+ : total_interactions)
  {
    long long local_count = 0;

#pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < n; ++i) {
      compute_force_on_particle(particles[i], tree_.get_root(), local_count);
    }

    total_interactions += local_count;
  }

  interaction_count_ = total_interactions;
}

void BarnesHutOMP::integrate(std::vector<Particle> &particles, double dt) {
  int n = static_cast<int>(particles.size());
  double half_dt = 0.5 * dt;

// Kick-Drift (parallel)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].velocity += particles[i].acceleration * half_dt;
    particles[i].position += particles[i].velocity * dt;
  }

  // Rebuild tree at new positions
  build_tree(particles);

  // Compute new forces
  compute_forces(particles);

// Final kick (parallel)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].velocity += particles[i].acceleration * half_dt;
  }
}

void BarnesHutOMP::step(std::vector<Particle> &particles, double dt) {
  integrate(particles, dt);
}

double BarnesHutOMP::compute_kinetic_energy(
    const std::vector<Particle> &particles) const {
  double KE = 0.0;
  int n = static_cast<int>(particles.size());

#pragma omp parallel for reduction(+ : KE) schedule(static)
  for (int i = 0; i < n; ++i) {
    KE += particles[i].kinetic_energy();
  }

  return KE;
}

double BarnesHutOMP::compute_total_energy(std::vector<Particle> &particles) {
  // Approximate potential energy using tree (parallel over particles)
  double PE = 0.0;
  int n = static_cast<int>(particles.size());

  if (!tree_.get_root()) {
    build_tree(particles);
  }

#pragma omp parallel reduction(+ : PE)
  {
    std::function<double(const Particle &, const OctreeNode *)> calc_pe;
    calc_pe = [&](const Particle &p, const OctreeNode *node) -> double {
      if (!node || node->get_type() == OctreeNode::Type::EMPTY) {
        return 0.0;
      }

      if (node->get_type() == OctreeNode::Type::LEAF) {
        const Particle *other = node->get_particle();
        if (other->id == p.id)
          return 0.0;

        Vector3D r = other->position - p.position;
        double dist = sqrt(r.magnitude_sq() + softening_sq_);
        return -G_ * p.mass * other->mass / dist;
      }

      if (node->can_approximate(p.position, theta_)) {
        Vector3D r = node->get_center_of_mass() - p.position;
        double dist = sqrt(r.magnitude_sq() + softening_sq_);
        return -G_ * p.mass * node->get_total_mass() / dist;
      }

      double sum = 0.0;
      for (int i = 0; i < 8; ++i) {
        sum += calc_pe(p, node->get_child(i));
      }
      return sum;
    };

#pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < n; ++i) {
      PE += calc_pe(particles[i], tree_.get_root());
    }
  }

  return compute_kinetic_energy(particles) + PE * 0.5;
}
