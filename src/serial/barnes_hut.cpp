#include "barnes_hut.h"
#include <cmath>
#include <functional>

BarnesHutSolver::BarnesHutSolver(double G, double theta, double softening)
    : G_(G), theta_(theta), theta_sq_(theta * theta), softening_(softening),
      softening_sq_(softening * softening), interaction_count_(0) {}

void BarnesHutSolver::build_tree(const std::vector<Particle> &particles) {
  tree_.build(particles);
}

void BarnesHutSolver::compute_force_on_particle(Particle &particle,
                                                const OctreeNode *node) {
  if (!node || node->get_type() == OctreeNode::Type::EMPTY) {
    return;
  }

  if (node->get_type() == OctreeNode::Type::LEAF) {
    // Leaf node - compute direct force if not same particle
    const Particle *other = node->get_particle();
    if (other->id == particle.id) {
      return; // Skip self-interaction
    }

    Vector3D r = other->position - particle.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * other->mass / dist_cubed;
    particle.acceleration += r * force_factor;

    interaction_count_++;
    return;
  }

  // Internal node - check MAC criterion
  if (node->can_approximate(particle.position, theta_)) {
    // Use center of mass approximation
    Vector3D r = node->get_center_of_mass() - particle.position;
    double dist_sq = r.magnitude_sq() + softening_sq_;
    double dist = sqrt(dist_sq);
    double dist_cubed = dist_sq * dist;

    double force_factor = G_ * node->get_total_mass() / dist_cubed;
    particle.acceleration += r * force_factor;

    interaction_count_++;
  } else {
    // Recurse into children
    for (int i = 0; i < 8; ++i) {
      compute_force_on_particle(particle, node->get_child(i));
    }
  }
}

void BarnesHutSolver::compute_forces(std::vector<Particle> &particles) {
  interaction_count_ = 0;

  // Reset accelerations
  for (auto &p : particles) {
    p.reset_acceleration();
  }

  // Compute force on each particle using tree traversal
  for (auto &p : particles) {
    compute_force_on_particle(p, tree_.get_root());
  }
}

void BarnesHutSolver::integrate(std::vector<Particle> &particles, double dt) {
  // Leapfrog integration (Kick-Drift-Kick)
  double half_dt = 0.5 * dt;

  for (auto &p : particles) {
    // Kick: update velocity by half step
    p.velocity += p.acceleration * half_dt;

    // Drift: update position by full step
    p.position += p.velocity * dt;
  }

  // Rebuild tree at new positions
  build_tree(particles);

  // Compute new forces
  compute_forces(particles);

  for (auto &p : particles) {
    // Kick: update velocity by half step
    p.velocity += p.acceleration * half_dt;
  }
}

void BarnesHutSolver::step(std::vector<Particle> &particles, double dt) {
  integrate(particles, dt);
}

double BarnesHutSolver::compute_kinetic_energy(
    const std::vector<Particle> &particles) const {
  double KE = 0.0;
  for (const auto &p : particles) {
    KE += p.kinetic_energy();
  }
  return KE;
}

double BarnesHutSolver::compute_potential_energy(
    const std::vector<Particle> &particles) {
  // Use tree-based approximation for potential energy
  double PE = 0.0;

  // Build tree if not already built
  if (!tree_.get_root()) {
    tree_.build(particles);
  }

  // For each particle, traverse tree (similar to force computation)
  for (const auto &p : particles) {
    std::function<double(const OctreeNode *)> calc_pe =
        [&](const OctreeNode *node) -> double {
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
        sum += calc_pe(node->get_child(i));
      }
      return sum;
    };

    PE += calc_pe(tree_.get_root());
  }

  // Divide by 2 to correct for double counting
  return PE * 0.5;
}

double
BarnesHutSolver::compute_total_energy(const std::vector<Particle> &particles) {
  return compute_kinetic_energy(particles) +
         compute_potential_energy(particles);
}
