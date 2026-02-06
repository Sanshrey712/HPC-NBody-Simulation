#include "direct_nbody.h"
#include <cmath>

DirectNBody::DirectNBody(double G, double softening)
    : G_(G), softening_(softening), softening_sq_(softening * softening),
      interaction_count_(0) {}

void DirectNBody::compute_forces(std::vector<Particle> &particles) {
  int n = particles.size();
  interaction_count_ = 0;

  // Reset accelerations
  for (int i = 0; i < n; ++i) {
    particles[i].reset_acceleration();
  }

  // Compute pairwise forces (use Newton's third law for efficiency)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      // Distance vector from i to j
      Vector3D r = particles[j].position - particles[i].position;
      double dist_sq = r.magnitude_sq() + softening_sq_;
      double dist = sqrt(dist_sq);
      double dist_cubed = dist_sq * dist;

      // Force magnitude: G * m_i * m_j / r²
      // Acceleration on i: G * m_j / r² in direction of r
      double force_factor = G_ / dist_cubed;

      Vector3D acc_i = r * (force_factor * particles[j].mass);
      Vector3D acc_j = r * (force_factor * particles[i].mass);

      particles[i].acceleration += acc_i;
      particles[j].acceleration -= acc_j; // Newton's third law

      interaction_count_++;
    }
  }
}

void DirectNBody::integrate(std::vector<Particle> &particles, double dt) {
  // Leapfrog integration (Kick-Drift-Kick variant)
  // This is symplectic and preserves energy well

  double half_dt = 0.5 * dt;

  for (auto &p : particles) {
    // Kick: update velocity by half step
    p.velocity += p.acceleration * half_dt;

    // Drift: update position by full step
    p.position += p.velocity * dt;
  }

  // Compute new forces at updated positions
  compute_forces(particles);

  for (auto &p : particles) {
    // Kick: update velocity by half step
    p.velocity += p.acceleration * half_dt;
  }
}

void DirectNBody::step(std::vector<Particle> &particles, double dt) {
  integrate(particles, dt);
}

double DirectNBody::compute_kinetic_energy(
    const std::vector<Particle> &particles) const {
  double KE = 0.0;
  for (const auto &p : particles) {
    KE += p.kinetic_energy();
  }
  return KE;
}

double DirectNBody::compute_potential_energy(
    const std::vector<Particle> &particles) const {
  double PE = 0.0;
  int n = particles.size();

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      Vector3D r = particles[j].position - particles[i].position;
      double dist = sqrt(r.magnitude_sq() + softening_sq_);
      PE -= G_ * particles[i].mass * particles[j].mass / dist;
    }
  }

  return PE;
}

double DirectNBody::compute_total_energy(
    const std::vector<Particle> &particles) const {
  return compute_kinetic_energy(particles) +
         compute_potential_energy(particles);
}

Vector3D DirectNBody::compute_center_of_mass(
    const std::vector<Particle> &particles) const {
  Vector3D com;
  double total_mass = 0.0;

  for (const auto &p : particles) {
    com += p.position * p.mass;
    total_mass += p.mass;
  }

  if (total_mass > 0.0) {
    com /= total_mass;
  }

  return com;
}

Vector3D DirectNBody::compute_total_momentum(
    const std::vector<Particle> &particles) const {
  Vector3D momentum;

  for (const auto &p : particles) {
    momentum += p.velocity * p.mass;
  }

  return momentum;
}
