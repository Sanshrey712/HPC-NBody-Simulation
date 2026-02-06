#include "direct_omp.h"
#include <cmath>
#include <omp.h>

DirectNBodyOMP::DirectNBodyOMP(double G, double softening)
    : G_(G), softening_(softening), softening_sq_(softening * softening),
      interaction_count_(0), num_threads_(omp_get_max_threads()) {}

void DirectNBodyOMP::set_num_threads(int n) {
  num_threads_ = n;
  omp_set_num_threads(n);
}

void DirectNBodyOMP::compute_forces(std::vector<Particle> &particles) {
  int n = static_cast<int>(particles.size());
  interaction_count_ = 0;

// Reset accelerations in parallel
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].reset_acceleration();
  }

  // Parallel force computation
  // Using dynamic scheduling for load balancing (triangular loop)
  long long local_interactions = 0;

#pragma omp parallel reduction(+ : local_interactions)
  {
    // Thread-local acceleration arrays to avoid race conditions
    std::vector<Vector3D> local_acc(n);

#pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        // Distance vector from i to j
        Vector3D r = particles[j].position - particles[i].position;
        double dist_sq = r.magnitude_sq() + softening_sq_;
        double dist = sqrt(dist_sq);
        double dist_cubed = dist_sq * dist;

        double force_factor = G_ / dist_cubed;

        Vector3D acc_i = r * (force_factor * particles[j].mass);
        Vector3D acc_j = r * (force_factor * particles[i].mass);

        local_acc[i] += acc_i;
        local_acc[j] -= acc_j; // Newton's third law

        local_interactions++;
      }
    }

// Reduce local accelerations to global
#pragma omp critical
    {
      for (int i = 0; i < n; ++i) {
        particles[i].acceleration += local_acc[i];
      }
    }
  }

  interaction_count_ = local_interactions;
}

void DirectNBodyOMP::integrate(std::vector<Particle> &particles, double dt) {
  int n = static_cast<int>(particles.size());
  double half_dt = 0.5 * dt;

// Kick: update velocity by half step (parallel)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].velocity += particles[i].acceleration * half_dt;
    particles[i].position += particles[i].velocity * dt;
  }

  // Compute new forces at updated positions
  compute_forces(particles);

// Kick: update velocity by half step (parallel)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    particles[i].velocity += particles[i].acceleration * half_dt;
  }
}

void DirectNBodyOMP::step(std::vector<Particle> &particles, double dt) {
  integrate(particles, dt);
}

double DirectNBodyOMP::compute_kinetic_energy(
    const std::vector<Particle> &particles) const {
  double KE = 0.0;
  int n = static_cast<int>(particles.size());

#pragma omp parallel for reduction(+ : KE) schedule(static)
  for (int i = 0; i < n; ++i) {
    KE += particles[i].kinetic_energy();
  }

  return KE;
}

double DirectNBodyOMP::compute_potential_energy(
    const std::vector<Particle> &particles) const {
  double PE = 0.0;
  int n = static_cast<int>(particles.size());

#pragma omp parallel for reduction(+ : PE) schedule(dynamic, 64)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      Vector3D r = particles[j].position - particles[i].position;
      double dist = sqrt(r.magnitude_sq() + softening_sq_);
      PE -= G_ * particles[i].mass * particles[j].mass / dist;
    }
  }

  return PE;
}

double DirectNBodyOMP::compute_total_energy(
    const std::vector<Particle> &particles) const {
  return compute_kinetic_energy(particles) +
         compute_potential_energy(particles);
}
