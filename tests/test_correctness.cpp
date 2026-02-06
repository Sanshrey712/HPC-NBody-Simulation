#include <cmath>
#include <iostream>
#include <vector>

#include "../src/common/config.h"
#include "../src/common/particle.h"
#include "../src/serial/barnes_hut.h"
#include "../src/serial/direct_nbody.h"
#include "../src/serial/generators.h"

// Test tolerance
const double TOLERANCE = 1e-6;

bool test_vector3d() {
  std::cout << "Testing Vector3D... ";

  Vector3D a(1.0, 2.0, 3.0);
  Vector3D b(4.0, 5.0, 6.0);

  // Addition
  Vector3D c = a + b;
  if (std::abs(c.x - 5.0) > TOLERANCE || std::abs(c.y - 7.0) > TOLERANCE ||
      std::abs(c.z - 9.0) > TOLERANCE) {
    std::cout << "FAILED (addition)\n";
    return false;
  }

  // Magnitude
  double mag = a.magnitude();
  if (std::abs(mag - std::sqrt(14.0)) > TOLERANCE) {
    std::cout << "FAILED (magnitude)\n";
    return false;
  }

  // Dot product
  double dot = a.dot(b);
  if (std::abs(dot - 32.0) > TOLERANCE) {
    std::cout << "FAILED (dot product)\n";
    return false;
  }

  std::cout << "PASSED\n";
  return true;
}

bool test_particle_generator() {
  std::cout << "Testing particle generators... ";

  int n = 1000;

  // Test uniform
  auto uniform = generate_uniform(n, 100.0, 1000.0);
  if (uniform.size() != n) {
    std::cout << "FAILED (uniform count)\n";
    return false;
  }

  // Test Plummer
  auto plummer = generate_plummer(n, 10.0, 1000.0);
  if (plummer.size() != n) {
    std::cout << "FAILED (plummer count)\n";
    return false;
  }

  // Check mass conservation
  double total_mass = 0.0;
  for (const auto &p : plummer) {
    total_mass += p.mass;
  }
  if (std::abs(total_mass - 1000.0) > TOLERANCE * 1000.0) {
    std::cout << "FAILED (mass conservation)\n";
    return false;
  }

  std::cout << "PASSED\n";
  return true;
}

bool test_direct_nbody() {
  std::cout << "Testing direct N-body... ";

  // Two-body test: two equal masses
  std::vector<Particle> particles(2);
  particles[0] =
      Particle(Vector3D(-1.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0), 1.0, 0);
  particles[1] =
      Particle(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0), 1.0, 1);

  DirectNBody solver(1.0, 0.01); // G=1, small softening
  solver.compute_forces(particles);

  // Forces should be equal and opposite
  double force_mag_0 = particles[0].acceleration.magnitude();
  double force_mag_1 = particles[1].acceleration.magnitude();

  if (std::abs(force_mag_0 - force_mag_1) > TOLERANCE) {
    std::cout << "FAILED (force symmetry)\n";
    return false;
  }

  // Force should point toward each other
  if (particles[0].acceleration.x <= 0 || particles[1].acceleration.x >= 0) {
    std::cout << "FAILED (force direction)\n";
    return false;
  }

  std::cout << "PASSED\n";
  return true;
}

bool test_energy_conservation() {
  std::cout << "Testing energy conservation... ";

  auto particles = generate_plummer(100, 5.0, 100.0, 42);

  DirectNBody solver(1.0, 0.1);
  solver.compute_forces(particles);

  double initial_energy = solver.compute_total_energy(particles);

  // Run 100 steps
  for (int i = 0; i < 100; ++i) {
    solver.step(particles, 0.001);
  }

  double final_energy = solver.compute_total_energy(particles);
  double drift =
      std::abs(final_energy - initial_energy) / std::abs(initial_energy);

  // Energy should be conserved to within 1%
  if (drift > 0.01) {
    std::cout << "FAILED (drift: " << drift * 100 << "%)\n";
    return false;
  }

  std::cout << "PASSED (drift: " << drift * 100 << "%)\n";
  return true;
}

bool test_barnes_hut() {
  std::cout << "Testing Barnes-Hut... ";

  auto particles = generate_plummer(500, 10.0, 100.0, 42);

  // Compute forces with direct method
  auto direct_particles = particles;
  DirectNBody direct_solver(1.0, 0.1);
  direct_solver.compute_forces(direct_particles);

  // Compute forces with Barnes-Hut
  auto bh_particles = particles;
  BarnesHutSolver bh_solver(1.0, 0.5, 0.1); // theta=0.5
  bh_solver.build_tree(bh_particles);
  bh_solver.compute_forces(bh_particles);

  // Compare forces (should be similar but not identical)
  double max_error = 0.0;
  for (size_t i = 0; i < particles.size(); ++i) {
    Vector3D diff =
        direct_particles[i].acceleration - bh_particles[i].acceleration;
    double error = diff.magnitude() /
                   (direct_particles[i].acceleration.magnitude() + 1e-10);
    max_error = std::max(max_error, error);
  }

  // With theta=0.5, errors should be < 10%
  if (max_error > 0.1) {
    std::cout << "FAILED (max error: " << max_error * 100 << "%)\n";
    return false;
  }

  std::cout << "PASSED (max error: " << max_error * 100 << "%)\n";
  return true;
}

bool test_barnes_hut_energy() {
  std::cout << "Testing Barnes-Hut energy conservation... ";

  auto particles = generate_plummer(200, 5.0, 100.0, 42);

  BarnesHutSolver solver(1.0, 0.5, 0.1);
  solver.build_tree(particles);
  solver.compute_forces(particles);

  double initial_energy = solver.compute_total_energy(particles);

  // Run 100 steps
  for (int i = 0; i < 100; ++i) {
    solver.step(particles, 0.001);
  }

  double final_energy = solver.compute_total_energy(particles);
  double drift =
      std::abs(final_energy - initial_energy) / std::abs(initial_energy);

  // Allow slightly more drift due to approximations
  if (drift > 0.02) {
    std::cout << "FAILED (drift: " << drift * 100 << "%)\n";
    return false;
  }

  std::cout << "PASSED (drift: " << drift * 100 << "%)\n";
  return true;
}

int main() {
  std::cout << "\n=== N-Body Simulation Correctness Tests ===\n\n";

  int passed = 0;
  int failed = 0;

  if (test_vector3d())
    passed++;
  else
    failed++;
  if (test_particle_generator())
    passed++;
  else
    failed++;
  if (test_direct_nbody())
    passed++;
  else
    failed++;
  if (test_energy_conservation())
    passed++;
  else
    failed++;
  if (test_barnes_hut())
    passed++;
  else
    failed++;
  if (test_barnes_hut_energy())
    passed++;
  else
    failed++;

  std::cout << "\n=== Results ===\n";
  std::cout << "Passed: " << passed << "/" << (passed + failed) << "\n";

  if (failed > 0) {
    std::cout << "SOME TESTS FAILED\n";
    return 1;
  }

  std::cout << "ALL TESTS PASSED\n";
  return 0;
}
