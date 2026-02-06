// OpenMP Parallel N-Body Simulation
// Usage: ./nbody_openmp --particles 100000 --steps 100 --threads 8

#include "common/config.h"
#include "common/particle.h"
#include "openmp/barnes_hut_omp.h"
#include "openmp/direct_omp.h"
#include "serial/generators.h"
#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {
  SimConfig config;
  config.parse_args(argc, argv);
  config.mode = SimConfig::Mode::OPENMP;

  if (config.num_threads <= 0) {
    config.num_threads = omp_get_max_threads();
  }
  omp_set_num_threads(config.num_threads);

  std::cout << "\n========================================\n";
  std::cout << "      OpenMP N-BODY SIMULATION\n";
  std::cout << "========================================\n";
  std::cout << "Threads: " << config.num_threads << "\n";
  config.print();

  // Generate particles
  std::cout << "Generating " << config.num_particles << " particles...\n";
  auto particles = generate_plummer(
      config.num_particles, config.domain_size / 4.0, config.total_mass, 42);

  Timer total_timer;
  PerfStats stats;
  double initial_energy = 0.0, final_energy = 0.0;

  total_timer.start();

  if (config.use_barnes_hut) {
    std::cout << "\nUsing Barnes-Hut + OpenMP (theta=" << config.theta << ")\n";
    BarnesHutOMP solver(Constants::G_NORMALIZED, config.theta,
                        config.softening);
    solver.set_num_threads(config.num_threads);

    solver.build_tree(particles);
    solver.compute_forces(particles);

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if ((step + 1) % 10 == 0) {
        std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                  << std::flush;
      }
    }

    stats.num_interactions = solver.get_interaction_count();

    if (config.enable_energy_check) {
      final_energy = solver.compute_total_energy(particles);
    }

  } else {
    std::cout << "\nUsing Direct O(N²) + OpenMP\n";
    DirectNBodyOMP solver(Constants::G_NORMALIZED, config.softening);
    solver.set_num_threads(config.num_threads);

    solver.compute_forces(particles);

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if ((step + 1) % 10 == 0) {
        std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                  << std::flush;
      }
    }

    stats.num_interactions = solver.get_interaction_count();

    if (config.enable_energy_check) {
      final_energy = solver.compute_total_energy(particles);
    }
  }

  total_timer.stop();
  stats.total_time = total_timer.elapsed_ms();

  // Print results
  std::cout << "\n\n========================================\n";
  std::cout << "      OpenMP PERFORMANCE RESULTS\n";
  std::cout << "========================================\n";
  std::cout << "Threads:            " << config.num_threads << "\n";
  std::cout << "Total time:         " << stats.total_time << " ms\n";
  std::cout << "Interactions/step:  " << stats.num_interactions << "\n";
  std::cout << "Time/step:          " << stats.total_time / config.num_steps
            << " ms\n";

  if (config.enable_energy_check) {
    double drift = std::abs(final_energy - initial_energy) /
                   std::abs(initial_energy) * 100;
    std::cout << "\nEnergy drift: " << drift << "%\n";
  }

  std::cout << "========================================\n\n";

  return 0;
}
