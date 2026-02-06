// Serial N-Body Simulation with Profiling
// Usage: ./nbody_serial --particles 10000 --steps 100

#include "common/config.h"
#include "common/particle.h"
#include "serial/barnes_hut.h"
#include "serial/direct_nbody.h"
#include "serial/generators.h"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  SimConfig config;
  config.parse_args(argc, argv);
  config.mode = SimConfig::Mode::SERIAL;

  std::cout << "\n========================================\n";
  std::cout << "       SERIAL N-BODY SIMULATION\n";
  std::cout << "========================================\n";
  config.print();

  // Generate particles
  std::cout << "Generating " << config.num_particles
            << " particles (Plummer sphere)...\n";
  auto particles = generate_plummer(
      config.num_particles, config.domain_size / 4.0, config.total_mass, 42);

  Timer total_timer, force_timer, tree_timer;
  PerfStats stats;

  double initial_energy = 0.0, final_energy = 0.0;

  total_timer.start();

  if (config.use_barnes_hut) {
    std::cout << "\nUsing Barnes-Hut algorithm (O(N log N), theta="
              << config.theta << ")\n";
    BarnesHutSolver solver(Constants::G_NORMALIZED, config.theta,
                           config.softening);

    // Initial tree build and force computation
    tree_timer.start();
    solver.build_tree(particles);
    tree_timer.stop();
    stats.tree_build_time = tree_timer.elapsed_ms();

    force_timer.start();
    solver.compute_forces(particles);
    force_timer.stop();

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if ((step + 1) % 10 == 0 || step == config.num_steps - 1) {
        std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                  << " | Interactions: " << solver.get_interaction_count()
                  << std::flush;
      }
    }

    stats.num_interactions = solver.get_interaction_count();
    stats.force_compute_time = force_timer.elapsed_ms();

    if (config.enable_energy_check) {
      final_energy = solver.compute_total_energy(particles);
    }

    std::cout << "\n\nTree nodes: " << solver.get_node_count()
              << " | Max depth: " << solver.get_max_depth() << "\n";

  } else {
    std::cout << "\nUsing Direct method (O(N²))\n";
    DirectNBody solver(Constants::G_NORMALIZED, config.softening);

    force_timer.start();
    solver.compute_forces(particles);
    force_timer.stop();

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if ((step + 1) % 10 == 0 || step == config.num_steps - 1) {
        std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                  << std::flush;
      }
    }

    stats.num_interactions = solver.get_interaction_count();
    stats.force_compute_time = force_timer.elapsed_ms();

    if (config.enable_energy_check) {
      final_energy = solver.compute_total_energy(particles);
    }
  }

  total_timer.stop();
  stats.total_time = total_timer.elapsed_ms();

  // Print results
  std::cout << "\n\n========================================\n";
  std::cout << "           PERFORMANCE RESULTS\n";
  std::cout << "========================================\n";
  std::cout << "Total time:         " << stats.total_time << " ms\n";
  std::cout << "Tree build time:    " << stats.tree_build_time << " ms\n";
  std::cout << "Force compute time: " << stats.force_compute_time << " ms\n";
  std::cout << "Interactions/step:  " << stats.num_interactions << "\n";
  std::cout << "Time/step:          " << stats.total_time / config.num_steps
            << " ms\n";

  if (config.enable_energy_check) {
    double drift = std::abs(final_energy - initial_energy) /
                   std::abs(initial_energy) * 100;
    std::cout << "\nEnergy conservation:\n";
    std::cout << "  Initial: " << initial_energy << "\n";
    std::cout << "  Final:   " << final_energy << "\n";
    std::cout << "  Drift:   " << drift << "%\n";
  }

  std::cout << "========================================\n\n";

  return 0;
}
