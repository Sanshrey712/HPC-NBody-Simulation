// CUDA GPU N-Body Simulation
// Usage: ./nbody_cuda --particles 1000000 --steps 100

#include "common/config.h"
#include "common/particle.h"
#include "cuda/nbody_cuda.h"
#include "serial/generators.h"
#include <iostream>

int main(int argc, char *argv[]) {
  SimConfig config;
  config.parse_args(argc, argv);
  config.mode = SimConfig::Mode::CUDA;

  std::cout << "\n========================================\n";
  std::cout << "       CUDA N-BODY SIMULATION\n";
  std::cout << "========================================\n";

  // Check CUDA availability
  if (!NBodyCUDA::is_available()) {
    std::cerr << "ERROR: No CUDA-capable device found!\n";
    return 1;
  }

  NBodyCUDA::print_device_info();
  config.print();

  // Generate particles
  std::cout << "Generating " << config.num_particles << " particles...\n";
  auto particles = generate_plummer(
      config.num_particles, config.domain_size / 4.0, config.total_mass, 42);

  // Initialize CUDA solver
  NBodyCUDA solver(Constants::G_NORMALIZED, config.softening);
  solver.set_block_size(config.block_size);

  std::cout << "Copying " << config.num_particles << " particles to GPU...\n";
  solver.initialize(particles);

  Timer total_timer;
  double total_kernel_time = 0.0;

  // Initial force computation
  solver.compute_forces_direct();

  double initial_energy = 0.0, final_energy = 0.0;
  if (config.enable_energy_check) {
    initial_energy = solver.compute_total_energy(particles);
    std::cout << "Initial energy: " << initial_energy << "\n";
  }

  std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";

  total_timer.start();

  for (int step = 0; step < config.num_steps; ++step) {
    solver.step(config.timestep);
    total_kernel_time += solver.get_kernel_time();

    if ((step + 1) % 10 == 0) {
      std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                << " | Kernel: " << solver.get_kernel_time() << " ms"
                << std::flush;
    }
  }

  total_timer.stop();

  // Sync back to host
  solver.synchronize(particles);

  if (config.enable_energy_check) {
    final_energy = solver.compute_total_energy(particles);
  }

  // Calculate performance
  double total_time = total_timer.elapsed_ms();
  long long interactions = solver.get_interaction_count();
  double gflops = (interactions * 20.0 * config.num_steps) /
                  (total_time * 1e6); // ~20 FLOPs per interaction

  std::cout << "\n\n========================================\n";
  std::cout << "       CUDA PERFORMANCE RESULTS\n";
  std::cout << "========================================\n";
  std::cout << "Particles:          " << config.num_particles << "\n";
  std::cout << "Block size:         " << config.block_size << "\n";
  std::cout << "Total time:         " << total_time << " ms\n";
  std::cout << "Total kernel time:  " << total_kernel_time << " ms\n";
  std::cout << "Avg kernel time:    " << total_kernel_time / config.num_steps
            << " ms/step\n";
  std::cout << "Interactions/step:  " << interactions << "\n";
  std::cout << "Performance:        " << gflops << " GFLOPS\n";

  if (config.enable_energy_check) {
    double drift = std::abs(final_energy - initial_energy) /
                   std::abs(initial_energy) * 100;
    std::cout << "\nEnergy drift: " << drift << "%\n";
  }

  std::cout << "========================================\n\n";

  return 0;
}
