// MPI Distributed N-Body Simulation
// Usage: mpirun -np 4 ./nbody_mpi --particles 100000 --steps 100

#include "common/config.h"
#include "common/particle.h"
#include "mpi/nbody_mpi.h"
#include "serial/generators.h"
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SimConfig config;
  config.parse_args(argc, argv);
  config.mode = SimConfig::Mode::MPI;
  config.my_rank = rank;
  config.num_ranks = size;

  if (rank == 0) {
    std::cout << "\n========================================\n";
    std::cout << "       MPI N-BODY SIMULATION\n";
    std::cout << "========================================\n";
    std::cout << "MPI Ranks: " << size << "\n";
    config.print();
  }

  // Generate particles on rank 0
  std::vector<Particle> all_particles;
  if (rank == 0) {
    std::cout << "Generating " << config.num_particles << " particles...\n";
    all_particles = generate_plummer(
        config.num_particles, config.domain_size / 4.0, config.total_mass, 42);
  }

  // Initialize MPI solver
  NBodyMPI solver(Constants::G_NORMALIZED, config.theta, config.softening);
  solver.set_use_barnes_hut(config.use_barnes_hut);
  solver.initialize(all_particles, config.domain_size);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  // Initial force computation
  solver.compute_forces();

  double initial_energy = 0.0, final_energy = 0.0;
  if (config.enable_energy_check) {
    initial_energy = solver.compute_total_energy();
    if (rank == 0) {
      std::cout << "Initial energy: " << initial_energy << "\n";
    }
  }

  if (rank == 0) {
    std::cout << "\nSimulating " << config.num_steps << " timesteps...\n";
  }

  for (int step = 0; step < config.num_steps; ++step) {
    solver.step(config.timestep);

    if (rank == 0 && (step + 1) % 10 == 0) {
      std::cout << "\rStep " << (step + 1) << "/" << config.num_steps
                << " | Local particles: " << solver.get_local_particles().size()
                << std::flush;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  if (config.enable_energy_check) {
    final_energy = solver.compute_total_energy();
  }

  // Gather statistics
  double total_time = (end_time - start_time) * 1000.0;
  double comm_time = solver.get_communication_time() * 1000.0;
  long long interactions = solver.get_interaction_count();

  // Reduce to get totals
  long long total_interactions;
  MPI_Reduce(&interactions, &total_interactions, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n\n========================================\n";
    std::cout << "       MPI PERFORMANCE RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "MPI Ranks:          " << size << "\n";
    std::cout << "Total time:         " << total_time << " ms\n";
    std::cout << "Communication time: " << comm_time << " ms\n";
    std::cout << "Compute time:       " << (total_time - comm_time) << " ms\n";
    std::cout << "Comm overhead:      " << (comm_time / total_time * 100)
              << "%\n";
    std::cout << "Total interactions: " << total_interactions << "\n";

    if (config.enable_energy_check) {
      double drift = std::abs(final_energy - initial_energy) /
                     std::abs(initial_energy) * 100;
      std::cout << "\nEnergy drift: " << drift << "%\n";
    }

    std::cout << "========================================\n\n";
  }

  MPI_Finalize();
  return 0;
}
