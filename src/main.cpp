#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

// Common
#include "common/config.h"
#include "common/particle.h"

// Serial
#include "serial/barnes_hut.h"
#include "serial/direct_nbody.h"
#include "serial/generators.h"

// OpenMP
#ifdef USE_OPENMP
#include "openmp/barnes_hut_omp.h"
#include "openmp/direct_omp.h"
#include <omp.h>
#endif

// MPI
#ifdef USE_MPI
#include "mpi/nbody_mpi.h"
#include <mpi.h>
#endif

// CUDA
#ifdef USE_CUDA
#include "cuda/nbody_cuda.h"
#endif

// LIKWID
#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

// Output particle positions
void output_particles(const std::vector<Particle> &particles,
                      const std::string &filename, int step) {
  std::ofstream file(filename, step == 0 ? std::ios::trunc : std::ios::app);
  file << "# Step " << step << ", N=" << particles.size() << "\n";
  for (const auto &p : particles) {
    file << p.position.x << " " << p.position.y << " " << p.position.z << "\n";
  }
  file << "\n";
}

// Generate initial conditions
std::vector<Particle> generate_particles(const SimConfig &config) {
  std::cout << "Generating " << config.num_particles << " particles...\n";
  return generate_plummer(config.num_particles, config.domain_size / 4.0,
                          config.total_mass, 42);
}

// Run serial simulation
void run_serial(SimConfig &config) {
  std::cout << "\n=== Serial Mode ===\n";

  auto particles = generate_particles(config);

  Timer timer;
  PerfStats stats;

  double initial_energy = 0.0;

  if (config.use_barnes_hut) {
    BarnesHutSolver solver(Constants::G_NORMALIZED, config.theta,
                           config.softening);

    // Initial force computation
    solver.build_tree(particles);
    solver.compute_forces(particles);

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    timer.start();

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if (config.output_positions && step % config.output_interval == 0) {
        output_particles(particles, config.output_file, step);
      }

      if (step % 10 == 0) {
        std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
      }
    }

    timer.stop();
    stats.force_compute_time = timer.elapsed_ms();
    stats.num_interactions = solver.get_interaction_count();

    if (config.enable_energy_check) {
      double final_energy = solver.compute_total_energy(particles);
      std::cout << "\nFinal energy: " << final_energy << " (drift: "
                << std::abs(final_energy - initial_energy) /
                       std::abs(initial_energy) * 100
                << "%)\n";
    }
  } else {
    DirectNBody solver(Constants::G_NORMALIZED, config.softening);

    solver.compute_forces(particles);

    if (config.enable_energy_check) {
      initial_energy = solver.compute_total_energy(particles);
      std::cout << "Initial energy: " << initial_energy << "\n";
    }

    timer.start();

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if (config.output_positions && step % config.output_interval == 0) {
        output_particles(particles, config.output_file, step);
      }

      if (step % 10 == 0) {
        std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
      }
    }

    timer.stop();
    stats.force_compute_time = timer.elapsed_ms();
    stats.num_interactions = solver.get_interaction_count();

    if (config.enable_energy_check) {
      double final_energy = solver.compute_total_energy(particles);
      std::cout << "\nFinal energy: " << final_energy << " (drift: "
                << std::abs(final_energy - initial_energy) /
                       std::abs(initial_energy) * 100
                << "%)\n";
    }
  }

  stats.total_time = stats.force_compute_time;
  std::cout << "\n";
  stats.print();
}

#ifdef USE_OPENMP
// Run OpenMP simulation
void run_openmp(SimConfig &config) {
  std::cout << "\n=== OpenMP Mode (" << config.num_threads << " threads) ===\n";
  omp_set_num_threads(config.num_threads);

  auto particles = generate_particles(config);

  Timer timer;
  PerfStats stats;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  if (config.use_barnes_hut) {
    BarnesHutOMP solver(Constants::G_NORMALIZED, config.theta,
                        config.softening);
    solver.set_num_threads(config.num_threads);

    solver.build_tree(particles);
    solver.compute_forces(particles);

    timer.start();

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("BH_FORCE_OMP");
#endif

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if (step % 10 == 0) {
        std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
      }
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("BH_FORCE_OMP");
#endif

    timer.stop();
    stats.num_interactions = solver.get_interaction_count();
  } else {
    DirectNBodyOMP solver(Constants::G_NORMALIZED, config.softening);
    solver.set_num_threads(config.num_threads);

    solver.compute_forces(particles);

    timer.start();

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("DIRECT_FORCE_OMP");
#endif

    for (int step = 0; step < config.num_steps; ++step) {
      solver.step(particles, config.timestep);

      if (step % 10 == 0) {
        std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
      }
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("DIRECT_FORCE_OMP");
#endif

    timer.stop();
    stats.num_interactions = solver.get_interaction_count();
  }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  stats.total_time = timer.elapsed_ms();
  std::cout << "\n";
  stats.print();
}
#endif

#ifdef USE_MPI
// Run MPI simulation
void run_mpi(SimConfig &config) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== MPI Mode (" << size << " ranks) ===\n";
  }

  std::vector<Particle> all_particles;
  if (rank == 0) {
    all_particles = generate_particles(config);
  }

  NBodyMPI solver(Constants::G_NORMALIZED, config.theta, config.softening);
  solver.set_use_barnes_hut(config.use_barnes_hut);
  solver.initialize(all_particles, config.domain_size);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  solver.compute_forces();

  for (int step = 0; step < config.num_steps; ++step) {
    solver.step(config.timestep);

    if (rank == 0 && step % 10 == 0) {
      std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  if (rank == 0) {
    std::cout << "\n\n=== MPI Performance ===\n";
    std::cout << "Total time: " << (end_time - start_time) * 1000.0 << " ms\n";
    std::cout << "Communication time: "
              << solver.get_communication_time() * 1000.0 << " ms\n";
    std::cout << "Interactions: " << solver.get_interaction_count() << "\n";
    std::cout << "=======================\n";
  }
}
#endif

#ifdef USE_CUDA
// Run CUDA simulation
void run_cuda(SimConfig &config) {
  std::cout << "\n=== CUDA Mode ===\n";

  if (!NBodyCUDA::is_available()) {
    std::cerr << "No CUDA device available!\n";
    return;
  }

  NBodyCUDA::print_device_info();

  auto particles = generate_particles(config);

  NBodyCUDA solver(Constants::G_NORMALIZED, config.softening);
  solver.set_block_size(config.block_size);
  solver.initialize(particles);

  Timer timer;
  timer.start();

  solver.compute_forces_direct();

  for (int step = 0; step < config.num_steps; ++step) {
    solver.step(config.timestep);

    if (step % 10 == 0) {
      std::cout << "\rStep " << step << "/" << config.num_steps << std::flush;
    }
  }

  solver.synchronize(particles);
  timer.stop();

  std::cout << "\n\n=== CUDA Performance ===\n";
  std::cout << "Total time: " << timer.elapsed_ms() << " ms\n";
  std::cout << "Kernel time (last): " << solver.get_kernel_time() << " ms\n";
  std::cout << "Interactions: " << solver.get_interaction_count() << "\n";
  std::cout << "========================\n";
}
#endif

int main(int argc, char *argv[]) {
  SimConfig config;
  config.parse_args(argc, argv);

#ifdef USE_MPI
  if (config.mode == SimConfig::Mode::MPI ||
      config.mode == SimConfig::Mode::HYBRID_MPI_OPENMP ||
      config.mode == SimConfig::Mode::HYBRID_MPI_CUDA) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &config.num_ranks);
  }
#endif

  if (config.my_rank == 0) {
    config.print();
  }

  switch (config.mode) {
  case SimConfig::Mode::SERIAL:
    run_serial(config);
    break;

#ifdef USE_OPENMP
  case SimConfig::Mode::OPENMP:
    run_openmp(config);
    break;
#endif

#ifdef USE_MPI
  case SimConfig::Mode::MPI:
    run_mpi(config);
    break;
#endif

#ifdef USE_CUDA
  case SimConfig::Mode::CUDA:
    run_cuda(config);
    break;
#endif

  default:
    if (config.my_rank == 0) {
      std::cerr << "Unsupported mode or feature not compiled.\n";
    }
    break;
  }

#ifdef USE_MPI
  if (config.mode == SimConfig::Mode::MPI ||
      config.mode == SimConfig::Mode::HYBRID_MPI_OPENMP ||
      config.mode == SimConfig::Mode::HYBRID_MPI_CUDA) {
    MPI_Finalize();
  }
#endif

  return 0;
}
