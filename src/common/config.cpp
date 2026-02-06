#include "config.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

void SimConfig::parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--particles") == 0 || strcmp(argv[i], "-n") == 0) {
      num_particles = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--steps") == 0 || strcmp(argv[i], "-s") == 0) {
      num_steps = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--dt") == 0) {
      timestep = std::atof(argv[++i]);
    } else if (strcmp(argv[i], "--theta") == 0) {
      theta = std::atof(argv[++i]);
    } else if (strcmp(argv[i], "--softening") == 0) {
      softening = std::atof(argv[++i]);
    } else if (strcmp(argv[i], "--mode") == 0) {
      const char *mode_str = argv[++i];
      if (strcmp(mode_str, "serial") == 0)
        mode = Mode::SERIAL;
      else if (strcmp(mode_str, "openmp") == 0)
        mode = Mode::OPENMP;
      else if (strcmp(mode_str, "mpi") == 0)
        mode = Mode::MPI;
      else if (strcmp(mode_str, "cuda") == 0)
        mode = Mode::CUDA;
      else if (strcmp(mode_str, "hybrid-mpi-omp") == 0)
        mode = Mode::HYBRID_MPI_OPENMP;
      else if (strcmp(mode_str, "hybrid-mpi-cuda") == 0)
        mode = Mode::HYBRID_MPI_CUDA;
    } else if (strcmp(argv[i], "--threads") == 0 ||
               strcmp(argv[i], "-t") == 0) {
      num_threads = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--block-size") == 0) {
      block_size = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--direct") == 0) {
      use_barnes_hut = false;
    } else if (strcmp(argv[i], "--barnes-hut") == 0) {
      use_barnes_hut = true;
    } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
      output_positions = true;
      output_file = argv[++i];
    } else if (strcmp(argv[i], "--output-interval") == 0) {
      output_interval = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--no-profile") == 0) {
      enable_profiling = false;
    } else if (strcmp(argv[i], "--no-energy") == 0) {
      enable_energy_check = false;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      std::cout
          << "N-Body Simulation\n"
          << "Usage: nbody [options]\n\n"
          << "Options:\n"
          << "  -n, --particles N    Number of particles (default: 10000)\n"
          << "  -s, --steps N        Number of timesteps (default: 100)\n"
          << "  --dt T               Timestep size (default: 0.001)\n"
          << "  --theta T            Barnes-Hut opening angle (default: 0.5)\n"
          << "  --softening S        Softening parameter (default: 0.01)\n"
          << "  --mode MODE          Execution mode: serial, openmp, mpi, "
             "cuda\n"
          << "  -t, --threads N      Number of OpenMP threads\n"
          << "  --block-size N       CUDA block size (default: 256)\n"
          << "  --direct             Use direct O(N²) method\n"
          << "  --barnes-hut         Use Barnes-Hut O(N log N) method\n"
          << "  -o, --output FILE    Output positions to file\n"
          << "  --output-interval N  Output every N steps\n"
          << "  --no-profile         Disable profiling\n"
          << "  --no-energy          Disable energy conservation check\n"
          << "  -h, --help           Show this help\n";
      std::exit(0);
    }
  }
}

void SimConfig::print() const {
  std::cout << "\n=== Simulation Configuration ===\n"
            << "Particles:       " << num_particles << "\n"
            << "Timesteps:       " << num_steps << "\n"
            << "dt:              " << timestep << "\n"
            << "Domain size:     " << domain_size << "\n"
            << "Algorithm:       "
            << (use_barnes_hut ? "Barnes-Hut" : "Direct N²") << "\n";

  if (use_barnes_hut) {
    std::cout << "Theta:           " << theta << "\n";
  }

  std::cout << "Softening:       " << softening << "\n"
            << "Mode:            ";

  switch (mode) {
  case Mode::SERIAL:
    std::cout << "Serial";
    break;
  case Mode::OPENMP:
    std::cout << "OpenMP (" << num_threads << " threads)";
    break;
  case Mode::MPI:
    std::cout << "MPI (" << num_ranks << " ranks)";
    break;
  case Mode::CUDA:
    std::cout << "CUDA (block size: " << block_size << ")";
    break;
  case Mode::HYBRID_MPI_OPENMP:
    std::cout << "Hybrid MPI+OpenMP";
    break;
  case Mode::HYBRID_MPI_CUDA:
    std::cout << "Hybrid MPI+CUDA";
    break;
  }
  std::cout << "\n================================\n\n";
}

// Timer implementation
static double get_time() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration<double>(duration).count();
}

void Timer::start() {
  start_time_ = get_time();
  running_ = true;
}

void Timer::stop() {
  end_time_ = get_time();
  running_ = false;
}

double Timer::elapsed_ms() const {
  if (running_) {
    return (get_time() - start_time_) * 1000.0;
  }
  return (end_time_ - start_time_) * 1000.0;
}

double Timer::elapsed_s() const { return elapsed_ms() / 1000.0; }

void Timer::reset() {
  start_time_ = 0;
  end_time_ = 0;
  running_ = false;
}

// PerfStats implementation
void PerfStats::print() const {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n=== Performance Statistics ===\n"
            << "Tree build time:    " << tree_build_time << " ms\n"
            << "Force compute time: " << force_compute_time << " ms\n"
            << "Integration time:   " << integration_time << " ms\n";
  if (communication_time > 0) {
    std::cout << "Communication time: " << communication_time << " ms\n";
  }
  std::cout << "Total time:         " << total_time << " ms\n"
            << "Interactions:       " << num_interactions << "\n";
  if (gflops > 0) {
    std::cout << "Performance:        " << gflops << " GFLOPS\n";
  }
  std::cout << "==============================\n\n";
}

void PerfStats::accumulate(const PerfStats &other) {
  tree_build_time += other.tree_build_time;
  force_compute_time += other.force_compute_time;
  integration_time += other.integration_time;
  communication_time += other.communication_time;
  total_time += other.total_time;
  num_interactions += other.num_interactions;
}
