#ifndef CONFIG_H
#define CONFIG_H

#include <string>

// Physical constants
namespace Constants {
constexpr double G = 6.67430e-11;    // Gravitational constant (m³/kg/s²)
constexpr double G_NORMALIZED = 1.0; // Normalized G for simulations
constexpr double SOFTENING = 0.01; // Softening parameter to avoid singularities
constexpr double SOFTENING_SQ = SOFTENING * SOFTENING;
} // namespace Constants

// Barnes-Hut parameters
namespace BarnesHut {
constexpr double THETA = 0.5; // Opening angle criterion (0.5 is typical)
constexpr double THETA_SQ = THETA * THETA;
constexpr int MAX_DEPTH = 50; // Maximum tree depth
} // namespace BarnesHut

// Simulation parameters
struct SimConfig {
  int num_particles;   // Number of particles
  double timestep;     // Time step (dt)
  int num_steps;       // Number of simulation steps
  double domain_size;  // Size of simulation domain
  double total_mass;   // Total mass of system
  bool use_barnes_hut; // Use Barnes-Hut (true) or direct N² (false)
  double theta;        // Barnes-Hut opening angle
  double softening;    // Softening parameter

  // Execution mode
  enum class Mode {
    SERIAL,
    OPENMP,
    MPI,
    CUDA,
    HYBRID_MPI_OPENMP,
    HYBRID_MPI_CUDA
  };
  Mode mode;

  // OpenMP specific
  int num_threads;

  // MPI specific
  int num_ranks;
  int my_rank;

  // CUDA specific
  int block_size;
  int device_id;

  // Output options
  bool output_positions;
  int output_interval;
  std::string output_file;

  // Profiling
  bool enable_profiling;
  bool enable_energy_check;

  // Default constructor with sensible defaults
  SimConfig()
      : num_particles(10000), timestep(0.001), num_steps(100),
        domain_size(100.0), total_mass(1000.0), use_barnes_hut(true),
        theta(BarnesHut::THETA), softening(Constants::SOFTENING),
        mode(Mode::SERIAL), num_threads(1), num_ranks(1), my_rank(0),
        block_size(256), device_id(0), output_positions(false),
        output_interval(10), output_file("output.dat"), enable_profiling(true),
        enable_energy_check(true) {}

  // Parse command line arguments
  void parse_args(int argc, char **argv);

  // Print configuration
  void print() const;
};

// Timer utility for profiling
class Timer {
public:
  void start();
  void stop();
  double elapsed_ms() const;
  double elapsed_s() const;
  void reset();

private:
  double start_time_;
  double end_time_;
  bool running_;
};

// Performance statistics
struct PerfStats {
  double tree_build_time;
  double force_compute_time;
  double integration_time;
  double communication_time; // For MPI
  double total_time;
  double gflops;
  double memory_bandwidth_gb_s;
  long long num_interactions;

  PerfStats()
      : tree_build_time(0), force_compute_time(0), integration_time(0),
        communication_time(0), total_time(0), gflops(0),
        memory_bandwidth_gb_s(0), num_interactions(0) {}

  void print() const;
  void accumulate(const PerfStats &other);
};

#endif // CONFIG_H
