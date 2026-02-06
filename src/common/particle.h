#ifndef PARTICLE_H
#define PARTICLE_H

#include "vector3d.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

// AOS (Array of Structures) - traditional layout
struct Particle {
  Vector3D position;
  Vector3D velocity;
  Vector3D acceleration;
  double mass;
  int id;

  CUDA_HOSTDEV Particle()
      : position(), velocity(), acceleration(), mass(1.0), id(0) {}

  CUDA_HOSTDEV Particle(const Vector3D &pos, const Vector3D &vel, double m,
                        int pid = 0)
      : position(pos), velocity(vel), acceleration(), mass(m), id(pid) {}

  // Reset acceleration for new timestep
  CUDA_HOSTDEV void reset_acceleration() { acceleration = Vector3D(); }

  // Compute kinetic energy
  CUDA_HOSTDEV double kinetic_energy() const {
    return 0.5 * mass * velocity.magnitude_sq();
  }
};

// SOA (Structure of Arrays) - GPU-optimized layout
struct ParticleSOA {
  double *pos_x;
  double *pos_y;
  double *pos_z;
  double *vel_x;
  double *vel_y;
  double *vel_z;
  double *acc_x;
  double *acc_y;
  double *acc_z;
  double *mass;
  int *id;
  int count;

  ParticleSOA()
      : pos_x(nullptr), pos_y(nullptr), pos_z(nullptr), vel_x(nullptr),
        vel_y(nullptr), vel_z(nullptr), acc_x(nullptr), acc_y(nullptr),
        acc_z(nullptr), mass(nullptr), id(nullptr), count(0) {}

  // Allocate memory for n particles
  void allocate(int n);

  // Free memory
  void deallocate();

  // Convert from AOS to SOA
  void from_aos(const Particle *particles, int n);

  // Convert from SOA to AOS
  void to_aos(Particle *particles) const;
};

#endif // PARTICLE_H
