#include "particle.h"
#include <cstdlib>

void ParticleSOA::allocate(int n) {
  count = n;
  pos_x = new double[n];
  pos_y = new double[n];
  pos_z = new double[n];
  vel_x = new double[n];
  vel_y = new double[n];
  vel_z = new double[n];
  acc_x = new double[n];
  acc_y = new double[n];
  acc_z = new double[n];
  mass = new double[n];
  id = new int[n];
}

void ParticleSOA::deallocate() {
  delete[] pos_x;
  delete[] pos_y;
  delete[] pos_z;
  delete[] vel_x;
  delete[] vel_y;
  delete[] vel_z;
  delete[] acc_x;
  delete[] acc_y;
  delete[] acc_z;
  delete[] mass;
  delete[] id;

  pos_x = pos_y = pos_z = nullptr;
  vel_x = vel_y = vel_z = nullptr;
  acc_x = acc_y = acc_z = nullptr;
  mass = nullptr;
  id = nullptr;
  count = 0;
}

void ParticleSOA::from_aos(const Particle *particles, int n) {
  if (count != n) {
    deallocate();
    allocate(n);
  }

  for (int i = 0; i < n; ++i) {
    pos_x[i] = particles[i].position.x;
    pos_y[i] = particles[i].position.y;
    pos_z[i] = particles[i].position.z;
    vel_x[i] = particles[i].velocity.x;
    vel_y[i] = particles[i].velocity.y;
    vel_z[i] = particles[i].velocity.z;
    acc_x[i] = particles[i].acceleration.x;
    acc_y[i] = particles[i].acceleration.y;
    acc_z[i] = particles[i].acceleration.z;
    mass[i] = particles[i].mass;
    id[i] = particles[i].id;
  }
}

void ParticleSOA::to_aos(Particle *particles) const {
  for (int i = 0; i < count; ++i) {
    particles[i].position.x = pos_x[i];
    particles[i].position.y = pos_y[i];
    particles[i].position.z = pos_z[i];
    particles[i].velocity.x = vel_x[i];
    particles[i].velocity.y = vel_y[i];
    particles[i].velocity.z = vel_z[i];
    particles[i].acceleration.x = acc_x[i];
    particles[i].acceleration.y = acc_y[i];
    particles[i].acceleration.z = acc_z[i];
    particles[i].mass = mass[i];
    particles[i].id = id[i];
  }
}
