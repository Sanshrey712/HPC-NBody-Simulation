#ifndef GENERATORS_H
#define GENERATORS_H

#include "../common/particle.h"
#include <vector>

// Generate particles with uniform random distribution
std::vector<Particle> generate_uniform(int n, double domain_size,
                                       double total_mass,
                                       unsigned int seed = 42);

// Generate particles in Gaussian clusters
std::vector<Particle> generate_gaussian_clusters(int n, int num_clusters,
                                                 double domain_size,
                                                 double cluster_std,
                                                 double total_mass,
                                                 unsigned int seed = 42);

// Generate spiral galaxy distribution
std::vector<Particle> generate_galaxy(int n, double radius, double total_mass,
                                      double arm_count = 2,
                                      unsigned int seed = 42);

// Generate two colliding galaxies
std::vector<Particle> generate_collision(int n, double separation,
                                         double radius, double total_mass,
                                         unsigned int seed = 42);

// Generate Plummer sphere (realistic stellar distribution)
std::vector<Particle> generate_plummer(int n, double scale_radius,
                                       double total_mass,
                                       unsigned int seed = 42);

#endif // GENERATORS_H
