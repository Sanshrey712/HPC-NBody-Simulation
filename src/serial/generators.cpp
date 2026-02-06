#include "generators.h"
#include <cmath>
#include <random>

std::vector<Particle> generate_uniform(int n, double domain_size,
                                       double total_mass, unsigned int seed) {
  std::vector<Particle> particles(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> pos_dist(0.0, domain_size);
  std::uniform_real_distribution<double> vel_dist(-0.1, 0.1);

  double mass_per_particle = total_mass / n;

  for (int i = 0; i < n; ++i) {
    Vector3D pos(pos_dist(rng), pos_dist(rng), pos_dist(rng));
    Vector3D vel(vel_dist(rng), vel_dist(rng), vel_dist(rng));
    particles[i] = Particle(pos, vel, mass_per_particle, i);
  }

  return particles;
}

std::vector<Particle> generate_gaussian_clusters(int n, int num_clusters,
                                                 double domain_size,
                                                 double cluster_std,
                                                 double total_mass,
                                                 unsigned int seed) {
  std::vector<Particle> particles(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> center_dist(
      cluster_std * 2, domain_size - cluster_std * 2);
  std::normal_distribution<double> offset_dist(0.0, cluster_std);
  std::uniform_real_distribution<double> vel_dist(-0.05, 0.05);

  double mass_per_particle = total_mass / n;
  int particles_per_cluster = n / num_clusters;

  std::vector<Vector3D> centers(num_clusters);
  for (int c = 0; c < num_clusters; ++c) {
    centers[c] = Vector3D(center_dist(rng), center_dist(rng), center_dist(rng));
  }

  for (int i = 0; i < n; ++i) {
    int cluster = i / particles_per_cluster;
    if (cluster >= num_clusters)
      cluster = num_clusters - 1;

    Vector3D offset(offset_dist(rng), offset_dist(rng), offset_dist(rng));
    Vector3D pos = centers[cluster] + offset;
    Vector3D vel(vel_dist(rng), vel_dist(rng), vel_dist(rng));

    particles[i] = Particle(pos, vel, mass_per_particle, i);
  }

  return particles;
}

std::vector<Particle> generate_galaxy(int n, double radius, double total_mass,
                                      double arm_count, unsigned int seed) {
  std::vector<Particle> particles(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> r_dist(0.1, 1.0);
  std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * M_PI);
  std::normal_distribution<double> z_dist(0.0, radius * 0.05);
  std::normal_distribution<double> vel_noise(0.0, 0.01);

  double mass_per_particle = total_mass / n;

  // Central bulge particles (10%)
  int bulge_count = n / 10;
  std::normal_distribution<double> bulge_dist(0.0, radius * 0.1);

  for (int i = 0; i < bulge_count; ++i) {
    Vector3D pos(bulge_dist(rng), bulge_dist(rng), bulge_dist(rng) * 0.5);
    Vector3D vel(vel_noise(rng), vel_noise(rng), vel_noise(rng));
    particles[i] = Particle(pos, vel, mass_per_particle * 2, i);
  }

  // Spiral arm particles
  for (int i = bulge_count; i < n; ++i) {
    double r = pow(r_dist(rng), 0.5) * radius;
    double base_theta = theta_dist(rng);
    double spiral_theta =
        base_theta + (r / radius) * 2.0 * M_PI * arm_count * 0.5;

    std::normal_distribution<double> theta_noise(0.0, 0.2);
    spiral_theta += theta_noise(rng);

    double x = r * cos(spiral_theta);
    double y = r * sin(spiral_theta);
    double z = z_dist(rng);

    Vector3D pos(x, y, z);

    double v_circ = sqrt(total_mass * 0.5 / (r + 0.1));
    double vx = -v_circ * sin(spiral_theta) + vel_noise(rng);
    double vy = v_circ * cos(spiral_theta) + vel_noise(rng);
    double vz = vel_noise(rng);

    Vector3D vel(vx, vy, vz);
    particles[i] = Particle(pos, vel, mass_per_particle, i);
  }

  return particles;
}

std::vector<Particle> generate_collision(int n, double separation,
                                         double radius, double total_mass,
                                         unsigned int seed) {
  int n_each = n / 2;

  auto galaxy1 = generate_galaxy(n_each, radius, total_mass / 2, 2, seed);
  auto galaxy2 =
      generate_galaxy(n - n_each, radius, total_mass / 2, 2, seed + 1);

  Vector3D offset(separation, 0, 0);
  Vector3D vel_offset(-0.5, 0.2, 0);

  for (auto &p : galaxy2) {
    p.position = p.position + offset;
    p.velocity = p.velocity + vel_offset;
    p.id += n_each;
  }

  std::vector<Particle> particles;
  particles.reserve(n);
  particles.insert(particles.end(), galaxy1.begin(), galaxy1.end());
  particles.insert(particles.end(), galaxy2.begin(), galaxy2.end());

  return particles;
}

std::vector<Particle> generate_plummer(int n, double scale_radius,
                                       double total_mass, unsigned int seed) {
  std::vector<Particle> particles(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u_dist(0.0, 1.0);

  double mass_per_particle = total_mass / n;

  for (int i = 0; i < n; ++i) {
    double u = u_dist(rng);
    double r = scale_radius / sqrt(pow(u, -2.0 / 3.0) - 1.0);
    r = std::min(r, scale_radius * 100.0);

    double cos_theta = 2.0 * u_dist(rng) - 1.0;
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    double phi = 2.0 * M_PI * u_dist(rng);

    double x = r * sin_theta * cos(phi);
    double y = r * sin_theta * sin(phi);
    double z = r * cos_theta;

    Vector3D pos(x, y, z);

    double v_escape =
        sqrt(2.0 * total_mass / sqrt(r * r + scale_radius * scale_radius));

    double q, g;
    do {
      q = u_dist(rng);
      g = q * q * pow(1.0 - q * q, 3.5);
    } while (u_dist(rng) > g / 0.1);

    double v = q * v_escape * 0.5;

    double cos_theta_v = 2.0 * u_dist(rng) - 1.0;
    double sin_theta_v = sqrt(1.0 - cos_theta_v * cos_theta_v);
    double phi_v = 2.0 * M_PI * u_dist(rng);

    double vx = v * sin_theta_v * cos(phi_v);
    double vy = v * sin_theta_v * sin(phi_v);
    double vz = v * cos_theta_v;

    Vector3D vel(vx, vy, vz);
    particles[i] = Particle(pos, vel, mass_per_particle, i);
  }

  return particles;
}
