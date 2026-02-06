#ifndef OCTREE_H
#define OCTREE_H

#include "../common/config.h"
#include "../common/particle.h"
#include <array>
#include <memory>
#include <vector>

// Axis-aligned bounding box for octree nodes
struct BoundingBox {
  Vector3D min;
  Vector3D max;
  Vector3D center;
  double size;

  BoundingBox() : size(0) {}
  BoundingBox(const Vector3D &min_, const Vector3D &max_);

  // Check if point is inside box
  bool contains(const Vector3D &point) const;

  // Get octant index (0-7) for a point
  int get_octant(const Vector3D &point) const;

  // Get child bounding box for given octant
  BoundingBox get_child_box(int octant) const;
};

// Octree node for Barnes-Hut algorithm
class OctreeNode {
public:
  // Node types
  enum class Type {
    EMPTY,   // No particles
    LEAF,    // Single particle
    INTERNAL // Has children
  };

  OctreeNode(const BoundingBox &bounds);
  ~OctreeNode() = default;

  // Insert a particle into the tree
  void insert(const Particle &particle, int depth = 0);

  // Compute center of mass for this node and all children
  void compute_mass_distribution();

  // Check if node can be approximated (MAC criterion)
  bool can_approximate(const Vector3D &point, double theta) const;

  // Getters
  Type get_type() const { return type_; }
  const BoundingBox &get_bounds() const { return bounds_; }
  const Vector3D &get_center_of_mass() const { return center_of_mass_; }
  double get_total_mass() const { return total_mass_; }
  const Particle *get_particle() const { return particle_; }
  OctreeNode *get_child(int i) const { return children_[i].get(); }
  int get_particle_count() const { return particle_count_; }

private:
  Type type_;
  BoundingBox bounds_;

  // For internal nodes
  std::array<std::unique_ptr<OctreeNode>, 8> children_;

  // For leaf nodes
  const Particle *particle_;

  // Mass distribution (computed after tree is built)
  Vector3D center_of_mass_;
  double total_mass_;
  int particle_count_;

  // Convert leaf to internal node
  void subdivide();
};

// Octree wrapper class for Barnes-Hut
class Octree {
public:
  Octree(double domain_size = 100.0);

  // Build tree from particles
  void build(const std::vector<Particle> &particles);

  // Clear the tree
  void clear();

  // Get root node
  OctreeNode *get_root() const { return root_.get(); }

  // Statistics
  int get_node_count() const;
  int get_max_depth() const;

private:
  std::unique_ptr<OctreeNode> root_;
  double domain_size_;

  // Compute bounding box for all particles
  BoundingBox
  compute_bounding_box(const std::vector<Particle> &particles) const;
};

#endif // OCTREE_H
