#include "octree.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

// BoundingBox implementation
BoundingBox::BoundingBox(const Vector3D &min_, const Vector3D &max_)
    : min(min_), max(max_) {
  center = (min + max) * 0.5;
  size = (max - min).magnitude();
}

bool BoundingBox::contains(const Vector3D &point) const {
  return point.x >= min.x && point.x <= max.x && point.y >= min.y &&
         point.y <= max.y && point.z >= min.z && point.z <= max.z;
}

int BoundingBox::get_octant(const Vector3D &point) const {
  int octant = 0;
  if (point.x >= center.x)
    octant |= 1;
  if (point.y >= center.y)
    octant |= 2;
  if (point.z >= center.z)
    octant |= 4;
  return octant;
}

BoundingBox BoundingBox::get_child_box(int octant) const {
  Vector3D child_min = min;
  Vector3D child_max = center;

  if (octant & 1) {
    child_min.x = center.x;
    child_max.x = max.x;
  }
  if (octant & 2) {
    child_min.y = center.y;
    child_max.y = max.y;
  }
  if (octant & 4) {
    child_min.z = center.z;
    child_max.z = max.z;
  }

  return BoundingBox(child_min, child_max);
}

// OctreeNode implementation
OctreeNode::OctreeNode(const BoundingBox &bounds)
    : type_(Type::EMPTY), bounds_(bounds), particle_(nullptr), total_mass_(0.0),
      particle_count_(0) {
  for (auto &child : children_) {
    child = nullptr;
  }
}

void OctreeNode::subdivide() {
  for (int i = 0; i < 8; ++i) {
    BoundingBox child_bounds = bounds_.get_child_box(i);
    children_[i] = std::make_unique<OctreeNode>(child_bounds);
  }
}

void OctreeNode::insert(const Particle &particle, int depth) {
  if (depth > BarnesHut::MAX_DEPTH) {
    // Max depth reached, just accumulate mass
    if (type_ == Type::EMPTY) {
      type_ = Type::LEAF;
      particle_ = &particle;
    }
    particle_count_++;
    return;
  }

  switch (type_) {
  case Type::EMPTY:
    // First particle in this node
    type_ = Type::LEAF;
    particle_ = &particle;
    particle_count_ = 1;
    break;

  case Type::LEAF: {
    // Need to subdivide
    const Particle *existing = particle_;
    type_ = Type::INTERNAL;
    particle_ = nullptr;
    subdivide();

    // Re-insert existing particle
    int octant = bounds_.get_octant(existing->position);
    children_[octant]->insert(*existing, depth + 1);

    // Insert new particle
    octant = bounds_.get_octant(particle.position);
    children_[octant]->insert(particle, depth + 1);
    particle_count_ = 2;
    break;
  }

  case Type::INTERNAL: {
    // Insert into appropriate child
    int octant = bounds_.get_octant(particle.position);
    children_[octant]->insert(particle, depth + 1);
    particle_count_++;
    break;
  }
  }
}

void OctreeNode::compute_mass_distribution() {
  switch (type_) {
  case Type::EMPTY:
    total_mass_ = 0.0;
    center_of_mass_ = Vector3D();
    break;

  case Type::LEAF:
    total_mass_ = particle_->mass;
    center_of_mass_ = particle_->position;
    break;

  case Type::INTERNAL:
    total_mass_ = 0.0;
    center_of_mass_ = Vector3D();

    for (auto &child : children_) {
      if (child && child->type_ != Type::EMPTY) {
        child->compute_mass_distribution();
        center_of_mass_ += child->center_of_mass_ * child->total_mass_;
        total_mass_ += child->total_mass_;
      }
    }

    if (total_mass_ > 0.0) {
      center_of_mass_ /= total_mass_;
    }
    break;
  }
}

bool OctreeNode::can_approximate(const Vector3D &point, double theta) const {
  // Multipole acceptance criterion (MAC)
  // If s/d < theta, we can approximate this node as a single mass
  // s = size of node, d = distance from point to center of mass

  double d_sq = (point - center_of_mass_).magnitude_sq();
  double s_sq = bounds_.size * bounds_.size;

  return s_sq < theta * theta * d_sq;
}

// Octree implementation
Octree::Octree(double domain_size) : domain_size_(domain_size) {}

BoundingBox
Octree::compute_bounding_box(const std::vector<Particle> &particles) const {
  if (particles.empty()) {
    return BoundingBox(Vector3D(),
                       Vector3D(domain_size_, domain_size_, domain_size_));
  }

  Vector3D min_pos(std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max());
  Vector3D max_pos(std::numeric_limits<double>::lowest(),
                   std::numeric_limits<double>::lowest(),
                   std::numeric_limits<double>::lowest());

  for (const auto &p : particles) {
    min_pos.x = std::min(min_pos.x, p.position.x);
    min_pos.y = std::min(min_pos.y, p.position.y);
    min_pos.z = std::min(min_pos.z, p.position.z);
    max_pos.x = std::max(max_pos.x, p.position.x);
    max_pos.y = std::max(max_pos.y, p.position.y);
    max_pos.z = std::max(max_pos.z, p.position.z);
  }

  // Make it a cube (octree requires cubic cells)
  double max_dim = std::max(
      {max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z});

  // Ensure non-zero size to prevent division by zero or infinite recursion
  if (max_dim < 1e-9)
    max_dim = 1.0;

  double size = max_dim * 1.01; // Small margin

  Vector3D center = (min_pos + max_pos) * 0.5;
  Vector3D half_size(size * 0.5, size * 0.5, size * 0.5);

  return BoundingBox(center - half_size, center + half_size);
}

void Octree::build(const std::vector<Particle> &particles) {
  clear();

  BoundingBox bounds = compute_bounding_box(particles);
  root_ = std::make_unique<OctreeNode>(bounds);

  for (const auto &p : particles) {
    root_->insert(p);
  }

  root_->compute_mass_distribution();
}

void Octree::clear() { root_.reset(); }

int Octree::get_node_count() const {
  if (!root_)
    return 0;

  std::function<int(const OctreeNode *)> count =
      [&](const OctreeNode *node) -> int {
    if (!node || node->get_type() == OctreeNode::Type::EMPTY)
      return 0;

    int c = 1;
    if (node->get_type() == OctreeNode::Type::INTERNAL) {
      for (int i = 0; i < 8; ++i) {
        c += count(node->get_child(i));
      }
    }
    return c;
  };

  return count(root_.get());
}

int Octree::get_max_depth() const {
  if (!root_)
    return 0;

  std::function<int(const OctreeNode *, int)> depth =
      [&](const OctreeNode *node, int d) -> int {
    if (!node || node->get_type() != OctreeNode::Type::INTERNAL)
      return d;

    int max_d = d;
    for (int i = 0; i < 8; ++i) {
      max_d = std::max(max_d, depth(node->get_child(i), d + 1));
    }
    return max_d;
  };

  return depth(root_.get(), 0);
}
