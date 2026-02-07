#include "domain.h"
#include <algorithm>
#include <cmath>

DomainDecomposition::DomainDecomposition(MPI_Comm comm)
    : comm_(comm), local_count_(0), global_count_(0), domain_size_(100.0),
      comm_time_(0.0) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  // Create Cartesian topology
  dims_[0] = dims_[1] = dims_[2] = 0;
  MPI_Dims_create(size_, 3, dims_);

  int periods[3] = {0, 0, 0}; // Non-periodic
  MPI_Cart_create(comm_, 3, dims_, periods, 1, &cart_comm_);
  MPI_Cart_coords(cart_comm_, rank_, 3, coords_);

  create_mpi_particle_type();
}

DomainDecomposition::~DomainDecomposition() {
  MPI_Type_free(&particle_type_);
  if (cart_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm_);
  }
}

void DomainDecomposition::create_mpi_particle_type() {
  // Create MPI datatype for Particle struct
  // Particle: 3 Vector3D (9 doubles) + 1 double (mass) + 1 int (id) = 10
  // doubles + 1 int

  const int nitems = 5;
  int blocklengths[5] = {3, 3, 3, 1,
                         1}; // position, velocity, acceleration, mass, id
  MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                           MPI_INT};
  MPI_Aint offsets[5];

  Particle dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.position, &offsets[0]);
  MPI_Get_address(&dummy.velocity, &offsets[1]);
  MPI_Get_address(&dummy.acceleration, &offsets[2]);
  MPI_Get_address(&dummy.mass, &offsets[3]);
  MPI_Get_address(&dummy.id, &offsets[4]);

  for (int i = 0; i < nitems; ++i) {
    offsets[i] -= base_address;
  }

  MPI_Datatype struct_type;
  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &struct_type);

  // Resize to match C++ struct alignment/size (essential for arrays)
  MPI_Type_create_resized(struct_type, 0, sizeof(Particle), &particle_type_);
  MPI_Type_commit(&particle_type_);
  MPI_Type_free(&struct_type);
}

void DomainDecomposition::initialize(int num_particles, double domain_size) {
  global_count_ = num_particles;
  domain_size_ = domain_size;

  // Compute local domain bounds based on Cartesian coordinates
  double dx = domain_size_ / dims_[0];
  double dy = domain_size_ / dims_[1];
  double dz = domain_size_ / dims_[2];

  local_min_.x = coords_[0] * dx;
  local_min_.y = coords_[1] * dy;
  local_min_.z = coords_[2] * dz;

  local_max_.x = (coords_[0] + 1) * dx;
  local_max_.y = (coords_[1] + 1) * dy;
  local_max_.z = (coords_[2] + 1) * dz;
}

bool DomainDecomposition::owns_particle(const Vector3D &position) const {
  return position.x >= local_min_.x && position.x < local_max_.x &&
         position.y >= local_min_.y && position.y < local_max_.y &&
         position.z >= local_min_.z && position.z < local_max_.z;
}

int DomainDecomposition::get_owner_rank(const Vector3D &position) const {
  double dx = domain_size_ / dims_[0];
  double dy = domain_size_ / dims_[1];
  double dz = domain_size_ / dims_[2];

  int cx = static_cast<int>(position.x / dx);
  int cy = static_cast<int>(position.y / dy);
  int cz = static_cast<int>(position.z / dz);

  // Clamp to valid range
  cx = std::max(0, std::min(cx, dims_[0] - 1));
  cy = std::max(0, std::min(cy, dims_[1] - 1));
  cz = std::max(0, std::min(cz, dims_[2] - 1));

  int coords[3] = {cx, cy, cz};
  int owner;
  MPI_Cart_rank(cart_comm_, coords, &owner);

  return owner;
}

void DomainDecomposition::distribute_particles(
    std::vector<Particle> &local_particles,
    const std::vector<Particle> &all_particles) {
  double start_time = MPI_Wtime();

  if (rank_ == 0) {
    // Rank 0 distributes particles
    std::vector<std::vector<Particle>> send_buffers(size_);

    for (const auto &p : all_particles) {
      int owner = get_owner_rank(p.position);
      send_buffers[owner].push_back(p);
    }

    // Keep local particles
    local_particles = std::move(send_buffers[0]);

    // Send to other ranks
    for (int r = 1; r < size_; ++r) {
      int count = static_cast<int>(send_buffers[r].size());
      MPI_Send(&count, 1, MPI_INT, r, 0, comm_);
      if (count > 0) {
        MPI_Send(send_buffers[r].data(), count, particle_type_, r, 1, comm_);
      }
    }
  } else {
    // Receive from rank 0
    int count;
    MPI_Recv(&count, 1, MPI_INT, 0, 0, comm_, MPI_STATUS_IGNORE);
    local_particles.resize(count);
    if (count > 0) {
      MPI_Recv(local_particles.data(), count, particle_type_, 0, 1, comm_,
               MPI_STATUS_IGNORE);
    }
  }

  local_count_ = static_cast<int>(local_particles.size());
  comm_time_ += MPI_Wtime() - start_time;
}

void DomainDecomposition::gather_particles(
    std::vector<Particle> &all_particles,
    const std::vector<Particle> &local_particles) {
  double start_time = MPI_Wtime();

  // Gather counts
  std::vector<int> counts(size_);
  int local_size = static_cast<int>(local_particles.size());
  MPI_Gather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);

  if (rank_ == 0) {
    // Compute displacements
    std::vector<int> displs(size_);
    displs[0] = 0;
    for (int i = 1; i < size_; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }

    int total = displs[size_ - 1] + counts[size_ - 1];
    all_particles.resize(total);

    MPI_Gatherv(local_particles.data(), local_size, particle_type_,
                all_particles.data(), counts.data(), displs.data(),
                particle_type_, 0, comm_);
  } else {
    MPI_Gatherv(local_particles.data(), local_size, particle_type_, nullptr,
                nullptr, nullptr, particle_type_, 0, comm_);
  }

  comm_time_ += MPI_Wtime() - start_time;
}

void DomainDecomposition::exchange_halo(std::vector<Particle> &local_particles,
                                        std::vector<Particle> &halo_particles,
                                        double halo_width) {
  double start_time = MPI_Wtime();
  halo_particles.clear();

  // Find particles near boundaries that need to be sent
  std::vector<std::vector<Particle>> send_buffers(size_);

  for (const auto &p : local_particles) {
    // Check each direction
    for (int dim = 0; dim < 3; ++dim) {
      double pos = (dim == 0)   ? p.position.x
                   : (dim == 1) ? p.position.y
                                : p.position.z;
      double min_bound = (dim == 0)   ? local_min_.x
                         : (dim == 1) ? local_min_.y
                                      : local_min_.z;
      double max_bound = (dim == 0)   ? local_max_.x
                         : (dim == 1) ? local_max_.y
                                      : local_max_.z;

      // Check lower neighbor
      if (pos - min_bound < halo_width && coords_[dim] > 0) {
        int neighbor_coords[3] = {coords_[0], coords_[1], coords_[2]};
        neighbor_coords[dim]--;
        int neighbor;
        MPI_Cart_rank(cart_comm_, neighbor_coords, &neighbor);
        send_buffers[neighbor].push_back(p);
      }

      // Check upper neighbor
      if (max_bound - pos < halo_width && coords_[dim] < dims_[dim] - 1) {
        int neighbor_coords[3] = {coords_[0], coords_[1], coords_[2]};
        neighbor_coords[dim]++;
        int neighbor;
        MPI_Cart_rank(cart_comm_, neighbor_coords, &neighbor);
        send_buffers[neighbor].push_back(p);
      }
    }
  }

  // Exchange with all neighbors using non-blocking communication
  std::vector<MPI_Request> requests;
  std::vector<int> send_counts(size_), recv_counts(size_);

  for (int r = 0; r < size_; ++r) {
    send_counts[r] = static_cast<int>(send_buffers[r].size());
  }

  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               comm_);

  // Receive particles
  int total_recv = 0;
  for (int r = 0; r < size_; ++r) {
    total_recv += recv_counts[r];
  }
  halo_particles.resize(total_recv);

  // Send and receive
  int recv_offset = 0;
  for (int r = 0; r < size_; ++r) {
    if (r != rank_) {
      if (recv_counts[r] > 0) {
        MPI_Request req;
        MPI_Irecv(&halo_particles[recv_offset], recv_counts[r], particle_type_,
                  r, 0, comm_, &req);
        requests.push_back(req);
      }
      if (send_counts[r] > 0) {
        MPI_Request req;
        MPI_Isend(send_buffers[r].data(), send_counts[r], particle_type_, r, 0,
                  comm_, &req);
        requests.push_back(req);
      }
    }
    recv_offset += recv_counts[r];
  }

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
              MPI_STATUSES_IGNORE);

  comm_time_ += MPI_Wtime() - start_time;
}

void DomainDecomposition::redistribute(std::vector<Particle> &local_particles) {
  double start_time = MPI_Wtime();

  // Find particles that have moved out of local domain
  std::vector<std::vector<Particle>> send_buffers(size_);
  std::vector<Particle> staying;

  for (auto &p : local_particles) {
    if (owns_particle(p.position)) {
      staying.push_back(p);
    } else {
      int owner = get_owner_rank(p.position);
      if (owner == rank_) {
        // Particle is outside our bounds but clamping assigns it to us
        // Keep it to avoiding sending to self (which causes data
        // loss/corruption)
        staying.push_back(p);
      } else {
        send_buffers[owner].push_back(p);
      }
    }
  }

  local_particles = std::move(staying);

  // Exchange counts
  std::vector<int> send_counts(size_), recv_counts(size_);
  for (int r = 0; r < size_; ++r) {
    send_counts[r] = static_cast<int>(send_buffers[r].size());
  }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               comm_);

  // Receive particles
  int total_recv = 0;
  for (int r = 0; r < size_; ++r) {
    if (r != rank_)
      total_recv += recv_counts[r];
  }

  std::vector<Particle> incoming(total_recv);
  std::vector<MPI_Request> requests;

  int recv_offset = 0;
  for (int r = 0; r < size_; ++r) {
    if (r != rank_) {
      if (recv_counts[r] > 0) {
        MPI_Request req;
        MPI_Irecv(&incoming[recv_offset], recv_counts[r], particle_type_, r, 0,
                  comm_, &req);
        requests.push_back(req);
        recv_offset += recv_counts[r];
      }
      if (send_counts[r] > 0) {
        MPI_Request req;
        MPI_Isend(send_buffers[r].data(), send_counts[r], particle_type_, r, 0,
                  comm_, &req);
        requests.push_back(req);
      }
    }
  }

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
              MPI_STATUSES_IGNORE);

  // Add incoming particles
  local_particles.insert(local_particles.end(), incoming.begin(),
                         incoming.end());
  local_count_ = static_cast<int>(local_particles.size());

  comm_time_ += MPI_Wtime() - start_time;
}

void DomainDecomposition::get_local_bounds(Vector3D &min, Vector3D &max) const {
  min = local_min_;
  max = local_max_;
}
