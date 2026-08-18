#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

enum Direction : int {
  XMinus = 0,
  XPlus,
  YMinus,
  YPlus,
  ZMinus,
  ZPlus,
  DirectionCount
};

constexpr std::array<int, DirectionCount> opposite = {XPlus,  XMinus, YPlus,
                                                      YMinus, ZPlus,  ZMinus};

void mpiCheck(const int error, const char *operation,
              const MPI_Comm communicator = MPI_COMM_WORLD) {
  if (error == MPI_SUCCESS)
    return;

  char message[MPI_MAX_ERROR_STRING];
  int messageLength = 0;
  MPI_Error_string(error, message, &messageLength);

  int rank = -1;
  MPI_Comm_rank(communicator, &rank);
  std::fprintf(stderr, "MPI error on rank %d in %s: %.*s\n", rank, operation,
               messageLength, message);
  MPI_Abort(communicator, error);
  std::abort();
}

bool checkedMultiply(const size_t a, const size_t b, size_t &product) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    return false;
  product = a * b;
  return true;
}

bool parseSize(const char *text, size_t &value) {
  if (*text == '-')
    return false;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (text == end || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max())
    return false;
  value = static_cast<size_t>(parsed);
  return true;
}

bool parseIterations(const char *text, int &value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (text == end || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max())
    return false;
  value = static_cast<int>(parsed);
  return true;
}

void printUsage(const char *programName) {
  std::printf("Usage: %s [options]\n", programName);
  std::printf("Options:\n");
  std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
  std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
  std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
  std::printf("  -i <num>     Number of time steps (default: 20)\n");
  std::printf("  -v           Enable validation\n");
  std::printf("  -r           Print results for external validation\n");
  std::printf("  -h           Show this help message\n");
}

// Minimize the exact number of cells crossing internal process interfaces.
// Unlike MPI_Dims_create, this also guarantees that no rank receives an empty
// block.
std::array<int, 3> chooseProcessGrid(const int processCount, const size_t nx,
                                     const size_t ny, const size_t nz) {
  std::array<int, 3> best = {0, 0, 0};
  long double bestInterfaceArea = std::numeric_limits<long double>::infinity();
  size_t bestMaximumVolume = std::numeric_limits<size_t>::max();

  for (int px = 1; px <= processCount; ++px) {
    if (processCount % px != 0 || static_cast<size_t>(px) > nx)
      continue;
    const int yzProcesses = processCount / px;
    for (int py = 1; py <= yzProcesses; ++py) {
      if (yzProcesses % py != 0 || static_cast<size_t>(py) > ny)
        continue;
      const int pz = yzProcesses / py;
      if (static_cast<size_t>(pz) > nz)
        continue;

      const long double interfaceArea =
          static_cast<long double>(px - 1) * static_cast<long double>(ny) *
              static_cast<long double>(nz) +
          static_cast<long double>(py - 1) * static_cast<long double>(nx) *
              static_cast<long double>(nz) +
          static_cast<long double>(pz - 1) * static_cast<long double>(nx) *
              static_cast<long double>(ny);

      size_t maximumVolume =
          (nx + static_cast<size_t>(px) - 1) / static_cast<size_t>(px);
      maximumVolume *=
          (ny + static_cast<size_t>(py) - 1) / static_cast<size_t>(py);
      maximumVolume *=
          (nz + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz);

      if (interfaceArea < bestInterfaceArea ||
          (interfaceArea == bestInterfaceArea &&
           maximumVolume < bestMaximumVolume)) {
        best = {px, py, pz};
        bestInterfaceArea = interfaceArea;
        bestMaximumVolume = maximumVolume;
      }
    }
  }
  return best;
}

void splitDimension(const size_t globalLength, const int processLength,
                    const int coordinate, size_t &localLength,
                    size_t &globalStart) {
  const size_t base = globalLength / static_cast<size_t>(processLength);
  const size_t remainder = globalLength % static_cast<size_t>(processLength);
  localLength = base + (static_cast<size_t>(coordinate) < remainder ? 1 : 0);
  globalStart = static_cast<size_t>(coordinate) * base +
                std::min(static_cast<size_t>(coordinate), remainder);
}

struct Domain {
  size_t nx = 0;
  size_t ny = 0;
  size_t nz = 0;
  size_t startX = 0;
  size_t startY = 0;
  size_t startZ = 0;
  size_t rowStride = 0;
  size_t planeStride = 0;

  size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
    return z * planeStride + y * rowStride + x;
  }
};

Domain makeDomain(const std::array<size_t, 3> &global,
                  const std::array<int, 3> &processGrid,
                  const std::array<int, 3> &coordinates) {
  Domain domain;
  splitDimension(global[0], processGrid[0], coordinates[0], domain.nx,
                 domain.startX);
  splitDimension(global[1], processGrid[1], coordinates[1], domain.ny,
                 domain.startY);
  splitDimension(global[2], processGrid[2], coordinates[2], domain.nz,
                 domain.startZ);
  domain.rowStride = domain.nx + 2;
  domain.planeStride = domain.rowStride * (domain.ny + 2);
  return domain;
}

class HaloExchange {
public:
  HaloExchange(const Domain &domain, const MPI_Comm communicator)
      : domain_(domain), communicator_(communicator) {
    int source = MPI_PROC_NULL;
    int destination = MPI_PROC_NULL;
    mpiCheck(MPI_Cart_shift(communicator_, 0, 1, &source, &destination),
             "MPI_Cart_shift(x)", communicator_);
    neighbors_[XMinus] = source;
    neighbors_[XPlus] = destination;
    mpiCheck(MPI_Cart_shift(communicator_, 1, 1, &source, &destination),
             "MPI_Cart_shift(y)", communicator_);
    neighbors_[YMinus] = source;
    neighbors_[YPlus] = destination;
    mpiCheck(MPI_Cart_shift(communicator_, 2, 1, &source, &destination),
             "MPI_Cart_shift(z)", communicator_);
    neighbors_[ZMinus] = source;
    neighbors_[ZPlus] = destination;

    counts_[XMinus] = counts_[XPlus] = domain_.ny * domain_.nz;
    counts_[YMinus] = counts_[YPlus] = domain_.nx * domain_.nz;
    counts_[ZMinus] = counts_[ZPlus] = domain_.nx * domain_.ny;

    size_t totalElements = 0;
    for (int direction = 0; direction < DirectionCount; ++direction) {
      offsets_[direction] = totalElements;
      totalElements += counts_[direction];
    }
    sendBuffers_.resize(totalElements);
    receiveBuffers_.resize(totalElements);

    // Persistent requests remove per-step matching/setup overhead. The packed
    // buffers keep their addresses for the lifetime of this object.
    receiveRequests_.reserve(DirectionCount);
    sendRequests_.reserve(DirectionCount);
    createPersistentRequests();
  }

  HaloExchange(const HaloExchange &) = delete;
  HaloExchange &operator=(const HaloExchange &) = delete;

  ~HaloExchange() {
    waitForSends();
    for (MPI_Request &request : receiveRequests_)
      mpiCheck(MPI_Request_free(&request), "MPI_Request_free(receive)",
               communicator_);
    for (MPI_Request &request : sendRequests_)
      mpiCheck(MPI_Request_free(&request), "MPI_Request_free(send)",
               communicator_);
  }

  bool hasNeighbor(const Direction direction) const noexcept {
    return neighbors_[direction] != MPI_PROC_NULL;
  }

  void start(std::vector<double> &field) {
    // The preceding send can finish while its boundary cells are computed.
    // It must be complete only before the fixed send buffers are repacked.
    waitForSends();
    fillPhysicalBoundaries(field);

    // Receives are posted first so peers can begin transferring while faces are
    // packed.
    if (!receiveRequests_.empty())
      mpiCheck(MPI_Startall(static_cast<int>(receiveRequests_.size()),
                            receiveRequests_.data()),
               "MPI_Startall(receive)", communicator_);
    for (int direction = 0; direction < DirectionCount; ++direction) {
      if (neighbors_[direction] != MPI_PROC_NULL) {
        pack(field, static_cast<Direction>(direction),
             sendBuffers_.data() + offsets_[direction]);
      }
    }
    if (!sendRequests_.empty()) {
      mpiCheck(MPI_Startall(static_cast<int>(sendRequests_.size()),
                            sendRequests_.data()),
               "MPI_Startall(send)", communicator_);
      sendsActive_ = true;
    }
  }

  void finish(std::vector<double> &field) {
    if (!receiveRequests_.empty()) {
      mpiCheck(MPI_Waitall(static_cast<int>(receiveRequests_.size()),
                           receiveRequests_.data(), MPI_STATUSES_IGNORE),
               "MPI_Waitall(receive)", communicator_);
    }
    for (int direction = 0; direction < DirectionCount; ++direction) {
      if (neighbors_[direction] != MPI_PROC_NULL) {
        unpack(field, static_cast<Direction>(direction),
               receiveBuffers_.data() + offsets_[direction]);
      }
    }
  }

  void complete() { waitForSends(); }

private:
  void createPersistentRequests() {
    constexpr size_t maximumCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    for (int direction = 0; direction < DirectionCount; ++direction) {
      if (neighbors_[direction] == MPI_PROC_NULL)
        continue;

      size_t remaining = counts_[direction];
      size_t offset = offsets_[direction];
      while (remaining != 0) {
        const int chunk = static_cast<int>(std::min(remaining, maximumCount));
        receiveRequests_.emplace_back();
        mpiCheck(MPI_Recv_init(receiveBuffers_.data() + offset, chunk,
                               MPI_DOUBLE, neighbors_[direction],
                               opposite[direction], communicator_,
                               &receiveRequests_.back()),
                 "MPI_Recv_init", communicator_);
        sendRequests_.emplace_back();
        mpiCheck(MPI_Send_init(sendBuffers_.data() + offset, chunk, MPI_DOUBLE,
                               neighbors_[direction], direction, communicator_,
                               &sendRequests_.back()),
                 "MPI_Send_init", communicator_);
        offset += static_cast<size_t>(chunk);
        remaining -= static_cast<size_t>(chunk);
      }
    }
  }

  void waitForSends() {
    if (sendsActive_) {
      mpiCheck(MPI_Waitall(static_cast<int>(sendRequests_.size()),
                           sendRequests_.data(), MPI_STATUSES_IGNORE),
               "MPI_Waitall(send)", communicator_);
      sendsActive_ = false;
    }
  }

  void fillPhysicalBoundaries(std::vector<double> &field) const {
    if (!hasNeighbor(XMinus)) {
      for (size_t z = 1; z <= domain_.nz; ++z)
        for (size_t y = 1; y <= domain_.ny; ++y)
          field[domain_.index(0, y, z)] = field[domain_.index(1, y, z)];
    }
    if (!hasNeighbor(XPlus)) {
      for (size_t z = 1; z <= domain_.nz; ++z)
        for (size_t y = 1; y <= domain_.ny; ++y)
          field[domain_.index(domain_.nx + 1, y, z)] =
              field[domain_.index(domain_.nx, y, z)];
    }
    if (!hasNeighbor(YMinus)) {
      for (size_t z = 1; z <= domain_.nz; ++z)
        std::memcpy(&field[domain_.index(1, 0, z)],
                    &field[domain_.index(1, 1, z)],
                    domain_.nx * sizeof(double));
    }
    if (!hasNeighbor(YPlus)) {
      for (size_t z = 1; z <= domain_.nz; ++z)
        std::memcpy(&field[domain_.index(1, domain_.ny + 1, z)],
                    &field[domain_.index(1, domain_.ny, z)],
                    domain_.nx * sizeof(double));
    }
    if (!hasNeighbor(ZMinus)) {
      for (size_t y = 1; y <= domain_.ny; ++y)
        std::memcpy(&field[domain_.index(1, y, 0)],
                    &field[domain_.index(1, y, 1)],
                    domain_.nx * sizeof(double));
    }
    if (!hasNeighbor(ZPlus)) {
      for (size_t y = 1; y <= domain_.ny; ++y)
        std::memcpy(&field[domain_.index(1, y, domain_.nz + 1)],
                    &field[domain_.index(1, y, domain_.nz)],
                    domain_.nx * sizeof(double));
    }
  }

  void pack(const std::vector<double> &field, const Direction direction,
            double *buffer) const {
    size_t offset = 0;
    if (direction == XMinus || direction == XPlus) {
      const size_t x = direction == XMinus ? 1 : domain_.nx;
      for (size_t z = 1; z <= domain_.nz; ++z)
        for (size_t y = 1; y <= domain_.ny; ++y)
          buffer[offset++] = field[domain_.index(x, y, z)];
    } else if (direction == YMinus || direction == YPlus) {
      const size_t y = direction == YMinus ? 1 : domain_.ny;
      for (size_t z = 1; z <= domain_.nz; ++z) {
        std::memcpy(buffer + offset, &field[domain_.index(1, y, z)],
                    domain_.nx * sizeof(double));
        offset += domain_.nx;
      }
    } else {
      const size_t z = direction == ZMinus ? 1 : domain_.nz;
      for (size_t y = 1; y <= domain_.ny; ++y) {
        std::memcpy(buffer + offset, &field[domain_.index(1, y, z)],
                    domain_.nx * sizeof(double));
        offset += domain_.nx;
      }
    }
  }

  void unpack(std::vector<double> &field, const Direction direction,
              const double *buffer) const {
    size_t offset = 0;
    if (direction == XMinus || direction == XPlus) {
      const size_t x = direction == XMinus ? 0 : domain_.nx + 1;
      for (size_t z = 1; z <= domain_.nz; ++z)
        for (size_t y = 1; y <= domain_.ny; ++y)
          field[domain_.index(x, y, z)] = buffer[offset++];
    } else if (direction == YMinus || direction == YPlus) {
      const size_t y = direction == YMinus ? 0 : domain_.ny + 1;
      for (size_t z = 1; z <= domain_.nz; ++z) {
        std::memcpy(&field[domain_.index(1, y, z)], buffer + offset,
                    domain_.nx * sizeof(double));
        offset += domain_.nx;
      }
    } else {
      const size_t z = direction == ZMinus ? 0 : domain_.nz + 1;
      for (size_t y = 1; y <= domain_.ny; ++y) {
        std::memcpy(&field[domain_.index(1, y, z)], buffer + offset,
                    domain_.nx * sizeof(double));
        offset += domain_.nx;
      }
    }
  }

  const Domain &domain_;
  MPI_Comm communicator_;
  std::array<int, DirectionCount> neighbors_{};
  std::array<size_t, DirectionCount> counts_{};
  std::array<size_t, DirectionCount> offsets_{};
  std::vector<double> sendBuffers_;
  std::vector<double> receiveBuffers_;
  std::vector<MPI_Request> receiveRequests_;
  std::vector<MPI_Request> sendRequests_;
  bool sendsActive_ = false;
};

struct Bounds {
  size_t xBegin;
  size_t xEnd;
  size_t yBegin;
  size_t yEnd;
  size_t zBegin;
  size_t zEnd;

  bool valid() const noexcept {
    return xBegin <= xEnd && yBegin <= yEnd && zBegin <= zEnd;
  }
};

Bounds independentBounds(const Domain &domain, const HaloExchange &halo) {
  return {
      halo.hasNeighbor(XMinus) ? 2UL : 1UL,
      halo.hasNeighbor(XPlus) ? domain.nx - 1 : domain.nx,
      halo.hasNeighbor(YMinus) ? 2UL : 1UL,
      halo.hasNeighbor(YPlus) ? domain.ny - 1 : domain.ny,
      halo.hasNeighbor(ZMinus) ? 2UL : 1UL,
      halo.hasNeighbor(ZPlus) ? domain.nz - 1 : domain.nz,
  };
}

struct PhysicalParameters {
  double dxSquared;
  double dySquared;
  double dzSquared;
  double dt;
  double eAA;
  double eBB;
  double eAB;
  double gamma;
  double diffusion;
};

void computeChemicalPotentialRange(const std::vector<double> &concentration,
                                   std::vector<double> &chemicalPotential,
                                   const Domain &domain,
                                   const PhysicalParameters &parameters,
                                   const size_t xBegin, const size_t xEnd,
                                   const size_t yBegin, const size_t yEnd,
                                   const size_t zBegin, const size_t zEnd) {
  if (xBegin > xEnd || yBegin > yEnd || zBegin > zEnd)
    return;
  const double *const c = concentration.data();
  double *const mu = chemicalPotential.data();
  const size_t row = domain.rowStride;
  const size_t plane = domain.planeStride;

  for (size_t z = zBegin; z <= zEnd; ++z) {
    for (size_t y = yBegin; y <= yEnd; ++y) {
      size_t index = domain.index(xBegin, y, z);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
      for (size_t x = xBegin; x <= xEnd; ++x, ++index) {
        const double value = c[index];
        const double cxx =
            (c[index + 1] + c[index - 1] - 2.0 * value) / parameters.dxSquared;
        const double cyy = (c[index + row] + c[index - row] - 2.0 * value) /
                           parameters.dySquared;
        const double czz = (c[index + plane] + c[index - plane] - 2.0 * value) /
                           parameters.dzSquared;
        mu[index] = 4.5 * ((value + 1.0) * parameters.eAA +
                           (value - 1.0) * parameters.eBB -
                           2.0 * value * parameters.eAB) +
                    3.0 * value + value * value * value -
                    parameters.gamma * (cxx + cyy + czz);
      }
    }
  }
}

void updateConcentrationRange(std::vector<double> &nextConcentration,
                              const std::vector<double> &concentration,
                              const std::vector<double> &chemicalPotential,
                              const Domain &domain,
                              const PhysicalParameters &parameters,
                              const size_t xBegin, const size_t xEnd,
                              const size_t yBegin, const size_t yEnd,
                              const size_t zBegin, const size_t zEnd) {
  if (xBegin > xEnd || yBegin > yEnd || zBegin > zEnd)
    return;
  double *const next = nextConcentration.data();
  const double *const current = concentration.data();
  const double *const mu = chemicalPotential.data();
  const size_t row = domain.rowStride;
  const size_t plane = domain.planeStride;

  for (size_t z = zBegin; z <= zEnd; ++z) {
    for (size_t y = yBegin; y <= yEnd; ++y) {
      size_t index = domain.index(xBegin, y, z);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
      for (size_t x = xBegin; x <= xEnd; ++x, ++index) {
        const double value = mu[index];
        const double muxx = (mu[index + 1] + mu[index - 1] - 2.0 * value) /
                            parameters.dxSquared;
        const double muyy = (mu[index + row] + mu[index - row] - 2.0 * value) /
                            parameters.dySquared;
        const double muzz =
            (mu[index + plane] + mu[index - plane] - 2.0 * value) /
            parameters.dzSquared;
        next[index] = current[index] + parameters.dt * parameters.diffusion *
                                           (muxx + muyy + muzz);
      }
    }
  }
}

template <typename RangeFunction>
void computeRemainder(const Domain &domain, const Bounds &interior,
                      RangeFunction &&computeRange) {
  if (!interior.valid()) {
    computeRange(1, domain.nx, 1, domain.ny, 1, domain.nz);
    return;
  }

  // Six disjoint slabs are the complement of the already computed interior box.
  computeRange(1, interior.xBegin - 1, 1, domain.ny, 1, domain.nz);
  computeRange(interior.xEnd + 1, domain.nx, 1, domain.ny, 1, domain.nz);
  computeRange(interior.xBegin, interior.xEnd, 1, interior.yBegin - 1, 1,
               domain.nz);
  computeRange(interior.xBegin, interior.xEnd, interior.yEnd + 1, domain.ny, 1,
               domain.nz);
  computeRange(interior.xBegin, interior.xEnd, interior.yBegin, interior.yEnd,
               1, interior.zBegin - 1);
  computeRange(interior.xBegin, interior.xEnd, interior.yBegin, interior.yEnd,
               interior.zEnd + 1, domain.nz);
}

void initializeConcentration(std::vector<double> &concentration,
                             const Domain &domain, const size_t globalNx,
                             const size_t globalNy, const size_t globalVolume) {
  for (size_t z = 1; z <= domain.nz; ++z) {
    const size_t globalZ = domain.startZ + z - 1;
    for (size_t y = 1; y <= domain.ny; ++y) {
      const size_t globalY = domain.startY + y - 1;
      for (size_t x = 1; x <= domain.nx; ++x) {
        const size_t globalX = domain.startX + x - 1;
        const size_t linearId =
            globalZ * (globalNx * globalNy) + globalY * globalNx + globalX;
        const double pseudo =
            (((linearId + 1) * static_cast<size_t>(1299709)) % globalVolume) /
            static_cast<double>(globalVolume);
        concentration[domain.index(x, y, z)] = -1.0 + 2.0 * pseudo;
      }
    }
  }
}

std::vector<double> packOwned(const std::vector<double> &field,
                              const Domain &domain) {
  std::vector<double> packed(domain.nx * domain.ny * domain.nz);
  size_t offset = 0;
  for (size_t z = 1; z <= domain.nz; ++z) {
    for (size_t y = 1; y <= domain.ny; ++y) {
      std::memcpy(packed.data() + offset, &field[domain.index(1, y, z)],
                  domain.nx * sizeof(double));
      offset += domain.nx;
    }
  }
  return packed;
}

void sendLargeBlocking(const double *buffer, size_t count,
                       const int destination, const int tag,
                       const MPI_Comm communicator) {
  constexpr size_t maximumCount =
      static_cast<size_t>(std::numeric_limits<int>::max());
  while (count != 0) {
    const int chunk = static_cast<int>(std::min(count, maximumCount));
    mpiCheck(
        MPI_Send(buffer, chunk, MPI_DOUBLE, destination, tag, communicator),
        "MPI_Send", communicator);
    buffer += chunk;
    count -= static_cast<size_t>(chunk);
  }
}

void receiveLargeBlocking(double *buffer, size_t count, const int source,
                          const int tag, const MPI_Comm communicator) {
  constexpr size_t maximumCount =
      static_cast<size_t>(std::numeric_limits<int>::max());
  while (count != 0) {
    const int chunk = static_cast<int>(std::min(count, maximumCount));
    mpiCheck(MPI_Recv(buffer, chunk, MPI_DOUBLE, source, tag, communicator,
                      MPI_STATUS_IGNORE),
             "MPI_Recv", communicator);
    buffer += chunk;
    count -= static_cast<size_t>(chunk);
  }
}

void placePackedBlock(const std::vector<double> &packed, const Domain &block,
                      const size_t globalNx, const size_t globalNy,
                      std::vector<double> &global) {
  size_t sourceOffset = 0;
  for (size_t z = 0; z < block.nz; ++z) {
    for (size_t y = 0; y < block.ny; ++y) {
      const size_t destination =
          ((block.startZ + z) * globalNy + block.startY + y) * globalNx +
          block.startX;
      std::memcpy(global.data() + destination, packed.data() + sourceOffset,
                  block.nx * sizeof(double));
      sourceOffset += block.nx;
    }
  }
}

std::vector<double> gatherGlobalField(const std::vector<double> &field,
                                      const Domain &localDomain,
                                      const std::array<size_t, 3> &global,
                                      const std::array<int, 3> &processGrid,
                                      const MPI_Comm communicator,
                                      const int rank, const int processCount) {
  constexpr int resultTag = 100;
  std::vector<double> packed = packOwned(field, localDomain);
  if (rank != 0) {
    sendLargeBlocking(packed.data(), packed.size(), 0, resultTag, communicator);
    return {};
  }

  std::vector<double> result(global[0] * global[1] * global[2]);
  placePackedBlock(packed, localDomain, global[0], global[1], result);

  for (int source = 1; source < processCount; ++source) {
    std::array<int, 3> coordinates{};
    mpiCheck(MPI_Cart_coords(communicator, source, 3, coordinates.data()),
             "MPI_Cart_coords", communicator);
    const Domain sourceDomain = makeDomain(global, processGrid, coordinates);
    packed.resize(sourceDomain.nx * sourceDomain.ny * sourceDomain.nz);
    receiveLargeBlocking(packed.data(), packed.size(), source, resultTag,
                         communicator);
    placePackedBlock(packed, sourceDomain, global[0], global[1], result);
  }
  return result;
}

bool validateResult(const std::vector<double> &field, const Domain &domain,
                    const MPI_Comm communicator, const int rank) {
  int localInvalid = 0;
  double localMinimum = std::numeric_limits<double>::infinity();
  double localMaximum = -std::numeric_limits<double>::infinity();
  for (size_t z = 1; z <= domain.nz; ++z) {
    for (size_t y = 1; y <= domain.ny; ++y) {
      for (size_t x = 1; x <= domain.nx; ++x) {
        const double value = field[domain.index(x, y, z)];
        localInvalid |= !std::isfinite(value);
        localMinimum = std::min(localMinimum, value);
        localMaximum = std::max(localMaximum, value);
      }
    }
  }

  int invalid = 0;
  double globalMinimum = 0.0;
  double globalMaximum = 0.0;
  mpiCheck(
      MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX, communicator),
      "MPI_Allreduce(validation)", communicator);
  mpiCheck(MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN,
                         communicator),
           "MPI_Allreduce(minimum)", communicator);
  mpiCheck(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX,
                         communicator),
           "MPI_Allreduce(maximum)", communicator);

  if (rank == 0) {
    if (invalid != 0)
      std::printf("Validation failed: found NaN or Inf value\n");
    std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum,
                globalMaximum);
    if (invalid == 0 && (globalMaximum > 10.0 || globalMinimum < -10.0))
      std::printf("Validation failed: values out of expected range\n");
  }
  return invalid == 0 && globalMaximum <= 10.0 && globalMinimum >= -10.0;
}

} // namespace

int main(int argc, char **argv) {
  mpiCheck(MPI_Init(&argc, &argv), "MPI_Init");

  int worldRank = 0;
  int processCount = 1;
  mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
  mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &processCount), "MPI_Comm_size");

  size_t nx = 64;
  size_t ny = 0;
  size_t nz = 0;
  int iterations = 20;
  bool validate = false;
  bool printResults = false;
  bool argumentsValid = true;
  bool showHelp = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
      argumentsValid &= parseSize(argv[++i], nx);
    } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
      argumentsValid &= parseSize(argv[++i], ny);
    } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
      argumentsValid &= parseSize(argv[++i], nz);
    } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      argumentsValid &= parseIterations(argv[++i], iterations);
    } else if (std::strcmp(argv[i], "-v") == 0) {
      validate = true;
    } else if (std::strcmp(argv[i], "-r") == 0) {
      printResults = true;
    } else if (std::strcmp(argv[i], "-h") == 0) {
      showHelp = true;
    } else {
      argumentsValid = false;
      if (worldRank == 0)
        std::printf("Unknown or incomplete option: %s\n", argv[i]);
    }
  }

  if (showHelp || !argumentsValid) {
    if (worldRank == 0)
      printUsage(argv[0]);
    MPI_Finalize();
    return argumentsValid ? 0 : 1;
  }
  if (ny == 0)
    ny = nx;
  if (nz == 0)
    nz = nx;

  size_t xy = 0;
  size_t gridSize = 0;
  if (!checkedMultiply(nx, ny, xy) || !checkedMultiply(xy, nz, gridSize) ||
      gridSize < static_cast<size_t>(processCount)) {
    if (worldRank == 0) {
      std::fprintf(stderr, "Grid dimensions overflow or there are more MPI "
                           "ranks than grid cells.\n");
    }
    MPI_Finalize();
    return 1;
  }

  const std::array<size_t, 3> global = {nx, ny, nz};
  const std::array<int, 3> processGrid =
      chooseProcessGrid(processCount, nx, ny, nz);
  if (processGrid[0] == 0) {
    if (worldRank == 0)
      std::fprintf(stderr, "Unable to form a nonempty MPI process grid.\n");
    MPI_Finalize();
    return 1;
  }

  const std::array<int, 3> periods = {0, 0, 0};
  MPI_Comm cartesian = MPI_COMM_NULL;
  mpiCheck(MPI_Cart_create(MPI_COMM_WORLD, 3, processGrid.data(),
                           periods.data(), 1, &cartesian),
           "MPI_Cart_create");
  int rank = 0;
  std::array<int, 3> coordinates{};
  mpiCheck(MPI_Comm_rank(cartesian, &rank), "MPI_Comm_rank(cartesian)",
           cartesian);
  mpiCheck(MPI_Cart_coords(cartesian, rank, 3, coordinates.data()),
           "MPI_Cart_coords", cartesian);

  const Domain domain = makeDomain(global, processGrid, coordinates);
  size_t ghostPlane = 0;
  size_t allocationSize = 0;
  if (!checkedMultiply(domain.nx + 2, domain.ny + 2, ghostPlane) ||
      !checkedMultiply(ghostPlane, domain.nz + 2, allocationSize)) {
    if (rank == 0)
      std::fprintf(stderr, "Local allocation size overflow.\n");
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("MPI process grid: %d x %d x %d (%d ranks)\n", processGrid[0],
                processGrid[1], processGrid[2], processCount);
  }

  const PhysicalParameters parameters = {
      1.0 * 1.0,    1.0 * 1.0,   1.0 * 1.0, 0.01, -(2.0 / 9.0),
      -(2.0 / 9.0), (2.0 / 9.0), 0.5,       1.0,
  };

  bool valid = true;
  {
    std::vector<double> concentration(allocationSize);
    std::vector<double> nextConcentration(allocationSize);
    std::vector<double> chemicalPotential(allocationSize);
    HaloExchange halo(domain, cartesian);
    const Bounds interior = independentBounds(domain, halo);

    if (rank == 0)
      std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, domain, nx, ny, gridSize);

    if (rank == 0)
      std::printf("Running Cahn-Hilliard simulation...\n");
    mpiCheck(MPI_Barrier(cartesian), "MPI_Barrier", cartesian);
    const double start = MPI_Wtime();

    for (int timeStep = 0; timeStep < iterations; ++timeStep) {
      halo.start(concentration);
      if (interior.valid()) {
        computeChemicalPotentialRange(
            concentration, chemicalPotential, domain, parameters,
            interior.xBegin, interior.xEnd, interior.yBegin, interior.yEnd,
            interior.zBegin, interior.zEnd);
      }
      halo.finish(concentration);
      computeRemainder(
          domain, interior,
          [&](const size_t xBegin, const size_t xEnd, const size_t yBegin,
              const size_t yEnd, const size_t zBegin, const size_t zEnd) {
            computeChemicalPotentialRange(concentration, chemicalPotential,
                                          domain, parameters, xBegin, xEnd,
                                          yBegin, yEnd, zBegin, zEnd);
          });

      halo.start(chemicalPotential);
      if (interior.valid()) {
        updateConcentrationRange(
            nextConcentration, concentration, chemicalPotential, domain,
            parameters, interior.xBegin, interior.xEnd, interior.yBegin,
            interior.yEnd, interior.zBegin, interior.zEnd);
      }
      halo.finish(chemicalPotential);
      computeRemainder(
          domain, interior,
          [&](const size_t xBegin, const size_t xEnd, const size_t yBegin,
              const size_t yEnd, const size_t zBegin, const size_t zEnd) {
            updateConcentrationRange(nextConcentration, concentration,
                                     chemicalPotential, domain, parameters,
                                     xBegin, xEnd, yBegin, yEnd, zBegin, zEnd);
          });
      std::swap(concentration, nextConcentration);
    }

    // Include completion of the final outgoing halo in the measured interval.
    halo.complete();

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpiCheck(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        cartesian),
             "MPI_Reduce(timing)", cartesian);

    if (rank == 0) {
      const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
      const double cellUpdates =
          static_cast<double>(gridSize) * static_cast<double>(iterations);
      const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
      std::printf("Computation time: %lld ms\n", milliseconds);
      std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
      std::vector<double> globalResult =
          gatherGlobalField(concentration, domain, global, processGrid,
                            cartesian, rank, processCount);
      if (rank == 0)
        print_results(globalResult, "Concentration");
    }

    if (validate) {
      if (rank == 0)
        std::printf("Validating result...\n");
      valid = validateResult(concentration, domain, cartesian, rank);
      if (rank == 0)
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
  } // Destroy persistent MPI requests before freeing their communicator.

  mpiCheck(MPI_Comm_free(&cartesian), "MPI_Comm_free");
  mpiCheck(MPI_Finalize(), "MPI_Finalize");
  return valid ? 0 : 1;
}
