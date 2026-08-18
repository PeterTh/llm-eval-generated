#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// A local field contains one ghost cell on every side.  The x coordinate is
// contiguous, matching the global row-major layout used by the original code.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t strideX, const size_t strideY) noexcept {
    return (z * strideY + y) * strideX + x;
}

bool checkedMultiply(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Pick a process grid that minimizes the ideal surface-to-volume ratio.  The
// explicit search also constrains every process-grid dimension to the global
// grid, which MPI_Dims_create does not guarantee for small benchmark grids.
bool chooseProcessGrid(const size_t nx, const size_t ny, const size_t nz,
                       const int processCount, int dimensions[3]) {
    double bestScore = std::numeric_limits<double>::infinity();
    int best[3] = {0, 0, 0};

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<size_t>(px) > nx) {
            continue;
        }
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<size_t>(py) > ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > nz) {
                continue;
            }

            const double score = static_cast<double>(px) / static_cast<double>(nx) +
                                 static_cast<double>(py) / static_cast<double>(ny) +
                                 static_cast<double>(pz) / static_cast<double>(nz);
            const bool better = score < bestScore - 1.0e-15;
            const bool equalAndMoreContiguous =
                std::abs(score - bestScore) <= 1.0e-15 &&
                (pz > best[2] || (pz == best[2] && py > best[1]));
            if (better || equalAndMoreContiguous) {
                bestScore = score;
                best[0] = px;
                best[1] = py;
                best[2] = pz;
            }
        }
    }

    if (best[0] == 0) {
        return false;
    }
    dimensions[0] = best[0];
    dimensions[1] = best[1];
    dimensions[2] = best[2];
    return true;
}

void partitionDimension(const size_t globalExtent, const int processCoordinate,
                        const int processExtent, size_t& start, size_t& extent) {
    const size_t base = globalExtent / static_cast<size_t>(processExtent);
    const size_t remainder = globalExtent % static_cast<size_t>(processExtent);
    const size_t coordinate = static_cast<size_t>(processCoordinate);
    start = coordinate * base + std::min(coordinate, remainder);
    extent = base + (coordinate < remainder ? 1 : 0);
}

struct Domain {
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int size = 0;
    int dimensions[3] = {1, 1, 1};
    int coordinates[3] = {0, 0, 0};
    int minusX = MPI_PROC_NULL;
    int plusX = MPI_PROC_NULL;
    int minusY = MPI_PROC_NULL;
    int plusY = MPI_PROC_NULL;
    int minusZ = MPI_PROC_NULL;
    int plusZ = MPI_PROC_NULL;

    size_t globalX = 0;
    size_t globalY = 0;
    size_t globalZ = 0;
    size_t startX = 0;
    size_t startY = 0;
    size_t startZ = 0;
    size_t localX = 0;
    size_t localY = 0;
    size_t localZ = 0;

    Domain(const size_t nx, const size_t ny, const size_t nz, const int processGrid[3])
        : globalX(nx), globalY(ny), globalZ(nz) {
        dimensions[0] = processGrid[0];
        dimensions[1] = processGrid[1];
        dimensions[2] = processGrid[2];
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, periods, 0, &communicator);
        MPI_Comm_rank(communicator, &rank);
        MPI_Comm_size(communicator, &size);
        MPI_Cart_coords(communicator, rank, 3, coordinates);

        partitionDimension(globalX, coordinates[0], dimensions[0], startX, localX);
        partitionDimension(globalY, coordinates[1], dimensions[1], startY, localY);
        partitionDimension(globalZ, coordinates[2], dimensions[2], startZ, localZ);

        MPI_Cart_shift(communicator, 0, 1, &minusX, &plusX);
        MPI_Cart_shift(communicator, 1, 1, &minusY, &plusY);
        MPI_Cart_shift(communicator, 2, 1, &minusZ, &plusZ);
    }

    ~Domain() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized && communicator != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator);
        }
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;
};

class HaloExchange {
  public:
    using Requests = std::array<MPI_Request, 12>;

    explicit HaloExchange(const Domain& domain)
        : domain_(domain), strideX_(domain.localX + 2), strideY_(domain.localY + 2) {
        const int localX = static_cast<int>(domain.localX);
        const int localY = static_cast<int>(domain.localY);
        const int localZ = static_cast<int>(domain.localZ);
        const int strideX = static_cast<int>(strideX_);
        const MPI_Aint planeBytes = static_cast<MPI_Aint>(strideX_ * strideY_ * sizeof(double));

        MPI_Datatype xRows = MPI_DATATYPE_NULL;
        MPI_Datatype yRows = MPI_DATATYPE_NULL;
        MPI_Type_vector(localY, 1, strideX, MPI_DOUBLE, &xRows);
        MPI_Type_create_hvector(localZ, 1, planeBytes, xRows, &xFace_);
        MPI_Type_contiguous(localX, MPI_DOUBLE, &yRows);
        MPI_Type_create_hvector(localZ, 1, planeBytes, yRows, &yFace_);
        MPI_Type_vector(localY, localX, strideX, MPI_DOUBLE, &zFace_);
        MPI_Type_commit(&xFace_);
        MPI_Type_commit(&yFace_);
        MPI_Type_commit(&zFace_);
        MPI_Type_free(&xRows);
        MPI_Type_free(&yRows);
    }

    ~HaloExchange() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            if (xFace_ != MPI_DATATYPE_NULL) {
                MPI_Type_free(&xFace_);
            }
            if (yFace_ != MPI_DATATYPE_NULL) {
                MPI_Type_free(&yFace_);
            }
            if (zFace_ != MPI_DATATYPE_NULL) {
                MPI_Type_free(&zFace_);
            }
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void begin(std::vector<double>& field, Requests& requests, int& requestCount) const {
        setPhysicalBoundaries(field);
        requestCount = 0;
        double* data = field.data();

        postReceive(data + index(0, 1, 1), xFace_, domain_.minusX, 101, requests, requestCount);
        postReceive(data + index(domain_.localX + 1, 1, 1), xFace_, domain_.plusX, 101,
                    requests, requestCount);
        postReceive(data + index(1, 0, 1), yFace_, domain_.minusY, 102, requests, requestCount);
        postReceive(data + index(1, domain_.localY + 1, 1), yFace_, domain_.plusY, 102,
                    requests, requestCount);
        postReceive(data + index(1, 1, 0), zFace_, domain_.minusZ, 103, requests, requestCount);
        postReceive(data + index(1, 1, domain_.localZ + 1), zFace_, domain_.plusZ, 103,
                    requests, requestCount);

        postSend(data + index(1, 1, 1), xFace_, domain_.minusX, 101, requests, requestCount);
        postSend(data + index(domain_.localX, 1, 1), xFace_, domain_.plusX, 101,
                 requests, requestCount);
        postSend(data + index(1, 1, 1), yFace_, domain_.minusY, 102, requests, requestCount);
        postSend(data + index(1, domain_.localY, 1), yFace_, domain_.plusY, 102,
                 requests, requestCount);
        postSend(data + index(1, 1, 1), zFace_, domain_.minusZ, 103, requests, requestCount);
        postSend(data + index(1, 1, domain_.localZ), zFace_, domain_.plusZ, 103,
                 requests, requestCount);
    }

    static void finish(Requests& requests, const int requestCount) {
        if (requestCount > 0) {
            MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return idx3(x, y, z, strideX_, strideY_);
    }

  private:
    const Domain& domain_;
    const size_t strideX_;
    const size_t strideY_;
    MPI_Datatype xFace_ = MPI_DATATYPE_NULL;
    MPI_Datatype yFace_ = MPI_DATATYPE_NULL;
    MPI_Datatype zFace_ = MPI_DATATYPE_NULL;

    void setPhysicalBoundaries(std::vector<double>& field) const {
        if (domain_.minusX == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.localZ; ++z) {
                for (size_t y = 1; y <= domain_.localY; ++y) {
                    field[index(0, y, z)] = field[index(1, y, z)];
                }
            }
        }
        if (domain_.plusX == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.localZ; ++z) {
                for (size_t y = 1; y <= domain_.localY; ++y) {
                    field[index(domain_.localX + 1, y, z)] =
                        field[index(domain_.localX, y, z)];
                }
            }
        }
        if (domain_.minusY == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.localZ; ++z) {
                for (size_t x = 1; x <= domain_.localX; ++x) {
                    field[index(x, 0, z)] = field[index(x, 1, z)];
                }
            }
        }
        if (domain_.plusY == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.localZ; ++z) {
                for (size_t x = 1; x <= domain_.localX; ++x) {
                    field[index(x, domain_.localY + 1, z)] =
                        field[index(x, domain_.localY, z)];
                }
            }
        }
        if (domain_.minusZ == MPI_PROC_NULL) {
            for (size_t y = 1; y <= domain_.localY; ++y) {
                for (size_t x = 1; x <= domain_.localX; ++x) {
                    field[index(x, y, 0)] = field[index(x, y, 1)];
                }
            }
        }
        if (domain_.plusZ == MPI_PROC_NULL) {
            for (size_t y = 1; y <= domain_.localY; ++y) {
                for (size_t x = 1; x <= domain_.localX; ++x) {
                    field[index(x, y, domain_.localZ + 1)] =
                        field[index(x, y, domain_.localZ)];
                }
            }
        }
    }

    void postReceive(double* address, const MPI_Datatype datatype, const int source,
                     const int tag, Requests& requests, int& requestCount) const {
        if (source != MPI_PROC_NULL) {
            MPI_Irecv(address, 1, datatype, source, tag, domain_.communicator,
                      &requests[requestCount++]);
        }
    }

    void postSend(double* address, const MPI_Datatype datatype, const int destination,
                  const int tag, Requests& requests, int& requestCount) const {
        if (destination != MPI_PROC_NULL) {
            MPI_Isend(address, 1, datatype, destination, tag, domain_.communicator,
                      &requests[requestCount++]);
        }
    }
};

inline double laplacianAt(const std::vector<double>& field, const size_t center,
                          const size_t strideX, const size_t strideY,
                          const double dx2, const double dy2, const double dz2) noexcept {
    const size_t strideZ = strideX * strideY;
    const double cxx = (field[center + 1] + field[center - 1] - 2.0 * field[center]) / dx2;
    const double cyy = (field[center + strideX] + field[center - strideX] -
                        2.0 * field[center]) /
                       dy2;
    const double czz = (field[center + strideZ] + field[center - strideZ] -
                        2.0 * field[center]) /
                       dz2;
    return cxx + cyy + czz;
}

inline void computeChemicalPotentialCell(const std::vector<double>& c, std::vector<double>& mu,
                                         const size_t center, const size_t strideX,
                                         const size_t strideY, const double dx2, const double dy2,
                                         const double dz2, const double gamma, const double e_AA,
                                         const double e_BB, const double e_AB) noexcept {
    const double cv = c[center];
    mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                 3.0 * cv + cv * cv * cv -
                 gamma * laplacianAt(c, center, strideX, strideY, dx2, dy2, dz2);
}

void computeChemicalPotentialCore(const std::vector<double>& c, std::vector<double>& mu,
                                  const size_t localX, const size_t localY, const size_t localZ,
                                  const size_t strideX, const size_t strideY, const double dx2,
                                  const double dy2, const double dz2, const double gamma,
                                  const double e_AA, const double e_BB, const double e_AB) {
    if (localX <= 2 || localY <= 2 || localZ <= 2) {
        return;
    }
    for (size_t z = 2; z < localZ; ++z) {
        for (size_t y = 2; y < localY; ++y) {
            for (size_t x = 2; x < localX; ++x) {
                computeChemicalPotentialCell(c, mu, idx3(x, y, z, strideX, strideY), strideX,
                                              strideY, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
            }
        }
    }
}

void computeChemicalPotentialBoundary(const std::vector<double>& c, std::vector<double>& mu,
                                      const size_t localX, const size_t localY,
                                      const size_t localZ, const size_t strideX,
                                      const size_t strideY, const double dx2, const double dy2,
                                      const double dz2, const double gamma, const double e_AA,
                                      const double e_BB, const double e_AB) {
    auto compute = [&](const size_t x, const size_t y, const size_t z) {
        computeChemicalPotentialCell(c, mu, idx3(x, y, z, strideX, strideY), strideX, strideY,
                                      dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
    };

    // x faces, including all y/z values
    for (size_t z = 1; z <= localZ; ++z) {
        for (size_t y = 1; y <= localY; ++y) {
            compute(1, y, z);
            if (localX > 1) {
                compute(localX, y, z);
            }
        }
    }
    // y faces, restricted to the x-interior so edges are not repeated
    for (size_t z = 1; z <= localZ; ++z) {
        for (size_t x = 2; x < localX; ++x) {
            compute(x, 1, z);
            if (localY > 1) {
                compute(x, localY, z);
            }
        }
    }
    // z faces, restricted to the x/y-interior
    for (size_t y = 2; y < localY; ++y) {
        for (size_t x = 2; x < localX; ++x) {
            compute(x, y, 1);
            if (localZ > 1) {
                compute(x, y, localZ);
            }
        }
    }
}

inline void computeUpdateCell(std::vector<double>& cnew, const std::vector<double>& cold,
                              const std::vector<double>& mu, const size_t center,
                              const size_t strideX, const size_t strideY, const double dx2,
                              const double dy2, const double dz2, const double D,
                              const double dt) noexcept {
    cnew[center] = cold[center] + dt * D *
                                      laplacianAt(mu, center, strideX, strideY, dx2, dy2, dz2);
}

void cahnHilliardUpdateCore(std::vector<double>& cnew, const std::vector<double>& cold,
                            const std::vector<double>& mu, const size_t localX,
                            const size_t localY, const size_t localZ, const size_t strideX,
                            const size_t strideY, const double dx2, const double dy2,
                            const double dz2, const double D, const double dt) {
    if (localX <= 2 || localY <= 2 || localZ <= 2) {
        return;
    }
    for (size_t z = 2; z < localZ; ++z) {
        for (size_t y = 2; y < localY; ++y) {
            for (size_t x = 2; x < localX; ++x) {
                computeUpdateCell(cnew, cold, mu, idx3(x, y, z, strideX, strideY), strideX,
                                  strideY, dx2, dy2, dz2, D, dt);
            }
        }
    }
}

void cahnHilliardUpdateBoundary(std::vector<double>& cnew, const std::vector<double>& cold,
                                const std::vector<double>& mu, const size_t localX,
                                const size_t localY, const size_t localZ, const size_t strideX,
                                const size_t strideY, const double dx2, const double dy2,
                                const double dz2, const double D, const double dt) {
    auto compute = [&](const size_t x, const size_t y, const size_t z) {
        computeUpdateCell(cnew, cold, mu, idx3(x, y, z, strideX, strideY), strideX, strideY, dx2,
                          dy2, dz2, D, dt);
    };

    for (size_t z = 1; z <= localZ; ++z) {
        for (size_t y = 1; y <= localY; ++y) {
            compute(1, y, z);
            if (localX > 1) {
                compute(localX, y, z);
            }
        }
    }
    for (size_t z = 1; z <= localZ; ++z) {
        for (size_t x = 2; x < localX; ++x) {
            compute(x, 1, z);
            if (localY > 1) {
                compute(x, localY, z);
            }
        }
    }
    for (size_t y = 2; y < localY; ++y) {
        for (size_t x = 2; x < localX; ++x) {
            compute(x, y, 1);
            if (localZ > 1) {
                compute(x, y, localZ);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const Domain& domain) {
    size_t volume = 0;
    size_t plane = 0;
    if (!checkedMultiply(nx, ny, plane) || !checkedMultiply(plane, nz, volume)) {
        return;
    }

    const size_t strideX = domain.localX + 2;
    const size_t strideY = domain.localY + 2;
    for (size_t z = 1; z <= domain.localZ; ++z) {
        const size_t globalZ = domain.startZ + z - 1;
        for (size_t y = 1; y <= domain.localY; ++y) {
            const size_t globalY = domain.startY + y - 1;
            for (size_t x = 1; x <= domain.localX; ++x) {
                const size_t globalX = domain.startX + x - 1;
                const size_t linearId = globalZ * plane + globalY * nx + globalX;
                const size_t pseudoInteger = ((linearId + 1) * static_cast<size_t>(1299709)) % volume;
                const double pseudo = static_cast<double>(pseudoInteger) /
                                      static_cast<double>(volume);
                c[idx3(x, y, z, strideX, strideY)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const Domain& domain) {
    const size_t strideX = domain.localX + 2;
    const size_t strideY = domain.localY + 2;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localNonFinite = 0;

    for (size_t z = 1; z <= domain.localZ; ++z) {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                const double value = c[idx3(x, y, z, strideX, strideY)];
                if (std::isnan(value) || std::isinf(value)) {
                    localNonFinite = 1;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    int nonFinite = 0;
    double minValue = 0.0;
    double maxValue = 0.0;
    MPI_Allreduce(&localNonFinite, &nonFinite, 1, MPI_INT, MPI_MAX, domain.communicator);
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, domain.communicator);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, domain.communicator);

    if (domain.rank == 0) {
        if (nonFinite != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
        }
    }

    return nonFinite == 0 && maxValue <= 10.0 && minValue >= -10.0;
}

void appendPackedInterior(const std::vector<double>& field, const Domain& domain,
                         std::vector<double>& packed) {
    const size_t strideX = domain.localX + 2;
    const size_t strideY = domain.localY + 2;
    packed.resize(domain.localX * domain.localY * domain.localZ);
    size_t output = 0;
    for (size_t z = 1; z <= domain.localZ; ++z) {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                packed[output++] = field[idx3(x, y, z, strideX, strideY)];
            }
        }
    }
}

void placePackedRange(const double* packed, const size_t packedOffset, const size_t count,
                      const unsigned long long* block, const size_t nx, const size_t ny,
                      std::vector<double>& global) {
    const size_t startX = static_cast<size_t>(block[0]);
    const size_t startY = static_cast<size_t>(block[1]);
    const size_t startZ = static_cast<size_t>(block[2]);
    const size_t localX = static_cast<size_t>(block[3]);
    const size_t localY = static_cast<size_t>(block[4]);
    const size_t plane = localX * localY;

    for (size_t n = 0; n < count; ++n) {
        const size_t localLinear = packedOffset + n;
        const size_t z = localLinear / plane;
        const size_t rem = localLinear - z * plane;
        const size_t y = rem / localX;
        const size_t x = rem - y * localX;
        global[idx3(startX + x, startY + y, startZ + z, nx, ny)] = packed[n];
    }
}

std::vector<double> gatherGlobalResult(const std::vector<double>& localField,
                                       const Domain& domain, const size_t nx, const size_t ny,
                                       const size_t nz) {
    std::vector<double> packed;
    appendPackedInterior(localField, domain, packed);

    const unsigned long long localBlock[7] = {
        static_cast<unsigned long long>(domain.startX),
        static_cast<unsigned long long>(domain.startY),
        static_cast<unsigned long long>(domain.startZ),
        static_cast<unsigned long long>(domain.localX),
        static_cast<unsigned long long>(domain.localY),
        static_cast<unsigned long long>(domain.localZ),
        static_cast<unsigned long long>(packed.size())};
    std::vector<unsigned long long> blocks;
    if (domain.rank == 0) {
        blocks.resize(static_cast<size_t>(domain.size) * 7);
    }
    MPI_Gather(localBlock, 7, MPI_UNSIGNED_LONG_LONG,
               domain.rank == 0 ? blocks.data() : nullptr, 7, MPI_UNSIGNED_LONG_LONG, 0,
               domain.communicator);

    std::vector<double> global;
    if (domain.rank == 0) {
        global.resize(nx * ny * nz);
        const size_t chunkLimit = 1U << 20;
        std::vector<double> receiveBuffer(chunkLimit);

        for (int source = 0; source < domain.size; ++source) {
            const unsigned long long* block = blocks.data() + static_cast<size_t>(source) * 7;
            const size_t blockCount = static_cast<size_t>(block[6]);
            if (source == 0) {
                placePackedRange(packed.data(), 0, blockCount, block, nx, ny, global);
                continue;
            }
            size_t offset = 0;
            while (offset < blockCount) {
                const size_t count = std::min(chunkLimit, blockCount - offset);
                MPI_Recv(receiveBuffer.data(), static_cast<int>(count), MPI_DOUBLE, source, 701,
                         domain.communicator, MPI_STATUS_IGNORE);
                placePackedRange(receiveBuffer.data(), offset, count, block, nx, ny, global);
                offset += count;
            }
        }
    } else {
        const size_t chunkLimit = 1U << 20;
        size_t offset = 0;
        while (offset < packed.size()) {
            const size_t count = std::min(chunkLimit, packed.size() - offset);
            MPI_Send(packed.data() + offset, static_cast<int>(count), MPI_DOUBLE, 0, 701,
                     domain.communicator);
            offset += count;
        }
    }
    return global;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], nx) || parseError;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], ny) || parseError;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseError = !parseSize(argv[++i], nz) || parseError;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parseError = !parseIterations(argv[++i], iterations) || parseError;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (worldRank == 0) {
            if (parseError) {
                std::printf("Invalid command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t planeSize = 0;
    size_t gridSize = 0;
    const bool validGrid = checkedMultiply(nx, ny, planeSize) &&
                           checkedMultiply(planeSize, nz, gridSize);
    int processGrid[3] = {0, 0, 0};
    const bool validProcessGrid = validGrid &&
                                  chooseProcessGrid(nx, ny, nz, worldSize, processGrid);
    if (!validProcessGrid) {
        if (worldRank == 0) {
            std::printf("Unable to decompose the grid across %d MPI processes\n", worldSize);
        }
        MPI_Finalize();
        return 1;
    }

    Domain domain(nx, ny, nz, processGrid);
    const size_t localStrideX = domain.localX + 2;
    const size_t localStrideY = domain.localY + 2;
    size_t localStorage = 0;
    if (!checkedMultiply(localStrideX, localStrideY, localStorage) ||
        !checkedMultiply(localStorage, domain.localZ + 2, localStorage)) {
        if (worldRank == 0) {
            std::printf("Local grid allocation size overflow\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI processes: %d\n", worldSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    std::vector<double> cold(localStorage);
    std::vector<double> cnew(localStorage);
    std::vector<double> mu(localStorage);

    if (worldRank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, nz, domain);

    HaloExchange halo(domain);
    HaloExchange::Requests requests;
    int requestCount = 0;

    if (worldRank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(domain.communicator);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration, compute the independent core while it is in flight,
        // then finish the boundary cells once all neighboring faces have arrived.
        halo.begin(cold, requests, requestCount);
        computeChemicalPotentialCore(cold, mu, domain.localX, domain.localY, domain.localZ,
                                     localStrideX, localStrideY, dx2, dy2, dz2, gamma, e_AA, e_BB,
                                     e_AB);
        HaloExchange::finish(requests, requestCount);
        computeChemicalPotentialBoundary(cold, mu, domain.localX, domain.localY, domain.localZ,
                                         localStrideX, localStrideY, dx2, dy2, dz2, gamma, e_AA,
                                         e_BB, e_AB);

        // Exchange chemical potential, overlap it with the independent update core,
        // and then update the boundary cells.
        halo.begin(mu, requests, requestCount);
        cahnHilliardUpdateCore(cnew, cold, mu, domain.localX, domain.localY, domain.localZ,
                               localStrideX, localStrideY, dx2, dy2, dz2, D, dt);
        HaloExchange::finish(requests, requestCount);
        cahnHilliardUpdateBoundary(cnew, cold, mu, domain.localX, domain.localY, domain.localZ,
                                   localStrideX, localStrideY, dx2, dy2, dz2, D, dt);
        std::swap(cold, cnew);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.communicator);

    if (worldRank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const double safeElapsed = std::max(elapsed, std::numeric_limits<double>::min());
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / safeElapsed / 1.0e6;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> globalResult = gatherGlobalResult(cold, domain, nx, ny, nz);
        if (worldRank == 0) {
            print_results(globalResult, "Concentration");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (worldRank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateResult(cold, domain);
        if (worldRank == 0) {
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
