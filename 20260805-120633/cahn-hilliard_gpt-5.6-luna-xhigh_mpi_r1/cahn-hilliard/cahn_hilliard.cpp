#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// The global field is split into rectangular blocks.  Each local field has a
// one-cell halo in every direction; the halo is exchanged before each stencil
// evaluation.  The array layout remains x-fastest, as in the original code.
enum Direction : int {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5,
};

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

struct Domain {
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};
    int neighbors[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                        MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t globalX = 0;
    size_t globalY = 0;
    size_t globalZ = 0;
    size_t startX = 0;
    size_t startY = 0;
    size_t startZ = 0;
    size_t localX = 0;
    size_t localY = 0;
    size_t localZ = 0;

    size_t strideX = 0;
    size_t strideY = 0;
    size_t strideZ = 0;

    MPI_Datatype sendTypes[6] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                 MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                 MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype receiveTypes[6] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                    MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                    MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
};

struct HaloExchange {
    MPI_Request requests[12] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int count = 0;
};

inline size_t localIndex(const Domain& domain, const size_t x, const size_t y,
                         const size_t z) noexcept {
    return idx3(x, y, z, domain.strideX, domain.strideY);
}

std::array<int, 3> chooseProcessGrid(const int processCount,
                                     const std::array<size_t, 3>& global) {
    // Enumerate factor triples in sorted order.  This is much cheaper than a
    // cubic search and lets us honor thin domains where MPI_Dims_create may
    // otherwise choose more partitions than there are cells in an axis.
    double bestScore = std::numeric_limits<double>::infinity();
    std::array<int, 3> best = {0, 0, 0};
    constexpr int permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
    };

    for (int first = 1;
         static_cast<long long>(first) * first * first <= processCount;
         ++first) {
        if (processCount % first != 0) {
            continue;
        }
        const int remainder = processCount / first;
        for (int second = first;
             static_cast<long long>(second) * second <= remainder;
             ++second) {
            if (remainder % second != 0) {
                continue;
            }
            const int third = remainder / second;
            const int factors[3] = {first, second, third};

            for (const auto& permutation : permutations) {
                const std::array<int, 3> candidate = {
                    factors[permutation[0]], factors[permutation[1]],
                    factors[permutation[2]]};
                if (static_cast<size_t>(candidate[0]) > global[0] ||
                    static_cast<size_t>(candidate[1]) > global[1] ||
                    static_cast<size_t>(candidate[2]) > global[2]) {
                    continue;
                }

                const double localX = static_cast<double>(global[0]) / candidate[0];
                const double localY = static_cast<double>(global[1]) / candidate[1];
                const double localZ = static_cast<double>(global[2]) / candidate[2];
                // Minimize the surface-to-volume ratio of a local block.
                const double score = 1.0 / localX + 1.0 / localY + 1.0 / localZ;
                if (score < bestScore) {
                    bestScore = score;
                    best = candidate;
                }
            }
        }
    }
    return best;
}

size_t blockStart(const size_t globalSize, const int partitions,
                  const int coordinate) noexcept {
    const size_t base = globalSize / static_cast<size_t>(partitions);
    const size_t remainder = globalSize % static_cast<size_t>(partitions);
    return static_cast<size_t>(coordinate) * base +
           std::min(static_cast<size_t>(coordinate), remainder);
}

size_t blockSize(const size_t globalSize, const int partitions,
                 const int coordinate) noexcept {
    const size_t base = globalSize / static_cast<size_t>(partitions);
    const size_t remainder = globalSize % static_cast<size_t>(partitions);
    return base + (static_cast<size_t>(coordinate) < remainder ? 1 : 0);
}

void createFaceDatatype(const Domain& domain, const int direction,
                        const bool receive, MPI_Datatype& datatype) {
    const int sizes[3] = {
        static_cast<int>(domain.localZ + 2),
        static_cast<int>(domain.localY + 2),
        static_cast<int>(domain.localX + 2),
    };
    int subsizes[3] = {0, 0, 0};
    int starts[3] = {1, 1, 1};

    switch (direction) {
        case XMinus:
        case XPlus:
            subsizes[0] = static_cast<int>(domain.localZ);
            subsizes[1] = static_cast<int>(domain.localY);
            subsizes[2] = 1;
            starts[2] = receive
                            ? (direction == XMinus ? 0 : static_cast<int>(domain.localX + 1))
                            : (direction == XMinus ? 1 : static_cast<int>(domain.localX));
            break;
        case YMinus:
        case YPlus:
            subsizes[0] = static_cast<int>(domain.localZ);
            subsizes[1] = 1;
            subsizes[2] = static_cast<int>(domain.localX);
            starts[1] = receive
                            ? (direction == YMinus ? 0 : static_cast<int>(domain.localY + 1))
                            : (direction == YMinus ? 1 : static_cast<int>(domain.localY));
            break;
        case ZMinus:
        case ZPlus:
            subsizes[0] = 1;
            subsizes[1] = static_cast<int>(domain.localY);
            subsizes[2] = static_cast<int>(domain.localX);
            starts[0] = receive
                            ? (direction == ZMinus ? 0 : static_cast<int>(domain.localZ + 1))
                            : (direction == ZMinus ? 1 : static_cast<int>(domain.localZ));
            break;
    }

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &datatype);
    MPI_Type_commit(&datatype);
}

bool initializeDomain(Domain& domain, MPI_Comm world,
                      const std::array<size_t, 3>& global,
                      const std::array<int, 3>& processGrid) {
    int periods[3] = {0, 0, 0};
    int dimensions[3] = {processGrid[0], processGrid[1], processGrid[2]};
    MPI_Cart_create(world, 3, dimensions, periods, 0, &domain.communicator);
    if (domain.communicator == MPI_COMM_NULL) {
        return false;
    }

    MPI_Comm_rank(domain.communicator, &domain.rank);
    MPI_Comm_size(domain.communicator, &domain.size);
    MPI_Cart_get(domain.communicator, 3, domain.dims, periods, domain.coords);
    MPI_Cart_shift(domain.communicator, 0, 1, &domain.neighbors[XMinus],
                   &domain.neighbors[XPlus]);
    MPI_Cart_shift(domain.communicator, 1, 1, &domain.neighbors[YMinus],
                   &domain.neighbors[YPlus]);
    MPI_Cart_shift(domain.communicator, 2, 1, &domain.neighbors[ZMinus],
                   &domain.neighbors[ZPlus]);

    domain.globalX = global[0];
    domain.globalY = global[1];
    domain.globalZ = global[2];
    domain.startX = blockStart(global[0], domain.dims[0], domain.coords[0]);
    domain.startY = blockStart(global[1], domain.dims[1], domain.coords[1]);
    domain.startZ = blockStart(global[2], domain.dims[2], domain.coords[2]);
    domain.localX = blockSize(global[0], domain.dims[0], domain.coords[0]);
    domain.localY = blockSize(global[1], domain.dims[1], domain.coords[1]);
    domain.localZ = blockSize(global[2], domain.dims[2], domain.coords[2]);
    domain.strideX = domain.localX + 2;
    domain.strideY = domain.localY + 2;
    domain.strideZ = domain.localZ + 2;

    const bool datatypeSizesFit =
        domain.localX <= static_cast<size_t>(std::numeric_limits<int>::max() - 2) &&
        domain.localY <= static_cast<size_t>(std::numeric_limits<int>::max() - 2) &&
        domain.localZ <= static_cast<size_t>(std::numeric_limits<int>::max() - 2);
    int allSizesFit = datatypeSizesFit ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &allSizesFit, 1, MPI_INT, MPI_MIN,
                  domain.communicator);
    if (allSizesFit == 0) {
        if (domain.rank == 0) {
            std::printf("Local grid dimensions are too large for MPI datatypes\n");
        }
        return false;
    }

    for (int direction = 0; direction < 6; ++direction) {
        createFaceDatatype(domain, direction, false, domain.sendTypes[direction]);
        createFaceDatatype(domain, direction, true, domain.receiveTypes[direction]);
    }
    return true;
}

void destroyDomain(Domain& domain) {
    if (domain.communicator == MPI_COMM_NULL) {
        return;
    }
    for (int direction = 0; direction < 6; ++direction) {
        if (domain.sendTypes[direction] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&domain.sendTypes[direction]);
        }
        if (domain.receiveTypes[direction] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&domain.receiveTypes[direction]);
        }
    }
    MPI_Comm_free(&domain.communicator);
}

void copyClampedFace(std::vector<double>& field, const Domain& domain,
                     const int direction) {
    if (direction == XMinus) {
        for (size_t z = 1; z <= domain.localZ; ++z) {
            for (size_t y = 1; y <= domain.localY; ++y) {
                field[localIndex(domain, 0, y, z)] = field[localIndex(domain, 1, y, z)];
            }
        }
    } else if (direction == XPlus) {
        for (size_t z = 1; z <= domain.localZ; ++z) {
            for (size_t y = 1; y <= domain.localY; ++y) {
                field[localIndex(domain, domain.localX + 1, y, z)] =
                    field[localIndex(domain, domain.localX, y, z)];
            }
        }
    } else if (direction == YMinus) {
        for (size_t z = 1; z <= domain.localZ; ++z) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                field[localIndex(domain, x, 0, z)] = field[localIndex(domain, x, 1, z)];
            }
        }
    } else if (direction == YPlus) {
        for (size_t z = 1; z <= domain.localZ; ++z) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                field[localIndex(domain, x, domain.localY + 1, z)] =
                    field[localIndex(domain, x, domain.localY, z)];
            }
        }
    } else if (direction == ZMinus) {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                field[localIndex(domain, x, y, 0)] = field[localIndex(domain, x, y, 1)];
            }
        }
    } else {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                field[localIndex(domain, x, y, domain.localZ + 1)] =
                    field[localIndex(domain, x, y, domain.localZ)];
            }
        }
    }
}

void beginHaloExchange(std::vector<double>& field, const Domain& domain,
                       const int phase, HaloExchange& exchange) {
    exchange.count = 0;
    for (int direction = 0; direction < 6; ++direction) {
        if (domain.neighbors[direction] == MPI_PROC_NULL) {
            copyClampedFace(field, domain, direction);
        } else {
            const int axis = direction / 2;
            MPI_Irecv(field.data(), 1, domain.receiveTypes[direction],
                      domain.neighbors[direction], phase * 3 + axis,
                      domain.communicator, &exchange.requests[exchange.count++]);
        }
    }
    for (int direction = 0; direction < 6; ++direction) {
        if (domain.neighbors[direction] != MPI_PROC_NULL) {
            const int axis = direction / 2;
            MPI_Isend(field.data(), 1, domain.sendTypes[direction],
                      domain.neighbors[direction], phase * 3 + axis,
                      domain.communicator, &exchange.requests[exchange.count++]);
        }
    }
}

void finishHaloExchange(HaloExchange& exchange) {
    if (exchange.count > 0) {
        MPI_Waitall(exchange.count, exchange.requests, MPI_STATUSES_IGNORE);
    }
}

inline double laplacian(const std::vector<double>& field, const Domain& domain,
                        const size_t x, const size_t y, const size_t z,
                        const double inverseDxSquared,
                        const double inverseDySquared,
                        const double inverseDzSquared) noexcept {
    const size_t centerIndex = localIndex(domain, x, y, z);
    const double center = field[centerIndex];
    const double xx = (field[centerIndex + 1] + field[centerIndex - 1] - 2.0 * center) *
                      inverseDxSquared;
    const double yy = (field[centerIndex + domain.strideX] +
                       field[centerIndex - domain.strideX] - 2.0 * center) *
                      inverseDySquared;
    const double zz = (field[centerIndex + domain.strideX * domain.strideY] +
                       field[centerIndex - domain.strideX * domain.strideY] - 2.0 * center) *
                      inverseDzSquared;
    return xx + yy + zz;
}

inline double chemicalPotentialAt(const std::vector<double>& concentration,
                                  const Domain& domain, const size_t x,
                                  const size_t y, const size_t z,
                                  const double inverseDxSquared,
                                  const double inverseDySquared,
                                  const double inverseDzSquared,
                                  const double gamma, const double e_AA,
                                  const double e_BB, const double e_AB) noexcept {
    const size_t index = localIndex(domain, x, y, z);
    const double value = concentration[index];
    return 4.5 * ((value + 1.0) * e_AA + (value - 1.0) * e_BB -
                  2.0 * value * e_AB) +
           3.0 * value + value * value * value -
           gamma * laplacian(concentration, domain, x, y, z,
                             inverseDxSquared, inverseDySquared,
                             inverseDzSquared);
}

inline double concentrationUpdateAt(const std::vector<double>& concentration,
                                    const std::vector<double>& chemicalPotential,
                                    const Domain& domain, const size_t x,
                                    const size_t y, const size_t z,
                                    const double D, const double dt,
                                    const double inverseDxSquared,
                                    const double inverseDySquared,
                                    const double inverseDzSquared) noexcept {
    const size_t index = localIndex(domain, x, y, z);
    return concentration[index] + dt * D *
           laplacian(chemicalPotential, domain, x, y, z,
                     inverseDxSquared, inverseDySquared,
                     inverseDzSquared);
}

template <typename CellOperation>
void applyInterior(const Domain& domain, CellOperation&& operation) {
    // Only cells whose six neighbors are local are evaluated while the halo
    // exchange is in flight.
    for (size_t z = 2; z < domain.localZ; ++z) {
        for (size_t y = 2; y < domain.localY; ++y) {
            for (size_t x = 2; x < domain.localX; ++x) {
                operation(x, y, z);
            }
        }
    }
}

template <typename CellOperation>
void applyBoundary(const Domain& domain, CellOperation&& operation) {
    for (size_t z = 1; z <= domain.localZ; ++z) {
        const bool zBoundary = (z == 1 || z == domain.localZ);
        for (size_t y = 1; y <= domain.localY; ++y) {
            const bool yBoundary = (y == 1 || y == domain.localY);
            for (size_t x = 1; x <= domain.localX; ++x) {
                if (zBoundary || yBoundary || x == 1 || x == domain.localX) {
                    operation(x, y, z);
                }
            }
        }
    }
}

void computeChemicalPotentialInterior(const std::vector<double>& concentration,
                                      std::vector<double>& chemicalPotential,
                                      const Domain& domain, const double gamma,
                                      const double e_AA, const double e_BB,
                                      const double e_AB, const double invDx2,
                                      const double invDy2, const double invDz2) {
    applyInterior(domain, [&](const size_t x, const size_t y, const size_t z) {
        chemicalPotential[localIndex(domain, x, y, z)] =
            chemicalPotentialAt(concentration, domain, x, y, z, invDx2, invDy2,
                                invDz2, gamma, e_AA, e_BB, e_AB);
    });
}

void computeChemicalPotentialBoundary(const std::vector<double>& concentration,
                                      std::vector<double>& chemicalPotential,
                                      const Domain& domain, const double gamma,
                                      const double e_AA, const double e_BB,
                                      const double e_AB, const double invDx2,
                                      const double invDy2, const double invDz2) {
    applyBoundary(domain, [&](const size_t x, const size_t y, const size_t z) {
        chemicalPotential[localIndex(domain, x, y, z)] =
            chemicalPotentialAt(concentration, domain, x, y, z, invDx2, invDy2,
                                invDz2, gamma, e_AA, e_BB, e_AB);
    });
}

void computeUpdateInterior(std::vector<double>& newConcentration,
                           const std::vector<double>& concentration,
                           const std::vector<double>& chemicalPotential,
                           const Domain& domain, const double D, const double dt,
                           const double invDx2, const double invDy2,
                           const double invDz2) {
    applyInterior(domain, [&](const size_t x, const size_t y, const size_t z) {
        newConcentration[localIndex(domain, x, y, z)] = concentrationUpdateAt(
            concentration, chemicalPotential, domain, x, y, z, D, dt, invDx2,
            invDy2, invDz2);
    });
}

void computeUpdateBoundary(std::vector<double>& newConcentration,
                           const std::vector<double>& concentration,
                           const std::vector<double>& chemicalPotential,
                           const Domain& domain, const double D, const double dt,
                           const double invDx2, const double invDy2,
                           const double invDz2) {
    applyBoundary(domain, [&](const size_t x, const size_t y, const size_t z) {
        newConcentration[localIndex(domain, x, y, z)] = concentrationUpdateAt(
            concentration, chemicalPotential, domain, x, y, z, D, dt, invDx2,
            invDy2, invDz2);
    });
}

void initializeConcentration(std::vector<double>& concentration,
                             const Domain& domain, const size_t volume) {
    for (size_t z = 1; z <= domain.localZ; ++z) {
        const size_t globalZ = domain.startZ + z - 1;
        for (size_t y = 1; y <= domain.localY; ++y) {
            const size_t globalY = domain.startY + y - 1;
            for (size_t x = 1; x <= domain.localX; ++x) {
                const size_t globalX = domain.startX + x - 1;
                const size_t linearId =
                    globalZ * (domain.globalX * domain.globalY) +
                    globalY * domain.globalX + globalX;
                const double pseudo =
                    ((((linearId + 1) * 1299709) % volume) /
                     static_cast<double>(volume));
                concentration[localIndex(domain, x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration,
                    const Domain& domain) {
    int localInvalid = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= domain.localZ; ++z) {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                const double value = concentration[localIndex(domain, x, y, z)];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = 1;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    int invalid = 0;
    MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX,
                  domain.communicator);
    if (invalid != 0) {
        if (domain.rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double minValue = 0.0;
    double maxValue = 0.0;
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN,
                  domain.communicator);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX,
                  domain.communicator);

    if (domain.rank == 0) {
        std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    }
    return maxValue <= 10.0 && minValue >= -10.0;
}

void packOwned(const std::vector<double>& local, const Domain& domain,
               std::vector<double>& packed) {
    packed.resize(domain.localX * domain.localY * domain.localZ);
    size_t output = 0;
    for (size_t z = 1; z <= domain.localZ; ++z) {
        for (size_t y = 1; y <= domain.localY; ++y) {
            for (size_t x = 1; x <= domain.localX; ++x) {
                packed[output++] = local[localIndex(domain, x, y, z)];
            }
        }
    }
}

void gatherResult(const std::vector<double>& local, const Domain& domain,
                  std::vector<double>& globalResult) {
    std::vector<double> packed;
    packOwned(local, domain, packed);
    const size_t localCount = packed.size();
    const int localCountInt = static_cast<int>(localCount);

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> gathered;
    if (domain.rank == 0) {
        counts.resize(static_cast<size_t>(domain.size));
        displacements.resize(static_cast<size_t>(domain.size));
        gathered.resize(domain.globalX * domain.globalY * domain.globalZ);
    }

    MPI_Gather(&localCountInt, 1, MPI_INT,
               domain.rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               domain.communicator);

    if (domain.rank == 0) {
        int displacement = 0;
        for (int rank = 0; rank < domain.size; ++rank) {
            displacements[static_cast<size_t>(rank)] = displacement;
            displacement += counts[static_cast<size_t>(rank)];
        }
    }

    MPI_Gatherv(packed.data(), localCountInt, MPI_DOUBLE,
                domain.rank == 0 ? gathered.data() : nullptr,
                domain.rank == 0 ? counts.data() : nullptr,
                domain.rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, domain.communicator);

    if (domain.rank != 0) {
        return;
    }

    globalResult.assign(domain.globalX * domain.globalY * domain.globalZ, 0.0);
    for (int rank = 0; rank < domain.size; ++rank) {
        int coordinates[3] = {0, 0, 0};
        MPI_Cart_coords(domain.communicator, rank, 3, coordinates);
        const size_t startX = blockStart(domain.globalX, domain.dims[0], coordinates[0]);
        const size_t startY = blockStart(domain.globalY, domain.dims[1], coordinates[1]);
        const size_t startZ = blockStart(domain.globalZ, domain.dims[2], coordinates[2]);
        const size_t localX = blockSize(domain.globalX, domain.dims[0], coordinates[0]);
        const size_t localY = blockSize(domain.globalY, domain.dims[1], coordinates[1]);
        const size_t localZ = blockSize(domain.globalZ, domain.dims[2], coordinates[2]);
        size_t input = static_cast<size_t>(displacements[static_cast<size_t>(rank)]);

        for (size_t z = 0; z < localZ; ++z) {
            for (size_t y = 0; y < localY; ++y) {
                const size_t globalOffset = idx3(startX, startY + y, startZ + z,
                                                 domain.globalX, domain.globalY);
                for (size_t x = 0; x < localX; ++x) {
                    globalResult[globalOffset + x] = gathered[input++];
                }
            }
        }
    }
}

void printUsage(const char* progName, const int rank) {
    if (rank != 0) {
        return;
    }
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0], worldRank);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            printUsage(argv[0], worldRank);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            std::printf("Grid dimensions must be positive and time steps non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t sizeLimit = std::numeric_limits<size_t>::max();
    if (nx > sizeLimit / ny || nx * ny > sizeLimit / nz) {
        if (worldRank == 0) {
            std::printf("Grid size overflows the addressable size\n");
        }
        MPI_Finalize();
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    if (static_cast<size_t>(worldSize) > gridSize) {
        if (worldRank == 0) {
            std::printf("MPI process count (%d) exceeds the number of grid cells (%zu)\n",
                        worldSize, gridSize);
        }
        MPI_Finalize();
        return 1;
    }

    const std::array<size_t, 3> global = {nx, ny, nz};
    const std::array<int, 3> processGrid = chooseProcessGrid(worldSize, global);
    if (processGrid[0] == 0) {
        if (worldRank == 0) {
            std::printf("Cannot decompose %zu x %zu x %zu over %d MPI processes\n",
                        nx, ny, nz, worldSize);
        }
        MPI_Finalize();
        return 1;
    }

    Domain domain;
    if (!initializeDomain(domain, MPI_COMM_WORLD, global, processGrid)) {
        if (domain.communicator != MPI_COMM_NULL) {
            destroyDomain(domain);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t localSizeLimit = std::numeric_limits<size_t>::max();
    bool localStorageFits = domain.strideX != 0 && domain.strideY != 0 &&
                            domain.strideZ != 0;
    if (localStorageFits && domain.strideX > localSizeLimit / domain.strideY) {
        localStorageFits = false;
    }
    const size_t localXY = localStorageFits ? domain.strideX * domain.strideY : 0;
    if (localStorageFits && localXY > localSizeLimit / domain.strideZ) {
        localStorageFits = false;
    }
    const size_t localStorage = localStorageFits ? localXY * domain.strideZ : 0;
    int allStorageFits = localStorageFits ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &allStorageFits, 1, MPI_INT, MPI_MIN,
                  domain.communicator);
    if (allStorageFits == 0) {
        if (domain.rank == 0) {
            std::printf("Local grid storage size overflows the addressable size\n");
        }
        destroyDomain(domain);
        MPI_Finalize();
        return 1;
    }
    int localAllocationOk =
        localStorage <= std::numeric_limits<size_t>::max() / sizeof(double);
    int allAllocationsOk = 0;
    MPI_Allreduce(&localAllocationOk, &allAllocationsOk, 1, MPI_INT, MPI_MIN,
                  domain.communicator);
    if (allAllocationsOk == 0) {
        if (domain.rank == 0) {
            std::printf("Local grid allocation is too large\n");
        }
        destroyDomain(domain);
        MPI_Finalize();
        return 1;
    }

    if (domain.rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI processes: %d (%d x %d x %d Cartesian grid)\n",
                    worldSize, domain.dims[0], domain.dims[1], domain.dims[2]);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical and numerical parameters.
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    std::vector<double> cold(localStorage);
    std::vector<double> cnew(localStorage);
    std::vector<double> chemicalPotential(localStorage);

    if (domain.rank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, domain, gridSize);

    if (domain.rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(domain.communicator);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        HaloExchange concentrationHalo;
        beginHaloExchange(cold, domain, 0, concentrationHalo);
        computeChemicalPotentialInterior(cold, chemicalPotential, domain, gamma,
                                          e_AA, e_BB, e_AB, invDx2, invDy2,
                                          invDz2);
        finishHaloExchange(concentrationHalo);
        computeChemicalPotentialBoundary(cold, chemicalPotential, domain, gamma,
                                         e_AA, e_BB, e_AB, invDx2, invDy2,
                                         invDz2);

        HaloExchange chemicalPotentialHalo;
        beginHaloExchange(chemicalPotential, domain, 1, chemicalPotentialHalo);
        computeUpdateInterior(cnew, cold, chemicalPotential, domain, D, dt,
                              invDx2, invDy2, invDz2);
        finishHaloExchange(chemicalPotentialHalo);
        computeUpdateBoundary(cnew, cold, chemicalPotential, domain, D, dt,
                               invDx2, invDy2, invDz2);
        cold.swap(cnew);
    }

    const double localSeconds = MPI_Wtime() - start;
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               domain.communicator);

    if (domain.rank == 0) {
        std::printf("Computation time: %.3f ms\n", computationSeconds * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = computationSeconds > 0.0
                                 ? cellUpdates / computationSeconds / 1e6
                                 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> globalResult;
        const bool gatherFitsInt = gridSize <= static_cast<size_t>(std::numeric_limits<int>::max());
        int allGatherFitsInt = gatherFitsInt ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &allGatherFitsInt, 1, MPI_INT, MPI_MIN,
                      domain.communicator);
        if (allGatherFitsInt == 0) {
            if (domain.rank == 0) {
                std::printf("Result printing requires a grid no larger than INT_MAX cells\n");
            }
        } else {
            gatherResult(cold, domain, globalResult);
            if (domain.rank == 0) {
                print_results(globalResult, "Concentration");
            }
        }
    }

    int result = 0;
    if (validate) {
        if (domain.rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateResult(cold, domain);
        if (domain.rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        result = valid ? 0 : 1;
    }

    destroyDomain(domain);
    MPI_Finalize();
    return result;
}
