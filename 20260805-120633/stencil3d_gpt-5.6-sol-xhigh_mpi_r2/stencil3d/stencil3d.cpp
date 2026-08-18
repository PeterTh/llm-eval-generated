#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

enum Direction : int { XMinus = 0, XPlus, YMinus, YPlus, ZMinus, ZPlus };

struct Options {
    int nx = 128;
    int ny = 0;
    int nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseInteger(const char* text, const long minimum, const long maximum, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parseOptions(const int argc, char** argv, Options& options, int& badArgument) {
    constexpr long maxExtent = std::numeric_limits<int>::max() - 2L;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0)) {
            if (i + 1 >= argc) {
                badArgument = i;
                return false;
            }
            int value = 0;
            const bool isIterations = std::strcmp(argv[i], "-i") == 0;
            if (!parseInteger(argv[i + 1], isIterations ? 0L : 1L,
                              isIterations ? static_cast<long>(std::numeric_limits<int>::max()) : maxExtent,
                              value)) {
                badArgument = i + 1;
                return false;
            }
            if (std::strcmp(argv[i], "-x") == 0) options.nx = value;
            if (std::strcmp(argv[i], "-y") == 0) options.ny = value;
            if (std::strcmp(argv[i], "-z") == 0) options.nz = value;
            if (isIterations) options.iterations = value;
            ++i;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            badArgument = i;
            return false;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return true;
}

std::array<int, 3> chooseProcessGrid(const int ranks, const std::array<int, 3>& global) {
    std::array<int, 3> best{0, 0, 0};
    long double bestCost = std::numeric_limits<long double>::infinity();

    // Minimize the communication surface per local volume.  Enumerating factor
    // triples also lets us avoid empty blocks on highly anisotropic grids.
    for (int px = 1; px <= ranks; ++px) {
        if (ranks % px != 0 || px > global[0]) continue;
        const int remainder = ranks / px;
        for (int py = 1; py <= remainder; ++py) {
            if (remainder % py != 0 || py > global[1]) continue;
            const int pz = remainder / py;
            if (pz > global[2]) continue;

            const long double cost =
                (px > 1 ? static_cast<long double>(px - 1) / global[0] : 0.0L) +
                (py > 1 ? static_cast<long double>(py - 1) / global[1] : 0.0L) +
                (pz > 1 ? static_cast<long double>(pz - 1) / global[2] : 0.0L);
            if (cost < bestCost) {
                bestCost = cost;
                best = {px, py, pz};
            }
        }
    }
    return best;
}

int blockExtent(const int global, const int partitions, const int coordinate) {
    return global / partitions + (coordinate < global % partitions ? 1 : 0);
}

int blockStart(const int global, const int partitions, const int coordinate) {
    return coordinate * (global / partitions) + std::min(coordinate, global % partitions);
}

class Domain {
public:
    explicit Domain(const std::array<int, 3>& globalExtents) : global(globalExtents) {
        int worldSize = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        dims = chooseProcessGrid(worldSize, global);
        if (dims[0] == 0) {
            int worldRank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
            if (worldRank == 0) {
                std::fprintf(stderr, "The grid has fewer cells than MPI ranks; a non-empty block cannot be assigned to every rank.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 1, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        MPI_Cart_coords(comm, rank, 3, coords.data());

        for (int axis = 0; axis < 3; ++axis) {
            local[axis] = blockExtent(global[axis], dims[axis], coords[axis]);
            start[axis] = blockStart(global[axis], dims[axis], coords[axis]);
        }
        MPI_Cart_shift(comm, 0, 1, &neighbor[XMinus], &neighbor[XPlus]);
        MPI_Cart_shift(comm, 1, 1, &neighbor[YMinus], &neighbor[YPlus]);
        MPI_Cart_shift(comm, 2, 1, &neighbor[ZMinus], &neighbor[ZPlus]);

        // Local array coordinates are one-based because index zero is a halo.
        for (int axis = 0; axis < 3; ++axis) {
            updateLo[axis] = std::max(1, 2 - start[axis]);
            updateHi[axis] = std::min(local[axis], global[axis] - 1 - start[axis]);
        }
    }

    ~Domain() {
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    size_t arrayIndex(const int x, const int y, const int z) const noexcept {
        return static_cast<size_t>(z) * planeStride() + static_cast<size_t>(y) * rowStride() +
               static_cast<size_t>(x);
    }

    size_t rowStride() const noexcept { return static_cast<size_t>(local[0] + 2); }
    size_t planeStride() const noexcept { return rowStride() * static_cast<size_t>(local[1] + 2); }
    size_t allocationSize() const noexcept {
        return planeStride() * static_cast<size_t>(local[2] + 2);
    }
    size_t ownedSize() const noexcept {
        return static_cast<size_t>(local[0]) * local[1] * local[2];
    }

    std::array<int, 3> global{};
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    std::array<int, 3> local{};
    std::array<int, 3> start{};
    std::array<int, 3> updateLo{};
    std::array<int, 3> updateHi{};
    std::array<int, 6> neighbor{};
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 0;
};

class HaloExchange {
public:
    explicit HaloExchange(const Domain& domain) : domain_(domain) {
        const std::array<size_t, 6> faceSizes = {
            static_cast<size_t>(domain.local[1]) * domain.local[2],
            static_cast<size_t>(domain.local[1]) * domain.local[2],
            static_cast<size_t>(domain.local[0]) * domain.local[2],
            static_cast<size_t>(domain.local[0]) * domain.local[2],
            static_cast<size_t>(domain.local[0]) * domain.local[1],
            static_cast<size_t>(domain.local[0]) * domain.local[1]};

        for (int direction = 0; direction < 6; ++direction) {
            if (domain.neighbor[direction] == MPI_PROC_NULL) continue;
            if (faceSizes[direction] > static_cast<size_t>(std::numeric_limits<int>::max())) {
                if (domain.rank == 0) std::fprintf(stderr, "An MPI halo face exceeds the supported message size.\n");
                MPI_Abort(domain.comm, 3);
            }
            count_[direction] = static_cast<int>(faceSizes[direction]);
            send_[direction].resize(faceSizes[direction]);
            receive_[direction].resize(faceSizes[direction]);
        }

        // Persistent requests avoid rebuilding the same communication schedule
        // on every Jacobi iteration.
        for (int direction = 0; direction < 6; ++direction) {
            if (domain.neighbor[direction] == MPI_PROC_NULL) continue;
            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Recv_init(receive_[direction].data(), count_[direction], MPI_DOUBLE,
                          domain.neighbor[direction], direction / 2, domain.comm, &request);
            requests_.push_back(request);
        }
        for (int direction = 0; direction < 6; ++direction) {
            if (domain.neighbor[direction] == MPI_PROC_NULL) continue;
            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Send_init(send_[direction].data(), count_[direction], MPI_DOUBLE,
                          domain.neighbor[direction], direction / 2, domain.comm, &request);
            requests_.push_back(request);
        }
    }

    ~HaloExchange() {
        for (MPI_Request& request : requests_) MPI_Request_free(&request);
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void start(const std::vector<Real>& grid) {
        pack(grid);
        if (!requests_.empty()) MPI_Startall(static_cast<int>(requests_.size()), requests_.data());
    }

    void finish(std::vector<Real>& grid) {
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
        }
        unpack(grid);
    }

private:
    void pack(const std::vector<Real>& grid) {
        const int lx = domain_.local[0];
        const int ly = domain_.local[1];
        const int lz = domain_.local[2];

        if (domain_.neighbor[XMinus] != MPI_PROC_NULL || domain_.neighbor[XPlus] != MPI_PROC_NULL) {
            size_t packed = 0;
            for (int z = 1; z <= lz; ++z) {
                for (int y = 1; y <= ly; ++y, ++packed) {
                    if (domain_.neighbor[XMinus] != MPI_PROC_NULL)
                        send_[XMinus][packed] = grid[domain_.arrayIndex(1, y, z)];
                    if (domain_.neighbor[XPlus] != MPI_PROC_NULL)
                        send_[XPlus][packed] = grid[domain_.arrayIndex(lx, y, z)];
                }
            }
        }
        if (domain_.neighbor[YMinus] != MPI_PROC_NULL || domain_.neighbor[YPlus] != MPI_PROC_NULL) {
            size_t packed = 0;
            for (int z = 1; z <= lz; ++z) {
                if (domain_.neighbor[YMinus] != MPI_PROC_NULL) {
                    std::copy_n(&grid[domain_.arrayIndex(1, 1, z)], lx, send_[YMinus].data() + packed);
                }
                if (domain_.neighbor[YPlus] != MPI_PROC_NULL) {
                    std::copy_n(&grid[domain_.arrayIndex(1, ly, z)], lx, send_[YPlus].data() + packed);
                }
                packed += static_cast<size_t>(lx);
            }
        }
        if (domain_.neighbor[ZMinus] != MPI_PROC_NULL) {
            packZFace(grid, 1, send_[ZMinus]);
        }
        if (domain_.neighbor[ZPlus] != MPI_PROC_NULL) {
            packZFace(grid, lz, send_[ZPlus]);
        }
    }

    void packZFace(const std::vector<Real>& grid, const int z, std::vector<Real>& buffer) {
        size_t packed = 0;
        for (int y = 1; y <= domain_.local[1]; ++y) {
            std::copy_n(&grid[domain_.arrayIndex(1, y, z)], domain_.local[0], buffer.data() + packed);
            packed += static_cast<size_t>(domain_.local[0]);
        }
    }

    void unpack(std::vector<Real>& grid) {
        const int lx = domain_.local[0];
        const int ly = domain_.local[1];
        const int lz = domain_.local[2];

        if (domain_.neighbor[XMinus] != MPI_PROC_NULL || domain_.neighbor[XPlus] != MPI_PROC_NULL) {
            size_t packed = 0;
            for (int z = 1; z <= lz; ++z) {
                for (int y = 1; y <= ly; ++y, ++packed) {
                    if (domain_.neighbor[XMinus] != MPI_PROC_NULL)
                        grid[domain_.arrayIndex(0, y, z)] = receive_[XMinus][packed];
                    if (domain_.neighbor[XPlus] != MPI_PROC_NULL)
                        grid[domain_.arrayIndex(lx + 1, y, z)] = receive_[XPlus][packed];
                }
            }
        }
        if (domain_.neighbor[YMinus] != MPI_PROC_NULL || domain_.neighbor[YPlus] != MPI_PROC_NULL) {
            size_t packed = 0;
            for (int z = 1; z <= lz; ++z) {
                if (domain_.neighbor[YMinus] != MPI_PROC_NULL) {
                    std::copy_n(receive_[YMinus].data() + packed, lx,
                                &grid[domain_.arrayIndex(1, 0, z)]);
                }
                if (domain_.neighbor[YPlus] != MPI_PROC_NULL) {
                    std::copy_n(receive_[YPlus].data() + packed, lx,
                                &grid[domain_.arrayIndex(1, ly + 1, z)]);
                }
                packed += static_cast<size_t>(lx);
            }
        }
        if (domain_.neighbor[ZMinus] != MPI_PROC_NULL) {
            unpackZFace(receive_[ZMinus], 0, grid);
        }
        if (domain_.neighbor[ZPlus] != MPI_PROC_NULL) {
            unpackZFace(receive_[ZPlus], lz + 1, grid);
        }
    }

    void unpackZFace(const std::vector<Real>& buffer, const int z, std::vector<Real>& grid) {
        size_t packed = 0;
        for (int y = 1; y <= domain_.local[1]; ++y) {
            std::copy_n(buffer.data() + packed, domain_.local[0],
                        &grid[domain_.arrayIndex(1, y, z)]);
            packed += static_cast<size_t>(domain_.local[0]);
        }
    }

    const Domain& domain_;
    std::array<int, 6> count_{};
    std::array<std::vector<Real>, 6> send_;
    std::array<std::vector<Real>, 6> receive_;
    std::vector<MPI_Request> requests_;
};

void initializeGrids(std::vector<Real>& first, std::vector<Real>& second, const Domain& domain) {
    for (int z = 1; z <= domain.local[2]; ++z) {
        const size_t globalZ = static_cast<size_t>(domain.start[2] + z - 1);
        for (int y = 1; y <= domain.local[1]; ++y) {
            const size_t globalY = static_cast<size_t>(domain.start[1] + y - 1);
            size_t globalIndex = (globalZ * static_cast<size_t>(domain.global[1]) + globalY) *
                                     static_cast<size_t>(domain.global[0]) +
                                 static_cast<size_t>(domain.start[0]);
            size_t localIndex = domain.arrayIndex(1, y, z);
            for (int x = 1; x <= domain.local[0]; ++x, ++globalIndex, ++localIndex) {
                const Real value = static_cast<Real>(globalIndex % 19);
                first[localIndex] = value;
                // Global boundary values are invariant.  Preinitializing both
                // buffers removes a boundary-copy pass from every iteration.
                second[localIndex] = value;
            }
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
#define STENCIL_RESTRICT __restrict__
#else
#define STENCIL_RESTRICT
#endif

inline void stencilBox(const Real* STENCIL_RESTRICT input, Real* STENCIL_RESTRICT output,
                       const size_t rowStride, const size_t planeStride,
                       const int xBegin, const int xEnd, const int yBegin, const int yEnd,
                       const int zBegin, const int zEnd) noexcept {
    if (xBegin > xEnd || yBegin > yEnd || zBegin > zEnd) return;
    for (int z = zBegin; z <= zEnd; ++z) {
        for (int y = yBegin; y <= yEnd; ++y) {
            size_t index = static_cast<size_t>(z) * planeStride + static_cast<size_t>(y) * rowStride +
                           static_cast<size_t>(xBegin);
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (int x = xBegin; x <= xEnd; ++x, ++index) {
                output[index] = (input[index] + input[index - 1] + input[index + 1] +
                                 input[index - rowStride] + input[index + rowStride] +
                                 input[index - planeStride] + input[index + planeStride]) /
                                7.0;
            }
        }
    }
}

#undef STENCIL_RESTRICT

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const Domain& domain, HaloExchange& halo) {
    const int xLo = domain.updateLo[0];
    const int xHi = domain.updateHi[0];
    const int yLo = domain.updateLo[1];
    const int yHi = domain.updateHi[1];
    const int zLo = domain.updateLo[2];
    const int zHi = domain.updateHi[2];

    const int coreXLo = std::max(xLo, 2);
    const int coreXHi = std::min(xHi, domain.local[0] - 1);
    const int coreYLo = std::max(yLo, 2);
    const int coreYHi = std::min(yHi, domain.local[1] - 1);
    const int coreZLo = std::max(zLo, 2);
    const int coreZHi = std::min(zHi, domain.local[2] - 1);
    const bool hasCore = coreXLo <= coreXHi && coreYLo <= coreYHi && coreZLo <= coreZHi;

    halo.start(input);
    if (hasCore) {
        stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
                   coreXLo, coreXHi, coreYLo, coreYHi, coreZLo, coreZHi);
    }
    halo.finish(input);

    if (!hasCore) {
        stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
                   xLo, xHi, yLo, yHi, zLo, zHi);
        return;
    }

    // Six non-overlapping boxes cover the one-cell shell around the core.
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               xLo, xHi, yLo, yHi, zLo, coreZLo - 1);
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               xLo, xHi, yLo, yHi, coreZHi + 1, zHi);
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               xLo, xHi, yLo, coreYLo - 1, coreZLo, coreZHi);
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               xLo, xHi, coreYHi + 1, yHi, coreZLo, coreZHi);
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               xLo, coreXLo - 1, coreYLo, coreYHi, coreZLo, coreZHi);
    stencilBox(input.data(), output.data(), domain.rowStride(), domain.planeStride(),
               coreXHi + 1, xHi, coreYLo, coreYHi, coreZLo, coreZHi);
}

std::vector<Real> packOwned(const std::vector<Real>& grid, const Domain& domain) {
    std::vector<Real> packed(domain.ownedSize());
    size_t offset = 0;
    for (int z = 1; z <= domain.local[2]; ++z) {
        for (int y = 1; y <= domain.local[1]; ++y) {
            std::copy_n(&grid[domain.arrayIndex(1, y, z)], domain.local[0], packed.data() + offset);
            offset += static_cast<size_t>(domain.local[0]);
        }
    }
    return packed;
}

std::vector<Real> gatherGlobalGrid(const std::vector<Real>& grid, const Domain& domain) {
    std::vector<Real> localPacked = packOwned(grid, domain);
    if (localPacked.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (domain.rank == 0) std::fprintf(stderr, "A local result block exceeds MPI_Gatherv limits.\n");
        MPI_Abort(domain.comm, 4);
    }
    const int localCount = static_cast<int>(localPacked.size());
    std::vector<int> counts;
    if (domain.rank == 0) counts.resize(domain.size);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, domain.comm);

    std::vector<int> displacements;
    std::vector<Real> rankOrdered;
    if (domain.rank == 0) {
        displacements.resize(domain.size);
        size_t total = 0;
        for (int rank = 0; rank < domain.size; ++rank) {
            if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "The global result exceeds MPI_Gatherv limits.\n");
                MPI_Abort(domain.comm, 4);
            }
            displacements[rank] = static_cast<int>(total);
            total += static_cast<size_t>(counts[rank]);
        }
        if (total > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "The global result exceeds MPI_Gatherv limits.\n");
            MPI_Abort(domain.comm, 4);
        }
        rankOrdered.resize(total);
    }
    MPI_Gatherv(localPacked.data(), localCount, MPI_DOUBLE, rankOrdered.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, domain.comm);

    if (domain.rank != 0) return {};

    const size_t totalCells = static_cast<size_t>(domain.global[0]) * domain.global[1] * domain.global[2];
    std::vector<Real> globalGrid(totalCells);
    for (int rank = 0; rank < domain.size; ++rank) {
        std::array<int, 3> coordinates{};
        MPI_Cart_coords(domain.comm, rank, 3, coordinates.data());
        std::array<int, 3> extent{};
        std::array<int, 3> begin{};
        for (int axis = 0; axis < 3; ++axis) {
            extent[axis] = blockExtent(domain.global[axis], domain.dims[axis], coordinates[axis]);
            begin[axis] = blockStart(domain.global[axis], domain.dims[axis], coordinates[axis]);
        }
        size_t source = static_cast<size_t>(displacements[rank]);
        for (int z = 0; z < extent[2]; ++z) {
            for (int y = 0; y < extent[1]; ++y) {
                const size_t destination =
                    (static_cast<size_t>(begin[2] + z) * domain.global[1] + begin[1] + y) *
                        domain.global[0] +
                    begin[0];
                std::copy_n(rankOrdered.data() + source, extent[0], globalGrid.data() + destination);
                source += static_cast<size_t>(extent[0]);
            }
        }
    }
    return globalGrid;
}

bool validateResult(const std::vector<Real>& grid, const Domain& domain) {
    bool localInvalid = false;
    Real localMinimum = std::numeric_limits<Real>::infinity();
    Real localMaximum = -std::numeric_limits<Real>::infinity();
    for (int z = 1; z <= domain.local[2]; ++z) {
        for (int y = 1; y <= domain.local[1]; ++y) {
            size_t index = domain.arrayIndex(1, y, z);
            for (int x = 1; x <= domain.local[0]; ++x, ++index) {
                const Real value = grid[index];
                localInvalid = localInvalid || !std::isfinite(value);
                localMinimum = std::min(localMinimum, value);
                localMaximum = std::max(localMaximum, value);
            }
        }
    }

    int invalid = localInvalid ? 1 : 0;
    int anyInvalid = 0;
    Real globalMinimum = 0.0;
    Real globalMaximum = 0.0;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_LOR, domain.comm);
    MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, domain.comm);
    MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, domain.comm);

    if (domain.rank == 0) {
        if (anyInvalid) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
        if (globalMaximum > 1e6 || globalMinimum < -1e6)
            std::printf("Validation failed: values out of expected range\n");
    }
    return !anyInvalid && globalMaximum <= 1e6 && globalMinimum >= -1e6;
}

int runBenchmark(const int argc, char** argv) {
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    Options options;
    int badArgument = 0;
    if (!parseOptions(argc, argv, options, badArgument)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Invalid or unknown argument: %s\n", argv[badArgument]);
            printUsage(argv[0]);
        }
        return 1;
    }
    if (options.help) {
        if (worldRank == 0) printUsage(argv[0]);
        return 0;
    }

    Domain domain({options.nx, options.ny, options.nz});
    if (domain.rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %d x %d x %d\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (%d x %d x %d Cartesian grid)\n",
                    domain.size, domain.dims[0], domain.dims[1], domain.dims[2]);
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> grid1(domain.allocationSize());
    std::vector<Real> grid2(domain.allocationSize());
    initializeGrids(grid1, grid2, domain);
    HaloExchange halo(domain);

    if (domain.rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(domain.comm);
    const double startTime = MPI_Wtime();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        if ((iteration & 1) == 0)
            stencilIteration(grid1, grid2, domain, halo);
        else
            stencilIteration(grid2, grid1, domain, halo);
    }
    const double localDuration = MPI_Wtime() - startTime;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);

    if (domain.rank == 0) {
        const double interiorX = std::max(options.nx - 2, 0);
        const double interiorY = std::max(options.ny - 2, 0);
        const double interiorZ = std::max(options.nz - 2, 0);
        const double cellUpdates = interiorX * interiorY * interiorZ * options.iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (options.iterations & 1) == 0 ? grid1 : grid2;
    if (options.printResults) {
        std::vector<Real> globalGrid = gatherGlobalGrid(finalGrid, domain);
        if (domain.rank == 0) print_results(globalGrid, "Grid");
    }

    bool valid = true;
    if (options.validate) {
        if (domain.rank == 0) std::printf("Validating result...\n");
        valid = validateResult(finalGrid, domain);
        if (domain.rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int result = runBenchmark(argc, argv);
    MPI_Finalize();
    return result;
}
