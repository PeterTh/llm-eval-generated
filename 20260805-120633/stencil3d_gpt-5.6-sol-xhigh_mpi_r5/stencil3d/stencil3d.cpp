#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct ProcessGrid {
    int px = 1;
    int py = 1;
    int pz = 1;
};

struct Layout {
    int lx = 0;
    int ly = 0;
    int lz = 0;
    size_t pitch = 0;
    size_t plane = 0;

    [[nodiscard]] size_t index(const int x, const int y, const int z) const noexcept {
        return static_cast<size_t>(z) * plane + static_cast<size_t>(y) * pitch +
               static_cast<size_t>(x);
    }
};

struct Domain {
    size_t nx = 0;
    size_t ny = 0;
    size_t nz = 0;
    size_t sx = 0;
    size_t sy = 0;
    size_t sz = 0;
    Layout local;
};

struct Neighbors {
    int xm = MPI_PROC_NULL;
    int xp = MPI_PROC_NULL;
    int ym = MPI_PROC_NULL;
    int yp = MPI_PROC_NULL;
    int zm = MPI_PROC_NULL;
    int zp = MPI_PROC_NULL;
};

struct FaceTypes {
    MPI_Datatype x = MPI_DATATYPE_NULL;
    MPI_Datatype y = MPI_DATATYPE_NULL;
    MPI_Datatype z = MPI_DATATYPE_NULL;
};

struct AxisRange {
    int begin = 1;
    int end = 1;
    int coreBegin = 1;
    int coreEnd = 1;
};

struct UpdateRanges {
    AxisRange x;
    AxisRange y;
    AxisRange z;
};

static std::pair<size_t, size_t> blockPartition(const size_t n, const int parts,
                                                 const int coordinate) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    const size_t count = base + (static_cast<size_t>(coordinate) < remainder ? 1 : 0);
    const size_t start = static_cast<size_t>(coordinate) * base +
                         std::min(static_cast<size_t>(coordinate), remainder);
    return {start, count};
}

// Keep the slowest rank close to the best possible load balance, then minimize
// its halo surface. This avoids communication-heavy slabs selected merely to
// eliminate a small block-rounding imbalance. Ties favor contiguous z/y faces.
static ProcessGrid chooseProcessGrid(const int processCount, const size_t nx,
                                     const size_t ny, const size_t nz) {
    struct Candidate {
        ProcessGrid grid;
        long double maxVolume;
        long double maxHalo;
        long double aggregateHalo;
    };
    std::vector<Candidate> candidates;
    long double minimumMaxVolume = std::numeric_limits<long double>::infinity();

    const auto neighborFaces = [](const int p) -> int {
        return p == 1 ? 0 : (p == 2 ? 1 : 2);
    };

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<size_t>(px) > nx) continue;
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<size_t>(py) > ny) continue;
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > nz) continue;

            const size_t bx = (nx + static_cast<size_t>(px) - 1) / static_cast<size_t>(px);
            const size_t by = (ny + static_cast<size_t>(py) - 1) / static_cast<size_t>(py);
            const size_t bz = (nz + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz);

            const long double maxVolume = static_cast<long double>(bx) * by * bz;
            const long double maxHalo =
                static_cast<long double>(neighborFaces(px)) * by * bz +
                static_cast<long double>(neighborFaces(py)) * bx * bz +
                static_cast<long double>(neighborFaces(pz)) * bx * by;
            const long double aggregate =
                static_cast<long double>(px - 1) * ny * nz +
                static_cast<long double>(py - 1) * nx * nz +
                static_cast<long double>(pz - 1) * nx * ny;

            candidates.push_back({{px, py, pz}, maxVolume, maxHalo, aggregate});
            minimumMaxVolume = std::min(minimumMaxVolume, maxVolume);
        }
    }

    ProcessGrid best{0, 0, 0};
    long double bestVolume = std::numeric_limits<long double>::infinity();
    long double bestHalo = std::numeric_limits<long double>::infinity();
    long double bestAggregate = std::numeric_limits<long double>::infinity();
    constexpr long double maximumImbalance = 1.05L;
    for (const Candidate& candidate : candidates) {
        if (candidate.maxVolume > minimumMaxVolume * maximumImbalance) continue;
        const bool better =
            candidate.maxHalo < bestHalo ||
            (candidate.maxHalo == bestHalo && candidate.aggregateHalo < bestAggregate) ||
            (candidate.maxHalo == bestHalo && candidate.aggregateHalo == bestAggregate &&
             candidate.maxVolume < bestVolume) ||
            (candidate.maxHalo == bestHalo && candidate.aggregateHalo == bestAggregate &&
             candidate.maxVolume == bestVolume &&
             (candidate.grid.pz > best.pz ||
              (candidate.grid.pz == best.pz && candidate.grid.py > best.py)));
        if (better) {
            best = candidate.grid;
            bestVolume = candidate.maxVolume;
            bestHalo = candidate.maxHalo;
            bestAggregate = candidate.aggregateHalo;
        }
    }
    return best;
}

static FaceTypes createFaceTypes(const Layout& layout) {
    FaceTypes types;
    MPI_Datatype column = MPI_DATATYPE_NULL;

    MPI_Type_vector(layout.ly, 1, static_cast<int>(layout.pitch), MPI_DOUBLE, &column);
    MPI_Type_create_hvector(layout.lz, 1,
                            static_cast<MPI_Aint>(layout.plane * sizeof(Real)),
                            column, &types.x);
    MPI_Type_commit(&types.x);
    MPI_Type_free(&column);

    MPI_Type_create_hvector(layout.lz, layout.lx,
                            static_cast<MPI_Aint>(layout.plane * sizeof(Real)),
                            MPI_DOUBLE, &types.y);
    MPI_Type_commit(&types.y);

    MPI_Type_vector(layout.ly, layout.lx, static_cast<int>(layout.pitch),
                    MPI_DOUBLE, &types.z);
    MPI_Type_commit(&types.z);
    return types;
}

static void freeFaceTypes(FaceTypes& types) {
    MPI_Type_free(&types.x);
    MPI_Type_free(&types.y);
    MPI_Type_free(&types.z);
}

enum MessageTag : int {
    XLow = 10,
    XHigh = 11,
    YLow = 12,
    YHigh = 13,
    ZLow = 14,
    ZHigh = 15,
    GatherResult = 20
};

static void addPersistentReceive(std::vector<MPI_Request>& requests, Real* buffer,
                                 const MPI_Datatype datatype, const int source,
                                 const int tag, MPI_Comm communicator) {
    if (source == MPI_PROC_NULL) return;
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Recv_init(buffer, 1, datatype, source, tag, communicator, &request);
    requests.push_back(request);
}

static void addPersistentSend(std::vector<MPI_Request>& requests, Real* buffer,
                              const MPI_Datatype datatype, const int destination,
                              const int tag, MPI_Comm communicator) {
    if (destination == MPI_PROC_NULL) return;
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Send_init(buffer, 1, datatype, destination, tag, communicator, &request);
    requests.push_back(request);
}

static std::vector<MPI_Request> createPersistentExchange(
        Real* grid, const Layout& layout, const Neighbors& neighbors,
        const FaceTypes& types, MPI_Comm communicator) {
    std::vector<MPI_Request> requests;
    requests.reserve(12);

    addPersistentReceive(requests, grid + layout.index(0, 1, 1), types.x,
                         neighbors.xm, XHigh, communicator);
    addPersistentReceive(requests, grid + layout.index(layout.lx + 1, 1, 1), types.x,
                         neighbors.xp, XLow, communicator);
    addPersistentReceive(requests, grid + layout.index(1, 0, 1), types.y,
                         neighbors.ym, YHigh, communicator);
    addPersistentReceive(requests, grid + layout.index(1, layout.ly + 1, 1), types.y,
                         neighbors.yp, YLow, communicator);
    addPersistentReceive(requests, grid + layout.index(1, 1, 0), types.z,
                         neighbors.zm, ZHigh, communicator);
    addPersistentReceive(requests, grid + layout.index(1, 1, layout.lz + 1), types.z,
                         neighbors.zp, ZLow, communicator);

    addPersistentSend(requests, grid + layout.index(1, 1, 1), types.x,
                      neighbors.xm, XLow, communicator);
    addPersistentSend(requests, grid + layout.index(layout.lx, 1, 1), types.x,
                      neighbors.xp, XHigh, communicator);
    addPersistentSend(requests, grid + layout.index(1, 1, 1), types.y,
                      neighbors.ym, YLow, communicator);
    addPersistentSend(requests, grid + layout.index(1, layout.ly, 1), types.y,
                      neighbors.yp, YHigh, communicator);
    addPersistentSend(requests, grid + layout.index(1, 1, 1), types.z,
                      neighbors.zm, ZLow, communicator);
    addPersistentSend(requests, grid + layout.index(1, 1, layout.lz), types.z,
                      neighbors.zp, ZHigh, communicator);
    return requests;
}

static void freePersistentExchange(std::vector<MPI_Request>& requests) {
    for (MPI_Request& request : requests) MPI_Request_free(&request);
}

static AxisRange makeAxisRange(const size_t globalStart, const int localCount,
                               const size_t globalCount, const int minusNeighbor,
                               const int plusNeighbor) {
    AxisRange range;
    if (globalCount > 2) {
        const size_t globalBegin = std::max<size_t>(globalStart, 1);
        const size_t globalEnd = std::min(globalStart + static_cast<size_t>(localCount),
                                          globalCount - 1);
        if (globalBegin < globalEnd) {
            range.begin = static_cast<int>(globalBegin - globalStart + 1);
            range.end = static_cast<int>(globalEnd - globalStart + 1);
        }
    }

    range.coreBegin = range.begin;
    if (minusNeighbor != MPI_PROC_NULL) range.coreBegin = std::max(range.coreBegin, 2);
    range.coreBegin = std::min(range.coreBegin, range.end);

    range.coreEnd = range.end;
    if (plusNeighbor != MPI_PROC_NULL) range.coreEnd = std::min(range.coreEnd, localCount);
    range.coreEnd = std::max(range.coreEnd, range.coreBegin);
    return range;
}

#if defined(__GNUC__) || defined(__clang__)
#define STENCIL_RESTRICT __restrict__
#else
#define STENCIL_RESTRICT
#endif

static inline void updateBox(const Real* STENCIL_RESTRICT input,
                             Real* STENCIL_RESTRICT output, const Layout& layout,
                             const int xb, const int xe, const int yb, const int ye,
                             const int zb, const int ze) {
    if (xb >= xe || yb >= ye || zb >= ze) return;
    for (int z = zb; z < ze; ++z) {
        for (int y = yb; y < ye; ++y) {
            const size_t row = layout.index(0, y, z);
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (int x = xb; x < xe; ++x) {
                const size_t index = row + static_cast<size_t>(x);
                output[index] = (input[index] + input[index - 1] + input[index + 1] +
                                 input[index - layout.pitch] + input[index + layout.pitch] +
                                 input[index - layout.plane] + input[index + layout.plane]) /
                                7.0;
            }
        }
    }
}

static void stencilIteration(const Real* input, Real* output, const Layout& layout,
                             const UpdateRanges& ranges,
                             std::vector<MPI_Request>& exchange) {
    if (!exchange.empty()) {
        MPI_Startall(static_cast<int>(exchange.size()), exchange.data());
    }

    // This core cannot depend on any in-flight receive and overlaps useful work
    // with all six halo transfers.
    updateBox(input, output, layout,
              ranges.x.coreBegin, ranges.x.coreEnd,
              ranges.y.coreBegin, ranges.y.coreEnd,
              ranges.z.coreBegin, ranges.z.coreEnd);

    if (!exchange.empty()) {
        MPI_Waitall(static_cast<int>(exchange.size()), exchange.data(), MPI_STATUSES_IGNORE);
    }

    // Six disjoint boxes cover the communication-dependent shell.
    updateBox(input, output, layout,
              ranges.x.begin, ranges.x.coreBegin,
              ranges.y.begin, ranges.y.end, ranges.z.begin, ranges.z.end);
    updateBox(input, output, layout,
              ranges.x.coreEnd, ranges.x.end,
              ranges.y.begin, ranges.y.end, ranges.z.begin, ranges.z.end);
    updateBox(input, output, layout,
              ranges.x.coreBegin, ranges.x.coreEnd,
              ranges.y.begin, ranges.y.coreBegin, ranges.z.begin, ranges.z.end);
    updateBox(input, output, layout,
              ranges.x.coreBegin, ranges.x.coreEnd,
              ranges.y.coreEnd, ranges.y.end, ranges.z.begin, ranges.z.end);
    updateBox(input, output, layout,
              ranges.x.coreBegin, ranges.x.coreEnd,
              ranges.y.coreBegin, ranges.y.coreEnd, ranges.z.begin, ranges.z.coreBegin);
    updateBox(input, output, layout,
              ranges.x.coreBegin, ranges.x.coreEnd,
              ranges.y.coreBegin, ranges.y.coreEnd, ranges.z.coreEnd, ranges.z.end);
}

static void initializeGrid(std::vector<Real>& grid1, std::vector<Real>& grid2,
                           const Domain& domain) {
    const Layout& layout = domain.local;
    const size_t xy = domain.nx * domain.ny;
    for (int z = 0; z < layout.lz; ++z) {
        const size_t gz = domain.sz + static_cast<size_t>(z);
        for (int y = 0; y < layout.ly; ++y) {
            const size_t gy = domain.sy + static_cast<size_t>(y);
            for (int x = 0; x < layout.lx; ++x) {
                const size_t gx = domain.sx + static_cast<size_t>(x);
                const Real value = static_cast<Real>((gz * xy + gy * domain.nx + gx) % 19);
                const size_t localIndex = layout.index(x + 1, y + 1, z + 1);
                grid1[localIndex] = value;
                if (gx == 0 || gx + 1 == domain.nx || gy == 0 ||
                    gy + 1 == domain.ny || gz == 0 || gz + 1 == domain.nz) {
                    // Global boundary values never change. Seeding both buffers
                    // avoids copying the six boundary planes on every iteration.
                    grid2[localIndex] = value;
                }
            }
        }
    }
}

static bool validateDistributed(const Real* grid, const Domain& domain,
                                MPI_Comm communicator, const int rank) {
    const Layout& layout = domain.local;
    int invalid = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (int z = 1; z <= layout.lz; ++z) {
        for (int y = 1; y <= layout.ly; ++y) {
            for (int x = 1; x <= layout.lx; ++x) {
                const Real value = grid[layout.index(x, y, z)];
                if (!std::isfinite(value)) {
                    invalid = 1;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    int anyInvalid = 0;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, communicator);
    if (anyInvalid) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, communicator);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, communicator);
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);

    const bool valid = globalMax <= 1e6 && globalMin >= -1e6;
    if (!valid && rank == 0) {
        printf("Validation failed: values out of expected range\n");
    }
    return valid;
}

static std::vector<Real> packLocalGrid(const Real* grid, const Layout& layout) {
    const size_t count = static_cast<size_t>(layout.lx) * layout.ly * layout.lz;
    std::vector<Real> packed(count);
    size_t position = 0;
    for (int z = 1; z <= layout.lz; ++z) {
        for (int y = 1; y <= layout.ly; ++y) {
            const Real* source = grid + layout.index(1, y, z);
            std::copy_n(source, layout.lx, packed.data() + position);
            position += static_cast<size_t>(layout.lx);
        }
    }
    return packed;
}

static void sendLargeBuffer(const Real* buffer, const size_t count, const int destination,
                            MPI_Comm communicator) {
    size_t sent = 0;
    while (sent < count) {
        const int chunk = static_cast<int>(std::min(
            count - sent, static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Send(buffer + sent, chunk, MPI_DOUBLE, destination, GatherResult, communicator);
        sent += static_cast<size_t>(chunk);
    }
}

static void receiveLargeBuffer(Real* buffer, const size_t count, const int source,
                               MPI_Comm communicator) {
    size_t received = 0;
    while (received < count) {
        const int chunk = static_cast<int>(std::min(
            count - received, static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Recv(buffer + received, chunk, MPI_DOUBLE, source, GatherResult,
                 communicator, MPI_STATUS_IGNORE);
        received += static_cast<size_t>(chunk);
    }
}

// Reconstruct only when external result reporting is requested. Normal benchmark
// runs never allocate the global grid on any rank.
static std::vector<Real> gatherGlobalGrid(const Real* grid, const Domain& domain,
                                          const ProcessGrid& processGrid,
                                          MPI_Comm communicator, const int rank,
                                          const int processCount) {
    std::vector<Real> localPacked = packLocalGrid(grid, domain.local);
    if (rank != 0) {
        sendLargeBuffer(localPacked.data(), localPacked.size(), 0, communicator);
        return {};
    }

    std::vector<Real> global(domain.nx * domain.ny * domain.nz);
    std::vector<Real> received;
    for (int sourceRank = 0; sourceRank < processCount; ++sourceRank) {
        int coordinates[3] = {0, 0, 0};
        MPI_Cart_coords(communicator, sourceRank, 3, coordinates);
        const auto [sx, lxSize] = blockPartition(domain.nx, processGrid.px, coordinates[2]);
        const auto [sy, lySize] = blockPartition(domain.ny, processGrid.py, coordinates[1]);
        const auto [sz, lzSize] = blockPartition(domain.nz, processGrid.pz, coordinates[0]);
        const size_t count = lxSize * lySize * lzSize;

        const Real* packed = nullptr;
        if (sourceRank == 0) {
            packed = localPacked.data();
        } else {
            received.resize(count);
            receiveLargeBuffer(received.data(), count, sourceRank, communicator);
            packed = received.data();
        }

        size_t position = 0;
        for (size_t z = 0; z < lzSize; ++z) {
            for (size_t y = 0; y < lySize; ++y) {
                Real* destination = global.data() + idx3(sx, sy + y, sz + z,
                                                          domain.nx, domain.ny);
                std::copy_n(packed + position, lxSize, destination);
                position += lxSize;
            }
        }
    }
    return global;
}

static void printUsage(const char* programName) {
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            argumentsValid = false;
            break;
        }
    }

    if (!argumentsValid || showHelp) {
        if (showHelp && worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const size_t maxIntDimension = static_cast<size_t>(std::numeric_limits<int>::max() - 2);
    bool dimensionsValid = nx > 0 && ny > 0 && nz > 0 &&
                           nx <= maxIntDimension && ny <= maxIntDimension &&
                           nz <= maxIntDimension && iterations >= 0;
    bool sizeValid = dimensionsValid;
    if (sizeValid) {
        sizeValid = nx <= std::numeric_limits<size_t>::max() / ny;
        if (sizeValid) {
            const size_t xy = nx * ny;
            sizeValid = xy <= std::numeric_limits<size_t>::max() / nz;
        }
    }
    if (!sizeValid) {
        if (worldRank == 0) {
            fprintf(stderr, "Grid dimensions and iteration count must be positive, valid values.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const ProcessGrid processGrid = chooseProcessGrid(processCount, nx, ny, nz);
    if (processGrid.px == 0) {
        if (worldRank == 0) {
            fprintf(stderr,
                    "Cannot map %d MPI processes onto a %zu x %zu x %zu grid without empty ranks.\n",
                    processCount, nx, ny, nz);
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[3] = {processGrid.pz, processGrid.py, processGrid.px};
    int periods[3] = {0, 0, 0};
    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, periods, 0, &cartesian);

    int rank = 0;
    int coordinates[3] = {0, 0, 0};
    MPI_Comm_rank(cartesian, &rank);
    MPI_Cart_coords(cartesian, rank, 3, coordinates);

    Domain domain;
    domain.nx = nx;
    domain.ny = ny;
    domain.nz = nz;
    const auto [sx, lxSize] = blockPartition(nx, processGrid.px, coordinates[2]);
    const auto [sy, lySize] = blockPartition(ny, processGrid.py, coordinates[1]);
    const auto [sz, lzSize] = blockPartition(nz, processGrid.pz, coordinates[0]);
    domain.sx = sx;
    domain.sy = sy;
    domain.sz = sz;
    domain.local.lx = static_cast<int>(lxSize);
    domain.local.ly = static_cast<int>(lySize);
    domain.local.lz = static_cast<int>(lzSize);
    domain.local.pitch = lxSize + 2;
    domain.local.plane = domain.local.pitch * (lySize + 2);

    Neighbors neighbors;
    MPI_Cart_shift(cartesian, 2, 1, &neighbors.xm, &neighbors.xp);
    MPI_Cart_shift(cartesian, 1, 1, &neighbors.ym, &neighbors.yp);
    MPI_Cart_shift(cartesian, 0, 1, &neighbors.zm, &neighbors.zp);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", processCount);
        printf("Process grid: %d x %d x %d\n",
               processGrid.px, processGrid.py, processGrid.pz);
        printf("Initializing grid...\n");
    }

    const size_t localGridSize = domain.local.plane * (lzSize + 2);
    std::vector<Real> grid1(localGridSize);
    std::vector<Real> grid2(localGridSize);
    initializeGrid(grid1, grid2, domain);

    FaceTypes faceTypes = createFaceTypes(domain.local);
    std::array<std::vector<MPI_Request>, 2> exchanges = {
        createPersistentExchange(grid1.data(), domain.local, neighbors, faceTypes, cartesian),
        createPersistentExchange(grid2.data(), domain.local, neighbors, faceTypes, cartesian)
    };
    const UpdateRanges ranges{
        makeAxisRange(domain.sx, domain.local.lx, nx, neighbors.xm, neighbors.xp),
        makeAxisRange(domain.sy, domain.local.ly, ny, neighbors.ym, neighbors.yp),
        makeAxisRange(domain.sz, domain.local.lz, nz, neighbors.zm, neighbors.zp)
    };

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cartesian);
    const double start = MPI_Wtime();

    Real* input = grid1.data();
    Real* output = grid2.data();
    int inputBuffer = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stencilIteration(input, output, domain.local, ranges, exchanges[inputBuffer]);
        std::swap(input, output);
        inputBuffer ^= 1;
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);

    if (rank == 0) {
        const long long durationMilliseconds = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMilliseconds);
        const double interiorCells = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                                     static_cast<double>(ny > 2 ? ny - 2 : 0) *
                                     static_cast<double>(nz > 2 ? nz - 2 : 0);
        const double cellUpdates = interiorCells * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<Real> global = gatherGlobalGrid(input, domain, processGrid,
                                                    cartesian, rank, processCount);
        if (rank == 0) print_results(global, "Grid");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateDistributed(input, domain, cartesian, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    freePersistentExchange(exchanges[0]);
    freePersistentExchange(exchanges[1]);
    freeFaceTypes(faceTypes);
    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return valid ? 0 : 1;
}
