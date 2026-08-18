#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

struct MpiGuard {
    MpiGuard(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
    ~MpiGuard() { MPI_Finalize(); }
    MpiGuard(const MpiGuard&) = delete;
    MpiGuard& operator=(const MpiGuard&) = delete;
};

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Every rank initializes its owned planes directly from their global indices.
void initializeGrid(std::vector<Real>& grid, const size_t planeSize,
                    const size_t localNz, const size_t globalZ0) {
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t globalBase = (globalZ0 + lz - 1) * planeSize;
        Real* const plane = grid.data() + lz * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            plane[i] = static_cast<Real>((globalBase + i) % 19);
        }
    }
}

// Compute one owned XY plane. Restrict-qualified pointers and a unit-stride
// inner loop allow the compiler to vectorize the bandwidth-bound stencil.
void stencilPlane(const Real* __restrict input, Real* __restrict output,
                  const size_t nx, const size_t ny, const size_t lz,
                  const size_t globalZ, const size_t nz) {
    const size_t planeSize = nx * ny;
    const size_t base = lz * planeSize;

    if (globalZ == 0 || globalZ + 1 == nz) {
        std::copy_n(input + base, planeSize, output + base);
        return;
    }

    // Global Y boundary rows are fixed.
    std::copy_n(input + base, nx, output + base);
    if (ny > 1) {
        std::copy_n(input + base + (ny - 1) * nx, nx,
                    output + base + (ny - 1) * nx);
    }

    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        output[row] = input[row];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - planeSize] + input[i + planeSize]) / 7.0;
        }
        if (nx > 1) output[row + nx - 1] = input[row + nx - 1];
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t localNz, const size_t globalZ0,
                      const int rank, const int ranks, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    const int planeCount = static_cast<int>(planeSize);
    MPI_Request receiveRequests[2];
    MPI_Request sendRequests[2];
    int receiveCount = 0;
    int sendCount = 0;

    // Receive first, then send. The lower and upper directions use distinct
    // tags, so this also works for two-rank periodic-looking tag patterns.
    if (rank > 0) {
        MPI_Irecv(input.data(), planeCount, MPI_DOUBLE, rank - 1, 1, comm,
                  &receiveRequests[receiveCount++]);
        MPI_Isend(input.data() + planeSize, planeCount, MPI_DOUBLE, rank - 1, 0,
                  comm, &sendRequests[sendCount++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(input.data() + (localNz + 1) * planeSize, planeCount,
                  MPI_DOUBLE, rank + 1, 0, comm, &receiveRequests[receiveCount++]);
        MPI_Isend(input.data() + localNz * planeSize, planeCount, MPI_DOUBLE,
                  rank + 1, 1, comm, &sendRequests[sendCount++]);
    }

    // Planes not adjacent to a rank boundary do not depend on incoming halos.
    for (size_t lz = 2; lz < localNz; ++lz) {
        stencilPlane(input.data(), output.data(), nx, ny, lz,
                     globalZ0 + lz - 1, nz);
    }

    // Only incoming halos gate the edge computation; outgoing transfers can
    // continue concurrently because the input buffer remains read-only.
    if (receiveCount != 0)
        MPI_Waitall(receiveCount, receiveRequests, MPI_STATUSES_IGNORE);

    stencilPlane(input.data(), output.data(), nx, ny, 1, globalZ0, nz);
    if (localNz > 1) {
        stencilPlane(input.data(), output.data(), nx, ny, localNz,
                     globalZ0 + localNz - 1, nz);
    }
    if (sendCount != 0) MPI_Waitall(sendCount, sendRequests, MPI_STATUSES_IGNORE);
}

bool validateResult(const std::vector<Real>& grid, const size_t planeSize,
                    const size_t localNz, MPI_Comm comm, const int rank) {
    int localFinite = 1;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t i = planeSize; i < (localNz + 1) * planeSize; ++i) {
        const Real value = grid[i];
        if (!std::isfinite(value)) localFinite = 0;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }

    int allFinite = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localFinite, &allFinite, 1, MPI_INT, MPI_LAND, comm);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (!allFinite) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6)
            std::printf("Validation failed: values out of expected range\n");
    }
    return allFinite && globalMax <= 1e6 && globalMin >= -1e6;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// MPI-3 uses int message counts. Transfer result chunks explicitly so result
// printing continues to work when the complete grid exceeds INT_MAX elements.
std::vector<Real> gatherResult(const std::vector<Real>& localGrid,
                               const size_t planeSize, const size_t localNz,
                               const size_t nz, const int rank, const int ranks,
                               MPI_Comm comm) {
    constexpr int tag = 7;
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    const Real* const localData = localGrid.data() + planeSize;

    if (rank != 0) {
        const size_t count = localNz * planeSize;
        for (size_t offset = 0; offset < count; offset += std::min(maxChunk, count - offset)) {
            const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
            MPI_Send(localData + offset, chunk, MPI_DOUBLE, 0, tag, comm);
        }
        return {};
    }

    std::vector<Real> result(planeSize * nz);
    std::copy_n(localData, localNz * planeSize, result.data());
    const size_t basePlanes = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    for (int source = 1; source < ranks; ++source) {
        const size_t sourceNz = basePlanes +
            (static_cast<size_t>(source) < remainder ? 1 : 0);
        const size_t sourceZ0 = static_cast<size_t>(source) * basePlanes +
            std::min(static_cast<size_t>(source), remainder);
        const size_t count = sourceNz * planeSize;
        for (size_t offset = 0; offset < count; offset += std::min(maxChunk, count - offset)) {
            const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
            MPI_Recv(result.data() + sourceZ0 * planeSize + offset, chunk,
                     MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE);
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    MpiGuard mpi(argc, argv);
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (help || !argumentsValid) {
        if (worldRank == 0) printUsage(argv[0]);
        return argumentsValid ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool dimensionsValid = nx >= 2 && ny >= 2 && nz >= 2 && iterations >= 0 &&
        nx <= std::numeric_limits<size_t>::max() / ny &&
        nx * ny <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
        nx * ny <= std::numeric_limits<size_t>::max() / nz;
    if (!dimensionsValid) {
        if (worldRank == 0)
            std::fprintf(stderr, "Grid dimensions must be at least 2, iterations non-negative, "
                                 "and an XY plane must fit in an MPI count.\n");
        return 1;
    }

    // No rank is assigned an empty slab. Excess launched ranks remain outside
    // the active communicator, which avoids invalid zero-length neighbors.
    const int activeRanks = nz < static_cast<size_t>(worldSize)
        ? static_cast<int>(nz) : worldSize;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) return 0;

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    const size_t basePlanes = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * basePlanes +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }

    std::vector<Real> grid1((localNz + 2) * planeSize);
    std::vector<Real> grid2((localNz + 2) * planeSize);
    initializeGrid(grid1, planeSize, localNz, globalZ0);

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0)
            stencilIteration(grid1, grid2, nx, ny, nz, localNz, globalZ0, rank, ranks, comm);
        else
            stencilIteration(grid2, grid1, nx, ny, nz, localNz, globalZ0, rank, ranks, comm);
    }

    const double localTime = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localTime, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double cellUpdates = static_cast<double>(nx - 2) *
            static_cast<double>(ny - 2) * static_cast<double>(nz - 2) * iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = ((iterations & 1) == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> globalGrid = gatherResult(finalGrid, planeSize, localNz,
                                                    nz, rank, ranks, comm);
        if (rank == 0) print_results(globalGrid, "Grid");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(finalGrid, planeSize, localNz, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}
