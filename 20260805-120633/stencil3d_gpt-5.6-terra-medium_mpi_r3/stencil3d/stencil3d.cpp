#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t firstZ, const size_t localNz) {
    const size_t planeSize = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalBase = (firstZ + z) * planeSize;
        Real* const plane = grid.data() + (z + 1) * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            plane[i] = static_cast<Real>((globalBase + i) % 19);
        }
    }
}

// Copy physical-domain boundary cells.  Halo planes are never part of output.
void copyBoundaries(const std::vector<Real>& input, std::vector<Real>& output,
                    const size_t nx, const size_t ny, const size_t nz,
                    const size_t firstZ, const size_t localNz) {
    const size_t planeSize = nx * ny;
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t globalZ = firstZ + lz - 1;
        const Real* const in = input.data() + lz * planeSize;
        Real* const out = output.data() + lz * planeSize;
        if (globalZ == 0 || globalZ + 1 == nz) {
            std::memcpy(out, in, planeSize * sizeof(Real));
            continue;
        }
        std::memcpy(out, in, nx * sizeof(Real));
        if (ny > 1) {
            std::memcpy(out + (ny - 1) * nx, in + (ny - 1) * nx, nx * sizeof(Real));
        }
        for (size_t y = 1; y + 1 < ny; ++y) {
            out[y * nx] = in[y * nx];
            if (nx > 1) out[y * nx + nx - 1] = in[y * nx + nx - 1];
        }
    }
}

inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t lz) {
    const size_t planeSize = nx * ny;
    const size_t base = lz * planeSize;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] +
                         input[i - planeSize] + input[i + planeSize]) / 7.0;
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t firstZ, const size_t localNz, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    copyBoundaries(input, output, nx, ny, nz, firstZ, localNz);

    MPI_Request requests[4];
    int requestCount = 0;
    if (rank > 0) {
        MPI_Irecv(input.data(), static_cast<int>(planeSize), MPI_DOUBLE,
                  rank - 1, 1, comm, &requests[requestCount++]);
        MPI_Isend(input.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
                  rank - 1, 0, comm, &requests[requestCount++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(input.data() + (localNz + 1) * planeSize,
                  static_cast<int>(planeSize), MPI_DOUBLE, rank + 1, 0, comm,
                  &requests[requestCount++]);
        MPI_Isend(input.data() + localNz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
                  rank + 1, 1, comm, &requests[requestCount++]);
    }

    // These planes do not depend on incoming halos, so communication overlaps useful work.
    for (size_t lz = 2; lz < localNz; ++lz) {
        stencilPlane(input, output, nx, ny, lz);
    }
    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);

    if (firstZ > 0) stencilPlane(input, output, nx, ny, 1);
    if (localNz > 1 && firstZ + localNz < nz) {
        stencilPlane(input, output, nx, ny, localNz);
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();
    for (size_t z = 1; z <= localNz; ++z) {
        const Real* plane = grid.data() + z * planeSize;
        for (size_t i = 0; i < planeSize; ++i) {
            const Real value = plane[i];
            localBad |= !std::isfinite(value);
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    }
    int bad;
    Real minValue, maxValue;
    MPI_Allreduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        if (bad) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
        if (maxValue > 1e6 || minValue < -1e6) {
            printf("Validation failed: values out of expected range\n");
            bad = 1;
        }
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, comm);
    return bad == 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (worldRank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) / ny) {
        if (worldRank == 0) printf("Grid dimensions must be at least 3 and each Z plane must fit MPI's count range.\n");
        MPI_Finalize();
        return 1;
    }

    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeRanks) { MPI_Finalize(); return 0; }

    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t firstZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\nMPI ranks: %d\nInitializing grid...\nRunning stencil computation...\n", validate ? "enabled" : "disabled", ranks);
    }
    std::vector<Real> grid1((localNz + 2) * planeSize), grid2((localNz + 2) * planeSize);
    initializeGrid(grid1, nx, ny, firstZ, localNz);

    MPI_Barrier(comm);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        stencilIteration(grid1, grid2, nx, ny, nz, firstZ, localNz, comm);
        grid1.swap(grid2);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const auto milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> fullGrid;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks); fullGrid.resize(nx * ny * nz);
            for (int r = 0; r < ranks; ++r) {
                const size_t countZ = base + (static_cast<size_t>(r) < remainder);
                counts[r] = static_cast<int>(countZ * planeSize);
                displacements[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * planeSize);
            }
        }
        MPI_Gatherv(grid1.data() + planeSize, static_cast<int>(localNz * planeSize), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(fullGrid, "Grid");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        if (!validateResult(grid1, nx, ny, localNz, comm)) exitCode = 1;
        if (rank == 0) printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED");
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
