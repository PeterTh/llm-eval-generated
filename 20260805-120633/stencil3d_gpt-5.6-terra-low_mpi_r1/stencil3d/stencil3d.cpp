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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t firstZ) {
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, firstZ + z - 1, nx, ny);
                grid[idx3(x, y, z, nx, ny)] = static_cast<Real>(globalIndex % 19);
            }
}

// Each rank owns localNz planes in [1, localNz].  Planes 0 and localNz+1
// are halo planes received from its adjacent Z-neighbours.
void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      size_t nx, size_t ny, size_t localNz, size_t firstZ,
                      size_t globalNz, int rank, int ranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    std::copy(input.begin(), input.end(), output.begin()); // preserves all physical boundaries

    MPI_Request requests[4];
    int requestCount = 0;
    if (rank > 0) {
        MPI_Irecv(output.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11, comm, &requests[requestCount++]);
        MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 12, comm, &requests[requestCount++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(output.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 12, comm, &requests[requestCount++]);
        MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 11, comm, &requests[requestCount++]);
    }

    // Most planes do not depend on a received halo, so compute them while the
    // two boundary planes are in flight.  Received planes are held in output's
    // halo slots; owned output values are therefore never clobbered.
    const auto computePlane = [&](size_t z) {
        const size_t globalZ = firstZ + z - 1;
        if (globalZ == 0 || globalZ + 1 == globalNz) return;
        for (size_t y = 1; y + 1 < ny; ++y)
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - nx] + input[i + nx] +
                             ((z == 1 && rank > 0) ? output[i - plane] : input[i - plane]) +
                             ((z == localNz && rank + 1 < ranks) ? output[i + plane] : input[i + plane])) / 7.0;
            }
    };
    for (size_t z = 2; z < localNz; ++z) computePlane(z);
    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    if (localNz > 0) computePlane(1);
    if (localNz > 1) computePlane(localNz);
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, MPI_Comm comm, int rank) {
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();
    for (size_t z = 1; z <= localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const Real value = grid[idx3(x, y, z, nx, ny)];
                localBad |= !std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
    int bad = 0;
    Real minValue, maxValue;
    MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, comm);
    MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;
    if (bad) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 1e6 || minValue < -1e6) { printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    bool usage = false, badOption = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) usage = true;
        else { badOption = true; if (worldRank == 0) printf("Unknown option: %s\n", argv[i]); }
    }
    if (usage || badOption) { if (worldRank == 0) printUsage(argv[0]); MPI_Finalize(); return badOption; }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 1 || ny < 1 || nz < 1 || iterations < 0 || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) printf("Grid dimensions and iteration count are invalid for MPI execution\n");
        MPI_Finalize(); return 1;
    }

    // Only ranks with a plane participate; this also supports more ranks than Z planes.
    const int activeRanks = std::min(worldSize, static_cast<int>(nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeRanks) { MPI_Finalize(); return 0; }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &ranks);
    const size_t base = nz / ranks, remainder = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t firstZ = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, firstZ);

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(comm);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, localNz, firstZ, nz, rank, ranks, comm);
        else stencilIteration(grid2, grid1, nx, ny, localNz, firstZ, nz, rank, ranks, comm);
    }
    MPI_Barrier(comm);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double updates = static_cast<double>((nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }
    const std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;
    if (printResults) {
        std::vector<int> counts, offsets;
        std::vector<Real> fullGrid;
        if (rank == 0) { counts.resize(ranks); offsets.resize(ranks); for (int r = 0; r < ranks; ++r) { const size_t n = base + (static_cast<size_t>(r) < remainder); counts[r] = static_cast<int>(n * plane); offsets[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * plane); } fullGrid.resize(nx * ny * nz); }
        MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE, fullGrid.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(fullGrid, "Grid");
    }
    bool valid = true;
    if (validate) { if (rank == 0) printf("Validating result...\n"); valid = validateResult(finalGrid, nx, ny, localNz, comm, rank); if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return (rank == 0 && !valid) ? 1 : 0;
}
