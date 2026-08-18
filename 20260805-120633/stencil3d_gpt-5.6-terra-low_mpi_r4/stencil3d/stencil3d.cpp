#include <algorithm>
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

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t globalZ) {
    const size_t plane = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t offset = (z + 1) * plane;
        const size_t globalOffset = (globalZ + z) * plane;
        for (size_t i = 0; i < plane; ++i) grid[offset + i] = static_cast<Real>((globalOffset + i) % 19);
    }
}

// Calculates complete planes in [first, last] in local (halo-padded) coordinates.
inline void stencilPlanes(const std::vector<Real>& input, std::vector<Real>& output,
                          size_t nx, size_t ny, size_t globalZ,
                          size_t globalNz, size_t first, size_t last) {
    if (first > last) return;
    const size_t plane = nx * ny;
    for (size_t z = first; z <= last; ++z) {
        const size_t gz = globalZ + z - 1;
        const size_t base = z * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = base + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = row + x;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == globalNz) {
                    output[i] = input[i];
                } else {
                    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                                 input[i - nx] + input[i + nx] +
                                 input[i - plane] + input[i + plane]) / 7.0;
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz,
                    MPI_Comm comm, int rank) {
    const size_t count = nx * ny * localNz;
    const size_t plane = nx * ny;
    int localFinite = 1;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const Real value = grid[plane + i];
        if (!std::isfinite(value)) localFinite = 0;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
    }
    int finite = 0;
    Real minVal, maxVal;
    MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_LAND, comm);
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (!finite) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    }
    return finite && maxVal <= 1e6 && minVal >= -1e6;
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
    int argumentError = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } argumentError = 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 1 || iterations < 0) argumentError = 1;
    int anyError = 0;
    MPI_Allreduce(&argumentError, &anyError, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (anyError) { if (worldRank == 0) printf("Invalid grid dimensions or iteration count\n"); MPI_Finalize(); return 1; }

    const int activeSize = std::min<size_t>(worldSize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeSize) { MPI_Finalize(); return 0; }
    int rank;
    MPI_Comm_rank(comm, &rank);
    const size_t baseNz = nz / activeSize, remainder = nz % activeSize;
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI ranks: %d)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n",
               activeSize, nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter & 1) ? grid2 : grid1;
        std::vector<Real>& output = (iter & 1) ? grid1 : grid2;
        MPI_Request requests[4]; int requestCount = 0;
        const int below = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int above = rank + 1 == activeSize ? MPI_PROC_NULL : rank + 1;
        MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE, below, 1, comm, &requests[requestCount++]);
        MPI_Irecv(input.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, above, 0, comm, &requests[requestCount++]);
        MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE, below, 0, comm, &requests[requestCount++]);
        MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, above, 1, comm, &requests[requestCount++]);
        if (localNz > 2) stencilPlanes(input, output, nx, ny, globalZ, nz, 2, localNz - 1);
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        stencilPlanes(input, output, nx, ny, globalZ, nz, 1, 1);
        if (localNz > 1) stencilPlanes(input, output, nx, ny, globalZ, nz, localNz, localNz);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const std::vector<Real>& finalGrid = (iterations & 1) ? grid2 : grid1;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz > 1 ? nz - 2 : 0)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }
    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> global;
        if (rank == 0) { counts.resize(activeSize); displacements.resize(activeSize); global.resize(nx * ny * nz); }
        const int sendCount = static_cast<int>(localNz * plane);
        if (rank == 0) for (int r = 0; r < activeSize; ++r) { const size_t n = baseNz + (static_cast<size_t>(r) < remainder); counts[r] = static_cast<int>(n * plane); displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder)) * plane); }
        MPI_Gatherv(finalGrid.data() + plane, sendCount, MPI_DOUBLE, rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(global, "Grid");
    }
    int valid = !validate || validateResult(finalGrid, nx, ny, localNz, comm, rank);
    if (validate && rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
