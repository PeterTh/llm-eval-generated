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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, const size_t globalZ0) {
    const size_t plane = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalBase = (globalZ0 + z) * plane;
        Real* const dst = grid.data() + (z + 1) * plane;
        for (size_t i = 0; i < plane; ++i)
            dst[i] = static_cast<Real>((globalBase + i) % 19);
    }
}

// Update one owned plane. localZ includes the lower ghost plane at index 0.
inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t localZ,
                         const size_t globalZ, const size_t globalNz) {
    const size_t plane = nx * ny;
    const size_t base = localZ * plane;
    if (globalZ == 0 || globalZ + 1 == globalNz) {
        std::copy_n(input.data() + base, plane, output.data() + base);
        return;
    }

    // The x/y boundary is immutable.  Copying the plane first also makes the
    // routine correct for thin x or y dimensions.
    std::copy_n(input.data() + base, plane, output.data() + base);
    if (nx < 3 || ny < 3)
        return;

    const Real* const below = input.data() + base - plane;
    const Real* const here = input.data() + base;
    const Real* const above = input.data() + base + plane;
    Real* const dst = output.data() + base;
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            dst[i] = (here[i] + here[i - 1] + here[i + 1] + here[i - nx] +
                      here[i + nx] + below[i] + above[i]) / 7.0;
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t localNz,
                      const size_t globalZ0, const size_t globalNz,
                      const int rank, const int activeRanks) {
    if (localNz == 0)
        return;

    const size_t plane = nx * ny;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == activeRanks ? MPI_PROC_NULL : rank + 1;
    MPI_Request requests[4];
    int count = 0;

    // Each neighbor needs the owned plane adjacent to it.  Separate receive
    // buffers avoid packing and permit computation while communication runs.
    MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 1,
              MPI_COMM_WORLD, &requests[count++]);
    MPI_Irecv(input.data() + (localNz + 1) * plane, static_cast<int>(plane),
              MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &requests[count++]);
    MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lower,
              0, MPI_COMM_WORLD, &requests[count++]);
    MPI_Isend(input.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
              upper, 1, MPI_COMM_WORLD, &requests[count++]);

    // These planes never consume a received halo.
    for (size_t z = 2; z < localNz; ++z)
        stencilPlane(input, output, nx, ny, z, globalZ0 + z - 1, globalNz);

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    stencilPlane(input, output, nx, ny, 1, globalZ0, globalNz);
    if (localNz > 1)
        stencilPlane(input, output, nx, ny, localNz, globalZ0 + localNz - 1, globalNz);
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    for (const Real val : grid) {
        if (!std::isfinite(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseStatus = 1; }
    }
    if (parseStatus) { MPI_Finalize(); return parseStatus; }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t localNz = rank < activeRanks
        ? nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks ? 1 : 0) : 0;
    const size_t globalZ0 = rank < activeRanks
        ? static_cast<size_t>(rank) * (nz / activeRanks) + std::min(static_cast<size_t>(rank), nz % activeRanks) : 0;
    const size_t plane = nx * ny;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI: %d ranks)\nGrid size: %zu x %zu x %zu\n", ranks, nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n", iterations, validate ? "enabled" : "disabled");
    }
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ0);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (rank < activeRanks)
            stencilIteration(iter % 2 == 0 ? grid1 : grid2, iter % 2 == 0 ? grid2 : grid1,
                             nx, ny, localNz, globalZ0, nz, rank, activeRanks);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::vector<Real>& finalLocal = iterations % 2 == 0 ? grid1 : grid2;
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < activeRanks; ++r) {
                const size_t slabs = nz / activeRanks + (static_cast<size_t>(r) < nz % activeRanks);
                counts[r] = static_cast<int>(slabs * plane);
                displacements[r] = static_cast<int>((static_cast<size_t>(r) * (nz / activeRanks) + std::min(static_cast<size_t>(r), nz % activeRanks)) * plane);
            }
            for (int r = activeRanks; r < ranks; ++r) counts[r] = displacements[r] = 0;
            finalGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(localNz ? finalLocal.data() + plane : nullptr, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int exitStatus = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) { printf("Validating result...\n"); exitStatus = validateResult(finalGrid) ? 0 : 1; printf("Validation: %s\n", exitStatus ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
