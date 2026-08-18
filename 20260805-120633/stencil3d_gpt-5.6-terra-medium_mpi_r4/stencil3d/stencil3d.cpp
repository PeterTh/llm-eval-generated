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
                    const size_t localNz, const size_t zBegin) {
    const size_t planeSize = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalZ = zBegin + z;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = globalZ * planeSize + y * nx + x;
                grid[idx3(x, y, z + 1, nx, ny)] = (globalIndex % 19) * 1.0;
            }
        }
    }
}

// Update one owned z plane.  The local arrays have a halo plane at z=0 and
// z=localNz+1; owned data starts at z=1.
inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t nz,
                         const size_t localZ, const size_t globalZ) {
    if (globalZ == 0 || globalZ + 1 == nz) return;
    for (size_t y = 1; y + 1 < ny; ++y) {
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t idx = idx3(x, y, localZ, nx, ny);
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                           input[idx - nx] + input[idx + nx] +
                           input[idx - nx * ny] + input[idx + nx * ny]) / 7.0;
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t localNz, const size_t zBegin,
                      const int previousRank, const int nextRank,
                      MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    const int planeCount = static_cast<int>(planeSize);

    // Preserve every global boundary (and x/y edges) before overwriting the
    // actual 3-D interior.  This also leaves output fully defined each step.
    std::copy(input.begin() + planeSize, input.begin() + (localNz + 1) * planeSize,
              output.begin() + planeSize);

    MPI_Request requests[4];
    MPI_Irecv(input.data(), planeCount, MPI_DOUBLE, previousRank, 1, comm, &requests[0]);
    MPI_Irecv(input.data() + (localNz + 1) * planeSize, planeCount, MPI_DOUBLE,
              nextRank, 0, comm, &requests[1]);
    MPI_Isend(input.data() + planeSize, planeCount, MPI_DOUBLE, previousRank, 0,
              comm, &requests[2]);
    MPI_Isend(input.data() + localNz * planeSize, planeCount, MPI_DOUBLE, nextRank, 1,
              comm, &requests[3]);

    // Planes not adjacent to a rank boundary do not depend on communication.
    for (size_t localZ = 2; localZ < localNz; ++localZ) {
        stencilPlane(input, output, nx, ny, nz, localZ, zBegin + localZ - 1);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    stencilPlane(input, output, nx, ny, nz, 1, zBegin);
    if (localNz > 1) {
        stencilPlane(input, output, nx, ny, nz, localNz, zBegin + localNz - 1);
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, MPI_Comm comm, const int rank) {
    const size_t planeSize = nx * ny;
    bool hasInvalid = false;
    Real minVal = grid[planeSize];
    Real maxVal = grid[planeSize];
    for (size_t i = planeSize; i < (localNz + 1) * planeSize; ++i) {
        const Real value = grid[i];
        hasInvalid = hasInvalid || std::isnan(value) || std::isinf(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    int invalid = hasInvalid ? 1 : 0;
    int globalInvalid = 0;
    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_LOR, 0, comm);
    MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank != 0) return globalInvalid == 0;
    if (globalInvalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, showHelp = false, badOption = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) showHelp = true;
        else { badOption = true; if (worldRank == 0) printf("Unknown option: %s\n", argv[i]); }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (showHelp || badOption || nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) printf("Invalid grid size or iteration count\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (showHelp && !badOption) ? 0 : 1;
    }

    const int activeSize = std::min<int>(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeSize) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const size_t zBegin = (nz * rank) / activeSize;
    const size_t zEnd = (nz * (rank + 1)) / activeSize;
    const size_t localNz = zEnd - zBegin;
    const size_t planeSize = nx * ny;
    if (planeSize > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        localNz * planeSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Grid is too large for MPI message counts\n");
        MPI_Comm_free(&comm);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("MPI ranks: %d\n", activeSize);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((localNz + 2) * planeSize);
    std::vector<Real> grid2((localNz + 2) * planeSize);
    initializeGrid(grid1, nx, ny, localNz, zBegin);

    const int previousRank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int nextRank = rank + 1 == activeSize ? MPI_PROC_NULL : rank + 1;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        stencilIteration(grid1, grid2, nx, ny, nz, localNz, zBegin,
                         previousRank, nextRank, comm);
        grid1.swap(grid2);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const size_t interiorX = nx > 2 ? nx - 2 : 0;
        const size_t interiorY = ny > 2 ? ny - 2 : 0;
        const size_t interiorZ = nz > 2 ? nz - 2 : 0;
        const double cellUpdates = static_cast<double>(interiorX * interiorY * interiorZ) * iterations;
        const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> globalGrid;
        if (rank == 0) {
            counts.resize(activeSize);
            displacements.resize(activeSize);
            for (int r = 0; r < activeSize; ++r) {
                const size_t begin = nz * r / activeSize;
                const size_t end = nz * (r + 1) / activeSize;
                counts[r] = static_cast<int>((end - begin) * planeSize);
                displacements[r] = static_cast<int>(begin * planeSize);
            }
            globalGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(grid1.data() + planeSize, static_cast<int>(localNz * planeSize), MPI_DOUBLE,
                    rank == 0 ? globalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(globalGrid, "Grid");
    }

    int localValid = !validate || validateResult(grid1, nx, ny, localNz, comm, rank);
    int valid = 0;
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, comm);
    if (validate && rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

    MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
