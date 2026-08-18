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

// A rank owns contiguous global z planes.  Planes 0 and localNz + 1 are halos.
void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t firstZ) {
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = firstZ + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, globalZ, nx, ny);
                grid[idx3(x, y, z, nx, ny)] = static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         size_t z, size_t globalZ, size_t nx, size_t ny, size_t nz) {
    const size_t planeSize = nx * ny;
    const size_t base = z * planeSize;
    // Copying the complete plane first preserves global z and x/y boundary values.
    std::memcpy(output.data() + base, input.data() + base, planeSize * sizeof(Real));
    if (globalZ == 0 || globalZ + 1 == nz) return;

    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t p = row + x;
            output[p] = (input[p] + input[p - 1] + input[p + 1] + input[p - nx] + input[p + nx]
                         + input[p - planeSize] + input[p + planeSize]) / 7.0;
        }
    }
}

// Exchange z-face halos, computing planes that do not depend on them while messages progress.
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      size_t nx, size_t ny, size_t nz, size_t localNz, size_t firstZ,
                      int previous, int next, MPI_Comm comm) {
    const size_t planeSize = nx * ny;
    MPI_Request requests[4];
    int requestCount = 0;
    MPI_Irecv(input.data(), static_cast<int>(planeSize), MPI_DOUBLE,
              previous, 1, comm, &requests[requestCount++]);
    MPI_Irecv(input.data() + (localNz + 1) * planeSize,
              static_cast<int>(planeSize), MPI_DOUBLE, next, 0, comm, &requests[requestCount++]);
    MPI_Isend(input.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
              previous, 0, comm, &requests[requestCount++]);
    MPI_Isend(input.data() + localNz * planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
              next, 1, comm, &requests[requestCount++]);

    for (size_t z = 2; z < localNz; ++z)
        stencilPlane(input, output, z, firstZ + z - 1, nx, ny, nz);

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    stencilPlane(input, output, 1, firstZ, nx, ny, nz);
    if (localNz > 1)
        stencilPlane(input, output, localNz, firstZ + localNz - 1, nx, ny, nz);
}

bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, MPI_Comm comm, int rank) {
    const size_t begin = nx * ny;
    const size_t end = begin + localNz * nx * ny;
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();
    for (size_t i = begin; i < end; ++i) {
        if (!std::isfinite(grid[i])) localBad = 1;
        localMin = std::min(localMin, grid[i]);
        localMax = std::max(localMax, grid[i]);
    }
    int bad = 0;
    Real minValue, maxValue;
    MPI_Allreduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (bad) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
        if (maxValue > 1e6 || minValue < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            bad = 1;
        }
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, comm);
    return bad == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("  -x <num>  Grid X size (default: 128)\n  -y <num>  Grid Y size (default: X)\n");
    std::printf("  -z <num>  Grid Z size (default: X)\n  -i <num>  Iterations (default: 10)\n");
    std::printf("  -v        Enable validation\n  -r        Print results\n  -h        Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else badArgs = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || badArgs || nx < 3 || ny < 3 || nz < 3 || iterations < 0 || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) { if (badArgs) std::printf("Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return (help && !badArgs) ? 0 : 1;
    }

    // Ranks beyond nz own no plane and are excluded, allowing arbitrary launch sizes.
    const int active = std::min<int>(worldSize, static_cast<int>(nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= active) { MPI_Finalize(); return 0; }
    int rank, size;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);

    const size_t basePlanes = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    const size_t localNz = basePlanes + (static_cast<size_t>(rank) < remainder);
    const size_t firstZ = static_cast<size_t>(rank) * basePlanes + std::min(static_cast<size_t>(rank), remainder);
    const size_t planeSize = nx * ny;
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == size ? MPI_PROC_NULL : rank + 1;

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI ranks: %d)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    size, nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }
    std::vector<Real> grid1((localNz + 2) * planeSize), grid2((localNz + 2) * planeSize);
    initializeGrid(grid1, nx, ny, localNz, firstZ);
    MPI_Barrier(comm);
    if (rank == 0) std::printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, nz, localNz, firstZ, previous, next, comm);
        else stencilIteration(grid2, grid1, nx, ny, nz, localNz, firstZ, previous, next, comm);
    }
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const std::vector<Real>& finalGrid = (iterations & 1) == 0 ? grid1 : grid2;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }
    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> globalGrid;
        if (rank == 0) { counts.resize(size); displacements.resize(size); }
        const int localCount = static_cast<int>(localNz * planeSize);
        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
        if (rank == 0) {
            int displacement = 0;
            for (int r = 0; r < size; ++r) { displacements[r] = displacement; displacement += counts[r]; }
            globalGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(finalGrid.data() + planeSize, localCount, MPI_DOUBLE, globalGrid.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(globalGrid, "Grid");
    }
    const bool valid = !validate || validateResult(finalGrid, nx, ny, localNz, comm, rank);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
