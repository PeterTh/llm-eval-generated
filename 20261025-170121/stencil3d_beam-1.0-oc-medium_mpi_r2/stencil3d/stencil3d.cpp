#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <mpi.h>

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz,
                    const size_t zStart) {
    const size_t localNz = grid.size() / (nx * ny);
    for (size_t lz = 1; lz + 1 < localNz; ++lz) {
        const size_t gz = zStart + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = (lz * (nx * ny)) + y * nx + x;
                const size_t gidx = idx3(x, y, gz, nx, ny);
                grid[idx] = (gidx % 19) * 1.0;
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t localNz,
                   int prevRank, int nextRank, MPI_Comm comm) {
    const size_t plane = nx * ny;
    // lower halo: recv from prev into lz=0; send lz=1 to prev (as its upper halo)
    MPI_Sendrecv(grid.data() + 1 * plane, plane, MPI_DOUBLE, prevRank, 0,
                 grid.data() + 0 * plane, plane, MPI_DOUBLE, prevRank, 1,
                 comm, MPI_STATUS_IGNORE);
    // upper halo: send last owned (lz=localNz-2) to next; recv into lz=localNz-1
    MPI_Sendrecv(grid.data() + (localNz - 2) * plane, plane, MPI_DOUBLE, nextRank, 1,
                 grid.data() + (localNz - 1) * plane, plane, MPI_DOUBLE, nextRank, 0,
                 comm, MPI_STATUS_IGNORE);
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t zStart, int prevRank, int nextRank, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t localNz = input.size() / plane;

    exchangeHalos(input, nx, ny, localNz, prevRank, nextRank, comm);

    // interior owned planes
    for (size_t lz = 1; lz + 1 < localNz; ++lz) {
        const size_t gz = zStart + (lz - 1);
        const bool zBoundary = (gz == 0 || gz == nz - 1);
        for (size_t y = 0; y < ny; ++y) {
            const bool yBoundary = (y == 0 || y == ny - 1);
            for (size_t x = 0; x < nx; ++x) {
                const bool xBoundary = (x == 0 || x == nx - 1);
                const size_t idx = lz * plane + y * nx + x;
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || zBoundary) {
                    output[idx] = input[idx];
                    continue;
                }
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - plane];
                const Real top = input[idx + plane];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    // halo planes of output unused
}

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){ printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // decompose z
    size_t base = nz / size, rem = nz % size;
    size_t count = base + (rank < (int)rem ? 1 : 0);
    size_t zStart = rank * base + std::min((size_t)rank, rem);
    int prevRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int nextRank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    size_t localNz = count + 2;
    std::vector<Real> grid1(localNz * nx * ny);
    std::vector<Real> grid2(localNz * nx * ny);
    initializeGrid(grid1, nx, ny, nz, zStart);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz, zStart, prevRank, nextRank, MPI_COMM_WORLD);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz, zStart, prevRank, nextRank, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = (double)((nx-2)*(ny-2)*(nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    // gather
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t c = base + (r < (int)rem ? 1 : 0);
        size_t zs = r * base + std::min((size_t)r, rem);
        counts[r] = (int)(c * nx * ny);
        displs[r] = (int)(zs * nx * ny);
    }
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    MPI_Gatherv(finalLocal.data() + 1 * (nx*ny), counts[rank], MPI_DOUBLE,
                finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
