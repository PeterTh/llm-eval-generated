#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t localNz, const size_t globalZ0) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = globalZ0 + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                grid[idx3(x, y, z, nx, ny)] =
                    static_cast<Real>((idx3(x, y, globalZ, nx, ny)) % 19);
            }
        }
    }
    // The ghost planes are populated by the first halo exchange.
    (void)plane;
}

void exchangeHalos(std::vector<Real>& grid, const size_t plane,
                   const size_t localNz, const int rank, const int nprocs) {
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == nprocs ? MPI_PROC_NULL : rank + 1;
    MPI_Request requests[4];
    MPI_Irecv(grid.data(), static_cast<int>(plane), MPI_DOUBLE, previous, 101,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(grid.data() + (localNz + 1) * plane, static_cast<int>(plane),
              MPI_DOUBLE, next, 100, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(grid.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
              previous, 100, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(grid.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
              next, 101, MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t localNz,
                      const size_t globalNz, const size_t globalZ0) {
    const size_t plane = nx * ny;
    // Preserve physical boundaries without copying the entire local buffer.
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = globalZ0 + z - 1;
        const size_t row0 = z * plane;
        for (size_t y = 0; y < ny; ++y) {
            output[row0 + y * nx] = input[row0 + y * nx];
            if (nx > 1) output[row0 + y * nx + nx - 1] = input[row0 + y * nx + nx - 1];
        }
        if (ny > 1) {
            std::copy_n(input.begin() + row0, nx, output.begin() + row0);
            std::copy_n(input.begin() + row0 + (ny - 1) * nx, nx,
                        output.begin() + row0 + (ny - 1) * nx);
        }
        if (globalZ == 0 || globalZ + 1 >= globalNz) {
            std::copy_n(input.begin() + row0, plane, output.begin() + row0);
        }
    }
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = globalZ0 + z - 1;
        if (globalZ == 0 || globalZ + 1 >= globalNz) continue;
        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z * plane + y * nx;
            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t p = row + x;
                output[p] = (input[p] + input[p - 1] + input[p + 1] +
                             input[p - nx] + input[p + nx] +
                             input[p - plane] + input[p + plane]) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    if (grid.empty()) return true;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const Real value : grid) {
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 1 || ny < 1 || nz < 1 || iterations < 0 || nprocs > static_cast<int>(nz)) {
        if (rank == 0) std::printf("Invalid dimensions/iterations (need positive sizes and nz >= MPI ranks)\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = nz / static_cast<size_t>(nprocs);
    const size_t extra = nz % static_cast<size_t>(nprocs);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    std::vector<Real> grid1((localNz + 2) * plane);
    std::vector<Real> grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ0);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI, %d ranks)\n", nprocs);
        std::printf("Grid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;
        exchangeHalos(input, plane, localNz, rank, nprocs);
        stencilIteration(input, output, nx, ny, localNz, nz, globalZ0);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<int> counts, displacements;
    std::vector<Real> globalGrid;
    if (rank == 0) {
        counts.resize(nprocs);
        displacements.resize(nprocs);
        size_t displacement = 0;
        for (int r = 0; r < nprocs; ++r) {
            const size_t count = (base + (static_cast<size_t>(r) < extra ? 1 : 0)) * plane;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()) || displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            counts[r] = static_cast<int>(count);
            displacements[r] = static_cast<int>(displacement);
            displacement += count;
        }
        globalGrid.resize(nx * ny * nz);
    }
    MPI_Gatherv(finalGrid.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double cells = static_cast<double>((nx > 2 && ny > 2 && nz > 2) ?
            (nx - 2) * (ny - 2) * (nz - 2) : 0) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? cells / elapsed / 1e6 : 0.0);
        if (printResults) print_results(globalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(globalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
