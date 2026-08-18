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

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz, size_t globalZ0) {
    for (size_t z = 0; z < localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, globalZ0 + z, nx, ny);
                grid[idx3(x, y, z + 1, nx, ny)] = static_cast<Real>(globalIndex % 19);
            }
}

// Computes one local plane.  Local plane 0 and localNz+1 are MPI halos.
inline void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                         size_t localZ, size_t globalZ, size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t base = localZ * plane;
    if (globalZ == 0 || globalZ + 1 == nz) {
        std::copy_n(input.data() + base, plane, output.data() + base);
        return;
    }

    for (size_t y = 0; y < ny; ++y) {
        const size_t row = base + y * nx;
        if (y == 0 || y + 1 == ny) {
            std::copy_n(input.data() + row, nx, output.data() + row);
            continue;
        }
        output[row] = input[row];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t i = row + x;
            output[i] = (input[i] + input[i - 1] + input[i + 1] + input[i - nx] +
                         input[i + nx] + input[i - plane] + input[i + plane]) / 7.0;
        }
        if (nx > 1) output[row + nx - 1] = input[row + nx - 1];
    }
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      size_t nx, size_t ny, size_t nz, size_t localNz, size_t globalZ0,
                      int rank, int ranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    MPI_Request requests[4];
    int requestCount = 0;
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int count = static_cast<int>(plane);

    MPI_Irecv(const_cast<Real*>(input.data()), count, MPI_DOUBLE, lower, 1, comm, &requests[requestCount++]);
    MPI_Irecv(const_cast<Real*>(input.data()) + (localNz + 1) * plane, count, MPI_DOUBLE, upper, 0, comm, &requests[requestCount++]);
    MPI_Isend(input.data() + plane, count, MPI_DOUBLE, lower, 0, comm, &requests[requestCount++]);
    MPI_Isend(input.data() + localNz * plane, count, MPI_DOUBLE, upper, 1, comm, &requests[requestCount++]);

    // These planes do not depend on received halo data.
    for (size_t z = 2; z < localNz; ++z)
        stencilPlane(input, output, z, globalZ0 + z - 1, nx, ny, nz);

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
    if (localNz != 0) {
        stencilPlane(input, output, 1, globalZ0, nx, ny, nz);
        if (localNz > 1)
            stencilPlane(input, output, localNz, globalZ0 + localNz - 1, nx, ny, nz);
    }
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    for (Real value : grid) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return maxVal <= 1e6 && minVal >= -1e6;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  X grid size (default: 128)\n  -y <num>  Y grid size (default: X)\n");
    std::printf("  -z <num>  Z grid size (default: X)\n  -i <num>  iterations (default: 10)\n");
    std::printf("  -v        validate\n  -r        print results\n  -h        show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseError = 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) parseError = 1;
    if (parseError) { if (rank == 0) { std::printf("Invalid option or dimensions\n"); printUsage(argv[0]); } MPI_Finalize(); return 1; }

    const size_t baseNz = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ0 = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max()) || localNz == 0) {
        if (rank == 0) std::printf("Grid is incompatible with this MPI decomposition\n");
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI ranks: %d)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    ranks, nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    std::vector<Real> grid1((localNz + 2) * plane), grid2((localNz + 2) * plane);
    initializeGrid(grid1, nx, ny, localNz, globalZ0);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, nz, localNz, globalZ0, rank, ranks, MPI_COMM_WORLD);
        else stencilIteration(grid2, grid1, nx, ny, nz, localNz, globalZ0, rank, ranks, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double updates = static_cast<double>((nx > 2 ? nx - 2 : 0) * (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }

    int valid = 1;
    if (validate || printResults) {
        std::vector<int> counts, displacements;
        std::vector<Real> complete;
        if (rank == 0) {
            counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t depth = baseNz + (static_cast<size_t>(r) < remainder);
                const size_t offset = static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder);
                counts[r] = static_cast<int>(depth * plane); displacements[r] = static_cast<int>(offset * plane);
            }
            complete.resize(nx * ny * nz);
        }
        const std::vector<Real>& result = (iterations & 1) ? grid2 : grid1;
        MPI_Gatherv(result.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? complete.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(complete, "Grid");
            if (validate) { std::printf("Validating result...\n"); valid = validateResult(complete) ? 1 : 0; std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
