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
                    const size_t nz, const size_t global_z0) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= nz; ++z) {
        const size_t global_z = global_z0 + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                grid[idx3(x, y, z, nx, ny)] =
                    static_cast<Real>(idx3(x, y, global_z, nx, ny) % 19);
            }
        }
    }
    // Keep the halo allocation initialized as well; halo values are replaced
    // by MPI before they can be used.
    std::fill_n(grid.begin(), plane, 0.0);
    std::fill_n(grid.end() - plane, plane, 0.0);
}

void stencilPlane(const std::vector<Real>& input, std::vector<Real>& output,
                  const size_t nx, const size_t ny, const size_t nz,
                  const size_t global_z0, const size_t local_z) {
    const size_t global_z = global_z0 + local_z - 1;
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            const size_t out = idx3(x, y, local_z, nx, ny);
            if (global_z == 0 || global_z + 1 == nz ||
                x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
                output[out] = input[out];
                continue;
            }
            const Real center = input[out];
            output[out] = (center + input[out - 1] + input[out + 1] +
                           input[out - nx] + input[out + nx] +
                           input[out - nx * ny] + input[out + nx * ny]) / 7.0;
        }
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t global_z0, const size_t local_nz,
                      const int lower, const int upper, MPI_Comm comm) {
    // The exchange uses the input buffer. Compute non-edge slabs while it is
    // in flight, then compute the two slabs that depend on the received halos.
    const size_t plane = nx * ny;
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    if (lower != MPI_PROC_NULL) {
        MPI_Request request;
        MPI_Irecv(input.data(), static_cast<int>(plane), MPI_DOUBLE,
                  lower, 1, comm, &request); requests.push_back(request);
        MPI_Isend(input.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                  lower, 0, comm, &request); requests.push_back(request);
    }
    if (upper != MPI_PROC_NULL) {
        MPI_Request request;
        MPI_Irecv(input.data() + (local_nz + 1) * plane,
                  static_cast<int>(plane), MPI_DOUBLE, upper, 0, comm, &request);
        requests.push_back(request);
        MPI_Isend(input.data() + local_nz * plane, static_cast<int>(plane),
                  MPI_DOUBLE, upper, 1, comm, &request); requests.push_back(request);
    }

    if (local_nz > 2) {
        for (size_t z = 2; z < local_nz; ++z)
            stencilPlane(input, output, nx, ny, nz, global_z0, z);
    }
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    stencilPlane(input, output, nx, ny, nz, global_z0, 1);
    if (local_nz > 1)
        stencilPlane(input, output, nx, ny, nz, global_z0, local_nz);
}

bool validateResult(const std::vector<Real>& grid) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0], maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return maxVal <= 1e6 && minVal >= -1e6;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else bad = true;
    }
    if (help) { if (world_rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    bad = bad || nx < 3 || ny < 3 || nz < 3 || iterations < 0 || world_size < 1;
    if (bad) { if (world_rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }

    const int active_count = std::min<size_t>(world_size, nz);
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_count ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (world_rank >= active_count) { MPI_Finalize(); return 0; }

    int rank = 0, size = 0;
    MPI_Comm_rank(active_comm, &rank); MPI_Comm_size(active_comm, &size);
    const size_t base = nz / static_cast<size_t>(size);
    const size_t extra = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t global_z0 = static_cast<size_t>(rank) * base +
                             std::min<size_t>(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) MPI_Abort(active_comm, 2);

    const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;
    std::vector<Real> grid1((local_nz + 2) * plane), grid2((local_nz + 2) * plane);
    initializeGrid(grid1, nx, ny, local_nz, global_z0);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(active_comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) stencilIteration(grid1, grid2, nx, ny, nz, global_z0, local_nz, lower, upper, active_comm);
        else stencilIteration(grid2, grid1, nx, ny, nz, global_z0, local_nz, lower, upper, active_comm);
    }
    const double local_time = MPI_Wtime() - start, end_time = [&] { double t = 0; MPI_Reduce(&local_time, &t, 1, MPI_DOUBLE, MPI_MAX, 0, active_comm); return t; }();
    if (rank == 0) {
        const long ms = static_cast<long>(end_time * 1000.0);
        std::printf("Computation time: %ld ms\n", ms);
        const double cells = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", cells / end_time / 1e6);
    }

    const std::vector<Real>& local_final = (iterations & 1) == 0 ? grid1 : grid2;
    std::vector<int> counts, displacements;
    std::vector<Real> finalGrid;
    if (rank == 0) { counts.resize(size); displacements.resize(size); finalGrid.resize(nx * ny * nz); }
    const int send_count = static_cast<int>(local_nz * plane);
    MPI_Gather(&send_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, active_comm);
    if (rank == 0) {
        int displacement = 0;
        for (int r = 0; r < size; ++r) { displacements[r] = displacement; displacement += counts[r]; }
    }
    MPI_Gatherv(local_final.data() + plane, send_count, MPI_DOUBLE,
                finalGrid.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active_comm);

    if (rank == 0) {
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Comm_free(&active_comm); MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    MPI_Comm_free(&active_comm);
    MPI_Finalize();
    return 0;
}
