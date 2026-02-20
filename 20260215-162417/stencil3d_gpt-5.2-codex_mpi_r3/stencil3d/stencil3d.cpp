#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cstdint>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Decomposition {
    size_t z_start;
    size_t local_nz;
};

Decomposition decomposeZ(const size_t nz, const int rank, const int size) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = (static_cast<size_t>(rank) < rem)
        ? static_cast<size_t>(rank) * (base + 1)
        : rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
    return {z_start, local_nz};
}

void initializeGridMPI(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz,
                       const size_t z_start, const size_t local_nz) {
    (void)nz;
    const size_t slice = nx * ny;
    std::fill(grid.begin(), grid.end(), Real{0});

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;
        const size_t local_base = lz * slice;
        const size_t global_base = gz * slice;
        for (size_t y = 0; y < ny; ++y) {
            const size_t local_row = local_base + y * nx;
            const size_t global_row = global_base + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = global_row + x;
                grid[local_row + x] = static_cast<Real>(gidx % 19);
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid, const size_t slice, const size_t local_nz,
                   const int rank, const int size, MPI_Comm comm, MPI_Request requests[4]) {
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    Real* data = grid.data();
    const int slice_count = static_cast<int>(slice);

    MPI_Irecv(data + 0 * slice, slice_count, MPI_DOUBLE, down, 200, comm, &requests[0]);
    MPI_Irecv(data + (local_nz + 1) * slice, slice_count, MPI_DOUBLE, up, 100, comm, &requests[1]);
    MPI_Isend(data + 1 * slice, slice_count, MPI_DOUBLE, down, 100, comm, &requests[2]);
    MPI_Isend(data + local_nz * slice, slice_count, MPI_DOUBLE, up, 200, comm, &requests[3]);
}

void computeSlice(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t nz,
                  const size_t slice, const size_t z_local, const size_t global_z) {
    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t z_base = z_local * slice;
    if (global_z == 0 || global_z + 1 == nz) {
        std::memcpy(output + z_base, input + z_base, slice * sizeof(Real));
        return;
    }

    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t y_base = z_base + y * nx;
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t idx = y_base + x;
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                + input[idx - nx] + input[idx + nx]
                + input[idx - slice] + input[idx + slice]) / 7.0;
        }
    }

    const size_t last_x = nx - 1;
    for (size_t y = 0; y < ny; ++y) {
        const size_t base = z_base + y * nx;
        output[base] = input[base];
        if (last_x != 0) {
            output[base + last_x] = input[base + last_x];
        }
    }

    const size_t last_y = ny - 1;
    const size_t base0 = z_base;
    const size_t base1 = z_base + last_y * nx;
    for (size_t x = 0; x < nx; ++x) {
        output[base0 + x] = input[base0 + x];
        if (last_y != 0) {
            output[base1 + x] = input[base1 + x];
        }
    }
}

void stencilIterationMPI(std::vector<Real>& input, std::vector<Real>& output,
                         const size_t nx, const size_t ny, const size_t nz,
                         const size_t z_start, const size_t local_nz,
                         const int rank, const int size, MPI_Comm comm) {
    const size_t slice = nx * ny;
    MPI_Request requests[4];
    exchangeHalos(input, slice, local_nz, rank, size, comm, requests);

    if (local_nz > 2) {
        for (size_t z = 2; z <= local_nz - 1; ++z) {
            const size_t global_z = z_start + z - 1;
            computeSlice(input.data(), output.data(), nx, ny, nz, slice, z, global_z);
        }
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    if (local_nz >= 1) {
        computeSlice(input.data(), output.data(), nx, ny, nz, slice, 1, z_start);
        if (local_nz > 1) {
            computeSlice(input.data(), output.data(), nx, ny, nz, slice, local_nz, z_start + local_nz - 1);
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

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
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    int showHelp = 0;
    int parseOk = 1;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = 0;
            }
        }

        if (showHelp || !parseOk) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, comm);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, comm);

    if (!parseOk || showHelp) {
        MPI_Finalize();
        return parseOk ? 0 : 1;
    }

    uint64_t nx64 = 0;
    uint64_t ny64 = 0;
    uint64_t nz64 = 0;
    int iterations_bcast = 0;
    int validate_bcast = 0;
    int print_bcast = 0;

    if (rank == 0) {
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        nx64 = static_cast<uint64_t>(nx);
        ny64 = static_cast<uint64_t>(ny);
        nz64 = static_cast<uint64_t>(nz);
        iterations_bcast = iterations;
        validate_bcast = validate ? 1 : 0;
        print_bcast = printResults ? 1 : 0;
    }

    MPI_Bcast(&nx64, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&ny64, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&nz64, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Bcast(&iterations_bcast, 1, MPI_INT, 0, comm);
    MPI_Bcast(&validate_bcast, 1, MPI_INT, 0, comm);
    MPI_Bcast(&print_bcast, 1, MPI_INT, 0, comm);

    nx = static_cast<size_t>(nx64);
    ny = static_cast<size_t>(ny64);
    nz = static_cast<size_t>(nz64);
    iterations = iterations_bcast;
    validate = (validate_bcast != 0);
    printResults = (print_bcast != 0);

    if (size > static_cast<int>(nz)) {
        if (rank == 0) {
            printf("Error: MPI ranks (%d) exceed z dimension (%zu).\n", size, nz);
        }
        MPI_Abort(comm, 1);
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t slice = nx * ny;
    if (slice > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Error: slice size exceeds MPI count limits.\n");
        }
        MPI_Abort(comm, 1);
        return 1;
    }

    const Decomposition decomp = decomposeZ(nz, rank, size);
    const size_t local_nz = decomp.local_nz;
    const size_t local_grid_size = slice * (local_nz + 2);

    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGridMPI(grid1, nx, ny, nz, decomp.z_start, local_nz);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationMPI(grid1, grid2, nx, ny, nz, decomp.z_start, local_nz, rank, size, comm);
        } else {
            stencilIterationMPI(grid2, grid1, nx, ny, nz, decomp.z_start, local_nz, rank, size, comm);
        }
    }

    MPI_Barrier(comm);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    std::vector<Real> gatheredGrid;
    if (printResults || validate) {
        if (local_nz * slice > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                printf("Error: local grid exceeds MPI gather limits.\n");
            }
            MPI_Abort(comm, 1);
            return 1;
        }

        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            gatheredGrid.resize(nx * ny * nz);
            recvCounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const Decomposition rd = decomposeZ(nz, r, size);
                const size_t count = rd.local_nz * slice;
                if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    printf("Error: gather count exceeds MPI limits.\n");
                    MPI_Abort(comm, 1);
                    return 1;
                }
                recvCounts[r] = static_cast<int>(count);
                displs[r] = static_cast<int>(rd.z_start * slice);
            }
        }

        const int sendCount = static_cast<int>(local_nz * slice);
        MPI_Gatherv(finalGrid.data() + slice, sendCount, MPI_DOUBLE,
                    rank == 0 ? gatheredGrid.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
    }

    if (printResults && rank == 0) {
        print_results(gatheredGrid, "Grid");
    }

    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            const bool ok = validateResult(gatheredGrid, nx, ny, nz);
            if (ok) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                valid = 0;
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
