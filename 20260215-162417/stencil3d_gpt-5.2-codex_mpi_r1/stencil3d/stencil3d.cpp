#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct ZDecomp {
    size_t local_nz;
    size_t z_start;
};

ZDecomp decomposeZ(const size_t nz, const int rank, const int size) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    return {local_nz, z_start};
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start) {
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_start + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[idx3(x, y, z, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const int rank, const int size, MPI_Comm comm) {
    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    const int plane = static_cast<int>(nx * ny);

    MPI_Sendrecv(&grid[idx3(0, 0, 1, nx, ny)], plane, MPI_DOUBLE, prev, 0,
                 &grid[idx3(0, 0, 0, nx, ny)], plane, MPI_DOUBLE, prev, 1,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&grid[idx3(0, 0, local_nz, nx, ny)], plane, MPI_DOUBLE, next, 1,
                 &grid[idx3(0, 0, local_nz + 1, nx, ny)], plane, MPI_DOUBLE, next, 0,
                 comm, MPI_STATUS_IGNORE);
}

void applyBoundaries(const std::vector<Real>& input, std::vector<Real>& output,
                     const size_t nx, const size_t ny, const size_t local_nz,
                     const size_t z_start, const size_t global_nz) {
    for (size_t z = 1; z <= local_nz; ++z) {
        const size_t global_z = z_start + (z - 1);
        const bool z_boundary = (global_z == 0) || (global_z + 1 == global_nz);
        for (size_t y = 0; y < ny; ++y) {
            const bool y_boundary = (y == 0) || (y + 1 == ny);
            for (size_t x = 0; x < nx; ++x) {
                if (z_boundary || y_boundary || x == 0 || x + 1 == nx) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t local_nz,
                      const size_t z_start, const size_t global_nz) {
    size_t z_begin = 1;
    size_t z_end_exclusive = local_nz + 1;

    if (z_start == 0) {
        z_begin = 2;
    }
    if (z_start + local_nz == global_nz && z_end_exclusive > 0) {
        z_end_exclusive = local_nz;
    }

    if (z_begin < z_end_exclusive) {
        for (size_t z = z_begin; z < z_end_exclusive; ++z) {
            for (size_t y = 1; y + 1 < ny; ++y) {
                for (size_t x = 1; x + 1 < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);

                    const Real center = input[idx];
                    const Real left = input[idx3(x - 1, y, z, nx, ny)];
                    const Real right = input[idx3(x + 1, y, z, nx, ny)];
                    const Real front = input[idx3(x, y - 1, z, nx, ny)];
                    const Real back = input[idx3(x, y + 1, z, nx, ny)];
                    const Real bottom = input[idx3(x, y, z - 1, nx, ny)];
                    const Real top = input[idx3(x, y, z + 1, nx, ny)];

                    // Simple averaging stencil
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }

    applyBoundaries(input, output, nx, ny, local_nz, z_start, global_nz);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

void gatherGrid(const std::vector<Real>& local, std::vector<Real>& global,
                const size_t nx, const size_t ny, const size_t local_nz,
                const size_t global_nz, const int rank, const int size, MPI_Comm comm) {
    const int sendcount = static_cast<int>(local_nz * nx * ny);
    const Real* sendbuf = local.data() + idx3(0, 0, 1, nx, ny);

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        const size_t base = global_nz / static_cast<size_t>(size);
        const size_t rem = global_nz % static_cast<size_t>(size);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t local = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(local * nx * ny);
            displs[r] = static_cast<int>(offset);
            offset += local * nx * ny;
        }
    }

    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseOk = true;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseOk = false;
            break;
        }
    }

    if (!parseOk) {
        MPI_Finalize();
        return 1;
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const ZDecomp decomp = decomposeZ(nz, rank, size);
    if (decomp.local_nz == 0) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds nz (%zu)\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const size_t local_nz = decomp.local_nz;
    const size_t local_grid_size = (local_nz + 2) * nx * ny;

    // Allocate grids (double buffering, includes halos)
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, nx, ny, local_nz, decomp.z_start);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
            stencilIteration(grid1, grid2, nx, ny, local_nz, decomp.z_start, nz);
        } else {
            exchangeHalos(grid2, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
            stencilIteration(grid2, grid1, nx, ny, local_nz, decomp.z_start, nz);
        }
    }

    const double end = MPI_Wtime();
    const double local_time = end - start;
    double max_time = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        double interior = 0.0;
        if (nx > 2 && ny > 2 && nz > 2) {
            interior = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2));
        }
        const double cellUpdates = interior * iterations;
        const double mcups = (max_time > 0.0) ? (cellUpdates / max_time / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults || validate) {
        std::vector<Real> globalGrid;
        if (rank == 0) {
            globalGrid.resize(nx * ny * nz);
        }
        gatherGrid(finalGrid, globalGrid, nx, ny, local_nz, nz, rank, size, MPI_COMM_WORLD);

        if (printResults && rank == 0) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            int validInt = 0;
            if (rank == 0) {
                printf("Validating result...\n");
                const bool valid = validateResult(globalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                    validInt = 1;
                } else {
                    printf("Validation: FAILED\n");
                }
            }

            MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return validInt ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
