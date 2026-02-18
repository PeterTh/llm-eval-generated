#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start) {
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;
        const size_t z_offset = lz * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z_offset + y * nx;
            const size_t global_row = (gz * ny + y) * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = global_row + x;
                grid[row + x] = (global_idx % 19) * 1.0;
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const int rank, const int active_ranks) {
    if (local_nz == 0 || active_ranks <= 1) {
        return;
    }

    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < active_ranks) ? rank + 1 : MPI_PROC_NULL;
    const size_t plane = nx * ny;

    MPI_Sendrecv(grid.data() + idx3(0, 0, 1, nx, ny), static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                 grid.data() + idx3(0, 0, 0, nx, ny), static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(grid.data() + idx3(0, 0, local_nz, nx, ny), static_cast<int>(plane), MPI_DOUBLE, next, 1,
                 grid.data() + idx3(0, 0, local_nz + 1, nx, ny), static_cast<int>(plane), MPI_DOUBLE, next, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// 7-point stencil computation on local slab (with halo layers)
void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t local_nz, const size_t z_start,
                      const int rank, const int active_ranks) {
    if (local_nz == 0) {
        return;
    }

    exchangeHalos(input, nx, ny, local_nz, rank, active_ranks);

    const size_t plane = nx * ny;
    const Real* in = input.data();
    Real* out = output.data();

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;
        const size_t z_offset = lz * plane;

        if (gz == 0 || gz + 1 == nz) {
            std::memcpy(out + z_offset, in + z_offset, plane * sizeof(Real));
            continue;
        }

        const size_t z_offset_minus = (lz - 1) * plane;
        const size_t z_offset_plus = (lz + 1) * plane;

        for (size_t y = 1; y + 1 < ny; ++y) {
            const size_t row = z_offset + y * nx;
            const size_t row_minus = z_offset_minus + y * nx;
            const size_t row_plus = z_offset_plus + y * nx;
            const size_t row_front = z_offset + (y - 1) * nx;
            const size_t row_back = z_offset + (y + 1) * nx;

            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t idx = row + x;
                out[idx] = (in[idx] +
                            in[idx - 1] + in[idx + 1] +
                            in[row_front + x] + in[row_back + x] +
                            in[row_minus + x] + in[row_plus + x]) / 7.0;
            }
        }

        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z_offset + y * nx;
            out[row] = in[row];
            if (nx > 1) {
                out[row + nx - 1] = in[row + nx - 1];
            }
        }

        if (ny > 0) {
            const size_t bottom = z_offset;
            const size_t top = z_offset + (ny - 1) * nx;
            for (size_t x = 0; x < nx; ++x) {
                out[bottom + x] = in[bottom + x];
                out[top + x] = in[top + x];
            }
        }
    }
}

bool validateResultMPI(const std::vector<Real>& grid, const size_t nx, const size_t ny,
                       const size_t local_nz, const int rank) {
    const size_t plane = nx * ny;
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    int local_ok = 1;

    if (local_nz > 0) {
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            const Real* slice = grid.data() + lz * plane;
            for (size_t i = 0; i < plane; ++i) {
                const Real val = slice[i];
                if (std::isnan(val) || std::isinf(val)) {
                    local_ok = 0;
                    break;
                }
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
            if (local_ok == 0) {
                break;
            }
        }
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    Real global_min = 0.0;
    Real global_max = 0.0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        if (global_ok == 0) {
            printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
            if (global_max > 1e6 || global_min < -1e6) {
                printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid == 1;
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
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    if (rank == 0) {
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
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            }
        }

        if (parseStatus == 0) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        MPI_Finalize();
        return (parseStatus == 1) ? 0 : 1;
    }

    unsigned long long dims[3] = {0, 0, 0};
    if (rank == 0) {
        dims[0] = nx;
        dims[1] = ny;
        dims[2] = nz;
    }
    MPI_Bcast(dims, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    nx = static_cast<size_t>(dims[0]);
    ny = static_cast<size_t>(dims[1]);
    nz = static_cast<size_t>(dims[2]);

    int flags[3] = {0, 0, 0};
    if (rank == 0) {
        flags[0] = iterations;
        flags[1] = validate ? 1 : 0;
        flags[2] = printResults ? 1 : 0;
    }
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    iterations = flags[0];
    validate = (flags[1] != 0);
    printResults = (flags[2] != 0);

    const int active_ranks = static_cast<int>(std::min<size_t>(static_cast<size_t>(world_size), nz));
    const bool is_active = rank < active_ranks;

    size_t local_nz = 0;
    size_t z_start = 0;
    if (is_active) {
        const size_t active_size = static_cast<size_t>(active_ranks);
        const size_t base = nz / active_size;
        const size_t rem = nz % active_size;
        const size_t rank_sz = static_cast<size_t>(rank);
        local_nz = base + (rank_sz < rem ? 1 : 0);
        z_start = rank_sz * base + std::min(rank_sz, rem);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane = nx * ny;
    std::vector<Real> grid1;
    std::vector<Real> grid2;
    if (is_active) {
        const size_t local_size = (local_nz + 2) * plane;
        grid1.assign(local_size, 0.0);
        grid2.assign(local_size, 0.0);
    }

    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    if (is_active) {
        initializeGrid(grid1, nx, ny, local_nz, z_start);
    }

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (is_active) {
            stencilIteration(grid1, grid2, nx, ny, nz, local_nz, z_start, rank, active_ranks);
            grid1.swap(grid2);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double local_time = MPI_Wtime() - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_time * 1000.0);
    }

    double cellUpdates = 0.0;
    if (nx > 2 && ny > 2 && nz > 2) {
        cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    }
    double mcups = 0.0;
    if (max_time > 0.0) {
        mcups = cellUpdates / max_time / 1e6;
    }
    if (rank == 0) {
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<Real> fullGrid;
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
            counts.assign(world_size, 0);
            displs.assign(world_size, 0);
            size_t offset = 0;
            for (int r = 0; r < active_ranks; ++r) {
                const size_t active_size = static_cast<size_t>(active_ranks);
                const size_t base = nz / active_size;
                const size_t rem = nz % active_size;
                const size_t r_sz = static_cast<size_t>(r);
                const size_t local_nz_r = base + (r_sz < rem ? 1 : 0);
                const size_t count = local_nz_r * plane;
                counts[r] = static_cast<int>(count);
                displs[r] = static_cast<int>(offset);
                offset += count;
            }
        }

        const int sendcount = is_active ? static_cast<int>(local_nz * plane) : 0;
        Real* sendbuf = is_active ? (grid1.data() + idx3(0, 0, 1, nx, ny)) : nullptr;

        MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResultMPI(grid1, nx, ny, local_nz, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
