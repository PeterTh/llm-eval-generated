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

// 3D index calculation (z is local index including ghost layers)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize owned layers using global coordinates
// Local layout: layer 0 = bottom ghost, layers 1..nz_local = owned, layer nz_local+1 = top ghost
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz_local, const size_t z_start) {
    const size_t layer_sz = nx * ny;
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        const size_t gz = z_start + (lz - 1);
        const size_t gz_offset = gz * layer_sz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                grid[lz * layer_sz + y * nx + x] = static_cast<Real>((gz_offset + y * nx + x) % 19);
            }
        }
    }
}

// 7-point stencil with non-blocking ghost exchange overlapped with computation
void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny,
                      const size_t nz_local,
                      const size_t z_start,
                      const size_t nz_total,
                      int rank_below, int rank_above,
                      MPI_Comm comm) {
    const size_t layer_sz = nx * ny;

    // Launch non-blocking ghost exchange
    MPI_Request reqs[4];
    int nreqs = 0;

    // Receive top ghost (local z=nz_local+1) from above
    MPI_Irecv(input.data() + (nz_local + 1) * layer_sz, static_cast<int>(layer_sz), MPI_DOUBLE,
              rank_above, 0, comm, &reqs[nreqs++]);
    // Send bottom owned layer (local z=1) to below
    MPI_Isend(input.data() + 1 * layer_sz, static_cast<int>(layer_sz), MPI_DOUBLE,
              rank_below, 0, comm, &reqs[nreqs++]);
    // Receive bottom ghost (local z=0) from below
    MPI_Irecv(input.data() + 0 * layer_sz, static_cast<int>(layer_sz), MPI_DOUBLE,
              rank_below, 1, comm, &reqs[nreqs++]);
    // Send top owned layer (local z=nz_local) to above
    MPI_Isend(input.data() + nz_local * layer_sz, static_cast<int>(layer_sz), MPI_DOUBLE,
              rank_above, 1, comm, &reqs[nreqs++]);

    // Phase 1: compute interior z-layers (local z=2..nz_local-1) that don't need ghost data
    for (size_t lz = 2; lz < nz_local; ++lz) {
        const size_t gz = z_start + (lz - 1);
        if (gz == 0 || gz == nz_total - 1) continue;

        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t base = lz * layer_sz + y * nx;
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = base + x;
                output[idx] = (input[idx]
                    + input[idx - 1] + input[idx + 1]
                    + input[idx - nx]  + input[idx + nx]
                    + input[idx - layer_sz] + input[idx + layer_sz]) / 7.0;
            }
        }
    }

    // Wait for ghost exchange to complete before accessing ghost layers
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Phase 2: compute boundary z-layers that need ghost data
    // Local z=1 (needs bottom ghost at local z=0)
    if (nz_local >= 1) {
        const size_t gz = z_start;
        if (gz != 0 && gz != nz_total - 1) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t base = 1 * layer_sz + y * nx;
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = base + x;
                    output[idx] = (input[idx]
                        + input[idx - 1] + input[idx + 1]
                        + input[idx - nx]  + input[idx + nx]
                        + input[idx - layer_sz] + input[idx + layer_sz]) / 7.0;
                }
            }
        }
    }

    // Local z=nz_local (needs top ghost at local z=nz_local+1), if distinct from z=1
    if (nz_local >= 2) {
        const size_t gz = z_start + nz_local - 1;
        if (gz != 0 && gz != nz_total - 1) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t base = nz_local * layer_sz + y * nx;
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = base + x;
                    output[idx] = (input[idx]
                        + input[idx - 1] + input[idx + 1]
                        + input[idx - nx]  + input[idx + nx]
                        + input[idx - layer_sz] + input[idx + layer_sz]) / 7.0;
                }
            }
        }
    }

    // Copy boundary values for all owned layers
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        const size_t gz = z_start + (lz - 1);
        const bool z_boundary = (gz == 0 || gz == nz_total - 1);

        if (z_boundary) {
            // Entire z-layer is boundary
            const size_t base = lz * layer_sz;
            for (size_t i = 0; i < layer_sz; ++i) {
                output[base + i] = input[base + i];
            }
        } else {
            // Only x and y boundaries
            for (size_t y = 0; y < ny; ++y) {
                const size_t row_base = lz * layer_sz + y * nx;
                if (y == 0 || y == ny - 1) {
                    for (size_t x = 0; x < nx; ++x) {
                        output[row_base + x] = input[row_base + x];
                    }
                } else {
                    output[row_base] = input[row_base];
                    if (nx > 1) output[row_base + nx - 1] = input[row_base + nx - 1];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D block distribution along Z-axis
    const size_t base_nz = nz / nprocs;
    const size_t remainder = nz % nprocs;
    size_t z_start, nz_local;
    if (static_cast<size_t>(rank) < remainder) {
        nz_local = base_nz + 1;
        z_start = static_cast<size_t>(rank) * (base_nz + 1);
    } else {
        nz_local = base_nz;
        z_start = remainder * (base_nz + 1) + (static_cast<size_t>(rank) - remainder) * base_nz;
    }

    // Create communicator for active ranks (nz_local > 0)
    MPI_Comm comp_comm;
    int color = (nz_local > 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comp_comm);

    int comp_rank = -1, comp_nprocs = 0;
    if (comp_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comp_comm, &comp_rank);
        MPI_Comm_size(comp_comm, &comp_nprocs);
    }

    int rank_below = (comp_rank > 0) ? comp_rank - 1 : MPI_PROC_NULL;
    int rank_above = (comp_rank < comp_nprocs - 1) ? comp_rank + 1 : MPI_PROC_NULL;

    const size_t layer_sz = nx * ny;

    // Allocate local grids: (nz_local + 2) layers including ghost layers
    // Layout: [ghost_bottom | owned layers 1..nz_local | ghost_top]
    const size_t local_size = (nz_local + 2) * layer_sz;
    std::vector<Real> grid1(local_size, 0.0);
    std::vector<Real> grid2(local_size, 0.0);

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz_local, z_start);

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (comp_comm != MPI_COMM_NULL) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz_local, z_start, nz,
                                 rank_below, rank_above, comp_comm);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz_local, z_start, nz,
                                 rank_below, rank_above, comp_comm);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long max_ms;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for printing/validation
    std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    int exit_code = 0;

    if (printResults || validate) {
        // Gather owned layer counts
        int my_count = static_cast<int>(nz_local * layer_sz);
        std::vector<int> recvcounts(nprocs);
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs, 0);
        std::vector<Real> global_grid;
        if (rank == 0) {
            global_grid.resize(nx * ny * nz);
            for (int r = 1; r < nprocs; ++r) {
                displs[r] = displs[r - 1] + recvcounts[r - 1];
            }
        }

        // Gather owned data (local z=1..nz_local, contiguous in memory)
        MPI_Gatherv(finalLocal.data() + layer_sz, my_count, MPI_DOUBLE,
                    global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (printResults && rank == 0) {
            print_results(global_grid, "Grid");
        }

        if (validate && rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(global_grid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    if (comp_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comp_comm);
    }

    MPI_Finalize();
    return exit_code;
}
