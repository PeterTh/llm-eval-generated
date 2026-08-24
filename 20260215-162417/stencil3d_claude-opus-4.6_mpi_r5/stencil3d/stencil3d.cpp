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

void initializeLocalGrid(Real* grid, const size_t nx, const size_t ny,
                         const size_t local_nz_total, const size_t global_nz,
                         const long long ghost_z_start) {
    const size_t plane_size = nx * ny;
    for (size_t lz = 0; lz < local_nz_total; ++lz) {
        long long gz = ghost_z_start + (long long)lz;
        if (gz < 0 || gz >= (long long)global_nz) {
            std::fill(grid + lz * plane_size, grid + (lz + 1) * plane_size, 0.0);
            continue;
        }
        size_t gzu = (size_t)gz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gzu * plane_size + y * nx + x;
                grid[lz * plane_size + y * nx + x] = (global_idx % 19) * 1.0;
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition along z-axis
    const size_t base_nz = nz / (size_t)nprocs;
    const size_t remainder = nz % (size_t)nprocs;

    size_t z_start, local_nz;
    if ((size_t)rank < remainder) {
        local_nz = base_nz + 1;
        z_start = (size_t)rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_start = remainder * (base_nz + 1) + ((size_t)rank - remainder) * base_nz;
    }

    // Neighbor ranks (MPI_PROC_NULL for no-neighbor)
    int last_active = (base_nz > 0) ? nprocs - 1 : (int)remainder - 1;
    int rank_below = (rank > 0 && local_nz > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_above = (rank < last_active && local_nz > 0) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane_size = nx * ny;
    const size_t local_nz_total = local_nz + 2; // +2 ghost planes
    const size_t local_size = plane_size * local_nz_total;
    const size_t stride_z = plane_size;
    const size_t stride_y = nx;

    std::vector<Real> grid1(local_size, 0.0);
    std::vector<Real> grid2(local_size, 0.0);

    if (rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(grid1.data(), nx, ny, local_nz_total, nz, (long long)z_start - 1);

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Stencil computation for a range of local z-planes
    auto computePlanes = [&](const Real* input, Real* output,
                             size_t lz_begin, size_t lz_end) {
        for (size_t lz = lz_begin; lz <= lz_end; ++lz) {
            size_t gz = z_start + lz - 1;
            // Global z boundary: copy entire plane
            if (gz == 0 || gz == nz - 1) {
                std::copy(&input[lz * stride_z],
                          &input[lz * stride_z + plane_size],
                          &output[lz * stride_z]);
                continue;
            }
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t i = lz * stride_z + y * stride_y + x;
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
                        output[i] = input[i];
                    } else {
                        output[i] = (input[i]
                                   + input[i - 1] + input[i + 1]
                                   + input[i - stride_y] + input[i + stride_y]
                                   + input[i - stride_z] + input[i + stride_z]) / 7.0;
                    }
                }
            }
        }
    };

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input  = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* output = (iter % 2 == 0) ? grid2.data() : grid1.data();

        // Non-blocking halo exchange
        MPI_Request reqs[4];
        MPI_Isend(&input[1 * stride_z], (int)plane_size, MPI_DOUBLE,
                  rank_below, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&input[0], (int)plane_size, MPI_DOUBLE,
                  rank_below, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&input[local_nz * stride_z], (int)plane_size, MPI_DOUBLE,
                  rank_above, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&input[(local_nz + 1) * stride_z], (int)plane_size, MPI_DOUBLE,
                  rank_above, 0, MPI_COMM_WORLD, &reqs[3]);

        // Compute interior planes (no ghost dependency) while halos exchange
        if (local_nz > 2) {
            computePlanes(input, output, 2, local_nz - 1);
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Compute boundary planes (depend on ghost data)
        if (local_nz >= 1) {
            computePlanes(input, output, 1, 1);
        }
        if (local_nz >= 2) {
            computePlanes(input, output, local_nz, local_nz);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int ret = 0;

    if (printResults || validate) {
        const Real* finalLocal = (iterations % 2 == 0) ? grid1.data() : grid2.data();

        // Extract owned planes (strip ghost layers)
        std::vector<Real> owned(local_nz * plane_size);
        if (local_nz > 0) {
            std::copy(finalLocal + plane_size,
                      finalLocal + (local_nz + 1) * plane_size,
                      owned.data());
        }

        // Gather on rank 0
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            size_t r_nz, r_start;
            if ((size_t)r < remainder) {
                r_nz = base_nz + 1;
                r_start = (size_t)r * (base_nz + 1);
            } else {
                r_nz = base_nz;
                r_start = remainder * (base_nz + 1) + ((size_t)r - remainder) * base_nz;
            }
            recvcounts[r] = (int)(r_nz * plane_size);
            displs[r] = (int)(r_start * plane_size);
        }

        std::vector<Real> fullGrid;
        if (rank == 0) fullGrid.resize(nx * ny * nz);

        MPI_Gatherv(owned.data(), (int)(local_nz * plane_size), MPI_DOUBLE,
                    fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    MPI_Finalize();
    return ret;
}
