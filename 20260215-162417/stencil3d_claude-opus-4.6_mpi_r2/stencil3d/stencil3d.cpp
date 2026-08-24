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

    // 1D decomposition along z-axis
    const size_t base_nz = nz / (size_t)nprocs;
    const size_t remainder = nz % (size_t)nprocs;
    const size_t local_nz = base_nz + ((size_t)rank < remainder ? 1 : 0);
    const size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, remainder);

    const int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane_size = nx * ny;
    // Local grid: local_nz owned planes + 2 ghost planes
    const size_t local_total = (local_nz + 2) * plane_size;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Real> grid1(local_total, 0.0);
    std::vector<Real> grid2(local_total, 0.0);

    // Initialize owned planes using global indices
    if (rank == 0) printf("Initializing grid...\n");
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = gz * plane_size + y * nx + x;
                grid1[lz * plane_size + y * nx + x] = (global_idx % 19) * 1.0;
            }
        }
    }

    // Determine local z range for stencil computation
    // local z=1 is global z=z_start, local z=local_nz is global z=z_start+local_nz-1
    size_t lz_begin, lz_end;
    if (local_nz == 0) {
        lz_begin = 1;
        lz_end = 0;
    } else {
        lz_begin = (z_start == 0) ? 2 : 1;
        lz_end = (z_start + local_nz >= nz) ? (local_nz > 0 ? local_nz - 1 : 0) : local_nz;
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input  = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? grid2 : grid1;

        // Non-blocking halo exchange
        MPI_Request reqs[4];
        MPI_Isend(&input[1 * plane_size], (int)plane_size, MPI_DOUBLE,
                  rank_below, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&input[0], (int)plane_size, MPI_DOUBLE,
                  rank_below, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&input[local_nz * plane_size], (int)plane_size, MPI_DOUBLE,
                  rank_above, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&input[(local_nz + 1) * plane_size], (int)plane_size, MPI_DOUBLE,
                  rank_above, 0, MPI_COMM_WORLD, &reqs[3]);

        // Compute interior planes (no ghost dependency: lz != 1 and lz != local_nz)
        for (size_t lz = lz_begin; lz <= lz_end; ++lz) {
            if (lz == 1 || lz == local_nz) continue;
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t i = lz * plane_size + y * nx + x;
                    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                                 input[i - nx] + input[i + nx] +
                                 input[i - plane_size] + input[i + plane_size]) / 7.0;
                }
            }
        }

        // Wait for halo exchange to complete
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Compute halo-dependent planes (lz=1 and/or lz=local_nz)
        if (lz_begin <= 1 && 1 <= lz_end) {
            const size_t lz = 1;
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t i = lz * plane_size + y * nx + x;
                    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                                 input[i - nx] + input[i + nx] +
                                 input[i - plane_size] + input[i + plane_size]) / 7.0;
                }
            }
        }
        if (local_nz >= 2 && lz_begin <= local_nz && local_nz <= lz_end) {
            const size_t lz = local_nz;
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t i = lz * plane_size + y * nx + x;
                    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                                 input[i - nx] + input[i + nx] +
                                 input[i - plane_size] + input[i + plane_size]) / 7.0;
                }
            }
        }

        // Copy boundary values for owned planes
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            const size_t gz = z_start + (lz - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                        gz == 0 || gz == nz - 1) {
                        const size_t i = lz * plane_size + y * nx + x;
                        output[i] = input[i];
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    double global_elapsed = 0.0;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long duration_ms = (long)(global_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / global_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults || validate) {
        // Gather full grid to rank 0
        int my_count = (int)(local_nz * plane_size);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < nprocs; ++r)
                displs[r] = displs[r - 1] + recvcounts[r - 1];
        }

        std::vector<Real> fullGrid;
        if (rank == 0) fullGrid.resize(nx * ny * nz);

        MPI_Gatherv(&localFinal[1 * plane_size], my_count, MPI_DOUBLE,
                     fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(fullGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = true;
                Real minVal = fullGrid[0], maxVal = fullGrid[0];
                for (const auto& val : fullGrid) {
                    if (std::isnan(val) || std::isinf(val)) {
                        printf("Validation failed: found NaN or Inf value\n");
                        valid = false;
                        break;
                    }
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                if (valid) {
                    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                    if (maxVal > 1e6 || minVal < -1e6) {
                        printf("Validation failed: values out of expected range\n");
                        valid = false;
                    }
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) {
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
