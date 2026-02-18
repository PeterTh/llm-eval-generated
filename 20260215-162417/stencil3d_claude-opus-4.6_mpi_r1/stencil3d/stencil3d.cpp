#include <algorithm>
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
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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

    // Decompose z-dimension among MPI ranks
    const size_t base_nz = nz / (size_t)nprocs;
    const size_t rem = nz % (size_t)nprocs;
    const size_t local_nz = base_nz + ((size_t)rank < rem ? 1 : 0);
    const size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, rem);

    const size_t plane_size = nx * ny;
    // Local grid: ghost plane at index 0, owned planes at 1..local_nz, ghost at local_nz+1
    const size_t local_total = plane_size * (local_nz + 2);

    // Neighbor ranks (inactive ranks with local_nz==0 are always at the tail)
    const int last_active = (int)std::min(nz, (size_t)nprocs) - 1;
    const int prev_rank = (rank > 0 && local_nz > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < last_active && local_nz > 0) ? rank + 1 : MPI_PROC_NULL;

    std::vector<Real> local_grid1(local_total, 0.0);
    std::vector<Real> local_grid2(local_total, 0.0);

    // Initialize owned cells using global indices
    if (rank == 0) printf("Initializing grid...\n");
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = lz * plane_size + y * nx + x;
                const size_t global_idx = gz * plane_size + y * nx + x;
                local_grid1[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }

    // Local z computation range (skip global boundary z-planes)
    size_t lz_comp_start = 1;
    size_t lz_comp_end = local_nz;
    if (z_start == 0 && local_nz > 0) lz_comp_start = 2;
    if (local_nz > 0 && z_start + local_nz - 1 == nz - 1) lz_comp_end = local_nz - 1;

    const bool dep_start = (prev_rank != MPI_PROC_NULL);
    const bool dep_end = (next_rank != MPI_PROC_NULL);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& input  = (iter % 2 == 0) ? local_grid1 : local_grid2;
        std::vector<Real>& output = (iter % 2 == 0) ? local_grid2 : local_grid1;

        if (local_nz == 0) continue;

        // Non-blocking halo exchange
        MPI_Request reqs[4];
        int nreqs = 0;
        if (prev_rank != MPI_PROC_NULL) {
            MPI_Irecv(&input[0], plane_size, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(&input[plane_size], plane_size, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (next_rank != MPI_PROC_NULL) {
            MPI_Irecv(&input[(local_nz + 1) * plane_size], plane_size, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(&input[local_nz * plane_size], plane_size, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        // Compute interior z-planes that do not depend on ghost data
        const size_t inner_start = lz_comp_start + (dep_start ? 1 : 0);
        const size_t inner_end   = lz_comp_end   - (dep_end   ? 1 : 0);
        if (lz_comp_start <= lz_comp_end && inner_start <= inner_end) {
            for (size_t lz = inner_start; lz <= inner_end; ++lz) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = lz * plane_size + y * nx + x;
                        output[idx] = (input[idx] +
                                       input[idx - 1] + input[idx + 1] +
                                       input[idx - nx] + input[idx + nx] +
                                       input[idx - plane_size] + input[idx + plane_size]) / 7.0;
                    }
                }
            }
        }

        // Wait for halo data
        if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Compute ghost-dependent z-planes
        if (lz_comp_start <= lz_comp_end) {
            if (dep_start) {
                const size_t lz = lz_comp_start;
                for (size_t y = 1; y < ny - 1; ++y) {
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = lz * plane_size + y * nx + x;
                        output[idx] = (input[idx] +
                                       input[idx - 1] + input[idx + 1] +
                                       input[idx - nx] + input[idx + nx] +
                                       input[idx - plane_size] + input[idx + plane_size]) / 7.0;
                    }
                }
            }
            if (dep_end && !(dep_start && lz_comp_end == lz_comp_start)) {
                const size_t lz = lz_comp_end;
                for (size_t y = 1; y < ny - 1; ++y) {
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = lz * plane_size + y * nx + x;
                        output[idx] = (input[idx] +
                                       input[idx - 1] + input[idx + 1] +
                                       input[idx - nx] + input[idx + nx] +
                                       input[idx - plane_size] + input[idx + plane_size]) / 7.0;
                    }
                }
            }
        }

        // Copy boundary values for owned planes
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            const size_t gz = z_start + lz - 1;
            if (gz == 0 || gz == nz - 1) {
                // Entire z-boundary plane
                std::memcpy(&output[lz * plane_size], &input[lz * plane_size],
                            plane_size * sizeof(Real));
            } else {
                // x boundaries
                for (size_t y = 0; y < ny; ++y) {
                    output[lz * plane_size + y * nx]          = input[lz * plane_size + y * nx];
                    output[lz * plane_size + y * nx + nx - 1] = input[lz * plane_size + y * nx + nx - 1];
                }
                // y boundaries
                for (size_t x = 1; x < nx - 1; ++x) {
                    output[lz * plane_size + x]                  = input[lz * plane_size + x];
                    output[lz * plane_size + (ny - 1) * nx + x]  = input[lz * plane_size + (ny - 1) * nx + x];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long duration_ms = (long)(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full grid on rank 0 for output / validation
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? local_grid1 : local_grid2;

    int local_count = (int)(local_nz * plane_size);
    std::vector<int> recvcounts, displs;
    if (rank == 0) { recvcounts.resize(nprocs); displs.resize(nprocs); }
    MPI_Gather(&local_count, 1, MPI_INT,
               rank == 0 ? recvcounts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < nprocs; ++r)
            displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    std::vector<Real> globalGrid;
    if (rank == 0) globalGrid.resize(nx * ny * nz);

    MPI_Gatherv(local_nz > 0 ? &finalLocal[plane_size] : nullptr, local_count, MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            Real minVal = globalGrid[0], maxVal = globalGrid[0];
            for (const auto& val : globalGrid) {
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
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
