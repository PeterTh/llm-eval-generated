#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // 1D domain decomposition along Z-axis
    const size_t plane_size = nx * ny;
    const size_t base_nz = nz / nprocs;
    const size_t rem_nz = nz % nprocs;

    size_t local_nz, z_start;
    if ((size_t)rank < rem_nz) {
        local_nz = base_nz + 1;
        z_start = (size_t)rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_start = rem_nz * (base_nz + 1) + ((size_t)rank - rem_nz) * base_nz;
    }

    if (local_nz == 0) {
        if (rank == 0) {
            printf("Error: Grid Z dimension (%zu) is smaller than number of MPI processes (%d).\n", nz, nprocs);
            printf("Please increase the Z dimension or reduce the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Buffer: halo(0) | owned(1..local_nz) | halo(local_nz+1)
    const size_t buf_nz = local_nz + 2;
    const size_t buf_size = plane_size * buf_nz;

    std::vector<Real> grid1(buf_size, 0.0);
    std::vector<Real> grid2(buf_size, 0.0);

    // Initialize using global indices to match serial semantics
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        size_t gz = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gz * plane_size + y * nx + x;
                grid1[lz * plane_size + y * nx + x] = (global_idx % 19) * 1.0;
            }
        }
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    const Real inv7 = 1.0 / 7.0;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* output = (iter % 2 == 0) ? grid2.data() : grid1.data();

        // Post non-blocking halo exchange
        MPI_Request reqs[4];
        int nreqs = 0;

        // Receive bottom halo (lz=0) from prev
        MPI_Irecv(input, (int)plane_size, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        // Receive top halo (lz=local_nz+1) from next
        MPI_Irecv(input + (local_nz + 1) * plane_size, (int)plane_size, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        // Send bottom owned (lz=1) to prev
        MPI_Isend(input + plane_size, (int)plane_size, MPI_DOUBLE, prev_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        // Send top owned (lz=local_nz) to next
        MPI_Isend(input + local_nz * plane_size, (int)plane_size, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);

        // Overlap: compute interior planes (lz=2..local_nz-1) that don't need halo data
        // Their Z-neighbors (lz-1, lz+1) are all within owned planes [1..local_nz]
        for (size_t lz = 2; lz < local_nz; ++lz) {
            size_t gz = z_start + (lz - 1);
            if (gz == 0 || gz == nz - 1) continue;
            const Real* in_base = input + lz * plane_size;
            const Real* in_below = input + (lz - 1) * plane_size;
            const Real* in_above = input + (lz + 1) * plane_size;
            Real* out_base = output + lz * plane_size;

            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t row = y * nx;
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = row + x;
                    out_base[idx] = (in_base[idx] + in_base[idx - 1] + in_base[idx + 1]
                                   + in_base[idx - nx] + in_base[idx + nx]
                                   + in_below[idx] + in_above[idx]) * inv7;
                }
            }
        }

        // Wait for halo exchange to complete
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Compute boundary-adjacent planes (lz=1 and lz=local_nz) that need halo data
        // Skip planes already computed in the overlap phase (lz in [2, local_nz-1])
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            if (lz >= 2 && lz < local_nz) continue;

            size_t gz = z_start + (lz - 1);
            if (gz == 0 || gz == nz - 1) continue;

            const Real* in_base = input + lz * plane_size;
            const Real* in_below = input + (lz - 1) * plane_size;
            const Real* in_above = input + (lz + 1) * plane_size;
            Real* out_base = output + lz * plane_size;

            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t row = y * nx;
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = row + x;
                    out_base[idx] = (in_base[idx] + in_base[idx - 1] + in_base[idx + 1]
                                   + in_base[idx - nx] + in_base[idx + nx]
                                   + in_below[idx] + in_above[idx]) * inv7;
                }
            }
        }

        // Copy boundary values (X/Y faces and global Z boundaries)
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            size_t gz = z_start + (lz - 1);
            size_t base = lz * plane_size;

            if (gz == 0 || gz == nz - 1) {
                // Copy entire plane for global Z boundary
                std::memcpy(output + base, input + base, plane_size * sizeof(Real));
            } else {
                // Copy Y=0 row
                std::memcpy(output + base, input + base, nx * sizeof(Real));
                // Copy Y=ny-1 row
                std::memcpy(output + base + (ny - 1) * nx, input + base + (ny - 1) * nx, nx * sizeof(Real));
                // Copy X=0 and X=nx-1 columns (interior Y rows only)
                for (size_t y = 1; y < ny - 1; ++y) {
                    size_t row_base = base + y * nx;
                    output[row_base] = input[row_base];
                    output[row_base + nx - 1] = input[row_base + nx - 1];
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double local_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double max_ms;
    MPI_Allreduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    long duration_ms = (long)max_ms;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        double cellUpdates = (double)(nx - 2) * (double)(ny - 2) * (double)(nz - 2) * iterations;
        double mcups = (duration_ms > 0) ? cellUpdates / (duration_ms / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Determine final grid
    Real* finalGrid = (iterations % 2 == 0) ? grid1.data() : grid2.data();

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            size_t r_local_nz = ((size_t)r < rem_nz) ? base_nz + 1 : base_nz;
            size_t r_z_start = ((size_t)r < rem_nz) ? (size_t)r * (base_nz + 1)
                                                     : rem_nz * (base_nz + 1) + ((size_t)r - rem_nz) * base_nz;
            recvcounts[r] = (int)(r_local_nz * plane_size);
            displs[r] = (int)(r_z_start * plane_size);
        }

        std::vector<Real> globalGrid;
        if (rank == 0) globalGrid.resize(nz * plane_size);

        MPI_Gatherv(finalGrid + plane_size, (int)(local_nz * plane_size), MPI_DOUBLE,
                     globalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(globalGrid, "Grid");
        }
    }

    // Validation
    if (validate) {
        // Check local portion for NaN/Inf
        int local_valid = 1;
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            size_t base = lz * plane_size;
            for (size_t i = 0; i < plane_size; ++i) {
                if (std::isnan(finalGrid[base + i]) || std::isinf(finalGrid[base + i])) {
                    local_valid = 0;
                    break;
                }
            }
            if (!local_valid) break;
        }

        // Compute local min/max
        Real local_min = 1e30, local_max = -1e30;
        for (size_t lz = 1; lz <= local_nz; ++lz) {
            size_t base = lz * plane_size;
            for (size_t i = 0; i < plane_size; ++i) {
                Real val = finalGrid[base + i];
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }

        int global_valid;
        Real global_min, global_max;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        }

        bool valid = (global_valid != 0) && (global_max <= 1e6) && (global_min >= -1e6);

        if (rank == 0) {
            printf("Validating result...\n");
            if (!global_valid) {
                printf("Validation failed: found NaN or Inf value\n");
            }
            if (global_max > 1e6 || global_min < -1e6) {
                printf("Validation failed: values out of expected range\n");
            }
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        int result = valid ? 0 : 1;
        MPI_Finalize();
        return result;
    }

    MPI_Finalize();
    return 0;
}
