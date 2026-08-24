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

// Compute stencil for a range of local z layers
inline void computeStencilRange(const std::vector<Real>& input, std::vector<Real>& output,
                                const size_t nx, const size_t ny,
                                const size_t lz_from, const size_t lz_to) {
    for (size_t lz = lz_from; lz <= lz_to; ++lz) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t i = idx3(x, y, lz, nx, ny);
                output[i] = (input[i]
                    + input[idx3(x-1, y, lz, nx, ny)]
                    + input[idx3(x+1, y, lz, nx, ny)]
                    + input[idx3(x, y-1, lz, nx, ny)]
                    + input[idx3(x, y+1, lz, nx, ny)]
                    + input[idx3(x, y, lz-1, nx, ny)]
                    + input[idx3(x, y, lz+1, nx, ny)]) / 7.0;
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

    // 1D domain decomposition along Z axis
    const size_t base_nz = nz / (size_t)nprocs;
    const size_t remainder = nz % (size_t)nprocs;
    const size_t local_nz = base_nz + ((size_t)rank < remainder ? 1 : 0);
    const size_t gz_start = (size_t)rank * base_nz + std::min((size_t)rank, remainder);

    // Local grids: local_nz real z-planes at indices [1..local_nz], ghost layers at 0 and local_nz+1
    const size_t plane_size = nx * ny;
    const size_t local_alloc = plane_size * (local_nz + 2);
    std::vector<Real> grid1(local_alloc, 0.0);
    std::vector<Real> grid2(local_alloc, 0.0);

    // Initialize local grid using global indices for deterministic values
    if (rank == 0) printf("Initializing grid...\n");
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = gz_start + lz - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = idx3(x, y, gz, nx, ny);
                grid1[idx3(x, y, lz, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (local_nz > 0) {
        // Precompute local z range for interior stencil updates (global interior: z in [1, nz-2])
        const size_t lz_begin = (gz_start == 0) ? 2 : 1;
        const size_t lz_end = (gz_start + local_nz >= nz && local_nz > 0) ? local_nz - 1 : local_nz;

        const bool has_bottom_neighbor = (rank > 0);
        const bool has_top_neighbor = (rank < nprocs - 1);

        // Safe interior range: layers that don't read from ghost planes
        const size_t safe_begin = has_bottom_neighbor ? std::max(lz_begin, (size_t)2) : lz_begin;
        const size_t safe_end = has_top_neighbor ? std::min(lz_end, local_nz - 1) : lz_end;

        for (int iter = 0; iter < iterations; ++iter) {
            auto& input = (iter % 2 == 0) ? grid1 : grid2;
            auto& output = (iter % 2 == 0) ? grid2 : grid1;

            // --- Non-blocking halo exchange ---
            MPI_Request reqs[4];
            int nreqs = 0;

            if (has_bottom_neighbor) {
                MPI_Isend(&input[1 * plane_size], plane_size, MPI_DOUBLE,
                          rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
                MPI_Irecv(&input[0], plane_size, MPI_DOUBLE,
                          rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            }
            if (has_top_neighbor) {
                MPI_Isend(&input[local_nz * plane_size], plane_size, MPI_DOUBLE,
                          rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
                MPI_Irecv(&input[(local_nz + 1) * plane_size], plane_size, MPI_DOUBLE,
                          rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            }

            // --- Compute safe interior (overlapped with communication) ---
            if (safe_begin <= safe_end) {
                computeStencilRange(input, output, nx, ny, safe_begin, safe_end);
            }

            // --- Wait for halo data ---
            if (nreqs > 0) {
                MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
            }

            // --- Compute ghost-dependent layers ---
            if (lz_begin <= lz_end) {
                for (size_t lz = lz_begin; lz <= lz_end; ++lz) {
                    if (lz >= safe_begin && lz <= safe_end) continue;
                    computeStencilRange(input, output, nx, ny, lz, lz);
                }
            }

            // --- Copy boundary values ---
            for (size_t lz = 1; lz <= local_nz; ++lz) {
                const size_t gz = gz_start + lz - 1;
                if (gz == 0 || gz == nz - 1) {
                    // Entire z-plane is boundary
                    for (size_t y = 0; y < ny; ++y) {
                        for (size_t x = 0; x < nx; ++x) {
                            const size_t i = idx3(x, y, lz, nx, ny);
                            output[i] = input[i];
                        }
                    }
                } else {
                    // x and y boundaries only
                    for (size_t y = 0; y < ny; ++y) {
                        output[idx3(0, y, lz, nx, ny)] = input[idx3(0, y, lz, nx, ny)];
                        output[idx3(nx-1, y, lz, nx, ny)] = input[idx3(nx-1, y, lz, nx, ny)];
                    }
                    for (size_t x = 0; x < nx; ++x) {
                        output[idx3(x, 0, lz, nx, ny)] = input[idx3(x, 0, lz, nx, ny)];
                        output[idx3(x, ny-1, lz, nx, ny)] = input[idx3(x, ny-1, lz, nx, ny)];
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather full grid to rank 0 ---
    const auto& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t rnz = base_nz + ((size_t)r < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(rnz * plane_size);
    }
    displs[0] = 0;
    for (int r = 1; r < nprocs; ++r) {
        displs[r] = displs[r-1] + recvcounts[r-1];
    }

    std::vector<Real> globalGrid;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
    }

    MPI_Gatherv(&finalGrid[1 * plane_size],
                static_cast<int>(local_nz * plane_size), MPI_DOUBLE,
                rank == 0 ? globalGrid.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);

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
