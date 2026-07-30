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

// 3D index calculation (works for both local grids with ghosts and global grids)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                              const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost layers with neighboring MPI ranks.
// The local grid has (nz_local + 2) z-layers: layer 0 and nz_local+1 are ghosts,
// layers 1..nz_local are owned interior cells.
static void exchangeHalos(std::vector<Real>& grid, const size_t nx, const size_t ny,
                           const size_t nz_local, const int rank, const int nprocs,
                           MPI_Comm comm) {
    const size_t plane = nx * ny;
    MPI_Status status;

    // Exchange with neighbor above: send our top owned layer, receive into top ghost
    if (rank < nprocs - 1) {
        MPI_Sendrecv(
            &grid[idx3(0, 0, nz_local, nx, ny)], plane, MPI_DOUBLE, rank + 1, 0,
            &grid[idx3(0, 0, nz_local + 1, nx, ny)], plane, MPI_DOUBLE, rank + 1, 1,
            comm, &status);
    }

    // Exchange with neighbor below: send our bottom owned layer, receive into bottom ghost
    if (rank > 0) {
        MPI_Sendrecv(
            &grid[idx3(0, 0, 1, nx, ny)], plane, MPI_DOUBLE, rank - 1, 1,
            &grid[idx3(0, 0, 0, nx, ny)], plane, MPI_DOUBLE, rank - 1, 0,
            comm, &status);
    }
}

// 7-point stencil computation on the local subdomain.
// input and output include ghost cells ((nz_local + 2) layers in z).
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny,
                      const size_t nz_local, const size_t nz_global,
                      const size_t z_start) {
    // Process interior points where all 7 neighbors are available
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t gz = z_start + z - 1;
        if (gz >= 1 && gz <= nz_global - 2) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t c = idx3(x, y, z, nx, ny);
                    output[c] = (input[c] +
                                 input[idx3(x - 1, y, z, nx, ny)] +
                                 input[idx3(x + 1, y, z, nx, ny)] +
                                 input[idx3(x, y - 1, z, nx, ny)] +
                                 input[idx3(x, y + 1, z, nx, ny)] +
                                 input[idx3(x, y, z - 1, nx, ny)] +
                                 input[idx3(x, y, z + 1, nx, ny)]) / 7.0;
                }
            }
        }
    }

    // Copy boundary values (x/y boundaries on all ranks, z boundaries on edge ranks)
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t gz = z_start + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                    gz == 0 || gz == nz_global - 1) {
                    const size_t c = idx3(x, y, z, nx, ny);
                    output[c] = input[c];
                }
            }
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

    // Parse command line arguments (all ranks parse identically)
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition along Z axis
    size_t nz_local = nz / nprocs;
    size_t nz_rem = nz % nprocs;
    if ((size_t)rank < nz_rem) nz_local++;

    size_t z_start = 0;
    for (int r = 0; r < rank; ++r) {
        size_t r_nz = nz / nprocs;
        if ((size_t)r < nz_rem) r_nz++;
        z_start += r_nz;
    }

    if (rank == 0) {
        printf("Initializing grid...\n");
    }

    // Allocate local grids with ghost cells (extra 2 z-layers)
    const size_t local_nz_total = nz_local + 2;
    const size_t local_grid_size = nx * ny * local_nz_total;
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    // Initialize owned cells (local z = 1 .. nz_local)
    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t gz = z_start + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_idx = gz * (nx * ny) + y * nx + x;
                grid1[idx3(x, y, z, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }

    // Fill ghost cells with initial halo exchange (sets interior ghost layers;
    // out-of-domain ghosts on edge ranks are never accessed)
    exchangeHalos(grid1, nx, ny, nz_local, rank, nprocs, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        auto& in  = (iter % 2 == 0) ? grid1 : grid2;
        auto& out = (iter % 2 == 0) ? grid2 : grid1;

        stencilIteration(in, out, nx, ny, nz_local, nz, z_start);

        if (iter < iterations - 1) {
            exchangeHalos(out, nx, ny, nz_local, rank, nprocs, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Reduce timing across all ranks (report maximum wall-clock time)
    long long local_ms = duration.count();
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full grid on rank 0 for output and validation
    std::vector<Real> fullGrid;
    if (printResults || validate) {
        // Build displacement table for MPI_Gatherv
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t r_nz = nz / nprocs;
            if ((size_t)r < nz_rem) r_nz++;
            recvcounts[r] = static_cast<int>(nx * ny * r_nz);
            displs[r] = offset;
            offset += recvcounts[r];
        }

        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
        }

        // Owned cells are at local_z = 1 .. nz_local (contiguous block after bottom ghost)
        const Real* sendbuf = &grid1[idx3(0, 0, 1, nx, ny)];
        if (iterations % 2 != 0) {
            sendbuf = &grid2[idx3(0, 0, 1, nx, ny)];
        }

        MPI_Gatherv(sendbuf, static_cast<int>(nx * ny * nz_local), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(fullGrid, "Grid");
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool local_valid = true;
        if (rank == 0) {
            local_valid = validateResult(fullGrid, nx, ny, nz);
        }
        // Broadcast the validation result so all ranks agree on the exit code
        int valid_int = local_valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid_int) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Finalize();
        return valid_int ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
