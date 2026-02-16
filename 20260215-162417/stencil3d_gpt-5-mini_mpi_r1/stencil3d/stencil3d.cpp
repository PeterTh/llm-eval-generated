#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t loc_idx3(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny) noexcept {
    return z_local * (nx * ny) + y * nx + x;
}

// initialize local slab using global coordinates so results match serial init
void initializeLocal(std::vector<Real>& local, const size_t nx, const size_t ny, const size_t local_nz, const size_t start_z_global) {
    const size_t slice = nx * ny;
    // local has halo slices: size (local_nz + 2) * slice
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        size_t gz = start_z_global + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = gz * (nx * ny) + y * nx + x;
                local[loc_idx3(x, y, zl, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

bool validateResultHost(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    // reuse previous validation logic
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints help/errors)
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

    const size_t global_slices = nz;
    // partition along Z
    size_t base = global_slices / (size_t)size;
    size_t rem = global_slices % (size_t)size;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    size_t start_z = (size_t)rank * base + std::min((size_t)rank, rem);

    const size_t slice = nx * ny;
    // local arrays include 2 halo slices
    std::vector<Real> local_in((local_nz + 2) * slice);
    std::vector<Real> local_out((local_nz + 2) * slice);

    // initialize interior real slices
    initializeLocal(local_in, nx, ny, local_nz, start_z);
    // set halos to 0 (will be filled via exchange or remain as boundary values)
    std::fill_n(local_in.data(), slice, 0.0); // bottom halo
    std::fill_n(local_in.data() + (local_nz + 1) * slice, slice, 0.0); // top halo

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Iterations
    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halos: send first real slice downward, receive into bottom halo; send last real slice upward, receive into top halo
        // sendrecv bottom
        MPI_Sendrecv(local_in.data() + slice, (int)slice, MPI_DOUBLE, prev, 0,
                     local_in.data(), (int)slice, MPI_DOUBLE, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // sendrecv top
        MPI_Sendrecv(local_in.data() + local_nz * slice, (int)slice, MPI_DOUBLE, next, 1,
                     local_in.data() + (local_nz + 1) * slice, (int)slice, MPI_DOUBLE, next, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute interior points (exclude global boundaries and x/y boundaries)
        for (size_t zl = 1; zl <= local_nz; ++zl) {
            size_t gz = start_z + (zl - 1);
            if (gz == 0 || gz == nz - 1) continue; // global z-boundary copied, not computed
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = loc_idx3(x, y, zl, nx, ny);
                    const Real center = local_in[idx];
                    const Real left = local_in[loc_idx3(x - 1, y, zl, nx, ny)];
                    const Real right = local_in[loc_idx3(x + 1, y, zl, nx, ny)];
                    const Real front = local_in[loc_idx3(x, y - 1, zl, nx, ny)];
                    const Real back = local_in[loc_idx3(x, y + 1, zl, nx, ny)];
                    const Real bottom = local_in[loc_idx3(x, y, zl - 1, nx, ny)];
                    const Real top = local_in[loc_idx3(x, y, zl + 1, nx, ny)];
                    local_out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }

        // Copy boundary values for points that are on global boundaries or x/y edges
        for (size_t zl = 1; zl <= local_nz; ++zl) {
            size_t gz = start_z + (zl - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
                        const size_t idx = loc_idx3(x, y, zl, nx, ny);
                        local_out[idx] = local_in[idx];
                    }
                }
            }
        }

        // swap buffers
        std::swap(local_in, local_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    // take max across ranks
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather results to rank 0 if needed
    std::vector<Real> fullGrid;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    std::vector<int> sendcounts(size);

    // compute sendcounts (number of doubles per rank)
    for (int r = 0; r < size; ++r) {
        size_t lnz = base + (r < (int)rem ? 1 : 0);
        sendcounts[r] = (int)(lnz * slice);
    }
    // displacements
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r - 1] + sendcounts[r - 1];

    if (rank == 0) {
        fullGrid.resize((size_t)nx * ny * nz);
    }

    // send only real slices (skip halos)
    MPI_Gatherv(local_in.data() + slice, (int)(local_nz * slice), MPI_DOUBLE,
                fullGrid.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_elapsed * 1000.0);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResultHost(fullGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
