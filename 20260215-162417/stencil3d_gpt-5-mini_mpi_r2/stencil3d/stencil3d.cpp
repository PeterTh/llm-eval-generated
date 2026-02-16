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

// 3D index calculation for global layout
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 3D index calculation for local (with halos) layout
inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny, const size_t local_nz_with_halo) noexcept {
    (void)local_nz_with_halo; // unused in formula, kept for clarity
    return z_local * (nx * ny) + y * nx + x;
}

void initializeLocalGrid(std::vector<Real>& grid_local, const size_t nx, const size_t ny, const size_t nz,
                         const size_t z_start, const size_t local_nz) {
    const size_t local_nz_with_halo = local_nz + 2;
    // For each local layer including halos, compute its global z coordinate and initialize if inside domain
    for (size_t zl = 0; zl < local_nz_with_halo; ++zl) {
        long global_z = static_cast<long>(z_start) + static_cast<long>(zl) - 1; // zl==1 => first owned slice
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t local_idx = idx3_local(x, y, zl, nx, ny, local_nz_with_halo);
                if (global_z >= 0 && static_cast<size_t>(global_z) < nz) {
                    size_t gidx = idx3(x, y, static_cast<size_t>(global_z), nx, ny);
                    grid_local[local_idx] = (gidx % 19) * 1.0;
                } else {
                    // Outside domain; initialize to zero (will be ignored or overwritten for boundaries)
                    grid_local[local_idx] = 0.0;
                }
            }
        }
    }
}

// Perform one local stencil iteration (input->output) on the owned slices; boundaries are copied.
void stencilIterationLocal(const std::vector<Real>& input_local,
                           std::vector<Real>& output_local,
                           const size_t nx, const size_t ny, const size_t nz,
                           const size_t z_start, const size_t local_nz) {
    const size_t local_nz_with_halo = local_nz + 2;

    // For each local layer including halos handle copying boundaries; compute interior points for owned region
    for (size_t zl = 0; zl < local_nz_with_halo; ++zl) {
        long global_z = static_cast<long>(z_start) + static_cast<long>(zl) - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t lidx = idx3_local(x, y, zl, nx, ny, local_nz_with_halo);
                // If this point is on global boundary, copy
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || global_z == 0 || global_z == static_cast<long>(nz)-1) {
                    output_local[lidx] = input_local[lidx];
                } else {
                    // Only compute for owned (non-halo) slices
                    if (zl >= 1 && zl <= local_nz) {
                        const Real center = input_local[lidx];
                        const Real left = input_local[idx3_local(x-1, y, zl, nx, ny, local_nz_with_halo)];
                        const Real right = input_local[idx3_local(x+1, y, zl, nx, ny, local_nz_with_halo)];
                        const Real front = input_local[idx3_local(x, y-1, zl, nx, ny, local_nz_with_halo)];
                        const Real back = input_local[idx3_local(x, y+1, zl, nx, ny, local_nz_with_halo)];
                        const Real bottom = input_local[idx3_local(x, y, zl-1, nx, ny, local_nz_with_halo)];
                        const Real top = input_local[idx3_local(x, y, zl+1, nx, ny, local_nz_with_halo)];
                        output_local[lidx] = (center + left + right + front + back + bottom + top) / 7.0;
                    } else {
                        // Halo regions that are not global boundaries: keep as-is
                        output_local[lidx] = input_local[lidx];
                    }
                }
            }
        }
    }
}

bool validateResultLocal(const std::vector<Real>& grid_local, const size_t nx, const size_t ny, const size_t local_nz,
                         const size_t z_start, const size_t nz, MPI_Comm comm) {
    // Compute local min/max and NaN presence over owned slices only
    Real local_min = std::numeric_limits<Real>::infinity();
    Real local_max = -std::numeric_limits<Real>::infinity();
    int local_nan = 0;
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        long global_z = static_cast<long>(z_start) + static_cast<long>(zl) - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx3_local(x, y, zl, nx, ny, local_nz+2);
                Real v = grid_local[idx];
                if (std::isnan(v) || std::isinf(v)) local_nan = 1;
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }
    // Reduce across ranks
    Real global_min, global_max;
    int global_nan;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_nan, &global_nan, 1, MPI_INT, MPI_LOR, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
    }
    if (global_nan) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (global_max > 1e6 || global_min < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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
    printf("  -r           Print results for external validation (only rank 0 prints)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    // Compute local partitioning along Z (contiguous blocks)
    size_t base = nz / static_cast<size_t>(procs);
    size_t rem = nz % static_cast<size_t>(procs);
    size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI tasks: %d\n", procs);
    }

    const size_t local_nz_with_halo = local_nz + 2;
    const size_t slice = nx * ny;
    const size_t local_size = slice * local_nz_with_halo; // includes halos

    // Allocate local grids (double buffering) including halos
    std::vector<Real> grid1_local(local_size);
    std::vector<Real> grid2_local(local_size);

    // Initialize local grid contents based on global coordinates
    initializeLocalGrid(grid1_local, nx, ny, nz, z_start, local_nz);
    // Make a copy for second buffer
    grid2_local = grid1_local;

    // Prepare neighbor ranks
    int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next = (rank == procs - 1) ? MPI_PROC_NULL : rank + 1;

    // Timing: synchronize and run local iterations with halo exchange
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? grid1_local : grid2_local;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2_local : grid1_local;

        // Exchange halos with neighbors: send owned lower slice to prev, receive upper halo from next
        // offsets
        Real* send_lower = nullptr;
        Real* send_upper = nullptr;
        Real* recv_lower = nullptr;
        Real* recv_upper = nullptr;
        if (local_nz_with_halo >= 2) {
            send_lower = const_cast<Real*>(in.data() + idx3_local(0, 0, 1, nx, ny, local_nz_with_halo));
            recv_lower = in.data() + idx3_local(0, 0, 0, nx, ny, local_nz_with_halo);
            send_upper = const_cast<Real*>(in.data() + idx3_local(0, 0, local_nz, nx, ny, local_nz_with_halo));
            recv_upper = in.data() + idx3_local(0, 0, local_nz + 1, nx, ny, local_nz_with_halo);
        }

        // Exchange lower/upper with appropriate partners; MPI handles MPI_PROC_NULL
        MPI_Status status;
        // Send lower to prev, recv upper from next
        MPI_Sendrecv(send_lower, static_cast<int>(slice), MPI_DOUBLE, prev, 0,
                     recv_upper, static_cast<int>(slice), MPI_DOUBLE, next, 0,
                     comm, &status);
        // Send upper to next, recv lower from prev
        MPI_Sendrecv(send_upper, static_cast<int>(slice), MPI_DOUBLE, next, 1,
                     recv_lower, static_cast<int>(slice), MPI_DOUBLE, prev, 1,
                     comm, &status);

        // For ranks without neighbors, ensure halos for outside-domain boundaries reflect boundary copy behavior
        if (prev == MPI_PROC_NULL) {
            // copy first owned slice into lower halo
            if (local_nz >= 1) {
                std::memcpy(in.data() + idx3_local(0,0,0,nx,ny,local_nz_with_halo),
                            in.data() + idx3_local(0,0,1,nx,ny,local_nz_with_halo),
                            slice * sizeof(Real));
            }
        }
        if (next == MPI_PROC_NULL) {
            if (local_nz >= 1) {
                std::memcpy(in.data() + idx3_local(0,0,local_nz+1,nx,ny,local_nz_with_halo),
                            in.data() + idx3_local(0,0,local_nz,nx,ny,local_nz_with_halo),
                            slice * sizeof(Real));
            }
        }

        // Now compute local stencil
        if (local_nz > 0) {
            stencilIterationLocal(in, out, nx, ny, nz, z_start, local_nz);
        } else {
            // No owned slices: ensure halos/output stay consistent
            out = in;
        }
    }

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    double local_duration = t1 - t0;
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // Compute performance on rank 0
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_duration * 1000.0));
        double cellUpdates = static_cast<double>((nx>2?nx-2:0) * (ny>2?ny-2:0) * (nz>2?nz-2:0)) * iterations;
        double mcups = (max_duration > 0.0) ? (cellUpdates / (max_duration) / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Prepare to gather final results to rank 0 if needed for printing/validation
    // Each rank contributes local_nz * slice elements
    int local_count = static_cast<int>(local_nz * slice);
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (rank == 0) {
        recvcounts.resize(procs);
        displs.resize(procs);
    }
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);

    std::vector<Real> final_global;
    if (rank == 0) {
        int total = 0;
        for (int p = 0; p < procs; ++p) {
            displs[p] = total;
            total += recvcounts[p];
        }
        final_global.resize(static_cast<size_t>(total));
    }

    // Send buffer pointer: skip halo (zl==1) if local_nz>0; else send nullptr with count 0
    Real* sendbuf = nullptr;
    if (local_nz > 0) sendbuf = ( ( ((iterations % 2) == 0) ? grid1_local.data() : grid2_local.data() ) + idx3_local(0,0,1,nx,ny,local_nz_with_halo));

    MPI_Gatherv(sendbuf, local_count, MPI_DOUBLE,
                final_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    // On rank 0, if requested, print results
    if (rank == 0) {
        if (printResults) {
            print_results(final_global, "Grid");
        }
        if (validate) {
            // run validation on gathered full grid
            bool valid = validateResult(final_global, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    } else {
        // For non-root ranks, optionally run lightweight validation to detect NaNs and range issues
        if (validate) {
            bool ok = validateResultLocal(((iterations % 2) == 0) ? grid1_local : grid2_local, nx, ny, local_nz, z_start, nz, comm);
            (void)ok; // root will print final outcome
        }
    }

    MPI_Finalize();
    return 0;
}
