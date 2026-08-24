#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation for local grid (includes ghost cells)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                              const size_t ldx, const size_t ldy) noexcept {
    return z * (ldx * ldy) + y * ldx + x;
}

// Compute global offset for a given Cartesian coordinate
inline size_t compute_global_offset(const int coord, const int ndim, const size_t size) {
    size_t offset = 0;
    const size_t base = size / static_cast<size_t>(ndim);
    const size_t rem = size % static_cast<size_t>(ndim);
    for (int i = 0; i < coord; ++i) {
        offset += base + (static_cast<size_t>(i) < rem ? 1 : 0);
    }
    return offset;
}

// Compute local size for a given Cartesian coordinate
inline size_t compute_local_size(const int coord, const int ndim, const size_t size) {
    const size_t base = size / static_cast<size_t>(ndim);
    const size_t rem = size % static_cast<size_t>(ndim);
    return base + (static_cast<size_t>(coord) < rem ? 1 : 0);
}

// Initialize local grid including ghost cells using global index
void initializeGrid(std::vector<Real>& grid,
                     const size_t local_nx, const size_t local_ny, const size_t local_nz,
                     const size_t ldx, const size_t ldy,
                     const size_t off_x, const size_t off_y, const size_t off_z,
                     const size_t global_nx, const size_t global_ny, const size_t global_nz) {
    for (size_t lz = 0; lz < local_nz + 2; ++lz) {
        for (size_t ly = 0; ly < local_ny + 2; ++ly) {
            for (size_t lx = 0; lx < local_nx + 2; ++lx) {
                const size_t idx = idx3(lx, ly, lz, ldx, ldy);

                // Compute global coordinates (ghost cells map to neighbor interior or boundary)
                ssize_t gx = static_cast<ssize_t>(off_x) + static_cast<ssize_t>(lx) - 1;
                ssize_t gy = static_cast<ssize_t>(off_y) + static_cast<ssize_t>(ly) - 1;
                ssize_t gz = static_cast<ssize_t>(off_z) + static_cast<ssize_t>(lz) - 1;

                // Clamp to global domain boundaries
                if (gx < 0) gx = 0;
                if (gx >= static_cast<ssize_t>(global_nx)) gx = static_cast<ssize_t>(global_nx) - 1;
                if (gy < 0) gy = 0;
                if (gy >= static_cast<ssize_t>(global_ny)) gy = static_cast<ssize_t>(global_ny) - 1;
                if (gz < 0) gz = 0;
                if (gz >= static_cast<ssize_t>(global_nz)) gz = static_cast<ssize_t>(global_nz) - 1;

                const size_t global_idx = static_cast<size_t>(gz) * global_nx * global_ny +
                                          static_cast<size_t>(gy) * global_nx +
                                          static_cast<size_t>(gx);
                grid[idx] = static_cast<Real>((global_idx % 19) * 1.0);
            }
        }
    }
}

// MPI face types for halo exchange
struct FaceTypes {
    MPI_Datatype send_left, recv_left;
    MPI_Datatype send_right, recv_right;
    MPI_Datatype send_front, recv_front;
    MPI_Datatype send_back, recv_back;
    MPI_Datatype send_bottom, recv_bottom;
    MPI_Datatype send_top, recv_top;
};

void createFaceTypes(const size_t local_nx, const size_t local_ny, const size_t local_nz,
                     FaceTypes& types) {
    const int ldx = static_cast<int>(local_nx + 2);
    const int ldy = static_cast<int>(local_ny + 2);
    const int ldz = static_cast<int>(local_nz + 2);
    const int fullsize[3] = {ldz, ldy, ldx};

    // X-direction: y-z planes (constant x)
    {
        const int subsize[3] = {static_cast<int>(local_nz), static_cast<int>(local_ny), 1};
        int start[3];
        start[0] = 1; start[1] = 1; start[2] = static_cast<int>(local_nx);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_right);
        start[2] = 0;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_left);
        start[2] = 1;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_left);
        start[2] = static_cast<int>(local_nx + 1);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_right);
    }

    // Y-direction: x-z planes (constant y)
    {
        const int subsize[3] = {static_cast<int>(local_nz), 1, static_cast<int>(local_nx)};
        int start[3];
        start[0] = 1; start[1] = 1; start[2] = 1;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_front);
        start[1] = 0;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_front);
        start[1] = static_cast<int>(local_ny);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_back);
        start[1] = static_cast<int>(local_ny + 1);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_back);
    }

    // Z-direction: x-y planes (constant z)
    {
        const int subsize[3] = {1, static_cast<int>(local_ny), static_cast<int>(local_nx)};
        int start[3];
        start[0] = 1; start[1] = 1; start[2] = 1;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_bottom);
        start[0] = 0;
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_bottom);
        start[0] = static_cast<int>(local_nz);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.send_top);
        start[0] = static_cast<int>(local_nz + 1);
        MPI_Type_create_subarray(3, fullsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &types.recv_top);
    }

    // Commit all types
    MPI_Type_commit(&types.send_right);
    MPI_Type_commit(&types.recv_left);
    MPI_Type_commit(&types.send_left);
    MPI_Type_commit(&types.recv_right);
    MPI_Type_commit(&types.send_front);
    MPI_Type_commit(&types.recv_front);
    MPI_Type_commit(&types.send_back);
    MPI_Type_commit(&types.recv_back);
    MPI_Type_commit(&types.send_bottom);
    MPI_Type_commit(&types.recv_bottom);
    MPI_Type_commit(&types.send_top);
    MPI_Type_commit(&types.recv_top);
}

void destroyFaceTypes(FaceTypes& types) {
    MPI_Type_free(&types.send_right);
    MPI_Type_free(&types.recv_left);
    MPI_Type_free(&types.send_left);
    MPI_Type_free(&types.recv_right);
    MPI_Type_free(&types.send_front);
    MPI_Type_free(&types.recv_front);
    MPI_Type_free(&types.send_back);
    MPI_Type_free(&types.recv_back);
    MPI_Type_free(&types.send_bottom);
    MPI_Type_free(&types.recv_bottom);
    MPI_Type_free(&types.send_top);
    MPI_Type_free(&types.recv_top);
}

// Exchange halo cells with all six neighbors using deadlock-free pairwise communication
void exchangeHalos(std::vector<Real>& grid, const FaceTypes& types, MPI_Comm cart_comm) {
    int left, right, front, back, bottom, top;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &front, &back);
    MPI_Cart_shift(cart_comm, 2, 1, &bottom, &top);

    MPI_Status status;

    // X direction: pair send-right with recv-left, then send-left with recv-right
    if (left != MPI_PROC_NULL || right != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[0], 1, types.send_right, right, 0,
                     &grid[0], 1, types.recv_left, left, 0,
                     cart_comm, &status);
        MPI_Sendrecv(&grid[0], 1, types.send_left, left, 1,
                     &grid[0], 1, types.recv_right, right, 1,
                     cart_comm, &status);
    }

    // Y direction
    if (front != MPI_PROC_NULL || back != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[0], 1, types.send_back, back, 2,
                     &grid[0], 1, types.recv_front, front, 2,
                     cart_comm, &status);
        MPI_Sendrecv(&grid[0], 1, types.send_front, front, 3,
                     &grid[0], 1, types.recv_back, back, 3,
                     cart_comm, &status);
    }

    // Z direction
    if (bottom != MPI_PROC_NULL || top != MPI_PROC_NULL) {
        MPI_Sendrecv(&grid[0], 1, types.send_top, top, 4,
                     &grid[0], 1, types.recv_bottom, bottom, 4,
                     cart_comm, &status);
        MPI_Sendrecv(&grid[0], 1, types.send_bottom, bottom, 5,
                     &grid[0], 1, types.recv_top, top, 5,
                     cart_comm, &status);
    }
}

// 7-point stencil with MPI halo exchange and global boundary preservation
void stencilIterationMPI(const std::vector<Real>& input,
                          std::vector<Real>& output,
                          const size_t local_nx, const size_t local_ny, const size_t local_nz,
                          const size_t ldx, const size_t ldy,
                          const size_t off_x, const size_t off_y, const size_t off_z,
                          const size_t global_nx, const size_t global_ny, const size_t global_nz,
                          const FaceTypes& types, MPI_Comm cart_comm) {

    // 1. Exchange halo cells (neighbor's interior -> our ghost cells)
    exchangeHalos(const_cast<std::vector<Real>&>(input), types, cart_comm);

    // 2. Compute stencil for all interior cells preserving global boundary values
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            for (size_t x = 1; x <= local_nx; ++x) {
                const size_t idx = idx3(x, y, z, ldx, ldy);
                const size_t gx = off_x + x - 1;
                const size_t gy = off_y + y - 1;
                const size_t gz = off_z + z - 1;

                // Global boundary cells are copied; interior cells use 7-point stencil
                if (gx == 0 || gx == global_nx - 1 ||
                    gy == 0 || gy == global_ny - 1 ||
                    gz == 0 || gz == global_nz - 1) {
                    output[idx] = input[idx];
                } else {
                    const Real center = input[idx];
                    const Real left = input[idx3(x - 1, y, z, ldx, ldy)];
                    const Real right = input[idx3(x + 1, y, z, ldx, ldy)];
                    const Real front = input[idx3(x, y - 1, z, ldx, ldy)];
                    const Real back = input[idx3(x, y + 1, z, ldx, ldy)];
                    const Real bottom = input[idx3(x, y, z - 1, ldx, ldy)];
                    const Real top = input[idx3(x, y, z + 1, ldx, ldy)];

                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
}

// Parallel validation using MPI reductions
bool validateResultMPI(const std::vector<Real>& grid,
                        const size_t local_nx, const size_t local_ny, const size_t local_nz,
                        const size_t ldx, const size_t ldy,
                        MPI_Comm cart_comm) {
    const int rank = [cart_comm]() { int r; MPI_Comm_rank(cart_comm, &r); return r; }();

    // 1. Check for NaN or Inf values in interior cells
    int local_nan_inf = 0;
    for (size_t z = 1; z <= local_nz && !local_nan_inf; ++z) {
        for (size_t y = 1; y <= local_ny && !local_nan_inf; ++y) {
            for (size_t x = 1; x <= local_nx && !local_nan_inf; ++x) {
                const Real val = grid[idx3(x, y, z, ldx, ldy)];
                if (std::isnan(val) || std::isinf(val)) {
                    local_nan_inf = 1;
                }
            }
        }
    }
    int global_nan_inf = 0;
    MPI_Allreduce(&local_nan_inf, &global_nan_inf, 1, MPI_INT, MPI_MAX, cart_comm);

    if (global_nan_inf) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Compute min/max values across all interior cells
    Real local_min = grid[idx3(1, 1, 1, ldx, ldy)];
    Real local_max = grid[idx3(1, 1, 1, ldx, ldy)];
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            for (size_t x = 1; x <= local_nx; ++x) {
                const Real val = grid[idx3(x, y, z, ldx, ldy)];
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);
            }
        }
    }
    Real global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, cart_comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
    }

    if (global_max > 1e6 || global_min < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Gather the full grid onto rank 0 for results output
void gatherFullGrid(const std::vector<Real>& local_grid,
                     std::vector<Real>& full_grid,
                     const size_t local_nx, const size_t local_ny, const size_t local_nz,
                     const size_t ldx, const size_t ldy,
                     const int rank, const int num_procs,
                     const int dims[3], const size_t nx, const size_t ny, const size_t nz,
                     MPI_Comm cart_comm) {
    // Pack interior cells into contiguous buffer
    const size_t local_size = local_nx * local_ny * local_nz;
    std::vector<Real> local_interior(local_size);
    size_t dst = 0;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 1; y <= local_ny; ++y) {
            for (size_t x = 1; x <= local_nx; ++x) {
                local_interior[dst++] = local_grid[idx3(x, y, z, ldx, ldy)];
            }
        }
    }

    // Gather sizes from all processes
    std::vector<int> recv_counts(num_procs);
    int my_size = static_cast<int>(local_size);
    MPI_Gather(&my_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, cart_comm);

    // Compute displacements in the flat gather buffer
    std::vector<int> displs(num_procs, 0);
    std::vector<Real> gathered;
    if (rank == 0) {
        for (int i = 1; i < num_procs; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
        const int total = displs[num_procs - 1] + recv_counts[num_procs - 1];
        gathered.resize(static_cast<size_t>(total));
    }

    // Gather all local interior data to rank 0
    MPI_Gatherv(local_interior.data(), my_size, MPI_DOUBLE,
                gathered.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, cart_comm);

    // Rank 0 reconstructs the full grid
    if (rank == 0) {
        full_grid.assign(nx * ny * nz, 0.0);
        for (int p = 0; p < num_procs; ++p) {
            int pcoords[3];
            MPI_Cart_coords(cart_comm, p, 3, pcoords);

            const size_t p_off_x = compute_global_offset(pcoords[0], dims[0], nx);
            const size_t p_off_y = compute_global_offset(pcoords[1], dims[1], ny);
            const size_t p_off_z = compute_global_offset(pcoords[2], dims[2], nz);
            const size_t p_nx = compute_local_size(pcoords[0], dims[0], nx);
            const size_t p_ny = compute_local_size(pcoords[1], dims[1], ny);
            const size_t p_nz = compute_local_size(pcoords[2], dims[2], nz);

            size_t src_off = static_cast<size_t>(displs[p]);
            for (size_t z = 0; z < p_nz; ++z) {
                for (size_t y = 0; y < p_ny; ++y) {
                    for (size_t x = 0; x < p_nx; ++x) {
                        const size_t gidx = (p_off_z + z) * nx * ny + (p_off_y + y) * nx + (p_off_x + x);
                        full_grid[gidx] = gathered[src_off++];
                    }
                }
            }
        }
    }
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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Parse command line arguments (duplicated on all ranks)
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
        printf("3D Stencil Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Create 3D Cartesian topology
    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    MPI_Dims_create(num_procs, 3, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Compute local grid sizes and global offsets
    const size_t local_nx = compute_local_size(coords[0], dims[0], nx);
    const size_t local_ny = compute_local_size(coords[1], dims[1], ny);
    const size_t local_nz = compute_local_size(coords[2], dims[2], nz);

    const size_t off_x = compute_global_offset(coords[0], dims[0], nx);
    const size_t off_y = compute_global_offset(coords[1], dims[1], ny);
    const size_t off_z = compute_global_offset(coords[2], dims[2], nz);

    // Local grid dimensions with ghost cells
    const size_t ldx = local_nx + 2;
    const size_t ldy = local_ny + 2;
    const size_t ldz = local_nz + 2;

    const size_t local_grid_size = ldx * ldy * ldz;

    // Create MPI face types
    FaceTypes types;
    createFaceTypes(local_nx, local_ny, local_nz, types);

    // Allocate local grids with ghost cells
    std::vector<Real> grid1(local_grid_size);
    std::vector<Real> grid2(local_grid_size);

    // Initialize local grid portion (including ghost cells)
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, local_nx, local_ny, local_nz, ldx, ldy,
                   off_x, off_y, off_z, nx, ny, nz);
    std::copy(grid1.begin(), grid1.end(), grid2.begin());

    // Initial halo exchange to fill ghost cells
    exchangeHalos(grid1, types, cart_comm);
    exchangeHalos(grid2, types, cart_comm);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart_comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationMPI(grid1, grid2, local_nx, local_ny, local_nz, ldx, ldy,
                                off_x, off_y, off_z, nx, ny, nz, types, cart_comm);
        } else {
            stencilIterationMPI(grid2, grid1, local_nx, local_ny, local_nz, ldx, ldy,
                                off_x, off_y, off_z, nx, ny, nz, types, cart_comm);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, cart_comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        // Calculate performance metrics (using global interior cell count)
        double cellUpdates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                             static_cast<double>(nz - 2) * static_cast<double>(iterations);
        double mcups = cellUpdates / (static_cast<double>(global_duration_ms) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather full grid on rank 0)
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> fullGrid;
        gatherFullGrid(finalGrid, fullGrid,
                        local_nx, local_ny, local_nz, ldx, ldy,
                        rank, num_procs, dims, nx, ny, nz, cart_comm);
        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResultMPI(finalGrid, local_nx, local_ny, local_nz, ldx, ldy, cart_comm);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        destroyFaceTypes(types);
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    destroyFaceTypes(types);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
