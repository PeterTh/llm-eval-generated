#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// ============================================================================
// CUDA kernels
// ============================================================================

__device__ inline size_t idx3d(const size_t x, const size_t y, const size_t z,
                                const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Combined stencil + boundary kernel
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3d(x, y, z, nx, ny);

    // Boundary cells: copy from input
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    // Interior cells: 7-point stencil
    const Real center = input[idx];
    const Real left   = input[idx3d(x-1, y, z, nx, ny)];
    const Real right  = input[idx3d(x+1, y, z, nx, ny)];
    const Real front  = input[idx3d(x, y-1, z, nx, ny)];
    const Real back   = input[idx3d(x, y+1, z, nx, ny)];
    const Real bottom = input[idx3d(x, y, z-1, nx, ny)];
    const Real top    = input[idx3d(x, y, z+1, nx, ny)];
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// ============================================================================
// Host helpers
// ============================================================================

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                    const size_t nz, const size_t z_global_base) {
    // z_global_base is the global Z index of local_z=0 (bottom halo)
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = z_global_base + z;
                const size_t global_idx = gz * (nx * ny) + y * nx + x;
                grid[z * (nx * ny) + y * nx + x] = (global_idx % 19) * 1.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    bool hasBad = false;
#pragma omp parallel for reduction(||:hasBad)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            hasBad = true;
        }
    }
    if (hasBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 1; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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

// ============================================================================
// Main: hybrid MPI + OpenMP + CUDA
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

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

    // Clamp nRanks to nz (can't have more ranks than Z cells)
    if (nRanks > static_cast<int>(nz)) nRanks = static_cast<int>(nz);

    // ---- Domain decomposition along Z axis ----
    // Interior Z cells: [1, nz-1)  (nz-2 cells, since z=0 and z=nz-1 are boundaries)
    // Each rank owns a chunk of interior Z cells plus 2 halo layers.
    // local_z=0 is the bottom halo (global z = z_start-1)
    // local_z=1..local_interior_z are interior (global z = z_start..z_start+local_interior_z-1)
    // local_z=local_nz-1 is the top halo (global z = z_start+local_interior_z)

    size_t interior_nz = nz - 2;
    size_t base_int_nz = interior_nz / nRanks;
    size_t int_remainder = interior_nz % nRanks;

    size_t local_interior_z = base_int_nz + (rank < static_cast<int>(int_remainder) ? 1 : 0);
    size_t local_nz = local_interior_z + 2;  // +2 for halo layers

    // Global Z index of the first interior cell for this rank
    size_t z_start = 1;
    for (int r = 0; r < rank; ++r) {
        z_start += base_int_nz + (r < static_cast<int>(int_remainder) ? 1 : 0);
    }
    // z_global_base is the global Z index of local_z=0 (bottom halo)
    size_t z_global_base = z_start - 1;

    size_t local_grid_size = nx * ny * local_nz;

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", nRanks);
        printf("Local Z per rank: %zu (interior: %zu)\n", local_nz, local_interior_z);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Initialize local grid with OpenMP ----
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> h_buffer(local_grid_size);
    initializeGrid(h_buffer, nx, ny, local_nz, z_global_base);

    // ---- Allocate GPU memory (double buffered) ----
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    size_t local_bytes = local_grid_size * sizeof(Real);
    cudaMalloc(&d_grid1, local_bytes);
    cudaMalloc(&d_grid2, local_bytes);

    cudaMemcpy(d_grid1, h_buffer.data(), local_bytes, cudaMemcpyHostToDevice);

    // ---- CUDA launch configuration ----
    dim3 block(8, 8, 8);
    dim3 grid_dim((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y,
                  (local_nz + block.z - 1) / block.z);

    // ---- MPI halo exchange buffers ----
    size_t halo_size = nx * ny;
    std::vector<Real> send_bottom(halo_size);
    std::vector<Real> send_top(halo_size);
    std::vector<Real> recv_bottom(halo_size);
    std::vector<Real> recv_top(halo_size);

    int rank_below = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int rank_above = (rank == nRanks - 1) ? MPI_PROC_NULL : rank + 1;

    // ---- Stencil computation with halo exchange ----
    if (rank == 0) printf("Running stencil computation...\n");

    Real* d_input = d_grid1;
    Real* d_output = d_grid2;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Launch combined stencil + boundary kernel
        stencilKernel<<<grid_dim, block, 0, 0>>>(d_input, d_output, nx, ny, local_nz);

        // Copy output to host for halo extraction
        cudaMemcpy(h_buffer.data(), d_output, local_bytes, cudaMemcpyDeviceToHost);

        // Extract first/last interior cells for sending
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                send_bottom[y * nx + x] = h_buffer[idx3(x, y, 1, nx, ny)];
                send_top[y * nx + x] = h_buffer[idx3(x, y, local_nz - 2, nx, ny)];
            }
        }

        // Non-blocking halo exchange
        // Rank R sends first interior (local_z=1) to rank R-1 with tag 100
        // Rank R sends last interior  (local_z=local_nz-2) to rank R+1 with tag 200
        // Rank R receives from rank R-1 with tag 200 → fills bottom halo (local_z=0)
        // Rank R receives from rank R+1 with tag 100 → fills top halo (local_z=local_nz-1)
        MPI_Request reqs[4];
        int nreqs = 0;

        if (rank_below != MPI_PROC_NULL) {
            MPI_Isend(send_bottom.data(), halo_size, MPI_DOUBLE, rank_below, 100,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank_above != MPI_PROC_NULL) {
            MPI_Isend(send_top.data(), halo_size, MPI_DOUBLE, rank_above, 200,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank_below != MPI_PROC_NULL) {
            MPI_Irecv(recv_bottom.data(), halo_size, MPI_DOUBLE, rank_below, 200,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank_above != MPI_PROC_NULL) {
            MPI_Irecv(recv_top.data(), halo_size, MPI_DOUBLE, rank_above, 100,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        if (rank_below != MPI_PROC_NULL) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    h_buffer[idx3(x, y, 0, nx, ny)] = recv_bottom[y * nx + x];
                }
            }
        }
        if (rank_above != MPI_PROC_NULL) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    h_buffer[idx3(x, y, local_nz - 1, nx, ny)] = recv_top[y * nx + x];
                }
            }
        }

        // Copy updated data back to GPU
        cudaMemcpy(d_output, h_buffer.data(), local_bytes, cudaMemcpyHostToDevice);

        // Swap for next iteration
        Real* tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- Gather final results to rank 0 ----
    // d_input holds the final result
    cudaMemcpy(h_buffer.data(), d_input, local_bytes, cudaMemcpyDeviceToHost);

    // Compute sendcounts and displacements for Gatherv
    std::vector<int> sendcounts(nRanks);
    std::vector<int> displs(nRanks);
    size_t offset = 0;
    for (int r = 0; r < nRanks; ++r) {
        size_t r_int_z = base_int_nz + (r < static_cast<int>(int_remainder) ? 1 : 0);
        size_t r_local_nz = r_int_z + 2;
        sendcounts[r] = static_cast<int>(nx * ny * r_local_nz);
        displs[r] = static_cast<int>(offset);
        offset += nx * ny * r_local_nz;
    }

    size_t gathered_size = offset;
    std::vector<Real> gathered(gathered_size);

    MPI_Gatherv(h_buffer.data(), static_cast<int>(local_grid_size), MPI_DOUBLE,
                gathered.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 assembles the global grid
    if (rank == 0) {
        std::vector<Real> assembled(nx * ny * nz);
        size_t g_z = 0;
        size_t l_offset = 0;

        for (int r = 0; r < nRanks; ++r) {
            size_t r_int_z = base_int_nz + (r < static_cast<int>(int_remainder) ? 1 : 0);
            // Copy bottom halo for rank 0 (global z=0)
            if (r == 0) {
                for (size_t j = 0; j < nx * ny; ++j) {
                    assembled[g_z * nx * ny + j] = gathered[l_offset * nx * ny + j];
                }
                g_z++;
                l_offset++;
            }
            // Copy interior cells
            for (size_t lz = 0; lz < r_int_z; ++lz) {
                for (size_t j = 0; j < nx * ny; ++j) {
                    assembled[g_z * nx * ny + j] = gathered[(l_offset + lz) * nx * ny + j];
                }
                g_z++;
            }
            // Copy top halo for last rank (global z=nz-1)
            if (r == nRanks - 1) {
                for (size_t j = 0; j < nx * ny; ++j) {
                    assembled[g_z * nx * ny + j] = gathered[(l_offset + r_int_z) * nx * ny + j];
                }
                g_z++;
            }
            l_offset += r_int_z + 2;
        }

        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(assembled, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(assembled, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    cudaFree(d_grid1);
    cudaFree(d_grid2);

    MPI_Finalize();
    return 0;
}
