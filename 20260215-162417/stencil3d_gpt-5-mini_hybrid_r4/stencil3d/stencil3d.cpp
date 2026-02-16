#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (generic)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local index within a buffer that includes halo planes (z padded by +2)
inline constexpr size_t local_idx3(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny) noexcept {
    return z_local * (nx * ny) + y * nx + x;
}

// CUDA kernel: 7-point stencil. The input and output buffers include halo planes in Z (first and last planes).
extern "C" __global__ void stencil_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                           size_t nx, size_t ny, size_t local_nz, size_t z_offset, size_t global_nz) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z; // z in [1..local_nz]

    if (x >= nx || y >= ny || z < 1 || z > local_nz) return;

    const size_t g_z = z_offset + (z - 1);
    const size_t idx = local_idx3(x, y, z, nx, ny);

    // Boundary points (global) are copied
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || g_z == 0 || g_z == global_nz - 1) {
        out[idx] = in[idx];
        return;
    }

    const Real center = in[idx];
    const Real left = in[local_idx3(x - 1, y, z, nx, ny)];
    const Real right = in[local_idx3(x + 1, y, z, nx, ny)];
    const Real front = in[local_idx3(x, y - 1, z, nx, ny)];
    const Real back = in[local_idx3(x, y + 1, z, nx, ny)];
    const Real bottom = in[local_idx3(x, y, z - 1, nx, ny)];
    const Real top = in[local_idx3(x, y, z + 1, nx, ny)];

    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks (parallelized with OpenMP)
    // 1. No NaN or Inf values
    bool ok = true;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
#pragma omp critical
            { ok = false; }
        }
    }
    if (!ok) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < grid.size(); ++i) {
#pragma omp critical
            {
                minVal = std::min(minVal, grid[i]);
                maxVal = std::max(maxVal, grid[i]);
            }
        }
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
    // Initialize MPI and CUDA-aware hybrid execution unconditionally
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 to keep semantics, but propagate)
    if (world_rank == 0) {
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
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
            }
        }
    }
    // Broadcast parsed values to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (world_rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU device per rank
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        int device_id = world_rank % device_count;
        cudaSetDevice(device_id);
    }

    // Partition Z dimension among MPI ranks
    size_t base = nz / world_size;
    size_t rem = nz % world_size;
    size_t local_nz = base + (world_rank < (int)rem ? 1 : 0);
    size_t z_offset = 0;
    for (int r = 0; r < world_rank; ++r) {
        z_offset += base + (r < (int)rem ? 1 : 0);
    }

    const size_t local_with_halo_nz = local_nz + 2; // include top and bottom halo planes
    const size_t plane = nx * ny;
    const size_t local_size = plane * local_with_halo_nz;

    // Host buffers (pinned for efficient GPU transfers)
    Real* h_buf_in = nullptr;
    Real* h_buf_out = nullptr;
    cudaHostAlloc(&h_buf_in, local_size * sizeof(Real), cudaHostAllocPortable);
    cudaHostAlloc(&h_buf_out, local_size * sizeof(Real), cudaHostAllocPortable);

    // Initialize local buffer with global indexing so that result matches sequential run
#pragma omp parallel for collapse(3)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t g_z = z_offset + (z - 1);
                size_t g_idx = idx3(x, y, g_z, nx, ny);
                h_buf_in[local_idx3(x, y, z, nx, ny)] = (g_idx % 19) * 1.0;
            }
        }
    }
    // Initialize halo planes to boundaries (for global boundaries) or zero; exchanges will fill interior halos
    // bottom halo (z=0)
    if (z_offset == 0) {
        // global z==0 plane
#pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
            h_buf_in[local_idx3(x, y, 0, nx, ny)] = h_buf_in[local_idx3(x, y, 1, nx, ny)];
    } else {
#pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
            h_buf_in[local_idx3(x, y, 0, nx, ny)] = 0.0; // will be filled by exchange
    }
    // top halo (z=local_nz+1)
    if (z_offset + local_nz == nz) {
#pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
            h_buf_in[local_idx3(x, y, local_nz + 1, nx, ny)] = h_buf_in[local_idx3(x, y, local_nz, nx, ny)];
    } else {
#pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
            h_buf_in[local_idx3(x, y, local_nz + 1, nx, ny)] = 0.0;
    }

    // Device buffers
    Real* d_in = nullptr;
    Real* d_out = nullptr;
    cudaMalloc(&d_in, local_size * sizeof(Real));
    cudaMalloc(&d_out, local_size * sizeof(Real));

    // Copy initial data to device
    cudaMemcpy(d_in, h_buf_in, local_size * sizeof(Real), cudaMemcpyHostToDevice);

    // Helper buffers for halo exchange
    std::vector<Real> send_top(plane), send_bottom(plane), recv_top(plane), recv_bottom(plane);

    auto exchange_halos = [&](Real* device_buf) {
        // copy interior boundary planes from device to host send buffers
        // bottom interior plane is z=1 -> send to rank-1
        cudaMemcpy(send_bottom.data(), device_buf + local_idx3(0, 0, 1, nx, ny), plane * sizeof(Real), cudaMemcpyDeviceToHost);
        // top interior plane is z=local_nz -> send to rank+1
        cudaMemcpy(send_top.data(), device_buf + local_idx3(0, 0, local_nz, nx, ny), plane * sizeof(Real), cudaMemcpyDeviceToHost);

        int prev = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
        int next = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

        MPI_Request reqs[4];
        MPI_Status stats[4];

        // Exchange bottom<->prev (send bottom, recv into bottom halo)
        MPI_Isend(send_bottom.data(), plane, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_top.data(), plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[1]);
        // Exchange top<->next (send top, recv into top halo)
        MPI_Isend(send_top.data(), plane, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_bottom.data(), plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, stats);

        // Copy received halos into device buffer
        if (next != MPI_PROC_NULL) {
            cudaMemcpy(device_buf + local_idx3(0, 0, local_nz + 1, nx, ny), recv_top.data(), plane * sizeof(Real), cudaMemcpyHostToDevice);
        }
        if (prev != MPI_PROC_NULL) {
            cudaMemcpy(device_buf + local_idx3(0, 0, 0, nx, ny), recv_bottom.data(), plane * sizeof(Real), cudaMemcpyHostToDevice);
        }
    };

    // Perform initial halo exchange to populate halos before iteration 0
    exchange_halos(d_in);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Launch configuration
    dim3 block(16, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              ((local_nz + 2) + block.z - 1) / block.z); // include halos but kernel skips z=0 and z>local_nz

    for (int iter = 0; iter < iterations; ++iter) {
        // compute: d_out = stencil(d_in)
        stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, local_nz, z_offset, nz);
        cudaDeviceSynchronize();

        // swap buffers
        std::swap(d_in, d_out);

        // exchange halos of the new d_in for next iteration (or final gather)
        exchange_halos(d_in);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double local_time_ms = (t1 - t0) * 1000.0;
    double max_time_ms = 0.0;
    MPI_Reduce(&local_time_ms, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather final data to rank 0 for validation/printing
    // Copy device final data to host (skip halos when sending)
    cudaMemcpy(h_buf_in, d_in, local_size * sizeof(Real), cudaMemcpyDeviceToHost);

    // Prepare gather counts and displs (in elements)
    std::vector<int> recvcounts(world_size), displs(world_size);
    int sendcount = (int)(plane * local_nz);
    for (int r = 0; r < world_size; ++r) {
        size_t r_nz = base + (r < (int)rem ? 1 : 0);
        recvcounts[r] = (int)(plane * r_nz);
    }
    if (world_rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < world_size; ++r) displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    std::vector<Real> fullGrid;
    if (world_rank == 0) fullGrid.resize((size_t)plane * nz);

    // Create contiguous send buffer that excludes halo planes
    std::vector<Real> sendbuf((size_t)plane * local_nz);
    for (size_t z = 1; z <= local_nz; ++z) {
        memcpy(sendbuf.data() + (z - 1) * plane, h_buf_in + local_idx3(0, 0, z, nx, ny), plane * sizeof(Real));
    }

    MPI_Gatherv(sendbuf.data(), sendcount, MPI_DOUBLE,
                world_rank == 0 ? fullGrid.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %.0f ms (max across ranks)\n", max_time_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (max_time_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        const std::vector<Real>& finalGrid = fullGrid;
        if (printResults) print_results(finalGrid, "Grid");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_buf_in);
    cudaFreeHost(h_buf_out);

    MPI_Finalize();
    return 0;
}
