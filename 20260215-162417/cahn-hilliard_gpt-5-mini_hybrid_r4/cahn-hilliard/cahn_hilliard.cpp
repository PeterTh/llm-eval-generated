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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device kernels for computing chemical potential and update
extern "C" __global__ void computeMuKernel(const double* __restrict__ c, double* __restrict__ mu,
                                            const size_t nx, const size_t ny, const size_t local_nz_with_ghosts,
                                            const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                            const size_t global_z_offset, const size_t global_nz) {
    const size_t nxny = nx * ny;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * local_nz_with_ghosts;
    if (gid >= total) return;
    size_t z = gid / nxny;
    size_t rem = gid % nxny;
    size_t y = rem / nx;
    size_t x = rem % nx;

    // Only compute for real slices (skip ghost layers 0 and local_nz+1)
    if (z == 0 || z == local_nz_with_ghosts - 1) return;

    auto local_idx = z * nxny + y * nx + x;

    // neighbor indices with clamped boundaries in x and y, z handled by ghosts
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zp = z + 1;
    size_t zn = z - 1;

    double center = c[local_idx];
    double cxx = (c[z * nxny + y * nx + xp] + c[z * nxny + y * nx + xn] - 2.0 * center);
    double cyy = (c[z * nxny + yp * nx + x] + c[z * nxny + yn * nx + x] - 2.0 * center);
    double czz = (c[zp * nxny + y * nx + x] + c[zn * nxny + y * nx + x] - 2.0 * center);

    // dx=dy=dz=1.0 in original
    double lap = cxx + cyy + czz;

    mu[local_idx] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
                    + 3.0 * center + center * center * center - gamma * lap;
}

extern "C" __global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu, double* __restrict__ cnew,
                                          const size_t nx, const size_t ny, const size_t local_nz_with_ghosts,
                                          const double D, const double dt) {
    const size_t nxny = nx * ny;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * local_nz_with_ghosts;
    if (gid >= total) return;
    size_t z = gid / nxny;
    size_t rem = gid % nxny;
    size_t y = rem / nx;
    size_t x = rem % nx;

    // Only update real slices
    if (z == 0 || z == local_nz_with_ghosts - 1) return;

    auto local_idx = z * nxny + y * nx + x;

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zp = z + 1;
    size_t zn = z - 1;

    double center_mu = mu[local_idx];
    double m_xx = (mu[z * nxny + y * nx + xp] + mu[z * nxny + y * nx + xn] - 2.0 * center_mu);
    double m_yy = (mu[z * nxny + yp * nx + x] + mu[z * nxny + yn * nx + x] - 2.0 * center_mu);
    double m_zz = (mu[zp * nxny + y * nx + x] + mu[zn * nxny + y * nx + x] - 2.0 * center_mu);

    double lap_mu = m_xx + m_yy + m_zz;

    cnew[local_idx] = cold[local_idx] + dt * D * lap_mu;
}

// Initialize concentration field on host (consistent with original global indexing)
void initializeConcentrationHost(double* c, const size_t nx, const size_t ny, const size_t local_nz, const size_t global_z_offset) {
    const size_t vol_global = nx * ny * (local_nz + 0); // not used directly for modulo, keep consistent per-slab
    const size_t nxny = nx * ny;

    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t local_idx = z * nxny + y * nx + x;
                size_t global_z = global_z_offset + z;
                size_t linear_id = global_z * (nx * ny) + y * nx + x;
                // Use vol based on full global grid to keep sequence similar
                size_t vol = nx * ny * (local_nz + global_z_offset + 1); // simple variant to avoid zero
                if (vol == 0) vol = 1;
                double pseudo = ((((linear_id + 1) * 1299709) % (nx*ny*(local_nz + global_z_offset + 1))) / static_cast<double>(nx*ny*(local_nz + global_z_offset + 1)));
                c[local_idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResultHost(const double* c_host, const size_t nx, const size_t ny, const size_t local_nz, const size_t global_z_offset, const size_t nz_global) {
    const size_t nxny = nx * ny;
    size_t total = nxny * local_nz;
    double minVal = c_host[0];
    double maxVal = c_host[0];
    for (size_t i = 0; i < total; ++i) {
        double val = c_host[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Local concentration range (rank slice): [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints usage/messages)
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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    // Decompose Z dimension among MPI ranks
    size_t base = nz / nprocs;
    size_t rem = nz % nprocs;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    // offset of this rank in global z
    size_t z_offset = rank * base + std::min<size_t>(rank, rem);

    const size_t nxny = nx * ny;
    const size_t local_size = nxny * local_nz;
    // include 2 ghost layers for z
    const size_t local_nz_with_ghosts = local_nz + 2;
    const size_t local_with_ghosts_size = nxny * local_nz_with_ghosts;

    // Host buffers for halo exchange and optional validation
    std::vector<double> host_buffer_send_top(nxny);
    std::vector<double> host_buffer_send_bottom(nxny);
    std::vector<double> host_buffer_recv_top(nxny);
    std::vector<double> host_buffer_recv_bottom(nxny);

    // Host initial allocation for local real data (no ghosts)
    std::vector<double> host_local(local_size);

    // Initialize host local data with the same pseudo-random scheme using global coordinates
    initializeConcentrationHost(host_local.data(), nx, ny, local_nz, z_offset);

    // CUDA device arrays include ghost layers
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    cudaSetDevice(0);
    cudaMalloc(&d_cold, local_with_ghosts_size * sizeof(double));
    cudaMalloc(&d_cnew, local_with_ghosts_size * sizeof(double));
    cudaMalloc(&d_mu, local_with_ghosts_size * sizeof(double));

    // Initialize device: set ghosts to clamped boundary copies and copy real data into middle
    // Set whole device region to zero first
    cudaMemset(d_cold, 0, local_with_ghosts_size * sizeof(double));
    cudaMemset(d_cnew, 0, local_with_ghosts_size * sizeof(double));
    cudaMemset(d_mu, 0, local_with_ghosts_size * sizeof(double));

    // Copy real slab into device offset by one layer
    size_t slab_bytes = nxny * local_nz * sizeof(double);
    cudaMemcpy(d_cold + nxny, host_local.data(), slab_bytes, cudaMemcpyHostToDevice);

    // For clamped boundaries at global domain edges, set ghost layers equal to edge slices
    // Prepare ghost copies on host from host_local
    // bottom ghost (z=0) gets local z=0 copy if this rank is first else will be overwritten by MPI
    memcpy(host_buffer_recv_bottom.data(), host_local.data(), nxny * sizeof(double));
    // top ghost (last) gets local z=local_nz-1
    memcpy(host_buffer_recv_top.data(), &host_local[(local_nz-1)*nxny], nxny * sizeof(double));
    cudaMemcpy(d_cold, host_buffer_recv_bottom.data(), nxny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cold + (local_nz_with_ghosts-1)*nxny, host_buffer_recv_top.data(), nxny * sizeof(double), cudaMemcpyHostToDevice);

    // MPI neighbors
    int rank_above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

    // Timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Launch parameters
    size_t total_threads = local_with_ghosts_size;
    int block = 256;
    int grid = (int)((total_threads + block - 1) / block);

    for (int it = 0; it < iterations; ++it) {
        // Prepare send buffers from device edges
        // copy real top slice (local_nz-1) to host send_top
        cudaMemcpy(host_buffer_send_top.data(), d_cold + (1 + (local_nz - 1)) * nxny, nxny * sizeof(double), cudaMemcpyDeviceToHost);
        // copy real bottom slice (0) to host send_bottom
        cudaMemcpy(host_buffer_send_bottom.data(), d_cold + (1) * nxny, nxny * sizeof(double), cudaMemcpyDeviceToHost);

        // Exchange halos with neighbors using MPI
        MPI_Request reqs[4];
        MPI_Status stats[4];
        // Send top real slice to rank_above, receive into top ghost
        MPI_Isend(host_buffer_send_top.data(), nxny, MPI_DOUBLE, rank_above, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(host_buffer_recv_top.data(), nxny, MPI_DOUBLE, rank_above, 1, MPI_COMM_WORLD, &reqs[1]);
        // Send bottom real slice to rank_below, receive into bottom ghost
        MPI_Isend(host_buffer_send_bottom.data(), nxny, MPI_DOUBLE, rank_below, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(host_buffer_recv_bottom.data(), nxny, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, stats);

        // For boundaries where neighbor is MPI_PROC_NULL, keep clamped edge (already set)
        if (rank_above != MPI_PROC_NULL) {
            // copy received top ghost into device top ghost layer
            cudaMemcpy(d_cold + (local_nz_with_ghosts - 1) * nxny, host_buffer_recv_top.data(), nxny * sizeof(double), cudaMemcpyHostToDevice);
        }
        if (rank_below != MPI_PROC_NULL) {
            cudaMemcpy(d_cold, host_buffer_recv_bottom.data(), nxny * sizeof(double), cudaMemcpyHostToDevice);
        }

        // Compute chemical potential on device (skipping ghosts)
        computeMuKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz_with_ghosts, gamma, e_AA, e_BB, e_AB, z_offset, nz);
        cudaDeviceSynchronize();

        // Update concentration on device
        updateKernel<<<grid, block>>>(d_cold, d_mu, d_cnew, nx, ny, local_nz_with_ghosts, D, dt);
        cudaDeviceSynchronize();

        // swap device pointers
        std::swap(d_cold, d_cnew);

        // For safety, ensure clamped boundaries remain for ranks at domain edges
        if (rank_below == MPI_PROC_NULL) {
            // copy first real slice into bottom ghost
            cudaMemcpy(d_cold, d_cold + nxny, nxny * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        if (rank_above == MPI_PROC_NULL) {
            cudaMemcpy(d_cold + (local_nz_with_ghosts-1)*nxny, d_cold + (local_nz_with_ghosts-2)*nxny, nxny * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double local_duration = t_end - t_start;
    double max_duration = 0.0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", max_duration);
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / max_duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy final data back to host_local (real slices only)
    cudaMemcpy(host_local.data(), d_cold + nxny, slab_bytes, cudaMemcpyDeviceToHost);

    // Optionally print results from rank 0 by gathering, but to avoid extra memory just print local slices
    if (printResults && rank == 0) {
        print_results(std::vector<double>(host_local.begin(), host_local.end()), "Concentration_local_rank0");
    }

    // Validation: each rank validates its local slab
    bool local_valid = true;
    if (validate) {
        local_valid = validateResultHost(host_local.data(), nx, ny, local_nz, z_offset, nz);
    }
    int all_valid = 1;
    int local_valid_int = local_valid ? 1 : 0;
    MPI_Reduce(&local_valid_int, &all_valid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0 && validate) {
        if (all_valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);

    MPI_Finalize();
    return 0;
}
