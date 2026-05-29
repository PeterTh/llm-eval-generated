#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================
// CUDA Kernels
// ============================================================

__global__ void initConcentrationKernel(double* __restrict__ c,
    const size_t nx, const size_t ny, const size_t nz_local,
    const size_t z_start, const size_t vol) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz_local;

    if (tid < total) {
        const size_t x = tid % nx;
        const size_t y = (tid / nx) % ny;
        const size_t z = tid / (nx * ny);  // 0 .. nz_local-1

        const size_t z_global = z_start + z;
        // +1 offset: local array has a ghost cell at z=0
        const size_t idx = (z + 1) * (nx * ny) + y * nx + x;
        const size_t linear_id = z_global * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                               static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz_local,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB,
    const double e_AB) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz_local;

    if (tid < total) {
        const size_t x = tid % nx;
        const size_t y = (tid / nx) % ny;
        const size_t z = tid / (nx * ny);  // 0 .. nz_local-1

        // z_local = z + 1  (1 .. nz_local); ghost at 0 and nz_local+1
        const size_t z_base  = (z + 1) * (nx * ny);
        const size_t z_base_p = (z + 2) * (nx * ny);  // z_local + 1
        const size_t z_base_n = z * (nx * ny);        // z_local - 1

        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;

        const size_t idx = z_base + y * nx + x;
        const double cv = c[idx];

        const double cxx = (c[z_base  + y * nx + xp] +
                            c[z_base  + y * nx + xn] - 2.0 * cv) / (dx * dx);
        const double cyy = (c[z_base  + yp * nx + x] +
                            c[z_base  + yn * nx + x] - 2.0 * cv) / (dy * dy);
        const double czz = (c[z_base_p + y * nx + x] +
                            c[z_base_n + y * nx + x] - 2.0 * cv) / (dz * dz);

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                          2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * (cxx + cyy + czz);
    }
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t nz_local,
    const double D, const double dt,
    const double dx, const double dy, const double dz) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz_local;

    if (tid < total) {
        const size_t x = tid % nx;
        const size_t y = (tid / nx) % ny;
        const size_t z = tid / (nx * ny);  // 0 .. nz_local-1

        const size_t z_base  = (z + 1) * (nx * ny);
        const size_t z_base_p = (z + 2) * (nx * ny);
        const size_t z_base_n = z * (nx * ny);

        const size_t xp = (x < nx - 1) ? x + 1 : x;
        const size_t xn = (x > 0) ? x - 1 : 0;
        const size_t yp = (y < ny - 1) ? y + 1 : y;
        const size_t yn = (y > 0) ? y - 1 : 0;

        const size_t idx = z_base + y * nx + x;
        const double mu_val = mu[idx];

        const double mu_xx = (mu[z_base  + y * nx + xp] +
                              mu[z_base  + y * nx + xn] - 2.0 * mu_val) /
                             (dx * dx);
        const double mu_yy = (mu[z_base  + yp * nx + x] +
                              mu[z_base  + yn * nx + x] - 2.0 * mu_val) /
                             (dy * dy);
        const double mu_zz = (mu[z_base_p + y * nx + x] +
                              mu[z_base_n + y * nx + x] - 2.0 * mu_val) /
                             (dz * dz);

        cnew[idx] = cold[idx] + dt * D * (mu_xx + mu_yy + mu_zz);
    }
}

// ============================================================
// Halo Exchange  (device  ->  pinned host  ->  MPI  ->  device)
// ============================================================

static void exchangeHalos(double* d_data, const size_t nx, const size_t ny,
                          const size_t nz_local,
                          double* h_send_bottom, double* h_send_top,
                          double* h_recv_bottom, double* h_recv_top,
                          const int rank, const int num_ranks) {
    const size_t layer = nx * ny;

    // Copy real boundary layers to pinned host
    cudaMemcpy(h_send_bottom, d_data + layer, layer * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_send_top, d_data + nz_local * layer, layer * sizeof(double),
               cudaMemcpyDeviceToHost);

    // --- bottom ghost: receive top-real from rank-1, send bottom-real to rank-1 ---
    if (rank > 0) {
        MPI_Sendrecv(h_send_bottom, static_cast<int>(layer), MPI_DOUBLE,
                     rank - 1, 0,
                     h_recv_bottom, static_cast<int>(layer), MPI_DOUBLE,
                     rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // --- top ghost: receive bottom-real from rank+1, send top-real to rank+1 ---
    if (rank < num_ranks - 1) {
        MPI_Sendrecv(h_send_top, static_cast<int>(layer), MPI_DOUBLE,
                     rank + 1, 0,
                     h_recv_top, static_cast<int>(layer), MPI_DOUBLE,
                     rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Write bottom ghost (z_local = 0)
    if (rank > 0) {
        cudaMemcpy(d_data, h_recv_bottom, layer * sizeof(double),
                   cudaMemcpyHostToDevice);
    } else {
        // Clamped BC: ghost mirrors the boundary real cell
        cudaMemcpy(d_data, d_data + layer, layer * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    // Write top ghost (z_local = nz_local + 1)
    if (rank < num_ranks - 1) {
        cudaMemcpy(d_data + (nz_local + 1) * layer, h_recv_top,
                   layer * sizeof(double), cudaMemcpyHostToDevice);
    } else {
        // Clamped BC: ghost mirrors the boundary real cell
        cudaMemcpy(d_data + (nz_local + 1) * layer,
                   d_data + nz_local * layer,
                   layer * sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

// ============================================================
// Usage
// ============================================================

static void printUsage(const char* progName) {
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

// ============================================================
// Main  –  hybrid MPI + CUDA + OpenMP
// ============================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // ---- parse arguments on rank 0, then broadcast ----
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    int validate_flag = 0, printResults_flag = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)
                nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)
                ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)
                nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
                iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate_flag = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults_flag = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate     = validate_flag     != 0;
    const bool printResults = printResults_flag != 0;

    // ---- 1-D domain decomposition along Z ----
    const size_t nz_base     = nz / num_ranks;
    const size_t nz_remainder = nz % num_ranks;
    const size_t nz_local    = nz_base + (rank < static_cast<int>(nz_remainder) ? 1 : 0);
    const size_t z_start     = nz_base * static_cast<size_t>(rank) +
                               std::min(static_cast<size_t>(rank), nz_remainder);

    const size_t layer   = nx * ny;
    const size_t local_sz = layer * (nz_local + 2);  // +2 ghost cells
    const size_t global_sz = nx * ny * nz;

    // ---- device memory ----
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    if (local_sz > 0) {
        cudaMalloc(&d_cold, local_sz * sizeof(double));
        cudaMalloc(&d_cnew, local_sz * sizeof(double));
        cudaMalloc(&d_mu,  local_sz * sizeof(double));
    }

    // ---- pinned host buffers for halo exchange ----
    double *h_sbot = nullptr, *h_stop = nullptr;
    double *h_rbot = nullptr, *h_rtop = nullptr;
    if (nz_local > 0) {
        cudaMallocHost(&h_sbot, layer * sizeof(double));
        cudaMallocHost(&h_stop, layer * sizeof(double));
        cudaMallocHost(&h_rbot, layer * sizeof(double));
        cudaMallocHost(&h_rtop, layer * sizeof(double));
    }

    // ---- info ----
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    printf("Initializing concentration field...\n");

    // ---- init on GPU ----
    if (nz_local > 0) {
        const int blk = 256;
        const int grd = static_cast<int>((nx * ny * nz_local + blk - 1) / blk);
        initConcentrationKernel<<<grd, blk>>>(d_cold, nx, ny, nz_local,
                                              z_start, global_sz);
        cudaDeviceSynchronize();
    }

    // ---- physical parameters ----
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int blk = 256;

    for (int t = 0; t < iterations; ++t) {
        if (nz_local > 0) {
            const int grd = static_cast<int>((nx * ny * nz_local + blk - 1) / blk);

            // 1. exchange cold halos
            exchangeHalos(d_cold, nx, ny, nz_local,
                          h_sbot, h_stop, h_rbot, h_rtop,
                          rank, num_ranks);

            // 2. compute chemical potential on GPU
            computeChemicalPotentialKernel<<<grd, blk>>>(
                d_cold, d_mu, nx, ny, nz_local,
                dx, dy, dz, gamma, e_AA, e_BB, e_AB);

            // 3. exchange mu halos
            exchangeHalos(d_mu, nx, ny, nz_local,
                          h_sbot, h_stop, h_rbot, h_rtop,
                          rank, num_ranks);

            // 4. update concentration on GPU
            cahnHilliardUpdateKernel<<<grd, blk>>>(
                d_cnew, d_cold, d_mu, nx, ny, nz_local,
                D, dt, dx, dy, dz);

            // 5. swap buffers
            std::swap(d_cold, d_cnew);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- copy result to host (real cells only) ----
    std::vector<double> h_local;
    if (nz_local > 0) {
        h_local.resize(nz_local * layer);
        cudaMemcpy(h_local.data(), d_cold + layer,
                   nz_local * layer * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // ---- gather to rank 0 ----
    std::vector<int> recv_counts(num_ranks);
    std::vector<int> displs(num_ranks);
    {
        size_t off = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const size_t nz_loc = nz_base + (r < static_cast<int>(nz_remainder) ? 1 : 0);
            recv_counts[r] = static_cast<int>(nz_loc * layer);
            displs[r]      = static_cast<int>(off);
            off += nz_loc * layer;
        }
    }

    std::vector<double> h_full;
    if (rank == 0) h_full.resize(global_sz);

    double dummy = 0.0;
    MPI_Gatherv(nz_local > 0 ? h_local.data() : &dummy,
                static_cast<int>(nz_local * layer), MPI_DOUBLE,
                h_full.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- timing & performance (rank 0) ----
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = static_cast<double>(global_sz) * iterations;
        double mcups = 0.0;
        if (duration.count() > 0)
            mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults)
            print_results(h_full, "Concentration");

        // ---- OpenMP-parallel validation ----
        if (validate) {
            printf("Validating result...\n");

            bool has_nan_inf = false;
            double minVal = h_full[0], maxVal = h_full[0];

            #pragma omp parallel
            {
                bool local_nan = false;
                double local_min = h_full[0], local_max = h_full[0];

                #pragma omp for
                for (size_t i = 0; i < global_sz; ++i) {
                    if (std::isnan(h_full[i]) || std::isinf(h_full[i]))
                        local_nan = true;
                    local_min = std::min(local_min, h_full[i]);
                    local_max = std::max(local_max, h_full[i]);
                }

                #pragma omp critical
                {
                    has_nan_inf = has_nan_inf || local_nan;
                    minVal = std::min(minVal, local_min);
                    maxVal = std::max(maxVal, local_max);
                }
            }

            if (has_nan_inf)
                printf("Validation failed: found NaN or Inf value\n");

            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 10.0 || minVal < -10.0)
                printf("Validation failed: values out of expected range\n");

            if (!has_nan_inf && maxVal <= 10.0 && minVal >= -10.0)
                printf("Validation: PASSED\n");
            else
                printf("Validation: FAILED\n");
        }
    }

    // ---- broadcast return code ----
    int rc = 0;
    if (rank == 0 && validate) {
        // re-derive from h_full for the broadcast
        bool has_nan_inf = false;
        double minVal = h_full[0], maxVal = h_full[0];
        #pragma omp parallel for reduction(||:has_nan_inf)
        for (size_t i = 0; i < global_sz; ++i) {
            if (std::isnan(h_full[i]) || std::isinf(h_full[i]))
                has_nan_inf = true;
        }
        #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
        for (size_t i = 0; i < global_sz; ++i) {
            minVal = std::min(minVal, h_full[i]);
            maxVal = std::max(maxVal, h_full[i]);
        }
        if (has_nan_inf || maxVal > 10.0 || minVal < -10.0)
            rc = 1;
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- cleanup ----
    if (d_cold)  cudaFree(d_cold);
    if (d_cnew)  cudaFree(d_cnew);
    if (d_mu)    cudaFree(d_mu);
    if (h_sbot)  cudaFreeHost(h_sbot);
    if (h_stop)  cudaFreeHost(h_stop);
    if (h_rbot)  cudaFreeHost(h_rbot);
    if (h_rtop)  cudaFreeHost(h_rtop);

    MPI_Finalize();
    return rc;
}
