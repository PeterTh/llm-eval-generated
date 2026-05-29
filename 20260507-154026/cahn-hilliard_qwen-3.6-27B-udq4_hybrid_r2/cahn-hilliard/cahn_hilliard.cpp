/**
 * Cahn-Hilliard Phase Separation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA parallelization:
 *   MPI   – domain decomposition along Z across ranks (halo exchange)
 *   CUDA  – all compute kernels (init, chemical potential, update)
 *   OpenMP – parallel validation on host
 */

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

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = call;                                               \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device: 3D index helper
// ---------------------------------------------------------------------------
__device__ inline size_t idx3d(size_t x, size_t y, size_t z,
                               size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernel – initialise concentration field
// ---------------------------------------------------------------------------
__global__ void initKernel(double* __restrict__ c,
                           size_t nx, size_t ny, size_t local_nz,
                           size_t vol, size_t z_start) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total  = nx * ny * local_nz;

    for (size_t i = tid; i < total; i += stride) {
        size_t x        = i % nx;
        size_t y        = (i / nx) % ny;
        size_t z_local  = i / (nx * ny) + 1;           // +1 ghost offset
        size_t z_global = z_start + i / (nx * ny);

        size_t linear_id = z_global * (nx * ny) + y * nx + x;
        double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) /
                         static_cast<double>(vol));
        c[idx3d(x, y, z_local, nx, ny)] = -1.0 + 2.0 * pseudo;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel – chemical potential (includes Laplacian of c)
// ---------------------------------------------------------------------------
__global__ void chemPotKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz,
    double dx, double dy, double dz,
    double gamma, double e_AA, double e_BB, double e_AB) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total  = nx * ny * local_nz;
    size_t nz_total = local_nz + 2;

    for (size_t i = tid; i < total; i += stride) {
        size_t x = i % nx;
        size_t y = (i / nx) % ny;
        size_t z = i / (nx * ny) + 1;

        size_t idx = idx3d(x, y, z, nx, ny);
        double cv  = __ldg(&c[idx]);

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t zp = (z < nz_total - 1) ? z + 1 : z;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yn = (y > 0) ? y - 1 : 0;
        size_t zn = (z > 0) ? z - 1 : 0;

        double cxx = (__ldg(&c[idx3d(xp, y, z, nx, ny)]) +
                      __ldg(&c[idx3d(xn, y, z, nx, ny)]) - 2.0 * cv) /
                     (dx * dx);
        double cyy = (__ldg(&c[idx3d(x, yp, z, nx, ny)]) +
                      __ldg(&c[idx3d(x, yn, z, nx, ny)]) - 2.0 * cv) /
                     (dy * dy);
        double czz = (__ldg(&c[idx3d(x, y, zp, nx, ny)]) +
                      __ldg(&c[idx3d(x, y, zn, nx, ny)]) - 2.0 * cv) /
                     (dz * dz);

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                         2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * (cxx + cyy + czz);
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel – concentration update (Laplacian of mu)
// ---------------------------------------------------------------------------
__global__ void updateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold, const double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz,
    double D, double dt, double dx, double dy, double dz) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t total  = nx * ny * local_nz;
    size_t nz_total = local_nz + 2;

    for (size_t i = tid; i < total; i += stride) {
        size_t x = i % nx;
        size_t y = (i / nx) % ny;
        size_t z = i / (nx * ny) + 1;

        size_t idx = idx3d(x, y, z, nx, ny);
        double muv = __ldg(&mu[idx]);

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t zp = (z < nz_total - 1) ? z + 1 : z;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yn = (y > 0) ? y - 1 : 0;
        size_t zn = (z > 0) ? z - 1 : 0;

        double mu_xx = (__ldg(&mu[idx3d(xp, y, z, nx, ny)]) +
                        __ldg(&mu[idx3d(xn, y, z, nx, ny)]) - 2.0 * muv) /
                       (dx * dx);
        double mu_yy = (__ldg(&mu[idx3d(x, yp, z, nx, ny)]) +
                        __ldg(&mu[idx3d(x, yn, z, nx, ny)]) - 2.0 * muv) /
                       (dy * dy);
        double mu_zz = (__ldg(&mu[idx3d(x, y, zp, nx, ny)]) +
                        __ldg(&mu[idx3d(x, y, zn, nx, ny)]) - 2.0 * muv) /
                       (dz * dz);

        cnew[idx] = __ldg(&cold[idx]) + dt * D * (mu_xx + mu_yy + mu_zz);
    }
}

// ---------------------------------------------------------------------------
// Host: halo exchange via pinned-memory staging
// ---------------------------------------------------------------------------
static void exchangeHalos(double* d_data,
                          size_t nx, size_t ny, size_t local_nz,
                          int rank, int n_ranks, MPI_Comm comm,
                          double* h_top_send, double* h_bottom_send,
                          double* h_top_recv, double* h_bottom_recv) {
    size_t row_size = nx * ny;

    // GPU -> host (boundary rows)
    CUDA_CHECK(cudaMemcpy(h_top_send,
                          &d_data[local_nz * row_size],
                          row_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bottom_send,
                          &d_data[row_size],
                          row_size * sizeof(double),
                          cudaMemcpyDeviceToHost));

    MPI_Request reqs[4];
    int nreq = 0;

    // Send last owned row (top) to rank+1
    if (rank < n_ranks - 1) {
        MPI_Isend(h_top_send, row_size, MPI_DOUBLE,
                  rank + 1, 0, comm, &reqs[nreq++]);
    }
    // Send first owned row (bottom) to rank-1
    if (rank > 0) {
        MPI_Isend(h_bottom_send, row_size, MPI_DOUBLE,
                  rank - 1, 1, comm, &reqs[nreq++]);
    }

    // Receive from rank-1 into bottom ghost
    if (rank > 0) {
        MPI_Irecv(h_bottom_recv, row_size, MPI_DOUBLE,
                  rank - 1, 0, comm, &reqs[nreq++]);
    }
    // Receive from rank+1 into top ghost
    if (rank < n_ranks - 1) {
        MPI_Irecv(h_top_recv, row_size, MPI_DOUBLE,
                  rank + 1, 1, comm, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Host -> GPU (ghost cells)
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(&d_data[0], h_bottom_recv,
                              row_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    } else {
        // Rank 0 – clamped BC
        CUDA_CHECK(cudaMemcpy(&d_data[0], &d_data[row_size],
                              row_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    if (rank < n_ranks - 1) {
        CUDA_CHECK(cudaMemcpy(&d_data[(local_nz + 1) * row_size], h_top_recv,
                              row_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    } else {
        // Last rank – clamped BC
        CUDA_CHECK(cudaMemcpy(&d_data[(local_nz + 1) * row_size],
                              &d_data[local_nz * row_size],
                              row_size * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    int validate_flag = 0, printResults_flag = 0;

    // ---- parse arguments on rank 0 ----
    if (rank == 0) {
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
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
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

    // broadcast to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool validate     = validate_flag != 0;
    bool printResults = printResults_flag != 0;

    // ---- domain decomposition along Z ----
    size_t base_nz   = nz / n_ranks;
    size_t remainder = nz % n_ranks;
    size_t z_start   = rank * base_nz + std::min(static_cast<size_t>(rank), remainder);
    size_t z_end     = (rank + 1) * base_nz + std::min(static_cast<size_t>(rank + 1), remainder);
    size_t local_nz  = z_end - z_start;
    size_t nz_total  = local_nz + 2;            // +2 ghost cells
    size_t row_size  = nx * ny;
    size_t local_size = nz_total * row_size;
    size_t gridSize  = nx * ny * nz;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", n_ranks);
        printf("CUDA + OpenMP hybrid parallelization\n");
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;

    // ---- pinned host memory for halo exchange ----
    double* h_top_send    = nullptr;
    double* h_bottom_send = nullptr;
    double* h_top_recv    = nullptr;
    double* h_bottom_recv = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_top_send,    row_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_bottom_send, row_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_top_recv,    row_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_bottom_recv, row_size * sizeof(double)));

    // ---- device memory (with ghost cells) ----
    double* d_cold  = nullptr;
    double* d_cnew  = nullptr;
    double* d_mu    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   local_size * sizeof(double)));

    // Launch configuration
    const int blockSize = 256;
    size_t totalThreads = nx * ny * local_nz;
    int numBlocks = static_cast<int>(
        std::min((totalThreads + blockSize - 1) / blockSize, (size_t)65536));

    // ---- initialise ----
    if (rank == 0) printf("Initializing concentration field...\n");
    if (numBlocks > 0) {
        initKernel<<<numBlocks, blockSize>>>(
            d_cold, nx, ny, local_nz, gridSize, z_start);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- simulation loop ----
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // 1. Exchange halos for cold
        exchangeHalos(d_cold, nx, ny, local_nz, rank, n_ranks, MPI_COMM_WORLD,
                      h_top_send, h_bottom_send, h_top_recv, h_bottom_recv);

        // 2. Compute chemical potential on GPU
        if (numBlocks > 0) {
            chemPotKernel<<<numBlocks, blockSize>>>(
                d_cold, d_mu, nx, ny, local_nz,
                dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Exchange halos for mu
        exchangeHalos(d_mu, nx, ny, local_nz, rank, n_ranks, MPI_COMM_WORLD,
                      h_top_send, h_bottom_send, h_top_recv, h_bottom_recv);

        // 4. Update concentration on GPU
        if (numBlocks > 0) {
            updateKernel<<<numBlocks, blockSize>>>(
                d_cnew, d_cold, d_mu, nx, ny, local_nz,
                D, dt, dx, dy, dz);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. Swap buffers
        std::swap(d_cold, d_cnew);
    }

    auto end     = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- copy owned cells GPU -> host ----
    std::vector<double> local_data(local_nz * row_size);
    if (local_nz > 0) {
        CUDA_CHECK(cudaMemcpy(local_data.data(), &d_cold[row_size],
                              local_nz * row_size * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // ---- gather to rank 0 ----
    std::vector<int> recv_counts(n_ranks);
    std::vector<int> displacements(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        size_t rs = r * base_nz + std::min(static_cast<size_t>(r), remainder);
        size_t re = (r + 1) * base_nz + std::min(static_cast<size_t>(r + 1), remainder);
        recv_counts[r]   = static_cast<int>((re - rs) * row_size);
        displacements[r] = static_cast<int>(rs * row_size);
    }

    std::vector<double> full_grid;
    if (rank == 0) full_grid.resize(gridSize);

    MPI_Gatherv(local_data.data(), static_cast<int>(local_nz * row_size),
                MPI_DOUBLE,
                rank == 0 ? full_grid.data() : nullptr,
                recv_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- post-processing on rank 0 ----
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(full_grid, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");

            bool   valid  = true;
            double minVal = full_grid[0];
            double maxVal = full_grid[0];

#pragma omp parallel reduction(|| : valid) reduction(min : minVal) reduction(max : maxVal)
            {
#pragma omp for nowait
                for (size_t i = 0; i < full_grid.size(); ++i) {
                    if (std::isnan(full_grid[i]) || std::isinf(full_grid[i])) {
                        valid = false;
                    }
                    minVal = std::min(minVal, full_grid[i]);
                    maxVal = std::max(maxVal, full_grid[i]);
                }
            }

            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 10.0 || minVal < -10.0) {
                valid = false;
            }

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // ---- cleanup ----
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_top_send));
    CUDA_CHECK(cudaFreeHost(h_bottom_send));
    CUDA_CHECK(cudaFreeHost(h_top_recv));
    CUDA_CHECK(cudaFreeHost(h_bottom_recv));

    MPI_Finalize();
    return 0;
}
