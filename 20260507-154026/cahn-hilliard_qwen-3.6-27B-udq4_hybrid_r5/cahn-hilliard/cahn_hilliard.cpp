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

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define MPI_CHECK(call)                                                          \
    do {                                                                         \
        int err = call;                                                          \
        if (err != MPI_SUCCESS) {                                                \
            char buf[MPI_MAX_ERROR_STRING];                                      \
            int len;                                                             \
            MPI_Error_string(err, buf, &len);                                    \
            fprintf(stderr, "MPI error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, buf);                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ==================== CUDA Kernels ====================

// Initialize concentration field on device
__global__ void initKernel(
    double* __restrict__ c,
    size_t nx, size_t ny, size_t nz_interior,
    size_t global_nz,
    size_t z_offset)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz_interior) {
        size_t global_z = z_offset + z;
        size_t linear_id = global_z * nx * ny + y * nx + x;
        size_t vol = nx * ny * global_nz;
        double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) / static_cast<double>(vol));
        c[(z + 1) * nx * ny + y * nx + x] = -1.0 + 2.0 * pseudo;
    }
}

// Compute chemical potential with 2D shared-memory tiling over x-y plane
__global__ void computeMuKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    constexpr int TILE = 16;
    __shared__ double sdata[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx_base = static_cast<int>(blockIdx.x) * TILE;
    int gy_base = static_cast<int>(blockIdx.y) * TILE;

    for (size_t gz = static_cast<size_t>(blockIdx.z); gz < nz_local; gz += gridDim.z) {
        // Load tile + 1 halo in x-y into shared memory
        int gx = gx_base + tx - 1;
        int gy = gy_base + ty - 1;

        int cx = (gx < 0) ? 0 : (gx < static_cast<int>(nx)) ? gx : static_cast<int>(nx) - 1;
        int cy = (gy < 0) ? 0 : (gy < static_cast<int>(ny)) ? gy : static_cast<int>(ny) - 1;
        sdata[ty][tx] = c[gz * nx * ny + static_cast<size_t>(cy) * nx + static_cast<size_t>(cx)];

        __syncthreads();

        // Compute for interior threads of the tile
        if (tx >= 1 && tx < TILE + 1 && ty >= 1 && ty < TILE + 1) {
            int igx = gx_base + tx - 1;
            int igy = gy_base + ty - 1;

            if (igx < static_cast<int>(nx) && igy < static_cast<int>(ny)) {
                double center = sdata[ty][tx];
                double xp_val = sdata[ty][tx + 1];
                double xn_val = sdata[ty][tx - 1];
                double yp_val = sdata[ty + 1][tx];
                double yn_val = sdata[ty - 1][tx];

                // z-neighbours from global memory (L2 cache provides reuse)
                int zp = (static_cast<int>(gz) + 1 < static_cast<int>(nz_local))
                         ? static_cast<int>(gz) + 1 : static_cast<int>(nz_local) - 1;
                int zn = (static_cast<int>(gz) > 0)
                         ? static_cast<int>(gz) - 1 : 0;
                double zp_val = c[static_cast<size_t>(zp) * nx * ny
                                  + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)];
                double zn_val = c[static_cast<size_t>(zn) * nx * ny
                                  + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)];

                double lap = ((xp_val + xn_val - 2.0 * center) * inv_dx2
                              + (yp_val + yn_val - 2.0 * center) * inv_dy2
                              + (zp_val + zn_val - 2.0 * center) * inv_dz2);

                double cv = center;
                mu[static_cast<size_t>(gz) * nx * ny
                   + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)]
                    = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv
                    - gamma * lap;
            }
        }

        __syncthreads();
    }
}

// Update concentration with 2D shared-memory tiling over x-y plane
__global__ void updateKernel(
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    double* __restrict__ cnew,
    size_t nx, size_t ny, size_t nz_local,
    double D, double dt, double inv_dx2, double inv_dy2, double inv_dz2)
{
    constexpr int TILE = 16;
    __shared__ double sdata[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx_base = static_cast<int>(blockIdx.x) * TILE;
    int gy_base = static_cast<int>(blockIdx.y) * TILE;

    for (size_t gz = static_cast<size_t>(blockIdx.z); gz < nz_local; gz += gridDim.z) {
        int gx = gx_base + tx - 1;
        int gy = gy_base + ty - 1;

        int cx = (gx < 0) ? 0 : (gx < static_cast<int>(nx)) ? gx : static_cast<int>(nx) - 1;
        int cy = (gy < 0) ? 0 : (gy < static_cast<int>(ny)) ? gy : static_cast<int>(ny) - 1;
        sdata[ty][tx] = mu[static_cast<size_t>(gz) * nx * ny
                            + static_cast<size_t>(cy) * nx + static_cast<size_t>(cx)];

        __syncthreads();

        if (tx >= 1 && tx < TILE + 1 && ty >= 1 && ty < TILE + 1) {
            int igx = gx_base + tx - 1;
            int igy = gy_base + ty - 1;

            if (igx < static_cast<int>(nx) && igy < static_cast<int>(ny)) {
                double center = sdata[ty][tx];
                double xp_val = sdata[ty][tx + 1];
                double xn_val = sdata[ty][tx - 1];
                double yp_val = sdata[ty + 1][tx];
                double yn_val = sdata[ty - 1][tx];

                int zp = (static_cast<int>(gz) + 1 < static_cast<int>(nz_local))
                         ? static_cast<int>(gz) + 1 : static_cast<int>(nz_local) - 1;
                int zn = (static_cast<int>(gz) > 0)
                         ? static_cast<int>(gz) - 1 : 0;
                double zp_val = mu[static_cast<size_t>(zp) * nx * ny
                                    + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)];
                double zn_val = mu[static_cast<size_t>(zn) * nx * ny
                                    + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)];

                double lap = ((xp_val + xn_val - 2.0 * center) * inv_dx2
                              + (yp_val + yn_val - 2.0 * center) * inv_dy2
                              + (zp_val + zn_val - 2.0 * center) * inv_dz2);

                cnew[static_cast<size_t>(gz) * nx * ny
                     + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)]
                    = cold[static_cast<size_t>(gz) * nx * ny
                           + static_cast<size_t>(igy) * nx + static_cast<size_t>(igx)]
                    + dt * D * lap;
            }
        }

        __syncthreads();
    }
}

// Clamp ghost-bottom layer: copy interior-bottom to ghost-bottom (global z = 0 boundary)
__global__ void clampGhostBottomKernel(
    double* __restrict__ c,
    size_t nx, size_t ny)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny) {
        c[y * nx + x] = c[nx * ny + y * nx + x];
    }
}

// Clamp ghost-top layer: copy interior-top to ghost-top (global z = nz-1 boundary)
__global__ void clampGhostTopKernel(
    double* __restrict__ c,
    size_t nx, size_t ny, size_t nz_interior)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny) {
        c[(nz_interior + 1) * nx * ny + y * nx + x]
            = c[nz_interior * nx * ny + y * nx + x];
    }
}

// ==================== Host Functions ====================

static void exchangeGhostLayers(
    const double* h_send_bottom,
    const double* h_send_top,
    double* h_recv_bottom,
    double* h_recv_top,
    size_t layer_size,
    int rank, int num_ranks)
{
    if (layer_size == 0) return;

    // Ghost-bottom exchange: send interior-top to r+1, receive from r-1
    if (rank > 0) {
        if (rank < num_ranks - 1) {
            MPI_CHECK(MPI_Sendrecv(
                const_cast<double*>(h_send_top), static_cast<int>(layer_size), MPI_DOUBLE,
                rank + 1, 0,
                h_recv_bottom, static_cast<int>(layer_size), MPI_DOUBLE,
                rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        } else {
            MPI_CHECK(MPI_Recv(
                h_recv_bottom, static_cast<int>(layer_size), MPI_DOUBLE,
                rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    } else if (rank < num_ranks - 1) {
        MPI_CHECK(MPI_Send(
            const_cast<double*>(h_send_top), static_cast<int>(layer_size), MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD));
    }

    // Ghost-top exchange: send interior-bottom to r-1, receive from r+1
    if (rank < num_ranks - 1) {
        if (rank > 0) {
            MPI_CHECK(MPI_Sendrecv(
                const_cast<double*>(h_send_bottom), static_cast<int>(layer_size), MPI_DOUBLE,
                rank - 1, 1,
                h_recv_top, static_cast<int>(layer_size), MPI_DOUBLE,
                rank + 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        } else {
            MPI_CHECK(MPI_Recv(
                h_recv_top, static_cast<int>(layer_size), MPI_DOUBLE,
                rank + 1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
    } else if (rank > 0) {
        MPI_CHECK(MPI_Send(
            const_cast<double*>(h_send_bottom), static_cast<int>(layer_size), MPI_DOUBLE,
            rank - 1, 1, MPI_COMM_WORLD));
    }
}

// ==================== Main ====================

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, num_ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    // ---- Parse arguments (rank 0) then broadcast ----
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    int validate_flag = 0;
    int printResults = 0;

    if (rank == 0) {
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
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -x <num>     Grid size in X dimension (default: 64)\n");
                printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
                printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
                printf("  -i <num>     Number of time steps (default: 20)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_CHECK(MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate_flag ? "enabled" : "disabled");
    }

    // ---- Physical parameters ----
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);

    // ---- Domain decomposition along Z ----
    size_t nz_per_rank = nz / num_ranks;
    size_t remainder   = nz % num_ranks;
    size_t z_start = rank * nz_per_rank + std::min(static_cast<size_t>(rank), remainder);
    size_t z_end   = (rank + 1) * nz_per_rank
                   + std::min(static_cast<size_t>(rank + 1), remainder);
    size_t nz_interior = z_end - z_start;
    size_t nz_local    = nz_interior + 2;          // ghost-bottom + interior + ghost-top
    size_t layer_size  = nx * ny;
    size_t local_size  = nz_local * layer_size;

    // ---- Device memory ----
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    if (nz_interior > 0) {
        CUDA_CHECK(cudaMalloc(&d_cold, local_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, local_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu,  local_size * sizeof(double)));
    }

    // ---- Pinned host buffers for halo exchange ----
    double *h_send_bottom = nullptr, *h_send_top = nullptr;
    double *h_recv_bottom = nullptr, *h_recv_top = nullptr;
    if (nz_interior > 0 && layer_size > 0) {
        CUDA_CHECK(cudaMallocHost(&h_send_bottom, layer_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_send_top,    layer_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_bottom, layer_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_top,    layer_size * sizeof(double)));
    }

    // ---- Initialise ----
    if (rank == 0) printf("Initializing concentration field...\n");

    if (nz_interior > 0) {
        // Interior cells
        {
            dim3 blk(8, 8, 4);
            dim3 grd((nx + blk.x - 1) / blk.x,
                     (ny + blk.y - 1) / blk.y,
                     (nz_interior + blk.z - 1) / blk.z);
            initKernel<<<grd, blk>>>(d_cold, nx, ny, nz_interior, nz, z_start);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // First halo exchange
        {
            CUDA_CHECK(cudaMemcpy(h_send_bottom, d_cold + layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_send_top, d_cold + nz_interior * layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            exchangeGhostLayers(h_send_bottom, h_send_top,
                                h_recv_bottom, h_recv_top,
                                layer_size, rank, num_ranks);
            CUDA_CHECK(cudaMemcpy(d_cold, h_recv_bottom,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cold + (nz_interior + 1) * layer_size, h_recv_top,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        // Clamp global-boundary ghost layers
        if (rank == 0) {
            dim3 blk(16, 16);
            dim3 grd((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);
            clampGhostBottomKernel<<<grd, blk>>>(d_cold, nx, ny);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (rank == num_ranks - 1) {
            dim3 blk(16, 16);
            dim3 grd((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);
            clampGhostTopKernel<<<grd, blk>>>(d_cold, nx, ny, nz_interior);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // ---- Simulation loop ----
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    auto start = std::chrono::high_resolution_clock::now();

    if (nz_interior > 0) {
        dim3 sm_blk(18, 18);
        size_t grid_z = std::max(static_cast<size_t>(1),
                                 std::min(nz_local, static_cast<size_t>(32)));
        dim3 sm_grd((nx + 16 - 1) / 16,
                    (ny + 16 - 1) / 16,
                    grid_z);

        dim3 cl_blk(16, 16);
        dim3 cl_grd((nx + cl_blk.x - 1) / cl_blk.x,
                    (ny + cl_blk.y - 1) / cl_blk.y);

        for (int t = 0; t < iterations; ++t) {
            // 1) Host-to-host halo exchange
            CUDA_CHECK(cudaMemcpy(h_send_bottom, d_cold + layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_send_top, d_cold + nz_interior * layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            exchangeGhostLayers(h_send_bottom, h_send_top,
                                h_recv_bottom, h_recv_top,
                                layer_size, rank, num_ranks);
            CUDA_CHECK(cudaMemcpy(d_cold, h_recv_bottom,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cold + (nz_interior + 1) * layer_size, h_recv_top,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));

            // 2) Clamp global-boundary ghosts
            if (rank == 0)
                clampGhostBottomKernel<<<cl_grd, cl_blk>>>(d_cold, nx, ny);
            if (rank == num_ranks - 1)
                clampGhostTopKernel<<<cl_grd, cl_blk>>>(d_cold, nx, ny, nz_interior);

            // 3) Chemical potential
            computeMuKernel<<<sm_grd, sm_blk>>>(
                d_cold, d_mu, nx, ny, nz_local,
                inv_dx2, inv_dy2, inv_dz2,
                gamma, e_AA, e_BB, e_AB);

            // 3.5) Exchange mu ghost layers for update kernel
            CUDA_CHECK(cudaMemcpy(h_send_bottom, d_mu + layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_send_top, d_mu + nz_interior * layer_size,
                                  layer_size * sizeof(double), cudaMemcpyDeviceToHost));
            exchangeGhostLayers(h_send_bottom, h_send_top,
                                h_recv_bottom, h_recv_top,
                                layer_size, rank, num_ranks);
            CUDA_CHECK(cudaMemcpy(d_mu, h_recv_bottom,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mu + (nz_interior + 1) * layer_size, h_recv_top,
                                  layer_size * sizeof(double), cudaMemcpyHostToDevice));

            // Clamp mu ghosts at global boundaries
            if (rank == 0)
                clampGhostBottomKernel<<<cl_grd, cl_blk>>>(d_mu, nx, ny);
            if (rank == num_ranks - 1)
                clampGhostTopKernel<<<cl_grd, cl_blk>>>(d_mu, nx, ny, nz_interior);

            // 4) Concentration update
            updateKernel<<<sm_grd, sm_blk>>>(
                d_cold, d_mu, d_cnew, nx, ny, nz_local,
                D, dt, inv_dx2, inv_dy2, inv_dz2);

            // 5) Swap buffers (pointer swap, O(1))
            double* tmp = d_cold; d_cold = d_cnew; d_cnew = tmp;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- Timing (max across ranks) ----
    long long local_ms = duration.count();
    long long max_ms;
    MPI_CHECK(MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG_INT,
                         MPI_MAX, 0, MPI_COMM_WORLD));

    double total_cells = static_cast<double>(nx) * ny * nz;
    double cellUpdates = total_cells * iterations;
    double mcups = cellUpdates
                 / (std::max(max_ms, static_cast<long long>(1)) / 1000.0)
                 / 1e6;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Results output ----
    if (printResults) {
        std::vector<double> local_data;
        if (nz_interior > 0) {
            local_data.resize(nz_interior * layer_size);
            CUDA_CHECK(cudaMemcpy(local_data.data(), d_cold + layer_size,
                                  nz_interior * layer_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> recvcounts(num_ranks), displs(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            size_t r_nz_pr = nz / num_ranks;
            size_t r_rem   = nz % num_ranks;
            size_t r_zs = r * r_nz_pr + std::min(static_cast<size_t>(r), r_rem);
            size_t r_ze = (r + 1) * r_nz_pr
                         + std::min(static_cast<size_t>(r + 1), r_rem);
            recvcounts[r] = static_cast<int>((r_ze - r_zs) * layer_size);
            displs[r]     = static_cast<int>(r_zs * layer_size);
        }

        std::vector<double> global_data;
        if (rank == 0) global_data.resize(nx * ny * nz);

        MPI_CHECK(MPI_Gatherv(
            local_data.data(), static_cast<int>(local_data.size()), MPI_DOUBLE,
            global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD));

        if (rank == 0) print_results(global_data, "Concentration");
    }

    // ---- Validation (OpenMP parallel reduction) ----
    if (validate_flag) {
        int local_valid = 1;
        double local_min = 1e300, local_max = -1e300;

        if (nz_interior > 0) {
            std::vector<double> local_data(nz_interior * layer_size);
            CUDA_CHECK(cudaMemcpy(local_data.data(), d_cold + layer_size,
                                  nz_interior * layer_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            local_min = local_data[0];
            local_max = local_data[0];

            #pragma omp parallel for reduction(min:local_min) \
                                     reduction(max:local_max) \
                                     reduction(min:local_valid)
            for (size_t i = 0; i < local_data.size(); ++i) {
                if (std::isnan(local_data[i]) || std::isinf(local_data[i]))
                    local_valid = 0;
                local_min = std::min(local_min, local_data[i]);
                local_max = std::max(local_max, local_data[i]);
            }
        }

        int global_valid = 1;
        double global_min = 1e300, global_max = -1e300;
        MPI_CHECK(MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT,
                                MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE,
                                MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE,
                                MPI_MAX, MPI_COMM_WORLD));

        if (rank == 0) {
            printf("Validating result...\n");
            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
            if (global_valid && global_max <= 10.0 && global_min >= -10.0) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed: values out of expected range\n");
                printf("Validation: FAILED\n");
                // cleanup before early return
                if (d_cold)  CUDA_CHECK(cudaFree(d_cold));
                if (d_cnew)  CUDA_CHECK(cudaFree(d_cnew));
                if (d_mu)    CUDA_CHECK(cudaFree(d_mu));
                if (h_send_bottom)  CUDA_CHECK(cudaFreeHost(h_send_bottom));
                if (h_send_top)     CUDA_CHECK(cudaFreeHost(h_send_top));
                if (h_recv_bottom)  CUDA_CHECK(cudaFreeHost(h_recv_bottom));
                if (h_recv_top)     CUDA_CHECK(cudaFreeHost(h_recv_top));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // ---- Cleanup ----
    if (d_cold)        CUDA_CHECK(cudaFree(d_cold));
    if (d_cnew)        CUDA_CHECK(cudaFree(d_cnew));
    if (d_mu)          CUDA_CHECK(cudaFree(d_mu));
    if (h_send_bottom) CUDA_CHECK(cudaFreeHost(h_send_bottom));
    if (h_send_top)    CUDA_CHECK(cudaFreeHost(h_send_top));
    if (h_recv_bottom) CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    if (h_recv_top)    CUDA_CHECK(cudaFreeHost(h_recv_top));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
