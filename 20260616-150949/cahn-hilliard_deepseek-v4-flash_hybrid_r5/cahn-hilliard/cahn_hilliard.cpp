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

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = call;                                               \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error @ %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));             \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int err_ = call;                                                       \
        if (err_ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error @ %s:%d\n", __FILE__, __LINE__);        \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 3D index helper (host + device)
// ---------------------------------------------------------------------------
__host__ __device__ inline size_t
idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernel: chemical potential
// Arrays have pre-filled ghost layers in z.
// ---------------------------------------------------------------------------
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double dx, double dy, double dz,
    double gamma, double e_AA, double e_BB, double e_AB,
    size_t nx_ny) {

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    size_t za  = zl + 1;          // skip ghost-bottom in the array
    size_t idx = za * nx_ny + y * nx + x;
    double cv  = c[idx];

    // x neighbours (clamped at physical boundaries)
    double c_xp = (x < nx - 1) ? c[za * nx_ny + y * nx + (x + 1)] : cv;
    double c_xn = (x > 0)      ? c[za * nx_ny + y * nx + (x - 1)] : cv;

    // y neighbours (clamped)
    double c_yp = (y < ny - 1) ? c[za * nx_ny + (y + 1) * nx + x] : cv;
    double c_yn = (y > 0)      ? c[za * nx_ny + (y - 1) * nx + x] : cv;

    // z neighbours (ghost layers already filled)
    double c_zp = c[(za + 1) * nx_ny + y * nx + x];
    double c_zn = c[(za - 1) * nx_ny + y * nx + x];

    double laplacian =
        (c_xp + c_xn - 2.0 * cv) / (dx * dx) +
        (c_yp + c_yn - 2.0 * cv) / (dy * dy) +
        (c_zp + c_zn - 2.0 * cv) / (dz * dz);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * laplacian;
}

// ---------------------------------------------------------------------------
// CUDA kernel: Cahn-Hilliard update
// mu ghost layers are pre-filled.
// ---------------------------------------------------------------------------
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double D, double dt, double dx, double dy, double dz,
    size_t nx_ny) {

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    size_t za  = zl + 1;
    size_t idx = za * nx_ny + y * nx + x;

    // Laplacian of mu (clamped x/y, pre-filled z ghosts)
    double mu_xp = (x < nx - 1) ? mu[za * nx_ny + y * nx + (x + 1)] : mu[idx];
    double mu_xn = (x > 0)      ? mu[za * nx_ny + y * nx + (x - 1)] : mu[idx];
    double mu_yp = (y < ny - 1) ? mu[za * nx_ny + (y + 1) * nx + x] : mu[idx];
    double mu_yn = (y > 0)      ? mu[za * nx_ny + (y - 1) * nx + x] : mu[idx];
    double mu_zp = mu[(za + 1) * nx_ny + y * nx + x];
    double mu_zn = mu[(za - 1) * nx_ny + y * nx + x];

    double lap_mu =
        (mu_xp + mu_xn - 2.0 * mu[idx]) / (dx * dx) +
        (mu_yp + mu_yn - 2.0 * mu[idx]) / (dy * dy) +
        (mu_zp + mu_zn - 2.0 * mu[idx]) / (dz * dz);

    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// ---------------------------------------------------------------------------
// CUDA kernel: initialise concentration on the local domain
// ---------------------------------------------------------------------------
__global__ void initializeConcentrationKernel(
    double* __restrict__ c,
    size_t nx, size_t ny, size_t nz_local,
    size_t local_z_offset, size_t total_vol,
    size_t nx_ny) {

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    size_t za       = zl + 1;
    size_t idx      = za * nx_ny + y * nx + x;
    size_t global_z = local_z_offset + zl;
    size_t linear_id = global_z * nx_ny + y * nx + x;
    double pseudo   = (((linear_id + 1) * 1299709ULL) % total_vol) /
                       static_cast<double>(total_vol);
    c[idx] = -1.0 + 2.0 * pseudo;
}

// ---------------------------------------------------------------------------
// Ghost-layer exchange in the z-direction (1 layer each side)
// Uses pinned host buffers for CUDA <-> MPI staging.
// ---------------------------------------------------------------------------
static void exchangeGhostsZ(
    double* d_arr,
    double* h_send_bottom, double* h_recv_bottom,
    double* h_send_top,    double* h_recv_top,
    size_t nx, size_t ny, size_t nz_local,
    int rank, int size, MPI_Comm comm)
{
    size_t nx_ny      = nx * ny;
    size_t slice_bytes = nx_ny * sizeof(double);

    // ---- single rank: clamped boundaries only ----
    if (size == 1) {
        CUDA_CHECK(cudaMemcpy(d_arr + 0 * nx_ny,
                               d_arr + 1 * nx_ny,
                               slice_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_arr + (nz_local + 1) * nx_ny,
                               d_arr + nz_local * nx_ny,
                               slice_bytes, cudaMemcpyDeviceToDevice));
        return;
    }

    // ---- copy local boundary slices from device to pinned host ----
    CUDA_CHECK(cudaMemcpy(h_send_bottom, d_arr + 1 * nx_ny,
                           slice_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_top,
                           d_arr + nz_local * nx_ny,
                           slice_bytes, cudaMemcpyDeviceToHost));

    // ---- MPI exchange (all use tag 0 for consistent matching) ----
    int has_bottom = (rank > 0);
    int has_top    = (rank < size - 1);

    if (has_bottom && has_top) {
        // Exchange with lower neighbour: send our bottom slice, receive into bottom ghost
        MPI_CHECK(MPI_Sendrecv(h_send_bottom, (int)nx_ny, MPI_DOUBLE,
                                rank - 1, 0,
                                h_recv_bottom, (int)nx_ny, MPI_DOUBLE,
                                rank - 1, 0, comm, MPI_STATUS_IGNORE));
        // Exchange with upper neighbour: send our top slice, receive into top ghost
        MPI_CHECK(MPI_Sendrecv(h_send_top,    (int)nx_ny, MPI_DOUBLE,
                                rank + 1, 0,
                                h_recv_top,    (int)nx_ny, MPI_DOUBLE,
                                rank + 1, 0, comm, MPI_STATUS_IGNORE));
    } else if (has_bottom) {
        // Last rank: only bottom neighbour exists
        MPI_CHECK(MPI_Sendrecv(h_send_bottom, (int)nx_ny, MPI_DOUBLE,
                                rank - 1, 0,
                                h_recv_bottom, (int)nx_ny, MPI_DOUBLE,
                                rank - 1, 0, comm, MPI_STATUS_IGNORE));
    } else if (has_top) {
        // First rank: only top neighbour exists
        MPI_CHECK(MPI_Sendrecv(h_send_top,    (int)nx_ny, MPI_DOUBLE,
                                rank + 1, 0,
                                h_recv_top,    (int)nx_ny, MPI_DOUBLE,
                                rank + 1, 0, comm, MPI_STATUS_IGNORE));
    }
    // else size == 1: already handled above

    // ---- copy received / clamped ghost layers to device ----
    if (has_bottom) {
        CUDA_CHECK(cudaMemcpy(d_arr + 0 * nx_ny,
                               h_recv_bottom, slice_bytes,
                               cudaMemcpyHostToDevice));
    } else {
        // physical bottom boundary: clamped
        CUDA_CHECK(cudaMemcpy(d_arr + 0 * nx_ny,
                               d_arr + 1 * nx_ny, slice_bytes,
                               cudaMemcpyDeviceToDevice));
    }
    if (has_top) {
        CUDA_CHECK(cudaMemcpy(d_arr + (nz_local + 1) * nx_ny,
                               h_recv_top, slice_bytes,
                               cudaMemcpyHostToDevice));
    } else {
        // physical top boundary: clamped
        CUDA_CHECK(cudaMemcpy(d_arr + (nz_local + 1) * nx_ny,
                               d_arr + nz_local * nx_ny, slice_bytes,
                               cudaMemcpyDeviceToDevice));
    }
}

// ---------------------------------------------------------------------------
// Validation (host side, uses OpenMP for parallel reduction)
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& c, [[maybe_unused]] size_t nx,
                    [[maybe_unused]] size_t ny, [[maybe_unused]] size_t nz) {
    size_t n = c.size();
    if (n == 0) return true;

    int has_nan_inf = 0;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel for reduction(+:has_nan_inf) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        double v = c[i];
        if (std::isnan(v) || std::isinf(v)) has_nan_inf = 1;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }

    if (has_nan_inf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // ---- parse command line (rank 0 broadcasts to all) ----
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    int validate = 0, printResults = 0;

    if (mpi_rank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
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
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ---- sanity check ----
    if (nz < (size_t)mpi_size) {
        if (mpi_rank == 0)
            fprintf(stderr, "Error: Z dimension (%zu) must be >= MPI size (%d)\n",
                    nz, mpi_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // ---- domain decomposition along z ----
    size_t base_nz    = nz / mpi_size;
    size_t rem        = nz % mpi_size;
    size_t nz_local   = base_nz + (mpi_rank < (int)rem ? 1 : 0);
    size_t local_z_offset = mpi_rank * base_nz + std::min((size_t)mpi_rank, rem);

    size_t nx_ny       = nx * ny;
    size_t local_slice = nx_ny * nz_local;          // interior elements
    size_t local_alloc = nx_ny * (nz_local + 2);    // + 2 ghost layers

    if (mpi_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Local Z per rank: %zu\n", nz_local);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- physical parameters ----
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB =  (2.0 / 9.0);
    const double gamma = 0.5;
    const double D    = 1.0;

    // ---- select GPU ----
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int gpu_id = mpi_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // ---- allocate device memory ----
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   local_alloc * sizeof(double)));

    // ---- pinned host buffers for ghost exchange ----
    double *h_send_bottom, *h_recv_bottom, *h_send_top, *h_recv_top;
    size_t slice_size = nx_ny;
    CUDA_CHECK(cudaHostAlloc(&h_send_bottom, slice_size * sizeof(double),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_bottom, slice_size * sizeof(double),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_send_top,    slice_size * sizeof(double),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_top,    slice_size * sizeof(double),
                              cudaHostAllocDefault));

    // ---- initialise concentration on device ----
    {
        dim3 block(8, 8, 4);
        dim3 grid((nx + block.x - 1) / block.x,
                  (ny + block.y - 1) / block.y,
                  (nz_local + block.z - 1) / block.z);
        size_t total_vol = nx * ny * nz;

        initializeConcentrationKernel<<<grid, block>>>(
            d_cold, nx, ny, nz_local, local_z_offset, total_vol, nx_ny);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- initial ghost-layer setup for cold ----
    exchangeGhostsZ(d_cold,
                    h_send_bottom, h_recv_bottom, h_send_top, h_recv_top,
                    nx, ny, nz_local, mpi_rank, mpi_size, MPI_COMM_WORLD);

    // ---- kernel launch configuration ----
    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz_local + block.z - 1) / block.z);

    // ---- simulation loop ----
    if (mpi_rank == 0)
        printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // 1. exchange ghost layers of c
        exchangeGhostsZ(d_cold,
                        h_send_bottom, h_recv_bottom, h_send_top, h_recv_top,
                        nx, ny, nz_local, mpi_rank, mpi_size, MPI_COMM_WORLD);

        // 2. compute chemical potential  (c -> mu)
        computeChemicalPotentialKernel<<<grid, block>>>(
            d_cold, d_mu, nx, ny, nz_local,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB, nx_ny);
        CUDA_CHECK(cudaGetLastError());

        // 3. exchange ghost layers of mu
        exchangeGhostsZ(d_mu,
                        h_send_bottom, h_recv_bottom, h_send_top, h_recv_top,
                        nx, ny, nz_local, mpi_rank, mpi_size, MPI_COMM_WORLD);

        // 4. update concentration  (cold, mu -> cnew)
        cahnHilliardUpdateKernel<<<grid, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local,
            D, dt, dx, dy, dz, nx_ny);
        CUDA_CHECK(cudaGetLastError());

        // 5. swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_duration_ms, &duration_ms, 1,
                         MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD));

    // ---- copy result to host ----
    std::vector<double> h_local(local_alloc);
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_cold, local_alloc * sizeof(double),
                           cudaMemcpyDeviceToHost));

    // ---- gather full result on rank 0 ----
    std::vector<double> full;
    std::vector<int> recv_counts(mpi_size), displs(mpi_size);

    for (int r = 0; r < mpi_size; ++r) {
        size_t rnz = base_nz + (r < (int)rem ? 1 : 0);
        recv_counts[r] = (int)(nx_ny * rnz);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + recv_counts[r - 1];
    }

    if (mpi_rank == 0)
        full.resize(nx * ny * nz);

    MPI_CHECK(MPI_Gatherv(h_local.data() + nx_ny,   // skip ghost-bottom
                           (int)local_slice, MPI_DOUBLE,
                           full.data(), recv_counts.data(), displs.data(),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // ---- report timing (rank 0) ----
    if (mpi_rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        size_t gridSize = nx * ny * nz;
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- print results (rank 0) ----
    if (printResults && mpi_rank == 0) {
        print_results(full, "Concentration");
    }

    // ---- validation (rank 0, uses OpenMP) ----
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(full, nx, ny, nz);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(h_send_bottom));
        CUDA_CHECK(cudaFreeHost(h_recv_bottom));
        CUDA_CHECK(cudaFreeHost(h_send_top));
        CUDA_CHECK(cudaFreeHost(h_recv_top));

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    // ---- cleanup ----
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_top));

    MPI_Finalize();
    return 0;
}
