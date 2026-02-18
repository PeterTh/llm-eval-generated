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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index for arrays with ghost layers in z (ghost offset already applied to z)
__host__ __device__ inline size_t idx3g(size_t x, size_t y, size_t z,
                                        size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: compute chemical potential
__global__ void chemPotKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB,
    int is_first, int is_last)
{
    size_t x  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y  = blockIdx.y * blockDim.y + threadIdx.y;
    size_t lz = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || lz >= nz_local) return;

    size_t z   = lz + 1; // +1 for ghost layer offset
    size_t idx = idx3g(x, y, z, nx, ny);
    double cv  = c[idx];

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0)      ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0)      ? y - 1 : 0;
    size_t zp = (is_last  && lz == nz_local - 1) ? z : z + 1;
    size_t zn = (is_first && lz == 0)             ? z : z - 1;

    double cxx = (c[idx3g(xp, y, z, nx, ny)] +
                  c[idx3g(xn, y, z, nx, ny)] - 2.0 * cv) * inv_dx2;
    double cyy = (c[idx3g(x, yp, z, nx, ny)] +
                  c[idx3g(x, yn, z, nx, ny)] - 2.0 * cv) * inv_dy2;
    double czz = (c[idx3g(x, y, zp, nx, ny)] +
                  c[idx3g(x, y, zn, nx, ny)] - 2.0 * cv) * inv_dz2;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * (cxx + cyy + czz);
}

// CUDA kernel: Cahn-Hilliard concentration update
__global__ void updateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double D_dt, double inv_dx2, double inv_dy2, double inv_dz2,
    int is_first, int is_last)
{
    size_t x  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y  = blockIdx.y * blockDim.y + threadIdx.y;
    size_t lz = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || lz >= nz_local) return;

    size_t z   = lz + 1;
    size_t idx = idx3g(x, y, z, nx, ny);

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0)      ? x - 1 : 0;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0)      ? y - 1 : 0;
    size_t zp = (is_last  && lz == nz_local - 1) ? z : z + 1;
    size_t zn = (is_first && lz == 0)             ? z : z - 1;

    double mv  = mu[idx];
    double mxx = (mu[idx3g(xp, y, z, nx, ny)] +
                  mu[idx3g(xn, y, z, nx, ny)] - 2.0 * mv) * inv_dx2;
    double myy = (mu[idx3g(x, yp, z, nx, ny)] +
                  mu[idx3g(x, yn, z, nx, ny)] - 2.0 * mv) * inv_dy2;
    double mzz = (mu[idx3g(x, y, zp, nx, ny)] +
                  mu[idx3g(x, y, zn, nx, ny)] - 2.0 * mv) * inv_dz2;

    cnew[idx] = cold[idx] + D_dt * (mxx + myy + mzz);
}

// Exchange ghost layers between MPI ranks via pinned host staging buffers
static void exchangeHalos(double* d_arr, size_t slice, size_t nz_local,
                          double* h_slo, double* h_shi,
                          double* h_rlo, double* h_rhi,
                          int rank, int is_first, int is_last) {
    // Copy boundary slices from device to pinned host memory
    CUDA_CHECK(cudaMemcpy(h_slo, d_arr + 1 * slice,
                          slice * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shi, d_arr + nz_local * slice,
                          slice * sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Request reqs[4];
    int nr = 0;
    if (!is_first) {
        MPI_Isend(h_slo, (int)slice, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[nr++]);
        MPI_Irecv(h_rlo, (int)slice, MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &reqs[nr++]);
    }
    if (!is_last) {
        MPI_Isend(h_shi, (int)slice, MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &reqs[nr++]);
        MPI_Irecv(h_rhi, (int)slice, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[nr++]);
    }
    if (nr > 0) MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);

    // Copy received ghost data from host to device
    if (!is_first)
        CUDA_CHECK(cudaMemcpy(d_arr, h_rlo,
                              slice * sizeof(double), cudaMemcpyHostToDevice));
    if (!is_last)
        CUDA_CHECK(cudaMemcpy(d_arr + (nz_local + 1) * slice, h_rhi,
                              slice * sizeof(double), cudaMemcpyHostToDevice));
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign one GPU per MPI rank (round-robin)
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(rank % ndev));

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)      nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)  ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)  nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)  iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)                   validate = true;
        else if (strcmp(argv[i], "-r") == 0)                   printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along z-axis
    size_t base_nz  = nz / (size_t)nprocs;
    size_t rem      = nz % (size_t)nprocs;
    size_t nz_local = base_nz + ((size_t)rank < rem ? 1 : 0);
    size_t z_start  = ((size_t)rank < rem)
                        ? (size_t)rank * (base_nz + 1)
                        : rem * (base_nz + 1) + ((size_t)rank - rem) * base_nz;

    int is_first = (rank == 0)            ? 1 : 0;
    int is_last  = (rank == nprocs - 1)   ? 1 : 0;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA GPUs: %d\n",
               nprocs, omp_get_max_threads(), ndev);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double D_dt    = D * dt;

    const size_t slice   = nx * ny;
    const size_t local_n = slice * nz_local;
    const size_t ghost_n = slice * (nz_local + 2);  // +2 ghost planes in z
    const size_t vol     = nx * ny * nz;

    // Initialize local portion on host using OpenMP
    std::vector<double> h_c(local_n);
    #pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < nz_local; ++lz) {
        size_t gz = z_start + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t li = lz * slice + y * nx + x;
                size_t gi = gz * slice + y * nx + x;
                double pseudo = (((gi + 1) * 1299709) % vol)
                                / static_cast<double>(vol);
                h_c[li] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Allocate device arrays (with ghost layers in z)
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, ghost_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, ghost_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   ghost_n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, ghost_n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, ghost_n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu,   0, ghost_n * sizeof(double)));

    // Copy initial data to device (offset by one z-plane for ghost layer)
    CUDA_CHECK(cudaMemcpy(d_cold + slice, h_c.data(),
                          local_n * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned host buffers for halo exchange
    double *h_slo, *h_shi, *h_rlo, *h_rhi;
    CUDA_CHECK(cudaMallocHost(&h_slo, slice * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_shi, slice * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_rlo, slice * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_rhi, slice * sizeof(double)));

    // Kernel launch configuration
    dim3 block(8, 8, 4);
    dim3 grid((unsigned)((nx + block.x - 1) / block.x),
              (unsigned)((ny + block.y - 1) / block.y),
              (unsigned)((nz_local + block.z - 1) / block.z));

    if (rank == 0) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto tstart = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers for concentration field
        exchangeHalos(d_cold, slice, nz_local,
                      h_slo, h_shi, h_rlo, h_rhi, rank, is_first, is_last);

        // Compute chemical potential on GPU
        chemPotKernel<<<grid, block>>>(
            d_cold, d_mu, nx, ny, nz_local,
            inv_dx2, inv_dy2, inv_dz2,
            gamma, e_AA, e_BB, e_AB, is_first, is_last);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exchange ghost layers for chemical potential
        exchangeHalos(d_mu, slice, nz_local,
                      h_slo, h_shi, h_rlo, h_rhi, rank, is_first, is_last);

        // Update concentration on GPU
        updateKernel<<<grid, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local,
            D_dt, inv_dx2, inv_dy2, inv_dz2, is_first, is_last);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap device pointers
        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto tend = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();

    // Copy result back to host (skip ghost offset)
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_cold + slice,
                          local_n * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather full result to rank 0 for validation / output
    std::vector<int> rcounts(nprocs), displs(nprocs);
    int mycount = (int)local_n;
    MPI_Gather(&mycount, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i)
            displs[i] = displs[i - 1] + rcounts[i - 1];
    }
    std::vector<double> global_c;
    if (rank == 0) global_c.resize(vol);
    MPI_Gatherv(h_c.data(), mycount, MPI_DOUBLE,
                global_c.data(), rcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", ms);
        double cellUpdates = (double)vol * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(global_c, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            double minVal = global_c[0], maxVal = global_c[0];
            #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
            for (size_t i = 0; i < vol; ++i) {
                if (std::isnan(global_c[i]) || std::isinf(global_c[i])) {
                    valid = false;
                }
                minVal = std::min(minVal, global_c[i]);
                maxVal = std::max(maxVal, global_c[i]);
            }
            if (!valid) {
                printf("Validation failed: found NaN or Inf value\n");
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) {
                CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu));
                CUDA_CHECK(cudaFreeHost(h_slo)); CUDA_CHECK(cudaFreeHost(h_shi));
                CUDA_CHECK(cudaFreeHost(h_rlo)); CUDA_CHECK(cudaFreeHost(h_rhi));
                MPI_Finalize(); return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_slo));
    CUDA_CHECK(cudaFreeHost(h_shi));
    CUDA_CHECK(cudaFreeHost(h_rlo));
    CUDA_CHECK(cudaFreeHost(h_rhi));
    MPI_Finalize();
    return 0;
}
