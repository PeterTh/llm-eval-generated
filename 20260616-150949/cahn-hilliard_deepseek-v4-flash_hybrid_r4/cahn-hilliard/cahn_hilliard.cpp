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

// ============================================================================
// Error checking macros
// ============================================================================
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err_ = (call);                                         \
    if (err_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err_));          \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                   \
} while (0)

#define MPI_CHECK(call) do {                                           \
    int err_ = (call);                                                 \
    if (err_ != MPI_SUCCESS) {                                         \
        fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                   \
} while (0)

// ============================================================================
// Host index helper (local array with ghost offset of 1)
// ============================================================================
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Initialize concentration field on GPU.
// Uses the same linear-congruential pseudo-random sequence as the serial code
// so results are bit-identical regardless of domain decomposition.
__global__ void initConcentrationKernel(double* c, int nx, int ny, int nz_local,
                                         int nz_global, int global_z_start) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    int z_global = global_z_start + zl;
    size_t vol = (size_t)nx * ny * nz_global;
    size_t linear_id = (size_t)z_global * nx * ny + (size_t)y * nx + (size_t)x;
    double pseudo = ((((linear_id + 1) * 1299709) % vol) / (double)vol);

    int zs = zl + 1; // ghost offset
    c[zs * (size_t)nx * ny + (size_t)y * nx + (size_t)x] = -1.0 + 2.0 * pseudo;
}

// Compute chemical potential:  mu = f'(c) - gamma * Laplacian(c)
// Ghost cells for Z are expected to be filled before launching.
__global__ void chemicalPotentialKernel(
    const double* __restrict__ c,
    double* __restrict__ mu,
    int nx, int ny, int nz_local,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    int zs = zl + 1; // ghost offset
    size_t sx = (size_t)nx;
    size_t sxy = sx * ny;
    size_t idx = zs * sxy + (size_t)y * sx + (size_t)x;

    double cv = c[idx];

    // X / Y boundaries: clamped (Neumann)
    int xp = min(x + 1, nx - 1);
    int xn = max(x - 1, 0);
    int yp = min(y + 1, ny - 1);
    int yn = max(y - 1, 0);

    // Z boundaries: handled by ghost cells (MPI halo exchange)
    int zsp = zs + 1;
    int zsn = zs - 1;

    double lap =
        (c[zs * sxy + (size_t)y * sx + xp] +
         c[zs * sxy + (size_t)y * sx + xn] - 2.0 * cv) * inv_dx2 +
        (c[zs * sxy + (size_t)yp * sx + x] +
         c[zs * sxy + (size_t)yn * sx + x] - 2.0 * cv) * inv_dy2 +
        (c[(size_t)zsp * sxy + (size_t)y * sx + x] +
         c[(size_t)zsn * sxy + (size_t)y * sx + x] - 2.0 * cv) * inv_dz2;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * lap;
}

// Cahn-Hilliard update:  cnew = cold + dt * D * Laplacian(mu)
// Ghost cells for mu are expected to be filled before launching.
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew,
    const double* __restrict__ cold,
    const double* __restrict__ mu,
    int nx, int ny, int nz_local,
    double D, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    int zs = zl + 1;
    size_t sx = (size_t)nx;
    size_t sxy = sx * ny;
    size_t idx = zs * sxy + (size_t)y * sx + (size_t)x;

    int xp = min(x + 1, nx - 1);
    int xn = max(x - 1, 0);
    int yp = min(y + 1, ny - 1);
    int yn = max(y - 1, 0);
    int zsp = zs + 1;
    int zsn = zs - 1;

    double lap_mu =
        (mu[zs * sxy + (size_t)y * sx + xp] +
         mu[zs * sxy + (size_t)y * sx + xn] - 2.0 * mu[idx]) * inv_dx2 +
        (mu[zs * sxy + (size_t)yp * sx + x] +
         mu[zs * sxy + (size_t)yn * sx + x] - 2.0 * mu[idx]) * inv_dy2 +
        (mu[(size_t)zsp * sxy + (size_t)y * sx + x] +
         mu[(size_t)zsn * sxy + (size_t)y * sx + x] - 2.0 * mu[idx]) * inv_dz2;

    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// ============================================================================
// MPI Halo exchange for ghost cells along Z
// ============================================================================
void exchangeHalos(double* d_data, int nx, int ny, int nz_local,
                   int rank, int n_ranks,
                   double* h_send_bottom, double* h_send_top,
                   double* h_recv_bottom, double* h_recv_top,
                   cudaStream_t stream)
{
    size_t plane_size = (size_t)nx * ny;
    size_t plane_bytes = plane_size * sizeof(double);

    int bottom_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int top_neighbor    = (rank < n_ranks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Copy boundary planes from device to host staging buffers
    CUDA_CHECK(cudaMemcpyAsync(h_send_bottom, d_data + 1 * plane_size,
                                plane_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_send_top,
                                d_data + nz_local * plane_size,
                                plane_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Non-blocking MPI halo exchange
    MPI_Request reqs[4];
    int nreqs = 0;

    // Bottom neighbor: send our bottom plane, receive their top plane
    if (bottom_neighbor != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(h_send_bottom, (int)plane_size, MPI_DOUBLE,
                             bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]));
        MPI_CHECK(MPI_Irecv(h_recv_bottom, (int)plane_size, MPI_DOUBLE,
                             bottom_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]));
    }
    // Top neighbor: send our top plane, receive their bottom plane
    if (top_neighbor != MPI_PROC_NULL) {
        MPI_CHECK(MPI_Isend(h_send_top, (int)plane_size, MPI_DOUBLE,
                             top_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]));
        MPI_CHECK(MPI_Irecv(h_recv_top, (int)plane_size, MPI_DOUBLE,
                             top_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]));
    }

    if (nreqs > 0) {
        MPI_CHECK(MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE));
    }

    // Copy received data into ghost cells on device
    // For boundary ranks with no neighbor, fill ghost with clamped (boundary plane) value
    if (bottom_neighbor != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_recv_bottom, plane_bytes,
                                    cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_data, d_data + 1 * plane_size, plane_bytes,
                                    cudaMemcpyDeviceToDevice, stream));
    }

    if (top_neighbor != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_data + (nz_local + 1) * plane_size,
                                    h_recv_top, plane_bytes,
                                    cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_data + (nz_local + 1) * plane_size,
                                    d_data + nz_local * plane_size, plane_bytes,
                                    cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================================
// Host-side validation
// ============================================================================
bool validateResultHost(const std::vector<double>& c, size_t n) {
    bool valid = true;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            #pragma omp atomic write
            valid = false;
        }
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    if (!valid) {
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

// ============================================================================
// Usage
// ============================================================================
void printUsage(const char* progName) {
    printf("Usage: mpirun -np <N> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // ------------------------------------------------------------------
    // MPI initialisation
    // ------------------------------------------------------------------
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, n_ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

    // ------------------------------------------------------------------
    // Parse command line arguments (every rank parses independently)
    // ------------------------------------------------------------------
    size_t nx = 64;
    size_t ny = 0;
    size_t nz_global = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz_global = (size_t)atoi(argv[++i]);
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
    if (nz_global == 0) nz_global = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz_global);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", n_ranks);
        #ifdef _OPENMP
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        #endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ------------------------------------------------------------------
    // Physical parameters
    // ------------------------------------------------------------------
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB =  (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    // ------------------------------------------------------------------
    // Domain decomposition along Z
    // ------------------------------------------------------------------
    size_t nz_per_rank = nz_global / (size_t)n_ranks;
    size_t nz_rem      = nz_global % (size_t)n_ranks;
    size_t nz_local    = nz_per_rank + ((size_t)rank < nz_rem ? 1 : 0);

    size_t global_z_start = 0;
    for (int r = 0; r < rank; ++r) {
        global_z_start += nz_per_rank + ((size_t)r < nz_rem ? 1 : 0);
    }

    const int num_ghosts = 1;
    size_t plane_size   = nx * ny;
    size_t local_stride_z = nz_local + 2 * num_ghosts;  // ghosts on both sides
    size_t local_alloc  = plane_size * local_stride_z;

    // ------------------------------------------------------------------
    // CUDA setup: round-robin GPU assignment across ranks
    // ------------------------------------------------------------------
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(dev_id));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ------------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------------
    double *c_d, *mu_d, *cnew_d;
    CUDA_CHECK(cudaMalloc(&c_d,    local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu_d,   local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew_d, local_alloc * sizeof(double)));

    // Pinned host buffers for MPI halo exchange
    double *h_send_bottom, *h_send_top, *h_recv_bottom, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_top,    plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top,    plane_size * sizeof(double)));

    // ------------------------------------------------------------------
    // Launch configuration
    // ------------------------------------------------------------------
    dim3 block_dim(16, 16, 4);
    dim3 grid_3d(
        (nx + 15) / 16,
        (ny + 15) / 16,
        ((size_t)nz_local + 3) / 4
    );

    // ------------------------------------------------------------------
    // Initialise concentration field on GPU
    // ------------------------------------------------------------------
    if (rank == 0) printf("Initializing concentration field...\n");

    initConcentrationKernel<<<grid_3d, block_dim, 0, stream>>>(
        c_d, (int)nx, (int)ny, (int)nz_local, (int)nz_global, (int)global_z_start);
    CUDA_CHECK(cudaGetLastError());

    // Initial ghost fill for c
    exchangeHalos(c_d, (int)nx, (int)ny, (int)nz_local,
                  rank, n_ranks,
                  h_send_bottom, h_send_top,
                  h_recv_bottom, h_recv_top, stream);

    // ------------------------------------------------------------------
    // Main time loop
    // ------------------------------------------------------------------
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // --- Step 1: compute chemical potential  (needs c ghosts) ---
        chemicalPotentialKernel<<<grid_3d, block_dim, 0, stream>>>(
            c_d, mu_d,
            (int)nx, (int)ny, (int)nz_local,
            inv_dx2, inv_dy2, inv_dz2,
            gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 2: fill mu ghost cells ---
        exchangeHalos(mu_d, (int)nx, (int)ny, (int)nz_local,
                      rank, n_ranks,
                      h_send_bottom, h_send_top,
                      h_recv_bottom, h_recv_top, stream);

        // --- Step 3: concentration update (needs mu ghosts) ---
        cahnHilliardUpdateKernel<<<grid_3d, block_dim, 0, stream>>>(
            cnew_d, c_d, mu_d,
            (int)nx, (int)ny, (int)nz_local,
            D, dt,
            inv_dx2, inv_dy2, inv_dz2);
        CUDA_CHECK(cudaGetLastError());

        // --- Step 4: swap buffers ---
        std::swap(c_d, cnew_d);

        // --- Step 5: fill c ghost cells for next iteration ---
        if (t < iterations - 1) {
            exchangeHalos(c_d, (int)nx, (int)ny, (int)nz_local,
                          rank, n_ranks,
                          h_send_bottom, h_send_top,
                          h_recv_bottom, h_recv_top, stream);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ------------------------------------------------------------------
    // Gather results to rank 0 for output / validation
    // ------------------------------------------------------------------
    // Counts and displacements for MPI_Gatherv
    std::vector<int> recv_counts(n_ranks);
    std::vector<int> recv_displs(n_ranks);

    int local_int = (int)(nz_local * plane_size);
    MPI_CHECK(MPI_Gather(&local_int, 1, MPI_INT,
                          recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::vector<double> full_result;
    if (rank == 0) {
        size_t total_local = 0;
        for (int r = 0; r < n_ranks; ++r) {
            recv_displs[r] = (int)total_local;
            total_local += (size_t)recv_counts[r];
        }
        full_result.resize(nz_global * plane_size);
    }

    // Copy local interior (excluding ghosts) to a host buffer
    std::vector<double> local_host(local_int);
    CUDA_CHECK(cudaMemcpy(local_host.data(),
                           c_d + num_ghosts * plane_size,
                           (size_t)local_int * sizeof(double),
                           cudaMemcpyDeviceToHost));

    MPI_CHECK(MPI_Gatherv(local_host.data(), local_int, MPI_DOUBLE,
                           rank == 0 ? full_result.data() : nullptr,
                           recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                           0, MPI_COMM_WORLD));

    // ------------------------------------------------------------------
    // Output and validation (rank 0 only)
    // ------------------------------------------------------------------
    int exit_code = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = (double)(nx * ny * nz_global) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(full_result, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResultHost(full_result, full_result.size());

            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    // Broadcast exit code so all ranks finalize consistently
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFree(c_d));
    CUDA_CHECK(cudaFree(mu_d));
    CUDA_CHECK(cudaFree(cnew_d));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return exit_code;
}
