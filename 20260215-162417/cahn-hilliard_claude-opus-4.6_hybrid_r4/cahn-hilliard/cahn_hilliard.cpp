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

// CUDA kernel: initialize concentration field with deterministic pseudo-random values
__global__ void initializeConcentrationKernel(
    double* __restrict__ c,
    const size_t nx, const size_t ny, const size_t local_nz,
    const size_t z_offset, const size_t vol)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) return;

    const size_t plane = nx * ny;
    const size_t local_idx = z * plane + y * nx + x;
    const size_t global_z = z_offset + (z - 1);
    const size_t linear_id = global_z * plane + y * nx + x;

    const double pseudo = (((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol);
    c[local_idx] = -1.0 + 2.0 * pseudo;
}

// CUDA kernel: compute chemical potential from concentration field
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) return;

    const size_t plane = nx * ny;
    const size_t center = z * plane + y * nx + x;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cv = c[center];

    const double laplacian =
        (c[z * plane + y * nx + xp] + c[z * plane + y * nx + xn] - 2.0 * cv) * inv_dx2 +
        (c[z * plane + yp * nx + x] + c[z * plane + yn * nx + x] - 2.0 * cv) * inv_dy2 +
        (c[(z + 1) * plane + y * nx + x] + c[(z - 1) * plane + y * nx + x] - 2.0 * cv) * inv_dz2;

    mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * laplacian;
}

// CUDA kernel: Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    const size_t nx, const size_t ny, const size_t local_nz,
    const double D_dt, const double inv_dx2, const double inv_dy2, const double inv_dz2)
{
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) return;

    const size_t plane = nx * ny;
    const size_t center = z * plane + y * nx + x;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double mu_c = mu[center];

    const double laplacian_mu =
        (mu[z * plane + y * nx + xp] + mu[z * plane + y * nx + xn] - 2.0 * mu_c) * inv_dx2 +
        (mu[z * plane + yp * nx + x] + mu[z * plane + yn * nx + x] - 2.0 * mu_c) * inv_dy2 +
        (mu[(z + 1) * plane + y * nx + x] + mu[(z - 1) * plane + y * nx + x] - 2.0 * mu_c) * inv_dz2;

    cnew[center] = cold[center] + D_dt * laplacian_mu;
}

// Exchange ghost layers between MPI ranks via pinned host buffers
void exchangeHalos(double* d_field, const size_t nx, const size_t ny, const size_t local_nz,
                   double* h_send_lo, double* h_send_hi, double* h_recv_lo, double* h_recv_hi,
                   const int rank, const int nprocs, cudaStream_t stream)
{
    const size_t plane_size = nx * ny;
    const size_t plane_bytes = plane_size * sizeof(double);

    // Copy boundary interior planes from device to pinned host
    CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_field + 1 * plane_size, plane_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_field + local_nz * plane_size, plane_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Non-blocking MPI halo exchange with neighbors
    MPI_Request reqs[4];
    int nreqs = 0;

    if (rank > 0) {
        MPI_Isend(h_send_lo, static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_lo, static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank < nprocs - 1) {
        MPI_Isend(h_send_hi, static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_hi, static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Copy received halos to device ghost layers
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_field, h_recv_lo, plane_bytes,
                                   cudaMemcpyHostToDevice, stream));
    } else {
        // Clamped BC: lower ghost = first interior plane
        CUDA_CHECK(cudaMemcpyAsync(d_field, d_field + plane_size, plane_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    if (rank < nprocs - 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_field + (local_nz + 1) * plane_size, h_recv_hi, plane_bytes,
                                   cudaMemcpyHostToDevice, stream));
    } else {
        // Clamped BC: upper ghost = last interior plane
        CUDA_CHECK(cudaMemcpyAsync(d_field + (local_nz + 1) * plane_size,
                                   d_field + local_nz * plane_size, plane_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// OpenMP-parallel validation on host
bool validateResult(const std::vector<double>& c, size_t nx, size_t ny, size_t nz) {
    int has_invalid = 0;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel for reduction(+:has_invalid) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            has_invalid++;
        }
        if (c[i] < minVal) minVal = c[i];
        if (c[i] > maxVal) maxVal = c[i];
    }

    if (has_invalid > 0) {
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign one GPU per MPI rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResultsFlag = false;

    // Parse command line arguments
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
            printResultsFlag = true;
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

    // Domain decomposition along Z
    const size_t base_nz = nz / static_cast<size_t>(nprocs);
    const size_t rem = nz % static_cast<size_t>(nprocs);
    const size_t local_nz = base_nz + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_offset = static_cast<size_t>(rank) * base_nz + std::min(static_cast<size_t>(rank), rem);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), num_devices);
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

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double D_dt = D * dt;

    const size_t gridSize = nx * ny * nz;
    const size_t plane_size = nx * ny;
    const size_t local_size = plane_size * (local_nz + 2); // +2 for ghost layers

    // Allocate device memory (interior + ghost layers in z)
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, local_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, local_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, local_size * sizeof(double)));

    // Pinned host buffers for halo exchange
    double *h_send_lo, *h_send_hi, *h_recv_lo, *h_recv_hi;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, plane_size * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Kernel launch configuration
    dim3 blockDim(8, 8, 4);
    dim3 gridDim(static_cast<unsigned int>((nx + blockDim.x - 1) / blockDim.x),
                 static_cast<unsigned int>((ny + blockDim.y - 1) / blockDim.y),
                 static_cast<unsigned int>((local_nz + blockDim.z - 1) / blockDim.z));

    // Initialize concentration field on GPU
    if (rank == 0) printf("Initializing concentration field...\n");
    if (local_nz > 0) {
        initializeConcentrationKernel<<<gridDim, blockDim, 0, stream>>>(
            d_cold, nx, ny, local_nz, z_offset, gridSize);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (local_nz > 0) {
            // Exchange halos for concentration field
            exchangeHalos(d_cold, nx, ny, local_nz,
                          h_send_lo, h_send_hi, h_recv_lo, h_recv_hi,
                          rank, nprocs, stream);

            // Compute chemical potential
            computeChemicalPotentialKernel<<<gridDim, blockDim, 0, stream>>>(
                d_cold, d_mu, nx, ny, local_nz,
                inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Exchange halos for chemical potential
            exchangeHalos(d_mu, nx, ny, local_nz,
                          h_send_lo, h_send_hi, h_recv_lo, h_recv_hi,
                          rank, nprocs, stream);

            // Update concentration
            cahnHilliardUpdateKernel<<<gridDim, blockDim, 0, stream>>>(
                d_cnew, d_cold, d_mu, nx, ny, local_nz,
                D_dt, inv_dx2, inv_dy2, inv_dz2);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Swap buffers (pointer swap)
            std::swap(d_cold, d_cnew);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration = duration.count();
    long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration);
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (global_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int ret = 0;

    // Gather results to rank 0 for output/validation
    if (printResultsFlag || validate) {
        // Copy local interior data from device to host
        std::vector<double> h_local(local_nz * plane_size);
        if (local_nz > 0) {
            CUDA_CHECK(cudaMemcpy(h_local.data(), d_cold + plane_size,
                                  local_nz * plane_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        // Gather counts and displacements
        int local_count = static_cast<int>(local_nz * plane_size);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> cold;
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
            cold.resize(gridSize);
        }

        MPI_Gatherv(h_local.data(), local_count, MPI_DOUBLE,
                     cold.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResultsFlag) {
                print_results(cold, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(cold, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return ret;
}
