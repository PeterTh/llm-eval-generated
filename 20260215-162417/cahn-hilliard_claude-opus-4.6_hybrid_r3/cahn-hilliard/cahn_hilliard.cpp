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
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index: z-major layout (z * nx*ny + y * nx + x)
__host__ __device__ inline size_t idx3(size_t x, size_t y, size_t z,
                                       size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential on GPU
// Local z indices: 1..nz_local are real cells; 0 and nz_local+1 are ghost layers
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || zl > nz_local) return;

    size_t id = idx3(x, y, zl, nx, ny);
    double cv = c[id];

    // Clamped BC for x and y; ghost cells handle z boundaries
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : y;
    size_t zp = zl + 1;
    size_t zn = zl - 1;

    double cxx = (c[idx3(xp, y, zl, nx, ny)] + c[idx3(xn, y, zl, nx, ny)] - 2.0 * cv) * inv_dx2;
    double cyy = (c[idx3(x, yp, zl, nx, ny)] + c[idx3(x, yn, zl, nx, ny)] - 2.0 * cv) * inv_dy2;
    double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * cv) * inv_dz2;

    mu[id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * (cxx + cyy + czz);
}

// Cahn-Hilliard update step on GPU
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    size_t nx, size_t ny, size_t nz_local,
    double D_dt, double inv_dx2, double inv_dy2, double inv_dz2)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || zl > nz_local) return;

    size_t id = idx3(x, y, zl, nx, ny);

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : y;
    size_t zp = zl + 1;
    size_t zn = zl - 1;

    double mval = mu[id];
    double mxx = (mu[idx3(xp, y, zl, nx, ny)] + mu[idx3(xn, y, zl, nx, ny)] - 2.0 * mval) * inv_dx2;
    double myy = (mu[idx3(x, yp, zl, nx, ny)] + mu[idx3(x, yn, zl, nx, ny)] - 2.0 * mval) * inv_dy2;
    double mzz = (mu[idx3(x, y, zp, nx, ny)] + mu[idx3(x, y, zn, nx, ny)] - 2.0 * mval) * inv_dz2;

    cnew[id] = cold[id] + D_dt * (mxx + myy + mzz);
}

// Exchange ghost layers between MPI ranks via host staging buffers.
// At global domain boundaries, fill ghosts with clamped BC (copy of boundary slab).
void exchangeHalos(double* d_arr, size_t nx, size_t ny, size_t nz_local,
                   double* h_send_lo, double* h_send_hi,
                   double* h_recv_lo, double* h_recv_hi,
                   int rank, int num_ranks, cudaStream_t stream)
{
    if (nz_local == 0) return;

    size_t slab_bytes = nx * ny * sizeof(double);
    int slab_count = static_cast<int>(nx * ny);

    // Copy outgoing boundary slabs from device to pinned host memory
    CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_arr + 1 * nx * ny,
                               slab_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_arr + nz_local * nx * ny,
                               slab_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Non-blocking MPI halo exchange
    MPI_Request reqs[4];
    int nreqs = 0;

    if (rank > 0) {
        MPI_Isend(h_send_lo, slab_count, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_lo, slab_count, MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (rank < num_ranks - 1) {
        MPI_Isend(h_send_hi, slab_count, MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_hi, slab_count, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Copy received halos to device ghost layers
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr, h_recv_lo,
                                   slab_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Global lower boundary: clamped BC — ghost = copy of first real slab
        CUDA_CHECK(cudaMemcpyAsync(d_arr, d_arr + nx * ny,
                                   slab_bytes, cudaMemcpyDeviceToDevice, stream));
    }

    if (rank < num_ranks - 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr + (nz_local + 1) * nx * ny, h_recv_hi,
                                   slab_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Global upper boundary: clamped BC — ghost = copy of last real slab
        CUDA_CHECK(cudaMemcpyAsync(d_arr + (nz_local + 1) * nx * ny,
                                   d_arr + nz_local * nx * ny,
                                   slab_bytes, cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Assign GPU round-robin across local ranks
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double D_dt = D * dt;

    // Domain decomposition along z
    size_t base_nz = nz / num_ranks;
    size_t remainder = nz % num_ranks;
    size_t nz_local = base_nz + ((size_t)rank < remainder ? 1 : 0);
    size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, remainder);

    // Local array dimensions: nz_local real slabs + 2 ghost slabs
    size_t slab_size = nx * ny;
    size_t local_total = slab_size * (nz_local + 2);

    // Initialize local concentration on host (with ghost offset z+1)
    std::vector<double> h_cold(local_total, 0.0);
    {
        size_t vol = nx * ny * nz;
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t z = 0; z < nz_local; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t global_z = z_start + z;
                    size_t linear_id = global_z * slab_size + y * nx + x;
                    double pseudo = (((linear_id + 1) * 1299709) % vol)
                                    / static_cast<double>(vol);
                    h_cold[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
                }
            }
        }
    }

    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   local_total * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, local_total * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu,   0, local_total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold, h_cold.data(),
                          local_total * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned staging buffers for halo exchange
    double *h_send_lo, *h_send_hi, *h_recv_lo, *h_recv_hi;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, slab_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, slab_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, slab_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, slab_size * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // CUDA kernel launch configuration
    dim3 block(8, 8, 4);
    dim3 grid((unsigned)((nx + block.x - 1) / block.x),
              (unsigned)((ny + block.y - 1) / block.y),
              (unsigned)((nz_local + block.z - 1) / block.z));

    if (rank == 0) printf("Initializing concentration field...\n");
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration field
        exchangeHalos(d_cold, nx, ny, nz_local,
                      h_send_lo, h_send_hi, h_recv_lo, h_recv_hi,
                      rank, num_ranks, stream);

        // Compute chemical potential
        if (nz_local > 0) {
            computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
                d_cold, d_mu, nx, ny, nz_local,
                inv_dx2, inv_dy2, inv_dz2,
                gamma, e_AA, e_BB, e_AB);
        }

        // Exchange halos for chemical potential
        exchangeHalos(d_mu, nx, ny, nz_local,
                      h_send_lo, h_send_hi, h_recv_lo, h_recv_hi,
                      rank, num_ranks, stream);

        // Update concentration
        if (nz_local > 0) {
            cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
                d_cnew, d_cold, d_mu,
                nx, ny, nz_local,
                D_dt, inv_dx2, inv_dy2, inv_dz2);
        }

        // Swap device pointers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full result on rank 0 for validation / output
    int ret = 0;
    if (printResults || validate) {
        // Copy local real cells from device (skip ghost layer at z=0)
        std::vector<double> h_local(nz_local * slab_size);
        if (nz_local > 0) {
            CUDA_CHECK(cudaMemcpy(h_local.data(), d_cold + slab_size,
                                  nz_local * slab_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        // MPI_Gatherv to rank 0
        int local_count = static_cast<int>(nz_local * slab_size);
        std::vector<int> recvcounts(num_ranks), displs(num_ranks);
        MPI_Gather(&local_count, 1, MPI_INT,
                   recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> cold_global;
        if (rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < num_ranks; ++r)
                displs[r] = displs[r - 1] + recvcounts[r - 1];
            cold_global.resize(nx * ny * nz);
        }

        MPI_Gatherv(h_local.data(), local_count, MPI_DOUBLE,
                     cold_global.data(), recvcounts.data(), displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(cold_global, "Concentration");
        }

        if (rank == 0 && validate) {
            printf("Validating result...\n");
            bool valid = true;

            #pragma omp parallel for schedule(static) reduction(&& : valid)
            for (size_t i = 0; i < cold_global.size(); ++i) {
                if (std::isnan(cold_global[i]) || std::isinf(cold_global[i]))
                    valid = false;
            }

            if (valid) {
                double minVal = cold_global[0], maxVal = cold_global[0];
                #pragma omp parallel for schedule(static) \
                    reduction(min : minVal) reduction(max : maxVal)
                for (size_t i = 0; i < cold_global.size(); ++i) {
                    if (cold_global[i] < minVal) minVal = cold_global[i];
                    if (cold_global[i] > maxVal) maxVal = cold_global[i];
                }
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            } else {
                printf("Validation failed: found NaN or Inf value\n");
            }

            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) ret = 1;
        }

        // Broadcast return code so all ranks exit consistently
        MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return ret;
}
