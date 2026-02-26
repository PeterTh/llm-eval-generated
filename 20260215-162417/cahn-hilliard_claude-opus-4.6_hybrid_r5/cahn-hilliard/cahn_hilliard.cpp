#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                __LINE__, cudaGetErrorString(err));                     \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                   \
} while(0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: compute chemical potential
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const int nx, const int ny, const int local_nz,
    const double inv_dx2, const double inv_dy2, const double inv_dz2,
    const double gamma, const double e_AA, const double e_BB, const double e_AB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int lz = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || lz >= local_nz) return;

    int z = lz + 1; // +1 for bottom halo
    size_t sxy = (size_t)nx * ny;
    size_t center = z * sxy + y * nx + x;

    int xp = min(x + 1, nx - 1);
    int xn = max(x - 1, 0);
    int yp = min(y + 1, ny - 1);
    int yn = max(y - 1, 0);

    double cv = c[center];

    double cxx = (c[z * sxy + y * nx + xp] +
                  c[z * sxy + y * nx + xn] - 2.0 * cv) * inv_dx2;
    double cyy = (c[z * sxy + yp * nx + x] +
                  c[z * sxy + yn * nx + x] - 2.0 * cv) * inv_dy2;
    double czz = (c[(z + 1) * sxy + y * nx + x] +
                  c[(z - 1) * sxy + y * nx + x] - 2.0 * cv) * inv_dz2;

    mu[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                         2.0 * cv * e_AB) +
                 3.0 * cv + cv * cv * cv -
                 gamma * (cxx + cyy + czz);
}

// CUDA kernel: Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const int nx, const int ny, const int local_nz,
    const double D_dt, const double inv_dx2,
    const double inv_dy2, const double inv_dz2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int lz = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || lz >= local_nz) return;

    int z = lz + 1;
    size_t sxy = (size_t)nx * ny;
    size_t center = z * sxy + y * nx + x;

    int xp = min(x + 1, nx - 1);
    int xn = max(x - 1, 0);
    int yp = min(y + 1, ny - 1);
    int yn = max(y - 1, 0);

    double mv = mu[center];

    double mxx = (mu[z * sxy + y * nx + xp] +
                  mu[z * sxy + y * nx + xn] - 2.0 * mv) * inv_dx2;
    double myy = (mu[z * sxy + yp * nx + x] +
                  mu[z * sxy + yn * nx + x] - 2.0 * mv) * inv_dy2;
    double mzz = (mu[(z + 1) * sxy + y * nx + x] +
                  mu[(z - 1) * sxy + y * nx + x] - 2.0 * mv) * inv_dz2;

    cnew[center] = cold[center] + D_dt * (mxx + myy + mzz);
}

// Exchange halo planes between MPI ranks via host-staged transfers
void exchangeHalos(double* d_data, double* h_send, double* h_recv,
                   size_t nx, size_t ny, size_t local_nz,
                   int rank, int nprocs, MPI_Comm comm)
{
    size_t slice = nx * ny;
    size_t bytes = slice * sizeof(double);

    double* slo = h_send;
    double* shi = h_send + slice;
    double* rlo = h_recv;
    double* rhi = h_recv + slice;

    int lo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int hi = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Device -> host: copy boundary interior slices
    CUDA_CHECK(cudaMemcpy(slo, d_data + slice, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(shi, d_data + slice * local_nz, bytes,
                          cudaMemcpyDeviceToHost));

    // MPI exchange with neighbors
    MPI_Sendrecv(slo, (int)slice, MPI_DOUBLE, lo, 0,
                 rlo, (int)slice, MPI_DOUBLE, lo, 1,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(shi, (int)slice, MPI_DOUBLE, hi, 1,
                 rhi, (int)slice, MPI_DOUBLE, hi, 0,
                 comm, MPI_STATUS_IGNORE);

    // Host -> device: fill halo planes
    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(d_data, rlo, bytes, cudaMemcpyHostToDevice));
    } else {
        // Clamped boundary: bottom halo mirrors first interior slice
        CUDA_CHECK(cudaMemcpy(d_data, d_data + slice, bytes,
                              cudaMemcpyDeviceToDevice));
    }

    if (rank < nprocs - 1) {
        CUDA_CHECK(cudaMemcpy(d_data + slice * (local_nz + 1), rhi, bytes,
                              cudaMemcpyHostToDevice));
    } else {
        // Clamped boundary: top halo mirrors last interior slice
        CUDA_CHECK(cudaMemcpy(d_data + slice * (local_nz + 1),
                              d_data + slice * local_nz, bytes,
                              cudaMemcpyDeviceToDevice));
    }
}

bool validateResult(const std::vector<double>& c,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz)
{
    double minVal = c[0], maxVal = c[0];
    bool valid = true;

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) \
        reduction(&&:valid)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) valid = false;
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

    // Assign one GPU per MPI rank
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(rank % ndev));

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

    if (nz < (size_t)nprocs) {
        if (rank == 0)
            fprintf(stderr,
                    "Error: Z dimension (%zu) must be >= MPI ranks (%d)\n",
                    nz, nprocs);
        MPI_Finalize();
        return 1;
    }

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

    // Domain decomposition along Z
    size_t base = nz / nprocs;
    size_t rem = nz % nprocs;
    size_t local_nz = base + ((size_t)rank < rem ? 1 : 0);
    size_t z_start = (size_t)rank * base + std::min((size_t)rank, rem);

    size_t slice = nx * ny;
    size_t local_total = slice * (local_nz + 2); // +2 for halo planes

    // Host-side initialization with OpenMP
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> h_init(local_total, 0.0);
    size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t lz = 0; lz < local_nz; ++lz)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                size_t gz = z_start + lz;
                size_t linear_id = gz * slice + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                                 static_cast<double>(vol));
                h_init[idx3(x, y, lz + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
            }

    // Device memory allocation
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_total * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_cold, h_init.data(),
                          local_total * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cnew, 0, local_total * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, local_total * sizeof(double)));

    // Pinned host buffers for halo exchange (2 slices send + 2 recv)
    double *h_send, *h_recv;
    CUDA_CHECK(cudaMallocHost(&h_send, 2 * slice * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv, 2 * slice * sizeof(double)));

    // Kernel launch configuration
    dim3 block(8, 8, 4);
    dim3 grid((unsigned)(nx + block.x - 1) / block.x,
              (unsigned)(ny + block.y - 1) / block.y,
              (unsigned)(local_nz + block.z - 1) / block.z);

    // Main simulation loop
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration field
        exchangeHalos(d_cold, h_send, h_recv, nx, ny, local_nz,
                      rank, nprocs, MPI_COMM_WORLD);

        // Compute chemical potential on GPU
        if (local_nz > 0) {
            computeChemicalPotentialKernel<<<grid, block>>>(
                d_cold, d_mu, (int)nx, (int)ny, (int)local_nz,
                inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exchange halos for chemical potential
        exchangeHalos(d_mu, h_send, h_recv, nx, ny, local_nz,
                      rank, nprocs, MPI_COMM_WORLD);

        // Update concentration on GPU
        if (local_nz > 0) {
            cahnHilliardUpdateKernel<<<grid, block>>>(
                d_cnew, d_cold, d_mu, (int)nx, (int)ny, (int)local_nz,
                D_dt, inv_dx2, inv_dy2, inv_dz2);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/output
    int local_count = (int)(local_nz * slice);
    std::vector<double> local_result(local_count);
    if (local_count > 0)
        CUDA_CHECK(cudaMemcpy(local_result.data(), d_cold + slice,
                              local_count * sizeof(double),
                              cudaMemcpyDeviceToHost));

    std::vector<int> counts(nprocs), displs(nprocs, 0);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);

    std::vector<double> cold;
    if (rank == 0) {
        for (int i = 1; i < nprocs; ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
        cold.resize(nx * ny * nz);
    }

    MPI_Gatherv(local_result.data(), local_count, MPI_DOUBLE,
                cold.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results for external validation
    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(cold, "Concentration");
        }

        // Validation
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

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return ret;
}
