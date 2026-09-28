#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// 3D index calculation (z includes halo offset in device arrays)
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in x/y; z neighbors come from
// halo planes (which are filled with a copy of the adjacent interior plane at global
// domain boundaries, reproducing the clamped condition exactly).
__device__ inline double laplacianHalo(const double* __restrict__ f, const size_t nx,
                                       const size_t ny, const double inv_dx2,
                                       const double inv_dy2, const double inv_dz2,
                                       const size_t x, const size_t y, const size_t zh) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double fc = f[idx3(x, y, zh, nx, ny)];
    const double fxx = (f[idx3(xp, y, zh, nx, ny)] + f[idx3(xn, y, zh, nx, ny)] - 2.0 * fc) * inv_dx2;
    const double fyy = (f[idx3(x, yp, zh, nx, ny)] + f[idx3(x, yn, zh, nx, ny)] - 2.0 * fc) * inv_dy2;
    const double fzz = (f[idx3(x, y, zh + 1, nx, ny)] + f[idx3(x, y, zh - 1, nx, ny)] - 2.0 * fc) * inv_dz2;

    return fxx + fyy + fzz;
}

// Compute chemical potential
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t lnz,
                                        const double inv_dx2, const double inv_dy2,
                                        const double inv_dz2, const double gamma,
                                        const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= lnz) return;

    const size_t zh = z + 1;  // shift into halo layout
    const size_t idx = idx3(x, y, zh, nx, ny);
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * laplacianHalo(c, nx, ny, inv_dx2, inv_dy2, inv_dz2, x, y, zh);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu, const size_t nx,
                                         const size_t ny, const size_t lnz, const double D,
                                         const double dt, const double inv_dx2,
                                         const double inv_dy2, const double inv_dz2) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= lnz) return;

    const size_t zh = z + 1;
    const size_t idx = idx3(x, y, zh, nx, ny);
    cnew[idx] = cold[idx] + dt * D * laplacianHalo(mu, nx, ny, inv_dx2, inv_dy2, inv_dz2, x, y, zh);
}

// Initialize concentration field (local slab, halo layout; global indices for the
// pseudo-random sequence so the result is independent of the decomposition)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t z0, const size_t lnz) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z0 + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf, and compute range
    bool bad = false;
    double minVal = c[0];
    double maxVal = c[0];

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        bad = bad || std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
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

// Exchange one-plane z-halos with neighbor ranks; at global domain boundaries fill the
// halo plane with a copy of the adjacent interior plane (clamped boundary condition).
static void exchangeHalos(double* d_field, const size_t plane, const size_t lnz, const int down,
                          const int up, double* h_send_dn, double* h_send_up, double* h_recv_dn,
                          double* h_recv_up) {
    const size_t bytes = plane * sizeof(double);

    if (down != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(h_send_dn, d_field + plane, bytes, cudaMemcpyDeviceToHost));
    if (up != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(h_send_up, d_field + lnz * plane, bytes, cudaMemcpyDeviceToHost));

    MPI_Sendrecv(h_send_dn, (int)plane, MPI_DOUBLE, down, 0,
                 h_recv_up, (int)plane, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_up, (int)plane, MPI_DOUBLE, up, 1,
                 h_recv_dn, (int)plane, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (down != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(d_field, h_recv_dn, bytes, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpy(d_field, d_field + plane, bytes, cudaMemcpyDeviceToDevice));

    if (up != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(d_field + (lnz + 1) * plane, h_recv_up, bytes, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpy(d_field + (lnz + 1) * plane, d_field + lnz * plane, bytes,
                              cudaMemcpyDeviceToDevice));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind each rank to a GPU by node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int nDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nDevices));
        CUDA_CHECK(cudaSetDevice(localRank % nDevices));
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
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

    const size_t gridSize = nx * ny * nz;

    // 1D slab decomposition along z (remainder planes go to the lowest ranks)
    const size_t base = nz / (size_t)size;
    const size_t rem = nz % (size_t)size;
    const size_t lnz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    const bool active = lnz > 0;

    auto rankPlanes = [&](int r) { return base + ((size_t)r < rem ? 1 : 0); };
    const int down = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (active && rank + 1 < size && rankPlanes(rank + 1) > 0) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localSize = plane * (lnz + 2);  // interior + 2 halo planes

    // Host buffer for the local slab (halo layout)
    std::vector<double> h_c(active ? localSize : 0, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    if (active) initializeConcentration(h_c, nx, ny, nz, z0, lnz);

    // Device arrays and pinned halo staging buffers
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    double *h_send_dn = nullptr, *h_send_up = nullptr, *h_recv_dn = nullptr, *h_recv_up = nullptr;
    if (active) {
        CUDA_CHECK(cudaMalloc(&d_cold, localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, localSize * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_send_dn, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_send_up, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_dn, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recv_up, plane * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cold, h_c.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    const dim3 block(32, 8, 1);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)(active ? lnz : 1));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // Exchange concentration halos, then compute chemical potential
            exchangeHalos(d_cold, plane, lnz, down, up, h_send_dn, h_send_up, h_recv_dn, h_recv_up);
            chemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, lnz, inv_dx2, inv_dy2,
                                                     inv_dz2, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Exchange chemical-potential halos, then update concentration
            exchangeHalos(d_mu, plane, lnz, down, up, h_send_dn, h_send_up, h_recv_dn, h_recv_up);
            cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, lnz, D, dt,
                                                      inv_dx2, inv_dy2, inv_dz2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Swap buffers
            std::swap(d_cold, d_cnew);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = (long)((tEnd - tStart) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for result printing / validation
    std::vector<double> cfull;
    if (validate || printResults) {
        if (active) {
            CUDA_CHECK(cudaMemcpy(h_c.data() + plane, d_cold + plane, plane * lnz * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (int)(rankPlanes(r) * plane);
            displs[r] = (int)(((size_t)r * base + std::min((size_t)r, rem)) * plane);
        }
        if (rank == 0) cfull.resize(gridSize);
        MPI_Gatherv(active ? h_c.data() + plane : nullptr, (int)(lnz * plane), MPI_DOUBLE,
                    cfull.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(cfull, "Concentration");
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(cfull, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (active) {
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(h_send_dn));
        CUDA_CHECK(cudaFreeHost(h_send_up));
        CUDA_CHECK(cudaFreeHost(h_recv_dn));
        CUDA_CHECK(cudaFreeHost(h_recv_up));
    }

    MPI_Finalize();
    return exitCode;
}
