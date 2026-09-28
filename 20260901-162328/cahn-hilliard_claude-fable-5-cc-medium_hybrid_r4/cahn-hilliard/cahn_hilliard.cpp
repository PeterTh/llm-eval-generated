#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),          \
                    __FILE__, __LINE__);                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions on a local slab with halo planes.
// The field layout has one halo plane below (plane 0) and one above (plane lnz+1);
// local planes occupy indices 1..lnz. zlo/zhi indicate whether a neighbor plane exists
// (halo filled by MPI) below/above; if not, the global clamped boundary applies.
__device__ inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                                          const double dx, const double dy, const double dz,
                                          const size_t x, const size_t y, const size_t z,
                                          const size_t lnz, const bool zlo, const bool zhi) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // z is the halo-offset plane index (1..lnz)
    const size_t zp = (z < lnz) ? z + 1 : (zhi ? z + 1 : z);
    const size_t zn = (z > 1) ? z - 1 : (zlo ? z - 1 : z);

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t lnz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB,
                                               const double e_AB, const bool zlo, const bool zhi) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;  // halo offset
    if (x >= nx || y >= ny || z > lnz) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z, lnz, zlo, zhi);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t lnz,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz,
                                         const bool zlo, const bool zhi) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;  // halo offset
    if (x >= nx || y >= ny || z > lnz) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
               computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z, lnz, zlo, zhi);
}

// Initialize local slab of the concentration field (values depend on global indices)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t global_nz, const size_t z_start, const size_t lnz) {
    const size_t vol = nx * ny * global_nz;

    #pragma omp parallel for collapse(2)
    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Local layout includes a lower halo plane at index 0
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z_start + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool bad = false;
    #pragma omp parallel for reduction(||:bad)
    for (size_t i = 0; i < c.size(); ++i) {
        bad = bad || std::isnan(c[i]) || std::isinf(c[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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

// Exchange the boundary planes of a device field with z-neighbors, filling halo planes.
// Layout: plane 0 = lower halo, planes 1..lnz = local data, plane lnz+1 = upper halo.
static void exchangeHalos(double* d_field, const size_t planeElems, const size_t lnz,
                          const int rankDown, const int rankUp,
                          double* h_sendLo, double* h_sendHi, double* h_recvLo, double* h_recvHi,
                          MPI_Comm comm) {
    const size_t planeBytes = planeElems * sizeof(double);

    if (rankDown != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(h_sendLo, d_field + planeElems, planeBytes, cudaMemcpyDeviceToHost));
    if (rankUp != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(h_sendHi, d_field + lnz * planeElems, planeBytes, cudaMemcpyDeviceToHost));

    MPI_Request reqs[4];
    int nreq = 0;
    if (rankDown != MPI_PROC_NULL) {
        MPI_Irecv(h_recvLo, (int)planeElems, MPI_DOUBLE, rankDown, 0, comm, &reqs[nreq++]);
        MPI_Isend(h_sendLo, (int)planeElems, MPI_DOUBLE, rankDown, 1, comm, &reqs[nreq++]);
    }
    if (rankUp != MPI_PROC_NULL) {
        MPI_Irecv(h_recvHi, (int)planeElems, MPI_DOUBLE, rankUp, 1, comm, &reqs[nreq++]);
        MPI_Isend(h_sendHi, (int)planeElems, MPI_DOUBLE, rankUp, 0, comm, &reqs[nreq++]);
    }
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (rankDown != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(d_field, h_recvLo, planeBytes, cudaMemcpyHostToDevice));
    if (rankUp != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(d_field + (lnz + 1) * planeElems, h_recvHi, planeBytes, cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks, omp_get_max_threads());
    }

    // Bind each rank on a node to its own GPU (round-robin over local ranks)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

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

    const size_t gridSize = nx * ny * nz;
    const size_t planeElems = nx * ny;

    // 1D slab decomposition of z-planes across ranks (contiguous, remainder to low ranks)
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    auto slabSize = [&](int r) -> size_t {
        return base + ((size_t)r < rem ? 1 : 0);
    };
    auto slabStart = [&](int r) -> size_t {
        return (size_t)r * base + std::min((size_t)r, rem);
    };
    const size_t lnz = slabSize(rank);
    const size_t z_start = slabStart(rank);

    const int rankDown = (lnz > 0 && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rankUp = (lnz > 0 && rank + 1 < nranks && slabSize(rank + 1) > 0) ? rank + 1 : MPI_PROC_NULL;
    const bool zlo = (rankDown != MPI_PROC_NULL);
    const bool zhi = (rankUp != MPI_PROC_NULL);

    const size_t localElems = planeElems * (lnz + 2);  // includes halo planes
    const size_t localBytes = localElems * sizeof(double);

    // Host slab (with halo planes) used for init and result gathering
    std::vector<double> hostSlab(localElems, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(hostSlab, nx, ny, nz, z_start, lnz);

    // Device arrays
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, localBytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, localBytes));
    CUDA_CHECK(cudaMalloc(&d_mu, localBytes));
    CUDA_CHECK(cudaMemcpy(d_cold, hostSlab.data(), localBytes, cudaMemcpyHostToDevice));

    // Pinned host staging buffers for halo exchange
    double *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_sendLo, planeElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_sendHi, planeElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvLo, planeElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvHi, planeElems * sizeof(double)));

    const dim3 block(32, 8, 1);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)(lnz > 0 ? (lnz + block.z - 1) / block.z : 1));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (lnz > 0) {
            // Exchange concentration halos, then compute chemical potential
            exchangeHalos(d_cold, planeElems, lnz, rankDown, rankUp,
                          h_sendLo, h_sendHi, h_recvLo, h_recvHi, MPI_COMM_WORLD);
            computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, lnz,
                                                            dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                                            zlo, zhi);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Exchange chemical potential halos, then update concentration
            exchangeHalos(d_mu, planeElems, lnz, rankDown, rankUp,
                          h_sendLo, h_sendHi, h_recvLo, h_recvHi, MPI_COMM_WORLD);
            cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, lnz,
                                                      D, dt, dx, dy, dz, zlo, zhi);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (globalDuration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy local result back and gather the full field on rank 0
    CUDA_CHECK(cudaMemcpy(hostSlab.data(), d_cold, localBytes, cudaMemcpyDeviceToHost));

    std::vector<double> cfull;
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = (int)(slabSize(r) * planeElems);
        displs[r] = (int)(slabStart(r) * planeElems);
    }
    if (rank == 0) cfull.resize(gridSize);
    MPI_Gatherv(hostSlab.data() + planeElems, (int)(lnz * planeElems), MPI_DOUBLE,
                rank == 0 ? cfull.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(cfull, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cfull, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));

    MPI_Finalize();
    return exitCode;
}
