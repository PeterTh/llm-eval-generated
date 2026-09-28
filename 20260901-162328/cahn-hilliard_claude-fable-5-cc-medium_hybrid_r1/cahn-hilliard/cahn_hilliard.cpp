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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions in x/y.
// The z direction always has valid neighbor planes: interior ranks hold halo
// planes exchanged via MPI, and global z boundaries hold a mirrored copy of the
// adjacent plane, which reproduces the clamped stencil exactly.
__device__ inline double computeLaplacian(const double* __restrict__ c,
                                          const size_t nx, const size_t ny,
                                          const double dx, const double dy, const double dz,
                                          const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential for local planes z in [z0, z1) (halo-inclusive indexing)
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t z0, const size_t z1,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB,
                                               const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, dx, dy, dz, x, y, z);
}

// Cahn-Hilliard update step for local planes z in [z0, z1)
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t z0, const size_t z1,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;

    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
               computeLaplacian(mu, nx, ny, dx, dy, dz, x, y, z);
}

// Initialize the local slab [zOffset, zOffset + nzLocal) of the concentration
// field. Values are a function of the global linear cell id, so the result is
// identical for any rank count.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nzGlobal, const size_t zOffset, const size_t nzLocal) {
    const size_t vol = nx * ny * nzGlobal;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nzLocal; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Local array includes one halo plane below the slab
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z + zOffset) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf, and track value range
    bool bad = false;
    double minVal = c[0];
    double maxVal = c[0];

#pragma omp parallel for reduction(||:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
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

namespace {

// Distributed halo state for one rank
struct HaloExchange {
    MPI_Comm comm = MPI_COMM_NULL;
    int lower = MPI_PROC_NULL;   // rank holding the slab below (smaller z)
    int upper = MPI_PROC_NULL;   // rank holding the slab above (larger z)
    size_t plane = 0;            // nx * ny
    size_t nzLocal = 0;
    double* h_sendLo = nullptr;  // pinned staging buffers
    double* h_sendHi = nullptr;
    double* h_recvLo = nullptr;
    double* h_recvHi = nullptr;

    // Exchange the halo planes of d_field (layout: (nzLocal + 2) planes, halos
    // at plane 0 and plane nzLocal + 1). Global boundaries mirror the adjacent
    // plane so the kernels always see a clamped z-stencil. All GPU work runs
    // on `stream` so it can overlap with interior computation on another stream.
    void run(double* d_field, cudaStream_t stream) const {
        const size_t bytes = plane * sizeof(double);

        if (lower != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_field + plane, bytes, cudaMemcpyDeviceToHost, stream));
        if (upper != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_field + nzLocal * plane, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Sendrecv(h_sendLo, static_cast<int>(plane), MPI_DOUBLE, lower, 0,
                     h_recvHi, static_cast<int>(plane), MPI_DOUBLE, upper, 0,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_sendHi, static_cast<int>(plane), MPI_DOUBLE, upper, 1,
                     h_recvLo, static_cast<int>(plane), MPI_DOUBLE, lower, 1,
                     comm, MPI_STATUS_IGNORE);

        if (lower != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_field, h_recvLo, bytes, cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_field, d_field + plane, bytes, cudaMemcpyDeviceToDevice, stream));
        }
        if (upper != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_field + (nzLocal + 1) * plane, h_recvHi, bytes,
                                       cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_field + (nzLocal + 1) * plane, d_field + nzLocal * plane, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }
};

dim3 launchGrid(const dim3 block, const size_t nx, const size_t ny, const size_t nzRange) {
    return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x),
                static_cast<unsigned>((ny + block.y - 1) / block.y),
                static_cast<unsigned>((nzRange + block.z - 1) / block.z));
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
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

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // 1D slab decomposition along z
    const size_t nzBase = nz / worldSize;
    const size_t nzRem = nz % worldSize;
    const size_t nzLocal = nzBase + (static_cast<size_t>(worldRank) < nzRem ? 1 : 0);
    const size_t zOffset = static_cast<size_t>(worldRank) * nzBase +
                           std::min<size_t>(worldRank, nzRem);

    // Ranks without any planes (more ranks than z planes) stay idle; the active
    // ranks form a contiguous, order-preserving subcommunicator so z-neighbors
    // are simply rank +/- 1 within it.
    const bool active = nzLocal > 0;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    // Select GPU round-robin among the ranks sharing this node
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));

    // Local slab plus one halo plane on each side
    const size_t localCells = nzLocal * plane;
    const size_t localAlloc = active ? (nzLocal + 2) * plane : 0;

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    HaloExchange halo;
    cudaStream_t streamBnd = nullptr; // boundary planes + halo traffic
    cudaStream_t streamInt = nullptr; // interior planes, overlaps the exchange

    if (active) {
        int activeRank = 0;
        int activeSize = 0;
        MPI_Comm_rank(activeComm, &activeRank);
        MPI_Comm_size(activeComm, &activeSize);

        CUDA_CHECK(cudaMalloc(&d_cold, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&streamBnd));
        CUDA_CHECK(cudaStreamCreate(&streamInt));

        halo.comm = activeComm;
        halo.lower = (activeRank > 0) ? activeRank - 1 : MPI_PROC_NULL;
        halo.upper = (activeRank < activeSize - 1) ? activeRank + 1 : MPI_PROC_NULL;
        halo.plane = plane;
        halo.nzLocal = nzLocal;
        CUDA_CHECK(cudaMallocHost(&halo.h_sendLo, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&halo.h_sendHi, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&halo.h_recvLo, plane * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&halo.h_recvHi, plane * sizeof(double)));
    }

    // Initialize concentration field
    if (worldRank == 0) printf("Initializing concentration field...\n");
    if (active) {
        std::vector<double> hostInit(localAlloc, 0.0);
        initializeConcentration(hostInit, nx, ny, nz, zOffset, nzLocal);
        CUDA_CHECK(cudaMemcpy(d_cold, hostInit.data(), localAlloc * sizeof(double),
                              cudaMemcpyHostToDevice));
        halo.run(d_cold, streamBnd);
        CUDA_CHECK(cudaStreamSynchronize(streamBnd));
    }

    const dim3 block(32, 4, 4);

    // Run simulation
    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // Local plane indices: 1 .. nzLocal (0 and nzLocal + 1 are halos).
            // Boundary planes run on streamBnd so their halo exchange can start
            // while streamInt processes the interior.
            const size_t intLo = 2;
            const size_t intHi = nzLocal; // interior range [2, nzLocal)

            // Chemical potential: boundary planes first, then interior
            computeChemicalPotentialKernel<<<launchGrid(block, nx, ny, 1), block, 0, streamBnd>>>(
                d_cold, d_mu, nx, ny, 1, 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (nzLocal > 1) {
                computeChemicalPotentialKernel<<<launchGrid(block, nx, ny, 1), block, 0, streamBnd>>>(
                    d_cold, d_mu, nx, ny, nzLocal, nzLocal + 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            if (intHi > intLo) {
                computeChemicalPotentialKernel<<<launchGrid(block, nx, ny, intHi - intLo), block, 0, streamInt>>>(
                    d_cold, d_mu, nx, ny, intLo, intHi, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Exchange mu halos while the interior kernel runs
            halo.run(d_mu, streamBnd);
            CUDA_CHECK(cudaStreamSynchronize(streamInt));
            CUDA_CHECK(cudaStreamSynchronize(streamBnd));

            // Concentration update: boundary planes first, then interior
            cahnHilliardUpdateKernel<<<launchGrid(block, nx, ny, 1), block, 0, streamBnd>>>(
                d_cnew, d_cold, d_mu, nx, ny, 1, 2, D, dt, dx, dy, dz);
            if (nzLocal > 1) {
                cahnHilliardUpdateKernel<<<launchGrid(block, nx, ny, 1), block, 0, streamBnd>>>(
                    d_cnew, d_cold, d_mu, nx, ny, nzLocal, nzLocal + 1, D, dt, dx, dy, dz);
            }
            if (intHi > intLo) {
                cahnHilliardUpdateKernel<<<launchGrid(block, nx, ny, intHi - intLo), block, 0, streamInt>>>(
                    d_cnew, d_cold, d_mu, nx, ny, intLo, intHi, D, dt, dx, dy, dz);
            }

            // Exchange the updated concentration halos for the next step while
            // the interior update finishes
            halo.run(d_cnew, streamBnd);
            CUDA_CHECK(cudaStreamSynchronize(streamInt));
            CUDA_CHECK(cudaStreamSynchronize(streamBnd));
        }

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    if (active) CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the full field on rank 0 for result printing and validation
    std::vector<double> cold;
    if (printResults || validate) {
        std::vector<double> localHost(localCells);
        if (active) {
            CUDA_CHECK(cudaMemcpy(localHost.data(), d_cold + plane, localCells * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(worldSize);
        std::vector<int> displs(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t nzr = nzBase + (static_cast<size_t>(r) < nzRem ? 1 : 0);
            const size_t off = static_cast<size_t>(r) * nzBase + std::min<size_t>(r, nzRem);
            counts[r] = static_cast<int>(nzr * plane);
            displs[r] = static_cast<int>(off * plane);
        }

        if (worldRank == 0) cold.resize(gridSize);
        MPI_Gatherv(localHost.data(), static_cast<int>(localCells), MPI_DOUBLE,
                    cold.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    // Print results for external validation
    if (printResults && worldRank == 0) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);

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
        CUDA_CHECK(cudaFreeHost(halo.h_sendLo));
        CUDA_CHECK(cudaFreeHost(halo.h_sendHi));
        CUDA_CHECK(cudaFreeHost(halo.h_recvLo));
        CUDA_CHECK(cudaFreeHost(halo.h_recvHi));
        CUDA_CHECK(cudaStreamDestroy(streamBnd));
        CUDA_CHECK(cudaStreamDestroy(streamInt));
        MPI_Comm_free(&activeComm);
    }
    MPI_Comm_free(&nodeComm);

    MPI_Finalize();
    return exitCode;
}
