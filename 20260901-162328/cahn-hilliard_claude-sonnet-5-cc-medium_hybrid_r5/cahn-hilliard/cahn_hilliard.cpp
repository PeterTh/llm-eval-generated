#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err__));                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

namespace {

constexpr int kBlockSize = 256;

// Index into a locally padded array: layout is [ghost-low][nz_local interior planes][ghost-high],
// each plane contiguous of size nx*ny (matches the global idx3 z-major layout used by the
// original serial code, so a rank's interior slab is bit-identical to the corresponding
// contiguous slice of the global array).
__device__ __forceinline__ size_t idx3Local(size_t x, size_t y, size_t zp, size_t nx, size_t ny) {
    return zp * (nx * ny) + y * nx + x;
}

__device__ double laplacianDevice(const double* __restrict__ c, size_t nx, size_t ny,
                                   double dx, double dy, double dz,
                                   size_t x, size_t y, size_t zp) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    // z neighbors: zp+1 / zp-1 are always valid in the padded array. Ghost planes are
    // filled each iteration either from the neighbor rank (interior boundary) or by
    // self-copy of the edge plane at the global domain boundary, which exactly reproduces
    // the clamped-boundary semantics of the original single-process implementation.
    const size_t zzp = zp + 1;
    const size_t zzn = zp - 1;

    const double center = c[idx3Local(x, y, zp, nx, ny)];
    const double cxx = (c[idx3Local(xp, y, zp, nx, ny)] + c[idx3Local(xn, y, zp, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3Local(x, yp, zp, nx, ny)] + c[idx3Local(x, yn, zp, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3Local(x, y, zzp, nx, ny)] + c[idx3Local(x, y, zzn, nx, ny)] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c, size_t nx, size_t ny, size_t nzLocal,
                                               size_t zStart, size_t nzGlobal) {
    const size_t total = nx * ny * nzLocal;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const size_t x = i % nx;
    const size_t y = (i / nx) % ny;
    const size_t lz = i / (nx * ny);
    const size_t z = zStart + lz;
    const size_t vol = nx * ny * nzGlobal;

    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = static_cast<double>(((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol);

    c[idx3Local(x, y, lz + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                size_t nx, size_t ny, size_t nzLocal,
                                                double dx, double dy, double dz,
                                                double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t total = nx * ny * nzLocal;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const size_t x = i % nx;
    const size_t y = (i / nx) % ny;
    const size_t lz = i / (nx * ny);
    const size_t zp = lz + 1;

    const double cv = c[idx3Local(x, y, zp, nx, ny)];
    const double lap = laplacianDevice(c, nx, ny, dx, dy, dz, x, y, zp);

    mu[idx3Local(x, y, zp, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                     + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                          const double* __restrict__ mu,
                                          size_t nx, size_t ny, size_t nzLocal,
                                          double D, double dt, double dx, double dy, double dz) {
    const size_t total = nx * ny * nzLocal;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const size_t x = i % nx;
    const size_t y = (i / nx) % ny;
    const size_t lz = i / (nx * ny);
    const size_t zp = lz + 1;

    const double lap = laplacianDevice(mu, nx, ny, dx, dy, dz, x, y, zp);
    cnew[idx3Local(x, y, zp, nx, ny)] = cold[idx3Local(x, y, zp, nx, ny)] + dt * D * lap;
}

// Exchanges the single-plane z-halos of a padded device array with the two neighboring
// MPI ranks (or self-copies the edge plane at the global domain boundary). Staged through
// pinned host buffers so the exchange works regardless of whether the MPI implementation
// is CUDA-aware.
void exchangeZHalo(double* d_field, double* h_sendLo, double* h_sendHi, double* h_recvLo, double* h_recvHi,
                    size_t planeElems, size_t nzLocal, int rank, int size, MPI_Comm comm) {
    const size_t planeBytes = planeElems * sizeof(double);
    double* d_ghostLo = d_field;
    double* d_first = d_field + planeElems;
    double* d_last = d_field + nzLocal * planeElems;
    double* d_ghostHi = d_field + (nzLocal + 1) * planeElems;

    const int down = rank - 1;
    const int up = rank + 1;
    constexpr int TAG_TO_DOWN = 0;
    constexpr int TAG_TO_UP = 1;

    MPI_Request reqs[4];
    int nreq = 0;

    if (down >= 0) {
        CUDA_CHECK(cudaMemcpy(h_sendLo, d_first, planeBytes, cudaMemcpyDeviceToHost));
        MPI_Isend(h_sendLo, static_cast<int>(planeElems), MPI_DOUBLE, down, TAG_TO_DOWN, comm, &reqs[nreq++]);
        MPI_Irecv(h_recvLo, static_cast<int>(planeElems), MPI_DOUBLE, down, TAG_TO_UP, comm, &reqs[nreq++]);
    }
    if (up < size) {
        CUDA_CHECK(cudaMemcpy(h_sendHi, d_last, planeBytes, cudaMemcpyDeviceToHost));
        MPI_Isend(h_sendHi, static_cast<int>(planeElems), MPI_DOUBLE, up, TAG_TO_UP, comm, &reqs[nreq++]);
        MPI_Irecv(h_recvHi, static_cast<int>(planeElems), MPI_DOUBLE, up, TAG_TO_DOWN, comm, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (down >= 0) {
        CUDA_CHECK(cudaMemcpy(d_ghostLo, h_recvLo, planeBytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_ghostLo, d_first, planeBytes, cudaMemcpyDeviceToDevice));
    }
    if (up < size) {
        CUDA_CHECK(cudaMemcpy(d_ghostHi, h_recvHi, planeBytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_ghostHi, d_last, planeBytes, cudaMemcpyDeviceToDevice));
    }
}

} // namespace

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                     [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();

    bool foundBad = false;
    #pragma omp parallel for reduction(||:foundBad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            foundBad = true;
        }
    }
    if (foundBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Bind this rank to a GPU local to its node so that multiple ranks per node
    // (and multiple nodes) each get a distinct accelerator.
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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

    if (static_cast<size_t>(worldSize) > nz) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid Z extent (%zu)\n", worldSize, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
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
    const size_t planeElems = nx * ny;

    // Z-slab domain decomposition across MPI ranks (contiguous ranges, remainder
    // distributed to the first ranks). Because z is the outermost (slowest-varying)
    // dimension in idx3, each rank's local slab is exactly a contiguous slice of the
    // global flattened array, which allows a plain Gatherv to reconstruct it.
    const size_t base = nz / static_cast<size_t>(worldSize);
    const size_t rem = nz % static_cast<size_t>(worldSize);
    const size_t nzLocal = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const size_t paddedLocal = planeElems * (nzLocal + 2);

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, paddedLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, paddedLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, paddedLocal * sizeof(double)));

    double *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_sendLo, planeElems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_sendHi, planeElems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvLo, planeElems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvHi, planeElems * sizeof(double), cudaHostAllocDefault));

    if (rank == 0) printf("Initializing concentration field...\n");
    {
        const size_t total = planeElems * nzLocal;
        const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
        if (blocks > 0) {
            initializeConcentrationKernel<<<blocks, kBlockSize>>>(d_cold, nx, ny, nzLocal, zStart, nz);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t total = planeElems * nzLocal;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);

    for (int t = 0; t < iterations; ++t) {
        exchangeZHalo(d_cold, h_sendLo, h_sendHi, h_recvLo, h_recvHi, planeElems, nzLocal, rank, worldSize, MPI_COMM_WORLD);

        if (blocks > 0) {
            computeChemicalPotentialKernel<<<blocks, kBlockSize>>>(d_cold, d_mu, nx, ny, nzLocal, dx, dy, dz,
                                                                    gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());
        }

        exchangeZHalo(d_mu, h_sendLo, h_sendHi, h_recvLo, h_recvHi, planeElems, nzLocal, rank, worldSize, MPI_COMM_WORLD);

        if (blocks > 0) {
            cahnHilliardUpdateKernel<<<blocks, kBlockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nzLocal, D, dt, dx, dy, dz);
            CUDA_CHECK(cudaGetLastError());
        }

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long durationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&durationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the local interior slabs (contiguous, matching the global z-major layout)
    // back to rank 0 to reproduce the original single-process result vector.
    std::vector<double> hostLocal(planeElems * nzLocal);
    if (!hostLocal.empty()) {
        CUDA_CHECK(cudaMemcpy(hostLocal.data(), d_cold + planeElems, planeElems * nzLocal * sizeof(double),
                               cudaMemcpyDeviceToHost));
    }

    std::vector<double> cold;
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        cold.resize(gridSize);
        recvCounts.resize(worldSize);
        displs.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const size_t rBase = nz / static_cast<size_t>(worldSize);
            const size_t rRem = nz % static_cast<size_t>(worldSize);
            const size_t rNzLocal = rBase + (static_cast<size_t>(r) < rRem ? 1 : 0);
            const size_t rZStart = static_cast<size_t>(r) * rBase + std::min(static_cast<size_t>(r), rRem);
            recvCounts[r] = static_cast<int>(planeElems * rNzLocal);
            displs[r] = static_cast<int>(planeElems * rZStart);
        }
    }
    MPI_Gatherv(hostLocal.data(), static_cast<int>(hostLocal.size()), MPI_DOUBLE,
                rank == 0 ? cold.data() : nullptr, rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(cold, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
