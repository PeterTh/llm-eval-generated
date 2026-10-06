// Hybrid MPI + OpenMP + CUDA Cahn-Hilliard benchmark.
//
// Parallelization:
//  - MPI: 1D slab decomposition along z. Each rank owns planes [z0, z1) and keeps
//    two ghost planes on each side for c (so that mu can be computed one plane
//    beyond the owned range and only one halo exchange per time step is needed).
//  - CUDA: one GPU per rank (selected by node-local rank); stencil kernels run on
//    the device. The (host-staged) halo exchange is overlapped with the interior
//    update and with the halo-independent part of the next chemical potential.
//  - OpenMP: host-side work (initialization, validation reductions).
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
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Clamped 7-point Laplacian. 'p' points at the center value; xm/xp/... are the
// (possibly zero, due to clamping) offsets to the -/+ neighbours. With unit grid
// spacing the (exact) multiplications by 1/(d*d) == 1.0 are skipped, which matters
// because the kernels are FP64-throughput bound on many GPUs.
template <bool UnitSpacing>
__device__ __forceinline__ double laplacian(const double* __restrict__ p, const ptrdiff_t xm, const ptrdiff_t xp,
                                            const ptrdiff_t ym, const ptrdiff_t yp, const ptrdiff_t zm,
                                            const ptrdiff_t zp, const double cv, const double idx2,
                                            const double idy2, const double idz2) {
    const double c2 = 2.0 * cv;
    double cxx = __ldg(p + xp) + __ldg(p - xm) - c2;
    double cyy = __ldg(p + yp) + __ldg(p - ym) - c2;
    double czz = __ldg(p + zp) + __ldg(p - zm) - c2;
    if constexpr (!UnitSpacing) {
        cxx *= idx2;
        cyy *= idy2;
        czz *= idz2;
    }
    return cxx + cyy + czz;
}

struct StencilParams {
    int nx, ny, nz;   // global grid
    int zlo;          // global z index of local plane 0
    double idx2, idy2, idz2;
};

// mu = f'(c) - gamma * lap(c) on global planes [zb, zb + gridDim.z).
// The bulk term 4.5*((c+1)e_AA + (c-1)e_BB - 2c e_AB) + 3c + c^3 is linear in c
// apart from c^3 and is evaluated as bulkA*c + bulkB + c^3 (coefficients from the host).
template <bool UnitSpacing>
__global__ void __launch_bounds__(256) chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                                const StencilParams P, const int zb,
                                                                const double gamma, const double bulkA,
                                                                const double bulkB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = zb + blockIdx.z;
    if (x >= P.nx || y >= P.ny) return;
    const ptrdiff_t plane = (ptrdiff_t)P.nx * P.ny;
    const size_t idx = (size_t)(z - P.zlo) * plane + (size_t)y * P.nx + x;
    const double* p = c + idx;
    const double cv = __ldg(p);
    const ptrdiff_t xp = (x < P.nx - 1) ? 1 : 0, xm = (x > 0) ? 1 : 0;
    const ptrdiff_t yp = (y < P.ny - 1) ? P.nx : 0, ym = (y > 0) ? P.nx : 0;
    const ptrdiff_t zp = (z < P.nz - 1) ? plane : 0, zm = (z > 0) ? plane : 0;
    mu[idx] = (bulkA * cv + bulkB) + cv * cv * cv -
              gamma * laplacian<UnitSpacing>(p, xm, xp, ym, yp, zm, zp, cv, P.idx2, P.idy2, P.idz2);
}

// cnew = cold + dt * D * lap(mu) on global planes [zb, zb + gridDim.z)
template <bool UnitSpacing>
__global__ void __launch_bounds__(256) updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                                     const double* __restrict__ mu, const StencilParams P,
                                                     const int zb, const double dtD) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = zb + blockIdx.z;
    if (x >= P.nx || y >= P.ny) return;
    const ptrdiff_t plane = (ptrdiff_t)P.nx * P.ny;
    const size_t idx = (size_t)(z - P.zlo) * plane + (size_t)y * P.nx + x;
    const double* p = mu + idx;
    const ptrdiff_t xp = (x < P.nx - 1) ? 1 : 0, xm = (x > 0) ? 1 : 0;
    const ptrdiff_t yp = (y < P.ny - 1) ? P.nx : 0, ym = (y > 0) ? P.nx : 0;
    const ptrdiff_t zp = (z < P.nz - 1) ? plane : 0, zm = (z > 0) ? plane : 0;
    cnew[idx] = __ldg(cold + idx) + dtD * laplacian<UnitSpacing>(p, xm, xp, ym, yp, zm, zp, __ldg(p), P.idx2, P.idy2, P.idz2);
}

// Initialize concentration field (global planes [zb, ze) into local buffer starting at plane zlo)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zb, const size_t ze, const size_t zlo) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = zb; z < ze; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z - zlo, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
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

static inline dim3 planeGrid(const dim3& block, int nx, int ny, int nplanes) {
    return dim3((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, nplanes);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (root) {
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

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // ---- Domain decomposition: every active rank owns >= 2 z-planes ----
    const int activeRanks = std::max(1, std::min(worldSize, static_cast<int>(nz / 2)));
    const bool active = worldRank < activeRanks;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    std::vector<int> counts(activeRanks), displs(activeRanks);
    for (int r = 0; r < activeRanks; ++r) {
        const size_t b = nz / activeRanks, rem = nz % activeRanks;
        const size_t zs = r * b + std::min<size_t>(r, rem);
        const size_t nl = b + (static_cast<size_t>(r) < rem ? 1 : 0);
        counts[r] = static_cast<int>(nl);
        displs[r] = static_cast<int>(zs);
    }

    std::vector<double> result;  // full field on root (for output)
    long long elapsedMs = 0;

    if (active) {
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Bind to a GPU by node-local rank
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));

        const int z0 = displs[rank];
        const int z1 = z0 + counts[rank];
        const int zlo = std::max(z0 - 2, 0);
        const int zhi = std::min(z1 + 2, static_cast<int>(nz));
        const int nLocalPlanes = zhi - zlo;
        const size_t localSize = static_cast<size_t>(nLocalPlanes) * planeSize;
        const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        const int upper = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        // mu is needed one plane beyond the owned range
        const int mb = std::max(z0 - 1, 0);
        const int me = std::min(z1 + 1, static_cast<int>(nz));

        // Host init of local slab including ghost planes
        if (root) printf("Initializing concentration field...\n");
        std::vector<double> hostC(localSize);
        initializeConcentration(hostC, nx, ny, nz, zlo, zhi, zlo);

        double *dC0 = nullptr, *dC1 = nullptr, *dMu = nullptr;
        CUDA_CHECK(cudaMalloc(&dC0, localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC1, localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dMu, localSize * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dC0, hostC.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC1, dC0, localSize * sizeof(double), cudaMemcpyDeviceToDevice));

        // Pinned halo buffers: [send lower | send upper | 2 x (recv lower | recv upper)].
        // Receive buffers are double-buffered: receives for step t are posted before the
        // (asynchronous) host-to-device copy of step t-1 is known to be complete.
        const size_t haloCount = 2 * planeSize;
        double* hHalo = nullptr;
        CUDA_CHECK(cudaMallocHost(&hHalo, 6 * haloCount * sizeof(double)));
        double* sendLo = hHalo;
        double* sendUp = hHalo + haloCount;

        cudaStream_t sMain, sHalo;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sHalo, cudaStreamNonBlocking));
        cudaEvent_t evBoundary, evHaloIn;
        CUDA_CHECK(cudaEventCreateWithFlags(&evBoundary, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evHaloIn, cudaEventDisableTiming));

        StencilParams P;
        P.nx = static_cast<int>(nx);
        P.ny = static_cast<int>(ny);
        P.nz = static_cast<int>(nz);
        P.zlo = zlo;
        P.idx2 = 1.0 / (dx * dx);
        P.idy2 = 1.0 / (dy * dy);
        P.idz2 = 1.0 / (dz * dz);
        const double dtD = dt * D;
        const bool unitSpacing = (P.idx2 == 1.0 && P.idy2 == 1.0 && P.idz2 == 1.0);
        // Linear part of the bulk chemical potential:
        // 4.5*((c+1)e_AA + (c-1)e_BB - 2c e_AB) + 3c = bulkA*c + bulkB
        const double bulkA = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
        const double bulkB = 4.5 * (e_AA - e_BB);

        const dim3 block(64, 4, 1);
        auto launchMu = [&](double* c, int zb, int ze) {
            if (ze <= zb) return;
            const dim3 grid = planeGrid(block, P.nx, P.ny, ze - zb);
            if (unitSpacing)
                chemicalPotentialKernel<true><<<grid, block, 0, sMain>>>(c, dMu, P, zb, gamma, bulkA, bulkB);
            else
                chemicalPotentialKernel<false><<<grid, block, 0, sMain>>>(c, dMu, P, zb, gamma, bulkA, bulkB);
        };
        auto launchUpdate = [&](double* cn, const double* co, int zb, int ze) {
            if (ze <= zb) return;
            const dim3 grid = planeGrid(block, P.nx, P.ny, ze - zb);
            if (unitSpacing)
                updateKernel<true><<<grid, block, 0, sMain>>>(cn, co, dMu, P, zb, dtD);
            else
                updateKernel<false><<<grid, block, 0, sMain>>>(cn, co, dMu, P, zb, dtD);
        };

        // Boundary planes (those sent to neighbours) and interior planes of the owned range
        const bool hasLower = lower != MPI_PROC_NULL, hasUpper = upper != MPI_PROC_NULL;
        const int bLoEnd = hasLower ? std::min(z0 + 2, z1) : z0;          // [z0, bLoEnd)
        const int bUpBeg = hasUpper ? std::max(z1 - 2, bLoEnd) : z1;      // [bUpBeg, z1)
        const int iBeg = bLoEnd, iEnd = bUpBeg;                           // interior [iBeg, iEnd)

        auto planePtr = [&](double* base, int gz) { return base + static_cast<size_t>(gz - zlo) * planeSize; };

        if (root) printf("Running Cahn-Hilliard simulation...\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        double* cold = dC0;
        double* cnew = dC1;
        // mu planes that only depend on owned c planes ("inner") can be computed while the
        // halo exchange of c is in flight; the remaining "edge" planes wait for the halos.
        const int muInBeg = hasLower ? z0 + 1 : mb;
        const int muInEnd = hasUpper ? z1 - 1 : me;

        for (int t = 0; t < iterations; ++t) {
            // Chemical potential on owned planes plus one ghost plane per side
            if (t == 0) {
                launchMu(cold, mb, me);
            } else {
                launchMu(cold, mb, muInBeg);   // inner part was launched during the previous step
                launchMu(cold, muInEnd, me);
            }

            // Boundary planes first, so they can be shipped while the interior is computed
            launchUpdate(cnew, cold, z0, bLoEnd);
            launchUpdate(cnew, cold, bUpBeg, z1);
            CUDA_CHECK(cudaEventRecord(evBoundary, sMain));
            launchUpdate(cnew, cold, iBeg, iEnd);
            // Inner chemical potential of the next step overlaps with the halo exchange
            if (t + 1 < iterations) launchMu(cnew, muInBeg, muInEnd);

            if (size > 1) {
                double* recvLo = hHalo + (2 + 2 * (t & 1)) * haloCount;
                double* recvUp = recvLo + haloCount;
                MPI_Request req[4];
                MPI_Irecv(recvLo, static_cast<int>(haloCount), MPI_DOUBLE, lower, 0, comm, &req[0]);
                MPI_Irecv(recvUp, static_cast<int>(haloCount), MPI_DOUBLE, upper, 1, comm, &req[1]);

                CUDA_CHECK(cudaStreamWaitEvent(sHalo, evBoundary, 0));
                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(sendLo, planePtr(cnew, z0), haloCount * sizeof(double),
                                               cudaMemcpyDeviceToHost, sHalo));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(sendUp, planePtr(cnew, z1 - 2), haloCount * sizeof(double),
                                               cudaMemcpyDeviceToHost, sHalo));
                CUDA_CHECK(cudaStreamSynchronize(sHalo));

                MPI_Isend(sendLo, static_cast<int>(haloCount), MPI_DOUBLE, lower, 1, comm, &req[2]);
                MPI_Isend(sendUp, static_cast<int>(haloCount), MPI_DOUBLE, upper, 0, comm, &req[3]);
                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(planePtr(cnew, z0 - 2), recvLo, haloCount * sizeof(double),
                                               cudaMemcpyHostToDevice, sHalo));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(planePtr(cnew, z1), recvUp, haloCount * sizeof(double),
                                               cudaMemcpyHostToDevice, sHalo));
                CUDA_CHECK(cudaEventRecord(evHaloIn, sHalo));
                CUDA_CHECK(cudaStreamWaitEvent(sMain, evHaloIn, 0));
            }

            // Swap buffers
            std::swap(cold, cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(sMain));
        CUDA_CHECK(cudaGetLastError());
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Gather owned planes to root if output is needed
        if (printResults || validate) {
            const size_t ownedCount = static_cast<size_t>(counts[rank]) * planeSize;
            std::vector<double> owned(ownedCount);
            CUDA_CHECK(cudaMemcpy(owned.data(), planePtr(cold, z0), ownedCount * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            if (root) result.resize(gridSize);
            // Gather in plane units (derived type) to avoid int overflow on large grids
            MPI_Datatype planeType;
            MPI_Type_contiguous(static_cast<int>(planeSize), MPI_DOUBLE, &planeType);
            MPI_Type_commit(&planeType);
            MPI_Gatherv(owned.data(), counts[rank], planeType, root ? result.data() : nullptr, counts.data(),
                        displs.data(), planeType, 0, comm);
            MPI_Type_free(&planeType);
        }

        CUDA_CHECK(cudaEventDestroy(evBoundary));
        CUDA_CHECK(cudaEventDestroy(evHaloIn));
        CUDA_CHECK(cudaStreamDestroy(sMain));
        CUDA_CHECK(cudaStreamDestroy(sHalo));
        CUDA_CHECK(cudaFreeHost(hHalo));
        CUDA_CHECK(cudaFree(dC0));
        CUDA_CHECK(cudaFree(dC1));
        CUDA_CHECK(cudaFree(dMu));
        MPI_Comm_free(&comm);
    }

    int exitCode = 0;
    if (root) {
        printf("Computation time: %lld ms\n", elapsedMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(result, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, nx, ny, nz);

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
