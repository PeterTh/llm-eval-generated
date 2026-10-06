// Hybrid MPI + OpenMP + CUDA Cahn-Hilliard benchmark.
//
// Parallelization:
//  - MPI: 1D slab decomposition along z, one rank per GPU. Each rank keeps two
//    ghost planes of c on each side, so a single halo exchange per time step
//    suffices (mu is recomputed on the first ghost plane locally).
//  - CUDA: stencil kernels for chemical potential and update; the halo exchange
//    (D2H copy -> MPI -> H2D copy) is overlapped with the interior kernels.
//  - OpenMP: host-side reductions/validation of the gathered field.
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

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// Physical parameters
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma_ = 0.5;
constexpr double D = 1.0;

constexpr int BX = 32;
constexpr int BY = 8;

// Laplacian with clamped boundary conditions. `p` points to the cell (x, y, z)
// of a local slab; z clamping is decided by the global z coordinate.
__device__ __forceinline__ double laplacian(const double* __restrict__ p, const int x, const int y,
                                            const long long gz, const int nx, const int ny,
                                            const long long nz, const long long plane) {
    const double cc = __ldg(p);
    const double cxp = __ldg(p + ((x < nx - 1) ? 1 : 0));
    const double cxn = __ldg(p - ((x > 0) ? 1 : 0));
    const double cyp = __ldg(p + ((y < ny - 1) ? nx : 0));
    const double cyn = __ldg(p - ((y > 0) ? nx : 0));
    const double czp = __ldg(p + ((gz < nz - 1) ? plane : 0));
    const double czn = __ldg(p - ((gz > 0) ? plane : 0));

    const double cxx = (cxp + cxn - 2.0 * cc) / (dx * dx);
    const double cyy = (cyp + cyn - 2.0 * cc) / (dy * dy);
    const double czz = (czp + czn - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// c and mu point to local plane 0; computes mu for local planes [zb, ze).
__global__ void __launch_bounds__(BX * BY)
chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu, const int nx,
                        const int ny, const long long nz, const long long z0, const int zb,
                        const int ze) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long long plane = (long long)nx * ny;
    for (int z = zb + (int)blockIdx.z; z < ze; z += gridDim.z) {
        const long long idx = (long long)z * plane + (long long)y * nx + x;
        const double cv = __ldg(c + idx);
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv +
                  cv * cv * cv - gamma_ * laplacian(c + idx, x, y, z0 + z, nx, ny, nz, plane);
    }
}

// Updates cnew for local planes [zb, ze).
__global__ void __launch_bounds__(BX * BY)
updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
             const double* __restrict__ mu, const int nx, const int ny, const long long nz,
             const long long z0, const int zb, const int ze) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long long plane = (long long)nx * ny;
    for (int z = zb + (int)blockIdx.z; z < ze; z += gridDim.z) {
        const long long idx = (long long)z * plane + (long long)y * nx + x;
        cnew[idx] = __ldg(cold + idx) +
                    dt * D * laplacian(mu + idx, x, y, z0 + z, nx, ny, nz, plane);
    }
}

// Initialize local slab (planes [0, nzl)) of the concentration field.
__global__ void initKernel(double* __restrict__ c, const int nx, const int ny, const size_t vol,
                           const long long z0, const int nzl) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = (size_t)nx * ny;
    for (int z = (int)blockIdx.z; z < nzl; z += gridDim.z) {
        const size_t linear_id = (size_t)(z0 + z) * plane + (size_t)y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[(size_t)z * plane + (size_t)y * nx + x] = -1.0 + 2.0 * pseudo;
    }
}

static dim3 gridFor(const int nx, const int ny, const int nplanes) {
    return dim3((nx + BX - 1) / BX, (ny + BY - 1) / BY, std::max(1, std::min(nplanes, 65535)));
}

static void launchMu(const double* c, double* mu, int nx, int ny, long long nz, long long z0,
                     int zb, int ze, cudaStream_t s) {
    if (ze <= zb) return;
    chemicalPotentialKernel<<<gridFor(nx, ny, ze - zb), dim3(BX, BY), 0, s>>>(c, mu, nx, ny, nz,
                                                                             z0, zb, ze);
}

static void launchUpdate(double* cnew, const double* cold, const double* mu, int nx, int ny,
                         long long nz, long long z0, int zb, int ze, cudaStream_t s) {
    if (ze <= zb) return;
    updateKernel<<<gridFor(nx, ny, ze - zb), dim3(BX, BY), 0, s>>>(cnew, cold, mu, nx, ny, nz, z0,
                                                                  zb, ze);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for reduction(|| : bad)
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
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
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

    const size_t gridSize = nx * ny * nz;
    const size_t planeSize = nx * ny;

    // Every active rank needs at least 2 planes (halo depth 2).
    const int nActive = (int)std::max<size_t>(1, std::min<size_t>(worldSize, nz / 2));
    const bool active = worldRank < nActive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    // Slab decomposition along z
    std::vector<int> planesOf(nActive), firstOf(nActive);
    for (int r = 0, off = 0; r < nActive; ++r) {
        planesOf[r] = (int)(nz / nActive + ((size_t)r < nz % nActive ? 1 : 0));
        firstOf[r] = off;
        off += planesOf[r];
    }

    // One GPU per rank within a node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int nDev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nDev));
        CUDA_CHECK(cudaSetDevice(localRank % nDev));
    }

    std::vector<double> cold;
    long long durationMs = 0;

    if (root) printf("Initializing concentration field...\n");

    if (active) {
        const int rank = worldRank;
        const int nzl = planesOf[rank];
        const long long z0 = firstOf[rank];
        const bool hasLower = rank > 0;
        const bool hasUpper = rank < nActive - 1;
        const int lower = rank - 1, upper = rank + 1;
        const int inx = (int)nx, iny = (int)ny;
        const long long gnz = (long long)nz;

        const size_t localAlloc = (size_t)(nzl + 4) * planeSize;
        const size_t haloCount = 2 * planeSize;
        double *dA, *dB, *dMu;
        CUDA_CHECK(cudaMalloc(&dA, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dMu, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(dA, 0, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(dB, 0, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(dMu, 0, localAlloc * sizeof(double)));
        // Pointers to local plane 0 (two ghost planes below)
        double* dCold = dA + 2 * planeSize;
        double* dCnew = dB + 2 * planeSize;
        double* dMu0 = dMu + 2 * planeSize;

        double *hSendLo, *hSendHi, *hRecvLo, *hRecvHi;
        CUDA_CHECK(cudaMallocHost(&hSendLo, haloCount * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hSendHi, haloCount * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hRecvLo, haloCount * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hRecvHi, haloCount * sizeof(double)));

        cudaStream_t sComp, sComm;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
        cudaEvent_t evReady, evHalo;
        CUDA_CHECK(cudaEventCreateWithFlags(&evReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

        initKernel<<<gridFor(inx, iny, nzl), dim3(BX, BY), 0, sComp>>>(dCold, inx, iny, gridSize,
                                                                       z0, nzl);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sComp));

        // Plane ranges. mu is needed on [muLo, muHi); the interior part
        // [muIntLo, muIntHi) does not depend on ghost data.
        const int muLo = hasLower ? -1 : 0;
        const int muHi = hasUpper ? nzl + 1 : nzl;
        int muIntLo = hasLower ? 1 : 0;
        int muIntHi = hasUpper ? nzl - 1 : nzl;
        if (muIntLo >= muIntHi) muIntLo = muIntHi = 0;
        // Update interior: needs only interior mu.
        int upIntLo = hasLower ? 2 : 0;
        int upIntHi = hasUpper ? nzl - 2 : nzl;
        if (muIntLo >= muIntHi || upIntLo >= upIntHi) upIntLo = upIntHi = 0;

        const bool exchange = hasLower || hasUpper;

        if (root) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            if (exchange) {
                // Send boundary planes of cold once it is complete
                CUDA_CHECK(cudaEventRecord(evReady, sComp));
                CUDA_CHECK(cudaStreamWaitEvent(sComm, evReady, 0));
                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(hSendLo, dCold, haloCount * sizeof(double),
                                               cudaMemcpyDeviceToHost, sComm));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(hSendHi, dCold + (size_t)(nzl - 2) * planeSize,
                                               haloCount * sizeof(double), cudaMemcpyDeviceToHost,
                                               sComm));
            }

            // Interior work, overlapped with the halo exchange
            launchMu(dCold, dMu0, inx, iny, gnz, z0, muIntLo, muIntHi, sComp);
            launchUpdate(dCnew, dCold, dMu0, inx, iny, gnz, z0, upIntLo, upIntHi, sComp);

            if (exchange) {
                MPI_Request reqs[4];
                int nreq = 0;
                // Send data staged; also guarantees the previous step's H2D
                // copies out of the receive buffers have completed.
                CUDA_CHECK(cudaStreamSynchronize(sComm));
                if (hasLower)
                    MPI_Irecv(hRecvLo, (int)haloCount, MPI_DOUBLE, lower, 1, comm, &reqs[nreq++]);
                if (hasUpper)
                    MPI_Irecv(hRecvHi, (int)haloCount, MPI_DOUBLE, upper, 0, comm, &reqs[nreq++]);
                if (hasLower)
                    MPI_Isend(hSendLo, (int)haloCount, MPI_DOUBLE, lower, 0, comm, &reqs[nreq++]);
                if (hasUpper)
                    MPI_Isend(hSendHi, (int)haloCount, MPI_DOUBLE, upper, 1, comm, &reqs[nreq++]);
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(dCold - 2 * planeSize, hRecvLo,
                                               haloCount * sizeof(double), cudaMemcpyHostToDevice,
                                               sComm));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(dCold + (size_t)nzl * planeSize, hRecvHi,
                                               haloCount * sizeof(double), cudaMemcpyHostToDevice,
                                               sComm));
                CUDA_CHECK(cudaEventRecord(evHalo, sComm));
                CUDA_CHECK(cudaStreamWaitEvent(sComp, evHalo, 0));
            }

            // Boundary mu planes (ghost-dependent)
            if (muIntLo == muIntHi) {
                launchMu(dCold, dMu0, inx, iny, gnz, z0, muLo, muHi, sComp);
            } else {
                launchMu(dCold, dMu0, inx, iny, gnz, z0, muLo, muIntLo, sComp);
                launchMu(dCold, dMu0, inx, iny, gnz, z0, muIntHi, muHi, sComp);
            }
            // Boundary update planes
            if (upIntLo == upIntHi) {
                launchUpdate(dCnew, dCold, dMu0, inx, iny, gnz, z0, 0, nzl, sComp);
            } else {
                launchUpdate(dCnew, dCold, dMu0, inx, iny, gnz, z0, 0, upIntLo, sComp);
                launchUpdate(dCnew, dCold, dMu0, inx, iny, gnz, z0, upIntHi, nzl, sComp);
            }

            // Swap buffers
            std::swap(dCold, dCnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(sComp));
        CUDA_CHECK(cudaGetLastError());
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        long long maxDurationMs = 0;
        MPI_Reduce(&durationMs, &maxDurationMs, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, comm);
        if (root) durationMs = maxDurationMs;

        // Gather the final field on rank 0 when needed
        if (printResults || validate) {
            MPI_Datatype planeType;
            MPI_Type_contiguous((int)planeSize, MPI_DOUBLE, &planeType);
            MPI_Type_commit(&planeType);
            double* hLocal;
            CUDA_CHECK(cudaMallocHost(&hLocal, (size_t)nzl * planeSize * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(hLocal, dCold, (size_t)nzl * planeSize * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            if (root) cold.resize(gridSize);
            MPI_Gatherv(hLocal, nzl, planeType, root ? cold.data() : nullptr, planesOf.data(),
                        firstOf.data(), planeType, 0, comm);
            CUDA_CHECK(cudaFreeHost(hLocal));
            MPI_Type_free(&planeType);
        }

        CUDA_CHECK(cudaEventDestroy(evReady));
        CUDA_CHECK(cudaEventDestroy(evHalo));
        CUDA_CHECK(cudaStreamDestroy(sComp));
        CUDA_CHECK(cudaStreamDestroy(sComm));
        CUDA_CHECK(cudaFreeHost(hSendLo));
        CUDA_CHECK(cudaFreeHost(hSendHi));
        CUDA_CHECK(cudaFreeHost(hRecvLo));
        CUDA_CHECK(cudaFreeHost(hRecvHi));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dMu));
        MPI_Comm_free(&comm);
    }

    int exitCode = 0;
    if (root) {
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
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
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
