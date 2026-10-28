// Hybrid MPI + OpenMP + CUDA 3D 7-point stencil benchmark.
//
// Decomposition:
//   * MPI:    1D slab decomposition along z (contiguous in memory). One rank per
//             GPU; ranks on a node are bound to GPUs by node-local rank.
//             Each iteration exchanges one z-plane with each neighbour.
//   * CUDA:   Each rank keeps its two grid buffers (plus one ghost plane on each
//             side) resident on its GPU. The interior planes are computed on one
//             stream while the boundary planes' halos are staged through pinned
//             host memory and exchanged with MPI; the boundary planes are then
//             computed once the halos have arrived.
//   * OpenMP: Host-side work: parallel grid initialisation, the validation scan,
//             and per-rank thread-count control.
//
// The arithmetic (operand order, division by 7.0) is identical to the serial
// reference, so results are bit-for-bit identical to the original program.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", __FILE__, __LINE__, \
                         #call, cudaGetErrorString(err_));                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// 3D index calculation (global)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Contiguous range of global z-planes owned by one rank.
struct Slab {
    size_t z0;
    size_t nLocal;
};

static Slab slabFor(int rank, int activeRanks, size_t nz) {
    if (rank >= activeRanks) return {0, 0};
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / static_cast<size_t>(activeRanks);
    const size_t rem = nz % static_cast<size_t>(activeRanks);
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// One thread per (x, y); loops over `numPlanes` local planes starting at
// `firstLocal`. Local layout: plane 0 is the lower ghost plane, planes
// 1..nLocal are owned, plane nLocal+1 is the upper ghost plane.
__global__ void stencilPlanesKernel(const Real* __restrict__ in, Real* __restrict__ out,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const size_t z0, const size_t firstLocal, const size_t numPlanes) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const bool xyBoundary = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1);

    for (size_t k = blockIdx.z; k < numPlanes; k += gridDim.z) {
        const size_t lz = firstLocal + k;
        const size_t gz = z0 + lz - 1;  // global z index
        const size_t idx = lz * plane + y * nx + x;

        if (xyBoundary || gz == 0 || gz == nz - 1) {
            out[idx] = in[idx];
            continue;
        }

        const Real center = in[idx];
        const Real left = in[idx - 1];
        const Real right = in[idx + 1];
        const Real front = in[idx - nx];
        const Real back = in[idx + nx];
        const Real bottom = in[idx - plane];
        const Real top = in[idx + plane];

        out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

static void launchStencil(cudaStream_t stream, const Real* in, Real* out,
                          size_t nx, size_t ny, size_t nz, size_t z0,
                          size_t firstLocal, size_t numPlanes) {
    if (numPlanes == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(numPlanes, 65535)));
    stencilPlanesKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, z0, firstLocal, numPlanes);
    CUDA_CHECK(cudaGetLastError());
}

// Distributed validation; every rank must call this. Returns the same result on all ranks.
static bool validateDistributed(const std::vector<Real>& owned, bool root) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    long long localBad = 0;

    const long long n = static_cast<long long>(owned.size());
    const Real* d = owned.data();
#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(+ : localBad)
    for (long long i = 0; i < n; ++i) {
        const Real v = d[i];
        if (!std::isfinite(v)) {
            ++localBad;
            continue;
        }
        if (v < localMin) localMin = v;
        if (v > localMax) localMax = v;
    }

    long long globalBad = 0;
    Real globalMin = 0, globalMax = 0;
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 1. No NaN or Inf values
    if (globalBad > 0) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    if (root) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        if (root) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int exitCode = -1;
    for (int i = 1; i < argc && exitCode < 0; ++i) {
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
            exitCode = 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Error: grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Device binding and OpenMP thread control --------------------------
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    // ---- Decomposition -----------------------------------------------------
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const int activeRanks = static_cast<int>(std::min<size_t>(static_cast<size_t>(size), nz));
    const Slab mine = slabFor(rank, activeRanks, nz);
    const size_t z0 = mine.z0;
    const size_t nLocal = mine.nLocal;
    const int prev = (rank > 0 && rank < activeRanks) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    const int planeCount = static_cast<int>(plane);

    // Device buffers (double buffering), each with ghost planes on both sides.
    Real* dev[2] = {nullptr, nullptr};
    Real* hSendLo = nullptr;
    Real* hSendHi = nullptr;
    Real* hRecvLo = nullptr;
    Real* hRecvHi = nullptr;

    cudaStream_t sCompute;
    cudaStream_t sCopy;
    cudaEvent_t evCompute;
    cudaEvent_t evHalo;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&evCompute, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

    if (nLocal > 0) {
        const size_t bytes = (nLocal + 2) * plane * sizeof(Real);
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dev[b], bytes));
            CUDA_CHECK(cudaMemset(dev[b], 0, bytes));
        }
        const size_t haloBytesAlloc = plane * sizeof(Real);
        CUDA_CHECK(cudaMallocHost(&hSendLo, haloBytesAlloc));
        CUDA_CHECK(cudaMallocHost(&hSendHi, haloBytesAlloc));
        CUDA_CHECK(cudaMallocHost(&hRecvLo, haloBytesAlloc));
        CUDA_CHECK(cudaMallocHost(&hRecvHi, haloBytesAlloc));
    }

    // ---- Initialize --------------------------------------------------------
    if (root) printf("Initializing grid...\n");
    if (nLocal > 0) {
        std::vector<Real> hOwned(nLocal * plane);
#pragma omp parallel for schedule(static)
        for (long long lp = 0; lp < static_cast<long long>(nLocal); ++lp) {
            Real* row = hOwned.data() + static_cast<size_t>(lp) * plane;
            const size_t zBase = (z0 + static_cast<size_t>(lp)) * plane;
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t gidx = zBase + y * nx + x;
                    row[y * nx + x] = static_cast<Real>(gidx % 19) * 1.0;
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(dev[0] + plane, hOwned.data(), nLocal * plane * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    // ---- Run stencil iterations -------------------------------------------
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(evCompute, sCompute));  // nothing pending yet
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        if (nLocal > 0) {
            Real* src = dev[cur];
            Real* dst = dev[1 - cur];
            const size_t haloBytes = plane * sizeof(Real);

            // (a) Interior planes (2..nLocal-1) need no halo data: start them now.
            if (nLocal >= 3) launchStencil(sCompute, src, dst, nx, ny, nz, z0, 2, nLocal - 2);

            // (b) Stage boundary planes to host once the previous sweep has finished.
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, evCompute, 0));
            if (prev != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSendLo, src + plane, haloBytes, cudaMemcpyDeviceToHost, sCopy));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSendHi, src + nLocal * plane, haloBytes, cudaMemcpyDeviceToHost, sCopy));
            CUDA_CHECK(cudaStreamSynchronize(sCopy));

            // (c) Exchange halo planes with neighbours (MPI_PROC_NULL is a no-op).
            MPI_Request req[4];
            MPI_Isend(hSendLo, planeCount, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Irecv(hRecvHi, planeCount, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &req[1]);
            MPI_Isend(hSendHi, planeCount, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(hRecvLo, planeCount, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &req[3]);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

            // (d) Upload received halos into the ghost planes of the source buffer.
            if (prev != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(src, hRecvLo, haloBytes, cudaMemcpyHostToDevice, sCopy));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(src + (nLocal + 1) * plane, hRecvHi, haloBytes, cudaMemcpyHostToDevice, sCopy));
            CUDA_CHECK(cudaEventRecord(evHalo, sCopy));

            // (e) Boundary planes (1 and nLocal) once halos are on the device.
            CUDA_CHECK(cudaStreamWaitEvent(sCompute, evHalo, 0));
            launchStencil(sCompute, src, dst, nx, ny, nz, z0, 1, 1);
            if (nLocal >= 2) launchStencil(sCompute, src, dst, nx, ny, nz, z0, nLocal, 1);
            CUDA_CHECK(cudaEventRecord(evCompute, sCompute));
        }
        cur = 1 - cur;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    const double localSec = std::chrono::duration<double>(end - start).count();
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxSec));
    if (root) printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

    // Calculate performance metrics (interior cells per iteration)
    auto interior = [](size_t n) { return n >= 2 ? n - 2 : size_t{0}; };
    double cellUpdates = (double)(interior(nx) * interior(ny) * interior(nz)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    if (root) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // ---- Gather / validate the final grid ---------------------------------
    const int finalIdx = (iterations % 2 == 0) ? 0 : 1;
    std::vector<Real> hOwned;
    if ((printResults || validate) && nLocal > 0) {
        hOwned.resize(nLocal * plane);
        CUDA_CHECK(cudaMemcpy(hOwned.data(), dev[finalIdx] + plane, nLocal * plane * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        std::vector<Real> full;
        if (root) full.resize(gridSize);

        MPI_Datatype planeType;
        MPI_Type_contiguous(planeCount, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        std::vector<int> counts(size, 0);
        std::vector<int> displs(size, 0);
        for (int r = 0; r < size; ++r) {
            const Slab s = slabFor(r, activeRanks, nz);
            counts[r] = static_cast<int>(s.nLocal);
            displs[r] = static_cast<int>(s.z0);
        }
        MPI_Gatherv(hOwned.data(), static_cast<int>(nLocal), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) print_results(full, "Grid");
    }

    int ret = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateDistributed(hOwned, root);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    // ---- Cleanup -----------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(evCompute));
    CUDA_CHECK(cudaEventDestroy(evHalo));
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
    CUDA_CHECK(cudaFreeHost(hSendLo));
    CUDA_CHECK(cudaFreeHost(hSendHi));
    CUDA_CHECK(cudaFreeHost(hRecvLo));
    CUDA_CHECK(cudaFreeHost(hRecvHi));
    CUDA_CHECK(cudaFree(dev[0]));
    CUDA_CHECK(cudaFree(dev[1]));

    MPI_Finalize();
    return ret;
}
