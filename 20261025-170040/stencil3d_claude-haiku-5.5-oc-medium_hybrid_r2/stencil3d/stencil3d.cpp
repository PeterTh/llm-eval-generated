// Hybrid MPI + OpenMP + CUDA 3D stencil benchmark.
//
// Decomposition: the global grid is split into contiguous z-slabs, one per MPI
// rank (min(nranks, nz) ranks own planes). Each rank drives one GPU, which holds
// its slab plus one ghost plane on each side (double-buffered).
//
// Per iteration on each rank:
//   * interior planes (no halo dependence) are computed on the GPU compute stream,
//   * boundary planes are staged to pinned host memory, halo planes are exchanged
//     with the z-neighbours via MPI, and copied back to the device ghost planes
//     on a separate copy stream while the interior kernel is still running,
//   * the two boundary planes are then computed once the ghosts have arrived.
// OpenMP is used for host-side work that is order-independent: slab
// initialization and the NaN/Inf/min/max validation scan. Rank 0 gathers the
// final grid and performs printing, hashing and validation exactly as before.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kTagUp = 0;    // message travelling to a higher-z rank
constexpr int kTagDown = 1;  // message travelling to a lower-z rank

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Owned z-range of a rank: global planes [z0, z0 + lnz).
struct Slab {
    size_t z0;
    size_t lnz;
};

Slab slabFor(int rank, size_t activeRanks, size_t nz) {
    if (static_cast<size_t>(rank) >= activeRanks) return {nz, 0};
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / activeRanks;
    const size_t rem = nz % activeRanks;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

}  // namespace

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Host-side initialization of the owned planes [z0, z0+lnz) of the global grid.
// Values depend only on the global index, exactly as in the serial version.
void initializeSlab(Real* slab, const size_t nx, const size_t ny, const size_t z0, const size_t lnz) {
    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < lnz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                slab[idx3(x, y, z, nx, ny)] = (idx3(x, y, z0 + z, nx, ny) % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil on a range of local planes of one slab.
// Local plane zl (0 = lower ghost, 1..lnz = owned, lnz+1 = upper ghost) corresponds
// to global plane z0 + zl - 1. Global boundary cells are copied unchanged.
// Summation order matches the original code exactly so results are bit-identical.
__global__ void stencilPlanesKernel(const Real* __restrict__ in, Real* __restrict__ out,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const size_t z0, const size_t zBegin, const size_t zCount) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    for (size_t p = blockIdx.z; p < zCount; p += gridDim.z) {
        const size_t zl = zBegin + p;
        const size_t idx = zl * plane + y * nx + x;
        const size_t gz = z0 + zl - 1;

        const bool interior = (x >= 1) && (x + 1 < nx) && (y >= 1) && (y + 1 < ny) &&
                              (gz >= 1) && (gz + 1 < nz);
        if (interior) {
            const Real center = in[idx];
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            const Real bottom = in[idx - plane];
            const Real top = in[idx + plane];
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        } else {
            out[idx] = in[idx];
        }
    }
}

static void launchStencil(const Real* in, Real* out, const size_t nx, const size_t ny, const size_t nz,
                          const size_t z0, const size_t zBegin, const size_t zCount, cudaStream_t stream) {
    if (zCount == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(zCount, 65535)));
    stencilPlanesKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, z0, zBegin, zCount);
    CUDA_CHECK(cudaGetLastError());
}

// Validation of the final grid (run on rank 0). The NaN/Inf and min/max scans are
// order-independent, so they are parallelized with OpenMP reductions.
bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool hasBad = false;
    const long long n = static_cast<long long>(grid.size());
    #pragma omp parallel for reduction(||:hasBad)
    for (long long i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) hasBad = true;
    }
    if (hasBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (long long i = 0; i < n; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

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

static int runBenchmark(int argc, char** argv) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    // Bind each rank to a GPU (round-robin over the GPUs of its node).
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);
    MPI_Comm_free(&shmComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Share host cores between the ranks on this node.
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t planeBytes = plane * sizeof(Real);
    const size_t activeRanks = std::min<size_t>(static_cast<size_t>(nranks), nz);
    const Slab slab = slabFor(rank, activeRanks, nz);
    const size_t lnz = slab.lnz;
    const size_t z0 = slab.z0;

    const int down = (lnz > 0 && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (lnz > 0 && static_cast<size_t>(rank) + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    const bool hasDown = (down != MPI_PROC_NULL);
    const bool hasUp = (up != MPI_PROC_NULL);

    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Device buffers: double-buffered slab with one ghost plane on each side.
    if (root) printf("Initializing grid...\n");
    Real* dBuf[2] = {nullptr, nullptr};
    for (Real*& b : dBuf) {
        CUDA_CHECK(cudaMalloc(&b, (lnz + 2) * planeBytes));
        CUDA_CHECK(cudaMemset(b, 0, (lnz + 2) * planeBytes));
    }
    {
        std::vector<Real> hOwned(lnz * plane);
        initializeSlab(hOwned.data(), nx, ny, z0, lnz);
        if (lnz > 0) {
            CUDA_CHECK(cudaMemcpy(dBuf[0] + plane, hOwned.data(), lnz * planeBytes, cudaMemcpyHostToDevice));
        }
    }

    // Pinned staging buffers for halo exchange.
    Real *hSendLo, *hSendHi, *hRecvLo, *hRecvHi;
    CUDA_CHECK(cudaMallocHost(&hSendLo, planeBytes));
    CUDA_CHECK(cudaMallocHost(&hSendHi, planeBytes));
    CUDA_CHECK(cudaMallocHost(&hRecvLo, planeBytes));
    CUDA_CHECK(cudaMallocHost(&hRecvHi, planeBytes));

    cudaStream_t sCompute, sCopy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    cudaEvent_t evGhost, evDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&evGhost, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evDone, cudaEventDisableTiming));

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = dBuf[iter % 2];
        Real* out = dBuf[1 - iter % 2];

        // The copy stream may only touch `in` after the previous iteration finished writing it.
        CUDA_CHECK(cudaStreamWaitEvent(sCopy, evDone, 0));

        // 1. Interior planes do not depend on halo data: compute them while the halo moves.
        if (lnz >= 3) {
            launchStencil(in, out, nx, ny, nz, z0, 2, lnz - 2, sCompute);
        }

        // 2. Stage the owned boundary planes to host memory.
        if (hasDown) {
            CUDA_CHECK(cudaMemcpyAsync(hSendLo, in + plane * 1, planeBytes, cudaMemcpyDeviceToHost, sCopy));
        }
        if (hasUp) {
            CUDA_CHECK(cudaMemcpyAsync(hSendHi, in + plane * lnz, planeBytes, cudaMemcpyDeviceToHost, sCopy));
        }
        CUDA_CHECK(cudaStreamSynchronize(sCopy));

        // 3. Exchange halo planes with z-neighbours.
        MPI_Request reqs[4];
        int nreq = 0;
        if (hasDown) {
            MPI_Irecv(hRecvLo, static_cast<int>(plane), MPI_DOUBLE, down, kTagUp, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hSendLo, static_cast<int>(plane), MPI_DOUBLE, down, kTagDown, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (hasUp) {
            MPI_Irecv(hRecvHi, static_cast<int>(plane), MPI_DOUBLE, up, kTagDown, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(hSendHi, static_cast<int>(plane), MPI_DOUBLE, up, kTagUp, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // 4. Bring received halos to the device ghost planes.
        if (hasDown) {
            CUDA_CHECK(cudaMemcpyAsync(in + 0, hRecvLo, planeBytes, cudaMemcpyHostToDevice, sCopy));
        }
        if (hasUp) {
            CUDA_CHECK(cudaMemcpyAsync(in + plane * (lnz + 1), hRecvHi, planeBytes, cudaMemcpyHostToDevice, sCopy));
        }
        CUDA_CHECK(cudaEventRecord(evGhost, sCopy));
        CUDA_CHECK(cudaStreamWaitEvent(sCompute, evGhost, 0));

        // 5. Boundary planes (they read the ghost planes).
        if (lnz >= 3) {
            launchStencil(in, out, nx, ny, nz, z0, 1, 1, sCompute);
            launchStencil(in, out, nx, ny, nz, z0, lnz, 1, sCompute);
        } else if (lnz > 0) {
            launchStencil(in, out, nx, ny, nz, z0, 1, lnz, sCompute);
        }
        CUDA_CHECK(cudaEventRecord(evDone, sCompute));
    }

    CUDA_CHECK(cudaStreamSynchronize(sCompute));
    CUDA_CHECK(cudaStreamSynchronize(sCopy));
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", max_ms);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0. Slabs are contiguous in z, so gathering
    // plane-by-plane reproduces the serial memory layout exactly.
    const Real* finalDev = (iterations % 2 == 0) ? dBuf[0] : dBuf[1];
    std::vector<Real> hLocal(lnz * plane);
    if (lnz > 0) {
        CUDA_CHECK(cudaMemcpy(hLocal.data(), finalDev + plane, lnz * planeBytes, cudaMemcpyDeviceToHost));
    }

    std::vector<Real> finalGrid;
    std::vector<int> counts(nranks, 0);
    std::vector<int> displs(nranks, 0);
    for (int r = 0; r < nranks; ++r) {
        const Slab sr = slabFor(r, activeRanks, nz);
        counts[r] = static_cast<int>(sr.lnz);
        displs[r] = static_cast<int>(sr.z0);
    }
    if (root) finalGrid.resize(gridSize);
    MPI_Gatherv(hLocal.data(), static_cast<int>(lnz), planeType,
                root ? finalGrid.data() : nullptr, counts.data(), displs.data(), planeType,
                0, MPI_COMM_WORLD);

    // Print results for external validation
    if (root && printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaEventDestroy(evGhost));
    CUDA_CHECK(cudaEventDestroy(evDone));
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
    CUDA_CHECK(cudaFreeHost(hSendLo));
    CUDA_CHECK(cudaFreeHost(hSendHi));
    CUDA_CHECK(cudaFreeHost(hRecvLo));
    CUDA_CHECK(cudaFreeHost(hRecvHi));
    CUDA_CHECK(cudaFree(dBuf[0]));
    CUDA_CHECK(cudaFree(dBuf[1]));
    MPI_Type_free(&planeType);

    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
