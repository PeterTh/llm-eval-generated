// 3D 7-point stencil benchmark -- hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy
//   * MPI   : 1D domain decomposition in Z (contiguous slabs of XY planes).
//             Each rank owns [z0, z0 + lnz) planes plus one halo plane on each
//             side.  Halo exchange is non-blocking and overlapped with the
//             interior stencil update on the GPU.  CUDA-aware MPI is used when
//             the runtime supports it, otherwise halos are staged through
//             pinned host memory.
//   * CUDA  : Every rank drives one GPU (local rank % #devices).  The stencil
//             is computed by a register-sliding-window kernel (one thread per
//             XY column, marching in Z) on two streams: one for the interior
//             planes and one for halo transfers + the two halo-dependent planes.
//   * OpenMP: All host-side bulk work (grid initialization, boundary/halo
//             packing, validation reductions) is multi-threaded.
//
// Semantics are identical to the original serial version: the same IEEE
// operations in the same order, boundaries copied unchanged, the final grid
// gathered on rank 0 for result printing and validation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                                    \
    do {                                                                                   \
        int err__ = (call);                                                                \
        if (err__ != MPI_SUCCESS) {                                                        \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, __LINE__, \
                    err__);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Host: initialize the local slab [z0, z0+lnz) exactly like the serial code
// (grid[idx] = (idx % 19) * 1.0 with the *global* linear index).
// The local buffer has lnz + 2 planes; plane k holds global plane z0 - 1 + k.
// ---------------------------------------------------------------------------
void initializeGrid(std::vector<Real>& local, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lnz) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t k = 0; k < lnz; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = z0 + k;
            Real* row = local.data() + (k + 1) * plane + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, gz, nx, ny);
                row[x] = (idx % 19) * 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA: 7-point stencil on planes [kbeg, kend) of the local buffer.  Only
// interior points are written; boundary values are left untouched in the
// output buffer (they were copied once at setup, matching the serial code's
// "output[idx] = input[idx]" for boundary cells).
// Each thread owns one (x, y) column and marches through a chunk of Z planes
// keeping bottom/center/top in registers.
// ---------------------------------------------------------------------------
namespace {
constexpr int BX = 32;
constexpr int BY = 8;
}

__global__ void __launch_bounds__(BX * BY)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
              const int nx, const int ny, const size_t plane,
              const int kbeg, const int kend, const int zchunk) {
    const int x = 1 + blockIdx.x * BX + threadIdx.x;
    const int y = 1 + blockIdx.y * BY + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    int k0 = kbeg + blockIdx.z * zchunk;
    int k1 = k0 + zchunk;
    if (k1 > kend) k1 = kend;
    if (k0 >= k1) return;

    const size_t col = static_cast<size_t>(y) * nx + x;
    const Real* p = in + static_cast<size_t>(k0) * plane + col;

    Real bottom = p[-static_cast<ptrdiff_t>(plane)];
    Real center = p[0];

    for (int k = k0; k < k1; ++k) {
        const Real top = p[plane];
        const Real left = p[-1];
        const Real right = p[1];
        const Real front = p[-nx];
        const Real back = p[nx];

        // Same evaluation order as the serial code.
        out[static_cast<size_t>(k) * plane + col] =
            (center + left + right + front + back + bottom + top) / 7.0;

        bottom = center;
        center = top;
        p += plane;
    }
}

static void launchStencil(const Real* in, Real* out, const int nx, const int ny,
                          const size_t plane, const int kbeg, const int kend,
                          const int numSMs, cudaStream_t stream) {
    if (kend <= kbeg || nx < 3 || ny < 3) return;
    const int nplanes = kend - kbeg;
    const unsigned gx = (static_cast<unsigned>(nx - 2) + BX - 1) / BX;
    const unsigned gy = (static_cast<unsigned>(ny - 2) + BY - 1) / BY;
    // Choose the Z chunk so that enough blocks are resident to fill the GPU
    // while keeping each thread's Z march as long as possible.
    const long long targetBlocks = 8LL * numSMs;
    long long zchunk = (static_cast<long long>(nplanes) * gx * gy) / targetBlocks;
    if (zchunk < 1) zchunk = 1;
    if (zchunk > nplanes) zchunk = nplanes;
    const unsigned gz = static_cast<unsigned>((nplanes + zchunk - 1) / zchunk);
    dim3 block(BX, BY, 1);
    dim3 grid(gx, gy, gz);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, plane, kbeg, kend,
                                              static_cast<int>(zchunk));
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host validation (rank 0 only, on the gathered grid) -- OpenMP reductions.
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const size_t n = grid.size();
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
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

// Determine whether device pointers can be handed to MPI directly.
static bool detectCudaAwareMpi() {
    if (const char* env = getenv("STENCIL3D_CUDA_AWARE_MPI")) {
        return atoi(env) != 0;
    }
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

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

    // ------------------------------------------------------------------
    // GPU selection: one device per rank, round-robin over node-local ranks.
    // ------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    const bool cudaAwareMpi = detectCudaAwareMpi();

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d per node), OpenMP threads/rank: %d, GPUs/node: %d, "
               "CUDA-aware MPI: %s\n",
               nranks, localSize, omp_get_max_threads(), numDevices,
               cudaAwareMpi ? "yes" : "no");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // ------------------------------------------------------------------
    // 1D decomposition along Z.  Rank r owns planes [z0[r], z0[r] + lnz[r]).
    // ------------------------------------------------------------------
    std::vector<size_t> zStart(nranks + 1, 0);
    for (int r = 0; r < nranks; ++r) {
        zStart[r + 1] = zStart[r] + nz / nranks + (static_cast<size_t>(r) < nz % nranks ? 1 : 0);
    }
    const size_t z0 = zStart[rank];
    const size_t lnz = zStart[rank + 1] - zStart[rank];
    // Neighbors: nearest non-empty rank below / above (MPI_PROC_NULL if none).
    int prevRank = MPI_PROC_NULL, nextRank = MPI_PROC_NULL;
    for (int r = rank - 1; r >= 0; --r) {
        if (zStart[r + 1] > zStart[r]) { prevRank = r; break; }
    }
    for (int r = rank + 1; r < nranks; ++r) {
        if (zStart[r + 1] > zStart[r]) { nextRank = r; break; }
    }
    if (lnz == 0) { prevRank = MPI_PROC_NULL; nextRank = MPI_PROC_NULL; }
    const bool haveLo = (prevRank != MPI_PROC_NULL);
    const bool haveHi = (nextRank != MPI_PROC_NULL);

    // Planes (local buffer indices, 1-based for owned planes) that must be
    // updated: global z in [1, nz-2] intersected with the owned range.
    int kbeg = 1, kend = 0;  // empty by default
    if (lnz > 0 && nz >= 3 && nx >= 3 && ny >= 3) {
        const size_t gzBeg = std::max<size_t>(z0, 1);
        const size_t gzEnd = std::min<size_t>(z0 + lnz, nz - 1);  // exclusive
        if (gzEnd > gzBeg) {
            kbeg = static_cast<int>(gzBeg - z0 + 1);
            kend = static_cast<int>(gzEnd - z0 + 1);
        }
    }
    // Split into halo-independent interior planes and halo-dependent edge planes.
    const bool loEdge = haveLo && kbeg == 1 && kend > kbeg;            // plane 1 needs plane 0
    const bool hiEdge = haveHi && kend == static_cast<int>(lnz) + 1 && kend > kbeg;  // plane lnz needs plane lnz+1
    const int ibeg = kbeg + (loEdge ? 1 : 0);
    const int iend = kend - (hiEdge ? 1 : 0);

    // ------------------------------------------------------------------
    // Allocate and initialize the local slab (host, OpenMP), then upload.
    // ------------------------------------------------------------------
    const size_t localElems = (lnz + 2) * plane;
    std::vector<Real> hostSlab(localElems, 0.0);

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(hostSlab, nx, ny, z0, lnz);

    Real* dA = nullptr;  // corresponds to grid1
    Real* dB = nullptr;  // corresponds to grid2
    CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dB, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dA, hostSlab.data(), localElems * sizeof(Real), cudaMemcpyHostToDevice));
    // grid2 starts zeroed (as in the serial code); its boundary cells receive
    // the input's boundary values on the first iteration, which we reproduce
    // with a single device-to-device copy (interior cells are overwritten by
    // the stencil anyway).
    CUDA_CHECK(cudaMemset(dB, 0, localElems * sizeof(Real)));
    if (iterations > 0) {
        CUDA_CHECK(cudaMemcpy(dB, dA, localElems * sizeof(Real), cudaMemcpyDeviceToDevice));
    }

    // Halo staging buffers (pinned) for non-CUDA-aware MPI.
    Real* hSendLo = nullptr; Real* hSendHi = nullptr;
    Real* hRecvLo = nullptr; Real* hRecvHi = nullptr;
    if (!cudaAwareMpi && (haveLo || haveHi)) {
        CUDA_CHECK(cudaMallocHost(&hSendLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&hSendHi, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&hRecvLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&hRecvHi, plane * sizeof(Real)));
    }

    cudaStream_t compStream, commStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
    cudaEvent_t evPack, evHalo, evCompute;
    CUDA_CHECK(cudaEventCreateWithFlags(&evPack, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evCompute, cudaEventDisableTiming));

    MPI_Datatype planeType;
    MPI_CHECK(MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType));
    MPI_CHECK(MPI_Type_commit(&planeType));

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // ------------------------------------------------------------------
    // Run stencil iterations
    // ------------------------------------------------------------------
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    Real* in = dA;
    Real* out = dB;
    const int TAG_LO = 100, TAG_HI = 101;  // TAG_LO: data travelling downward, TAG_HI: upward

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request reqs[4];
        int nreq = 0;

        Real* loHaloDst = in;                                    // plane 0
        Real* hiHaloDst = in + (lnz + 1) * plane;                // plane lnz+1
        const Real* loSendSrc = in + plane;                      // plane 1
        const Real* hiSendSrc = in + lnz * plane;                // plane lnz

        if (haveLo || haveHi) {
            // The pinned receive buffers are reused: make sure the previous
            // iteration's host-to-device halo copy has drained them before MPI
            // may write into them again.
            if (!cudaAwareMpi) CUDA_CHECK(cudaEventSynchronize(evHalo));
            // Receive halos.
            if (haveLo) {
                MPI_CHECK(MPI_Irecv(cudaAwareMpi ? loHaloDst : hRecvLo, 1, planeType, prevRank,
                                    TAG_HI, MPI_COMM_WORLD, &reqs[nreq++]));
            }
            if (haveHi) {
                MPI_CHECK(MPI_Irecv(cudaAwareMpi ? hiHaloDst : hRecvHi, 1, planeType, nextRank,
                                    TAG_LO, MPI_COMM_WORLD, &reqs[nreq++]));
            }
            // The send planes were produced by the previous iteration's edge kernels.
            CUDA_CHECK(cudaStreamWaitEvent(commStream, evCompute, 0));
            if (!cudaAwareMpi) {
                if (haveLo) {
                    CUDA_CHECK(cudaMemcpyAsync(hSendLo, loSendSrc, plane * sizeof(Real),
                                               cudaMemcpyDeviceToHost, commStream));
                }
                if (haveHi) {
                    CUDA_CHECK(cudaMemcpyAsync(hSendHi, hiSendSrc, plane * sizeof(Real),
                                               cudaMemcpyDeviceToHost, commStream));
                }
            }
            CUDA_CHECK(cudaEventRecord(evPack, commStream));
        }

        // Interior planes (independent of halos) overlap with communication.
        launchStencil(in, out, static_cast<int>(nx), static_cast<int>(ny), plane, ibeg, iend,
                      numSMs, compStream);

        if (haveLo || haveHi) {
            CUDA_CHECK(cudaEventSynchronize(evPack));
            if (haveLo) {
                MPI_CHECK(MPI_Isend(cudaAwareMpi ? loSendSrc : hSendLo, 1, planeType, prevRank,
                                    TAG_LO, MPI_COMM_WORLD, &reqs[nreq++]));
            }
            if (haveHi) {
                MPI_CHECK(MPI_Isend(cudaAwareMpi ? hiSendSrc : hSendHi, 1, planeType, nextRank,
                                    TAG_HI, MPI_COMM_WORLD, &reqs[nreq++]));
            }
            MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

            if (!cudaAwareMpi) {
                if (haveLo) {
                    CUDA_CHECK(cudaMemcpyAsync(loHaloDst, hRecvLo, plane * sizeof(Real),
                                               cudaMemcpyHostToDevice, commStream));
                }
                if (haveHi) {
                    CUDA_CHECK(cudaMemcpyAsync(hiHaloDst, hRecvHi, plane * sizeof(Real),
                                               cudaMemcpyHostToDevice, commStream));
                }
            }
            CUDA_CHECK(cudaEventRecord(evHalo, commStream));
            CUDA_CHECK(cudaStreamWaitEvent(compStream, evHalo, 0));

            // Halo-dependent edge planes.
            if (loEdge) {
                launchStencil(in, out, static_cast<int>(nx), static_cast<int>(ny), plane, 1, 2,
                              numSMs, compStream);
            }
            if (hiEdge) {
                launchStencil(in, out, static_cast<int>(nx), static_cast<int>(ny), plane,
                              static_cast<int>(lnz), static_cast<int>(lnz) + 1, numSMs,
                              compStream);
            }
        }
        CUDA_CHECK(cudaEventRecord(evCompute, compStream));

        std::swap(in, out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ------------------------------------------------------------------
    // Gather the final grid on rank 0 (same buffer selection as the serial
    // code: grid1 for an even iteration count, grid2 otherwise).
    // ------------------------------------------------------------------
    const Real* dFinal = (iterations % 2 == 0) ? dA : dB;
    CUDA_CHECK(cudaMemcpy(hostSlab.data() + plane, dFinal + plane, lnz * plane * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    std::vector<Real> finalGrid;
    std::vector<int> counts, displs;
    if (rank == 0) {
        finalGrid.resize(gridSize);
        counts.resize(nranks);
        displs.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(zStart[r + 1] - zStart[r]);
            displs[r] = static_cast<int>(zStart[r]);
        }
    }
    MPI_CHECK(MPI_Gatherv(hostSlab.data() + plane, static_cast<int>(lnz), planeType,
                          rank == 0 ? finalGrid.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displs.data() : nullptr, planeType, 0, MPI_COMM_WORLD));

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    MPI_CHECK(MPI_Type_free(&planeType));
    CUDA_CHECK(cudaEventDestroy(evPack));
    CUDA_CHECK(cudaEventDestroy(evHalo));
    CUDA_CHECK(cudaEventDestroy(evCompute));
    CUDA_CHECK(cudaStreamDestroy(compStream));
    CUDA_CHECK(cudaStreamDestroy(commStream));
    if (hSendLo) CUDA_CHECK(cudaFreeHost(hSendLo));
    if (hSendHi) CUDA_CHECK(cudaFreeHost(hSendHi));
    if (hRecvLo) CUDA_CHECK(cudaFreeHost(hRecvLo));
    if (hRecvHi) CUDA_CHECK(cudaFreeHost(hRecvHi));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    MPI_Finalize();
    return exitCode;
}
