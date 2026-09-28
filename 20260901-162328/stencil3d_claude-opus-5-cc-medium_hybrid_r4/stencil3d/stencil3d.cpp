#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,       \
                    __LINE__);                                                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Domain decomposition: the grid is split along Z across MPI ranks. Every rank
// keeps its own slab of nzLocal planes plus one halo plane on each side, so the
// local device array holds (nzLocal + 2) planes. Local plane k (1..nzLocal)
// corresponds to global plane zOffset + k - 1.
// ---------------------------------------------------------------------------

// 7-point stencil kernel, one thread per (x,y) column sweeping in Z with the
// Z-neighbours kept in registers and the current plane staged in shared memory.
static constexpr Real INV7 = 1.0 / 7.0;
static constexpr int BX = 32;
static constexpr int BY = 8;

__global__ __launch_bounds__(BX* BY) void stencilKernel(const Real* __restrict__ input,
                                                        Real* __restrict__ output, const int nx,
                                                        const int ny, const int kBegin,
                                                        const int kEnd) {
    __shared__ Real tile[BY + 2][BX + 2];

    const int x = blockIdx.x * BX + threadIdx.x + 1;
    const int y = blockIdx.y * BY + threadIdx.y + 1;
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;

    const size_t slice = (size_t)nx * (size_t)ny;
    const bool active = (x < nx - 1) && (y < ny - 1);

    size_t idx = (size_t)kBegin * slice + (size_t)y * nx + x;

    Real prev = 0.0, cur = 0.0, next = 0.0;
    if (active) {
        prev = input[idx - slice];
        cur = input[idx];
    }

    for (int k = kBegin; k < kEnd; ++k) {
        if (active) {
            next = input[idx + slice];
        }

        __syncthreads();
        if (active) {
            tile[ty][tx] = cur;
            if (threadIdx.x == 0) tile[ty][0] = input[idx - 1];
            if (threadIdx.x == BX - 1 || x == nx - 2) tile[ty][tx + 1] = input[idx + 1];
            if (threadIdx.y == 0) tile[0][tx] = input[idx - nx];
            if (threadIdx.y == BY - 1 || y == ny - 2) tile[ty + 1][tx] = input[idx + nx];
        }
        __syncthreads();

        if (active) {
            const Real left = tile[ty][tx - 1];
            const Real right = tile[ty][tx + 1];
            const Real front = tile[ty - 1][tx];
            const Real back = tile[ty + 1][tx];

            // Simple averaging stencil (identical operand order to the serial code).
            // The division by 7 is done with the FMA-refined reciprocal (Markstein),
            // which produces the correctly rounded quotient, i.e. bit-identical
            // results to "/ 7.0" but far cheaper than a full FP64 divide.
            const Real sum = cur + left + right + front + back + prev + next;
            const Real q0 = sum * INV7;
            const Real res = __fma_rn(-7.0, q0, sum);
            output[idx] = __fma_rn(res, INV7, q0);
        }

        prev = cur;
        cur = next;
        idx += slice;
    }
}

void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                         const size_t nzLocalWithHalo, const long long zOffset,
                         const long long nz) {
    const size_t slice = nx * ny;
#pragma omp parallel for schedule(static)
    for (long long k = 0; k < (long long)nzLocalWithHalo; ++k) {
        const long long gz = zOffset + k - 1;
        Real* dst = grid.data() + (size_t)k * slice;
        if (gz < 0 || gz >= nz) {
            // Halo plane outside of the global domain, never read.
            std::fill(dst, dst + slice, Real(0));
            continue;
        }
        const size_t base = (size_t)gz * slice;
        for (size_t i = 0; i < slice; ++i) {
            dst[i] = (Real)((base + i) % 19) * 1.0;
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = true;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    (void)provided;

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t slice = nx * ny;

    // ---- Bind one GPU per rank (round robin over the devices of each node) ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // ---- Z decomposition ----
    const long long nzll = (long long)nz;
    const long long base = nzll / nranks;
    const long long rem = nzll % nranks;
    const long long zOffset = rank * base + std::min<long long>(rank, rem);
    const long long nzLocal = base + (rank < rem ? 1 : 0);
    const size_t nzWithHalo = (size_t)nzLocal + 2;
    const size_t localSize = slice * nzWithHalo;

    // With more ranks than Z planes some ranks own nothing; the halo chain skips
    // them so that any rank count stays valid.
    const auto planesOf = [&](const int r) { return base + (r < rem ? 1 : 0); };
    int leftRank = MPI_PROC_NULL;
    int rightRank = MPI_PROC_NULL;
    if (nzLocal > 0) {
        for (int r = rank - 1; r >= 0; --r) {
            if (planesOf(r) > 0) { leftRank = r; break; }
        }
        for (int r = rank + 1; r < nranks; ++r) {
            if (planesOf(r) > 0) { rightRank = r; break; }
        }
    }

    // Local plane range that is interior in the global grid (global z in [1, nz-2]).
    const int kInteriorBegin = (int)std::max<long long>(1, 2 - zOffset);
    const int kInteriorEnd = (int)std::min<long long>(nzLocal + 1, nzll - zOffset);
    const bool hasWork = (kInteriorEnd > kInteriorBegin) && nx > 2 && ny > 2;

    // ---- Host buffer (used for initialization and for gathering results) ----
    std::vector<Real> hostGrid(localSize);
    if (rank == 0) printf("Initializing grid...\n");
    initializeLocalGrid(hostGrid, nx, ny, nzWithHalo, zOffset, nzll);

    // ---- Device buffers (double buffering); both start with identical data so
    //      the invariant boundary values are already correct in both buffers ----
    Real* dBuf[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&dBuf[0], localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dBuf[1], localSize * sizeof(Real)));
    CUDA_CHECK(
        cudaMemcpy(dBuf[0], hostGrid.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBuf[1], dBuf[0], localSize * sizeof(Real), cudaMemcpyDeviceToDevice));

    bool cudaAwareMPI = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cudaAwareMPI = (MPIX_Query_cuda_support() == 1);
#endif

    // ---- Halo transport ----
    // Neighbours on the same node exchange through an MPI shared-memory window:
    // the GPU writes its edge plane straight into the neighbour's slot and reads
    // its own slots back, so no MPI message copy is involved at all. Neighbours
    // on other nodes use MPI messages (directly from device memory when the MPI
    // implementation is CUDA-aware, otherwise through pinned staging buffers).
    // Node-local ranks of the two Z neighbours (MPI_UNDEFINED if on another node).
    int nodeNb[2] = {MPI_UNDEFINED, MPI_UNDEFINED};
    {
        MPI_Group worldGroup, nodeGroup;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
        MPI_Comm_group(nodeComm, &nodeGroup);
        const int worldNb[2] = {leftRank == MPI_PROC_NULL ? rank : leftRank,
                                rightRank == MPI_PROC_NULL ? rank : rightRank};
        MPI_Group_translate_ranks(worldGroup, 2, worldNb, nodeGroup, nodeNb);
        MPI_Group_free(&worldGroup);
        MPI_Group_free(&nodeGroup);
        if (leftRank == MPI_PROC_NULL) nodeNb[0] = MPI_UNDEFINED;
        if (rightRank == MPI_PROC_NULL) nodeNb[1] = MPI_UNDEFINED;
    }
    const bool shmLeft = (nodeNb[0] != MPI_UNDEFINED);
    const bool shmRight = (nodeNb[1] != MPI_UNDEFINED);

    // Four plane slots per rank: [iteration parity][from-left, from-right].
    const size_t slotElems = 4 * slice;
    Real* shmBase = nullptr;
    Real* shmPeer[2] = {nullptr, nullptr};
    MPI_Win shmWin = MPI_WIN_NULL;
    std::vector<void*> pinned;
    {
        MPI_Info winInfo;
        MPI_Info_create(&winInfo);
        // Separate, page-aligned blocks per rank so they can be pinned individually.
        MPI_Info_set(winInfo, "alloc_shared_noncontig", "true");
        MPI_Win_allocate_shared((MPI_Aint)(slotElems * sizeof(Real)), (int)sizeof(Real), winInfo,
                                nodeComm, &shmBase, &shmWin);
        MPI_Info_free(&winInfo);
        for (int s = 0; s < 2; ++s) {
            if (nodeNb[s] == MPI_UNDEFINED) continue;
            MPI_Aint sz = 0;
            int du = 0;
            void* p = nullptr;
            MPI_Win_shared_query(shmWin, nodeNb[s], &sz, &du, &p);
            shmPeer[s] = (Real*)p;
        }
        // Pin the blocks the GPU DMAs into/out of (best effort: works unpinned too).
        void* toPin[3] = {shmBase, (void*)shmPeer[0], (void*)shmPeer[1]};
        for (int i = 0; i < 3; ++i) {
            if (toPin[i] == nullptr) continue;
            bool dup = false;
            for (int j = 0; j < i; ++j) dup = dup || (toPin[j] == toPin[i]);
            if (dup) continue;
            if (cudaHostRegister(toPin[i], slotElems * sizeof(Real), cudaHostRegisterDefault) ==
                cudaSuccess) {
                pinned.push_back(toPin[i]);
            } else {
                cudaGetLastError();
            }
        }
    }

    // Ranks still reached through MPI messages.
    const int mpiLeft = shmLeft ? MPI_PROC_NULL : leftRank;
    const int mpiRight = shmRight ? MPI_PROC_NULL : rightRank;
    const bool needMpiExchange = (mpiLeft != MPI_PROC_NULL) || (mpiRight != MPI_PROC_NULL);

    // Pinned staging buffers for message-based links without CUDA-aware MPI.
    Real* hSend[2] = {nullptr, nullptr};
    Real* hRecv[2] = {nullptr, nullptr};
    if (needMpiExchange && !cudaAwareMPI) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMallocHost(&hSend[i], slice * sizeof(Real)));
            CUDA_CHECK(cudaMallocHost(&hRecv[i], slice * sizeof(Real)));
        }
    }

    cudaStream_t sEdge, sBulk;
    CUDA_CHECK(cudaStreamCreate(&sEdge));
    CUDA_CHECK(cudaStreamCreate(&sBulk));

    const dim3 block(BX, BY);
    const dim3 grid((unsigned)(((nx > 2 ? nx - 2 : 0) + BX - 1) / BX),
                    (unsigned)(((ny > 2 ? ny - 2 : 0) + BY - 1) / BY));

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;  // dBuf[cur] is the input, dBuf[1-cur] the output of an iteration
    for (int iter = 0; iter < iterations; ++iter) {
        Real* dIn = dBuf[cur];
        Real* dOut = dBuf[1 - cur];

        // Edge planes first so their results can be shipped to the neighbours
        // while the bulk of the slab is still being computed.
        if (hasWork) {
            const int kFirst = kInteriorBegin;
            const int kLast = kInteriorEnd - 1;
            if (kLast - kFirst >= 2) {
                stencilKernel<<<grid, block, 0, sEdge>>>(dIn, dOut, (int)nx, (int)ny, kFirst,
                                                         kFirst + 1);
                stencilKernel<<<grid, block, 0, sEdge>>>(dIn, dOut, (int)nx, (int)ny, kLast,
                                                         kLast + 1);
                stencilKernel<<<grid, block, 0, sBulk>>>(dIn, dOut, (int)nx, (int)ny, kFirst + 1,
                                                         kLast);
            } else {
                stencilKernel<<<grid, block, 0, sEdge>>>(dIn, dOut, (int)nx, (int)ny,
                                                         kInteriorBegin, kInteriorEnd);
            }
        }

        if (nranks > 1) {
            const size_t planeBytes = slice * sizeof(Real);
            const size_t par = (size_t)(iter & 1);  // double-buffered shared slots
            Real* sendLeft = dOut + slice;                        // local plane 1
            Real* sendRight = dOut + (size_t)nzLocal * slice;     // local plane nzLocal
            Real* recvLeft = dOut;                                // local plane 0
            Real* recvRight = dOut + (size_t)(nzLocal + 1) * slice;

            // Publish the edge planes: straight into the neighbours' shared-memory
            // slots, and into pinned staging buffers for message-based links.
            if (shmLeft) {
                CUDA_CHECK(cudaMemcpyAsync(shmPeer[0] + (par * 2 + 1) * slice, sendLeft, planeBytes,
                                           cudaMemcpyDeviceToHost, sEdge));
            }
            if (shmRight) {
                CUDA_CHECK(cudaMemcpyAsync(shmPeer[1] + (par * 2 + 0) * slice, sendRight,
                                           planeBytes, cudaMemcpyDeviceToHost, sEdge));
            }
            if (needMpiExchange && !cudaAwareMPI) {
                if (mpiLeft != MPI_PROC_NULL) {
                    CUDA_CHECK(cudaMemcpyAsync(hSend[0], sendLeft, planeBytes,
                                               cudaMemcpyDeviceToHost, sEdge));
                }
                if (mpiRight != MPI_PROC_NULL) {
                    CUDA_CHECK(cudaMemcpyAsync(hSend[1], sendRight, planeBytes,
                                               cudaMemcpyDeviceToHost, sEdge));
                }
            }
            // Everything published by this rank must be visible before the rendezvous.
            CUDA_CHECK(cudaStreamSynchronize(sEdge));

            MPI_Request reqs[4];
            if (needMpiExchange) {
                Real* rl = cudaAwareMPI ? recvLeft : hRecv[0];
                Real* rr = cudaAwareMPI ? recvRight : hRecv[1];
                Real* sl = cudaAwareMPI ? sendLeft : hSend[0];
                Real* sr = cudaAwareMPI ? sendRight : hSend[1];
                MPI_Irecv(rl, (int)slice, MPI_DOUBLE, mpiLeft, 0, MPI_COMM_WORLD, &reqs[0]);
                MPI_Irecv(rr, (int)slice, MPI_DOUBLE, mpiRight, 1, MPI_COMM_WORLD, &reqs[1]);
                MPI_Isend(sl, (int)slice, MPI_DOUBLE, mpiLeft, 1, MPI_COMM_WORLD, &reqs[2]);
                MPI_Isend(sr, (int)slice, MPI_DOUBLE, mpiRight, 0, MPI_COMM_WORLD, &reqs[3]);
            }

            if (shmLeft || shmRight) {
                // Pairwise rendezvous with the shared-memory neighbours: their
                // slots for this iteration are filled. The slots alternate per
                // iteration, so a writer never catches up with a reader.
                MPI_Request tok[4];
                char out = 0, inL = 0, inR = 0;
                int nt = 0;
                if (shmLeft) {
                    MPI_Irecv(&inL, 1, MPI_BYTE, leftRank, 20, MPI_COMM_WORLD, &tok[nt++]);
                    MPI_Isend(&out, 1, MPI_BYTE, leftRank, 21, MPI_COMM_WORLD, &tok[nt++]);
                }
                if (shmRight) {
                    MPI_Irecv(&inR, 1, MPI_BYTE, rightRank, 21, MPI_COMM_WORLD, &tok[nt++]);
                    MPI_Isend(&out, 1, MPI_BYTE, rightRank, 20, MPI_COMM_WORLD, &tok[nt++]);
                }
                MPI_Waitall(nt, tok, MPI_STATUSES_IGNORE);
                if (shmLeft) {
                    CUDA_CHECK(cudaMemcpyAsync(recvLeft, shmBase + (par * 2 + 0) * slice,
                                               planeBytes, cudaMemcpyHostToDevice, sEdge));
                }
                if (shmRight) {
                    CUDA_CHECK(cudaMemcpyAsync(recvRight, shmBase + (par * 2 + 1) * slice,
                                               planeBytes, cudaMemcpyHostToDevice, sEdge));
                }
            }

            if (needMpiExchange) {
                MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
                if (!cudaAwareMPI) {
                    if (mpiLeft != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(recvLeft, hRecv[0], planeBytes,
                                                   cudaMemcpyHostToDevice, sEdge));
                    }
                    if (mpiRight != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(recvRight, hRecv[1], planeBytes,
                                                   cudaMemcpyHostToDevice, sEdge));
                    }
                }
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(sBulk));
        CUDA_CHECK(cudaStreamSynchronize(sEdge));

        cur = 1 - cur;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Assemble the final grid on rank 0 if it is needed ----
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        // dBuf[cur] holds the result of the last iteration after the final swap.
        CUDA_CHECK(cudaMemcpy(hostGrid.data(), dBuf[cur], localSize * sizeof(Real),
                              cudaMemcpyDeviceToHost));

        MPI_Datatype planeType;
        MPI_Type_contiguous((int)slice, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(nranks);
            displs.resize(nranks);
            for (int r = 0; r < nranks; ++r) {
                const long long off = r * base + std::min<long long>(r, rem);
                counts[r] = (int)(base + (r < rem ? 1 : 0));
                displs[r] = (int)off;
            }
            finalGrid.resize(gridSize);
        }
        MPI_Gatherv(hostGrid.data() + slice, (int)nzLocal, planeType, finalGrid.data(),
                    counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);
    }

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
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaStreamDestroy(sBulk));
    for (void* p : pinned) CUDA_CHECK(cudaHostUnregister(p));
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dBuf[0]));
    CUDA_CHECK(cudaFree(dBuf[1]));
    for (int i = 0; i < 2; ++i) {
        if (hSend[i]) CUDA_CHECK(cudaFreeHost(hSend[i]));
        if (hRecv[i]) CUDA_CHECK(cudaFreeHost(hRecv[i]));
    }
    MPI_Win_free(&shmWin);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
