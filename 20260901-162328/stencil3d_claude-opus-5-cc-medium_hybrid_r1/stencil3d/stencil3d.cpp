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

using Real = double;

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA 3D stencil
//
// Decomposition: the grid is split into contiguous slabs along Z (the slowest
// varying dimension), one slab per MPI rank.  Each rank drives one GPU and
// keeps its slab resident in device memory surrounded by one ghost plane on
// each side.  OpenMP is used for the host-side work (initialization of the
// local slab, assembly of the global result, validation).  Halo planes are
// exchanged every iteration through pinned host staging buffers because the
// MPI installation is not necessarily CUDA aware.
// ---------------------------------------------------------------------------

// Halo chunks per face; each chunk/direction pair is driven by its own OpenMP
// thread and CUDA stream so that the D2H, MPI and H2D stages overlap.
static constexpr int kHaloChunks = 4;
static constexpr int kXferStreams = 2 * kHaloChunks;

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Initialize the locally owned planes of the global grid.
// The global value of a cell is (globalLinearIndex % 19).
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t zOffset,
                    const size_t localNz) {
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static)
    for (long long zl = 0; zl < (long long)localNz; ++zl) {
        const size_t base = ((size_t)zl + zOffset) * plane;
        Real* dst = grid.data() + (size_t)zl * plane;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = (Real)((base + i) % 19);
        }
    }
}

// 7-point stencil over the local planes [zBegin, zEnd) of a slab that carries
// one ghost plane at local index 0 and one at local index localNz+1.
// The arithmetic (operand order, division by 7.0) mirrors the serial version
// exactly so results stay bit-identical.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out, const int nx, const int ny,
                              const int nz, const int zOffset, const int zBegin, const int zEnd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = (size_t)nx * (size_t)ny;
    size_t idx = (size_t)zBegin * plane + (size_t)y * (size_t)nx + (size_t)x;

    // Cells on an X/Y face are always copied through; test once per thread.
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    // March along Z keeping the previous/current planes in registers.
    Real bottom = in[idx - plane];
    Real center = in[idx];

    for (int zl = zBegin; zl < zEnd; ++zl) {
        const Real top = in[idx + plane];
        const int gz = zOffset + zl - 1;

        if (xyBoundary || gz == 0 || gz == nz - 1) {
            out[idx] = center;
        } else {
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }

        bottom = center;
        center = top;
        idx += plane;
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const size_t n = grid.size();
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(|:bad)
    for (long long i = 0; i < (long long)n; ++i) {
        const Real val = grid[(size_t)i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (long long i = 0; i < (long long)n; ++i) {
        const Real val = grid[(size_t)i];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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

// Transfer of a slab between a worker rank and rank 0, split into chunks so
// that element counts always fit into the int-typed MPI count arguments.
static void sendSlab(const Real* data, size_t count, int dest, MPI_Comm comm) {
    constexpr size_t kChunk = 1u << 26;
    for (size_t off = 0; off < count; off += kChunk) {
        const int n = (int)std::min(kChunk, count - off);
        MPI_Send(data + off, n, MPI_DOUBLE, dest, 7, comm);
    }
}

static void recvSlab(Real* data, size_t count, int src, MPI_Comm comm) {
    constexpr size_t kChunk = 1u << 26;
    for (size_t off = 0; off < count; off += kChunk) {
        const int n = (int)std::min(kChunk, count - off);
        MPI_Recv(data + off, n, MPI_DOUBLE, src, 7, comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    size_t gridSize = nx * ny * nz;

    if (gridSize == 0) {
        if (rank == 0) printf("Empty grid, nothing to do\n");
        MPI_Finalize();
        return 0;
    }

    // --- Bind one GPU per rank, round-robin over the devices of this node ---
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // --- Slab decomposition along Z ---
    const size_t plane = nx * ny;
    const size_t base = nz / (size_t)numRanks;
    const size_t rem = nz % (size_t)numRanks;
    const size_t localNz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t zOffset = (size_t)rank * base + std::min((size_t)rank, rem);
    const bool active = localNz > 0;

    // Ranks beyond the Z extent stay idle; neighbours are the adjacent active ranks.
    const int nActive = (int)std::min((size_t)numRanks, nz);
    const int down = (rank > 0 && active) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank + 1 < nActive && active) ? rank + 1 : MPI_PROC_NULL;

    const size_t localCells = localNz * plane;              // owned cells
    const size_t paddedCells = (localNz + 2) * plane;       // owned + 2 ghost planes

    // Initialize (host, OpenMP) and upload
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> hostSlab(localCells);
    initializeGrid(hostSlab, nx, ny, zOffset, localNz);

    Real* d_in = nullptr;
    Real* d_out = nullptr;
    Real* h_sendLo = nullptr;
    Real* h_sendHi = nullptr;
    Real* h_recvLo = nullptr;
    Real* h_recvHi = nullptr;
    cudaStream_t streamBnd = nullptr, streamInt = nullptr;
    cudaStream_t xferStream[kXferStreams] = {};
    cudaEvent_t evBnd = nullptr;

    if (active) {
        CUDA_CHECK(cudaMalloc(&d_in, paddedCells * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_out, paddedCells * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_in, 0, paddedCells * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_out, 0, paddedCells * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_in + plane, hostSlab.data(), localCells * sizeof(Real), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaHostAlloc(&h_sendLo, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_sendHi, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recvLo, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recvHi, plane * sizeof(Real), cudaHostAllocDefault));

        // The boundary planes are on the critical path of the halo exchange, so
        // give them the highest available stream priority.
        int prioLow = 0, prioHigh = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streamBnd, cudaStreamNonBlocking, prioHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streamInt, cudaStreamNonBlocking, prioLow));
        for (int i = 0; i < kXferStreams; ++i) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&xferStream[i], cudaStreamNonBlocking, prioHigh));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&evBnd, cudaEventDisableTiming));

        // Seed the ghost planes so that iteration 0 already sees valid halos.
        MPI_Request req[4];
        CUDA_CHECK(cudaMemcpy(h_sendLo, d_in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sendHi, d_in + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Irecv(h_recvLo, (int)plane, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(h_recvHi, (int)plane, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(h_sendLo, (int)plane, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(h_sendHi, (int)plane, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_in, h_recvLo, plane * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in + (localNz + 1) * plane, h_recvHi, plane * sizeof(Real), cudaMemcpyHostToDevice));
    }

    const dim3 block(32, 4);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x), (unsigned)((ny + block.y - 1) / block.y));

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        const int inx = (int)nx, iny = (int)ny, inz = (int)nz, izOff = (int)zOffset;
        const int lnz = (int)localNz;
        const bool exchange = (down != MPI_PROC_NULL) || (up != MPI_PROC_NULL);
        const bool threadedMPI = (provided >= MPI_THREAD_MULTIPLE);
        const int xferTasks = kXferStreams;
        // Only used by the OpenMP clause below, which the CUDA frontend does not
        // count as a use.
        [[maybe_unused]] const int xferThreads = threadedMPI ? xferTasks : 1;
        const size_t chunkLen = (plane + kHaloChunks - 1) / kHaloChunks;

        for (int iter = 0; iter < iterations; ++iter) {
            // Boundary planes first so their halo data can go on the wire while
            // the interior of the slab is still being computed.
            stencilKernel<<<grid, block, 0, streamBnd>>>(d_in, d_out, inx, iny, inz, izOff, 1, 2);
            if (lnz > 1) {
                stencilKernel<<<grid, block, 0, streamBnd>>>(d_in, d_out, inx, iny, inz, izOff, lnz, lnz + 1);
            }
            CUDA_CHECK(cudaEventRecord(evBnd, streamBnd));

            // Interior planes (2 .. localNz-1) run concurrently with the exchange.
            if (lnz > 2) {
                stencilKernel<<<grid, block, 0, streamInt>>>(d_in, d_out, inx, iny, inz, izOff, 2, lnz);
            }

            if (exchange) {
                // Each task owns one chunk of one face and drives the whole
                // device -> host -> MPI -> host -> device pipeline for it, so the
                // transfers of the different chunks and directions overlap.
#pragma omp parallel for num_threads(xferThreads) schedule(static, 1)
                for (int task = 0; task < xferTasks; ++task) {
                    const int chunk = task / 2;
                    const bool lo = (task % 2) == 0;
                    const int peer = lo ? down : up;
                    if (peer == MPI_PROC_NULL) continue;

                    const size_t off = (size_t)chunk * chunkLen;
                    const size_t len = std::min(chunkLen, plane - off);
                    Real* const hSend = (lo ? h_sendLo : h_sendHi) + off;
                    Real* const hRecv = (lo ? h_recvLo : h_recvHi) + off;
                    const Real* const dSend = d_out + (lo ? plane : localNz * plane) + off;
                    Real* const dRecv = d_out + (lo ? (size_t)0 : (localNz + 1) * plane) + off;
                    // Tags are symmetric: what a rank sends upwards its neighbour
                    // receives from below.
                    const int sendTag = lo ? 2 * chunk + 1 : 2 * chunk;
                    const int recvTag = lo ? 2 * chunk : 2 * chunk + 1;
                    cudaStream_t s = xferStream[task];

                    CUDA_CHECK(cudaSetDevice(device));
                    CUDA_CHECK(cudaStreamWaitEvent(s, evBnd, 0));
                    CUDA_CHECK(cudaMemcpyAsync(hSend, dSend, len * sizeof(Real), cudaMemcpyDeviceToHost, s));
                    CUDA_CHECK(cudaStreamSynchronize(s));

                    MPI_Request req[2];
                    MPI_Irecv(hRecv, (int)len, MPI_DOUBLE, peer, recvTag, MPI_COMM_WORLD, &req[0]);
                    MPI_Isend(hSend, (int)len, MPI_DOUBLE, peer, sendTag, MPI_COMM_WORLD, &req[1]);
                    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

                    CUDA_CHECK(cudaMemcpyAsync(dRecv, hRecv, len * sizeof(Real), cudaMemcpyHostToDevice, s));
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(d_in, d_out);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxDuration / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int status = 0;

    // The result lives in d_in after the final swap (and in the freshly
    // initialized buffer when no iteration was run).
    if (printResults || validate) {
        if (active) {
            CUDA_CHECK(cudaMemcpy(hostSlab.data(), d_in + plane, localCells * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        std::vector<Real> finalGrid;
        if (rank == 0) {
            finalGrid.resize(gridSize);
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)localCells; ++i) {
                finalGrid[(size_t)i] = hostSlab[(size_t)i];
            }
            for (int r = 1; r < nActive; ++r) {
                const size_t rCnt = base + ((size_t)r < rem ? 1 : 0);
                const size_t rOff = (size_t)r * base + std::min((size_t)r, rem);
                recvSlab(finalGrid.data() + rOff * plane, rCnt * plane, r, MPI_COMM_WORLD);
            }
        } else if (active) {
            sendSlab(hostSlab.data(), localCells, 0, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(finalGrid, nx, ny, nz);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }

        if (validate) {
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (active) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_sendLo);
        cudaFreeHost(h_sendHi);
        cudaFreeHost(h_recvLo);
        cudaFreeHost(h_recvHi);
        cudaStreamDestroy(streamBnd);
        cudaStreamDestroy(streamInt);
        for (int i = 0; i < kXferStreams; ++i) cudaStreamDestroy(xferStream[i]);
        cudaEventDestroy(evBnd);
    }

    MPI_Finalize();
    return status;
}
