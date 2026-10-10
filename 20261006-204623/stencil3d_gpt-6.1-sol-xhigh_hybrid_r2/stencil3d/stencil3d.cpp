#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#ifdef STENCIL3D_MPI_CUDA_QUERY
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include "../common/results_output.hpp"

using Real = double;

// Only the main thread calls MPI. Abort all ranks on errors, including those
// with no assigned planes, so peers cannot get stuck in a halo exchange.
static void mpiCheck(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "%s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d, %s: %s\n", rank, operation,
            cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr unsigned TileX = 32;
constexpr unsigned TileY = 8;
constexpr size_t TileZ = 8;

// Coalesced XY tiles share their four lateral neighbors. A register window
// reuses the center and the two Z neighbors while sweeping eight planes.
// Boundary mode updates just the two MPI faces, independently of bulk work.
template <bool Boundary>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t zBegin,
                              size_t zEnd, size_t ownedPlanes,
                              size_t tilesX, size_t tilesY, size_t tilesZ) {
    __shared__ Real tile[2][TileY + 2][TileX + 2];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const size_t plane = nx * ny;
    // A native 3D launch avoids costly 64-bit division to decode tile indices.
    // Grid-stride loops still support dimensions beyond CUDA's grid limits.
    for (size_t bz = blockIdx.z; bz < tilesZ; bz += gridDim.z) {
        for (size_t by = blockIdx.y; by < tilesY; by += gridDim.y) {
            for (size_t bx = blockIdx.x; bx < tilesX; bx += gridDim.x) {
                const size_t x = bx * TileX + tx + 1;
                const size_t y = by * TileY + ty + 1;
                const size_t first = Boundary ? (bz == 0 ? 1 : ownedPlanes)
                                             : zBegin + bz * TileZ;
                const size_t last = Boundary ? first + 1
                                            : (first + TileZ < zEnd ? first + TileZ : zEnd);
                // This predicate is uniform across the block.
                if (first < zBegin || first >= zEnd) continue;
                const bool load = x < nx && y < ny;
                const bool update = x < nx - 1 && y < ny - 1;
                size_t index = first * plane + y * nx + x;
                Real below = load ? input[index - plane] : 0.0;
                Real center = load ? input[index] : 0.0;
                for (size_t z = first; z < last; ++z, index += plane) {
                    auto& layer = tile[(z - first) & 1];
                    const Real above = load ? input[index + plane] : 0.0;
                    layer[ty + 1][tx + 1] = center;
                    if (tx == 0)
                        layer[ty + 1][0] = load ? input[index - 1] : 0.0;
                    if (tx == TileX - 1)
                        layer[ty + 1][TileX + 1] = (x + 1 < nx && y < ny)
                                                     ? input[index + 1] : 0.0;
                    if (ty == 0)
                        layer[0][tx + 1] = load ? input[index - nx] : 0.0;
                    if (ty == TileY - 1)
                        layer[TileY + 1][tx + 1] = (x < nx && y + 1 < ny)
                                                     ? input[index + nx] : 0.0;
                    __syncthreads();
                    if (update) {
                        // Preserve the original addition order and correctly rounded
                        // division. Reciprocal multiplication would change hashes.
                        const Real sum = center + layer[ty + 1][tx]
                            + layer[ty + 1][tx + 2] + layer[ty][tx + 1]
                            + layer[ty + 2][tx + 1] + below + above;
                        output[index] = __ddiv_rn(sum, 7.0);
                    }
                    // Alternate shared layers: the next iteration's barrier protects
                    // the preceding reads, avoiding a second barrier on every plane.
                    below = center;
                    center = above;
                }
                __syncthreads();
            }
        }
    }
}

template <bool Boundary>
static void launchStencil(const Real* input, Real* output, size_t nx,
                          size_t ny, size_t zBegin, size_t zEnd,
                          size_t ownedPlanes, cudaStream_t stream) {
    if (zBegin >= zEnd) return;
    const size_t tilesX = (nx - 3) / TileX + 1;
    const size_t tilesY = (ny - 3) / TileY + 1;
    const size_t tilesZ = Boundary ? (ownedPlanes == 1 ? 1 : 2)
                                   : (zEnd - zBegin - 1) / TileZ + 1;
    const dim3 blocks(static_cast<unsigned>(std::min<size_t>(tilesX, INT_MAX)),
                      static_cast<unsigned>(std::min<size_t>(tilesY, 65535)),
                      static_cast<unsigned>(std::min<size_t>(tilesZ, 65535)));
    stencilKernel<Boundary><<<blocks, dim3(TileX, TileY), 0, stream>>>(
        input, output, nx, ny, zBegin, zEnd, ownedPlanes, tilesX, tilesY, tilesZ);
    CUDA_CHECK(cudaGetLastError());
}

static size_t slabCount(size_t nz, int ranks, int rank) {
    return nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
}

static size_t slabStart(size_t nz, int ranks, int rank) {
    return (nz / ranks) * rank + std::min<size_t>(rank, nz % ranks);
}

// Chunk MPI's int-sized counts without truncating global dimensions or offsets.
static void postHalo(Real* data, size_t count, int peer, int tag, bool send,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Request request;
        if (send)
            MPI_CHECK(MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag,
                               comm, &request));
        else
            MPI_CHECK(MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag,
                               comm, &request));
        requests.push_back(request);
        offset += chunk;
    }
}

static void gatherGrid(const std::vector<Real>& local, size_t nx, size_t ny,
                       size_t nz, int rank, int ranks, MPI_Comm comm) {
    constexpr int resultTag = 3;
    const size_t plane = nx * ny;
    if (rank == 0) {
        std::vector<Real> global(plane * nz);
        std::copy(local.begin(), local.end(), global.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t start = slabStart(nz, ranks, source) * plane;
            const size_t count = slabCount(nz, ranks, source) * plane;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
                MPI_CHECK(MPI_Recv(global.data() + start + offset, chunk,
                                   MPI_DOUBLE, source, resultTag, comm, MPI_STATUS_IGNORE));
                offset += chunk;
            }
        }
        // Keep global order for the original Kahan sum, samples and byte hash.
        print_results(global, "Grid");
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int chunk = static_cast<int>(std::min<size_t>(local.size() - offset, INT_MAX));
            MPI_CHECK(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE,
                               0, resultTag, comm));
            offset += chunk;
        }
    }
}

static bool validateResult(const std::vector<Real>& local, int rank, MPI_Comm comm) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t i = 0; i < local.size(); ++i) {
        const Real value = local[i];
        invalid |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    Real globalMin, globalMax;
    int globalInvalid;
    MPI_CHECK(MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm));
    MPI_CHECK(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm));
    if (rank == 0) {
        if (globalInvalid)
            printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 1e6 || globalMin < -1e6)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !globalInvalid && globalMax <= 1e6 && globalMin >= -1e6;
}

static void printUsage(const char* progName) {
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

static bool parseSize(const char* text, size_t& value) {
    if (*text < '0' || *text > '9') return false;
    char* end;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

static int runBenchmark(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if ((!strcmp(option, "-x") || !strcmp(option, "-y") ||
             !strcmp(option, "-z") || !strcmp(option, "-i")) && i + 1 < argc) {
            size_t value;
            if (!parseSize(argv[++i], value) ||
                (!strcmp(option, "-i") && value > INT_MAX)) {
                if (worldRank == 0) fprintf(stderr, "Invalid value for %s: %s\n", option, argv[i]);
                return 1;
            }
            if (!strcmp(option, "-x")) nx = value;
            else if (!strcmp(option, "-y")) ny = value;
            else if (!strcmp(option, "-z")) nz = value;
            else iterations = static_cast<int>(value);
        } else if (!strcmp(option, "-v")) validate = true;
        else if (!strcmp(option, "-r")) printResults = true;
        else if (!strcmp(option, "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > maxCells / ny || nx * ny > maxCells / nz) {
        if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive and fit in address space.\n");
        return 1;
    }

    // Excess ranks do not allocate a slab or a GPU. Each active rank owns at
    // least one plane, including thin and uneven decompositions.
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED,
                             worldRank, &comm));
    if (comm == MPI_COMM_NULL) return 0;
    const int rank = worldRank;
    const size_t owned = slabCount(nz, ranks, rank);
    const size_t startZ = slabStart(nz, ranks, rank);
    const size_t plane = nx * ny;
    if (owned > maxCells / plane || maxCells / plane - owned < 2) {
        fprintf(stderr, "Rank %d: local grid with halo planes is too large.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const size_t cells = (owned + 2) * plane;
    const size_t bytes = cells * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);

    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Supports launchers that expose only the assigned GPU to each rank.
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, omp_get_max_threads()));

    int cudaAware = 0;
#ifdef STENCIL3D_MPI_CUDA_QUERY
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, comm));
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA (%s halos)\n",
               ranks, omp_get_max_threads(), cudaAware ? "device" : "pinned host");
        printf("Initializing grid...\n");
    }

    Real *input, *output;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
    {
        std::vector<Real> initial(cells);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells; ++i) {
            const bool outside = (startZ == 0 && i < plane) ||
                (startZ + owned == nz && i >= (owned + 1) * plane);
            initial[i] = outside ? 0.0 : static_cast<Real>((startZ * plane + i - plane) % 19);
        }
        CUDA_CHECK(cudaMemcpy(input, initial.data(), bytes, cudaMemcpyHostToDevice));
        // Seed fixed faces in both buffers to avoid per-iteration copies.
        CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    }

    cudaStream_t bulkStream, edgeStream, transfer[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&bulkStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&edgeStream, cudaStreamNonBlocking));
    cudaEvent_t bulkDone, edgeDone, sendReady[2], haloReady[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&edgeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(bulkDone, bulkStream));
    CUDA_CHECK(cudaEventRecord(edgeDone, edgeStream));
    Real *sendBuffer[2] = {nullptr, nullptr}, *recvBuffer[2] = {nullptr, nullptr};
    const int peers[2] = {rank == 0 ? MPI_PROC_NULL : rank - 1,
                          rank + 1 == ranks ? MPI_PROC_NULL : rank + 1};
    for (int side = 0; side < 2; ++side) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer[side], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&sendReady[side], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&haloReady[side], cudaEventDisableTiming));
        if (!cudaAware && peers[side] != MPI_PROC_NULL && iterations > 1 && nx > 2 && ny > 2 && nz > 2) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendBuffer[side]), planeBytes));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvBuffer[side]), planeBytes));
        }
    }
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane - 1) / INT_MAX + 1));
    const size_t zBegin = startZ == 0 ? 2 : 1;
    const size_t zEnd = owned + 1 - (startZ + owned == nz);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(comm));
    if (rank == 0) printf("Running stencil computation...\n");
    const double begin = MPI_Wtime();
    for (int iter = 0; iter < iterations && hasInterior; ++iter) {
        if (ranks == 1) {
            launchStencil<false>(input, output, nx, ny, zBegin, zEnd, owned, bulkStream);
        } else {
            // Capture the PREVIOUS events before re-recording either event.
            // The two kernels write disjoint output planes. MPI can send the
            // previous edges while the previous bulk kernel is still running.
            CUDA_CHECK(cudaStreamWaitEvent(bulkStream, edgeDone, 0));
            CUDA_CHECK(cudaStreamWaitEvent(edgeStream, bulkDone, 0));
            launchStencil<false>(input, output, nx, ny, std::max<size_t>(zBegin, 2),
                                 std::min(zEnd, owned), owned, bulkStream);
            CUDA_CHECK(cudaEventRecord(bulkDone, bulkStream));

            // Initial ghost planes have already been initialized analytically.
            if (iter > 0) {
                requests.clear();
                for (int side = 0; side < 2; ++side) {
                    if (peers[side] == MPI_PROC_NULL) continue;
                    // MPI must not overwrite a pinned receive buffer while
                    // the preceding asynchronous H2D copy still reads it.
                    if (!cudaAware && iter > 1)
                        CUDA_CHECK(cudaEventSynchronize(haloReady[side]));
                    Real* ghost = input + (side == 0 ? 0 : owned + 1) * plane;
                    postHalo(cudaAware ? ghost : recvBuffer[side], plane,
                             peers[side], 1 - side, false, comm, requests);
                    CUDA_CHECK(cudaStreamWaitEvent(transfer[side], edgeDone, 0));
                    if (!cudaAware)
                        CUDA_CHECK(cudaMemcpyAsync(sendBuffer[side],
                            input + (side == 0 ? 1 : owned) * plane, planeBytes,
                            cudaMemcpyDeviceToHost, transfer[side]));
                    CUDA_CHECK(cudaEventRecord(sendReady[side], transfer[side]));
                }
                for (int side = 0; side < 2; ++side) {
                    if (peers[side] == MPI_PROC_NULL) continue;
                    CUDA_CHECK(cudaEventSynchronize(sendReady[side]));
                    postHalo(cudaAware ? input + (side == 0 ? 1 : owned) * plane
                                       : sendBuffer[side], plane, peers[side], side,
                             true, comm, requests);
                }
                // Drive MPI progress while the GPU does bulk work.
                for (size_t offset = 0; offset < requests.size();) {
                    const int count = static_cast<int>(std::min<size_t>(requests.size() - offset, INT_MAX));
                    MPI_CHECK(MPI_Waitall(count, requests.data() + offset, MPI_STATUSES_IGNORE));
                    offset += count;
                }
                for (int side = 0; side < 2; ++side) {
                    if (peers[side] == MPI_PROC_NULL) continue;
                    if (!cudaAware)
                        CUDA_CHECK(cudaMemcpyAsync(
                            input + (side == 0 ? 0 : owned + 1) * plane,
                            recvBuffer[side], planeBytes, cudaMemcpyHostToDevice,
                            transfer[side]));
                    CUDA_CHECK(cudaEventRecord(haloReady[side], transfer[side]));
                    CUDA_CHECK(cudaStreamWaitEvent(edgeStream, haloReady[side], 0));
                }
            }
            launchStencil<true>(input, output, nx, ny, zBegin, zEnd, owned, edgeStream);
            CUDA_CHECK(cudaEventRecord(edgeDone, edgeStream));
        }
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(bulkStream));
    CUDA_CHECK(cudaStreamSynchronize(edgeStream));
    const double elapsed = MPI_Wtime() - begin;
    double seconds;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        const double updates = hasInterior ? static_cast<double>(nx - 2) *
            static_cast<double>(ny - 2) * static_cast<double>(nz - 2) * iterations : 0.0;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
    }

    bool valid = true;
    if (validate || printResults) {
        std::vector<Real> local(owned * plane);
        CUDA_CHECK(cudaMemcpy(local.data(), input + plane, local.size() * sizeof(Real),
                              cudaMemcpyDeviceToHost));
        if (printResults) gatherGrid(local, nx, ny, nz, rank, ranks, comm);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            valid = validateResult(local, rank, comm);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (int side = 0; side < 2; ++side) {
        if (sendBuffer[side]) CUDA_CHECK(cudaFreeHost(sendBuffer[side]));
        if (recvBuffer[side]) CUDA_CHECK(cudaFreeHost(recvBuffer[side]));
        CUDA_CHECK(cudaEventDestroy(sendReady[side]));
        CUDA_CHECK(cudaEventDestroy(haloReady[side]));
        CUDA_CHECK(cudaStreamDestroy(transfer[side]));
    }
    CUDA_CHECK(cudaEventDestroy(bulkDone));
    CUDA_CHECK(cudaEventDestroy(edgeDone));
    CUDA_CHECK(cudaStreamDestroy(bulkStream));
    CUDA_CHECK(cudaStreamDestroy(edgeStream));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_CHECK(MPI_Comm_free(&comm));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
