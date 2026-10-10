#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#ifdef STENCIL3D_MPI_CUDA_QUERY
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkMPI(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
        std::abort();
    }
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define MPI_CHECK(operation) checkMPI((operation), #operation)
#define CUDA_CHECK(operation) checkCUDA((operation), #operation)

constexpr unsigned tileX = 32;
constexpr unsigned tileY = 4;
constexpr size_t tileZ = 8;

__device__ __forceinline__ Real average(Real sum) {
    // For this initialization, nonzero stencil sums are normal and stay in
    // [0, 126]. For division by 7, the exact quotient is separated from
    // any rounding midpoint by at least 1/14 ulp. The exact FMA residual and
    // a second rounding therefore produce the same result as rounded division,
    // without the general divider's reciprocal refinement and special cases.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real remainder = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(remainder, reciprocal, quotient);
}

// Each warp accesses contiguous X values. Keeping three Z values in registers
// reuses the previous and next planes without shared-memory barriers.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t firstZ, size_t endZ) {
    const size_t plane = nx * ny;
    for (size_t y = 1 + size_t(blockIdx.y) * tileY + threadIdx.y;
         y < ny - 1; y += size_t(gridDim.y) * tileY) {
        for (size_t x = 1 + size_t(blockIdx.x) * tileX + threadIdx.x;
             x < nx - 1; x += size_t(gridDim.x) * tileX) {
            for (size_t z0 = firstZ + size_t(blockIdx.z) * tileZ;
                 z0 < endZ; z0 += size_t(gridDim.z) * tileZ) {
                size_t index = z0 * plane + y * nx + x;
                Real bottom = input[index - plane];
                Real center = input[index];
                const size_t stop = (z0 + tileZ < endZ) ? z0 + tileZ : endZ;
                for (size_t z = z0; z < stop; ++z, index += plane) {
                    const Real top = input[index + plane];
                    // Preserve the original addition order and rounded division.
                    output[index] = average(center + input[index - 1] + input[index + 1]
                                            + input[index - nx] + input[index + nx]
                                            + bottom + top);
                    bottom = center;
                    center = top;
                }
            }
        }
    }
}

// Compute the two outgoing planes first, so communication can run concurrently
// with stencilKernel on the rest of the slab. A one-plane slab is written once.
__global__ void facesKernel(const Real* __restrict__ input,
                           Real* __restrict__ output,
                           size_t nx, size_t ny, size_t localZ,
                           size_t firstZ, size_t endZ) {
    const size_t z = blockIdx.z == 0 ? 1 : localZ;
    if (z < firstZ || z >= endZ) return;
    const size_t plane = nx * ny;
    for (size_t y = 1 + size_t(blockIdx.y) * tileY + threadIdx.y;
         y < ny - 1; y += size_t(gridDim.y) * tileY) {
        for (size_t x = 1 + size_t(blockIdx.x) * tileX + threadIdx.x;
             x < nx - 1; x += size_t(gridDim.x) * tileX) {
            const size_t index = z * plane + y * nx + x;
            output[index] = average(input[index] + input[index - 1] + input[index + 1]
                                    + input[index - nx] + input[index + nx]
                                    + input[index - plane] + input[index + plane]);
        }
    }
}

dim3 blocksXY(size_t nx, size_t ny) {
    // Grid-stride loops also cover grids exceeding CUDA's Y/Z launch limits.
    return dim3(static_cast<unsigned>(std::min<size_t>((nx - 2 + tileX - 1) / tileX, INT_MAX)),
                static_cast<unsigned>(std::min<size_t>((ny - 2 + tileY - 1) / tileY, 65535)), 1);
}

void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t firstZ, size_t endZ, cudaStream_t stream) {
    if (firstZ >= endZ) return;
    dim3 blocks = blocksXY(nx, ny);
    blocks.z = static_cast<unsigned>(std::min<size_t>((endZ - firstZ + tileZ - 1) / tileZ, 65535));
    stencilKernel<<<blocks, dim3(tileX, tileY), 0, stream>>>(input, output, nx, ny, firstZ, endZ);
    CUDA_CHECK(cudaGetLastError());
}

void initializeGrid(Real* grid, size_t count, size_t globalOffset) {
    // Unsigned wraparound is intentional: rank zero's unused lower ghost plane
    // precedes global index zero. Every owned cell uses its original global index.
    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < count; ++index) {
        grid[index] = static_cast<Real>((globalOffset + index) % 19);
    }
}

bool parseNumber(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Transfer in chunks to avoid MPI's int count limit for large grids.
void gatherResults(const Real* local, size_t localCount, size_t plane, size_t nz,
                   int ranks, MPI_Comm comm) {
    std::vector<Real> full;
    if (worldRank == 0) full.resize(plane * nz);
    if (plane * nz <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> counts, offsets;
        if (worldRank == 0) {
            counts.resize(ranks);
            offsets.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                const size_t z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
                const size_t start = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
                counts[rank] = static_cast<int>(z * plane);
                offsets[rank] = static_cast<int>(start * plane);
            }
        }
        MPI_CHECK(MPI_Gatherv(local, static_cast<int>(localCount), MPI_DOUBLE,
                             full.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, comm));
    } else if (worldRank == 0) {
        std::copy_n(local, localCount, full.data());
        for (int rank = 1; rank < ranks; ++rank) {
            const size_t count = (nz / ranks + (static_cast<size_t>(rank) < nz % ranks)) * plane;
            const size_t start = (size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks)) * plane;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
                MPI_CHECK(MPI_Recv(full.data() + start + offset, chunk, MPI_DOUBLE, rank,
                                   2, comm, MPI_STATUS_IGNORE));
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < localCount;) {
            const int chunk = static_cast<int>(std::min<size_t>(localCount - offset, INT_MAX));
            MPI_CHECK(MPI_Send(local + offset, chunk, MPI_DOUBLE, 0, 2, comm));
            offset += chunk;
        }
    }
    if (worldRank == 0) print_results(full, "Grid");
}

bool validateResult(const Real* grid, size_t count, size_t beginZ,
                    size_t nx, size_t ny, size_t nz, MPI_Comm comm) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static) reduction(min:localMin) reduction(max:localMax) reduction(|:invalid)
    for (size_t i = 0; i < count; ++i) {
        const Real value = grid[i];
        if (!std::isfinite(value)) invalid |= 1;
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
        const size_t z = beginZ + i / plane;
        const size_t y = (i % plane) / nx;
        const size_t x = i % nx;
        if ((x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
            && value != static_cast<Real>((beginZ * plane + i) % 19)) invalid |= 2;
    }
    Real globalMin = 0, globalMax = 0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
    MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (localMin < -1e6 || localMax > 1e6) invalid |= 4;
    MPI_CHECK(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, comm));
    if (worldRank == 0) {
        if (globalInvalid & 1) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalInvalid & 2) std::printf("Validation failed: boundary values changed\n");
        if (globalInvalid & 4) std::printf("Validation failed: values out of expected range\n");
        std::printf("Validation: %s\n", globalInvalid == 0 ? "PASSED" : "FAILED");
    }
    return globalInvalid == 0;
}

int run(int argc, char** argv, int worldSize) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0
                    || std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0)
                   && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value = 0;
            if (!parseNumber(argv[++i], value) || (option == 'i' && value > INT_MAX)) {
                if (worldRank == 0) std::fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') iterations = static_cast<int>(value);
        } else {
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxElements / ny || nx * ny > maxElements / nz) {
        if (worldRank == 0) std::fprintf(stderr, "Invalid or overflowing grid dimensions\n");
        return 1;
    }

    // Ranks beyond the number of Z planes have no data and need no GPU context.
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (comm == MPI_COMM_NULL) return 0;
    MPI_CHECK(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) fail("The MPI/OpenMP/CUDA benchmark requires a CUDA device on every active rank");
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    if (std::getenv("OMP_NUM_THREADS") == nullptr) omp_set_num_threads(std::min(8, omp_get_max_threads()));

    int deviceMPI = 0;
#ifdef STENCIL3D_MPI_CUDA_QUERY
    deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
    // This also enables CUDA-aware implementations without a query extension.
    if (const char* overrideMode = std::getenv("STENCIL3D_CUDA_AWARE_MPI")) {
        if (std::strcmp(overrideMode, "0") == 0) deviceMPI = 0;
        else if (std::strcmp(overrideMode, "1") == 0) deviceMPI = 1;
        else fail("STENCIL3D_CUDA_AWARE_MPI must be 0 or 1");
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &deviceMPI, 1, MPI_INT, MPI_MIN, comm));

    const size_t plane = nx * ny;
    const size_t localZ = nz / ranks + (static_cast<size_t>(worldRank) < nz % ranks);
    const size_t beginZ = size_t(worldRank) * (nz / ranks) + std::min<size_t>(worldRank, nz % ranks);
    if (localZ > maxElements / plane || maxElements / plane - localZ < 2) fail("Local grid size overflows");
    const size_t count = (localZ + 2) * plane;
    const size_t ownedCount = localZ * plane;
    const size_t bytes = count * sizeof(Real);
    const size_t faceBytes = plane * sizeof(Real);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t firstZ = beginZ == 0 ? 2 : 1;
    const size_t endZ = localZ + 1 - (beginZ + localZ == nz ? 1 : 0);

    if (worldRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d; OpenMP threads/rank: %d; CUDA halo exchange: %s\n",
                    ranks, omp_get_max_threads(), deviceMPI ? "device buffers" : "pinned host buffers");
        std::printf("Initializing grid...\n");
    }
    std::unique_ptr<Real, decltype(&std::free)> host(static_cast<Real*>(std::malloc(bytes)), &std::free);
    if (!host) fail("Cannot allocate the local host grid");
    initializeGrid(host.get(), count, beginZ * plane - plane);
    Real* input = nullptr;
    Real* output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), bytes));
    CUDA_CHECK(cudaMemcpy(input, host.get(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    // Both buffers retain the fixed global boundaries for all iterations.
    if (!printResults && !validate) host.reset();

    cudaStream_t compute = nullptr, faces = nullptr;
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&compute, cudaStreamNonBlocking, leastPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&faces, cudaStreamNonBlocking, greatestPriority));
    cudaEvent_t bulkDone = nullptr, halosReady = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));

    const int lower = worldRank == 0 ? MPI_PROC_NULL : worldRank - 1;
    const int upper = worldRank == ranks - 1 ? MPI_PROC_NULL : worldRank + 1;
    const bool exchange = ranks > 1 && hasInterior && iterations > 1;
    Real* staging = nullptr;
    if (exchange && !deviceMPI) {
        if (plane > maxElements / 4) fail("Halo buffer size overflows");
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&staging), 4 * faceBytes, cudaHostAllocDefault));
    }
    const size_t chunks = (plane - 1) / INT_MAX + 1;
    if (chunks > static_cast<size_t>(INT_MAX / 4)) fail("Too many MPI halo messages");
    std::vector<MPI_Request> requests(exchange ? 4 * chunks : 0);
    // Load CUDA modules before timing the steady-state computation.
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, stencilKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, facesKernel));
    cudaGraphExec_t graphExec = nullptr;
    const int graphSteps = std::min(iterations - iterations % 2, 32);
    if (ranks == 1 && hasInterior && graphSteps > 0) {
        // Replaying an even number of steps returns to the same buffer, and
        // avoids repeated host launch overhead when no MPI exchange is needed.
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
        for (int step = 0; step < graphSteps; ++step) {
            launchStencil(step % 2 == 0 ? input : output, step % 2 == 0 ? output : input,
                          nx, ny, firstZ, endZ, compute);
        }
        CUDA_CHECK(cudaStreamEndCapture(compute, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphUpload(graphExec, compute));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (worldRank == 0) std::printf("Running stencil computation...\n");
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();

    if (graphExec) {
        int iteration = 0;
        for (; iterations - iteration >= graphSteps; iteration += graphSteps)
            CUDA_CHECK(cudaGraphLaunch(graphExec, compute));
        for (; iteration < iterations; ++iteration) {
            launchStencil(input, output, nx, ny, firstZ, endZ, compute);
            std::swap(input, output);
        }
    } else if (hasInterior) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (!exchange || iteration == iterations - 1) {
                launchStencil(input, output, nx, ny, firstZ, endZ, compute);
            } else {
                // A staged receive buffer cannot be reused until its previous
                // asynchronous host-to-device transfer has completed.
                if (iteration > 0 && !deviceMPI) CUDA_CHECK(cudaEventSynchronize(halosReady));
                int requestCount = 0;
                // Receives touch only ghosts. The previous bulk kernel reads
                // owned planes only; the previous face kernel is already done.
                for (int side = 0; side < 2; ++side) {
                    const int neighbor = side == 0 ? lower : upper;
                    if (neighbor == MPI_PROC_NULL) continue;
                    Real* receive = deviceMPI ? output + (side == 0 ? 0 : (localZ + 1) * plane)
                                              : staging + (2 + side) * plane;
                    for (size_t offset = 0; offset < plane;) {
                        const int chunk = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
                        MPI_CHECK(MPI_Irecv(receive + offset, chunk, MPI_DOUBLE, neighbor, 1 - side,
                                            comm, &requests[requestCount++]));
                        offset += chunk;
                    }
                }
                dim3 faceBlocks = blocksXY(nx, ny);
                faceBlocks.z = localZ == 1 ? 1 : 2;
                facesKernel<<<faceBlocks, dim3(tileX, tileY), 0, faces>>>
                    (input, output, nx, ny, localZ, firstZ, endZ);
                CUDA_CHECK(cudaGetLastError());
                launchStencil(input, output, nx, ny, std::max<size_t>(firstZ, 2),
                              std::min(endZ, localZ), compute);
                CUDA_CHECK(cudaEventRecord(bulkDone, compute));
                if (!deviceMPI) {
                    if (lower != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(staging, output + plane, faceBytes, cudaMemcpyDeviceToHost, faces));
                    if (upper != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(staging + plane, output + localZ * plane, faceBytes,
                                                   cudaMemcpyDeviceToHost, faces));
                }
                // Only outgoing faces are synchronized; the bulk GPU kernel
                // continues while the host drives MPI progress in MPI_Waitall.
                CUDA_CHECK(cudaStreamSynchronize(faces));
                for (int side = 0; side < 2; ++side) {
                    const int neighbor = side == 0 ? lower : upper;
                    if (neighbor == MPI_PROC_NULL) continue;
                    const Real* send = deviceMPI ? output + (side == 0 ? plane : localZ * plane)
                                                : staging + side * plane;
                    for (size_t offset = 0; offset < plane;) {
                        const int chunk = static_cast<int>(std::min<size_t>(plane - offset, INT_MAX));
                        MPI_CHECK(MPI_Isend(send + offset, chunk, MPI_DOUBLE, neighbor, side,
                                            comm, &requests[requestCount++]));
                        offset += chunk;
                    }
                }
                MPI_CHECK(MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE));
                if (!deviceMPI) {
                    if (lower != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(output, staging + 2 * plane, faceBytes, cudaMemcpyHostToDevice, faces));
                    if (upper != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(output + (localZ + 1) * plane, staging + 3 * plane,
                                                   faceBytes, cudaMemcpyHostToDevice, faces));
                }
                CUDA_CHECK(cudaEventRecord(halosReady, faces));
                CUDA_CHECK(cudaStreamWaitEvent(compute, halosReady, 0));
                CUDA_CHECK(cudaStreamWaitEvent(faces, bulkDone, 0));
            }
            std::swap(input, output);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (worldRank == 0) {
        const double updates = hasInterior ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0;
        std::printf("Computation time: %.3f ms\n", duration * 1000);
        std::printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }

    if (printResults || validate)
        CUDA_CHECK(cudaMemcpy(host.get(), input + plane, ownedCount * sizeof(Real), cudaMemcpyDeviceToHost));
    if (printResults) gatherResults(host.get(), ownedCount, plane, nz, ranks, comm);
    bool valid = true;
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        valid = validateResult(host.get(), ownedCount, beginZ, nx, ny, nz, comm);
    }
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaEventDestroy(bulkDone));
    CUDA_CHECK(cudaEventDestroy(halosReady));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(faces));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    MPI_CHECK(MPI_Comm_free(&comm));
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    const int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (error != MPI_SUCCESS) {
        std::fprintf(stderr, "Cannot initialize MPI\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI must support MPI_THREAD_FUNNELED");
    int worldSize = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    int result = 1;
    try {
        result = run(argc, argv, worldSize);
    } catch (const std::exception& exception) {
        fail(exception.what());
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
