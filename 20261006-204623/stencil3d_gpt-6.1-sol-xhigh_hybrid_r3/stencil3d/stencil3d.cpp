#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

[[noreturn]] static void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        fail("CUDA operation failed");
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fail(message);
    }
}
#define MPI_CHECK(operation) mpiCheck((operation))

// Only owned planes are gathered. Both grids retain their initially fixed
// physical boundaries, eliminating the original per-iteration boundary copy.
struct DeviceGrid {
    Real* grids[2] = {nullptr, nullptr};
    Real* staging = nullptr;
    cudaStream_t compute = nullptr, boundary = nullptr;
    cudaEvent_t interiorDone = nullptr, haloReady = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    ~DeviceGrid() {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        if (interiorDone) cudaEventDestroy(interiorDone);
        if (haloReady) cudaEventDestroy(haloReady);
        if (compute) cudaStreamDestroy(compute);
        if (boundary) cudaStreamDestroy(boundary);
        if (staging) cudaFreeHost(staging);
        for (Real* grid : grids) if (grid) cudaFree(grid);
    }
};

// All stencil sums are in [0,126]. Correct a reciprocal product with its exact
// FMA residual. Division by seven cannot land on a binary rounding midpoint:
// its distance is at least 1/14 ulp, much larger than the correction's error.
// The slow path also covers zero and values too small for a normal residual.
__device__ __forceinline__ Real divideBySeven(Real sum) {
    if (sum < 0x1p-900) return __ddiv_rn(sum, 7.0);
    constexpr Real reciprocal = 0x1.2492492492492p-3;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real residual = __fma_rn(-quotient, 7.0, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

// A warp spans 32 adjacent X positions. Advancing along Z retains three planes
// in registers; the read-only cache shares X/Y neighbors. Short Z tiles leave
// enough independent blocks to fill large accelerators.
template<int Depth, typename Index>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              Index nx, Index ny, Index plane,
                              Index zBegin, Index zEnd, Index zStep,
                              Index zTiles) {
    const Index x = Index(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    for (Index tile = blockIdx.z; tile < zTiles; tile += gridDim.z) {
        const Index z = zBegin + tile * zStep;
        for (Index y = Index(blockIdx.y) * blockDim.y + threadIdx.y + 1;
             y < ny - 1; y += Index(gridDim.y) * blockDim.y) {
            Index index = z * plane + y * nx + x;
            Real bottom = __ldg(input + index - plane);
            Real center = __ldg(input + index);
            Real top = __ldg(input + index + plane);
#pragma unroll
            for (int k = 0; k < Depth; ++k) {
                if (z + k >= zEnd) break;
                // Preserve the original summation order and correctly rounded
                // division; reciprocal multiplication changes result hashes.
                const Real sum = center + __ldg(input + index - 1)
                    + __ldg(input + index + 1) + __ldg(input + index - nx)
                    + __ldg(input + index + nx) + bottom + top;
                output[index] = divideBySeven(sum);
                if (k + 1 < Depth && z + k + 1 < zEnd) {
                    bottom = center;
                    center = top;
                    index += plane;
                    top = __ldg(input + index + plane);
                }
            }
        }
    }
}

// Share each XY plane explicitly for large slabs. Register rotation supplies
// Z neighbors; only the tile's narrow XY halos need additional global reads.
template<int Depth, typename Index>
__global__ void tiledStencilKernel(const Real* __restrict__ input,
                                   Real* __restrict__ output,
                                   Index nx, Index ny, Index plane,
                                   Index zBegin, Index zEnd, Index zStep,
                                   Index zTiles) {
    __shared__ Real tile[10][34];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const Index x = Index(blockIdx.x) * 32 + tx + 1;
    const Index yTiles = (ny - 3) / 8 + 1;
    for (Index zTile = blockIdx.z; zTile < zTiles; zTile += gridDim.z) {
        const Index z = zBegin + zTile * zStep;
        for (Index yTile = blockIdx.y; yTile < yTiles; yTile += gridDim.y) {
            const Index y = yTile * 8 + ty + 1;
            const bool valid = x < nx && y < ny;
            Index index = z * plane + y * nx + x;
            Real bottom = valid ? __ldg(input + index - plane) : 0;
            Real center = valid ? __ldg(input + index) : 0;
            Real top = valid ? __ldg(input + index + plane) : 0;
#pragma unroll
            for (int k = 0; k < Depth; ++k) {
                if (z + k >= zEnd) break;
                tile[ty + 1][tx + 1] = center;
                if (tx == 0)
                    tile[ty + 1][0] = y < ny ? __ldg(input + index - 1) : 0;
                if (tx == 31)
                    tile[ty + 1][33] = x + 1 < nx && y < ny ? __ldg(input + index + 1) : 0;
                if (ty == 0)
                    tile[0][tx + 1] = x < nx ? __ldg(input + index - nx) : 0;
                if (ty == 7)
                    tile[9][tx + 1] = x < nx && y + 1 < ny ? __ldg(input + index + nx) : 0;
                __syncthreads();
                if (x < nx - 1 && y < ny - 1) {
                    const Real sum = center + tile[ty + 1][tx] + tile[ty + 1][tx + 2]
                        + tile[ty][tx + 1] + tile[ty + 2][tx + 1] + bottom + top;
                    output[index] = divideBySeven(sum);
                }
                __syncthreads();
                if (k + 1 < Depth && z + k + 1 < zEnd) {
                    bottom = center;
                    center = top;
                    index += plane;
                    top = valid ? __ldg(input + index + plane) : 0;
                }
            }
        }
    }
}

template<int Depth, typename Index>
static void launchTyped(const Real* input, Real* output, Index nx, Index ny,
                        Index begin, Index end, Index step, Index tiles,
                        dim3 grid, dim3 block, cudaStream_t stream) {
    if constexpr (Depth > 1) {
        if (nx >= 34 && ny >= 10 && size_t(nx) * ny * (end - begin) >= 4 * 1024 * 1024) {
            tiledStencilKernel<Depth, Index><<<grid, block, 0, stream>>>(
                input, output, nx, ny, nx * ny, begin, end, step, tiles);
            return;
        }
    }
    stencilKernel<Depth, Index><<<grid, block, 0, stream>>>(
        input, output, nx, ny, nx * ny, begin, end, step, tiles);
}

template<int Depth>
static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t begin, size_t end, size_t step,
                          cudaStream_t stream) {
    if (nx <= 2 || ny <= 2 || begin >= end) return;
    const size_t tiles = (end - begin - 1) / step + 1;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((nx - 3) / block.x + 1),
                    static_cast<unsigned>(std::min<size_t>((ny - 3) / block.y + 1, 65535)),
                    static_cast<unsigned>(std::min<size_t>(tiles, 65535)));
    // Most slabs fit 32-bit element indices. Avoid wide integer arithmetic
    // there, retaining the wide-index kernel for larger device allocations.
    if (end + 1 <= UINT_MAX / ny / nx) {
        launchTyped<Depth, unsigned>(input, output, unsigned(nx), unsigned(ny),
            unsigned(begin), unsigned(end), unsigned(step), unsigned(tiles), grid, block, stream);
    } else {
        launchTyped<Depth, size_t>(input, output, nx, ny, begin, end, step, tiles, grid, block, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void initializeGrid(Real* grid, size_t nx, size_t ny, size_t nz,
                           size_t zStart, size_t localZ) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static) if ((localZ + 2) * plane >= 65536)
    for (size_t z = 0; z < localZ + 2; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const bool valid = zStart + z > 0 && zStart + z <= nz;
            const size_t globalRow = valid ? (zStart + z - 1) * plane + y * nx : 0;
            const size_t localRow = z * plane + y * nx;
#pragma omp simd
            for (size_t x = 0; x < nx; ++x)
                grid[localRow + x] = valid ? Real((globalRow + x) % 19) : 0.0;
        }
    }
}

// Chunk messages instead of narrowing size_t to MPI's int count. Identical
// tags are safe because MPI preserves the order of messages from each peer.
static void postReceive(Real* buffer, size_t count, int peer, int tag,
                        MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Irecv(buffer + offset, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back()));
        offset += chunk;
    }
}

static void postSend(const Real* buffer, size_t count, int peer, int tag,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Isend(buffer + offset, chunk, MPI_DOUBLE, peer, tag, comm, &requests.back()));
        offset += chunk;
    }
}

static void waitRequests(std::vector<MPI_Request>& requests) {
    MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    requests.clear();
}

static void gatherResults(const std::vector<Real>& local, size_t plane, size_t nz,
                          int rank, int ranks, MPI_Comm comm) {
    std::vector<Real> result;
    std::vector<MPI_Request> requests;
    if (rank == 0) {
        result.resize(plane * nz);
        std::copy(local.begin(), local.end(), result.begin());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t count = nz / ranks + (size_t(peer) < nz % ranks);
            const size_t start = size_t(peer) * (nz / ranks) + std::min<size_t>(peer, nz % ranks);
            postReceive(result.data() + start * plane, count * plane, peer, 200, comm, requests);
        }
    } else {
        postSend(local.data(), local.size(), 0, 200, comm, requests);
    }
    waitRequests(requests);
    if (rank == 0) print_results(result, "Grid");
}

static bool validateResult(const std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t nz, size_t zStart, MPI_Comm comm, int rank) {
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -std::numeric_limits<Real>::infinity();
    int invalid = 0, badBoundary = 0;
    const size_t plane = nx * ny;
#pragma omp parallel for schedule(static) reduction(min:minValue) reduction(max:maxValue) reduction(|:invalid,badBoundary) if (grid.size() >= 65536)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        invalid |= !std::isfinite(value);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        const size_t x = i % nx, y = (i / nx) % ny, z = zStart + i / plane;
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1)
            badBoundary |= value != Real((zStart * plane + i) % 19);
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minValue, 1, MPI_DOUBLE, MPI_MIN, comm));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxValue, 1, MPI_DOUBLE, MPI_MAX, comm));
    int errors[2] = {invalid, badBoundary};
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_INT, MPI_BOR, comm));
    const bool valid = !errors[0] && !errors[1] && minValue >= -1e6 && maxValue <= 1e6;
    if (rank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
        if (errors[0]) std::printf("Validation failed: found NaN or Inf value\n");
        if (errors[1]) std::printf("Validation failed: boundary values changed\n");
        if (minValue < -1e6 || maxValue > 1e6)
            std::printf("Validation failed: values out of expected range\n");
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

static int runBenchmark(size_t nx, size_t ny, size_t nz, int iterations,
                        bool validate, bool printResults, MPI_Comm comm) {
    int rank = 0, ranks = 0;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &ranks));
    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 0;
    MPI_CHECK(MPI_Comm_rank(node, &localRank));
    MPI_CHECK(MPI_Comm_size(node, &localRanks));
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("A CUDA device is required on every participating rank");
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&node));

    const size_t localZ = nz / ranks + (size_t(rank) < nz % ranks);
    const size_t zStart = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
    const size_t plane = nx * ny;
    if (localZ + 2 > std::numeric_limits<size_t>::max() / sizeof(Real) / plane)
        fail("Local grid size overflows address space");
    const size_t localCells = (localZ + 2) * plane;
    // Respect explicit thread counts. Bound default host teams by both the
    // available CPUs and the amount of work, avoiding costly tiny teams.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int cpuBudget = std::max(1, omp_get_num_procs() / localRanks);
        const int workBudget = static_cast<int>(std::min<size_t>(32, (localCells - 1) / 262144 + 1));
        omp_set_num_threads(std::min(cpuBudget, workBudget));
    }
    const size_t bytes = localCells * sizeof(Real);
    const size_t planeBytes = plane * sizeof(Real);
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const size_t first = zStart == 0 ? 2 : 1;
    const size_t end = zStart + localZ == nz ? localZ : localZ + 1;
    const bool hasWork = nx > 2 && ny > 2 && nz > 2;

    int cudaAware = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, comm));

    DeviceGrid device;
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithPriority(&device.boundary, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaEventCreateWithFlags(&device.interiorDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&device.haloReady, cudaEventDisableTiming));
    for (Real*& grid : device.grids) CUDA_CHECK(cudaMalloc(&grid, bytes));
    if (ranks > 1 && iterations > 1 && hasWork && !cudaAware) {
        if (planeBytes > std::numeric_limits<size_t>::max() / 4)
            fail("Halo staging size overflows address space");
        CUDA_CHECK(cudaMallocHost(&device.staging, 4 * planeBytes));
    }
    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks, %d OpenMP threads on rank 0, CUDA\n", ranks, omp_get_max_threads());
        std::printf("Initializing grid...\n");
    }
    {
        std::unique_ptr<Real[]> host(new Real[localCells]);
        initializeGrid(host.get(), nx, ny, nz, zStart, localZ);
        CUDA_CHECK(cudaMemcpy(device.grids[0], host.get(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device.grids[1], device.grids[0], bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(device.interiorDone, device.compute));
    CUDA_CHECK(cudaEventRecord(device.haloReady, device.boundary));

    Real* input = device.grids[0];
    Real* output = device.grids[1];
    // Reuse a two-step graph for a single slab. Graph construction is outside
    // the timed computation and preserves the double-buffer parity.
    if (ranks == 1 && hasWork && iterations >= 2) {
        CUDA_CHECK(cudaStreamBeginCapture(device.compute, cudaStreamCaptureModeThreadLocal));
        launchStencil<8>(input, output, nx, ny, first, end, 8, device.compute);
        launchStencil<8>(output, input, nx, ny, first, end, 8, device.compute);
        CUDA_CHECK(cudaStreamEndCapture(device.compute, &device.graph));
        CUDA_CHECK(cudaGraphInstantiate(&device.graphExec, device.graph, nullptr, nullptr, 0));
#if CUDART_VERSION >= 11010
        CUDA_CHECK(cudaGraphUpload(device.graphExec, device.compute));
#endif
    }
    std::vector<MPI_Request> requests;
    requests.reserve(4 * ((plane - 1) / INT_MAX + 1));
    if (rank == 0) std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(comm));
    const double startTime = MPI_Wtime();
    if (hasWork && ranks == 1) {
        int iter = 0;
        for (; iter + 1 < iterations; iter += 2)
            CUDA_CHECK(cudaGraphLaunch(device.graphExec, device.compute));
        if (iter < iterations) {
            launchStencil<8>(input, output, nx, ny, first, end, 8, device.compute);
            std::swap(input, output);
        }
    } else if (hasWork) {
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaStreamWaitEvent(device.compute, device.haloReady, 0));
            if (iter == iterations - 1) {
                // No rank needs the last output's ghost planes.
                launchStencil<8>(input, output, nx, ny, first, end, 8, device.compute);
                std::swap(input, output);
                break;
            }
            CUDA_CHECK(cudaStreamWaitEvent(device.boundary, device.interiorDone, 0));
            CUDA_CHECK(cudaStreamWaitEvent(device.boundary, device.haloReady, 0));
            const bool firstEdge = first <= 1 && 1 < end;
            const bool lastEdge = first <= localZ && localZ < end;
            const size_t edgeFirst = firstEdge ? 1 : (lastEdge ? localZ : 0);
            const size_t edgeEnd = lastEdge ? localZ + 1 : (firstEdge ? 2 : 0);
            launchStencil<1>(input, output, nx, ny, edgeFirst, edgeEnd,
                             std::max<size_t>(1, localZ - 1), device.boundary);
            launchStencil<8>(input, output, nx, ny, std::max<size_t>(first, 2),
                             std::min(end, localZ), 8, device.compute);
            CUDA_CHECK(cudaEventRecord(device.interiorDone, device.compute));

            if (!cudaAware) {
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(device.staging, output + plane, planeBytes,
                                               cudaMemcpyDeviceToHost, device.boundary));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(device.staging + plane, output + localZ * plane,
                                               planeBytes, cudaMemcpyDeviceToHost, device.boundary));
            }
            // MPI is called only by the main thread (MPI_THREAD_FUNNELED).
            // This synchronization also releases last step's host receive
            // buffers while the independent interior kernel keeps running.
            CUDA_CHECK(cudaStreamSynchronize(device.boundary));
            Real* receiveLower = cudaAware ? output : device.staging + 2 * plane;
            Real* receiveUpper = cudaAware ? output + (localZ + 1) * plane : device.staging + 3 * plane;
            const Real* sendLower = cudaAware ? output + plane : device.staging;
            const Real* sendUpper = cudaAware ? output + localZ * plane : device.staging + plane;
            postReceive(receiveLower, plane, lower, 102, comm, requests);
            postReceive(receiveUpper, plane, upper, 101, comm, requests);
            postSend(sendLower, plane, lower, 101, comm, requests);
            postSend(sendUpper, plane, upper, 102, comm, requests);
            waitRequests(requests);
            if (!cudaAware) {
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(output, receiveLower, planeBytes,
                                               cudaMemcpyHostToDevice, device.boundary));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(output + (localZ + 1) * plane, receiveUpper,
                                               planeBytes, cudaMemcpyHostToDevice, device.boundary));
            }
            CUDA_CHECK(cudaEventRecord(device.haloReady, device.boundary));
            std::swap(input, output);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(device.compute));
    CUDA_CHECK(cudaStreamSynchronize(device.boundary));
    const double elapsed = MPI_Wtime() - startTime;
    double maxElapsed = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        const double updates = hasWork ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0;
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0);
    }
    if (printResults || validate) {
        std::vector<Real> local(localZ * plane);
        CUDA_CHECK(cudaMemcpy(local.data(), input + plane, local.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) gatherResults(local, plane, nz, rank, ranks, comm);
        if (validate) {
            if (rank == 0) std::printf("Validating result...\n");
            return validateResult(local, nx, ny, nz, zStart, comm, rank) ? 0 : 1;
        }
    }
    return 0;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static size_t parseNumber(const char* text) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '\0' || *text == '-' || *end != '\0' || errno == ERANGE
        || value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Options require nonnegative integers");
    return static_cast<size_t>(value);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int rank = 0, ranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false;
    try {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-v") == 0) validate = true;
            else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
            else if (std::strcmp(argv[i], "-h") == 0) { help = true; break; }
            else if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0
                      || std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
                const char option = argv[i][1];
                const size_t value = parseNumber(argv[++i]);
                if (option == 'x') nx = value;
                else if (option == 'y') ny = value;
                else if (option == 'z') nz = value;
                else {
                    if (value > INT_MAX) throw std::invalid_argument("Iteration count exceeds INT_MAX");
                    iterations = static_cast<int>(value);
                }
            } else throw std::invalid_argument("Unknown option or missing option value");
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        if (!help && (nx == 0 || nx > std::numeric_limits<size_t>::max() / ny
                      || nx * ny > std::numeric_limits<size_t>::max() / sizeof(Real) / nz))
            throw std::invalid_argument("Invalid grid size or grid size overflows address space");
    } catch (const std::exception& error) {
        if (rank == 0) { std::fprintf(stderr, "%s\n", error.what()); printUsage(argv[0]); }
        MPI_Finalize();
        return 1;
    }
    if (help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    // Empty slabs do not allocate a GPU or enter the stencil communicator.
    const int activeRanks = static_cast<int>(std::min<size_t>(ranks, nz));
    MPI_Comm active;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active));
    int result = 0;
    if (active != MPI_COMM_NULL) {
        try {
            result = runBenchmark(nx, ny, nz, iterations, validate, printResults, active);
        } catch (const std::exception& error) {
            fail(error.what());
        }
        MPI_CHECK(MPI_Comm_free(&active));
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return result;
}
