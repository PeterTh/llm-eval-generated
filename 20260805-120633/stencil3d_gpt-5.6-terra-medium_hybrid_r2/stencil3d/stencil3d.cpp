#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do {                                                        \
    const cudaError_t error_ = (call);                                               \
    if (error_ != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                cudaGetErrorString(error_));                                         \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_));                         \
    }                                                                                 \
} while (0)

__global__ void initializeKernel(Real* grid, size_t nx, size_t ny, size_t localNz,
                                 size_t globalZ0) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < localNz) {
        const size_t globalIndex = (globalZ0 + z) * nx * ny + y * nx + x;
        grid[(z + 1) * nx * ny + y * nx + x] = static_cast<Real>(globalIndex % 19);
    }
}

// zBegin/zEnd use local physical coordinates [0, localNz).  Ghost planes are at 0
// and localNz+1, so the same kernel can process both the bulk and received edges.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t localNz, size_t globalZ0,
                              size_t globalNz, size_t zBegin, size_t zEnd) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = zBegin + static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= zEnd || z >= localNz) return;

    const size_t plane = nx * ny;
    const size_t i = (z + 1) * plane + y * nx + x;
    const size_t gz = globalZ0 + z;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == globalNz) {
        output[i] = input[i];
    } else {
        output[i] = (input[i] + input[i - 1] + input[i + 1] + input[i - nx] +
                     input[i + nx] + input[i - plane] + input[i + plane]) / 7.0;
    }
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (rank == 0) fprintf(stderr, "Dimensions must be at least 3, iterations non-negative, and MPI ranks no more than z dimension.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign GPUs by node-local rank; this works with arbitrary numbers of nodes.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device visible on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t base = nz / static_cast<size_t>(ranks), remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny, localCount = (localNz + 2) * plane;
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    Real *current = nullptr, *nextGrid = nullptr;
    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    CUDA_CHECK(cudaMalloc(&current, localCount * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&nextGrid, localCount * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendUpper, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvUpper, plane * sizeof(Real)));
    cudaStream_t computeStream, haloStream;
    cudaEvent_t ghostsReady;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ghostsReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaMemset(current, 0, localCount * sizeof(Real)));
    CUDA_CHECK(cudaMemset(nextGrid, 0, localCount * sizeof(Real)));
    const dim3 block(32, 4, 2);
    const dim3 initGrid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                        (localNz + block.z - 1) / block.z);
    initializeKernel<<<initGrid, block>>>(current, nx, ny, localNz, globalZ0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP host threads: %d\nInitializing grid...\nRunning stencil computation...\n", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        // Stage just two planes through pinned memory. This permits MPI stacks
        // without CUDA-aware transports while compute overlaps the MPI transfer.
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaMemcpyAsync(sendLower, current + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(sendUpper, current + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
        MPI_Request requests[4]; int requestCount = 0;
        MPI_Irecv(recvLower, static_cast<int>(plane), MPI_DOUBLE, prev, 17, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Irecv(recvUpper, static_cast<int>(plane), MPI_DOUBLE, next, 23, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendLower, static_cast<int>(plane), MPI_DOUBLE, prev, 23, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendUpper, static_cast<int>(plane), MPI_DOUBLE, next, 17, MPI_COMM_WORLD, &requests[requestCount++]);

        // While halos travel, update all planes that do not depend on them.
        const size_t bulkBegin = prev == MPI_PROC_NULL ? 0 : 1;
        const size_t bulkEnd = localNz - (next == MPI_PROC_NULL ? 0 : 1);
        if (bulkBegin < bulkEnd) {
            const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                            (bulkEnd - bulkBegin + block.z - 1) / block.z);
            stencilKernel<<<grid, block, 0, computeStream>>>(current, nextGrid, nx, ny, localNz, globalZ0, nz, bulkBegin, bulkEnd);
            CUDA_CHECK(cudaGetLastError());
        }
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current, recvLower, plane * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current + (localNz + 1) * plane, recvUpper, plane * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        CUDA_CHECK(cudaEventRecord(ghostsReady, haloStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, ghostsReady, 0));
        if (prev != MPI_PROC_NULL) {
            stencilKernel<<<dim3((nx + 31) / 32, (ny + 3) / 4, 1), block, 0, computeStream>>>(current, nextGrid, nx, ny, localNz, globalZ0, nz, 0, 1);
        }
        if (next != MPI_PROC_NULL && localNz > 1) {
            stencilKernel<<<dim3((nx + 31) / 32, (ny + 3) / 4, 1), block, 0, computeStream>>>(current, nextGrid, nx, ny, localNz, globalZ0, nz, localNz - 1, localNz);
        }
        CUDA_CHECK(cudaGetLastError());
        std::swap(current, nextGrid);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<Real> finalGrid;
    if (rank == 0 && (validate || printResults)) finalGrid.resize(nx * ny * nz);
    std::vector<int> recvCounts, displacements;
    if (rank == 0) { recvCounts.resize(ranks); displacements.resize(ranks); }
    const int sendCount = static_cast<int>(localNz * plane);
    MPI_Gather(&sendCount, 1, MPI_INT, rank == 0 ? recvCounts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) { for (int r = 1; r < ranks; ++r) displacements[r] = displacements[r - 1] + recvCounts[r - 1]; }
    // Result collection is deliberately host-staged: it is outside the timed path
    // and keeps -r/-v usable with MPI installations lacking CUDA-aware collectives.
    std::vector<Real> localResult;
    if (validate || printResults) {
        localResult.resize(localNz * plane);
        CUDA_CHECK(cudaMemcpy(localResult.data(), current + plane, localResult.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Gatherv(localResult.data(), sendCount, MPI_DOUBLE, rank == 0 ? finalGrid.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / maxElapsed / 1.0e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            bool valid = true; Real lo = std::numeric_limits<Real>::max(), hi = -lo;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:valid)
            for (size_t i = 0; i < finalGrid.size(); ++i) {
                const Real v = finalGrid[i]; lo = std::min(lo, v); hi = std::max(hi, v);
                valid = valid && std::isfinite(v);
            }
            printf("Value range: [%.6f, %.6f]\nValidation: %s\n", lo, hi,
                   valid && hi <= 1e6 && lo >= -1e6 ? "PASSED" : "FAILED");
            if (!(valid && hi <= 1e6 && lo >= -1e6)) { CUDA_CHECK(cudaFree(current)); CUDA_CHECK(cudaFree(nextGrid)); MPI_Finalize(); return 1; }
        }
    }
    CUDA_CHECK(cudaEventDestroy(ghostsReady));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaFreeHost(sendLower)); CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(recvLower)); CUDA_CHECK(cudaFreeHost(recvUpper));
    CUDA_CHECK(cudaFree(current)); CUDA_CHECK(cudaFree(nextGrid));
    MPI_Finalize();
    return 0;
}
