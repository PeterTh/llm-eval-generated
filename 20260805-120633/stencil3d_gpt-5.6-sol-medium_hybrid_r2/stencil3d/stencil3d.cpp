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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t cuda_error_ = (call);                                     \
    if (cuda_error_ != cudaSuccess) {                                           \
        int rank_ = -1; MPI_Comm_rank(MPI_COMM_WORLD, &rank_);                  \
        std::fprintf(stderr, "Rank %d CUDA error at %s:%d: %s\n", rank_,      \
                     __FILE__, __LINE__, cudaGetErrorString(cuda_error_));       \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t plane,
                              size_t globalZ0, size_t globalNz,
                              size_t localZBegin, size_t localZEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t localZ = localZBegin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || localZ >= localZEnd) return;

    const size_t p = localZ * plane + y * nx + x;
    const size_t globalZ = globalZ0 + localZ - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        globalZ == 0 || globalZ + 1 == globalNz) {
        output[p] = input[p];
    } else {
        output[p] = (input[p] + input[p - 1] + input[p + 1] +
                     input[p - nx] + input[p + nx] +
                     input[p - plane] + input[p + plane]) / Real{7.0};
    }
}

static void launchRange(const Real* input, Real* output, size_t nx, size_t ny,
                        size_t globalZ0, size_t globalNz,
                        size_t localBegin, size_t localEnd,
                        cudaStream_t stream) {
    if (localBegin >= localEnd) return;
    constexpr dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (localEnd - localBegin + block.z - 1) / block.z);
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, nx * ny,
                                              globalZ0, globalNz,
                                              localBegin, localEnd);
    CUDA_CHECK(cudaGetLastError());
}

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny,
                           size_t localNz, size_t globalZ0) {
    const size_t plane = nx * ny;
#pragma omp parallel for collapse(2) schedule(static)
    for (long long lz = 0; lz < static_cast<long long>(localNz); ++lz) {
        for (long long y = 0; y < static_cast<long long>(ny); ++y) {
            const size_t globalBase = (globalZ0 + static_cast<size_t>(lz)) * plane +
                                      static_cast<size_t>(y) * nx;
            const size_t localBase = (static_cast<size_t>(lz) + 1) * plane +
                                     static_cast<size_t>(y) * nx;
            for (size_t x = 0; x < nx; ++x)
                grid[localBase + x] = static_cast<Real>((globalBase + x) % 19);
        }
    }
}

static bool validateResult(const std::vector<Real>& grid) {
    int bad = 0;
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
#pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        bad |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (bad) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); parseError = true; }
    }
    if (help || parseError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const bool invalid = nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
                         nz < static_cast<size_t>(ranks) ||
                         nx > std::numeric_limits<size_t>::max() / ny ||
                         nx * ny > static_cast<size_t>(std::numeric_limits<int>::max());
    if (invalid) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be >= 2, Z must be >= MPI ranks, iterations nonnegative, and an XY plane must fit MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "At least one CUDA GPU is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder);
    const size_t globalZ0 = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    if ((localNz + 2) > std::numeric_limits<size_t>::max() / plane ||
        nz > std::numeric_limits<size_t>::max() / plane) {
        if (rank == 0) std::fprintf(stderr, "Grid is too large for address space\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }

    std::vector<Real> hostLocal((localNz + 2) * plane);
    initializeGrid(hostLocal, nx, ny, localNz, globalZ0);
    Real *deviceA = nullptr, *deviceB = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, hostLocal.size() * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceB, hostLocal.size() * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(deviceA, hostLocal.data(), hostLocal.size() * sizeof(Real), cudaMemcpyHostToDevice));
    std::vector<Real>().swap(hostLocal); // do not retain a full host-side slab during the run

    Real *sendLower = nullptr, *sendUpper = nullptr, *recvLower = nullptr, *recvUpper = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendUpper, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLower, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvUpper, plane * sizeof(Real)));
    cudaStream_t computeStream, commStream;
    cudaEvent_t deepDone, edgeDone;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&deepDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&edgeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(deepDone, computeStream));
    CUDA_CHECK(cudaEventRecord(edgeDone, commStream));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Running stencil computation...\n");
    const double start = MPI_Wtime();
    Real* input = deviceA;
    Real* output = deviceB;
    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, edgeDone, 0));
        launchRange(input, output, nx, ny, globalZ0, nz, 2, localNz, computeStream);

        CUDA_CHECK(cudaStreamWaitEvent(commStream, deepDone, 0));
        CUDA_CHECK(cudaStreamWaitEvent(commStream, edgeDone, 0));
        MPI_Request requests[4];
        int requestCount = 0;
        const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
        if (prev != MPI_PROC_NULL)
            MPI_Irecv(recvLower, static_cast<int>(plane), MPI_DOUBLE, prev, 11, MPI_COMM_WORLD, &requests[requestCount++]);
        if (next != MPI_PROC_NULL)
            MPI_Irecv(recvUpper, static_cast<int>(plane), MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &requests[requestCount++]);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendLower, input + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, commStream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendUpper, input + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, commStream));
        CUDA_CHECK(cudaStreamSynchronize(commStream));
        if (prev != MPI_PROC_NULL)
            MPI_Isend(sendLower, static_cast<int>(plane), MPI_DOUBLE, prev, 10, MPI_COMM_WORLD, &requests[requestCount++]);
        if (next != MPI_PROC_NULL)
            MPI_Isend(sendUpper, static_cast<int>(plane), MPI_DOUBLE, next, 11, MPI_COMM_WORLD, &requests[requestCount++]);
        if (requestCount) MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input, recvLower, plane * sizeof(Real), cudaMemcpyHostToDevice, commStream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(input + (localNz + 1) * plane, recvUpper, plane * sizeof(Real), cudaMemcpyHostToDevice, commStream));
        launchRange(input, output, nx, ny, globalZ0, nz, 1, 2, commStream);
        if (localNz > 1)
            launchRange(input, output, nx, ny, globalZ0, nz, localNz, localNz + 1, commStream);
        CUDA_CHECK(cudaEventRecord(deepDone, computeStream));
        CUDA_CHECK(cudaEventRecord(edgeDone, commStream));
        std::swap(input, output);
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(commStream));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool needHostOutput = printResults || validate;
    std::vector<Real> owned;
    std::vector<int> counts, offsets;
    std::vector<Real> finalGrid;
    if (needHostOutput) {
        owned.resize(localNz * plane);
        CUDA_CHECK(cudaMemcpy(owned.data(), input + plane, owned.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (needHostOutput && rank == 0) {
        counts.resize(ranks); offsets.resize(ranks);
        size_t offset = 0;
        for (int r = 0; r < ranks; ++r) {
            const size_t rz = base + (static_cast<size_t>(r) < remainder);
            if (rz * plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                offset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "Gather size exceeds MPI int-count interface\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            counts[r] = static_cast<int>(rz * plane);
            offsets[r] = static_cast<int>(offset);
            offset += rz * plane;
        }
        finalGrid.resize(nz * plane);
    }
    if (needHostOutput)
        MPI_Gatherv(owned.data(), static_cast<int>(owned.size()), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? offsets.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double updates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventDestroy(deepDone)); CUDA_CHECK(cudaEventDestroy(edgeDone));
    CUDA_CHECK(cudaStreamDestroy(computeStream)); CUDA_CHECK(cudaStreamDestroy(commStream));
    CUDA_CHECK(cudaFreeHost(sendLower)); CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(recvLower)); CUDA_CHECK(cudaFreeHost(recvUpper));
    CUDA_CHECK(cudaFree(deviceA)); CUDA_CHECK(cudaFree(deviceB));
    MPI_Finalize();
    return result;
}
