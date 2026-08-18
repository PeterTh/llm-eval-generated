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

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error_ = (call);                                            \
    if (error_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(error_));                                 \
        MPI_Abort(MPI_COMM_WORLD, 2);                                             \
    }                                                                             \
} while (0)

static inline size_t checkedGridSize(size_t nx, size_t ny, size_t nz) {
    if (nx != 0 && ny > std::numeric_limits<size_t>::max() / nx)
        return 0;
    const size_t plane = nx * ny;
    if (plane != 0 && nz > std::numeric_limits<size_t>::max() / plane)
        return 0;
    return plane * nz;
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny,
                              size_t firstZ, size_t lastZ) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + firstZ;
    if (x >= nx - 1 || y >= ny - 1 || z > lastZ)
        return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                 input[i - nx] + input[i + nx] +
                 input[i - plane] + input[i + plane]) * (1.0 / 7.0);
}

static void launchStencil(const Real* input, Real* output, size_t nx, size_t ny,
                          size_t firstZ, size_t lastZ, cudaStream_t stream) {
    if (firstZ > lastZ || nx <= 2 || ny <= 2)
        return;
    const dim3 block(32, 4, 1);
    const dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                    static_cast<unsigned>((ny - 2 + block.y - 1) / block.y),
                    static_cast<unsigned>(lastZ - firstZ + 1));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, firstZ, lastZ);
    CUDA_CHECK(cudaGetLastError());
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>  Grid size in X (default: 128)\n");
    std::printf("  -y <num>  Grid size in Y (default: same as X)\n");
    std::printf("  -z <num>  Grid size in Z (default: same as X)\n");
    std::printf("  -i <num>  Iterations (default: 10)\n");
    std::printf("  -v        Validate result\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, argsOk = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else argsOk = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t globalSize = checkedGridSize(nx, ny, nz);
    argsOk = argsOk && nx >= 3 && ny >= 3 && nz >= 3 && iterations >= 0 &&
             globalSize != 0 && static_cast<size_t>(ranks) <= nz &&
             nx * ny <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (help || !argsOk) {
        if (rank == 0) {
            if (!argsOk) std::fprintf(stderr, "Grid dimensions must be at least 3, iterations nonnegative, and MPI ranks <= Z planes.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argsOk ? 0 : 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t owned = base + (static_cast<size_t>(rank) < extra);
    const size_t globalFirst = static_cast<size_t>(rank) * base +
                               std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    const size_t localPlanes = owned + 2;
    std::vector<Real> host(localPlanes * plane);

    // OpenMP initializes both owned cells and ghosts using their global linear index.
    #pragma omp parallel for schedule(static)
    for (long long lz = 0; lz < static_cast<long long>(localPlanes); ++lz) {
        const long long gz = static_cast<long long>(globalFirst) + lz - 1;
        for (size_t i = 0; i < plane; ++i) {
            const size_t globalIndex = static_cast<size_t>(std::max(0LL, gz)) * plane + i;
            host[static_cast<size_t>(lz) * plane + i] = static_cast<Real>(globalIndex % 19);
        }
    }

    Real *current = nullptr, *next = nullptr;
    const size_t bytes = host.size() * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&current, bytes));
    CUDA_CHECK(cudaMalloc(&next, bytes));
    CUDA_CHECK(cudaMemcpy(current, host.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(next, host.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Real *sendLower = nullptr, *sendUpper = nullptr;
    Real *recvLower = nullptr, *recvUpper = nullptr;
    const size_t planeBytes = plane * sizeof(Real);
    CUDA_CHECK(cudaMallocHost(&sendLower, planeBytes));
    CUDA_CHECK(cudaMallocHost(&sendUpper, planeBytes));
    CUDA_CHECK(cudaMallocHost(&recvLower, planeBytes));
    CUDA_CHECK(cudaMallocHost(&recvUpper, planeBytes));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, OpenMP host parallelism, CUDA GPUs\n", ranks);
        std::printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int following = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
    const size_t firstCompute = globalFirst == 0 ? 2 : 1;
    const size_t lastCompute = (globalFirst + owned == nz) ? owned - 1 : owned;
    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaMemcpyAsync(sendLower, current + plane, planeBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sendUpper, current + owned * plane, planeBytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Request requests[4];
        MPI_Irecv(recvLower, static_cast<int>(plane), MPI_DOUBLE, prev, 101, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recvUpper, static_cast<int>(plane), MPI_DOUBLE, following, 100, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(sendLower, static_cast<int>(plane), MPI_DOUBLE, prev, 100, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(sendUpper, static_cast<int>(plane), MPI_DOUBLE, following, 101, MPI_COMM_WORLD, &requests[3]);

        // Planes independent of incoming halos execute while MPI progresses.
        if (firstCompute < lastCompute && firstCompute + 1 <= lastCompute - 1)
            launchStencil(current, next, nx, ny, firstCompute + 1, lastCompute - 1, stream);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current, recvLower, planeBytes,
                                       cudaMemcpyHostToDevice, stream));
        if (following != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(current + (owned + 1) * plane, recvUpper, planeBytes,
                                       cudaMemcpyHostToDevice, stream));
        if (firstCompute <= lastCompute) {
            launchStencil(current, next, nx, ny, firstCompute, firstCompute, stream);
            if (lastCompute != firstCompute)
                launchStencil(current, next, nx, ny, lastCompute, lastCompute, stream);
        }
        std::swap(current, next);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<Real> ownedHost(owned * plane);
    CUDA_CHECK(cudaMemcpy(ownedHost.data(), current + plane, ownedHost.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<Real> global;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        global.resize(globalSize);
        counts.resize(ranks);
        displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t ro = base + (static_cast<size_t>(r) < extra);
            const size_t rf = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
            if (ro * plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                rf * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "Result is too large for MPI_Gatherv integer counts.\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            counts[r] = static_cast<int>(ro * plane);
            displacements[r] = static_cast<int>(rf * plane);
        }
    }
    MPI_Gatherv(ownedHost.data(), static_cast<int>(ownedHost.size()), MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
        if (printResults) print_results(global, "Grid");
        if (validate) {
            int invalid = 0;
            Real minValue = std::numeric_limits<Real>::infinity();
            Real maxValue = -std::numeric_limits<Real>::infinity();
            #pragma omp parallel for reduction(+:invalid) reduction(min:minValue) reduction(max:maxValue)
            for (long long i = 0; i < static_cast<long long>(global.size()); ++i) {
                const Real value = global[static_cast<size_t>(i)];
                invalid += !std::isfinite(value);
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
            std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
            status = invalid || minValue < -1e6 || maxValue > 1e6;
            std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(sendLower));
    CUDA_CHECK(cudaFreeHost(sendUpper));
    CUDA_CHECK(cudaFreeHost(recvLower));
    CUDA_CHECK(cudaFreeHost(recvUpper));
    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(next));
    MPI_Finalize();
    return status;
}
