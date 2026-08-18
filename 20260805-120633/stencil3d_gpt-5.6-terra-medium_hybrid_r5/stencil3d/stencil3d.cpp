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

#define CUDA_CHECK(call) do {                                                         \
    const cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                     cudaGetErrorString(error_));                                     \
        MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
    }                                                                                 \
} while (0)

__host__ __device__ inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t localNz, size_t globalZ0,
                              size_t globalNz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || localZ >= localNz) return;

    const size_t z = localZ + 1;  // plane zero is the lower MPI halo
    const size_t globalZ = globalZ0 + localZ;
    const size_t index = idx3(x, y, z, nx, ny);
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || globalZ == 0 || globalZ + 1 == globalNz) {
        output[index] = input[index];
        return;
    }
    const size_t plane = nx * ny;
    output[index] = (input[index] + input[index - 1] + input[index + 1] +
                     input[index - nx] + input[index + nx] +
                     input[index - plane] + input[index + plane]) * (1.0 / 7.0);
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    bool badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArgs = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (badArgs || nx == 0 || ny == 0 || nz == 0 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (rank == 0) std::fprintf(stderr, "Invalid dimensions/iterations (MPI ranks must not exceed Z dimension).\n");
        MPI_Finalize(); return 1;
    }

    // Assign one accelerator per node-local MPI rank.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, gpuCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    if (gpuCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % gpuCount));
    MPI_Comm_free(&localComm);

    const size_t baseNz = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < extra);
    const size_t globalZ0 = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    const size_t localCells = (localNz + 2) * plane;
    // Pinned buffers make the device-to-host halo staging transfers DMA-capable.
    Real *hostA = nullptr, *hostB = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostA, localCells * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hostB, localCells * sizeof(Real)));

    // The global-index formula preserves the original deterministic initialization exactly.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < localNz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = idx3(x, y, globalZ0 + z, nx, ny);
                hostA[idx3(x, y, z + 1, nx, ny)] = static_cast<Real>(globalIndex % 19);
            }

    Real *deviceA = nullptr, *deviceB = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, localCells * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceB, localCells * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, localCells * sizeof(Real), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d, MPI ranks: %d, validation: %s\n", iterations, ranks, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    Real* deviceIn = deviceA;
    Real* deviceOut = deviceB;
    Real* hostIn = hostA;
    Real* hostOut = hostB;
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (localNz + block.z - 1) / block.z);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        // Stage just the two boundary planes; this works with both CUDA-aware and ordinary MPI.
        CUDA_CHECK(cudaMemcpyAsync(hostIn + plane, deviceIn + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hostIn + localNz * plane, deviceIn + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Sendrecv(hostIn + plane, static_cast<int>(plane), MPI_DOUBLE, previous, 11,
                     hostIn + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hostIn + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 12,
                     hostIn, static_cast<int>(plane), MPI_DOUBLE, previous, 12,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (previous != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(deviceIn, hostIn, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(deviceIn + (localNz + 1) * plane, hostIn + (localNz + 1) * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, stream));
        stencilKernel<<<grid, block, 0, stream>>>(deviceIn, deviceOut, nx, ny, localNz, globalZ0, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(deviceIn, deviceOut);
        std::swap(hostIn, hostOut);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(hostIn + plane, deviceIn + plane, localNz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<int> counts, displacements;
    std::vector<Real> globalGrid;
    if (rank == 0) { counts.resize(ranks); displacements.resize(ranks); globalGrid.resize(nx * ny * nz); }
    const int sendCount = static_cast<int>(localNz * plane);
    MPI_Gather(&sendCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) for (int r = 1; r < ranks; ++r) displacements[r] = displacements[r - 1] + counts[r - 1];
    MPI_Gatherv(hostIn + plane, sendCount, MPI_DOUBLE, rank == 0 ? globalGrid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0.0 ? updates / seconds / 1e6 : 0.0);
        if (printResults) print_results(globalGrid, "Grid");
        if (validate) {
            const auto [minIt, maxIt] = std::minmax_element(globalGrid.begin(), globalGrid.end());
            const bool finite = std::all_of(globalGrid.begin(), globalGrid.end(), [](Real v) { return std::isfinite(v); });
            std::printf("Value range: [%.6f, %.6f]\n", *minIt, *maxIt);
            result = (!finite || *maxIt > 1e6 || *minIt < -1e6);
            std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFreeHost(hostA));
    CUDA_CHECK(cudaFreeHost(hostB));
    MPI_Finalize();
    return result;
}
