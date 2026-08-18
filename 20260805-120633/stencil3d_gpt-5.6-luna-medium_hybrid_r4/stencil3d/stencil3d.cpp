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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

[[noreturn]] static void cudaFailure(cudaError_t error, const char* expression, const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) cudaFailure(e, #call, __FILE__, __LINE__); } while (false)

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              size_t nx, size_t ny, size_t ownedNz, size_t globalZStart,
                              size_t globalNz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t localZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || localZ > ownedNz) return;

    const size_t plane = nx * ny;
    const size_t out = localZ * plane + y * nx + x;
    const bool boundary = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                          globalZStart == 0 && localZ == 1 ||
                          globalZStart + localZ == globalNz;
    if (boundary) {
        output[out] = input[out];
        return;
    }
    output[out] = (input[out] + input[out - 1] + input[out + 1] +
                   input[out - nx] + input[out + nx] +
                   input[out - plane] + input[out + plane]) / 7.0;
}

static void initializeSlab(std::vector<Real>& grid, size_t nx, size_t ny, size_t ownedNz,
                           size_t globalZStart) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(ownedNz); ++z) {
        const size_t globalZ = globalZStart + static_cast<size_t>(z);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t globalIndex = globalZ * plane + y * nx + x;
                grid[(static_cast<size_t>(z) + 1) * plane + y * nx + x] =
                    static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

static void exchangeHalos(Real* deviceGrid, size_t nx, size_t ny, size_t ownedNz,
                          int lower, int upper, int rank, std::vector<Real>& sendLower,
                          std::vector<Real>& sendUpper, std::vector<Real>& recvLower,
                          std::vector<Real>& recvUpper) {
    const size_t plane = nx * ny;
    if (lower != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(sendLower.data(), deviceGrid + plane,
                                                       plane * sizeof(Real), cudaMemcpyDeviceToHost));
    if (upper != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(sendUpper.data(), deviceGrid + ownedNz * plane,
                                                       plane * sizeof(Real), cudaMemcpyDeviceToHost));

    MPI_Sendrecv(lower == MPI_PROC_NULL ? nullptr : sendLower.data(), lower == MPI_PROC_NULL ? 0 : static_cast<int>(plane), MPI_DOUBLE,
                 lower, 101, upper == MPI_PROC_NULL ? nullptr : recvUpper.data(), upper == MPI_PROC_NULL ? 0 : static_cast<int>(plane), MPI_DOUBLE,
                 upper, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(upper == MPI_PROC_NULL ? nullptr : sendUpper.data(), upper == MPI_PROC_NULL ? 0 : static_cast<int>(plane), MPI_DOUBLE,
                 upper, 102, lower == MPI_PROC_NULL ? nullptr : recvLower.data(), lower == MPI_PROC_NULL ? 0 : static_cast<int>(plane), MPI_DOUBLE,
                 lower, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (lower != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(deviceGrid, recvLower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
    if (upper != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(deviceGrid + (ownedNz + 1) * plane, recvUpper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
    (void)rank;
}

static bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) return false;
    Real minVal = grid[0], maxVal = grid[0];
    bool finite = true;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:finite)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        if (!std::isfinite(value)) finite = false;
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }
    if (!finite) {
        std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return maxVal <= 1e6 && minVal >= -1e6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -x <num> grid X (default 128)\n"
                "  -y <num> grid Y (default X)\n  -z <num> grid Z (default X)\n"
                "  -i <num> iterations (default 10)\n  -v validate\n  -r print results\n  -h help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || nz < static_cast<size_t>(ranks)) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be at least 3 and Z must cover MPI ranks.\n");
        MPI_Finalize(); return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Finalize(); return 1; }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t ownedNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t globalZStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny;
    const size_t localSize = (ownedNz + 2) * plane;
    std::vector<Real> hostInput(localSize), hostOutput(localSize);
    initializeSlab(hostInput, nx, ny, ownedNz, globalZStart);
    Real* deviceInput = nullptr;
    Real* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceInput, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));

    std::vector<Real> sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventRecord(begin));
    const dim3 block(32, 4, 1);
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                    static_cast<unsigned>((ownedNz + block.z - 1) / block.z));
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(deviceInput, nx, ny, ownedNz, lower, upper, rank,
                      sendLower, sendUpper, recvLower, recvUpper);
        stencilKernel<<<grid, block>>>(deviceInput, deviceOutput, nx, ny, ownedNz, globalZStart, nz);
        CUDA_CHECK(cudaGetLastError());
        std::swap(deviceInput, deviceOutput);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    CUDA_CHECK(cudaMemcpy(hostInput.data(), deviceInput, localSize * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<int> counts, displacements;
    std::vector<Real> finalGrid;
    if (rank == 0) { counts.resize(ranks); displacements.resize(ranks); finalGrid.resize(nx * ny * nz); }
    int localCount = static_cast<int>(ownedNz * plane);
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            const size_t rOwned = base + (static_cast<size_t>(r) < extra ? 1 : 0);
            const size_t rStart = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
            counts[r] = static_cast<int>(rOwned * plane);
            displacements[r] = static_cast<int>(rStart * plane);
        }
    }
    MPI_Gatherv(hostInput.data() + plane, localCount, MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", milliseconds,
                    static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations /
                    (milliseconds / 1000.0) / 1e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) { std::printf("Validating result...\n"); const bool ok = validateResult(finalGrid); MPI_Finalize(); return ok ? 0 : 1; }
    }
    MPI_Finalize();
    return 0;
}
