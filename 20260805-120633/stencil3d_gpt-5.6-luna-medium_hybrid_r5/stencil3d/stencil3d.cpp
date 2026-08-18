#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t localNz,
                              const size_t globalZStart, const size_t globalNz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= localNz) return;

    const size_t p = idx3(x, y, z, nx, ny);
    output[p] = input[p];
    const size_t globalZ = globalZStart + z - 1;
    if (x > 0 && x + 1 < nx && y > 0 && y + 1 < ny &&
        z > 0 && z + 1 < localNz && globalZ > 0 && globalZ + 1 < globalNz) {
        output[p] = (input[p] + input[p - 1] + input[p + 1] +
                     input[p - nx] + input[p + nx] +
                     input[p - nx * ny] + input[p + nx * ny]) / 7.0;
    }
}

static void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                           const size_t localNz, const size_t globalZStart) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(localNz); ++z) {
        const size_t globalZ = globalZStart + static_cast<size_t>(z) - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = idx3(x, y, static_cast<size_t>(z), nx, ny);
                const size_t globalIndex = globalZ * plane + y * nx + x;
                grid[p] = static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

static void exchangeHalos(std::vector<Real>& sendLower, std::vector<Real>& sendUpper,
                          std::vector<Real>& recvLower,
                          std::vector<Real>& recvUpper, const size_t nx, const size_t ny,
                          const int rank, const int ranks) {
    if (ranks == 1) return;
    const size_t plane = nx * ny;
    const int lower = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int upper = (rank + 1 == ranks) ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(sendLower.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 101,
                 recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE, upper, 101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendUpper.data(), static_cast<int>(plane), MPI_DOUBLE, upper, 102,
                 recvLower.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 102,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

static bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny,
                           const size_t ownedNz, const size_t globalZStart, const int rank) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localBad = 0;
    #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(|:localBad)
    for (long long z = 1; z <= static_cast<long long>(ownedNz); ++z) {
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const Real value = grid[idx3(x, y, static_cast<size_t>(z), nx, ny)];
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
            localBad |= (std::isnan(value) || std::isinf(value)) ? 1 : 0;
        }
    }
    Real minVal = 0.0, maxVal = 0.0;
    int bad = 0;
    MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (bad) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 1e6 || minVal < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            bad = 1;
        }
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    (void)globalZStart;
    return bad == 0;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

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
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz - 2) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be at least 3 and support one interior Z plane per MPI rank.\n");
        MPI_Finalize(); return 1;
    }

    const size_t interiorNz = nz - 2;
    const size_t base = interiorNz / static_cast<size_t>(ranks);
    const size_t remainder = interiorNz % static_cast<size_t>(ranks);
    const size_t ownedNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZStart = 1 + static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    const size_t localSize = (ownedNz + 2) * plane;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) MPI_Abort(MPI_COMM_WORLD, 1);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) MPI_Abort(MPI_COMM_WORLD, 1);
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    if (rank == 0) {
        std::printf("3D Stencil Benchmark (MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs: %d)\n", ranks, omp_get_max_threads(), deviceCount);
        std::printf("Grid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localSize), grid2(localSize);
    initializeGrid(grid1, nx, ny, ownedNz + 2, globalZStart);
    Real* d1 = nullptr; Real* d2 = nullptr;
    cudaCheck(cudaMalloc(&d1, localSize * sizeof(Real)), "cudaMalloc d1");
    cudaCheck(cudaMalloc(&d2, localSize * sizeof(Real)), "cudaMalloc d2");
    cudaCheck(cudaMemcpy(d1, grid1.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice), "initial copy");
    std::vector<Real> sendLower(plane), sendUpper(plane), recvLower(plane), recvUpper(plane);
    dim3 block(32, 4, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, ownedNz + 2);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        if (ranks > 1) {
            cudaCheck(cudaMemcpy(sendLower.data(), d1 + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "lower halo download");
            cudaCheck(cudaMemcpy(sendUpper.data(), d1 + ownedNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "upper halo download");
            exchangeHalos(sendLower, sendUpper, recvLower, recvUpper, nx, ny, rank, ranks);
            if (rank != 0) cudaCheck(cudaMemcpy(d1, recvLower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "lower halo upload");
            if (rank + 1 != ranks) cudaCheck(cudaMemcpy(d1 + (ownedNz + 1) * plane, recvUpper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "upper halo upload");
        }
        stencilKernel<<<grid, block>>>(d1, d2, nx, ny, ownedNz + 2, globalZStart, nz);
        cudaCheck(cudaGetLastError(), "stencil launch");
        cudaCheck(cudaDeviceSynchronize(), "stencil synchronize");
        std::swap(d1, d2);
    }
    cudaCheck(cudaMemcpy(grid1.data(), d1, localSize * sizeof(Real), cudaMemcpyDeviceToHost), "final copy");
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", updates / seconds / 1e6);
    }

    std::vector<Real> finalGrid;
    if (printResults) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t count = (base + (static_cast<size_t>(r) < remainder ? 1 : 0)) * plane;
            counts[r] = static_cast<int>(count);
            displacements[r] = (r == 0) ? 0 : displacements[r - 1] + counts[r - 1];
        }
        if (rank == 0) finalGrid.resize(nx * ny * nz);
        MPI_Gatherv(grid1.data() + plane, counts[rank], MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() + plane : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for
            for (long long z = 0; z < static_cast<long long>(nz); ++z)
                for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
                    if (z == 0 || z == static_cast<long long>(nz - 1)) finalGrid[idx3(x, y, static_cast<size_t>(z), nx, ny)] = static_cast<Real>(idx3(x, y, static_cast<size_t>(z), nx, ny) % 19);
            print_results(finalGrid, "Grid");
        }
    }
    const bool valid = validate ? validateResult(grid1, nx, ny, ownedNz, globalZStart, rank) : true;
    cudaFree(d1); cudaFree(d2);
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
