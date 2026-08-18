#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static void checkCuda(const cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t localNz,
                              const size_t globalZ0, const size_t globalNz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > localNz) return;

    const size_t p = idx3(x, y, z, nx, ny);
    const size_t globalZ = globalZ0 + z - 1;
    const bool boundary = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                          globalZ == 0 || globalZ + 1 == globalNz;
    if (boundary) {
        output[p] = input[p];
    } else {
        output[p] = (input[p] + input[p - 1] + input[p + 1] +
                     input[p - nx] + input[p + nx] +
                     input[p - nx * ny] + input[p + nx * ny]) / 7.0;
    }
}

static void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                           const size_t localNz, const size_t globalZ0) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(localNz + 2); ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const long long globalZ = static_cast<long long>(globalZ0) + z - 1;
                grid[static_cast<size_t>(z) * plane + y * nx + x] =
                    static_cast<Real>((static_cast<size_t>(globalZ) * plane + y * nx + x) % 19);
            }
        }
    }
}

static bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) return true;
    Real minVal = grid[0], maxVal = grid[0];
    int bad = 0;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(|:bad)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real value = grid[static_cast<size_t>(i)];
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
        bad |= std::isnan(value) || std::isinf(value);
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

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n");
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
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be >= 3 and iterations non-negative\n");
        MPI_Finalize(); return 1;
    }
    if (static_cast<size_t>(ranks) > nz) {
        if (rank == 0) std::fprintf(stderr, "MPI ranks (%d) cannot exceed the Z dimension (%zu)\n", ranks, nz);
        MPI_Finalize(); return 1;
    }

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalZ0 = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    const size_t localElements = (localNz + 2) * plane;
    std::vector<Real> host1(localElements), host2(localElements);
    initializeGrid(host1, nx, ny, localNz, globalZ0);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { std::fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    Real *device1 = nullptr, *device2 = nullptr;
    checkCuda(cudaMalloc(&device1, localElements * sizeof(Real)), "cudaMalloc");
    checkCuda(cudaMalloc(&device2, localElements * sizeof(Real)), "cudaMalloc");
    checkCuda(cudaMemcpy(device1, host1.data(), localElements * sizeof(Real), cudaMemcpyHostToDevice), "initial copy");
    std::vector<Real> lowerSend(plane), upperSend(plane), lowerRecv(plane), upperRecv(plane);
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
                    (localNz + block.z - 1) / block.z);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Running MPI(%d) + OpenMP + CUDA stencil computation...\n", ranks);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        stencilKernel<<<grid, block>>>(device1, device2, nx, ny, localNz, globalZ0, nz);
        checkCuda(cudaGetLastError(), "stencilKernel launch");
        checkCuda(cudaDeviceSynchronize(), "stencilKernel synchronization");
        // Stage halo planes through host memory so this also works with MPI builds
        // that do not provide CUDA-aware device-buffer communication.
        if (localNz != 0) {
            if (rank > 0) {
                checkCuda(cudaMemcpy(lowerSend.data(), device2 + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "lower halo copy");
                MPI_Sendrecv(lowerSend.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 10,
                             lowerRecv.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                checkCuda(cudaMemcpy(device2, lowerRecv.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "lower halo restore");
            }
            if (rank + 1 < ranks) {
                checkCuda(cudaMemcpy(upperSend.data(), device2 + localNz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost), "upper halo copy");
                MPI_Sendrecv(upperSend.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 11,
                             upperRecv.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                checkCuda(cudaMemcpy(device2 + (localNz + 1) * plane, upperRecv.data(), plane * sizeof(Real), cudaMemcpyHostToDevice), "upper halo restore");
            }
        }
        std::swap(device1, device2);
    }
    checkCuda(cudaMemcpy(host1.data(), device1, localElements * sizeof(Real), cudaMemcpyDeviceToHost), "final copy");
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(rz * plane);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * plane);
    }
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx * ny * nz);
    MPI_Gatherv(localNz ? host1.data() + plane : nullptr, counts[rank], MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double cells = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        const double safeElapsed = std::max(maxElapsed, std::numeric_limits<double>::min());
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",
                    maxElapsed * 1000.0, cells / safeElapsed / 1e6);
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            checkCuda(cudaFree(device1), "cudaFree"); checkCuda(cudaFree(device2), "cudaFree");
            MPI_Finalize(); return valid ? 0 : 1;
        }
    }
    checkCuda(cudaFree(device1), "cudaFree");
    checkCuda(cudaFree(device2), "cudaFree");
    MPI_Finalize();
    return 0;
}
