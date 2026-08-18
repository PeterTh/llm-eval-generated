#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int threadsPerBlock = 256;
constexpr dim3 stencilBlock(32, 4, 2);

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double laplacian(const double* const field, const size_t index,
                                             const size_t x, const size_t y, const size_t z,
                                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);

    const double center = field[index];
    const double cxx = field[z * plane + y * nx + xp] + field[z * plane + y * nx + xn] - 2.0 * center;
    const double cyy = field[z * plane + yp * nx + x] + field[z * plane + yn * nx + x] - 2.0 * center;
    const double czz = field[zp * plane + y * nx + x] + field[zn * plane + y * nx + x] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void initializeConcentration(double* const concentration, const size_t volume) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < volume) {
        const double pseudo = ((index + 1) * static_cast<size_t>(1299709) % volume) /
                              static_cast<double>(volume);
        concentration[index] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotential(const double* const concentration, double* const chemicalPotential,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double gamma, const double eAA, const double eBB, const double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = z * nx * ny + y * nx + x;
        const double c = concentration[index];
        chemicalPotential[index] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB)
                                 + 3.0 * c + c * c * c
                                 - gamma * laplacian(concentration, index, x, y, z, nx, ny, nz);
    }
}

__global__ void cahnHilliardUpdate(double* const nextConcentration, const double* const concentration,
                                   const double* const chemicalPotential,
                                   const size_t nx, const size_t ny, const size_t nz,
                                   const double diffusionTimeStep) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = z * nx * ny + y * nx + x;
        nextConcentration[index] = concentration[index]
                                 + diffusionTimeStep * laplacian(chemicalPotential, index, x, y, z, nx, ny, nz);
    }
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto [minIt, maxIt] = std::minmax_element(concentration.begin(), concentration.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *minIt, *maxIt);
    if (*maxIt > 10.0 || *minIt < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid positive values.\n");
        return 1;
    }

    const size_t volume = nx * ny * nz;
    const size_t initBlocksRequired = (volume + threadsPerBlock - 1) / threadsPerBlock;
    const size_t gridX = (nx + stencilBlock.x - 1) / stencilBlock.x;
    const size_t gridY = (ny + stencilBlock.y - 1) / stencilBlock.y;
    const size_t gridZ = (nz + stencilBlock.z - 1) / stencilBlock.z;
    if (initBlocksRequired > std::numeric_limits<unsigned int>::max() ||
        gridX > std::numeric_limits<unsigned int>::max() || gridY > std::numeric_limits<unsigned int>::max() ||
        gridZ > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Grid is too large for CUDA launch dimensions.\n");
        return 1;
    }
    const dim3 initBlocks(static_cast<unsigned int>(initBlocksRequired));
    const dim3 stencilGrid(static_cast<unsigned int>(gridX), static_cast<unsigned int>(gridY),
                           static_cast<unsigned int>(gridZ));

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusionTimeStep = dt;
    const size_t bytes = volume * sizeof(double);

    double* cold = nullptr;
    double* cnew = nullptr;
    double* chemicalPotential = nullptr;
    checkCuda(cudaMalloc(&cold, bytes), "allocating concentration buffer");
    checkCuda(cudaMalloc(&cnew, bytes), "allocating update buffer");
    checkCuda(cudaMalloc(&chemicalPotential, bytes), "allocating chemical potential buffer");

    std::printf("Initializing concentration field...\n");
    initializeConcentration<<<initBlocks, threadsPerBlock>>>(cold, volume);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration field");

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start{};
    cudaEvent_t end{};
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential<<<stencilGrid, stencilBlock>>>(cold, chemicalPotential, nx, ny, nz,
                                                               gamma, eAA, eBB, eAB);
        checkCuda(cudaGetLastError(), "launching chemical potential kernel");
        cahnHilliardUpdate<<<stencilGrid, stencilBlock>>>(cnew, cold, chemicalPotential, nx, ny, nz,
                                                        diffusionTimeStep);
        checkCuda(cudaGetLastError(), "launching update kernel");
        std::swap(cold, cnew);
    }
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "synchronizing simulation");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, end), "measuring elapsed time");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    const double mcups = elapsedMilliseconds > 0.0F
                             ? static_cast<double>(volume) * iterations / elapsedMilliseconds / 1.0e3
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> result(volume);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying result to host");
        if (printResults) print_results(result, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(result);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            checkCuda(cudaFree(cold), "freeing concentration buffer");
            checkCuda(cudaFree(cnew), "freeing update buffer");
            checkCuda(cudaFree(chemicalPotential), "freeing chemical potential buffer");
            return valid ? 0 : 1;
        }
    }

    checkCuda(cudaFree(cold), "freeing concentration buffer");
    checkCuda(cudaFree(cnew), "freeing update buffer");
    checkCuda(cudaFree(chemicalPotential), "freeing chemical potential buffer");
    return 0;
}
