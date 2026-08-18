#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned BX = 8;
constexpr unsigned BY = 8;
constexpr unsigned BZ = 4;

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t index3(const size_t x, const size_t y, const size_t z,
                                         const size_t nx, const size_t ny) {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ double laplacianFromTile(const double* tile, const unsigned sx,
                                                    const unsigned sy, const unsigned sz,
                                                    const unsigned tx, const unsigned ty,
                                                    const double idx2, const double idy2,
                                                    const double idz2) {
    const unsigned plane = tx * ty;
    const unsigned center = (sz * ty + sy) * tx + sx;
    const double c = tile[center];
    return (tile[center + 1] + tile[center - 1] - 2.0 * c) * idx2
         + (tile[center + tx] + tile[center - tx] - 2.0 * c) * idy2
         + (tile[center + plane] + tile[center - plane] - 2.0 * c) * idz2;
}

// Each block stages its 3-D stencil (including the one-cell halo) in shared memory.
// The X dimension is contiguous, so the main field accesses are coalesced.
template <bool ChemicalPotential>
__global__ void stencilKernel(const double* __restrict__ input,
                              const double* __restrict__ centerInput,
                              double* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx2, const double dy2, const double dz2,
                              const double gamma, const double e_AA, const double e_BB,
                              const double e_AB, const double coefficient) {
    constexpr unsigned TX = BX + 2;
    constexpr unsigned TY = BY + 2;
    constexpr unsigned TZ = BZ + 2;
    constexpr unsigned TILE_SIZE = TX * TY * TZ;
    __shared__ double tile[TILE_SIZE];

    const unsigned tid = threadIdx.z * blockDim.y * blockDim.x
                       + threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned p = tid; p < TILE_SIZE; p += threads) {
        const unsigned sx = p % TX;
        const unsigned sy = (p / TX) % TY;
        const unsigned sz = p / (TX * TY);
        const size_t baseX = static_cast<size_t>(blockIdx.x) * BX;
        const size_t baseY = static_cast<size_t>(blockIdx.y) * BY;
        const size_t baseZ = static_cast<size_t>(blockIdx.z) * BZ;
        // Handle the lower halo explicitly: unsigned subtraction would otherwise
        // wrap and incorrectly select the upper boundary cell.
        const size_t x = sx == 0 ? (baseX == 0 ? 0 : baseX - 1)
                                 : min(baseX + sx - 1, nx - 1);
        const size_t y = sy == 0 ? (baseY == 0 ? 0 : baseY - 1)
                                 : min(baseY + sy - 1, ny - 1);
        const size_t z = sz == 0 ? (baseZ == 0 ? 0 : baseZ - 1)
                                 : min(baseZ + sz - 1, nz - 1);
        tile[p] = input[index3(x, y, z, nx, ny)];
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BY + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const unsigned sx = threadIdx.x + 1;
    const unsigned sy = threadIdx.y + 1;
    const unsigned sz = threadIdx.z + 1;
    const size_t outputIndex = index3(x, y, z, nx, ny);
    const double lap = laplacianFromTile(tile, sx, sy, sz, TX, TY, dx2, dy2, dz2);

    if constexpr (ChemicalPotential) {
        const double cv = tile[(sz * TY + sy) * TX + sx];
        output[outputIndex] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                            + 3.0 * cv + cv * cv * cv - gamma * lap;
    } else {
        output[outputIndex] = centerInput[outputIndex] + coefficient * lap;
    }
}

__global__ void initializeKernel(double* c, const size_t count) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        const size_t pseudo = ((id + 1) * static_cast<size_t>(1299709)) % count;
        c[id] = -1.0 + 2.0 * (pseudo / static_cast<double>(count));
    }
}

void launchChemicalPotential(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
                             double gamma, double e_AA, double e_BB, double e_AB) {
    const dim3 block(BX, BY, BZ);
    const dim3 grid((static_cast<unsigned long long>(nx) + BX - 1) / BX,
                    (static_cast<unsigned long long>(ny) + BY - 1) / BY,
                    (static_cast<unsigned long long>(nz) + BZ - 1) / BZ);
    stencilKernel<true><<<grid, block>>>(c, nullptr, mu, nx, ny, nz, 1.0, 1.0, 1.0,
                                          gamma, e_AA, e_BB, e_AB, 0.0);
    cudaCheck(cudaGetLastError(), "chemical-potential kernel launch");
}

void launchUpdate(const double* cold, double* cnew, const double* mu,
                  size_t nx, size_t ny, size_t nz, double D, double dt) {
    const dim3 block(BX, BY, BZ);
    const dim3 grid((static_cast<unsigned long long>(nx) + BX - 1) / BX,
                    (static_cast<unsigned long long>(ny) + BY - 1) / BY,
                    (static_cast<unsigned long long>(nz) + BZ - 1) / BZ);
    stencilKernel<false><<<grid, block>>>(mu, cold, cnew, nx, ny, nz, 1.0, 1.0, 1.0,
                                           0.0, 0.0, 0.0, 0.0, D * dt);
    cudaCheck(cudaGetLastError(), "update kernel launch");
}

} // namespace

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

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minValue = c[0];
    double maxValue = c[0];
    for (const double value : c) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double gamma = 0.5, D = 1.0;
    const size_t gridSize = nx * ny * nz;
    std::vector<double> result(gridSize);
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    cudaCheck(cudaMalloc(&cold, gridSize * sizeof(double)), "cold allocation");
    cudaCheck(cudaMalloc(&cnew, gridSize * sizeof(double)), "cnew allocation");
    cudaCheck(cudaMalloc(&mu, gridSize * sizeof(double)), "mu allocation");

    std::printf("Initializing concentration field...\n");
    constexpr unsigned initBlock = 256;
    initializeKernel<<<(gridSize + initBlock - 1) / initBlock, initBlock>>>(cold, gridSize);
    cudaCheck(cudaGetLastError(), "initialization kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "initialization");

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start), "start event creation");
    cudaCheck(cudaEventCreate(&end), "end event creation");
    cudaCheck(cudaEventRecord(start), "start event recording");
    for (int t = 0; t < iterations; ++t) {
        launchChemicalPotential(cold, mu, nx, ny, nz, gamma, e_AA, e_BB, e_AB);
        launchUpdate(cold, cnew, mu, nx, ny, nz, D, dt);
        std::swap(cold, cnew);
    }
    cudaCheck(cudaEventRecord(end), "end event recording");
    cudaCheck(cudaEventSynchronize(end), "simulation completion");
    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, end), "elapsed-time measurement");
    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double mcups = static_cast<double>(gridSize) * iterations
                       / (static_cast<double>(elapsedMs) / 1000.0) / 1.0e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    cudaCheck(cudaMemcpy(result.data(), cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost),
              "result copy");
    if (printResults) print_results(result, "Concentration");
    cudaCheck(cudaEventDestroy(start), "start event destruction");
    cudaCheck(cudaEventDestroy(end), "end event destruction");
    cudaCheck(cudaFree(cold), "cold deallocation");
    cudaCheck(cudaFree(cnew), "cnew deallocation");
    cudaCheck(cudaFree(mu), "mu deallocation");

    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(result);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
