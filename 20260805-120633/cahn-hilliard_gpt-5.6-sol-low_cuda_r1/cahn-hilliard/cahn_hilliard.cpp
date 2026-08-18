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

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error_ = (call);                                            \
    if (error_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(error_));                                 \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

// One-dimensional thread indexing makes the launch efficient for arbitrary grid
// shapes.  X is the unit-stride dimension, so neighboring threads coalesce.
__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                        double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t nz,
                                        double gamma,
                                        double e_AA, double e_BB, double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t zm = z ? i - plane : i;
    const size_t zp = z + 1 < nz ? i + plane : i;
    const double cv = c[i];
    const double lap = c[xm] + c[xp] + c[ym] + c[yp] + c[zm] + c[zp] - 6.0 * cv;

    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                   - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             size_t nx, size_t ny, size_t nz, double dtD) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xm = x ? i - 1 : i;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t zm = z ? i - plane : i;
    const size_t zp = z + 1 < nz ? i + plane : i;
    const double center = mu[i];
    const double lap = mu[xm] + mu[xp] + mu[ym] + mu[yp] + mu[zm] + mu[zp]
                       - 6.0 * center;
    cnew[i] = cold[i] + dtD * lap;
}

void initializeConcentration(std::vector<double>& c) {
    const size_t vol = c.size();
    for (size_t i = 0; i < vol; ++i) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    for (double val : c) {
        if (!std::isfinite(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto range = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *range.first, *range.second);
    if (*range.second > 10.0 || *range.first < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    if ((nx + 31) / 32 > std::numeric_limits<unsigned int>::max()
        || (ny + 3) / 4 > 65535 || (nz + 1) / 2 > 65535) {
        std::fprintf(stderr, "Grid dimensions exceed CUDA launch limits\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");

    std::vector<double> result(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(result);

    double *d_cold, *d_cnew, *d_mu;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, result.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 threads(32, 4, 2);
    const dim3 blocks(static_cast<unsigned>((nx + threads.x - 1) / threads.x),
                      static_cast<unsigned>((ny + threads.y - 1) / threads.y),
                      static_cast<unsigned>((nz + threads.z - 1) / threads.z));
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5, dtD = 0.01;
    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(d_cold, d_mu, nx, ny, nz,
                                                     gamma, e_AA, e_BB, e_AB);
        updateKernel<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, nz, dtD);
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(result.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double mcups = seconds > 0.0 ? static_cast<double>(gridSize) * iterations / seconds / 1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    if (printResults) print_results(result, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(result);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
