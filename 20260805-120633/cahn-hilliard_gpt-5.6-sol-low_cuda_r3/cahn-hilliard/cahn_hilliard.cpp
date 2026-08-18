#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error__ = (call);                                         \
    if (error__ != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error__));                              \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

// The x dimension is contiguous, so adjacent threads issue coalesced accesses.
// __restrict__ lets nvcc use the read-only cache aggressively for the stencil.
__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                             size_t i, size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = a[i];
    const double xm = x ? a[i - 1] : center;
    const double xp = x + 1 < nx ? a[i + 1] : center;
    const double ym = y ? a[i - nx] : center;
    const double yp = y + 1 < ny ? a[i + nx] : center;
    const double zm = z ? a[i - plane] : center;
    const double zp = z + 1 < nz ? a[i + plane] : center;
    // dx == dy == dz == 1 in this benchmark.
    return xm + xp + ym + yp + zm + zp - 6.0 * center;
}

__global__ void initialize_kernel(double* c, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        // Exactly the same deterministic initialization as the scalar program.
        const size_t value = ((i + 1) * static_cast<size_t>(1299709)) % n;
        c[i] = -1.0 + 2.0 * (static_cast<double>(value) / static_cast<double>(n));
    }
}

__global__ void chemical_potential_kernel(const double* __restrict__ c,
                                           double* __restrict__ mu,
                                           size_t nx, size_t ny, size_t nz,
                                           size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t plane = nx * ny;
    const size_t z = i / plane;
    const size_t rem = i - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const double cv = c[i];
    // Algebraically identical to the original expression with its fixed
    // physical constants; written explicitly to preserve operation ordering.
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB =  (2.0 / 9.0);
    constexpr double gamma = 0.5;
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                   2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
            gamma * laplacian(c, i, x, y, z, nx, ny, nz);
}

__global__ void update_kernel(double* __restrict__ cnew,
                              const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              size_t nx, size_t ny, size_t nz, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t plane = nx * ny;
    const size_t z = i / plane;
    const size_t rem = i - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    constexpr double dt_D = 0.01;
    cnew[i] = cold[i] + dt_D * laplacian(mu, i, x, y, z, nx, ny, nz);
}

bool validateResult(const std::vector<double>& c) {
    for (double val : c) {
        if (!std::isfinite(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto bounds = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 10.0 || *bounds.first < -10.0) {
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
    if (!nx || !ny || !nz || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t n = nx * ny * nz;
    if (n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid allocation is too large\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, n * sizeof(double)));
    constexpr unsigned threads = 256;
    const size_t block_count = (n + threads - 1) / threads;
    if (block_count > static_cast<size_t>(std::numeric_limits<unsigned>::max())) {
        std::fprintf(stderr, "Grid is too large for a CUDA launch\n");
        return 1;
    }
    const unsigned blocks = static_cast<unsigned>(block_count);
    std::printf("Initializing concentration field...\n");
    initialize_kernel<<<blocks, threads>>>(d_cold, n);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iterations; ++t) {
        chemical_potential_kernel<<<blocks, threads>>>(d_cold, d_mu, nx, ny, nz, n);
        update_kernel<<<blocks, threads>>>(d_cnew, d_cold, d_mu, nx, ny, nz, n);
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf("Computation time: %.3f ms\n", elapsed_ms);
    const double mcups = elapsed_ms > 0.0f ? static_cast<double>(n) * iterations / (elapsed_ms * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int result = 0;
    if (printResults || validate) {
        std::vector<double> output(n);
        CUDA_CHECK(cudaMemcpy(output.data(), d_cold, n * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) print_results(output, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(output);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));
    return result;
}
