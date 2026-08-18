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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        checkCuda(cudaMalloc(&data_, count * sizeof(double)), "device allocation");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    double* get() const { return data_; }
private:
    double* data_ = nullptr;
    size_t count_;
};

// Each block updates a 3D tile.  The two passes stay separate because the
// chemical potential of every cell is required before its Laplacian is used.
__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                        double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t nz,
                                        double invDx2, double invDy2, double invDz2,
                                        double gamma, double eAA, double eBB,
                                        double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);
    const double cv = c[i];
    const double laplacian = (c[z * plane + y * nx + xp] + c[z * plane + y * nx + xn] - 2.0 * cv) * invDx2
                           + (c[z * plane + yp * nx + x] + c[z * plane + yn * nx + x] - 2.0 * cv) * invDy2
                           + (c[zp * plane + y * nx + x] + c[zn * plane + y * nx + x] - 2.0 * cv) * invDz2;
    mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
          + 3.0 * cv + cv * cv * cv - gamma * laplacian;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t nx, size_t ny, size_t nz,
                             double invDx2, double invDy2, double invDz2, double Ddt) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);
    const double center = mu[i];
    const double laplacian = (mu[z * plane + y * nx + xp] + mu[z * plane + y * nx + xn] - 2.0 * center) * invDx2
                           + (mu[z * plane + yp * nx + x] + mu[z * plane + yn * nx + x] - 2.0 * center) * invDy2
                           + (mu[zp * plane + y * nx + x] + mu[zn * plane + y * nx + x] - 2.0 * center) * invDz2;
    cnew[i] = cold[i] + Ddt * laplacian;
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, z, nx, ny);
                const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[i] = -1.0 + 2.0 * pseudo;
            }
}

bool validateResult(const std::vector<double>& c, size_t, size_t, size_t) {
    for (const double val : c)
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    double minVal = c[0], maxVal = c[0];
    for (const double val : c) { minVal = std::min(minVal, val); maxVal = std::max(maxVal, val); }
    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n"); return 1;
    }
    const size_t gridSize = nx * ny * nz;
    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");

    constexpr double dt = 0.01, eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5, D = 1.0;
    std::vector<double> cold(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    DeviceBuffer dCold(gridSize), dNew(gridSize), dMu(gridSize);
    checkCuda(cudaMemcpy(dCold.get(), cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice), "initial data transfer");

    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz + block.z - 1) / block.z);
    double* current = dCold.get();
    double* next = dNew.get();
    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, block>>>(current, dMu.get(), nx, ny, nz, 1.0, 1.0, 1.0, gamma, eAA, eBB, eAB);
        updateKernel<<<grid, block>>>(next, current, dMu.get(), nx, ny, nz, 1.0, 1.0, 1.0, D * dt);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "simulation synchronization");
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::printf("Computation time: %ld ms\n", duration.count());
    const double elapsedSeconds = duration.count() / 1000.0;
    const double mcups = elapsedSeconds > 0.0 ? static_cast<double>(gridSize) * iterations / elapsedSeconds / 1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(cold.data(), current, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "result transfer");
    }
    if (printResults) print_results(cold, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(cold, nx, ny, nz)) { std::printf("Validation: PASSED\n"); return 0; }
        std::printf("Validation: FAILED\n"); return 1;
    }
    return 0;
}
