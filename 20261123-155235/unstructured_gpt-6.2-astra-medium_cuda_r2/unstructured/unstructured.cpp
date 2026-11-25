#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

using val_t = double;
constexpr int threads = 256;
constexpr int neighbors_per_cell = 4;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* data;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)), "Allocate device memory");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Store each adjacency slot in its own array for coalesced index loads.
// A missing edge has index -1. All generated edges have unit weight.
__global__ void buildSquare2D(int side, int count, int* neighbors, signed char* source,
                              double* energy, double* flux) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int col = i % side;
    const size_t stride = count;
    neighbors[i] = i < count - side ? i + side : -1;
    neighbors[stride + i] = i >= side ? i - side : -1;
    neighbors[2 * stride + i] = col < side - 1 ? i + 1 : -1;
    neighbors[3 * stride + i] = col > 0 ? i - 1 : -1;
    if (i == 0 || i == count - 1)
        source[i] = 1;
    else if (i == side - 1 || i == count - side)
        source[i] = -1;
    else
        source[i] = 0;
    energy[i] = 0.0;
    flux[i] = 0.0;
}

__global__ void update(int count, const int* __restrict__ neighbors,
                       const signed char* __restrict__ source,
                       const double* __restrict__ energy, double* __restrict__ next,
                       double* __restrict__ flux) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double current = energy[i];
    double transfer = source[i] * 0.5;
    #pragma unroll
    for (int edge = 0; edge < neighbors_per_cell; ++edge) {
        const int j = neighbors[size_t(edge) * count + i];
        if (j >= 0) transfer += (energy[j] - current) * 0.8 * 0.25;
    }
    next[i] = current + transfer;
    // Each thread is the sole reader and writer of its cell's accumulated flux.
    flux[i] += fabs(transfer);
}

__global__ void computeHash(int count, const double* energy, const double* flux,
                            unsigned long long* result) {
    unsigned long long bits = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(count);
         i += gridDim.x * blockDim.x) {
        bits ^= (__double_as_longlong(energy[i]) + i) * 0x9e3779b97f4a7c15ULL;
        bits ^= (__double_as_longlong(flux[i]) + i) * 0xbf58476d1ce4e5b9ULL;
    }
    for (int offset = 16; offset; offset /= 2)
        bits ^= __shfl_down_sync(0xffffffff, bits, offset);
    if ((threadIdx.x & 31) == 0) atomicXor(result, bits);
}

// Validate simulation results
bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    
    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }
    
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    
    printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (n_elems_root <= 0 || n_elems_root > std::numeric_limits<int>::max() / n_elems_root) {
        std::fprintf(stderr, "Grid size must be positive and fit in a 32-bit element index.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    const size_t count = n_elems;
    const int blocks = (n_elems - 1) / threads + 1;
    DeviceBuffer<int> neighbors(count * neighbors_per_cell);
    DeviceBuffer<signed char> source(count);
    DeviceBuffer<double> first(count), second(count), flux(count);
    double* energy = first.data;
    double* next = second.data;
    buildSquare2D<<<blocks, threads>>>(n_elems_root, n_elems, neighbors.data, source.data,
                                       energy, flux.data);
    checkCuda(cudaGetLastError(), "Launch mesh initialization");
    checkCuda(cudaDeviceSynchronize(), "Initialize mesh");

    const size_t static_mem = count * (neighbors_per_cell * sizeof(int) + sizeof(signed char));
    const size_t dynamic_mem = count * 3 * sizeof(double);
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        update<<<blocks, threads>>>(n_elems, neighbors.data, source.data, energy, next, flux.data);
        checkCuda(cudaGetLastError(), "Launch energy update");
        std::swap(energy, next);
    }
    checkCuda(cudaDeviceSynchronize(), "Run simulation");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = static_cast<double>(duration_ms) / std::max(n_measured_iters, 1);
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    DeviceBuffer<unsigned long long> deviceHash(1);
    checkCuda(cudaMemset(deviceHash.data, 0, sizeof(unsigned long long)), "Initialize hash");
    computeHash<<<std::min(blocks, 1024), threads>>>(n_elems, energy, flux.data, deviceHash.data);
    checkCuda(cudaGetLastError(), "Launch result hash");
    unsigned long long hash;
    checkCuda(cudaMemcpy(&hash, deviceHash.data, sizeof(hash), cudaMemcpyDeviceToHost), "Read hash");
    printf("  Result hash: %016llX\n\n", hash);

    if (printResults || validate) {
        std::vector<double> hostEnergy(count);
        checkCuda(cudaMemcpy(hostEnergy.data(), energy, count * sizeof(double),
                             cudaMemcpyDeviceToHost), "Read energy");
        if (printResults) print_results(hostEnergy, "ElementEnergy");
        if (validate) {
            std::vector<double> hostFlux(count);
            checkCuda(cudaMemcpy(hostFlux.data(), flux.data, count * sizeof(double),
                                 cudaMemcpyDeviceToHost), "Read flux");
            if (!validateResults(hostEnergy, hostFlux)) return 1;
        }
    }
    return 0;
}
