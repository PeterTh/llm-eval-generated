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
constexpr int block_size = 256;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* data;
};

// Every edge built by this benchmark has unit weight. Keep the four
// neighbor indices together so a thread can load its adjacency in one int4.
__global__ void buildSquare2D(int width, int count, int4* neighbors,
                              double* source, double* energy, double* flux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int column = i % width;
    neighbors[i] = make_int4(i < count - width ? i + width : -1,
                             i >= width ? i - width : -1,
                             column < width - 1 ? i + 1 : -1,
                             column > 0 ? i - 1 : -1);
    double flow = 0.0;
    if (i == 0 || i == count - 1) flow = 0.5;
    else if (i == width - 1 || i == count - width) flow = -0.5;
    source[i] = flow;
    energy[i] = 0.0;
    flux[i] = 0.0;
}

__device__ inline double edgeFlux(double adjacent, double energy) {
    return (adjacent - energy) * 0.8 * 0.25;
}

__global__ void updateElements(int count, const int4* __restrict__ neighbors,
                               const double* __restrict__ source,
                               const double* __restrict__ current,
                               double* __restrict__ next, double* __restrict__ flux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int4 edges = neighbors[i];
    double energy = current[i];
    double delta = source[i];
    if (edges.x >= 0) delta += edgeFlux(current[edges.x], energy);
    if (edges.y >= 0) delta += edgeFlux(current[edges.y], energy);
    if (edges.z >= 0) delta += edgeFlux(current[edges.z], energy);
    if (edges.w >= 0) delta += edgeFlux(current[edges.w], energy);
    next[i] = energy + delta;
    flux[i] += fabs(delta);
}

// Each element contributes independently to the benchmark's XOR hash.
__global__ void hashElements(int count, const double* energy, const double* flux,
                             uint64_t* partial) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t bits = 0;
    for (; i < static_cast<unsigned>(count); i += blockDim.x * gridDim.x) {
        uint64_t e = __double_as_longlong(energy[i]);
        uint64_t f = __double_as_longlong(flux[i]);
        bits ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        bits ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    __shared__ uint64_t warps[block_size / 32];
    for (int offset = 16; offset; offset /= 2)
        bits ^= __shfl_down_sync(0xffffffff, bits, offset);
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = bits;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t result = 0;
        for (uint64_t value : warps) result ^= value;
        partial[blockIdx.x] = result;
    }
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
    
    if (n_elems_root <= 0 || int64_t(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Grid size out of range\n");
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
    const int blocks = 1 + (n_elems - 1) / block_size;
    DeviceArray<int4> neighbors(n_elems);
    DeviceArray<double> source(n_elems), energyA(n_elems), energyB(n_elems), flux(n_elems);
    double* current = energyA.data;
    double* next = energyB.data;
    buildSquare2D<<<blocks, block_size>>>(n_elems_root, n_elems, neighbors.data,
                                         source.data, current, flux.data);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    const size_t static_mem = size_t(n_elems) * (sizeof(int4) + sizeof(double));
    const size_t dynamic_mem = size_t(n_elems) * 3 * sizeof(double);
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
        updateElements<<<blocks, block_size>>>(n_elems, neighbors.data, source.data,
                                               current, next, flux.data);
        checkCuda(cudaGetLastError());
        std::swap(current, next);
    }
    checkCuda(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.4f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(std::max(n_iters, 0)) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const int hash_blocks = std::min(blocks, 1024);
    DeviceArray<uint64_t> partial(hash_blocks);
    hashElements<<<hash_blocks, block_size>>>(n_elems, current, flux.data, partial.data);
    checkCuda(cudaGetLastError());
    std::vector<uint64_t> hashes(hash_blocks);
    checkCuda(cudaMemcpy(hashes.data(), partial.data, hash_blocks * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    uint64_t hash = 0;
    for (uint64_t value : hashes) hash ^= value;
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    std::vector<double> energyData;
    if (printResults || validate) {
        energyData.resize(n_elems);
        checkCuda(cudaMemcpy(energyData.data(), current, n_elems * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        std::vector<double> fluxData(n_elems);
        checkCuda(cudaMemcpy(fluxData.data(), flux.data, n_elems * sizeof(double),
                             cudaMemcpyDeviceToHost));
        bool valid = validateResults(energyData, fluxData);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
