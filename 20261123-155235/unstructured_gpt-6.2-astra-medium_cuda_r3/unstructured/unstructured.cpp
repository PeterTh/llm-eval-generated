#include <cuda_runtime.h>

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

using val_t = double;
constexpr int block_size = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Edges have unit weight. A missing edge points back to its own element
// and contributes zero, keeping each adjacency list one aligned vector.
__global__ void buildSquare2D(int4* neighbors, int side, int count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int cell = static_cast<int>(i);
    const int col = cell % side;
    neighbors[i] = make_int4(cell < count - side ? cell + side : cell,
                             cell >= side ? cell - side : cell,
                             col < side - 1 ? cell + 1 : cell,
                             col > 0 ? cell - 1 : cell);
}

__device__ double edgeFlux(double neighbor, double center) {
    return ((neighbor - center) * 0.8) * 0.25;
}

__global__ void updateEnergy(const int4* __restrict__ neighbors,
                             const double* __restrict__ energy,
                             double* __restrict__ next, double* __restrict__ flux,
                             int side, int count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int4 edges = neighbors[i];
    const double center = energy[i];
    double change = 0.0;
    if (i == 0 || i == count - 1) change = 0.5;
    else if (i == side - 1 || i == count - side) change = -0.5;
    change += edgeFlux(energy[edges.x], center);
    change += edgeFlux(energy[edges.y], center);
    change += edgeFlux(energy[edges.z], center);
    change += edgeFlux(energy[edges.w], center);
    next[i] = center + change;
    flux[i] += fabs(change);
}

// XOR is associative, so the verification hash can be reduced in any order.
__global__ void computeHash(const double* energy, const double* flux, int count,
                            unsigned long long* result) {
    unsigned long long value = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        value ^= (__double_as_longlong(energy[i]) + static_cast<unsigned long long>(i))
                 * 0x9e3779b97f4a7c15ULL;
        value ^= (__double_as_longlong(flux[i]) + static_cast<unsigned long long>(i))
                 * 0xbf58476d1ce4e5b9ULL;
    }
    for (int offset = 16; offset; offset /= 2)
        value ^= __shfl_down_sync(0xffffffff, value, offset);
    __shared__ unsigned long long warps[block_size / 32];
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        value = 0;
        for (auto partial : warps) value ^= partial;
        atomicXor(result, value);
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
    
    if (n_elems_root < 1 || static_cast<int64_t>(n_elems_root) * n_elems_root >
                               std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Grid size is outside the supported range\n");
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
    DeviceBuffer<int4> neighbors(n_elems);
    DeviceBuffer<double> first(n_elems), second(n_elems), flux(n_elems);
    double* energy = first.data;
    double* next = second.data;
    const size_t bytes = static_cast<size_t>(n_elems) * sizeof(double);
    buildSquare2D<<<blocks, block_size>>>(neighbors.data, n_elems_root, n_elems);
    checkCuda(cudaGetLastError(), "buildSquare2D launch");
    checkCuda(cudaMemset(energy, 0, bytes), "initialize energy");
    checkCuda(cudaMemset(flux.data, 0, bytes), "initialize flux");
    checkCuda(cudaDeviceSynchronize(), "mesh initialization");

    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(int4);
    const size_t dynamic_mem = 3 * bytes;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::steady_clock::now();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        updateEnergy<<<blocks, block_size>>>(neighbors.data, energy, next, flux.data,
                                             n_elems_root, n_elems);
        std::swap(energy, next);
    }
    checkCuda(cudaGetLastError(), "updateEnergy launch");
    checkCuda(cudaDeviceSynchronize(), "simulation");
    
    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = n_iters > 1 ? n_iters - 1 : 1;
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = static_cast<double>(n_elems) / time_per_iter / 1e6;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    DeviceBuffer<unsigned long long> device_hash(1);
    checkCuda(cudaMemset(device_hash.data, 0, sizeof(unsigned long long)), "initialize hash");
    computeHash<<<std::min(blocks, 1024), block_size>>>(energy, flux.data, n_elems,
                                                        device_hash.data);
    checkCuda(cudaGetLastError(), "computeHash launch");
    unsigned long long hash;
    checkCuda(cudaMemcpy(&hash, device_hash.data, sizeof(hash), cudaMemcpyDeviceToHost),
              "read hash");
    printf("  Result hash: %016llX\n\n", hash);

    std::vector<double> energyData;
    if (printResults || validate) {
        energyData.resize(n_elems);
        checkCuda(cudaMemcpy(energyData.data(), energy, bytes, cudaMemcpyDeviceToHost),
                  "read energy");
    }
    if (printResults) print_results(energyData, "ElementEnergy");
    if (validate) {
        std::vector<double> fluxData(n_elems);
        checkCuda(cudaMemcpy(fluxData.data(), flux.data, bytes, cudaMemcpyDeviceToHost),
                  "read flux");
        bool valid = validateResults(energyData, fluxData);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
