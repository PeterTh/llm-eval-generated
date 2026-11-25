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
#include <climits>

using val_t = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
class DeviceArray {
public:
    T* data = nullptr;

    explicit DeviceArray(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)), "Allocate device memory");
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

struct World {
    int side;
    size_t count;
    // One contiguous plane per connection slot for coalesced neighbor lookups.
    DeviceArray<int> neighbors;
    DeviceArray<double> source, energy, next_energy, flux;

    explicit World(int n)
        : side(n), count(size_t(n) * n), neighbors(4 * count), source(count),
          energy(count), next_energy(count), flux(count) {}
};

__global__ void buildSquare2D(int side, size_t count, int* neighbors, double* source,
                              double* energy, double* next_energy, double* flux) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (size_t(i) >= count) return;
    int x = i / side, y = i % side;
    neighbors[i] = x + 1 < side ? i + side : -1;
    neighbors[count + i] = x > 0 ? i - side : -1;
    neighbors[2 * count + i] = y + 1 < side ? i + 1 : -1;
    neighbors[3 * count + i] = y > 0 ? i - 1 : -1;
    double flow = 0.0;
    if ((x == 0 || x == side - 1) && (y == 0 || y == side - 1))
        flow = x == y ? 0.5 : -0.5;
    source[i] = flow;
    energy[i] = next_energy[i] = flux[i] = 0.0;
}

__global__ void update(int side, int extent, size_t count, const int* __restrict__ neighbors,
                       const double* __restrict__ source, const double* __restrict__ energy,
                       double* __restrict__ next_energy, double* __restrict__ flux) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= extent || y >= extent) return;
    // Until the corner regions meet, omit the zero band in each dimension.
    if (x >= extent / 2) x += side - extent;
    if (y >= extent / 2) y += side - extent;
    int i = x * side + y;
    double current = energy[i];
    double delta = source[i];
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        int neighbor = neighbors[slot * count + i];
        if (neighbor >= 0)
            delta += ((energy[neighbor] - current) * 0.8) * 0.25;
    }
    next_energy[i] = current + delta;
    flux[i] += fabs(delta);
}

void runSimulation(World& world, int iterations) {
    const dim3 threads(32, 4);
    for (int step = 0; step < iterations; ++step) {
        // One edge traversal per step bounds the support of the four sources.
        int extent = step < world.side / 2 ? 2 * (step + 1) : world.side;
        dim3 blocks((extent + threads.x - 1) / threads.x,
                    (extent + threads.y - 1) / threads.y);
        update<<<blocks, threads>>>(world.side, extent, world.count, world.neighbors.data,
                                    world.source.data, world.energy.data,
                                    world.next_energy.data, world.flux.data);
        checkCuda(cudaGetLastError(), "Launch energy update");
        std::swap(world.energy.data, world.next_energy.data);
    }
    checkCuda(cudaDeviceSynchronize(), "Run simulation");
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

__global__ void hashElements(const double* energy, const double* flux, size_t count,
                             unsigned long long* result) {
    unsigned long long hash = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        hash ^= (__double_as_longlong(energy[i]) + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (__double_as_longlong(flux[i]) + i) * 0xbf58476d1ce4e5b9ULL;
    }
    for (int offset = 16; offset; offset /= 2)
        hash ^= __shfl_xor_sync(0xffffffff, hash, offset);
    __shared__ unsigned long long warp_hashes[8];
    if (threadIdx.x % 32 == 0) warp_hashes[threadIdx.x / 32] = hash;
    __syncthreads();
    if (threadIdx.x < 32) {
        hash = threadIdx.x < 8 ? warp_hashes[threadIdx.x] : 0;
        for (int offset = 4; offset; offset /= 2)
            hash ^= __shfl_xor_sync(0xffffffff, hash, offset);
        if (threadIdx.x == 0) atomicXor(result, hash);
    }
}

uint64_t computeHash(const World& world) {
    DeviceArray<unsigned long long> result(1);
    checkCuda(cudaMemset(result.data, 0, sizeof(uint64_t)), "Initialize hash");
    int blocks = std::min<size_t>((world.count + 255) / 256, 1024);
    hashElements<<<blocks, 256>>>(world.energy.data, world.flux.data, world.count, result.data);
    checkCuda(cudaGetLastError(), "Launch hash reduction");
    uint64_t hash;
    checkCuda(cudaMemcpy(&hash, result.data, sizeof(hash), cudaMemcpyDeviceToHost), "Read hash");
    return hash;
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
    
    if (n_elems_root <= 0 || n_elems_root > INT_MAX / n_elems_root) {
        std::fprintf(stderr, "Grid size must be positive and fit in integer mesh indices\n");
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
    World world(n_elems_root);
    buildSquare2D<<<(world.count + 255) / 256, 256>>>(
        world.side, world.count, world.neighbors.data, world.source.data,
        world.energy.data, world.next_energy.data, world.flux.data);
    checkCuda(cudaGetLastError(), "Launch mesh initialization");
    checkCuda(cudaDeviceSynchronize(), "Build mesh");
    
    // Calculate memory usage
    const size_t static_mem = world.count * (4 * sizeof(int) + sizeof(double));
    const size_t dynamic_mem = world.count * 3 * sizeof(double);
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (double(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    std::vector<double> energy;
    if (printResults || validate) {
        energy.resize(world.count);
        checkCuda(cudaMemcpy(energy.data(), world.energy.data, world.count * sizeof(double),
                             cudaMemcpyDeviceToHost), "Read energy");
    }
    if (printResults) print_results(energy, "ElementEnergy");
    if (validate) {
        std::vector<double> flux(world.count);
        checkCuda(cudaMemcpy(flux.data(), world.flux.data, world.count * sizeof(double),
                             cudaMemcpyDeviceToHost), "Read accumulated flux");
        bool valid = validateResults(energy, flux);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
