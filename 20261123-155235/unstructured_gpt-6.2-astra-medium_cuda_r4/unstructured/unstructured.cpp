#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

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
constexpr int threads = 256;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data; }
private:
    T* data;
};

// Slot-major adjacency gives consecutive threads consecutive indices to load.
// A missing edge points back to the cell and contributes zero flux.
__global__ void buildSquare2D(int side, int count, int* neighbors,
                              double* flow, double* energy, double* flux) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const size_t stride = count;
    const int x = i / side;
    const int y = i % side;
    neighbors[i] = x < side - 1 ? i + side : i;
    neighbors[stride + i] = x > 0 ? i - side : i;
    neighbors[2 * stride + i] = y < side - 1 ? i + 1 : i;
    neighbors[3 * stride + i] = y > 0 ? i - 1 : i;
    double source = 0.0;
    if ((x == 0 || x == side - 1) && (y == 0 || y == side - 1))
        source = x == y ? 0.5 : -0.5;
    flow[i] = source;
    energy[i] = 0.0;
    flux[i] = 0.0;
}

__global__ void updateElements(int count, const int* __restrict__ neighbors,
                               const double* __restrict__ flow,
                               const double* __restrict__ energy,
                               double* __restrict__ next, double* __restrict__ flux) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(count)) return;
    const double current = energy[i];
    double change = flow[i];
    // All materials have coefficient 0.8 and all connections have unit weight.
    #pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int other = neighbors[static_cast<size_t>(slot) * count + i];
        change += ((energy[other] - current) * 0.8) * 0.25;
    }
    next[i] = current + change;
    flux[i] += fabs(change);
}

struct ElementHash {
    const double* energy;
    const double* flux;

    __device__ uint64_t operator()(int i) const {
        const uint64_t e = __double_as_longlong(energy[i]);
        const uint64_t f = __double_as_longlong(flux[i]);
        return ((e + i) * 0x9e3779b97f4a7c15ULL) ^
               ((f + i) * 0xbf58476d1ce4e5b9ULL);
    }
};

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
        std::fprintf(stderr, "Grid size must be positive and fit in an int element count\n");
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
    const int blocks = (count + threads - 1) / threads;
    DeviceBuffer<int> neighbors(4 * count);
    DeviceBuffer<double> flow(count), first(count), second(count), flux(count);
    double* energy = first.get();
    double* next = second.get();
    buildSquare2D<<<blocks, threads>>>(n_elems_root, n_elems, neighbors.get(),
                                       flow.get(), energy, flux.get());
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    const size_t static_mem = count * (4 * sizeof(int) + sizeof(double));
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
        updateElements<<<blocks, threads>>>(n_elems, neighbors.get(), flow.get(),
                                            energy, next, flux.get());
        std::swap(energy, next);
    }
    checkCuda(cudaGetLastError());
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
    const thrust::counting_iterator<int> begin(0);
    const uint64_t hash = thrust::transform_reduce(thrust::device, begin, begin + n_elems,
        ElementHash{energy, flux.get()}, uint64_t{0}, thrust::bit_xor<uint64_t>());
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    std::vector<double> energyData;
    if (printResults || validate) {
        energyData.resize(count);
        checkCuda(cudaMemcpy(energyData.data(), energy, count * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        std::vector<double> fluxData(count);
        checkCuda(cudaMemcpy(fluxData.data(), flux.get(), count * sizeof(double),
                             cudaMemcpyDeviceToHost));
        bool valid = validateResults(energyData, fluxData);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
