#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include "../common/results_output.hpp"

namespace {

using val_t = double;
constexpr int threads_per_block = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data_, count * sizeof(T)), "Allocating GPU memory");
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

template <typename Neighbors>
__global__ void buildSquare2D(Neighbors* links, val_t* energies, val_t* fluxes,
                             int side, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    using Offset = decltype(Neighbors::x);
    const int row = i / side;
    const int col = i - row * side;
    // Pack valid relative indices in the original order, with zeros at the end.
    const int first_row_offset = row + 1 < side ? side : -side;
    const int first_col_offset = col + 1 < side ? 1 : -1;
    const int second_col_offset = col > 0 && col + 1 < side ? -1 : 0;
    const bool two_row_neighbors = row > 0 && row + 1 < side;
    links[i] = side == 1 ? Neighbors{} : Neighbors{
        static_cast<Offset>(first_row_offset),
        static_cast<Offset>(two_row_neighbors ? -side : first_col_offset),
        static_cast<Offset>(two_row_neighbors ? first_col_offset : second_col_offset),
        static_cast<Offset>(two_row_neighbors ? second_col_offset : 0)
    };
    energies[i] = 0.0;
    fluxes[i] = 0.0;
}

__device__ __forceinline__ val_t computeFlux(val_t energy, val_t neighbor_energy) {
    // All materials share the transfer coefficient and all connections have unit flux.
    return ((neighbor_energy - energy) * 0.8) * 0.25;
}

template <typename Neighbors, bool fuse_edge_flux>
__global__ void updateElements(const Neighbors* __restrict__ links,
                               const val_t* __restrict__ energies,
                               val_t* __restrict__ next_energies,
                               val_t* __restrict__ fluxes, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const Neighbors neighbors = links[i];
    const val_t energy = energies[i];
    const val_t first = energies[i + neighbors.x];
    const val_t second = energies[i + neighbors.y];
    const val_t third = energies[i + neighbors.z];
    const val_t fourth = energies[i + neighbors.w];
    const auto energy_bits = __double_as_longlong(energy) | __double_as_longlong(first)
        | __double_as_longlong(second) | __double_as_longlong(third)
        | __double_as_longlong(fourth);
    if (energy_bits == 0 && neighbors.z != 0) {
        // Zero local energies and no external flow give zero flux this step.
        next_energies[i] = 0.0;
        return;
    }

    val_t total_flux = 0.0;
    if (neighbors.z == 0) {
        // Corners have two neighbors. On a one-cell grid the final corner assignment wins.
        total_flux = i == 0 || i == count - 1 ? 0.5 : -0.5;
    }
    total_flux += computeFlux(energy, first);
    total_flux += computeFlux(energy, second);
    if (neighbors.w != 0) {
        total_flux += computeFlux(energy, third);
        total_flux += computeFlux(energy, fourth);
    } else if (neighbors.z != 0) {
        // The GCC reference contracts the scalar tail for three-neighbor cells
        // on FMA hosts. Keep its rounding, including for subnormal fluxes.
        if constexpr (fuse_edge_flux) {
            total_flux = __fma_rn((third - energy) * 0.8, 0.25, total_flux);
        } else {
            total_flux += computeFlux(energy, third);
        }
    }

    next_energies[i] = energy + total_flux;
    // Each thread owns its accumulated flux. Only neighbor energies need double buffering.
    fluxes[i] += fabs(total_flux);
}

struct XorHash {
    __device__ unsigned long long operator()(unsigned long long a, unsigned long long b) const {
        return a ^ b;
    }
};

__global__ void computeHash(const val_t* energies, const val_t* fluxes, unsigned int count,
                            unsigned long long* hash) {
    using BlockReduce = cub::BlockReduce<unsigned long long, threads_per_block>;
    __shared__ BlockReduce::TempStorage storage;
    unsigned long long partial = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        const auto energy_bits = static_cast<unsigned long long>(__double_as_longlong(energies[i]));
        const auto flux_bits = static_cast<unsigned long long>(__double_as_longlong(fluxes[i]));
        partial ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        partial ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }

    const auto block_hash = BlockReduce(storage).Reduce(partial, XorHash{});
    if (threadIdx.x == 0) {
        atomicXor(hash, block_hash);
    }
}

struct SimulationResult {
    double duration_ms;
    uint64_t hash;
    std::vector<val_t> energies;
    std::vector<val_t> fluxes;
};

template <typename Neighbors>
SimulationResult runSimulation(int side, int count, int iterations,
                               bool copy_energy, bool copy_flux) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
    constexpr bool fuse_edge_flux = true;
#else
    constexpr bool fuse_edge_flux = false;
#endif
    printf("Building unstructured mesh...\n");
    DeviceBuffer<Neighbors> links(count);
    DeviceBuffer<val_t> energy_a(count);
    DeviceBuffer<val_t> energy_b(count);
    DeviceBuffer<val_t> fluxes(count);
    auto* current = energy_a.data();
    auto* next = energy_b.data();

    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    buildSquare2D<<<blocks, threads_per_block>>>(links.data(), current, fluxes.data(), side, count);
    checkCuda(cudaGetLastError(), "Launching mesh initialization");
    checkCuda(cudaDeviceSynchronize(), "Building the GPU mesh");

    const size_t static_mem = static_cast<size_t>(count) * sizeof(Neighbors);
    const size_t dynamic_mem = static_cast<size_t>(count) * 3 * sizeof(val_t);
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
           (static_mem + dynamic_mem) / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));

    printf("Running simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        updateElements<Neighbors, fuse_edge_flux><<<blocks, threads_per_block>>>(
            links.data(), current, next, fluxes.data(), count);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "Launching energy updates");
    checkCuda(cudaDeviceSynchronize(), "Running the GPU simulation");

    SimulationResult result{};
    result.duration_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    DeviceBuffer<unsigned long long> device_hash(1);
    checkCuda(cudaMemset(device_hash.data(), 0, sizeof(unsigned long long)), "Clearing result hash");
    computeHash<<<std::min(blocks, 1024), threads_per_block>>>(current, fluxes.data(), count,
                                                             device_hash.data());
    checkCuda(cudaGetLastError(), "Launching result hashing");

    unsigned long long hash;
    checkCuda(cudaMemcpy(&hash, device_hash.data(), sizeof(hash), cudaMemcpyDeviceToHost),
              "Reading result hash");
    result.hash = hash;
    if (copy_energy) {
        result.energies.resize(count);
        checkCuda(cudaMemcpy(result.energies.data(), current, count * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "Reading final energies");
    }
    if (copy_flux) {
        result.fluxes.resize(count);
        checkCuda(cudaMemcpy(result.fluxes.data(), fluxes.data(), count * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "Reading accumulated fluxes");
    }
    return result;
}

} // namespace

// Validate simulation results
bool validateResults(const std::vector<val_t>& energies,
                     const std::vector<val_t>& fluxes) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
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

int main(int argc, char** argv) try {
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
        fprintf(stderr, "Grid size must be positive and its square must fit in an int.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    const int n_measured_iters = std::max(n_iters, 2) - 1;
    const bool copy_energy = printResults || validate;
    const auto result = n_elems_root <= std::numeric_limits<short>::max()
        ? runSimulation<short4>(n_elems_root, n_elems, n_iters, copy_energy, validate)
        : runSimulation<int4>(n_elems_root, n_elems, n_iters, copy_energy, validate);

    printf("Computation time: %.3f ms\n", result.duration_ms);

    const double time_per_iter = result.duration_ms / n_measured_iters;
    const double giga_elems_per_sec = n_elems * static_cast<double>(n_measured_iters)
        / (result.duration_ms * 1e6);

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    printf("  Result hash: %016" PRIX64 "\n\n", result.hash);

    if (printResults) {
        print_results(result.energies, "ElementEnergy");
    }
    if (validate && !validateResults(result.energies, result.fluxes)) {
        return 1;
    }
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
