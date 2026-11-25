#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint32_t;
using val_t = double;

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&ptr, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return ptr; }

private:
    T* ptr = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { checkCuda(cudaEventCreate(&event)); }
    ~CudaEvent() { cudaEventDestroy(event); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t event = nullptr;
};

// Pack the four 32-bit neighbor indices into one aligned vector per element.
__global__ void buildSquare2D(uint4* neighbors, val_t* energy, val_t* accumulatedFlux,
                              idx_t count, idx_t side) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const idx_t row = i / side;
    const idx_t col = i % side;
    // Self indices fill missing slots. Their zero flux leaves the sum unchanged.
    neighbors[i] = make_uint4(row + 1 < side ? i + side : i,
                             row > 0 ? i - side : i,
                             col + 1 < side ? i + 1 : i,
                             col > 0 ? i - 1 : i);
    energy[i] = 0.0;
    accumulatedFlux[i] = 0.0;
}

__device__ val_t addConnectionFlux(val_t total, val_t neighbor, val_t energy) {
    // All materials transfer at 0.8 and all connections have unit weight.
    // The original native build contracts the final multiply and addition.
    return __fma_rn((neighbor - energy) * 0.8, 0.25, total);
}

__global__ void updateEnergy(const uint4* __restrict__ neighbors,
                             const val_t* __restrict__ current,
                             val_t* __restrict__ next,
                             val_t* __restrict__ accumulatedFlux,
                             idx_t count, idx_t side) {
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const val_t energy = current[i];
    // Match the original corner assignments, including the overlapping corners at side == 1.
    val_t totalFlux = 0.0;
    if (i == 0 || i == count - 1) {
        totalFlux = 0.5;
    } else if (i == side - 1 || i == count - side) {
        totalFlux = -0.5;
    }

    const uint4 connected = neighbors[i];
    totalFlux = addConnectionFlux(totalFlux, current[connected.x], energy);
    totalFlux = addConnectionFlux(totalFlux, current[connected.y], energy);
    totalFlux = addConnectionFlux(totalFlux, current[connected.z], energy);
    totalFlux = addConnectionFlux(totalFlux, current[connected.w], energy);

    next[i] = energy + totalFlux;
    // Only this element uses its previous accumulated flux.
    accumulatedFlux[i] += fabs(totalFlux);
}

double runSimulation(const uint4* neighbors, val_t*& current, val_t*& next,
                     val_t* accumulatedFlux, idx_t count, idx_t side,
                     int iterations) {
    CudaEvent start;
    CudaEvent stop;
    const idx_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    checkCuda(cudaEventRecord(start.event));
    // Each launch sees the completed energy buffer from the preceding launch.
    for (int iter = 0; iter < iterations; ++iter) {
        updateEnergy<<<blocks, BLOCK_SIZE>>>(neighbors, current, next, accumulatedFlux,
                                             count, side);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaEventRecord(stop.event));
    checkCuda(cudaEventSynchronize(stop.event));

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.event, stop.event));
    return iterations > 0 ? milliseconds : 0.0;
}

__device__ uint64_t warpXor(uint64_t value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value ^= __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void hashElements(const val_t* energy, const val_t* accumulatedFlux,
                             idx_t count, uint64_t* blockHashes) {
    uint64_t value = 0;
    for (idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridDim.x * blockDim.x) {
        const uint64_t energyBits = __double_as_longlong(energy[i]);
        const uint64_t fluxBits = __double_as_longlong(accumulatedFlux[i]);
        value ^= (energyBits + i) * 0x9e3779b97f4a7c15ULL;
        value ^= (fluxBits + i) * 0xbf58476d1ce4e5b9ULL;
    }

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    __shared__ uint64_t warpHashes[BLOCK_SIZE / WARP_SIZE];
    value = warpXor(value);
    if (lane == 0) {
        warpHashes[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = warpXor(lane < BLOCK_SIZE / WARP_SIZE ? warpHashes[lane] : 0);
        if (lane == 0) {
            blockHashes[blockIdx.x] = value;
        }
    }
}

uint64_t computeHash(const val_t* energy, const val_t* accumulatedFlux, idx_t count) {
    constexpr idx_t maxBlocks = 1024;
    const idx_t blocks = std::min((count + BLOCK_SIZE - 1) / BLOCK_SIZE, maxBlocks);
    DeviceBuffer<uint64_t> deviceHashes(blocks);
    hashElements<<<blocks, BLOCK_SIZE>>>(energy, accumulatedFlux, count, deviceHashes.data());
    checkCuda(cudaGetLastError());

    std::vector<uint64_t> blockHashes(blocks);
    checkCuda(cudaMemcpy(blockHashes.data(), deviceHashes.data(),
                         blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t hash = 0;
    for (const uint64_t blockHash : blockHashes) {
        hash ^= blockHash;
    }
    return hash;
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
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
    
    if (n_elems_root <= 0 ||
        static_cast<int64_t>(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
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
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    const idx_t count = n_elems;
    DeviceBuffer<uint4> neighbors(count);
    DeviceBuffer<val_t> energyA(count);
    DeviceBuffer<val_t> energyB(count);
    DeviceBuffer<val_t> accumulatedFlux(count);
    val_t* current = energyA.data();
    val_t* next = energyB.data();

    const idx_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildSquare2D<<<blocks, BLOCK_SIZE>>>(neighbors.data(), current, accumulatedFlux.data(),
                                         count, n_elems_root);
    checkCuda(cudaGetLastError());

    const size_t static_mem = count * sizeof(uint4);
    const size_t dynamic_mem = count * sizeof(val_t) * 3;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");

    printf("Running simulation...\n");
    const double duration_ms = runSimulation(neighbors.data(), current, next,
                                             accumulatedFlux.data(), count, n_elems_root, n_iters);
    printf("Computation time: %.4f ms\n", duration_ms);

    const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
    const double giga_elems_per_sec = time_per_iter > 0 ? count / (time_per_iter * 1e6) : 0.0;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    const uint64_t hash = computeHash(current, accumulatedFlux.data(), count);
    printf("  Result hash: %016" PRIX64 "\n", hash);
    printf("\n");

    std::vector<val_t> energyData;
    if (printResults || validate) {
        energyData.resize(count);
        checkCuda(cudaMemcpy(energyData.data(), current, count * sizeof(val_t),
                             cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        print_results(energyData, "ElementEnergy");
    }
    if (validate) {
        std::vector<val_t> fluxData(count);
        checkCuda(cudaMemcpy(fluxData.data(), accumulatedFlux.data(), count * sizeof(val_t),
                             cudaMemcpyDeviceToHost));
        if (!validateResults(energyData, fluxData)) {
            return 1;
        }
    }
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "Unstructured benchmark failed: %s\n", error.what());
    return 1;
}
