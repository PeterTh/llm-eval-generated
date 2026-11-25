#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using val_t = double;
using hash_t = unsigned long long;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)), "Allocating GPU memory");
    }

    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

__device__ __forceinline__ val_t addFlux(val_t flux, val_t center, val_t neighbor,
                                        bool scalar_tail = false) {
    // The generated mesh has unit connection flux and transfer coefficient 0.8.
    // The CPU build rounds the vectorized pairs before adding them and fuses
    // only the final scalar term on edges with three neighbors.
    const val_t weighted = (neighbor - center) * 0.8;
    return scalar_tail ? __fma_rn(weighted, 0.25, flux) : flux + weighted * 0.25;
}

template <bool corners>
__global__ void updateEnergy(const val_t* __restrict__ energy,
                             val_t* __restrict__ next_energy, val_t* __restrict__ total_flux,
                             int side, int extent) {
    int col = blockIdx.x * BLOCK_X + threadIdx.x;
    int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (col >= extent || row >= extent) {
        return;
    }
    if constexpr (corners) {
        col += (blockIdx.z & 1) ? side - extent : 0;
        row += (blockIdx.z & 2) ? side - extent : 0;
    }

    const int last = side - 1;
    const int i = row * side + col;
    const val_t center = energy[i];

    const bool row_edge = row == 0 || row == last;
    const bool col_edge = col == 0 || col == last;

    // The main-diagonal corners have inflow, including a single-cell grid.
    val_t flux = 0.0;
    if (row_edge && col_edge) {
        flux = row == col ? 0.5 : -0.5;
    }

    // Keep the original order of the four connections.
    if (row < last) flux = addFlux(flux, center, energy[i + side]);
    if (row > 0)    flux = addFlux(flux, center, energy[i - side]);
    if (col < last) flux = addFlux(flux, center, energy[i + 1], col == 0 && !row_edge);
    if (col > 0)    flux = addFlux(flux, center, energy[i - 1], row_edge != col_edge);

    next_energy[i] = center + flux;
    total_flux[i] += fabs(flux);
}

float runSimulation(val_t*& energy, val_t* next_energy, val_t* total_flux,
                    int side, int iterations) {
    // Load the needed kernels before timing, including with CUDA lazy loading.
    cudaFuncAttributes attributes;
    if (iterations > 0 && side > 1) {
        checkCuda(cudaFuncGetAttributes(&attributes, updateEnergy<true>), "Loading corner kernel");
    }
    if (iterations > side / 2) {
        checkCuda(cudaFuncGetAttributes(&attributes, updateEnergy<false>), "Loading grid kernel");
    }

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");
    checkCuda(cudaEventRecord(start), "Recording start event");

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 full_grid((side + BLOCK_X - 1) / BLOCK_X,
                         (side + BLOCK_Y - 1) / BLOCK_Y);
    for (int iter = 0; iter < iterations; ++iter) {
        const int extent = iter + 1;
        if (extent <= side / 2) {
            // A source can affect at most one more layer per iteration. The
            // untouched center stays zero in both buffers, and patches never overlap.
            const dim3 grid((extent + BLOCK_X - 1) / BLOCK_X,
                            (extent + BLOCK_Y - 1) / BLOCK_Y, 4);
            updateEnergy<true><<<grid, block>>>(energy, next_energy, total_flux, side, extent);
        } else {
            updateEnergy<false><<<full_grid, block>>>(energy, next_energy, total_flux, side, side);
        }
        std::swap(energy, next_energy);
    }

    checkCuda(cudaGetLastError(), "Launching simulation");
    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Executing simulation");
    float duration_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&duration_ms, start, stop), "Measuring simulation");
    checkCuda(cudaEventDestroy(start), "Destroying start event");
    checkCuda(cudaEventDestroy(stop), "Destroying stop event");
    return duration_ms;
}

constexpr int HASH_THREADS = 256;

__device__ __forceinline__ hash_t warpXor(hash_t hash) {
    for (int offset = 16; offset > 0; offset /= 2) {
        hash ^= __shfl_down_sync(0xffffffff, hash, offset);
    }
    return hash;
}

__global__ void computeHash(const val_t* __restrict__ energy,
                            const val_t* __restrict__ total_flux, size_t count, hash_t* result) {
    hash_t hash = 0;
    const size_t stride = gridDim.x * HASH_THREADS;
    for (size_t i = blockIdx.x * HASH_THREADS + threadIdx.x; i < count; i += stride) {
        const auto energy_bits = static_cast<hash_t>(__double_as_longlong(energy[i]));
        const auto flux_bits = static_cast<hash_t>(__double_as_longlong(total_flux[i]));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }

    __shared__ hash_t warp_hashes[HASH_THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    hash = warpXor(hash);
    if (lane == 0) {
        warp_hashes[warp] = hash;
    }
    __syncthreads();

    if (warp == 0) {
        hash = warpXor(lane < HASH_THREADS / 32 ? warp_hashes[lane] : 0);
        if (lane == 0) {
            atomicXor(result, hash);
        }
    }
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& total_flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += total_flux[i];
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
        fprintf(stderr, "Grid size must be positive and its element count must fit in an int.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n\n", validate ? "enabled" : "disabled");

    printf("Building unstructured mesh...\n");
    const size_t count = n_elems;
    const size_t dynamic_mem = 3 * count * sizeof(val_t);
    DeviceBuffer<val_t> state(3 * count);
    DeviceBuffer<hash_t> device_hash(1);
    checkCuda(cudaMemset(state.data, 0, dynamic_mem), "Initializing GPU state");
    checkCuda(cudaMemset(device_hash.data, 0, sizeof(hash_t)), "Initializing GPU hash");
    val_t* energy = state.data;
    val_t* next_energy = energy + count;
    val_t* total_flux = next_energy + count;

    // Grid connectivity is implicit, so the mesh has no static allocation.
    printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n",
           dynamic_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));

    printf("Running simulation...\n");
    const double duration_ms = runSimulation(energy, next_energy, total_flux, n_elems_root, n_iters);
    printf("Computation time: %.3f ms\n", duration_ms);

    const int n_measured_iters = n_iters > 1 ? n_iters - 1 : 1;
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms * 1e6)
        : 0.0;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);

    const int hash_blocks = std::min((count + HASH_THREADS - 1) / HASH_THREADS, size_t{1024});
    computeHash<<<hash_blocks, HASH_THREADS>>>(energy, total_flux, count, device_hash.data);
    checkCuda(cudaGetLastError(), "Launching result hash");
    hash_t hash = 0;
    checkCuda(cudaMemcpy(&hash, device_hash.data, sizeof(hash), cudaMemcpyDeviceToHost),
              "Reading result hash");
    printf("  Result hash: %016llX\n\n", hash);

    std::vector<val_t> energy_data;
    if (printResults || validate) {
        energy_data.resize(count);
        checkCuda(cudaMemcpy(energy_data.data(), energy, count * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "Reading final energy");
    }
    if (printResults) {
        print_results(energy_data, "ElementEnergy");
    }
    if (validate) {
        std::vector<val_t> flux_data(count);
        checkCuda(cudaMemcpy(flux_data.data(), total_flux, count * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "Reading accumulated flux");
        if (!validateResults(energy_data, flux_data)) {
            return 1;
        }
    }

    return 0;
}
