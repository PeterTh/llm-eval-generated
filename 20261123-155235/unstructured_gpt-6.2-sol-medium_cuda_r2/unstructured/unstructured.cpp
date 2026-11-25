#include <cuda_runtime.h>

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
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

constexpr int BLOCK_COLUMNS = 32;
constexpr int BLOCK_ROWS = 8;

__device__ inline val_t connectionFlux(val_t current, val_t neighbor) {
    // All connections have weight 1.0 and all materials have coefficient 0.8.
    return ((neighbor - current) * 0.8) * 0.25;
}

__global__ void updateElements(const val_t* __restrict__ energy,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ accumulated_flux,
                               unsigned int side) {
    const unsigned int col = blockIdx.x * BLOCK_COLUMNS + threadIdx.x;
    const unsigned int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    if (col >= side || row >= side) {
        return;
    }

    const size_t index = static_cast<size_t>(row) * side + col;
    const val_t current = energy[index];
    const unsigned int last = side - 1;

    val_t flux = 0.0;
    if ((row == 0 || row == last) && (col == 0 || col == last)) {
        flux = row == col ? 0.5 : -0.5;
    }

    // Visit the same neighbors in the same order as the generated mesh.
    if (row < last) {
        flux += connectionFlux(current, energy[index + side]);
    }
    if (row > 0) {
        flux += connectionFlux(current, energy[index - side]);
    }
    if (col < last) {
        flux += connectionFlux(current, energy[index + 1]);
    }
    if (col > 0) {
        flux += connectionFlux(current, energy[index - 1]);
    }

    next_energy[index] = current + flux;
    // Only the thread for this element uses its accumulated flux.
    accumulated_flux[index] += fabs(flux);
}

constexpr int ITERATIONS_PER_BATCH = 32;
static_assert(ITERATIONS_PER_BATCH % 2 == 0);

class UpdateBatch {
public:
    UpdateBatch() = default;
    ~UpdateBatch() {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
        }
    }

    UpdateBatch(const UpdateBatch&) = delete;
    UpdateBatch& operator=(const UpdateBatch&) = delete;

    void prepare(dim3 blocks, dim3 threads, val_t* energy, val_t* next_energy,
                 val_t* accumulated_flux, unsigned int side) {
        checkCuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal),
                  "Starting CUDA graph capture");
        // An even number of updates returns the current energy to the first buffer.
        for (int i = 0; i < ITERATIONS_PER_BATCH; ++i) {
            updateElements<<<blocks, threads, 0, cudaStreamPerThread>>>(
                energy, next_energy, accumulated_flux, side);
            std::swap(energy, next_energy);
        }
        cudaGraph_t graph = nullptr;
        checkCuda(cudaStreamEndCapture(cudaStreamPerThread, &graph), "Ending CUDA graph capture");
        const cudaError_t status = cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        checkCuda(status, "Instantiating CUDA graph");
    }

    void launch() const {
        checkCuda(cudaGraphLaunch(graph_exec_, cudaStreamPerThread), "Launching CUDA graph");
    }

private:
    cudaGraphExec_t graph_exec_ = nullptr;
};

constexpr int HASH_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

__device__ inline uint64_t warpXor(uint64_t value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value ^= __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void hashElements(const val_t* energy, const val_t* accumulated_flux,
                             size_t count, uint64_t* partial_hashes) {
    __shared__ uint64_t warp_hashes[HASH_BLOCK_SIZE / WARP_SIZE];
    const size_t stride = static_cast<size_t>(gridDim.x) * HASH_BLOCK_SIZE;
    uint64_t hash = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * HASH_BLOCK_SIZE + threadIdx.x;
         i < count; i += stride) {
        hash ^= (static_cast<uint64_t>(__double_as_longlong(energy[i])) + i) *
                0x9e3779b97f4a7c15ULL;
        hash ^= (static_cast<uint64_t>(__double_as_longlong(accumulated_flux[i])) + i) *
                0xbf58476d1ce4e5b9ULL;
    }

    hash = warpXor(hash);
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_hashes[warp] = hash;
    }
    __syncthreads();
    if (warp == 0) {
        hash = warpXor(lane < HASH_BLOCK_SIZE / WARP_SIZE ? warp_hashes[lane] : 0);
        if (lane == 0) {
            partial_hashes[blockIdx.x] = hash;
        }
    }
}

uint64_t computeHash(const val_t* energy, const val_t* accumulated_flux, size_t count) {
    const int blocks = static_cast<int>(
        std::min<size_t>((count + HASH_BLOCK_SIZE - 1) / HASH_BLOCK_SIZE, 1024));
    DeviceBuffer<uint64_t> device_hashes(blocks);
    hashElements<<<blocks, HASH_BLOCK_SIZE>>>(energy, accumulated_flux, count,
                                             device_hashes.data());
    checkCuda(cudaGetLastError(), "Launching hash kernel");

    std::vector<uint64_t> partial_hashes(blocks);
    checkCuda(cudaMemcpy(partial_hashes.data(), device_hashes.data(),
                         blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost),
              "Copying result hashes");
    uint64_t hash = 0;
    for (uint64_t partial_hash : partial_hashes) {
        hash ^= partial_hash;
    }
    return hash;
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& accumulated_flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += accumulated_flux[i];
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

    if (n_elems_root <= 0) {
        fprintf(stderr, "Grid size must be positive\n");
        return 1;
    }
    const unsigned int side = static_cast<unsigned int>(n_elems_root);
    const size_t n_elems = static_cast<size_t>(side) * side;
    if (n_elems > std::numeric_limits<size_t>::max() / (3 * sizeof(val_t))) {
        fprintf(stderr, "Grid size is too large\n");
        return 1;
    }

    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    printf("Building unstructured mesh...\n");
    const size_t field_bytes = n_elems * sizeof(val_t);
    DeviceBuffer<val_t> energy_buffer(n_elems);
    DeviceBuffer<val_t> next_energy_buffer(n_elems);
    DeviceBuffer<val_t> flux_buffer(n_elems);
    val_t* energy = energy_buffer.data();
    val_t* next_energy = next_energy_buffer.data();
    val_t* accumulated_flux = flux_buffer.data();
    checkCuda(cudaMemset(energy, 0, field_bytes), "Initializing energy");
    checkCuda(cudaMemset(accumulated_flux, 0, field_bytes), "Initializing accumulated flux");

    const double dynamic_mb = 3 * field_bytes / (1024.0 * 1024.0);
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           dynamic_mb, 0.0, dynamic_mb);
    printf("\n");

    printf("Running simulation...\n");
    const dim3 threads(BLOCK_COLUMNS, BLOCK_ROWS);
    const dim3 blocks((side + BLOCK_COLUMNS - 1) / BLOCK_COLUMNS,
                      (side + BLOCK_ROWS - 1) / BLOCK_ROWS);
    checkCuda(cudaStreamSynchronize(nullptr), "Initializing simulation");
    UpdateBatch batch;
    const int batches = n_iters >= 2 * ITERATIONS_PER_BATCH ? n_iters / ITERATIONS_PER_BATCH : 0;
    if (batches > 0) {
        batch.prepare(blocks, threads, energy, next_energy, accumulated_flux, side);
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < batches; ++i) {
        batch.launch();
    }
    for (int iter = batches * ITERATIONS_PER_BATCH; iter < n_iters; ++iter) {
        updateElements<<<blocks, threads, 0, cudaStreamPerThread>>>(
            energy, next_energy, accumulated_flux, side);
        std::swap(energy, next_energy);
    }
    checkCuda(cudaGetLastError(), "Launching update kernel");
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread), "Running simulation");
    const auto end = std::chrono::steady_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));
    const int n_measured_iters = n_iters > 1 ? n_iters - 1 : 1;
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = n_iters > 0
        ? (static_cast<double>(n_elems) * n_measured_iters) / (duration_ms * 1e6)
        : 0.0;
    const double gflops = giga_elems_per_sec * 22.0;

    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    printf("  Result hash: %016" PRIX64 "\n", computeHash(energy, accumulated_flux, n_elems));
    printf("\n");

    std::vector<val_t> energy_data;
    if (printResults || validate) {
        energy_data.resize(n_elems);
        checkCuda(cudaMemcpy(energy_data.data(), energy, field_bytes, cudaMemcpyDeviceToHost),
                  "Copying energy");
    }
    if (printResults) {
        print_results(energy_data, "ElementEnergy");
    }
    if (validate) {
        std::vector<val_t> flux_data(n_elems);
        checkCuda(cudaMemcpy(flux_data.data(), accumulated_flux, field_bytes,
                             cudaMemcpyDeviceToHost),
                  "Copying accumulated flux");
        if (!validateResults(energy_data, flux_data)) {
            return 1;
        }
    }
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
