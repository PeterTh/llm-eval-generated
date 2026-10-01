#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Mesh topology is stored by connection slot (ELL format), so adjacent CUDA
// threads read adjacent indices and coefficients. The square builder uses four
// slots; the update kernel itself does not assume a structured topology.
using idx_t = uint32_t;
using val_t = double;
constexpr int SQUARE_CONNECTIONS = 4;
constexpr int BLOCK_SIZE = 256;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

struct World {
    size_t size = 0;
    size_t stride = 0;
    DeviceBuffer<idx_t> neighbors;
    DeviceBuffer<unsigned char> counts;
    DeviceBuffer<val_t> connection_flux, transfer, external;
    DeviceBuffer<val_t> energy_a, energy_b, flux;
    DeviceBuffer<ElementDynamic> output;
    std::vector<ElementDynamic> elements_dynamic;
    cudaStream_t stream = nullptr;
    World() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~World() { if (stream) cudaStreamDestroy(stream); }
};

__global__ void buildSquareKernel(int root, size_t size, size_t stride,
                                  idx_t* neighbors, unsigned char* counts,
                                  val_t* connection_flux, val_t* transfer,
                                  val_t* external, val_t* energy, val_t* flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const int x = i / root;
    const int y = i % root;
    unsigned char count = 0;
    // Preserve the original connection order, including at the boundaries.
    if (x + 1 < root) neighbors[size_t(count++) * stride + i] = i + root;
    if (x > 0)        neighbors[size_t(count++) * stride + i] = i - root;
    if (y + 1 < root) neighbors[size_t(count++) * stride + i] = i + 1;
    if (y > 0)        neighbors[size_t(count++) * stride + i] = i - 1;
    for (int j = 0; j < count; ++j) connection_flux[size_t(j) * stride + i] = 1.0;
    counts[i] = count;
    transfer[i] = 0.8;
    val_t source = 0.0;
    // Assign in the same order as the CPU builder (also handles root == 1).
    if (i == 0) source = 0.5;
    if (i == size_t(root - 1)) source = -0.5;
    if (i == size_t(root - 1) * root) source = -0.5;
    if (i == size - 1) source = 0.5;
    external[i] = source;
    energy[i] = 0.0;
    flux[i] = 0.0;
}

void buildSquare2D(World& world, const int root) {
    world.size = size_t(root) * root;
    world.stride = (world.size + 31) & ~size_t(31);
    world.neighbors.allocate(SQUARE_CONNECTIONS * world.stride);
    world.connection_flux.allocate(SQUARE_CONNECTIONS * world.stride);
    world.counts.allocate(world.stride);
    world.transfer.allocate(world.stride);
    world.external.allocate(world.stride);
    world.energy_a.allocate(world.stride);
    world.energy_b.allocate(world.stride);
    world.flux.allocate(world.stride);
    world.output.allocate(world.size);
    world.elements_dynamic.resize(world.size);
    const unsigned blocks = (world.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildSquareKernel<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
        root, world.size, world.stride, world.neighbors.data, world.counts.data,
        world.connection_flux.data, world.transfer.data, world.external.data,
        world.energy_a.data, world.flux.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

__global__ void updateElements(size_t size, size_t stride,
                               const idx_t* __restrict__ neighbors,
                               const unsigned char* __restrict__ counts,
                               const val_t* __restrict__ connection_flux,
                               const val_t* __restrict__ transfer,
                               const val_t* __restrict__ external,
                               const val_t* __restrict__ energy,
                               val_t* __restrict__ next,
                               val_t* __restrict__ flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const val_t current = energy[i];
    const val_t coefficient = transfer[i];
    val_t total = external[i];
    for (unsigned j = 0; j < counts[i]; ++j) {
        const size_t slot = size_t(j) * stride + i;
        total += (energy[neighbors[slot]] - current) * coefficient *
                 connection_flux[slot] * 0.25;
    }
    next[i] = current + total;
    // Only this thread uses its accumulated flux, so it can stay in place.
    flux[i] += fabs(total);
}

__global__ void packResults(size_t size, const val_t* energy, const val_t* flux,
                            ElementDynamic* output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) output[i] = ElementDynamic{energy[i], flux[i]};
}

void runSimulation(World& world, const int n_iters) {
    const unsigned blocks = (world.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    val_t* current = world.energy_a.data;
    val_t* next = world.energy_b.data;
    auto step = [&] {
        updateElements<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
            world.size, world.stride, world.neighbors.data, world.counts.data,
            world.connection_flux.data, world.transfer.data, world.external.data,
            current, next, world.flux.data);
        std::swap(current, next);
    };

    // Replay even-sized batches to reduce CPU launch overhead without changing
    // buffer parity. Stream dependencies provide the global iteration barrier.
    constexpr int batch = 32;
    int completed = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (n_iters >= 2 * batch) {
        CUDA_CHECK(cudaStreamBeginCapture(world.stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < batch; ++j) step();
        CUDA_CHECK(cudaStreamEndCapture(world.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (; completed <= n_iters - batch; completed += batch)
            CUDA_CHECK(cudaGraphLaunch(executable, world.stream));
    }
    for (; completed < n_iters; ++completed) step();
    CUDA_CHECK(cudaGetLastError());
    packResults<<<blocks, BLOCK_SIZE, 0, world.stream>>>(
        world.size, current, world.flux.data, world.output.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), world.output.data,
                              world.size * sizeof(ElementDynamic),
                              cudaMemcpyDeviceToHost, world.stream));
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    // Retain the final device state if this world is simulated again.
    world.energy_a.data = current;
    world.energy_b.data = next;
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
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
    
    if (n_elems_root <= 0 || n_elems_root > 46340) {
        fprintf(stderr, "Grid size must be between 1 and 46340.\n");
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
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.stride *
        (SQUARE_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t)) +
         sizeof(unsigned char) + 2 * sizeof(val_t));
    const size_t dynamic_mem = world.stride * 3 * sizeof(val_t) +
        world.size * sizeof(ElementDynamic);
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
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = n_measured_iters > 0 ? duration_ms / n_measured_iters : 0.0;
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
