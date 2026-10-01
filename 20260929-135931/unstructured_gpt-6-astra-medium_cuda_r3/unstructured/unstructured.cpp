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

// The generated square mesh has at most four connections. Connectivity stays
// explicit: the simulation kernel also handles irregular neighbor lists.
using idx_t = uint32_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 4;
constexpr int BLOCK_SIZE = 256;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation,
                cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

// Structure-of-arrays storage gives consecutive threads consecutive metadata
// accesses. Each connection slot occupies one plane of n elements.
struct World {
    size_t size = 0;
    DeviceBuffer<idx_t> neighbors;
    DeviceBuffer<val_t> connection_flux;
    DeviceBuffer<unsigned char> counts, material_ids;
    DeviceBuffer<val_t> energy, energy_swap, total_flux;
    DeviceBuffer<ElementDynamic> results;
    std::vector<ElementDynamic> elements_dynamic;
};

__constant__ Material materials[3];

__global__ void buildMesh(int root, size_t n, idx_t* neighbors,
                          val_t* connection_flux, unsigned char* counts,
                          unsigned char* material_ids, val_t* energy,
                          val_t* total_flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = i / root;
    const int y = i % root;
    int count = 0;
    // Preserve the CPU neighbor order, including at edges and corners.
    if (x + 1 < root) neighbors[size_t(count++) * n + i] = i + root;
    if (x > 0)        neighbors[size_t(count++) * n + i] = i - root;
    if (y + 1 < root) neighbors[size_t(count++) * n + i] = i + 1;
    if (y > 0)        neighbors[size_t(count++) * n + i] = i - 1;
    for (int j = 0; j < count; ++j) connection_flux[size_t(j) * n + i] = 1.0;
    counts[i] = count;
    unsigned char material = 0;
    if (i == 0) material = 1;
    if (i == size_t(root - 1)) material = 2;
    if (i == size_t(root - 1) * root) material = 2;
    if (i == n - 1) material = 1; // Also preserves the 1x1 assignment order.
    material_ids[i] = material;
    energy[i] = 0.0;
    total_flux[i] = 0.0;
}

void buildSquare2D(World& world, const int root) {
    world.size = size_t(root) * root;
    const size_t n = world.size;
    world.neighbors.allocate(MAX_CONNECTIONS * n);
    world.connection_flux.allocate(MAX_CONNECTIONS * n);
    world.counts.allocate(n);
    world.material_ids.allocate(n);
    world.energy.allocate(n);
    world.energy_swap.allocate(n);
    world.total_flux.allocate(n);
    world.results.allocate(n);
    world.elements_dynamic.resize(n);
    const Material host_materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    CUDA_CHECK(cudaMemcpyToSymbol(materials, host_materials, sizeof(host_materials)));
    buildMesh<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        root, n, world.neighbors.data, world.connection_flux.data,
        world.counts.data, world.material_ids.data, world.energy.data,
        world.total_flux.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void simulationStep(size_t n, const idx_t* __restrict__ neighbors,
                               const val_t* __restrict__ connection_flux,
                               const unsigned char* __restrict__ counts,
                               const unsigned char* __restrict__ material_ids,
                               const val_t* __restrict__ energy,
                               val_t* __restrict__ next,
                               val_t* __restrict__ accumulated_flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const val_t current = energy[i];
    const Material mat = materials[material_ids[i]];
    val_t flux = mat.external_flow;
    for (unsigned int j = 0; j < counts[i]; ++j) {
        const size_t slot = size_t(j) * n + i;
        flux += (energy[neighbors[slot]] - current) * mat.transfer_coeff *
                connection_flux[slot] * 0.25;
    }
    next[i] = current + flux;
    // Only the owning thread reads or writes this accumulator, so it needs
    // neither atomics nor a second buffer.
    accumulated_flux[i] += fabs(flux);
}

__global__ void collectResults(size_t n, const val_t* energy,
                               const val_t* flux, ElementDynamic* results) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) results[i] = {energy[i], flux[i]};
}

void runSimulation(World& world, const int n_iters) {
    const size_t n = world.size;
    const unsigned int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    val_t* current = world.energy.data;
    val_t* next = world.energy_swap.data;
    auto step = [&]() {
        simulationStep<<<blocks, BLOCK_SIZE, 0, stream>>>(
            n, world.neighbors.data, world.connection_flux.data,
            world.counts.data, world.material_ids.data, current, next,
            world.total_flux.data);
        std::swap(current, next);
    };
    // An even batch returns to the same ping-pong buffer. Graph replay keeps
    // launch overhead small on short meshes without growing with n_iters.
    constexpr int batch = 32;
    int completed = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (n_iters >= 2 * batch) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < batch; ++j) step();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (; completed <= n_iters - batch; completed += batch)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
    for (; completed < n_iters; ++completed) step();
    collectResults<<<blocks, BLOCK_SIZE, 0, stream>>>(
        n, current, world.total_flux.data, world.results.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(world.elements_dynamic.data(), world.results.data,
                               n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Keep the current device state available for another simulation call.
    world.energy.data = current;
    world.energy_swap.data = next;
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
    
    if (n_elems_root <= 0 || n_elems_root > 46340 || n_iters < 0) {
        fprintf(stderr, "Grid size must be in [1, 46340] and iterations must be nonnegative.\n");
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
    const size_t static_mem = world.size * (MAX_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t)) + 2);
    const size_t dynamic_mem = world.size * (3 * sizeof(val_t) + 2 * sizeof(ElementDynamic));
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
    const int n_measured_iters = n_iters;
    const double time_per_iter = duration_ms / std::max(n_measured_iters, 1);
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
