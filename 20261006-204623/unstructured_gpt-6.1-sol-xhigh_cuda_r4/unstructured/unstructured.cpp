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

using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

__constant__ Material device_materials[3] = {
    {0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}
};

constexpr unsigned DEFAULT_MAT_ID = 0;
constexpr unsigned INFLOW_MAT_ID = 1;
constexpr unsigned OUTFLOW_MAT_ID = 2;
constexpr unsigned CONNECTIONS = 4;
constexpr unsigned BLOCK_SIZE = 256;
constexpr int GRAPH_STEPS = 32;  // Even: a replay returns to the same energy buffer.

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Device allocations live for the entire simulation, never inside an iteration.
template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }
};

struct World {
    unsigned n_elems = 0;
    unsigned blocks = 0;
    // Neighbor slots are separate contiguous arrays so consecutive threads read
    // consecutive indices. Only four unit-flux connections exist in this mesh.
    DeviceBuffer<uint32_t> neighbors;
    // Low three bits: connection count. Remaining bits: material index.
    DeviceBuffer<uint8_t> metadata;
    DeviceBuffer<val_t> energy[2];
    DeviceBuffer<val_t> accumulated_flux;
    std::vector<val_t> host_energy;
    std::vector<val_t> host_flux;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    int graph_steps = 0;
    unsigned current_buffer = 0;

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        if (executable_graph) cudaGraphExecDestroy(executable_graph);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Construct exactly the original ordered adjacency lists, directly on the GPU.
__global__ void buildMeshKernel(unsigned root, unsigned n_elems,
                                uint32_t* __restrict__ neighbors,
                                uint8_t* __restrict__ metadata,
                                val_t* __restrict__ energy,
                                val_t* __restrict__ accumulated_flux) {
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const unsigned x = i / root;
        const unsigned y = i % root;
        unsigned count = 0;
        if (x + 1 < root) neighbors[size_t(count++) * n_elems + i] = i + root;
        if (x > 0)        neighbors[size_t(count++) * n_elems + i] = i - root;
        if (y + 1 < root) neighbors[size_t(count++) * n_elems + i] = i + 1;
        if (y > 0)        neighbors[size_t(count++) * n_elems + i] = i - 1;

        unsigned material = DEFAULT_MAT_ID;
        // Assignment order matters for the one-element grid.
        if (i == 0) material = INFLOW_MAT_ID;
        if (i == root - 1) material = OUTFLOW_MAT_ID;
        if (i == n_elems - root) material = OUTFLOW_MAT_ID;
        if (i == n_elems - 1) material = INFLOW_MAT_ID;
        metadata[i] = uint8_t(count | (material << 3));
        energy[i] = 0.0;
        accumulated_flux[i] = 0.0;
    }
}

__global__ void updateElementsKernel(unsigned n_elems,
                                     const uint32_t* __restrict__ neighbors,
                                     const uint8_t* __restrict__ metadata,
                                     const val_t* __restrict__ current,
                                     val_t* __restrict__ next,
                                     val_t* __restrict__ accumulated_flux) {
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const unsigned info = metadata[i];
        const unsigned count = info & 7;
        const Material mat = device_materials[info >> 3];
        const val_t this_energy = current[i];
        val_t total_flux = mat.external_flow;
        // Preserve the original neighbor order and arithmetic. Multiplication
        // by connected_flux is omitted because buildSquare2D sets it to 1.0.
        #pragma unroll
        for (unsigned j = 0; j < CONNECTIONS; ++j) {
            if (j < count) {
                const uint32_t neighbor = neighbors[size_t(j) * n_elems + i];
                total_flux += (current[neighbor] - this_energy) * mat.transfer_coeff * 0.25;
            }
        }
        next[i] = this_energy + total_flux;
        // No other thread reads this element's accumulated flux, so this field
        // needs just one buffer and one writer, with no atomic operations.
        accumulated_flux[i] += fabs(total_flux);
    }
}

void launchUpdate(World& world, unsigned current) {
    updateElementsKernel<<<world.blocks, BLOCK_SIZE, 0, world.stream>>>(
        world.n_elems, world.neighbors.data, world.metadata.data,
        world.energy[current].data, world.energy[current ^ 1].data,
        world.accumulated_flux.data);
    cudaCheck(cudaGetLastError(), "updateElementsKernel launch");
}

void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems = unsigned(n_elems_root) * unsigned(n_elems_root);
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    cudaCheck(cudaStreamCreateWithFlags(&world.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    // A bounded grid sustains occupancy without requiring one block per chunk
    // of a very large mesh; threads then traverse the mesh with a grid stride.
    world.blocks = std::min((world.n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE,
                            unsigned(properties.multiProcessorCount) * 32);
    world.neighbors.allocate(size_t(CONNECTIONS) * world.n_elems);
    world.metadata.allocate(world.n_elems);
    world.energy[0].allocate(world.n_elems);
    world.energy[1].allocate(world.n_elems);
    world.accumulated_flux.allocate(world.n_elems);
    world.host_energy.resize(world.n_elems);
    world.host_flux.resize(world.n_elems);
    buildMeshKernel<<<world.blocks, BLOCK_SIZE, 0, world.stream>>>(
        unsigned(n_elems_root), world.n_elems, world.neighbors.data, world.metadata.data,
        world.energy[0].data, world.accumulated_flux.data);
    cudaCheck(cudaGetLastError(), "buildMeshKernel launch");
    cudaCheck(cudaStreamSynchronize(world.stream), "mesh initialization");
}

// Capture a bounded batch once. Replaying it avoids per-iteration CPU launch
// overhead, while stream dependencies supply the global timestep barriers.
void prepareSimulation(World& world, int n_iters) {
    if (n_iters <= 1) return;
    world.graph_steps = std::min(n_iters, GRAPH_STEPS);
    cudaCheck(cudaStreamBeginCapture(world.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    for (int iter = 0; iter < world.graph_steps; ++iter) {
        launchUpdate(world, unsigned(iter) & 1);
    }
    cudaCheck(cudaStreamEndCapture(world.stream, &world.graph), "cudaStreamEndCapture");
    cudaCheck(cudaGraphInstantiate(&world.executable_graph, world.graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    cudaCheck(cudaGraphUpload(world.executable_graph, world.stream), "cudaGraphUpload");
    cudaCheck(cudaStreamSynchronize(world.stream), "graph preparation");
}

void runSimulation(World& world, const int n_iters) {
    int iter = 0;
    if (world.executable_graph) {
        // Repeated batches have an even length. An odd batch only occurs when
        // the entire simulation fits in a single graph replay.
        for (; iter <= n_iters - world.graph_steps; iter += world.graph_steps) {
            cudaCheck(cudaGraphLaunch(world.executable_graph, world.stream), "cudaGraphLaunch");
        }
    }
    for (; iter < n_iters; ++iter) launchUpdate(world, unsigned(iter) & 1);
    cudaCheck(cudaStreamSynchronize(world.stream), "simulation completion");
    world.current_buffer = n_iters > 0 ? unsigned(n_iters) & 1 : 0;
}

void downloadResults(World& world) {
    const size_t bytes = size_t(world.n_elems) * sizeof(val_t);
    cudaCheck(cudaMemcpyAsync(world.host_energy.data(), world.energy[world.current_buffer].data,
                              bytes, cudaMemcpyDeviceToHost, world.stream), "energy download");
    cudaCheck(cudaMemcpyAsync(world.host_flux.data(), world.accumulated_flux.data,
                              bytes, cudaMemcpyDeviceToHost, world.stream), "flux download");
    cudaCheck(cudaStreamSynchronize(world.stream), "result download");
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.host_energy.size(); ++i) {
        energy_sum += world.host_energy[i];
        flux_sum += world.host_flux[i];
        energy_max = std::max(world.host_energy[i], energy_max);
        energy_min = std::min(world.host_energy[i], energy_min);
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
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (size_t i = 0; i < world.host_energy.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &world.host_energy[i], sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.host_flux[i], sizeof(flux_bits));
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
    
    if (n_elems_root <= 0 ||
        int64_t(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Grid size must be positive and its element count must fit in a signed integer.\n");
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
    const size_t static_mem = size_t(world.n_elems) * (CONNECTIONS * sizeof(uint32_t) + sizeof(uint8_t));
    const size_t dynamic_mem = size_t(world.n_elems) * sizeof(val_t) * 3;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    prepareSimulation(world, n_iters);
    auto start = std::chrono::steady_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::steady_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    downloadResults(world);

    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (double(std::max(n_iters, 0)) * n_elems) / (duration_ms / 1000.0) / 1e9 : 0.0;
    
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
    if (printResults) {
        print_results(world.host_energy, "ElementEnergy");
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
