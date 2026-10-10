#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// The benchmark's square mesh has at most four connections per element.
constexpr int SQUARE_CONNECTIONS = 4;
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int GRAPH_BATCH_SIZE = 64;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t size) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)), "cudaMalloc");
        count = size;
    }

    size_t bytes() const { return count * sizeof(T); }
};

struct CudaStream {
    cudaStream_t handle = nullptr;
    CudaStream() { checkCuda(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "cudaStreamCreate"); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() { if (handle) cudaStreamDestroy(handle); }
};

struct CudaGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CudaGraph() = default;
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    ~CudaGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

// Connectivity is stored slot-major: neighboring CUDA threads access adjacent
// indices and weights even though energy is gathered through mesh connections.
// 32-bit indices are sufficient for every grid supported by the original int
// element count. Energy is double-buffered; accumulated flux is private to each
// element, so it can be updated in place without races or a second flux buffer.
struct World {
    CudaStream stream;
    size_t n_elems = 0;
    DeviceBuffer<Material> materials;
    DeviceBuffer<uint32_t> connected_idx;
    DeviceBuffer<val_t> connected_flux;
    DeviceBuffer<uint8_t> num_connections;
    DeviceBuffer<uint8_t> material_idx;
    DeviceBuffer<val_t> energy;
    DeviceBuffer<val_t> energy_swap;
    DeviceBuffer<val_t> total_flux;
    DeviceBuffer<ElementDynamic> results;
    CudaGraph batch_graph;
    CudaGraph tail_graph;
    int graph_iterations = 0;
    std::vector<ElementDynamic> elements_dynamic;
};

__global__ void buildSquareKernel(uint32_t* __restrict__ connected_idx,
                                 val_t* __restrict__ connected_flux,
                                 uint8_t* __restrict__ num_connections,
                                 uint8_t* __restrict__ material_idx,
                                 val_t* __restrict__ energy,
                                 val_t* __restrict__ total_flux,
                                 size_t n_elems, uint32_t root) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const uint32_t x = i / root;
    const uint32_t y = i % root;
    const int dx[SQUARE_CONNECTIONS] = {1, -1, 0, 0};
    const int dy[SQUARE_CONNECTIONS] = {0, 0, 1, -1};
    uint8_t count = 0;
    // Keep exactly the same connection order, including at boundaries.
    #pragma unroll
    for (int j = 0; j < SQUARE_CONNECTIONS; ++j) {
        const int nx = static_cast<int>(x) + dx[j];
        const int ny = static_cast<int>(y) + dy[j];
        if (nx >= 0 && nx < static_cast<int>(root) &&
            ny >= 0 && ny < static_cast<int>(root)) {
            const size_t slot = static_cast<size_t>(count++) * n_elems + i;
            connected_idx[slot] = static_cast<uint32_t>(nx) * root + ny;
            connected_flux[slot] = 1.0;
        }
    }
    num_connections[i] = count;

    uint8_t material = DEFAULT_MAT_ID;
    // Separate assignments preserve the original overwrite order for root=1.
    if (i == 0) material = INFLOW_MAT_ID;
    if (i == root - 1) material = OUTFLOW_MAT_ID;
    if (i == n_elems - root) material = OUTFLOW_MAT_ID;
    if (i == n_elems - 1) material = INFLOW_MAT_ID;
    material_idx[i] = material;
    energy[i] = 0.0;
    total_flux[i] = 0.0;
}

__global__ void simulationKernel(const Material* __restrict__ materials,
                                 const uint32_t* __restrict__ connected_idx,
                                 const val_t* __restrict__ connected_flux,
                                 const uint8_t* __restrict__ num_connections,
                                 const uint8_t* __restrict__ material_idx,
                                 const val_t* __restrict__ energy,
                                 val_t* __restrict__ next_energy,
                                 val_t* __restrict__ accumulated_flux,
                                 size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = materials[material_idx[i]];
    const val_t current = energy[i];
    val_t flux = mat.external_flow;
    const unsigned int count = num_connections[i];
    #pragma unroll
    for (unsigned int j = 0; j < SQUARE_CONNECTIONS; ++j) {
        if (j < count) {
            const size_t slot = static_cast<size_t>(j) * n_elems + i;
            flux += (energy[connected_idx[slot]] - current) *
                    mat.transfer_coeff * connected_flux[slot] * 0.25;
        }
    }
    next_energy[i] = current + flux;
    accumulated_flux[i] += fabs(flux);
}

// A mesh that fits in one block can keep both Jacobi buffers in shared memory
// for the entire run. Every thread participates in the iteration barrier,
// including inactive threads in a partially filled block.
__global__ void smallSimulationKernel(const Material* __restrict__ materials,
                                      const uint32_t* __restrict__ connected_idx,
                                      const val_t* __restrict__ connected_flux,
                                      const uint8_t* __restrict__ num_connections,
                                      const uint8_t* __restrict__ material_idx,
                                      val_t* __restrict__ energy,
                                      val_t* __restrict__ energy_swap,
                                      val_t* __restrict__ accumulated_flux,
                                      size_t n_elems, int n_iters) {
    __shared__ val_t states[2][CUDA_BLOCK_SIZE];
    const unsigned int i = threadIdx.x;
    const bool active = i < n_elems;
    Material mat{0.0, 0.0};
    unsigned int count = 0;
    uint32_t neighbors[SQUARE_CONNECTIONS];
    val_t weights[SQUARE_CONNECTIONS];
    val_t accumulated = 0.0;
    if (active) {
        mat = materials[material_idx[i]];
        count = num_connections[i];
        accumulated = accumulated_flux[i];
        states[0][i] = energy[i];
        #pragma unroll
        for (unsigned int j = 0; j < SQUARE_CONNECTIONS; ++j) {
            if (j < count) {
                const size_t slot = static_cast<size_t>(j) * n_elems + i;
                neighbors[j] = connected_idx[slot];
                weights[j] = connected_flux[slot];
            }
        }
    }
    __syncthreads();
    unsigned int current_buffer = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (active) {
            const val_t current = states[current_buffer][i];
            val_t flux = mat.external_flow;
            #pragma unroll
            for (unsigned int j = 0; j < SQUARE_CONNECTIONS; ++j) {
                if (j < count) {
                    flux += (states[current_buffer][neighbors[j]] - current) *
                            mat.transfer_coeff * weights[j] * 0.25;
                }
            }
            states[current_buffer ^ 1][i] = current + flux;
            accumulated += fabs(flux);
        }
        __syncthreads();
        current_buffer ^= 1;
    }
    if (active) {
        if (n_iters % 2 != 0) energy_swap[i] = states[current_buffer][i];
        else energy[i] = states[current_buffer][i];
        accumulated_flux[i] = accumulated;
    }
}

__global__ void packResultsKernel(const val_t* __restrict__ energy,
                                 const val_t* __restrict__ total_flux,
                                 ElementDynamic* __restrict__ results,
                                 size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_elems) results[i] = ElementDynamic{energy[i], total_flux[i]};
}

unsigned int blocksFor(size_t n_elems) {
    return static_cast<unsigned int>((n_elems + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
}

// Build the actual mesh on the GPU, avoiding serial host construction and
// uploading a large array of padded structures.
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    world.materials.allocate(3);
    world.connected_idx.allocate(SQUARE_CONNECTIONS * world.n_elems);
    world.connected_flux.allocate(SQUARE_CONNECTIONS * world.n_elems);
    world.num_connections.allocate(world.n_elems);
    world.material_idx.allocate(world.n_elems);
    world.energy.allocate(world.n_elems);
    world.energy_swap.allocate(world.n_elems);
    world.total_flux.allocate(world.n_elems);
    world.results.allocate(world.n_elems);
    world.elements_dynamic.resize(world.n_elems);

    const Material materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    checkCuda(cudaMemcpy(world.materials.data, materials, sizeof(materials),
                         cudaMemcpyHostToDevice), "copy materials");
    buildSquareKernel<<<blocksFor(world.n_elems), CUDA_BLOCK_SIZE, 0, world.stream.handle>>>(
        world.connected_idx.data, world.connected_flux.data, world.num_connections.data,
        world.material_idx.data, world.energy.data, world.total_flux.data,
        world.n_elems, static_cast<uint32_t>(n_elems_root));
    checkCuda(cudaGetLastError(), "buildSquareKernel");
    checkCuda(cudaStreamSynchronize(world.stream.handle), "build mesh");
}

void captureIterations(World& world, CudaGraph& graph, int iterations) {
    checkCuda(cudaStreamBeginCapture(world.stream.handle, cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture");
    if (world.n_elems <= CUDA_BLOCK_SIZE) {
        smallSimulationKernel<<<1, CUDA_BLOCK_SIZE, 0, world.stream.handle>>>(
            world.materials.data, world.connected_idx.data, world.connected_flux.data,
            world.num_connections.data, world.material_idx.data,
            world.energy.data, world.energy_swap.data, world.total_flux.data,
            world.n_elems, iterations);
    } else {
        val_t* current = world.energy.data;
        val_t* next = world.energy_swap.data;
        for (int iter = 0; iter < iterations; ++iter) {
            simulationKernel<<<blocksFor(world.n_elems), CUDA_BLOCK_SIZE, 0, world.stream.handle>>>(
                world.materials.data, world.connected_idx.data, world.connected_flux.data,
                world.num_connections.data, world.material_idx.data, current, next,
                world.total_flux.data, world.n_elems);
            std::swap(current, next);
        }
    }
    checkCuda(cudaGetLastError(), "capture simulation kernel");
    checkCuda(cudaStreamEndCapture(world.stream.handle, &graph.graph), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0),
              "cudaGraphInstantiate");
    checkCuda(cudaGraphUpload(graph.executable, world.stream.handle), "cudaGraphUpload");
}

void prepareSimulation(World& world, int n_iters) {
    // An even batch returns to the original energy buffer. Reusing bounded
    // graphs removes per-iteration host launches without O(n_iters) graph memory.
    world.graph_iterations = std::max(n_iters, 0);
    if (world.n_elems <= CUDA_BLOCK_SIZE) {
        if (n_iters > 0) captureIterations(world, world.tail_graph, n_iters);
    } else {
        if (n_iters >= GRAPH_BATCH_SIZE) captureIterations(world, world.batch_graph, GRAPH_BATCH_SIZE);
        if (n_iters > 0 && n_iters % GRAPH_BATCH_SIZE != 0) {
            captureIterations(world, world.tail_graph, n_iters % GRAPH_BATCH_SIZE);
        }
    }
    checkCuda(cudaStreamSynchronize(world.stream.handle), "prepare simulation");
}

// Dependencies between graph nodes provide the global Jacobi barrier. No
// transfers, allocations, atomics, or host synchronizations occur per iteration.
void runSimulation(World& world, const int n_iters) {
    if (std::max(n_iters, 0) != world.graph_iterations) {
        throw std::runtime_error("simulation iteration count differs from prepared CUDA graphs");
    }
    if (n_iters > 0 && world.n_elems <= CUDA_BLOCK_SIZE) {
        checkCuda(cudaGraphLaunch(world.tail_graph.executable, world.stream.handle), "cudaGraphLaunch");
    } else {
        for (int batch = 0; batch < n_iters / GRAPH_BATCH_SIZE; ++batch) {
            checkCuda(cudaGraphLaunch(world.batch_graph.executable, world.stream.handle), "cudaGraphLaunch");
        }
        if (n_iters > 0 && n_iters % GRAPH_BATCH_SIZE != 0) {
            checkCuda(cudaGraphLaunch(world.tail_graph.executable, world.stream.handle), "cudaGraphLaunch");
        }
    }
    checkCuda(cudaStreamSynchronize(world.stream.handle), "run simulation");
    if (n_iters > 0 && n_iters % 2 != 0) {
        std::swap(world.energy.data, world.energy_swap.data);
    }
}

void collectResults(World& world) {
    packResultsKernel<<<blocksFor(world.n_elems), CUDA_BLOCK_SIZE, 0, world.stream.handle>>>(
        world.energy.data, world.total_flux.data, world.results.data, world.n_elems);
    checkCuda(cudaGetLastError(), "packResultsKernel");
    checkCuda(cudaMemcpyAsync(world.elements_dynamic.data(), world.results.data,
                              world.results.bytes(), cudaMemcpyDeviceToHost,
                              world.stream.handle), "copy results");
    checkCuda(cudaStreamSynchronize(world.stream.handle), "collect results");
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
    
    // The original mesh uses an int element count; reject grids that would
    // overflow it before allocation or device indexing.
    if (n_elems_root <= 0 ||
        static_cast<uint64_t>(n_elems_root) * n_elems_root >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
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
    World world;
    buildSquare2D(world, n_elems_root);
    prepareSimulation(world, n_iters);
    
    // Calculate memory usage
    const size_t static_mem = world.materials.bytes() + world.connected_idx.bytes() +
                              world.connected_flux.bytes() + world.num_connections.bytes() +
                              world.material_idx.bytes();
    const size_t dynamic_mem = world.energy.bytes() + world.energy_swap.bytes() +
                               world.total_flux.bytes() + world.results.bytes();
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::steady_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::steady_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    collectResults(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = duration_ms / std::max(n_measured_iters, 1);
    const double giga_elems_per_sec = duration_ms > 0.0 ?
        (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9 : 0.0;
    
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
} catch (const std::exception& error) {
    fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
