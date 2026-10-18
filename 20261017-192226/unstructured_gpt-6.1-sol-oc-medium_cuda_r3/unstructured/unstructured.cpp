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

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
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
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

// Connection-major arrays coalesce accesses by adjacent CUDA threads, even for
// an irregular mesh. All simulation data remains resident between time steps.
struct DeviceMesh {
    DeviceBuffer<idx_t> neighbors;
    DeviceBuffer<val_t> connection_flux;
    DeviceBuffer<unsigned char> degree;
    DeviceBuffer<Material> material;
    DeviceBuffer<ElementDynamic> state[2];
    cudaStream_t stream = nullptr;
    ~DeviceMesh() { if (stream) cudaStreamDestroy(stream); }
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    DeviceMesh device;
};

__global__ void packMesh(const ElementStatic* source, const Material* materials,
                         idx_t* neighbors, val_t* flux, unsigned char* degree,
                         Material* material, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const ElementStatic& element = source[i];
    degree[i] = static_cast<unsigned char>(element.num_connections);
    material[i] = materials[element.material_idx];
    for (idx_t j = 0; j < element.num_connections; ++j) {
        neighbors[j * count + i] = element.connected_idx[j];
        flux[j * count + i] = element.connected_flux[j];
    }
}

void uploadMesh(World& world) {
    const size_t count = world.elements_static.size();
    auto& device = world.device;
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    size_t max_degree = 0;
    for (const auto& element : world.elements_static)
        max_degree = std::max(max_degree, size_t(element.num_connections));
    device.neighbors.allocate(count * max_degree);
    device.connection_flux.allocate(count * max_degree);
    device.degree.allocate(count);
    device.material.allocate(count);
    for (auto& state : device.state) state.allocate(count);

    DeviceBuffer<ElementStatic> source;
    DeviceBuffer<Material> materials;
    source.allocate(count);
    materials.allocate(world.materials.size());
    CUDA_CHECK(cudaMemcpyAsync(source.data, world.elements_static.data(),
        count * sizeof(ElementStatic), cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(materials.data, world.materials.data(),
        world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(device.state[0].data, world.elements_dynamic.data(),
        count * sizeof(ElementDynamic), cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(device.state[1].data, world.elements_dynamic_swap.data(),
        count * sizeof(ElementDynamic), cudaMemcpyHostToDevice, device.stream));
    packMesh<<<(count + 255) / 256, 256, 0, device.stream>>>(source.data, materials.data,
        device.neighbors.data, device.connection_flux.data, device.degree.data,
        device.material.data, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
}

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    uploadMesh(world);
}

__global__ void updateElements(const idx_t* __restrict__ neighbors,
                               const val_t* __restrict__ connection_flux,
                               const unsigned char* __restrict__ degree,
                               const Material* __restrict__ materials,
                               const ElementDynamic* __restrict__ current,
                               ElementDynamic* __restrict__ next, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const ElementDynamic element = current[i];
    const Material mat = materials[i];
    val_t total_flux = mat.external_flow;
    // Preserve the original neighbor and arithmetic order for reproducibility.
    #pragma unroll 4
    for (unsigned int j = 0; j < degree[i]; ++j) {
        const size_t offset = size_t(j) * count + i;
        total_flux += (current[neighbors[offset]].current_energy - element.current_energy)
                      * mat.transfer_coeff * connection_flux[offset] * 0.25;
    }
    next[i].current_energy = element.current_energy + total_flux;
    next[i].total_flux = element.total_flux + fabs(total_flux);
}

// A kernel boundary provides the global Jacobi barrier between iterations.
void runSimulation(World& world, const int n_iters) {
    const size_t count = world.elements_static.size();
    auto& device = world.device;
    auto* current = device.state[0].data;
    auto* next = device.state[1].data;
    auto step = [&]() {
        updateElements<<<(count + 255) / 256, 256, 0, device.stream>>>(
            device.neighbors.data, device.connection_flux.data, device.degree.data,
            device.material.data, current, next, count);
        std::swap(current, next);
    };

    // Reuse a bounded-size graph for long runs, amortizing host launch overhead.
    // An even batch returns the ping-pong pointers to their starting positions.
    constexpr int batch = 32;
    int iter = 0;
    if (n_iters >= 2 * batch) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < batch; ++j) step();
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (; iter <= n_iters - batch; iter += batch)
            CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    for (; iter < n_iters; ++iter) step();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), current,
        count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic_swap.data(), next,
        count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    // Keep the device's logical current buffer consistent for subsequent calls.
    if (current != device.state[0].data)
        std::swap(device.state[0].data, device.state[1].data);
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
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
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
    const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (static_cast<double>(n_iters) * n_elems) / (duration_ms * 1e6) : 0.0;
    
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
