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

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

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
}

// All simulation work runs on CUDA; failures are reported instead of falling
// back to a different execution path.
void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Slot-major connectivity coalesces the reads made by adjacent GPU threads.
// Energies and accumulated fluxes are separate so neighbor gathers fetch only
// the energy. Each thread exclusively owns its accumulated-flux entry.
__global__ void updateElements(size_t count, const idx_t* __restrict__ neighbors,
                               const val_t* __restrict__ weights,
                               const unsigned char* __restrict__ degrees,
                               const val_t* __restrict__ coefficients,
                               const val_t* __restrict__ sources,
                               const val_t* __restrict__ energy,
                               val_t* __restrict__ next,
                               val_t* __restrict__ accumulated) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const val_t current = energy[i];
        val_t flux = sources[i];
        for (unsigned int j = 0; j < degrees[i]; ++j) {
            const size_t slot = size_t(j) * count + i;
            flux += (energy[neighbors[slot]] - current) * coefficients[i]
                    * weights[slot] * 0.25;
        }
        next[i] = current + flux;
        accumulated[i] += fabs(flux);
    }
}

struct DeviceWorld {
    size_t count;
    idx_t* neighbors = nullptr;
    unsigned char* degrees = nullptr;
    val_t *weights = nullptr, *coefficients = nullptr, *sources = nullptr;
    val_t *energy = nullptr, *next = nullptr, *accumulated = nullptr;
    cudaStream_t stream;

    template<class T>
    static void allocate(T*& pointer, size_t n) {
        checkCuda(cudaMalloc(&pointer, std::max(n, size_t(1)) * sizeof(T)));
    }

    template<class T>
    static void upload(T* pointer, const std::vector<T>& values) {
        if (!values.empty())
            checkCuda(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
    }

    explicit DeviceWorld(const World& world) : count(world.elements_static.size()) {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        size_t max_degree = 0;
        for (const auto& e : world.elements_static)
            max_degree = std::max(max_degree, size_t(e.num_connections));
        std::vector<idx_t> host_neighbors(count * max_degree);
        std::vector<val_t> host_weights(count * max_degree);
        std::vector<unsigned char> host_degrees(count);
        std::vector<val_t> host_coefficients(count), host_sources(count);
        std::vector<val_t> host_energy(count), host_accumulated(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& e = world.elements_static[i];
            const auto& mat = world.materials[e.material_idx];
            host_degrees[i] = static_cast<unsigned char>(e.num_connections);
            host_coefficients[i] = mat.transfer_coeff;
            host_sources[i] = mat.external_flow;
            host_energy[i] = world.elements_dynamic[i].current_energy;
            host_accumulated[i] = world.elements_dynamic[i].total_flux;
            for (size_t j = 0; j < e.num_connections; ++j) {
                host_neighbors[j * count + i] = e.connected_idx[j];
                host_weights[j * count + i] = e.connected_flux[j];
            }
        }
        allocate(neighbors, host_neighbors.size());
        allocate(weights, host_weights.size());
        allocate(degrees, count);
        allocate(coefficients, count);
        allocate(sources, count);
        allocate(energy, count);
        allocate(next, count);
        allocate(accumulated, count);
        upload(neighbors, host_neighbors);
        upload(weights, host_weights);
        upload(degrees, host_degrees);
        upload(coefficients, host_coefficients);
        upload(sources, host_sources);
        upload(energy, host_energy);
        upload(accumulated, host_accumulated);
        // Load the kernel before timing, without modifying the state.
        checkCuda(cudaFuncSetCacheConfig(updateElements, cudaFuncCachePreferL1));
        checkCuda(cudaDeviceSynchronize());
    }

    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;
    ~DeviceWorld() {
        cudaFree(neighbors);
        cudaFree(weights);
        cudaFree(degrees);
        cudaFree(coefficients);
        cudaFree(sources);
        cudaFree(energy);
        cudaFree(next);
        cudaFree(accumulated);
        cudaStreamDestroy(stream);
    }

    void step() {
        const unsigned blocks = static_cast<unsigned>(
            std::min((count + 255) / 256, size_t(65535)));
        updateElements<<<blocks, 256, 0, stream>>>(count, neighbors, weights,
            degrees, coefficients, sources, energy, next, accumulated);
        std::swap(energy, next);
    }
};

// Stream ordering supplies the global barrier between Jacobi iterations.
// An even-sized CUDA graph can be replayed without changing buffer bindings.
void runSimulation(World& world, DeviceWorld& device, const int n_iters) {
    constexpr int batch = 32;
    int iter = 0;
    if (n_iters >= 2 * batch) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        checkCuda(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < batch; ++j) device.step();
        checkCuda(cudaStreamEndCapture(device.stream, &graph));
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (; iter <= n_iters - batch; iter += batch)
            checkCuda(cudaGraphLaunch(executable, device.stream));
        checkCuda(cudaGraphExecDestroy(executable));
        checkCuda(cudaGraphDestroy(graph));
    }
    for (; iter < n_iters; ++iter) device.step();
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(device.stream));

    // Preserve the CPU result layout used by validation and hashing.
    std::vector<val_t> energy(device.count), flux(device.count);
    checkCuda(cudaMemcpy(energy.data(), device.energy, device.count * sizeof(val_t),
                         cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(flux.data(), device.accumulated, device.count * sizeof(val_t),
                         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < device.count; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
    }
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
        fprintf(stderr, "Grid size must be between 1 and 46340\n");
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
    DeviceWorld device(world);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, device, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
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
