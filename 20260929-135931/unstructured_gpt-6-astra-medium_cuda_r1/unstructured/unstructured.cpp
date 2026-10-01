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

// Keep every iteration on the GPU. Slot-major connectivity coalesces accesses
// across elements while retaining each element's original neighbor order.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void updateElements(size_t count, const unsigned* __restrict__ degree,
                               const unsigned* __restrict__ neighbors,
                               const val_t* __restrict__ weights,
                               const val_t* __restrict__ coeff,
                               const val_t* __restrict__ source,
                               const val_t* __restrict__ energy,
                               const val_t* __restrict__ flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const val_t e = energy[i];
    val_t total = source[i];
    for (unsigned j = 0; j < degree[i]; ++j) {
        const size_t slot = size_t(j) * count + i;
        total += (energy[neighbors[slot]] - e) * coeff[i] * weights[slot] * 0.25;
    }
    next_energy[i] = e + total;
    next_flux[i] = flux[i] + fabs(total);
}

class CudaSimulation {
    size_t count;
    unsigned* indices = nullptr;
    val_t* values = nullptr;
    unsigned *degree, *neighbors;
    val_t *weights, *coeff, *source, *energy, *flux, *next_energy, *next_flux;
    cudaStream_t stream{};
    cudaGraphExec_t graph = nullptr;
    static constexpr int batch = 32; // Even batch keeps ping-pong pointers stable.

    void step() {
        updateElements<<<(count + 255) / 256, 256, 0, stream>>>(
            count, degree, neighbors, weights, coeff, source,
            energy, flux, next_energy, next_flux);
        std::swap(energy, next_energy);
        std::swap(flux, next_flux);
    }

public:
    CudaSimulation(const World& world, int n_iters) : count(world.elements_static.size()) {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        size_t slots = 0;
        for (const auto& elem : world.elements_static)
            slots = std::max(slots, size_t(elem.num_connections));
        std::vector<unsigned> host_indices((slots + 1) * count);
        std::vector<val_t> host_values((slots + 6) * count);
        for (size_t i = 0; i < count; ++i) {
            const auto& elem = world.elements_static[i];
            const auto& mat = world.materials[elem.material_idx];
            host_indices[i] = unsigned(elem.num_connections);
            for (size_t j = 0; j < elem.num_connections; ++j) {
                host_indices[(j + 1) * count + i] = unsigned(elem.connected_idx[j]);
                host_values[j * count + i] = elem.connected_flux[j];
            }
            host_values[slots * count + i] = mat.transfer_coeff;
            host_values[(slots + 1) * count + i] = mat.external_flow;
            host_values[(slots + 2) * count + i] = world.elements_dynamic[i].current_energy;
            host_values[(slots + 3) * count + i] = world.elements_dynamic[i].total_flux;
            host_values[(slots + 4) * count + i] = world.elements_dynamic_swap[i].current_energy;
            host_values[(slots + 5) * count + i] = world.elements_dynamic_swap[i].total_flux;
        }
        cudaCheck(cudaMalloc(&indices, host_indices.size() * sizeof(unsigned)));
        cudaCheck(cudaMalloc(&values, host_values.size() * sizeof(val_t)));
        cudaCheck(cudaMemcpyAsync(indices, host_indices.data(), host_indices.size() * sizeof(unsigned),
                                  cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(values, host_values.data(), host_values.size() * sizeof(val_t),
                                  cudaMemcpyHostToDevice, stream));
        degree = indices;
        neighbors = indices + count;
        weights = values;
        coeff = weights + slots * count;
        source = coeff + count;
        energy = source + count;
        flux = energy + count;
        next_energy = flux + count;
        next_flux = next_energy + count;
        cudaCheck(cudaStreamSynchronize(stream));
        if (n_iters >= 2 * batch) {
            cudaGraph_t captured;
            cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < batch; ++i) step();
            cudaCheck(cudaStreamEndCapture(stream, &captured));
            cudaCheck(cudaGraphInstantiate(&graph, captured, nullptr, nullptr, 0));
            cudaCheck(cudaGraphDestroy(captured));
            cudaCheck(cudaGraphUpload(graph, stream));
            cudaCheck(cudaStreamSynchronize(stream));
        }
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    void run(int n_iters) {
        int iter = 0;
        if (graph) {
            for (; n_iters - iter >= batch; iter += batch)
                cudaCheck(cudaGraphLaunch(graph, stream));
        }
        for (; iter < n_iters; ++iter) step();
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void download(World& world) {
        std::vector<val_t> host(4 * count);
        const val_t* buffers[] = {energy, flux, next_energy, next_flux};
        for (int j = 0; j < 4; ++j)
            cudaCheck(cudaMemcpyAsync(host.data() + j * count, buffers[j], count * sizeof(val_t),
                                      cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < count; ++i) {
            world.elements_dynamic[i] = {host[i], host[count + i]};
            world.elements_dynamic_swap[i] = {host[2 * count + i], host[3 * count + i]};
        }
    }

    ~CudaSimulation() {
        if (graph) cudaCheck(cudaGraphExecDestroy(graph));
        cudaCheck(cudaFree(values));
        cudaCheck(cudaFree(indices));
        cudaCheck(cudaStreamDestroy(stream));
    }
};

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
    // Device allocation, mesh upload and graph preparation are setup work.
    CudaSimulation simulation(world, n_iters);
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    simulation.download(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = duration_ms / std::max(n_measured_iters, 1);
    const double giga_elems_per_sec = (double(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
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
