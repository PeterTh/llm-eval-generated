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

// CUDA Error checking helper
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Device function to compute flux
__device__ inline val_t computeFluxDevice(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA Kernel for simulation
__global__ void simulationKernel(
    const int n_elems,
    const Material* __restrict__ materials,
    const idx_t* __restrict__ static_material_idx,
    const idx_t* __restrict__ static_num_connections,
    const idx_t* __restrict__ static_connected_idx,
    const val_t* __restrict__ static_connected_flux,
    const val_t* __restrict__ dynamic_current_energy,
    const val_t* __restrict__ dynamic_total_flux,
    val_t* __restrict__ dynamic_swap_current_energy,
    val_t* __restrict__ dynamic_swap_total_flux
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    idx_t mat_idx = static_material_idx[i];
    const Material& mat = materials[mat_idx];
    
    val_t current_energy = dynamic_current_energy[i];
    val_t current_total_flux = dynamic_total_flux[i];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    idx_t num_connections = static_num_connections[i];
    
    // Add flux from all connected elements
    #pragma unroll
    for (idx_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < num_connections) {
             const size_t offset = (size_t)j * n_elems + i;
             const idx_t neighbor_idx = static_connected_idx[offset];
             val_t neighbor_energy = dynamic_current_energy[neighbor_idx];
             val_t connection_flux = static_connected_flux[offset];
             
             total_flux += (neighbor_energy - current_energy) * mat.transfer_coeff * connection_flux * 0.25;
        }
    }
    
    // Update element state
    dynamic_swap_current_energy[i] = current_energy + total_flux;
    dynamic_swap_total_flux[i] = current_total_flux + abs(total_flux);
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
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, double& duration_ms) {
    const int n_elems = world.elements_static.size();
    const int n_materials = world.materials.size();

    // Allocate device memory for SoA layout
    Material* d_materials;
    
    // ElementStatic SoA
    idx_t* d_static_material_idx;
    idx_t* d_static_num_connections;
    idx_t* d_static_connected_idx; // Size: n_elems * MAX_CONNECTIONS
    val_t* d_static_connected_flux; // Size: n_elems * MAX_CONNECTIONS
    
    // ElementDynamic SoA
    val_t* d_dynamic_current_energy;
    val_t* d_dynamic_total_flux;
    
    val_t* d_dynamic_swap_current_energy;
    val_t* d_dynamic_swap_total_flux;

    CHECK_CUDA(cudaMalloc(&d_materials, n_materials * sizeof(Material)));
    
    CHECK_CUDA(cudaMalloc(&d_static_material_idx, n_elems * sizeof(idx_t)));
    CHECK_CUDA(cudaMalloc(&d_static_num_connections, n_elems * sizeof(idx_t)));
    CHECK_CUDA(cudaMalloc(&d_static_connected_idx, n_elems * MAX_CONNECTIONS * sizeof(idx_t)));
    CHECK_CUDA(cudaMalloc(&d_static_connected_flux, n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    
    CHECK_CUDA(cudaMalloc(&d_dynamic_current_energy, n_elems * sizeof(val_t)));
    CHECK_CUDA(cudaMalloc(&d_dynamic_total_flux, n_elems * sizeof(val_t)));
    
    CHECK_CUDA(cudaMalloc(&d_dynamic_swap_current_energy, n_elems * sizeof(val_t)));
    CHECK_CUDA(cudaMalloc(&d_dynamic_swap_total_flux, n_elems * sizeof(val_t)));

    // Prepare host buffers for SoA transformation
    std::vector<idx_t> h_static_material_idx(n_elems);
    std::vector<idx_t> h_static_num_connections(n_elems);
    std::vector<idx_t> h_static_connected_idx(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_static_connected_flux(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_dynamic_current_energy(n_elems);
    std::vector<val_t> h_dynamic_total_flux(n_elems);

    // Transform AoS to SoA
    for (int i = 0; i < n_elems; ++i) {
        h_static_material_idx[i] = world.elements_static[i].material_idx;
        h_static_num_connections[i] = world.elements_static[i].num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            // Transpose: [j * n_elems + i] for coalesced access
            h_static_connected_idx[j * n_elems + i] = world.elements_static[i].connected_idx[j];
            h_static_connected_flux[j * n_elems + i] = world.elements_static[i].connected_flux[j];
        }
        h_dynamic_current_energy[i] = world.elements_dynamic[i].current_energy;
        h_dynamic_total_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_materials, world.materials.data(), n_materials * sizeof(Material), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(d_static_material_idx, h_static_material_idx.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_static_num_connections, h_static_num_connections.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_static_connected_idx, h_static_connected_idx.data(), n_elems * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_static_connected_flux, h_static_connected_flux.data(), n_elems * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(d_dynamic_current_energy, h_dynamic_current_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dynamic_total_flux, h_dynamic_total_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    
    // Configure grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n_elems + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernel<<<numBlocks, blockSize>>>(
            n_elems,
            d_materials,
            d_static_material_idx,
            d_static_num_connections,
            d_static_connected_idx,
            d_static_connected_flux,
            d_dynamic_current_energy,
            d_dynamic_total_flux,
            d_dynamic_swap_current_energy,
            d_dynamic_swap_total_flux
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Swap buffers
        std::swap(d_dynamic_current_energy, d_dynamic_swap_current_energy);
        std::swap(d_dynamic_total_flux, d_dynamic_swap_total_flux);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_dynamic_current_energy.data(), d_dynamic_current_energy, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dynamic_total_flux.data(), d_dynamic_total_flux, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    
    // Transform SoA back to AoS
    for (int i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_dynamic_current_energy[i];
        world.elements_dynamic[i].total_flux = h_dynamic_total_flux[i];
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_materials));
    CHECK_CUDA(cudaFree(d_static_material_idx));
    CHECK_CUDA(cudaFree(d_static_num_connections));
    CHECK_CUDA(cudaFree(d_static_connected_idx));
    CHECK_CUDA(cudaFree(d_static_connected_flux));
    CHECK_CUDA(cudaFree(d_dynamic_current_energy));
    CHECK_CUDA(cudaFree(d_dynamic_total_flux));
    CHECK_CUDA(cudaFree(d_dynamic_swap_current_energy));
    CHECK_CUDA(cudaFree(d_dynamic_swap_total_flux));
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
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    double kernel_duration_ms = 0.0;
    runSimulation(world, n_iters, kernel_duration_ms);
    
    printf("Computation time: %ld ms\n", (long)kernel_duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = kernel_duration_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (kernel_duration_ms / 1000.0) / 1e9;
    
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
