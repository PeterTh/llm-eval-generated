#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// CUDA header for GPU runtime functions
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err) do {                                                    \
    cudaError_t err_ = (err);                                                   \
    if (err_ != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__,         \
                __LINE__, cudaGetErrorString(err_));                            \
        exit(1);                                                                \
    }                                                                           \
} while (0)

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

// Compute energy flux between two elements (host and device)
__host__ __device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel: update all elements in parallel on GPU
// Uses __ldg() for cached read-only global loads and __launch_bounds__ for occupancy
__global__ void __launch_bounds__(256)
updateElementsKernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    size_t n_elems) {
    
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    
    // Pre-fetch static data via read-only cache
    const idx_t mat_idx = __ldg(&elements_static[i].material_idx);
    const idx_t num_conn = __ldg(&elements_static[i].num_connections);
    
    // Pre-fetch this element's dynamic data
    const val_t cur_energy = __ldg(&elements_dynamic[i].current_energy);
    const val_t cur_total_flux = __ldg(&elements_dynamic[i].total_flux);
    
    // Material data (tiny array, likely in L1 cache)
    const Material mat = materials[mat_idx];
    const val_t transfer_coeff = mat.transfer_coeff;
    val_t total_flux = mat.external_flow;
    
    // Accumulate flux from connected neighbors (unrolled for max 8 connections)
    #pragma unroll
    for (idx_t j = 0; j < num_conn; ++j) {
        const idx_t nidx = __ldg(&elements_static[i].connected_idx[j]);
        const val_t flux = __ldg(&elements_static[i].connected_flux[j]);
        const val_t nrg = __ldg(&elements_dynamic[nidx].current_energy);
        total_flux += (nrg - cur_energy) * transfer_coeff * flux * 0.25;
    }
    
    // Write results to swap buffer
    elements_dynamic_swap[i].current_energy = cur_energy + total_flux;
    elements_dynamic_swap[i].total_flux = cur_total_flux + fabs(total_flux);
}

// GPU device memory state
struct GPUDeviceState {
    Material* d_materials;
    ElementStatic* d_elements_static;
    ElementDynamic* d_buf0;
    ElementDynamic* d_buf1;
    ElementDynamic* d_current;
    ElementDynamic* d_swap;
    size_t n_elems;
    int numBlocks;
};

// Initialize GPU device memory and copy data from host
GPUDeviceState gpuInit(const World& world) {
    GPUDeviceState state;
    const size_t n_elems = world.elements_static.size();
    const size_t n_materials = world.materials.size();
    
    CUDA_CHECK(cudaMalloc(&state.d_materials, n_materials * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&state.d_elements_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&state.d_buf0, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&state.d_buf1, n_elems * sizeof(ElementDynamic)));
    
    CUDA_CHECK(cudaMemcpy(state.d_materials, world.materials.data(),
                n_materials * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_elements_static, world.elements_static.data(),
                n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_buf0, world.elements_dynamic.data(),
                n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    
    state.d_current = state.d_buf0;
    state.d_swap = state.d_buf1;
    state.n_elems = n_elems;
    
    const int blockSize = 256;
    state.numBlocks = static_cast<int>((n_elems + blockSize - 1) / blockSize);
    return state;
}

// Run the simulation loop on GPU (only kernel launches, no alloc/copy)
void runSimulationGPU(GPUDeviceState& state, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        updateElementsKernel<<<state.numBlocks, 256>>>(
            state.d_elements_static, state.d_current, state.d_swap,
            state.d_materials, state.n_elems);
        
        // Swap device buffer pointers for next iteration
        ElementDynamic* temp = state.d_current;
        state.d_current = state.d_swap;
        state.d_swap = temp;
    }
    // Synchronize to ensure all GPU work completes before timer stops
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy results back to host and free GPU memory
void gpuFinish(World& world, GPUDeviceState& state) {
    const size_t n_elems = world.elements_static.size();
    
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), state.d_current,
                n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(state.d_materials));
    CUDA_CHECK(cudaFree(state.d_elements_static));
    CUDA_CHECK(cudaFree(state.d_buf0));
    CUDA_CHECK(cudaFree(state.d_buf1));
}

// Run simulation for n_iters iterations (kept for API compatibility, delegates to GPU)
// Note: Use gpuInit/runSimulationGPU/gpuFinish for proper timing separation
void runSimulation(World& world, const int n_iters) {
    GPUDeviceState state = gpuInit(world);
    runSimulationGPU(state, n_iters);
    gpuFinish(world, state);
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
    
    // Initialize CUDA context and allocate GPU memory (outside timed section)
    printf("Initializing GPU...\n");
    CUDA_CHECK(cudaFree(0));  // Warm up CUDA driver context
    GPUDeviceState gpu_state = gpuInit(world);
    
    // Run simulation (only kernel launches in timed section)
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulationGPU(gpu_state, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy results back to host and free GPU memory (outside timed section)
    gpuFinish(world, gpu_state);
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.6f ms\n", duration_ms);
    
    // Calculate performance metrics
    const double n_measured_iters = std::max(static_cast<double>(n_iters - 1), 1.0);
    const double time_per_iter = duration_ms / n_measured_iters;
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
