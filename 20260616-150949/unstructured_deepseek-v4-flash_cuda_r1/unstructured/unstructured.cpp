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

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel: one thread per element, reads from elements_dynamic, writes to elements_dynamic_swap
__global__ void simulation_kernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    const size_t n_elems)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const ElementStatic& es = elements_static[i];
    const val_t cur_energy = elements_dynamic[i].current_energy;
    const val_t cur_total_flux = elements_dynamic[i].total_flux;
    const Material& mat = materials[es.material_idx];
    const val_t tc = mat.transfer_coeff;

    val_t total_flux = mat.external_flow;

    for (idx_t j = 0; j < es.num_connections; ++j) {
        const val_t nbr_energy = __ldg(&elements_dynamic[es.connected_idx[j]].current_energy);
        total_flux += (nbr_energy - cur_energy) * tc * es.connected_flux[j] * 0.25;
    }

    elements_dynamic_swap[i].current_energy = cur_energy + total_flux;
    elements_dynamic_swap[i].total_flux = cur_total_flux + fabs(total_flux);
}

// GPU state for simulation
struct GPUState {
    Material* materials;
    ElementStatic* elements_static;
    ElementDynamic *dyn_a, *dyn_b;
    size_t n_elems;
};

// Initialize GPU memory for simulation
GPUState initGPU(World& world) {
    GPUState gpu;
    gpu.n_elems = world.elements_static.size();
    const size_t static_bytes = gpu.n_elems * sizeof(ElementStatic);
    const size_t dyn_bytes = gpu.n_elems * sizeof(ElementDynamic);
    const size_t mat_bytes = world.materials.size() * sizeof(Material);

    cudaMalloc(&gpu.materials, mat_bytes);
    cudaMalloc(&gpu.elements_static, static_bytes);
    cudaMalloc(&gpu.dyn_a, dyn_bytes);
    cudaMalloc(&gpu.dyn_b, dyn_bytes);

    cudaMemcpy(gpu.materials, world.materials.data(), mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.elements_static, world.elements_static.data(), static_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.dyn_a, world.elements_dynamic.data(), dyn_bytes, cudaMemcpyHostToDevice);

    return gpu;
}

// Run simulation on GPU for n_iters iterations (timed portion)
void runSimulationGPU(const GPUState& gpu, const int n_iters) {
    constexpr int block_size = 512;
    const int grid_size = static_cast<int>((gpu.n_elems + block_size - 1) / block_size);

    ElementDynamic* gpu_read = gpu.dyn_a;
    ElementDynamic* gpu_write = gpu.dyn_b;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulation_kernel<<<grid_size, block_size>>>(
            gpu.elements_static, gpu_read, gpu_write, gpu.materials, gpu.n_elems);

        ElementDynamic* tmp = gpu_read;
        gpu_read = gpu_write;
        gpu_write = tmp;
    }
}

// Copy results back from GPU and free memory
void cleanupGPU(World& world, const GPUState& gpu) {
    const size_t dyn_bytes = gpu.n_elems * sizeof(ElementDynamic);
    cudaMemcpy(world.elements_dynamic.data(), gpu.dyn_a, dyn_bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu.materials);
    cudaFree(gpu.elements_static);
    cudaFree(gpu.dyn_a);
    cudaFree(gpu.dyn_b);

    world.elements_dynamic_swap.resize(gpu.n_elems);
}

// Run simulation for n_iters iterations using CUDA (legacy entry point)
void runSimulation(World& world, const int n_iters) {
    GPUState gpu = initGPU(world);
    runSimulationGPU(gpu, n_iters);
    cleanupGPU(world, gpu);
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
    
    // Initialize GPU (outside timed section)
    printf("Setting up GPU...\n");
    cudaFree(0);  // Warm up CUDA context
    GPUState gpu = initGPU(world);
    
    // Run simulation (timed)
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start);
    
    runSimulationGPU(gpu, n_iters);
    
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get precise kernel timing
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    // Copy results back and cleanup
    cleanupGPU(world, gpu);
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Computation time: %ld ms (kernel: %.3f ms)\n", duration_ms, kernel_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double kernel_iter_ms = kernel_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (kernel_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Kernel time per iteration: %.4f ms\n", kernel_iter_ms);
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
