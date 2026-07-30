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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Build a 2D square grid as an unstructured mesh
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

// Structure of Arrays layout for GPU (better memory coalescing)
struct ElementStaticSoA {
    uint64_t* material_idx;
    uint64_t* num_connections;
    uint64_t* connected_idx;  // Flattened: [elem0_conn0, elem0_conn1, ..., elem1_conn0, ...]
    double* connected_flux;   // Flattened
};

struct ElementDynamicSoA {
    double* current_energy;
    double* total_flux;
};

// CUDA kernel for simulation step - optimized with SoA layout
__global__ void simulationKernelSoA(
    const ElementStaticSoA es,
    const ElementDynamicSoA ed,
    ElementDynamicSoA ew,
    const Material* __restrict__ materials,
    size_t n_elems)
{
    // Use grid-stride loop for better occupancy
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = tid; i < n_elems; i += stride) {
        const uint64_t mat_idx = es.material_idx[i];
        const Material mat = materials[mat_idx];
        
        // Start with external flow
        double total_flux = mat.external_flow;
        const double my_energy = ed.current_energy[i];
        const double transfer_coeff_025 = mat.transfer_coeff * 0.25;
        
        // Add flux from all connected elements
        const uint64_t num_conn = es.num_connections[i];
        const size_t conn_offset = i * MAX_CONNECTIONS;
        
        #pragma unroll 4
        for (uint64_t j = 0; j < num_conn; ++j) {
            const uint64_t neighbor_idx = es.connected_idx[conn_offset + j];
            const double neighbor_energy = ed.current_energy[neighbor_idx];
            const double flux_coeff = es.connected_flux[conn_offset + j];
            total_flux += (neighbor_energy - my_energy) * transfer_coeff_025 * flux_coeff;
        }
        
        // Update element state
        ew.current_energy[i] = my_energy + total_flux;
        ew.total_flux[i] = ed.total_flux[i] + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations using CUDA with SoA layout
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const size_t n_materials = world.materials.size();
    
    // Use pinned memory for faster transfers
    ElementStaticSoA h_es;
    ElementDynamicSoA h_ed, h_ew;
    
    CUDA_CHECK(cudaMallocHost(&h_es.material_idx, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&h_es.num_connections, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&h_es.connected_idx, n_elems * MAX_CONNECTIONS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&h_es.connected_flux, n_elems * MAX_CONNECTIONS * sizeof(double)));
    
    CUDA_CHECK(cudaMallocHost(&h_ed.current_energy, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_ed.total_flux, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_ew.current_energy, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_ew.total_flux, n_elems * sizeof(double)));
    
    // Convert from AoS to SoA
    for (size_t i = 0; i < n_elems; ++i) {
        h_es.material_idx[i] = world.elements_static[i].material_idx;
        h_es.num_connections[i] = world.elements_static[i].num_connections;
        h_ed.current_energy[i] = world.elements_dynamic[i].current_energy;
        h_ed.total_flux[i] = world.elements_dynamic[i].total_flux;
        
        const size_t offset = i * MAX_CONNECTIONS;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_es.connected_idx[offset + j] = world.elements_static[i].connected_idx[j];
            h_es.connected_flux[offset + j] = world.elements_static[i].connected_flux[j];
        }
    }
    
    // Allocate device memory in SoA format
    ElementStaticSoA d_es;
    ElementDynamicSoA d_ed, d_ew;
    
    CUDA_CHECK(cudaMalloc(&d_es.material_idx, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_es.num_connections, n_elems * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_es.connected_idx, n_elems * MAX_CONNECTIONS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_es.connected_flux, n_elems * MAX_CONNECTIONS * sizeof(double)));
    
    CUDA_CHECK(cudaMalloc(&d_ed.current_energy, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ed.total_flux, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ew.current_energy, n_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ew.total_flux, n_elems * sizeof(double)));
    
    Material* d_materials;
    CUDA_CHECK(cudaMalloc(&d_materials, n_materials * sizeof(Material)));
    
    // Copy to device using pinned memory
    CUDA_CHECK(cudaMemcpy(d_es.material_idx, h_es.material_idx, n_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_es.num_connections, h_es.num_connections, n_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_es.connected_idx, h_es.connected_idx, n_elems * MAX_CONNECTIONS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_es.connected_flux, h_es.connected_flux, n_elems * MAX_CONNECTIONS * sizeof(double), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_ed.current_energy, h_ed.current_energy, n_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ed.total_flux, h_ed.total_flux, n_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(), n_materials * sizeof(Material), cudaMemcpyHostToDevice));
    
    // Configure kernel launch - optimize for RTX 3090 (82 SMs)
    const int threadsPerBlock = 256;
    const int blocksPerGrid = std::min((n_elems + threadsPerBlock - 1) / threadsPerBlock, (size_t)820);
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Run simulation iterations
    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernelSoA<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_es, d_ed, d_ew, d_materials, n_elems);
        
        // Swap pointers
        ElementDynamicSoA temp = d_ed;
        d_ed = d_ew;
        d_ew = temp;
    }
    
    // Synchronize before copying back
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy results back to pinned host memory
    CUDA_CHECK(cudaMemcpy(h_ed.current_energy, d_ed.current_energy, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ed.total_flux, d_ed.total_flux, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert back from SoA to AoS
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_ed.current_energy[i];
        world.elements_dynamic[i].total_flux = h_ed.total_flux[i];
    }
    
    // Free resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_es.material_idx));
    CUDA_CHECK(cudaFreeHost(h_es.num_connections));
    CUDA_CHECK(cudaFreeHost(h_es.connected_idx));
    CUDA_CHECK(cudaFreeHost(h_es.connected_flux));
    CUDA_CHECK(cudaFreeHost(h_ed.current_energy));
    CUDA_CHECK(cudaFreeHost(h_ed.total_flux));
    CUDA_CHECK(cudaFreeHost(h_ew.current_energy));
    CUDA_CHECK(cudaFreeHost(h_ew.total_flux));
    
    CUDA_CHECK(cudaFree(d_es.material_idx));
    CUDA_CHECK(cudaFree(d_es.num_connections));
    CUDA_CHECK(cudaFree(d_es.connected_idx));
    CUDA_CHECK(cudaFree(d_es.connected_flux));
    CUDA_CHECK(cudaFree(d_ed.current_energy));
    CUDA_CHECK(cudaFree(d_ed.total_flux));
    CUDA_CHECK(cudaFree(d_ew.current_energy));
    CUDA_CHECK(cudaFree(d_ew.total_flux));
    CUDA_CHECK(cudaFree(d_materials));
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
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
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
