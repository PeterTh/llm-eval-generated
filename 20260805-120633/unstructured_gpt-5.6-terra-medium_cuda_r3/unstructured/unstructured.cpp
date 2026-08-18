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

[[noreturn]] void cudaFail(cudaError_t error, const char* operation,
                           const char* file, int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d while %s: %s\n", file, line,
                 operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t cuda_error_ = (operation);                            \
        if (cuda_error_ != cudaSuccess) {                                       \
            cudaFail(cuda_error_, #operation, __FILE__, __LINE__);              \
        }                                                                       \
    } while (false)

// One thread owns one element.  Reads are exclusively from the current state
// buffer and writes are exclusively to the next buffer, so every update can
// run independently without atomics or inter-block synchronization.
__global__ void updateElementsKernel(const idx_t* __restrict__ material_indices,
                                     const idx_t* __restrict__ connection_counts,
                                     const idx_t* __restrict__ connection_indices,
                                     const val_t* __restrict__ connection_fluxes,
                                     const Material* __restrict__ materials,
                                     const ElementDynamic* __restrict__ current,
                                     ElementDynamic* __restrict__ next,
                                     const size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const ElementDynamic elem_dyn = current[i];
    const Material mat = materials[material_indices[i]];
    val_t total_flux = mat.external_flow;

    // Preserve the original connection order, which preserves the numerical
    // update order for arbitrary unstructured meshes.
    for (idx_t j = 0; j < connection_counts[i]; ++j) {
        // Connection data is transposed on the device, making adjacent threads
        // read adjacent addresses for each neighbor slot.
        const size_t connection_offset = static_cast<size_t>(j) * n_elems + i;
        const ElementDynamic neighbor_dyn = current[connection_indices[connection_offset]];
        total_flux += (neighbor_dyn.current_energy - elem_dyn.current_energy) *
                      mat.transfer_coeff * connection_fluxes[connection_offset] * 0.25;
    }

    next[i].current_energy = elem_dyn.current_energy + total_flux;
    next[i].total_flux = elem_dyn.total_flux + fabs(total_flux);
}

struct GpuSimulation {
    size_t n_elems = 0;
    idx_t* material_indices = nullptr;
    idx_t* connection_counts = nullptr;
    idx_t* connection_indices = nullptr;
    val_t* connection_fluxes = nullptr;
    Material* device_materials = nullptr;
    ElementDynamic* device_current = nullptr;
    ElementDynamic* device_next = nullptr;
};

// Upload static mesh information once.  The device representation is
// structure-of-arrays, unlike the convenient host representation, so a warp
// loads each field with coalesced memory transactions.
GpuSimulation initializeGpuSimulation(const World& world) {
    GpuSimulation simulation;
    simulation.n_elems = world.elements_static.size();
    const size_t n_elems = simulation.n_elems;
    const size_t dynamic_bytes = n_elems * sizeof(ElementDynamic);
    const size_t material_bytes = world.materials.size() * sizeof(Material);
    const size_t index_bytes = n_elems * sizeof(idx_t);
    const size_t connection_bytes = n_elems * MAX_CONNECTIONS * sizeof(idx_t);
    const size_t flux_bytes = n_elems * MAX_CONNECTIONS * sizeof(val_t);

    if (n_elems == 0) {
        return simulation;
    }

    std::vector<idx_t> material_indices(n_elems);
    std::vector<idx_t> connection_counts(n_elems);
    std::vector<idx_t> connection_indices(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> connection_fluxes(n_elems * MAX_CONNECTIONS);
    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& element = world.elements_static[i];
        material_indices[i] = element.material_idx;
        connection_counts[i] = element.num_connections;
        for (idx_t j = 0; j < element.num_connections; ++j) {
            const size_t offset = static_cast<size_t>(j) * n_elems + i;
            connection_indices[offset] = element.connected_idx[j];
            connection_fluxes[offset] = element.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&simulation.material_indices, index_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.connection_counts, index_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.connection_indices, connection_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.connection_fluxes, flux_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.device_materials, material_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.device_current, dynamic_bytes));
    CUDA_CHECK(cudaMalloc(&simulation.device_next, dynamic_bytes));
    CUDA_CHECK(cudaMemcpy(simulation.material_indices, material_indices.data(), index_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.connection_counts, connection_counts.data(), index_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.connection_indices, connection_indices.data(), connection_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.connection_fluxes, connection_fluxes.data(), flux_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.device_materials, world.materials.data(), material_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.device_current, world.elements_dynamic.data(), dynamic_bytes,
                          cudaMemcpyHostToDevice));
    return simulation;
}

// Run simulation for n_iters iterations entirely on the GPU.
void runSimulation(GpuSimulation& simulation, const int n_iters) {
    if (simulation.n_elems == 0) {
        return;
    }
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((simulation.n_elems + threads_per_block - 1) /
                                        threads_per_block);
    for (int iter = 0; iter < n_iters; ++iter) {
        updateElementsKernel<<<blocks, threads_per_block>>>(
            simulation.material_indices, simulation.connection_counts,
            simulation.connection_indices, simulation.connection_fluxes,
            simulation.device_materials, simulation.device_current,
            simulation.device_next, simulation.n_elems);
        CUDA_CHECK(cudaGetLastError());
        std::swap(simulation.device_current, simulation.device_next);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void finalizeGpuSimulation(World& world, GpuSimulation& simulation) {
    const size_t dynamic_bytes = simulation.n_elems * sizeof(ElementDynamic);
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), simulation.device_current, dynamic_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(simulation.device_next));
    CUDA_CHECK(cudaFree(simulation.device_current));
    CUDA_CHECK(cudaFree(simulation.device_materials));
    CUDA_CHECK(cudaFree(simulation.connection_fluxes));
    CUDA_CHECK(cudaFree(simulation.connection_indices));
    CUDA_CHECK(cudaFree(simulation.connection_counts));
    CUDA_CHECK(cudaFree(simulation.material_indices));
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
    
    // Keep one-time CUDA initialization and mesh upload out of the timed
    // region, matching the original benchmark's compute-only timing.
    GpuSimulation simulation = initializeGpuSimulation(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(simulation, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // The final state is needed by the existing host-side hash, validation,
    // and external-results code, but is not part of timestep execution.
    finalizeGpuSimulation(world, simulation);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
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
