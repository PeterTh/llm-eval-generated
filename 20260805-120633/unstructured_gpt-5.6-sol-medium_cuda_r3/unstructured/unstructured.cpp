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

// Device data is stored as structure-of-arrays.  In particular, connection
// slot j for all elements is contiguous, which makes the irregular gather's
// metadata reads coalesced across a warp.
struct CudaState {
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    uint32_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    Material* materials = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};
    int final_buffer = 0;
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    CudaState cuda;
};


inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
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
__global__ void updateElementsKernel(
        size_t n_elems,
        const uint8_t* __restrict__ material_idx,
        const uint8_t* __restrict__ num_connections,
        const uint32_t* __restrict__ connected_idx,
        const val_t* __restrict__ connected_flux,
        const Material* __restrict__ materials,
        const val_t* __restrict__ energy_in,
        const val_t* __restrict__ accumulated_flux_in,
        val_t* __restrict__ energy_out,
        val_t* __restrict__ accumulated_flux_out) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = materials[material_idx[i]];
    const val_t this_energy = energy_in[i];
    const int connection_count = num_connections[i];
    val_t flux = mat.external_flow;

#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j >= connection_count) break;
        const size_t connection = static_cast<size_t>(j) * n_elems + i;
        const val_t neighbor_energy = energy_in[connected_idx[connection]];
        flux += (neighbor_energy - this_energy) * mat.transfer_coeff *
                connected_flux[connection] * 0.25;
    }

    energy_out[i] = this_energy + flux;
    accumulated_flux_out[i] = accumulated_flux_in[i] + fabs(flux);
}

// Allocate and upload immutable mesh data before the timed region.  The
// compact device representation cuts metadata bandwidth substantially versus
// copying the padded host structs directly.
void prepareCuda(World& world) {
    const size_t n_elems = world.elements_static.size();
    const size_t connections = n_elems * MAX_CONNECTIONS;
    std::vector<uint8_t> material_idx(n_elems);
    std::vector<uint8_t> num_connections(n_elems);
    std::vector<uint32_t> connected_idx(connections);
    std::vector<val_t> connected_flux(connections);
    std::vector<val_t> energy(n_elems);
    std::vector<val_t> total_flux(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        material_idx[i] = static_cast<uint8_t>(elem.material_idx);
        num_connections[i] = static_cast<uint8_t>(elem.num_connections);
        energy[i] = world.elements_dynamic[i].current_energy;
        total_flux[i] = world.elements_dynamic[i].total_flux;
        for (size_t j = 0; j < elem.num_connections; ++j) {
            const size_t connection = j * n_elems + i;
            connected_idx[connection] = static_cast<uint32_t>(elem.connected_idx[j]);
            connected_flux[connection] = elem.connected_flux[j];
        }
    }

    CudaState& gpu = world.cuda;
    checkCuda(cudaMalloc(&gpu.material_idx, n_elems * sizeof(uint8_t)), "allocating material indices");
    checkCuda(cudaMalloc(&gpu.num_connections, n_elems * sizeof(uint8_t)), "allocating connection counts");
    checkCuda(cudaMalloc(&gpu.connected_idx, connections * sizeof(uint32_t)), "allocating connectivity");
    checkCuda(cudaMalloc(&gpu.connected_flux, connections * sizeof(val_t)), "allocating connection fluxes");
    checkCuda(cudaMalloc(&gpu.materials, world.materials.size() * sizeof(Material)), "allocating materials");
    for (int buffer = 0; buffer < 2; ++buffer) {
        checkCuda(cudaMalloc(&gpu.energy[buffer], n_elems * sizeof(val_t)), "allocating energy buffers");
        checkCuda(cudaMalloc(&gpu.total_flux[buffer], n_elems * sizeof(val_t)), "allocating flux buffers");
    }

    checkCuda(cudaMemcpy(gpu.material_idx, material_idx.data(), n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice), "uploading material indices");
    checkCuda(cudaMemcpy(gpu.num_connections, num_connections.data(), n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice), "uploading connection counts");
    checkCuda(cudaMemcpy(gpu.connected_idx, connected_idx.data(), connections * sizeof(uint32_t), cudaMemcpyHostToDevice), "uploading connectivity");
    checkCuda(cudaMemcpy(gpu.connected_flux, connected_flux.data(), connections * sizeof(val_t), cudaMemcpyHostToDevice), "uploading connection fluxes");
    checkCuda(cudaMemcpy(gpu.materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "uploading materials");
    checkCuda(cudaMemcpy(gpu.energy[0], energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "uploading energy");
    checkCuda(cudaMemcpy(gpu.total_flux[0], total_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "uploading accumulated flux");
}

// Run all element updates on CUDA.  Each launch is the global synchronization
// point required by the original Jacobi iteration semantics.
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((n_elems + threads - 1) / threads);
    CudaState& gpu = world.cuda;

    int source = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int destination = source ^ 1;
        updateElementsKernel<<<blocks, threads>>>(
            n_elems, gpu.material_idx, gpu.num_connections, gpu.connected_idx,
            gpu.connected_flux, gpu.materials, gpu.energy[source],
            gpu.total_flux[source], gpu.energy[destination],
            gpu.total_flux[destination]);
        source = destination;
    }
    gpu.final_buffer = source;
    checkCuda(cudaGetLastError(), "launching simulation kernel");
    checkCuda(cudaDeviceSynchronize(), "running simulation kernel");
}

void downloadCudaResults(World& world) {
    const size_t n_elems = world.elements_dynamic.size();
    std::vector<val_t> energy(n_elems);
    std::vector<val_t> total_flux(n_elems);
    const CudaState& gpu = world.cuda;
    checkCuda(cudaMemcpy(energy.data(), gpu.energy[gpu.final_buffer], n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "downloading energy");
    checkCuda(cudaMemcpy(total_flux.data(), gpu.total_flux[gpu.final_buffer], n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "downloading accumulated flux");
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i] = {energy[i], total_flux[i]};
    }
}

void releaseCuda(World& world) {
    CudaState& gpu = world.cuda;
    checkCuda(cudaFree(gpu.material_idx), "freeing material indices");
    checkCuda(cudaFree(gpu.num_connections), "freeing connection counts");
    checkCuda(cudaFree(gpu.connected_idx), "freeing connectivity");
    checkCuda(cudaFree(gpu.connected_flux), "freeing connection fluxes");
    checkCuda(cudaFree(gpu.materials), "freeing materials");
    for (int buffer = 0; buffer < 2; ++buffer) {
        checkCuda(cudaFree(gpu.energy[buffer]), "freeing energy buffers");
        checkCuda(cudaFree(gpu.total_flux[buffer]), "freeing flux buffers");
    }
    gpu = {};
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

    // One-time device allocation and transfer are setup costs, just like mesh
    // construction, and are intentionally outside the solver timing.
    prepareCuda(world);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);

    downloadCudaResults(world);
    releaseCuda(world);
    
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
