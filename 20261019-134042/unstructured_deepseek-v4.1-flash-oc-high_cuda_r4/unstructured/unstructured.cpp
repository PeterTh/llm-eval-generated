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

// GPU-resident mirror of the world.  Connectivity is stored in CSR form so
// the kernel only touches the data it needs, and the dynamic state is stored
// as separate energy/flux arrays for fully coalesced accesses.
struct GpuWorld {
    int n_elems = 0;
    int n_materials = 0;

    int* d_offsets = nullptr;
    uint32_t* d_neighbors = nullptr;
    val_t* d_weights = nullptr;
    uint32_t* d_material_idx = nullptr;

    val_t* d_transfer_coeff = nullptr;
    val_t* d_external_flow = nullptr;

    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};

    int current = 0;
    bool unit_weights = true;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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

// Upload the static mesh and initial dynamic state to the GPU.  This is done
// before the timed simulation so the timed region contains only computation.
void initGpuWorld(const World& world, GpuWorld& gpu) {
    const int n = static_cast<int>(world.elements_static.size());
    gpu.n_elems = n;
    gpu.n_materials = static_cast<int>(world.materials.size());
    gpu.current = 0;

    std::vector<int> offsets(n + 1);
    std::vector<uint32_t> neighbors;
    std::vector<val_t> weights;
    std::vector<uint32_t> material_idx(n);
    std::vector<val_t> energy(n);
    std::vector<val_t> flux(n);

    offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        const ElementStatic& e = world.elements_static[i];
        material_idx[i] = static_cast<uint32_t>(e.material_idx);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            neighbors.push_back(static_cast<uint32_t>(e.connected_idx[j]));
            weights.push_back(e.connected_flux[j]);
        }
        offsets[i + 1] = static_cast<int>(neighbors.size());
        energy[i] = world.elements_dynamic[i].current_energy;
        flux[i] = world.elements_dynamic[i].total_flux;
    }

    gpu.unit_weights = true;
    for (val_t w : weights) {
        if (w != 1.0) { gpu.unit_weights = false; break; }
    }

    std::vector<val_t> transfer(gpu.n_materials);
    std::vector<val_t> external(gpu.n_materials);
    for (int m = 0; m < gpu.n_materials; ++m) {
        transfer[m] = world.materials[m].transfer_coeff;
        external[m] = world.materials[m].external_flow;
    }

    CUDA_CHECK(cudaMalloc(&gpu.d_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_material_idx, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_transfer_coeff, gpu.n_materials * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&gpu.d_external_flow, gpu.n_materials * sizeof(val_t)));
    if (!neighbors.empty()) {
        CUDA_CHECK(cudaMalloc(&gpu.d_neighbors, neighbors.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&gpu.d_weights, weights.size() * sizeof(val_t)));
    }
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&gpu.d_energy[b], n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&gpu.d_flux[b], n * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(gpu.d_offsets, offsets.data(), (n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_material_idx, material_idx.data(),
                          n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_transfer_coeff, transfer.data(),
                          gpu.n_materials * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_external_flow, external.data(),
                          gpu.n_materials * sizeof(val_t), cudaMemcpyHostToDevice));
    if (!neighbors.empty()) {
        CUDA_CHECK(cudaMemcpy(gpu.d_neighbors, neighbors.data(),
                              neighbors.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu.d_weights, weights.data(),
                              weights.size() * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(gpu.d_energy[0], energy.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_flux[0], flux.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
}

// Copy the final GPU state back into the host world.
void downloadGpuWorld(const GpuWorld& gpu, World& world) {
    const int n = gpu.n_elems;
    std::vector<val_t> energy(n);
    std::vector<val_t> flux(n);
    CUDA_CHECK(cudaMemcpy(energy.data(), gpu.d_energy[gpu.current],
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), gpu.d_flux[gpu.current],
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
    }
}

void destroyGpuWorld(GpuWorld& gpu) {
    cudaFree(gpu.d_offsets);
    cudaFree(gpu.d_neighbors);
    cudaFree(gpu.d_weights);
    cudaFree(gpu.d_material_idx);
    cudaFree(gpu.d_transfer_coeff);
    cudaFree(gpu.d_external_flow);
    for (int b = 0; b < 2; ++b) {
        cudaFree(gpu.d_energy[b]);
        cudaFree(gpu.d_flux[b]);
    }
    gpu = GpuWorld{};
}

// One simulation iteration.  Every element is independent: it reads the old
// energy of itself and its neighbours and writes only its own new state, so
// an arbitrary number of threads can process elements concurrently.
template <bool UnitWeights>
__global__ void energyStepKernel(
    const int* __restrict__ offsets,
    const uint32_t* __restrict__ neighbors,
    const val_t* __restrict__ weights,
    const uint32_t* __restrict__ material_idx,
    const val_t* __restrict__ transfer_coeff,
    const val_t* __restrict__ external_flow,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ flux_out,
    int n_elems) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const int begin = offsets[i];
        const int end = offsets[i + 1];
        const uint32_t mat = material_idx[i];
        const val_t this_energy = energy_in[i];
        const val_t coeff = transfer_coeff[mat];

        val_t total_flux = external_flow[mat];
#pragma unroll 4
        for (int j = begin; j < end; ++j) {
            const val_t other_energy = energy_in[neighbors[j]];
            if constexpr (UnitWeights) {
                total_flux += (other_energy - this_energy) * coeff * 0.25;
            } else {
                total_flux += (other_energy - this_energy) * coeff * weights[j] * 0.25;
            }
        }

        energy_out[i] = this_energy + total_flux;
        flux_out[i] = flux_in[i] + fabs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World&, GpuWorld& gpu, const int n_iters) {
    const int n_elems = gpu.n_elems;
    if (n_elems == 0) {
        return;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    constexpr int threads = 256;
    const int needed_blocks = (n_elems + threads - 1) / threads;
    const int max_blocks = sm_count * 8;
    const int blocks = std::min(needed_blocks, max_blocks);

    int current = gpu.current;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int next = current ^ 1;
        if (gpu.unit_weights) {
            energyStepKernel<true><<<blocks, threads>>>(
                gpu.d_offsets, gpu.d_neighbors, gpu.d_weights, gpu.d_material_idx,
                gpu.d_transfer_coeff, gpu.d_external_flow,
                gpu.d_energy[current], gpu.d_flux[current],
                gpu.d_energy[next], gpu.d_flux[next], n_elems);
        } else {
            energyStepKernel<false><<<blocks, threads>>>(
                gpu.d_offsets, gpu.d_neighbors, gpu.d_weights, gpu.d_material_idx,
                gpu.d_transfer_coeff, gpu.d_external_flow,
                gpu.d_energy[current], gpu.d_flux[current],
                gpu.d_energy[next], gpu.d_flux[next], n_elems);
        }
        current = next;
    }
    gpu.current = current;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    
    // Prepare GPU-resident mesh and state before timing the simulation.
    GpuWorld gpu;
    initGpuWorld(world, gpu);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, gpu, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    downloadGpuWorld(gpu, world);
    destroyGpuWorld(gpu);
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double elapsed_ms = std::max(static_cast<double>(duration_us) / 1000.0, 1e-9);
    const double time_per_iter = elapsed_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (elapsed_ms / 1000.0) / 1e9;
    
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
