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

// GPU device pointers
struct DeviceWorld {
    Material* d_materials = nullptr;
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;

    cudaStream_t stream = 0;
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
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

// CUDA kernel: one thread per element, computes one simulation step
// Uses __restrict__ for alias analysis, grid-stride for scalability
__global__ void simulationKernel(
    const ElementStatic* __restrict__ elements_static,
    const Material* __restrict__ materials,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const size_t n_elems)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n_elems) return;

    const ElementStatic& elem_static = elements_static[i];
    const Material& mat = materials[elem_static.material_idx];
    const ElementDynamic& elem_dyn = elements_dynamic[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        total_flux += (neighbor_dyn.current_energy - elem_dyn.current_energy) *
                      mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
    }

    // Update element state
    ElementDynamic& elem_write = elements_dynamic_swap[i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Free GPU resources
void freeDeviceWorld(DeviceWorld& dev) {
    if (dev.d_materials)         cudaFree(dev.d_materials);
    if (dev.d_elements_static)   cudaFree(dev.d_elements_static);
    if (dev.d_elements_dynamic)  cudaFree(dev.d_elements_dynamic);
    if (dev.d_elements_dynamic_swap) cudaFree(dev.d_elements_dynamic_swap);
    if (dev.start_evt)           cudaEventDestroy(dev.start_evt);
    if (dev.stop_evt)            cudaEventDestroy(dev.stop_evt);
    if (dev.stream)              cudaStreamDestroy(dev.stream);
}

// Run simulation for n_iters iterations on GPU
// Returns GPU kernel execution time in milliseconds
double runSimulation(World& world, DeviceWorld& dev, const int n_iters) {
    const size_t n_elems = world.elements_static.size();

    // Create CUDA stream and events for accurate timing
    cudaStreamCreate(&dev.stream);
    cudaEventCreate(&dev.start_evt);
    cudaEventCreate(&dev.stop_evt);

    // Prefer L1 cache for random neighbor access patterns
    cudaFuncSetCacheConfig(simulationKernel, cudaFuncCachePreferL1);

    // Allocate and upload materials (small: 3 entries)
    cudaMalloc(&dev.d_materials, world.materials.size() * sizeof(Material));
    cudaMemcpy(dev.d_materials, world.materials.data(),
               world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    // Allocate and upload static connectivity (read-only during simulation)
    cudaMalloc(&dev.d_elements_static, n_elems * sizeof(ElementStatic));
    cudaMemcpy(dev.d_elements_static, world.elements_static.data(),
               n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice);

    // Allocate and upload dynamic state (double-buffered)
    cudaMalloc(&dev.d_elements_dynamic, n_elems * sizeof(ElementDynamic));
    cudaMemcpy(dev.d_elements_dynamic, world.elements_dynamic.data(),
               n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice);

    cudaMalloc(&dev.d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic));
    cudaMemset(dev.d_elements_dynamic_swap, 0, n_elems * sizeof(ElementDynamic));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = static_cast<int>((n_elems + threadsPerBlock - 1) / threadsPerBlock);

    // Record start event
    cudaEventRecord(dev.start_evt, dev.stream);

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernel<<<blocksPerGrid, threadsPerBlock, 0, dev.stream>>>(
            dev.d_elements_static,
            dev.d_materials,
            dev.d_elements_dynamic,
            dev.d_elements_dynamic_swap,
            n_elems
        );

        // Swap device pointers for next iteration (zero-cost pointer swap)
        std::swap(dev.d_elements_dynamic, dev.d_elements_dynamic_swap);
    }

    // Record stop event and synchronize
    cudaEventRecord(dev.stop_evt, dev.stream);
    cudaEventSynchronize(dev.stop_evt);

    // Calculate GPU kernel execution time
    float ms = 0;
    cudaEventElapsedTime(&ms, dev.start_evt, dev.stop_evt);

    // Download final results (asynchronous, after timing)
    cudaMemcpyAsync(world.elements_dynamic.data(), dev.d_elements_dynamic,
                    n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost, dev.stream);
    cudaStreamSynchronize(dev.stream);

    return static_cast<double>(ms);
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

    // Query GPU info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable device found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("GPU: %s (SM %d.%d, %zu MB global memory)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
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

    // Run simulation on GPU
    printf("Running simulation on GPU...\n");
    DeviceWorld dev;
    double duration_ms = runSimulation(world, dev, n_iters);

    printf("Computation time: %.2f ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = static_cast<double>(n_measured_iters) * n_elems / (duration_ms / 1000.0) / 1e9;

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
            freeDeviceWorld(dev);
            return 1;
        }
    }

    // Cleanup GPU memory
    freeDeviceWorld(dev);

    return 0;
}
