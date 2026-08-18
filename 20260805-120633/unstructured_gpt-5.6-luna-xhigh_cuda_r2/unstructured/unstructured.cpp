#include <algorithm>
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

constexpr int CUDA_BLOCK_SIZE = 256;

// Materials are read by every thread and there are only three of them. Keeping
// them in constant memory gives all threads in a warp a cached, broadcast load.
__constant__ Material d_materials[3];

void cudaCheck(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d while calling %s: %s\n",
                file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)

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

// The host representation is convenient for mesh construction, but its AoS
// connectivity layout makes adjacent CUDA threads stride through large
// records. This device representation keeps each connectivity slot in its own
// contiguous array so reads by a warp are coalesced.
struct DeviceBuffers {
    uint8_t* material_indices = nullptr;
    uint8_t* connection_counts = nullptr;
    idx_t* connected_indices = nullptr;
    val_t* connected_fluxes = nullptr;
    ElementDynamic* dynamic_a = nullptr;
    ElementDynamic* dynamic_b = nullptr;

    void release() {
        if (material_indices != nullptr) {
            CUDA_CHECK(cudaFree(material_indices));
        }
        if (connection_counts != nullptr) {
            CUDA_CHECK(cudaFree(connection_counts));
        }
        if (connected_indices != nullptr) {
            CUDA_CHECK(cudaFree(connected_indices));
        }
        if (connected_fluxes != nullptr) {
            CUDA_CHECK(cudaFree(connected_fluxes));
        }
        if (dynamic_a != nullptr) {
            CUDA_CHECK(cudaFree(dynamic_a));
        }
        if (dynamic_b != nullptr) {
            CUDA_CHECK(cudaFree(dynamic_b));
        }
    }
};

__global__ void updateElementsKernel(
        const uint8_t* __restrict__ material_indices,
        const uint8_t* __restrict__ connection_counts,
        const idx_t* __restrict__ connected_indices,
        const val_t* __restrict__ connected_fluxes,
        const ElementDynamic* __restrict__ dynamic_in,
        ElementDynamic* __restrict__ dynamic_out,
        const size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const Material mat = d_materials[material_indices[i]];
    const ElementDynamic this_elem = dynamic_in[i];
    val_t total_flux = mat.external_flow;

    // The maximum is fixed at compile time, while the branch preserves the
    // original variable-length connection list for boundary elements.
    const uint8_t num_connections = connection_counts[i];
#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < num_connections) {
            const size_t connection_offset = static_cast<size_t>(j) * n_elems + i;
            const idx_t neighbor_idx = connected_indices[connection_offset];
            const val_t neighbor_energy = dynamic_in[neighbor_idx].current_energy;

            // Keep the operation order of the scalar implementation. In
            // particular, this is intentionally not fast-math arithmetic.
            total_flux += (neighbor_energy - this_elem.current_energy) *
                          mat.transfer_coeff * connected_fluxes[connection_offset] *
                          0.25;
        }
    }

    dynamic_out[i].current_energy = this_elem.current_energy + total_flux;
    dynamic_out[i].total_flux = this_elem.total_flux + fabs(total_flux);
}

// Run simulation for n_iters iterations on the GPU. The returned time covers
// only the repeated GPU update kernels, excluding one-time transfers and the
// final result copy.
double runSimulation(World& world, const int n_iters) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        fprintf(stderr, "CUDA device required for this benchmark: %s\n",
                cudaGetErrorString(device_status));
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));

    const size_t n_elems = world.elements_static.size();
    if (world.materials.size() > 3) {
        fprintf(stderr, "Too many materials for the CUDA constant-memory table\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint8_t> material_indices(n_elems);
    std::vector<uint8_t> connection_counts(n_elems);
    std::vector<idx_t> connected_indices(n_elems * MAX_CONNECTIONS, 0);
    std::vector<val_t> connected_fluxes(n_elems * MAX_CONNECTIONS, 0.0);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        material_indices[i] = static_cast<uint8_t>(elem.material_idx);
        connection_counts[i] = static_cast<uint8_t>(elem.num_connections);

        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const size_t connection_offset = static_cast<size_t>(j) * n_elems + i;
            connected_indices[connection_offset] = elem.connected_idx[j];
            connected_fluxes[connection_offset] = elem.connected_flux[j];
        }
    }

    DeviceBuffers device;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.material_indices),
                          n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.connection_counts),
                          n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.connected_indices),
                          n_elems * MAX_CONNECTIONS * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.connected_fluxes),
                          n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.dynamic_a),
                          n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.dynamic_b),
                          n_elems * sizeof(ElementDynamic)));

    CUDA_CHECK(cudaMemcpy(device.material_indices, material_indices.data(),
                          n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.connection_counts, connection_counts.data(),
                          n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.connected_indices, connected_indices.data(),
                          n_elems * MAX_CONNECTIONS * sizeof(idx_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.connected_fluxes, connected_fluxes.data(),
                          n_elems * MAX_CONNECTIONS * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.dynamic_a, world.elements_dynamic.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.dynamic_b, world.elements_dynamic_swap.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_materials, world.materials.data(),
                                  world.materials.size() * sizeof(Material)));

    CUDA_CHECK(cudaFuncSetCacheConfig(updateElementsKernel, cudaFuncCachePreferL1));

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    ElementDynamic* dynamic_in = device.dynamic_a;
    ElementDynamic* dynamic_out = device.dynamic_b;
    const unsigned int grid_size = static_cast<unsigned int>(
        (n_elems + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    for (int iter = 0; iter < n_iters; ++iter) {
        updateElementsKernel<<<grid_size, CUDA_BLOCK_SIZE>>>(
            device.material_indices, device.connection_counts,
            device.connected_indices, device.connected_fluxes,
            dynamic_in, dynamic_out, n_elems);
        std::swap(dynamic_in, dynamic_out);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), dynamic_in,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic_swap.data(), dynamic_out,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    device.release();
    return static_cast<double>(elapsed_ms);
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
    const double duration_ms = runSimulation(world, n_iters);

    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) /
                                     (duration_ms / 1000.0) / 1e9;
    
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
