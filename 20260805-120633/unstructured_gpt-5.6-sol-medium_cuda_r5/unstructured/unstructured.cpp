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

// Fail immediately on CUDA errors.  This benchmark has no serial fallback: a
// usable CUDA device is part of its execution contract.
inline void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

template <typename T>
T* deviceAllocate(size_t count) {
    T* pointer = nullptr;
    if (count != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                              count * sizeof(T)));
    }
    return pointer;
}

// Structure-of-arrays storage makes every memory transaction in a warp
// contiguous.  Connectivity is transposed from [element][connection] on the
// CPU to [connection][element] on the GPU for the same reason.
struct DeviceWorld {
    size_t n_elems;
    Material* materials = nullptr;
    idx_t* material_idx = nullptr;
    idx_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};
    ElementDynamic* final_state = nullptr;

    explicit DeviceWorld(const World& world) : n_elems(world.elements_static.size()) {
        materials = deviceAllocate<Material>(world.materials.size());
        material_idx = deviceAllocate<idx_t>(n_elems);
        num_connections = deviceAllocate<idx_t>(n_elems);
        connected_idx = deviceAllocate<idx_t>(n_elems * MAX_CONNECTIONS);
        connected_flux = deviceAllocate<val_t>(n_elems * MAX_CONNECTIONS);
        energy[0] = deviceAllocate<val_t>(n_elems);
        energy[1] = deviceAllocate<val_t>(n_elems);
        total_flux[0] = deviceAllocate<val_t>(n_elems);
        total_flux[1] = deviceAllocate<val_t>(n_elems);
        final_state = deviceAllocate<ElementDynamic>(n_elems);

        std::vector<idx_t> host_material_idx(n_elems);
        std::vector<idx_t> host_num_connections(n_elems);
        std::vector<idx_t> host_connected_idx(n_elems * MAX_CONNECTIONS, 0);
        std::vector<val_t> host_connected_flux(n_elems * MAX_CONNECTIONS, 0.0);
        std::vector<val_t> host_energy(n_elems);
        std::vector<val_t> host_total_flux(n_elems);

        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& src_static = world.elements_static[i];
            host_material_idx[i] = src_static.material_idx;
            host_num_connections[i] = src_static.num_connections;
            host_energy[i] = world.elements_dynamic[i].current_energy;
            host_total_flux[i] = world.elements_dynamic[i].total_flux;
            for (idx_t j = 0; j < src_static.num_connections; ++j) {
                const size_t slot = static_cast<size_t>(j) * n_elems + i;
                host_connected_idx[slot] = src_static.connected_idx[j];
                host_connected_flux[slot] = src_static.connected_flux[j];
            }
        }

        CUDA_CHECK(cudaMemcpy(materials, world.materials.data(),
                              world.materials.size() * sizeof(Material),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(material_idx, host_material_idx.data(),
                              n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(num_connections, host_num_connections.data(),
                              n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_idx, host_connected_idx.data(),
                              n_elems * MAX_CONNECTIONS * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_flux, host_connected_flux.data(),
                              n_elems * MAX_CONNECTIONS * sizeof(val_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(energy[0], host_energy.data(),
                              n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(total_flux[0], host_total_flux.data(),
                              n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;

    ~DeviceWorld() {
        // cudaFree is synchronizing, and errors cannot usefully be recovered
        // from in a destructor.
        cudaFree(materials);
        cudaFree(material_idx);
        cudaFree(num_connections);
        cudaFree(connected_idx);
        cudaFree(connected_flux);
        cudaFree(energy[0]);
        cudaFree(energy[1]);
        cudaFree(total_flux[0]);
        cudaFree(total_flux[1]);
        cudaFree(final_state);
    }
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
__global__ void updateElementsKernel(
    size_t n_elems, const Material* __restrict__ materials,
    const idx_t* __restrict__ material_idx,
    const idx_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ accumulated_flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ accumulated_flux_out) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += stride) {
        const Material mat = materials[material_idx[i]];
        const val_t current_energy = energy_in[i];
        val_t iteration_flux = mat.external_flow;
        const idx_t count = num_connections[i];

#pragma unroll
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (static_cast<idx_t>(j) >= count) {
                break;
            }
            const size_t slot = static_cast<size_t>(j) * n_elems + i;
            const val_t neighbor_energy = energy_in[connected_idx[slot]];
            iteration_flux += (neighbor_energy - current_energy) *
                              mat.transfer_coeff * connected_flux[slot] * 0.25;
        }

        energy_out[i] = current_energy + iteration_flux;
        accumulated_flux_out[i] =
            accumulated_flux_in[i] + fabs(iteration_flux);
    }
}

__global__ void packResultsKernel(size_t n_elems,
                                  const val_t* __restrict__ energy,
                                  const val_t* __restrict__ total_flux,
                                  ElementDynamic* __restrict__ output) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += stride) {
        output[i] = ElementDynamic{energy[i], total_flux[i]};
    }
}

// Run the Jacobi iterations exclusively on the GPU, then copy only the final
// state back for the existing validation and result-reporting paths.
void runSimulation(World& world, DeviceWorld& device, const int n_iters) {
    constexpr int threads_per_block = 256;
    int read_buffer = 0;

    if (device.n_elems != 0) {
        cudaDeviceProp properties{};
        int active_device = 0;
        CUDA_CHECK(cudaGetDevice(&active_device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, active_device));

        // A capped grid retains enough resident work to saturate the GPU while
        // grid-stride iteration keeps the launch scalable to arbitrarily large
        // meshes and avoids excessive block scheduling overhead.
        const size_t blocks_needed =
            (device.n_elems + threads_per_block - 1) / threads_per_block;
        const size_t saturation_blocks =
            static_cast<size_t>(properties.multiProcessorCount) * 32;
        const int blocks = static_cast<int>(std::min(blocks_needed,
                                                      saturation_blocks));

        for (int iter = 0; iter < n_iters; ++iter) {
            const int write_buffer = read_buffer ^ 1;
            updateElementsKernel<<<blocks, threads_per_block>>>(
                device.n_elems, device.materials, device.material_idx,
                device.num_connections, device.connected_idx,
                device.connected_flux, device.energy[read_buffer],
                device.total_flux[read_buffer], device.energy[write_buffer],
                device.total_flux[write_buffer]);
            CUDA_CHECK(cudaGetLastError());
            read_buffer = write_buffer;
        }

        packResultsKernel<<<blocks, threads_per_block>>>(
            device.n_elems, device.energy[read_buffer],
            device.total_flux[read_buffer], device.final_state);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), device.final_state,
                              device.n_elems * sizeof(ElementDynamic),
                              cudaMemcpyDeviceToHost));
    }
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

    if (n_elems_root <= 0 || n_elems_root > 46340) {
        std::fprintf(stderr, "Grid size must be in the range [1, 46340]\n");
        return 1;
    }
    if (n_iters < 0) {
        std::fprintf(stderr, "Iteration count must be non-negative\n");
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

    // Allocate and upload before timing so the measured region represents the
    // simulation rather than one-time mesh layout conversion and CUDA setup.
    DeviceWorld device_world(world);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, device_world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    
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
