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

inline void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The connectivity is stored as structure-of-arrays on the device.  A warp
// therefore reads consecutive indices/fluxes for every connection slot.
// This keeps the original arbitrary connectivity and connection ordering.
struct DeviceWorld {
    idx_t* material_idx = nullptr;
    idx_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    Material* materials = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};

    ~DeviceWorld() {
        cudaFree(material_idx);
        cudaFree(num_connections);
        cudaFree(connected_idx);
        cudaFree(connected_flux);
        cudaFree(materials);
        cudaFree(energy[0]);
        cudaFree(energy[1]);
        cudaFree(total_flux[0]);
        cudaFree(total_flux[1]);
    }
};

__global__ void updateElementsKernel(const idx_t* __restrict__ material_idx,
                                     const idx_t* __restrict__ num_connections,
                                     const idx_t* __restrict__ connected_idx,
                                     const val_t* __restrict__ connected_flux,
                                     const Material* __restrict__ materials,
                                     const val_t* __restrict__ energy_read,
                                     const val_t* __restrict__ flux_read,
                                     val_t* __restrict__ energy_write,
                                     val_t* __restrict__ flux_write,
                                     size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = materials[material_idx[i]];
    const val_t this_energy = energy_read[i];
    val_t total = mat.external_flow;
    const idx_t connection_count = num_connections[i];

    // Do not unroll: MAX_CONNECTIONS is only an upper bound, while this
    // loop preserves the source's left-to-right floating-point accumulation.
    for (idx_t j = 0; j < connection_count; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_elems + i;
        const val_t other_energy = energy_read[connected_idx[slot]];
        total += (other_energy - this_energy) * mat.transfer_coeff *
                 connected_flux[slot] * 0.25;
    }
    energy_write[i] = this_energy + total;
    flux_write[i] = flux_read[i] + fabs(total);
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

// Run simulation for n_iters iterations entirely on the GPU.  The device
// layout is intentionally separate from World so host-side reporting and the
// externally-visible data model retain their original form.
double runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0 || n_iters <= 0) return 0.0;

    std::vector<idx_t> material_idx(n_elems);
    std::vector<idx_t> num_connections(n_elems);
    std::vector<idx_t> connected_idx(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> connected_flux(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> energy(n_elems);
    std::vector<val_t> total_flux(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& source = world.elements_static[i];
        material_idx[i] = source.material_idx;
        num_connections[i] = source.num_connections;
        energy[i] = world.elements_dynamic[i].current_energy;
        total_flux[i] = world.elements_dynamic[i].total_flux;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            const size_t slot = static_cast<size_t>(j) * n_elems + i;
            connected_idx[slot] = source.connected_idx[j];
            connected_flux[slot] = source.connected_flux[j];
        }
    }

    DeviceWorld device;
    const size_t scalar_bytes = n_elems * sizeof(val_t);
    const size_t index_bytes = n_elems * sizeof(idx_t);
    const size_t connection_bytes = n_elems * MAX_CONNECTIONS * sizeof(idx_t);
    const size_t connection_flux_bytes = n_elems * MAX_CONNECTIONS * sizeof(val_t);
    cudaCheck(cudaMalloc(&device.material_idx, index_bytes), "allocating material indices");
    cudaCheck(cudaMalloc(&device.num_connections, index_bytes), "allocating connection counts");
    cudaCheck(cudaMalloc(&device.connected_idx, connection_bytes), "allocating neighbor indices");
    cudaCheck(cudaMalloc(&device.connected_flux, connection_flux_bytes), "allocating connection fluxes");
    cudaCheck(cudaMalloc(&device.materials, world.materials.size() * sizeof(Material)), "allocating materials");
    for (int buffer = 0; buffer < 2; ++buffer) {
        cudaCheck(cudaMalloc(&device.energy[buffer], scalar_bytes), "allocating energy buffer");
        cudaCheck(cudaMalloc(&device.total_flux[buffer], scalar_bytes), "allocating flux buffer");
    }

    cudaCheck(cudaMemcpy(device.material_idx, material_idx.data(), index_bytes, cudaMemcpyHostToDevice), "uploading material indices");
    cudaCheck(cudaMemcpy(device.num_connections, num_connections.data(), index_bytes, cudaMemcpyHostToDevice), "uploading connection counts");
    cudaCheck(cudaMemcpy(device.connected_idx, connected_idx.data(), connection_bytes, cudaMemcpyHostToDevice), "uploading neighbor indices");
    cudaCheck(cudaMemcpy(device.connected_flux, connected_flux.data(), connection_flux_bytes, cudaMemcpyHostToDevice), "uploading connection fluxes");
    cudaCheck(cudaMemcpy(device.materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "uploading materials");
    cudaCheck(cudaMemcpy(device.energy[0], energy.data(), scalar_bytes, cudaMemcpyHostToDevice), "uploading energy");
    cudaCheck(cudaMemcpy(device.total_flux[0], total_flux.data(), scalar_bytes, cudaMemcpyHostToDevice), "uploading total flux");

    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks = static_cast<unsigned int>((n_elems + threads_per_block - 1) / threads_per_block);
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaCheck(cudaEventCreate(&start_event), "creating start timing event");
    cudaCheck(cudaEventCreate(&stop_event), "creating stop timing event");
    cudaCheck(cudaEventRecord(start_event), "recording start timing event");
    int read_buffer = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int write_buffer = read_buffer ^ 1;
        updateElementsKernel<<<blocks, threads_per_block>>>(
            device.material_idx, device.num_connections, device.connected_idx,
            device.connected_flux, device.materials, device.energy[read_buffer],
            device.total_flux[read_buffer], device.energy[write_buffer],
            device.total_flux[write_buffer], n_elems);
        cudaCheck(cudaGetLastError(), "launching update kernel");
        read_buffer = write_buffer;
    }
    cudaCheck(cudaEventRecord(stop_event), "recording stop timing event");
    cudaCheck(cudaEventSynchronize(stop_event), "executing simulation kernels");
    float elapsed_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "measuring simulation kernels");
    cudaCheck(cudaEventDestroy(start_event), "destroying start timing event");
    cudaCheck(cudaEventDestroy(stop_event), "destroying stop timing event");

    cudaCheck(cudaMemcpy(energy.data(), device.energy[read_buffer], scalar_bytes, cudaMemcpyDeviceToHost), "downloading energy");
    cudaCheck(cudaMemcpy(total_flux.data(), device.total_flux[read_buffer], scalar_bytes, cudaMemcpyDeviceToHost), "downloading total flux");
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = total_flux[i];
    }
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
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9
        : 0.0;
    
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
