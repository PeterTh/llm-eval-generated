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

// Check a CUDA runtime call and abort with a readable message on failure
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Device-resident copy of the world. Connectivity is stored slot-major
// (index = slot * n_elems + i) so consecutive threads read consecutive
// addresses. Dynamic state is double-buffered: one buffer is read while the
// other is written, then the roles swap.
struct DeviceWorld {
    size_t n_elems = 0;
    size_t n_slots = 0;
    Material* materials = nullptr;
    idx_t* material_idx = nullptr;
    idx_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux[2] = {nullptr, nullptr};
};

template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
T* toDevice(const std::vector<T>& host) {
    T* ptr = deviceAlloc<T>(host.size());
    CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

// Upload the mesh and its initial state to the GPU
DeviceWorld uploadWorld(const World& world) {
    DeviceWorld dw;
    dw.n_elems = world.elements_static.size();
    for (const auto& elem : world.elements_static) {
        dw.n_slots = std::max<size_t>(dw.n_slots, elem.num_connections);
    }
    dw.n_slots = std::max<size_t>(dw.n_slots, 1);

    const size_t n = dw.n_elems;
    std::vector<idx_t> material_idx(n);
    std::vector<idx_t> num_connections(n);
    std::vector<idx_t> connected_idx(dw.n_slots * n, 0);
    std::vector<val_t> connected_flux(dw.n_slots * n, 0.0);
    std::vector<val_t> energy(n);
    std::vector<val_t> flux(n);
    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        material_idx[i] = elem_static.material_idx;
        num_connections[i] = elem_static.num_connections;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            connected_idx[j * n + i] = elem_static.connected_idx[j];
            connected_flux[j * n + i] = elem_static.connected_flux[j];
        }
        energy[i] = world.elements_dynamic[i].current_energy;
        flux[i] = world.elements_dynamic[i].total_flux;
    }

    dw.materials = toDevice(world.materials);
    dw.material_idx = toDevice(material_idx);
    dw.num_connections = toDevice(num_connections);
    dw.connected_idx = toDevice(connected_idx);
    dw.connected_flux = toDevice(connected_flux);
    dw.energy[0] = toDevice(energy);
    dw.flux[0] = toDevice(flux);
    // The first step writes every element of the second buffer before reading it
    dw.energy[1] = deviceAlloc<val_t>(n);
    dw.flux[1] = deviceAlloc<val_t>(n);
    return dw;
}

// Compute energy flux between two elements (same operation order as the serial reference)
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// One simulation step: each thread updates one element from buffer 'cur' and
// writes buffer '1 - cur'. Reads and writes never alias, so elements are
// independent and need no synchronization.
__global__ void stepKernel(const DeviceWorld dw, const int cur) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= dw.n_elems) {
        return;
    }

    const val_t* energy_in = dw.energy[cur];
    const val_t* flux_in = dw.flux[cur];
    const Material mat = dw.materials[dw.material_idx[i]];
    const val_t this_energy = energy_in[i];

    // Start with external flow, then add flux from all connected elements
    val_t total_flux = mat.external_flow;
    const idx_t num_connections = dw.num_connections[i];
    for (idx_t j = 0; j < num_connections; ++j) {
        const size_t slot = j * dw.n_elems + i;
        total_flux += computeFlux(mat.transfer_coeff, this_energy,
                                  dw.connected_flux[slot], energy_in[dw.connected_idx[slot]]);
    }

    dw.energy[1 - cur][i] = this_energy + total_flux;
    dw.flux[1 - cur][i] = flux_in[i] + fabs(total_flux);
}

// Run simulation for n_iters iterations
// Run simulation for n_iters iterations on the GPU.
// Returns the index of the dynamic buffer that holds the final state.
int runSimulation(const DeviceWorld& dw, const int n_iters) {
    constexpr unsigned threads_per_block = 256;
    const unsigned n_blocks = static_cast<unsigned>((dw.n_elems + threads_per_block - 1) / threads_per_block);

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        stepKernel<<<n_blocks, threads_per_block>>>(dw, cur);
        CUDA_CHECK(cudaGetLastError());
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return cur;
}

// Copy the final dynamic state back into the host world
void downloadWorld(const DeviceWorld& dw, const int cur, World& world) {
    std::vector<val_t> energy(dw.n_elems);
    std::vector<val_t> flux(dw.n_elems);
    CUDA_CHECK(cudaMemcpy(energy.data(), dw.energy[cur], dw.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), dw.flux[cur], dw.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < dw.n_elems; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
    }
}

void freeWorld(DeviceWorld& dw) {
    CUDA_CHECK(cudaFree(dw.materials));
    CUDA_CHECK(cudaFree(dw.material_idx));
    CUDA_CHECK(cudaFree(dw.num_connections));
    CUDA_CHECK(cudaFree(dw.connected_idx));
    CUDA_CHECK(cudaFree(dw.connected_flux));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(dw.energy[b]));
        CUDA_CHECK(cudaFree(dw.flux[b]));
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
    
    // Upload the mesh to the GPU (setup, not timed, like building the mesh)
    DeviceWorld dworld = uploadWorld(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const int final_buf = runSimulation(dworld, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    downloadWorld(dworld, final_buf, world);
    freeWorld(dworld);

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
