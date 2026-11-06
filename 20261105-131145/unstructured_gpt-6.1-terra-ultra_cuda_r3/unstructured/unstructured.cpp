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

// Report CUDA failures immediately rather than producing incomplete results.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceElement {
    val_t transfer_coeff;
    val_t external_flow;
    idx_t num_connections;
};

// Connectivity is stored connection-major, so a warp reads adjacent entries.
// Dynamic energy and accumulated flux are separate for efficient neighbor reads.
__global__ void updateElements(size_t n, const DeviceElement* __restrict__ elements,
                               const uint32_t* __restrict__ neighbors,
                               const val_t* __restrict__ connection_flux,
                               const val_t* __restrict__ energy,
                               const val_t* __restrict__ accumulated_flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_accumulated_flux) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const DeviceElement elem = elements[i];
    const val_t current = energy[i];
    val_t total_flux = elem.external_flow;
    for (idx_t j = 0; j < elem.num_connections; ++j) {
        const size_t edge = j * n + i;
        const val_t difference = energy[neighbors[edge]] - current;
        total_flux += difference * elem.transfer_coeff * connection_flux[edge] * 0.25;
    }
    next_energy[i] = current + total_flux;
    next_accumulated_flux[i] = accumulated_flux[i] + fabs(total_flux);
}

class DeviceSimulation {
public:
    explicit DeviceSimulation(const World& world) : n_(world.elements_static.size()) {
        std::vector<DeviceElement> elements(n_);
        size_t max_connections = 0;
        for (size_t i = 0; i < n_; ++i) {
            const auto& element = world.elements_static[i];
            const auto& material = world.materials[element.material_idx];
            elements[i] = {material.transfer_coeff, material.external_flow, element.num_connections};
            max_connections = std::max(max_connections, static_cast<size_t>(element.num_connections));
        }
        // The square-grid builder uses int indices, so they fit in 32 bits.
        std::vector<uint32_t> neighbors(n_ * max_connections);
        std::vector<val_t> connection_flux(n_ * max_connections);
        std::vector<val_t> energy(n_), accumulated_flux(n_);
        for (size_t i = 0; i < n_; ++i) {
            const auto& element = world.elements_static[i];
            for (size_t j = 0; j < element.num_connections; ++j) {
                neighbors[j * n_ + i] = static_cast<uint32_t>(element.connected_idx[j]);
                connection_flux[j * n_ + i] = element.connected_flux[j];
            }
            energy[i] = world.elements_dynamic[i].current_energy;
            accumulated_flux[i] = world.elements_dynamic[i].total_flux;
        }
        allocateAndCopy(elements_, elements);
        allocateAndCopy(neighbors_, neighbors);
        allocateAndCopy(connection_flux_, connection_flux);
        allocateAndCopy(energy_[0], energy);
        allocateAndCopy(accumulated_flux_[0], accumulated_flux);
        allocate(energy_[1], n_);
        allocate(accumulated_flux_[1], n_);
    }

    ~DeviceSimulation() {
        cudaFree(elements_);
        cudaFree(neighbors_);
        cudaFree(connection_flux_);
        for (int i = 0; i < 2; ++i) {
            cudaFree(energy_[i]);
            cudaFree(accumulated_flux_[i]);
        }
    }

    void run(int n_iters) {
        constexpr unsigned block_size = 256;
        const unsigned blocks = static_cast<unsigned>((n_ + block_size - 1) / block_size);
        for (int iter = 0; iter < n_iters; ++iter) {
            updateElements<<<blocks, block_size>>>(n_, elements_, neighbors_, connection_flux_,
                energy_[current_], accumulated_flux_[current_], energy_[1 - current_],
                accumulated_flux_[1 - current_]);
            current_ = 1 - current_;
        }
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "simulation");
    }

    void copyBack(World& world) const {
        std::vector<val_t> energy(n_), accumulated_flux(n_);
        checkCuda(cudaMemcpy(energy.data(), energy_[current_], n_ * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "copy energy to host");
        checkCuda(cudaMemcpy(accumulated_flux.data(), accumulated_flux_[current_],
                             n_ * sizeof(val_t), cudaMemcpyDeviceToHost), "copy flux to host");
        for (size_t i = 0; i < n_; ++i) {
            world.elements_dynamic[i] = {energy[i], accumulated_flux[i]};
        }
    }

private:
    template <typename T> static void allocate(T*& ptr, size_t count) {
        if (count != 0) checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "allocation");
    }
    template <typename T> static void allocateAndCopy(T*& ptr, const std::vector<T>& data) {
        allocate(ptr, data.size());
        if (!data.empty()) checkCuda(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T),
                                                cudaMemcpyHostToDevice), "copy to device");
    }
    size_t n_;
    int current_ = 0;
    DeviceElement* elements_ = nullptr;
    uint32_t* neighbors_ = nullptr;
    val_t* connection_flux_ = nullptr;
    val_t* energy_[2] = {};
    val_t* accumulated_flux_[2] = {};
};

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
    
    // Transfer mesh data before timing the GPU computation.
    DeviceSimulation simulation(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    simulation.copyBack(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(std::max(n_iters, 0)) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
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
