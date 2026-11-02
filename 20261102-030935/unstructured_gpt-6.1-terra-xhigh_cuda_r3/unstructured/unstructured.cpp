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

// Store each connection slot contiguously so adjacent GPU threads read adjacent
// entries, while retaining the original per-element connection order.
struct DeviceConnection {
    idx_t neighbor;
    val_t flux;
};

struct DeviceElement {
    val_t transfer_coeff;
    val_t external_flow;
    idx_t num_connections;
};

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

__global__ void updateElements(const DeviceElement* __restrict__ elements,
                               const DeviceConnection* __restrict__ connections,
                               const val_t* __restrict__ energy_in,
                               val_t* __restrict__ energy_out,
                               val_t* __restrict__ accumulated_flux,
                               size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const DeviceElement elem = elements[i];
    const val_t energy = energy_in[i];
    val_t total_flux = elem.external_flow;
    for (idx_t j = 0; j < elem.num_connections; ++j) {
        const DeviceConnection connection = connections[j * n_elems + i];
        total_flux += (energy_in[connection.neighbor] - energy) *
                      elem.transfer_coeff * connection.flux * 0.25;
    }
    energy_out[i] = energy + total_flux;
    accumulated_flux[i] += fabs(total_flux);
}

// Device state is prepared before the timed region, just as the host mesh is.
class DeviceWorld {
public:
    explicit DeviceWorld(const World& world) : n_elems(world.elements_static.size()) {
        std::vector<DeviceElement> elements(n_elems);
        size_t max_connections = 0;
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            const Material& mat = world.materials[elem.material_idx];
            elements[i] = {mat.transfer_coeff, mat.external_flow, elem.num_connections};
            max_connections = std::max(max_connections, static_cast<size_t>(elem.num_connections));
        }
        std::vector<DeviceConnection> connections(n_elems * max_connections);
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            for (size_t j = 0; j < elem.num_connections; ++j) {
                connections[j * n_elems + i] = {elem.connected_idx[j], elem.connected_flux[j]};
            }
        }

        CUDA_CHECK(cudaMalloc(&d_elements, n_elems * sizeof(DeviceElement)));
        // A one-element allocation also supports a mesh with no connections.
        CUDA_CHECK(cudaMalloc(&d_connections, std::max(connections.size(), size_t{1}) * sizeof(DeviceConnection)));
        CUDA_CHECK(cudaMalloc(&d_energy[0], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_energy[1], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_flux, n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(d_elements, elements.data(), n_elems * sizeof(DeviceElement), cudaMemcpyHostToDevice));
        if (!connections.empty()) {
            CUDA_CHECK(cudaMemcpy(d_connections, connections.data(), connections.size() * sizeof(DeviceConnection), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy2D(d_energy[0], sizeof(val_t), &world.elements_dynamic[0].current_energy,
                                sizeof(ElementDynamic), sizeof(val_t), n_elems, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(d_flux, sizeof(val_t), &world.elements_dynamic[0].total_flux,
                                sizeof(ElementDynamic), sizeof(val_t), n_elems, cudaMemcpyHostToDevice));
    }

    void run(int n_iters) {
        constexpr unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>((n_elems + threads - 1) / threads);
        for (int iter = 0; iter < n_iters; ++iter) {
            updateElements<<<blocks, threads>>>(d_elements, d_connections, d_energy[current],
                                                 d_energy[1 - current], d_flux, n_elems);
            current = 1 - current;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void copyBack(World& world) const {
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic[0].current_energy, sizeof(ElementDynamic),
                                d_energy[current], sizeof(val_t), sizeof(val_t), n_elems, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic[0].total_flux, sizeof(ElementDynamic),
                                d_flux, sizeof(val_t), sizeof(val_t), n_elems, cudaMemcpyDeviceToHost));
    }

    ~DeviceWorld() {
        cudaFree(d_elements);
        cudaFree(d_connections);
        cudaFree(d_energy[0]);
        cudaFree(d_energy[1]);
        cudaFree(d_flux);
    }
    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;

private:
    size_t n_elems;
    DeviceElement* d_elements = nullptr;
    DeviceConnection* d_connections = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux = nullptr;
    int current = 0;
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
    
    // Prepare GPU data before measuring the iteration loop.
    DeviceWorld device_world(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    device_world.run(n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    device_world.copyBack(world);
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %.3f ms\n", duration_ms);

    // All iterations are timed, including the first one.
    const int n_measured_iters = std::max(n_iters, 0);
    const double time_per_iter = n_measured_iters ? duration_ms / n_measured_iters : 0.0;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9
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
