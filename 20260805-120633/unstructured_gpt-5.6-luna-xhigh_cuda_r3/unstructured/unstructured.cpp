#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

namespace {

[[noreturn]] void failCuda(const cudaError_t error, const char* expression,
                           const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cuda_status = (expression); \
        if (cuda_status != cudaSuccess) { \
            failCuda(cuda_status, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// Materials are read by every thread but never modified during a run.  Keeping
// them in constant memory makes the material lookup a broadcast for a warp.
__constant__ Material device_materials[3];

// One thread computes one element.  The connectivity is stored in a
// structure-of-arrays layout on the device, so neighboring threads issue
// coalesced loads for each connection slot while retaining the original
// connection order and accumulation order.
__global__ void updateElements(
    const idx_t* __restrict__ material_indices,
    const idx_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_indices,
    const val_t* __restrict__ connected_flux,
    const ElementDynamic* __restrict__ current,
    ElementDynamic* __restrict__ next,
    const std::size_t n_elems) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const ElementDynamic this_element = current[i];
    const Material material = device_materials[material_indices[i]];
    val_t total_flux = material.external_flow;
    const std::size_t connection_base = i * MAX_CONNECTIONS;

    for (idx_t j = 0; j < num_connections[i]; ++j) {
        const idx_t connection_offset = connection_base + j;
        const idx_t neighbor_idx = connected_indices[connection_offset];
        const val_t neighbor_energy = current[neighbor_idx].current_energy;
        total_flux += (neighbor_energy - this_element.current_energy) *
                      material.transfer_coeff * connected_flux[connection_offset] * 0.25;
    }

    next[i].current_energy = this_element.current_energy + total_flux;
    next[i].total_flux = this_element.total_flux + fabs(total_flux);
}

class CudaSimulation {
  public:
    explicit CudaSimulation(const World& world)
        : n_elems_(world.elements_static.size()) {
        int device_count = 0;
        const cudaError_t device_count_status = cudaGetDeviceCount(&device_count);
        if (device_count_status != cudaSuccess) {
            failCuda(device_count_status, "cudaGetDeviceCount(&device_count)", __FILE__, __LINE__);
        }
        if (device_count == 0) {
            std::fprintf(stderr, "CUDA error: no CUDA device is available\n");
            std::exit(EXIT_FAILURE);
        }

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpyToSymbol(device_materials, world.materials.data(),
                                      world.materials.size() * sizeof(Material)));

        if (n_elems_ == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&material_indices_),
                              n_elems_ * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&num_connections_),
                              n_elems_ * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&connected_indices_),
                              n_elems_ * MAX_CONNECTIONS * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&connected_flux_),
                              n_elems_ * MAX_CONNECTIONS * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dynamic_[0]),
                              n_elems_ * sizeof(ElementDynamic)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dynamic_[1]),
                              n_elems_ * sizeof(ElementDynamic)));

        const char* static_data = reinterpret_cast<const char*>(world.elements_static.data());
        CUDA_CHECK(cudaMemcpy2D(
            material_indices_, sizeof(idx_t),
            static_data + offsetof(ElementStatic, material_idx), sizeof(ElementStatic),
            sizeof(idx_t), n_elems_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(
            num_connections_, sizeof(idx_t),
            static_data + offsetof(ElementStatic, num_connections), sizeof(ElementStatic),
            sizeof(idx_t), n_elems_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(
            connected_indices_, MAX_CONNECTIONS * sizeof(idx_t),
            static_data + offsetof(ElementStatic, connected_idx), sizeof(ElementStatic),
            MAX_CONNECTIONS * sizeof(idx_t), n_elems_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(
            connected_flux_, MAX_CONNECTIONS * sizeof(val_t),
            static_data + offsetof(ElementStatic, connected_flux), sizeof(ElementStatic),
            MAX_CONNECTIONS * sizeof(val_t), n_elems_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dynamic_[0], world.elements_dynamic.data(),
                              n_elems_ * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        cudaFree(dynamic_[0]);
        cudaFree(dynamic_[1]);
        cudaFree(connected_flux_);
        cudaFree(connected_indices_);
        cudaFree(num_connections_);
        cudaFree(material_indices_);
    }

    // Run all iterations in the default stream.  Kernel launches in one
    // stream are ordered, providing the global buffer-swap barrier required
    // by the original Jacobi update without a per-element synchronization.
    void run(const int n_iters) const {
        if (n_elems_ == 0 || n_iters <= 0) {
            return;
        }

        constexpr unsigned int threads_per_block = 256;
        const unsigned int blocks = static_cast<unsigned int>(
            (n_elems_ + threads_per_block - 1) / threads_per_block);
        int current_buffer = 0;

        for (int iter = 0; iter < n_iters; ++iter) {
            updateElements<<<blocks, threads_per_block>>>(
                material_indices_, num_connections_, connected_indices_, connected_flux_,
                dynamic_[current_buffer], dynamic_[1 - current_buffer], n_elems_);
            CUDA_CHECK(cudaGetLastError());
            current_buffer = 1 - current_buffer;
        }
    }

    void download(World& world, const int n_iters) const {
        if (n_elems_ == 0) {
            return;
        }

        const int current_buffer = (n_iters > 0) ? (n_iters & 1) : 0;
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), dynamic_[current_buffer],
                              n_elems_ * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    }

  private:
    std::size_t n_elems_ = 0;
    idx_t* material_indices_ = nullptr;
    idx_t* num_connections_ = nullptr;
    idx_t* connected_indices_ = nullptr;
    val_t* connected_flux_ = nullptr;
    ElementDynamic* dynamic_[2] = {nullptr, nullptr};
};

}  // namespace

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
    
    // Allocate and populate the GPU state before timing the iteration loop.
    // The benchmark's computation time therefore measures the CUDA workload,
    // rather than one-time mesh upload and allocation costs.
    CudaSimulation simulation(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(n_iters);
    simulation.download(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
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
