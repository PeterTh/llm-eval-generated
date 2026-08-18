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

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

[[noreturn]] void cudaFail(const cudaError_t status, const char* operation) {
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFail(status, operation);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// CSR is a compact GPU-friendly representation of the original fixed-width
// connectivity.  The ordering of every element's connections is retained, so
// the arithmetic order and mesh semantics are unchanged.
__global__ __launch_bounds__(256)
void updateElementsKernel(const size_t n_elems,
                          const idx_t* __restrict__ connection_offsets,
                          const uint32_t* __restrict__ connected_idx,
                          const val_t* __restrict__ connected_flux,
                          const val_t* __restrict__ transfer_coeff,
                          const val_t* __restrict__ external_flow,
                          const val_t* __restrict__ energy_in,
                          const val_t* __restrict__ accumulated_flux_in,
                          val_t* __restrict__ energy_out,
                          val_t* __restrict__ accumulated_flux_out) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const val_t this_energy = energy_in[i];
    const val_t this_transfer_coeff = transfer_coeff[i];
    val_t total_flux = external_flow[i];

    const idx_t begin = connection_offsets[i];
    const idx_t end = connection_offsets[i + 1];
    for (idx_t connection = begin; connection < end; ++connection) {
        const val_t neighbor_energy = __ldg(&energy_in[connected_idx[connection]]);
        const val_t flux = (neighbor_energy - this_energy) * this_transfer_coeff *
                           connected_flux[connection] * 0.25;
        total_flux += flux;
    }

    energy_out[i] = this_energy + total_flux;
    accumulated_flux_out[i] = accumulated_flux_in[i] + fabs(total_flux);
}

class GpuSimulation {
public:
    explicit GpuSimulation(const World& world) : n_elems_(world.elements_static.size()) {
        std::vector<idx_t> connection_offsets(n_elems_ + 1);
        std::vector<val_t> transfer_coeff(n_elems_);
        std::vector<val_t> external_flow(n_elems_);
        std::vector<val_t> current_energy(n_elems_);
        std::vector<val_t> accumulated_flux(n_elems_);
        std::vector<val_t> swap_energy(n_elems_);
        std::vector<val_t> swap_flux(n_elems_);

        size_t n_connections = 0;
        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            const Material& material = world.materials[elem.material_idx];
            connection_offsets[i] = static_cast<idx_t>(n_connections);
            n_connections += elem.num_connections;
            transfer_coeff[i] = material.transfer_coeff;
            external_flow[i] = material.external_flow;
            current_energy[i] = world.elements_dynamic[i].current_energy;
            accumulated_flux[i] = world.elements_dynamic[i].total_flux;
            swap_energy[i] = world.elements_dynamic_swap[i].current_energy;
            swap_flux[i] = world.elements_dynamic_swap[i].total_flux;
        }
        connection_offsets[n_elems_] = static_cast<idx_t>(n_connections);

        std::vector<uint32_t> connected_idx(n_connections);
        std::vector<val_t> connected_flux(n_connections);
        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            idx_t connection = connection_offsets[i];
            for (idx_t j = 0; j < elem.num_connections; ++j, ++connection) {
                connected_idx[connection] = static_cast<uint32_t>(elem.connected_idx[j]);
                connected_flux[connection] = elem.connected_flux[j];
            }
        }

        copyToDevice(d_connection_offsets_, connection_offsets);
        copyToDevice(d_connected_idx_, connected_idx);
        copyToDevice(d_connected_flux_, connected_flux);
        copyToDevice(d_transfer_coeff_, transfer_coeff);
        copyToDevice(d_external_flow_, external_flow);
        copyToDevice(d_energy_current_, current_energy);
        copyToDevice(d_accumulated_flux_current_, accumulated_flux);
        copyToDevice(d_energy_swap_, swap_energy);
        copyToDevice(d_accumulated_flux_swap_, swap_flux);

        CUDA_CHECK(cudaFuncSetCacheConfig(updateElementsKernel, cudaFuncCachePreferL1));
    }

    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    ~GpuSimulation() {
        cudaFree(d_connection_offsets_);
        cudaFree(d_connected_idx_);
        cudaFree(d_connected_flux_);
        cudaFree(d_transfer_coeff_);
        cudaFree(d_external_flow_);
        cudaFree(d_energy_current_);
        cudaFree(d_accumulated_flux_current_);
        cudaFree(d_energy_swap_);
        cudaFree(d_accumulated_flux_swap_);
    }

    // Run exclusively on the GPU.  One thread updates one element, and the
    // ping-pong buffers retain the synchronous iteration semantics.
    void run(const int n_iters) {
        if (n_elems_ == 0 || n_iters <= 0) {
            return;
        }

        constexpr unsigned int block_size = 256;
        const unsigned int grid_size = static_cast<unsigned int>(
            (n_elems_ + block_size - 1) / block_size);

        for (int iter = 0; iter < n_iters; ++iter) {
            updateElementsKernel<<<grid_size, block_size>>>(
                n_elems_, d_connection_offsets_, d_connected_idx_, d_connected_flux_,
                d_transfer_coeff_, d_external_flow_, d_energy_current_,
                d_accumulated_flux_current_, d_energy_swap_, d_accumulated_flux_swap_);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_energy_current_, d_energy_swap_);
            std::swap(d_accumulated_flux_current_, d_accumulated_flux_swap_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void copyResults(World& world) const {
        std::vector<val_t> current_energy(n_elems_);
        std::vector<val_t> accumulated_flux(n_elems_);
        if (n_elems_ != 0) {
            CUDA_CHECK(cudaMemcpy(current_energy.data(), d_energy_current_,
                                  n_elems_ * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(accumulated_flux.data(), d_accumulated_flux_current_,
                                  n_elems_ * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        for (size_t i = 0; i < n_elems_; ++i) {
            world.elements_dynamic[i].current_energy = current_energy[i];
            world.elements_dynamic[i].total_flux = accumulated_flux[i];
        }
    }

private:
    template <typename T>
    static void copyToDevice(T*& destination, const std::vector<T>& source) {
        const size_t bytes = source.size() * sizeof(T);
        if (bytes == 0) {
            destination = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&destination), bytes));
        CUDA_CHECK(cudaMemcpy(destination, source.data(), bytes, cudaMemcpyHostToDevice));
    }

    size_t n_elems_;
    idx_t* d_connection_offsets_ = nullptr;
    uint32_t* d_connected_idx_ = nullptr;
    val_t* d_connected_flux_ = nullptr;
    val_t* d_transfer_coeff_ = nullptr;
    val_t* d_external_flow_ = nullptr;
    val_t* d_energy_current_ = nullptr;
    val_t* d_accumulated_flux_current_ = nullptr;
    val_t* d_energy_swap_ = nullptr;
    val_t* d_accumulated_flux_swap_ = nullptr;
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

    // Build and upload the compact device representation before timing the
    // simulation itself.  Result download remains after the timed region.
    GpuSimulation simulation(world);
    
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
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    simulation.copyResults(world);
    
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
