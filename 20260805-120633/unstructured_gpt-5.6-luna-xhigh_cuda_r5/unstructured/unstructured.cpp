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
constexpr size_t MATERIAL_COUNT = 3;
constexpr unsigned CUDA_BLOCK_SIZE = 256;

// The benchmark is deliberately GPU-only: CUDA failures are fatal instead of
// silently changing the execution path or its performance characteristics.
inline void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__constant__ Material device_materials[MATERIAL_COUNT];

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
__host__ __device__ __forceinline__ val_t computeFlux(const Material& mat,
                                                      val_t this_energy,
                                                      val_t connection_flux,
                                                      val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Device-side connectivity is stored slot-major.  Threads in a warp then
// read contiguous neighbor indices and flux coefficients for the same slot,
// while the dynamic state is kept structure-of-arrays for coalesced access.
__global__ void updateElementsKernel(const idx_t* __restrict__ material_indices,
                                     const uint8_t* __restrict__ connection_counts,
                                     const idx_t* __restrict__ connected_indices,
                                     const val_t* __restrict__ connected_flux,
                                     const val_t* __restrict__ current_energy,
                                     const val_t* __restrict__ total_flux,
                                     val_t* __restrict__ current_energy_next,
                                     val_t* __restrict__ total_flux_next,
                                     const size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const Material mat = device_materials[material_indices[i]];
    const val_t this_energy = current_energy[i];
    val_t element_flux = mat.external_flow;

    const uint8_t n_connections = connection_counts[i];
    for (uint8_t j = 0; j < n_connections; ++j) {
        const size_t connection_offset = static_cast<size_t>(j) * n_elems + i;
        const idx_t neighbor_idx = connected_indices[connection_offset];
        element_flux += computeFlux(mat, this_energy,
                                    connected_flux[connection_offset],
                                    current_energy[neighbor_idx]);
    }

    current_energy_next[i] = this_energy + element_flux;
    total_flux_next[i] = total_flux[i] + fabs(element_flux);
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

// Own the device representation for one simulation.  Mesh/state transfers are
// performed before the timed region; the timed path consists solely of CUDA
// kernel launches and the required iteration ordering.
class CudaSimulation {
  public:
    explicit CudaSimulation(const World& world)
        : n_elems_(world.elements_static.size()),
          d_material_indices_(n_elems_),
          d_connection_counts_(n_elems_),
          d_connected_indices_(n_elems_ * MAX_CONNECTIONS),
          d_connected_flux_(n_elems_ * MAX_CONNECTIONS),
          d_current_energy_(n_elems_),
          d_next_energy_(n_elems_),
          d_total_flux_(n_elems_),
          d_next_total_flux_(n_elems_) {
        std::vector<idx_t> material_indices(n_elems_);
        std::vector<uint8_t> connection_counts(n_elems_);
        std::vector<idx_t> connected_indices(n_elems_ * MAX_CONNECTIONS, 0);
        std::vector<val_t> connected_flux(n_elems_ * MAX_CONNECTIONS, 0.0);
        std::vector<val_t> current_energy(n_elems_);
        std::vector<val_t> total_flux(n_elems_);

        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            material_indices[i] = elem.material_idx;
            connection_counts[i] = static_cast<uint8_t>(elem.num_connections);

            for (idx_t j = 0; j < elem.num_connections; ++j) {
                const size_t connection_offset = static_cast<size_t>(j) * n_elems_ + i;
                connected_indices[connection_offset] = elem.connected_idx[j];
                connected_flux[connection_offset] = elem.connected_flux[j];
            }

            current_energy[i] = world.elements_dynamic[i].current_energy;
            total_flux[i] = world.elements_dynamic[i].total_flux;
        }

        CUDA_CHECK(cudaMemcpyToSymbol(device_materials,
                                      world.materials.data(),
                                      MATERIAL_COUNT * sizeof(Material),
                                      0,
                                      cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_material_indices_.data(), material_indices.data(),
                              n_elems_ * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_connection_counts_.data(), connection_counts.data(),
                              n_elems_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_connected_indices_.data(), connected_indices.data(),
                              n_elems_ * MAX_CONNECTIONS * sizeof(idx_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_connected_flux_.data(), connected_flux.data(),
                              n_elems_ * MAX_CONNECTIONS * sizeof(val_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_current_energy_.data(), current_energy.data(),
                              n_elems_ * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_total_flux_.data(), total_flux.data(),
                              n_elems_ * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    void run(const int n_iters) {
        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid(static_cast<unsigned>((n_elems_ + CUDA_BLOCK_SIZE - 1) /
                                               CUDA_BLOCK_SIZE));

        val_t* current_energy = active_first_ ? d_current_energy_.data()
                                              : d_next_energy_.data();
        val_t* next_energy = active_first_ ? d_next_energy_.data()
                                           : d_current_energy_.data();
        val_t* total_flux = active_first_ ? d_total_flux_.data()
                                          : d_next_total_flux_.data();
        val_t* next_total_flux = active_first_ ? d_next_total_flux_.data()
                                               : d_total_flux_.data();

        for (int iter = 0; iter < n_iters; ++iter) {
            updateElementsKernel<<<grid, block>>>(
                d_material_indices_.data(),
                d_connection_counts_.data(),
                d_connected_indices_.data(),
                d_connected_flux_.data(),
                current_energy,
                total_flux,
                next_energy,
                next_total_flux,
                n_elems_);

            std::swap(current_energy, next_energy);
            std::swap(total_flux, next_total_flux);
            active_first_ = !active_first_;
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void copyResults(World& world) const {
        std::vector<val_t> current_energy(n_elems_);
        std::vector<val_t> total_flux(n_elems_);

        const val_t* active_energy = active_first_ ? d_current_energy_.data()
                                                   : d_next_energy_.data();
        const val_t* active_flux = active_first_ ? d_total_flux_.data()
                                                 : d_next_total_flux_.data();

        CUDA_CHECK(cudaMemcpy(current_energy.data(), active_energy,
                              n_elems_ * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(total_flux.data(), active_flux,
                              n_elems_ * sizeof(val_t), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < n_elems_; ++i) {
            world.elements_dynamic[i].current_energy = current_energy[i];
            world.elements_dynamic[i].total_flux = total_flux[i];
        }
    }

  private:
    size_t n_elems_;
    DeviceBuffer<idx_t> d_material_indices_;
    DeviceBuffer<uint8_t> d_connection_counts_;
    DeviceBuffer<idx_t> d_connected_indices_;
    DeviceBuffer<val_t> d_connected_flux_;
    DeviceBuffer<val_t> d_current_energy_;
    DeviceBuffer<val_t> d_next_energy_;
    DeviceBuffer<val_t> d_total_flux_;
    DeviceBuffer<val_t> d_next_total_flux_;
    bool active_first_ = true;
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
    
    // Run simulation
    printf("Running simulation...\n");
    CudaSimulation simulation(world);
    auto start = std::chrono::high_resolution_clock::now();
    
    simulation.run(n_iters);
    auto end = std::chrono::high_resolution_clock::now();
    simulation.copyResults(world);

    const std::chrono::duration<double, std::milli> elapsed = end - start;
    const double duration_ms = elapsed.count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
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
