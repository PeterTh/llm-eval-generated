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

[[noreturn]] void cudaCheckFailed(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
            expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_status_ = (expression);                           \
        if (cuda_status_ != cudaSuccess) {                                       \
            cudaCheckFailed(cuda_status_, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

// Static connectivity is converted to a slot-major structure of arrays on the
// device.  Consecutive threads therefore read consecutive headers and each
// connection slot, which avoids the highly-strided accesses of ElementStatic.
__global__ void updateElementsKernel(
    const idx_t* __restrict__ material_indices,
    const uint8_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_indices,
    const val_t* __restrict__ connected_fluxes,
    const Material* __restrict__ materials,
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ current_total_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux,
    const size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const val_t element_energy = current_energy[i];
    const Material material = materials[material_indices[i]];
    const uint8_t connection_count = num_connections[i];

    val_t total_flux = material.external_flow;

#pragma unroll
    for (int connection = 0; connection < MAX_CONNECTIONS; ++connection) {
        if (connection < connection_count) {
            const size_t offset = static_cast<size_t>(connection) * n_elems + i;
            const val_t neighbor_energy = current_energy[connected_indices[offset]];
            // Keep the operation order of computeFlux so the device calculation
            // has the same numerical semantics as the original update.
            total_flux += (neighbor_energy - element_energy) *
                          material.transfer_coeff * connected_fluxes[offset] * 0.25;
        }
    }

    next_energy[i] = element_energy + total_flux;
    next_total_flux[i] = current_total_flux[i] + fabs(total_flux);
}

class DeviceWorld {
public:
    explicit DeviceWorld(const World& world) : n_elems_(world.elements_static.size()) {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            fprintf(stderr, "CUDA error: no CUDA-capable device is available\n");
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaSetDevice(0));

        CUDA_CHECK(cudaMalloc(&material_indices_, n_elems_ * sizeof(*material_indices_)));
        CUDA_CHECK(cudaMalloc(&num_connections_, n_elems_ * sizeof(*num_connections_)));
        CUDA_CHECK(cudaMalloc(&connected_indices_,
                              MAX_CONNECTIONS * n_elems_ * sizeof(*connected_indices_)));
        CUDA_CHECK(cudaMalloc(&connected_fluxes_,
                              MAX_CONNECTIONS * n_elems_ * sizeof(*connected_fluxes_)));
        CUDA_CHECK(cudaMalloc(&materials_, world.materials.size() * sizeof(*materials_)));
        CUDA_CHECK(cudaMalloc(&current_energy_, n_elems_ * sizeof(*current_energy_)));
        CUDA_CHECK(cudaMalloc(&next_energy_, n_elems_ * sizeof(*next_energy_)));
        CUDA_CHECK(cudaMalloc(&current_total_flux_, n_elems_ * sizeof(*current_total_flux_)));
        CUDA_CHECK(cudaMalloc(&next_total_flux_, n_elems_ * sizeof(*next_total_flux_)));

        std::vector<idx_t> material_indices(n_elems_);
        std::vector<uint8_t> num_connections(n_elems_);
        std::vector<idx_t> connected_indices(MAX_CONNECTIONS * n_elems_);
        std::vector<val_t> connected_fluxes(MAX_CONNECTIONS * n_elems_);

        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& element = world.elements_static[i];
            material_indices[i] = element.material_idx;
            num_connections[i] = static_cast<uint8_t>(element.num_connections);
            for (size_t connection = 0; connection < MAX_CONNECTIONS; ++connection) {
                const size_t offset = connection * n_elems_ + i;
                connected_indices[offset] = element.connected_idx[connection];
                connected_fluxes[offset] = element.connected_flux[connection];
            }
        }

        CUDA_CHECK(cudaMemcpy(material_indices_, material_indices.data(),
                              n_elems_ * sizeof(*material_indices_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(num_connections_, num_connections.data(),
                              n_elems_ * sizeof(*num_connections_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_indices_, connected_indices.data(),
                              MAX_CONNECTIONS * n_elems_ * sizeof(*connected_indices_),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_fluxes_, connected_fluxes.data(),
                              MAX_CONNECTIONS * n_elems_ * sizeof(*connected_fluxes_),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(materials_, world.materials.data(),
                              world.materials.size() * sizeof(*materials_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(current_energy_, sizeof(*current_energy_),
                                &world.elements_dynamic.front().current_energy,
                                sizeof(ElementDynamic), sizeof(*current_energy_), n_elems_,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(current_total_flux_, sizeof(*current_total_flux_),
                                &world.elements_dynamic.front().total_flux,
                                sizeof(ElementDynamic), sizeof(*current_total_flux_), n_elems_,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(next_energy_, sizeof(*next_energy_),
                                &world.elements_dynamic_swap.front().current_energy,
                                sizeof(ElementDynamic), sizeof(*next_energy_), n_elems_,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(next_total_flux_, sizeof(*next_total_flux_),
                                &world.elements_dynamic_swap.front().total_flux,
                                sizeof(ElementDynamic), sizeof(*next_total_flux_), n_elems_,
                                cudaMemcpyHostToDevice));

        int minimum_grid_size = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minimum_grid_size, &block_size_, updateElementsKernel, 0, 0));
        if (block_size_ <= 0) {
            fprintf(stderr, "CUDA error: could not determine a kernel launch configuration\n");
            std::exit(EXIT_FAILURE);
        }
    }

    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;

    ~DeviceWorld() {
        cudaFree(next_total_flux_);
        cudaFree(current_total_flux_);
        cudaFree(next_energy_);
        cudaFree(current_energy_);
        cudaFree(materials_);
        cudaFree(connected_fluxes_);
        cudaFree(connected_indices_);
        cudaFree(num_connections_);
        cudaFree(material_indices_);
    }

    float run(const int n_iters) {
        const size_t grid_size = (n_elems_ + static_cast<size_t>(block_size_) - 1) /
                                 static_cast<size_t>(block_size_);

        cudaEvent_t start_event = nullptr;
        cudaEvent_t end_event = nullptr;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&end_event));
        CUDA_CHECK(cudaEventRecord(start_event));

        for (int iter = 0; iter < n_iters; ++iter) {
            updateElementsKernel<<<static_cast<unsigned int>(grid_size), block_size_>>>(
                material_indices_, num_connections_, connected_indices_, connected_fluxes_,
                materials_, current_energy_, current_total_flux_, next_energy_, next_total_flux_,
                n_elems_);
            CUDA_CHECK(cudaGetLastError());
            std::swap(current_energy_, next_energy_);
            std::swap(current_total_flux_, next_total_flux_);
        }

        CUDA_CHECK(cudaEventRecord(end_event));
        CUDA_CHECK(cudaEventSynchronize(end_event));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
        CUDA_CHECK(cudaEventDestroy(end_event));
        CUDA_CHECK(cudaEventDestroy(start_event));
        return elapsed_ms;
    }

    void copyResultToHost(World& world) const {
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic.front().current_energy,
                                sizeof(ElementDynamic), current_energy_,
                                sizeof(*current_energy_), sizeof(*current_energy_), n_elems_,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic.front().total_flux,
                                sizeof(ElementDynamic), current_total_flux_,
                                sizeof(*current_total_flux_), sizeof(*current_total_flux_), n_elems_,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic_swap.front().current_energy,
                                sizeof(ElementDynamic), next_energy_,
                                sizeof(*next_energy_), sizeof(*next_energy_), n_elems_,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(&world.elements_dynamic_swap.front().total_flux,
                                sizeof(ElementDynamic), next_total_flux_,
                                sizeof(*next_total_flux_), sizeof(*next_total_flux_), n_elems_,
                                cudaMemcpyDeviceToHost));
    }

private:
    size_t n_elems_ = 0;
    int block_size_ = 0;
    idx_t* material_indices_ = nullptr;
    uint8_t* num_connections_ = nullptr;
    idx_t* connected_indices_ = nullptr;
    val_t* connected_fluxes_ = nullptr;
    Material* materials_ = nullptr;
    val_t* current_energy_ = nullptr;
    val_t* next_energy_ = nullptr;
    val_t* current_total_flux_ = nullptr;
    val_t* next_total_flux_ = nullptr;
};

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

// Run every timestep on the GPU.  The two dynamic buffers stay resident on the
// device for the whole simulation, so the iterative dependency needs no host
// synchronization or transfer between kernel launches.
float runSimulation(DeviceWorld& device_world, const int n_iters) {
    return device_world.run(n_iters);
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

    // CUDA is the required execution path.  DeviceWorld owns all device memory
    // and performs the one-time upload outside the timed simulation region.
    DeviceWorld device_world(world);
    
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
    const float duration_ms = runSimulation(device_world, n_iters);

    // The existing reporting, result export, hash, and validation operate on
    // the final dynamic buffer, so transfer it once after timing the kernels.
    device_world.copyResultToHost(world);
    
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
