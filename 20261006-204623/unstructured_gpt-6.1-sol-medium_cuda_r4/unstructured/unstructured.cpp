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

// Fail explicitly if CUDA is unavailable; there is no CPU simulation fallback.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    size_t size = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t count) {
        size = count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    void upload(const std::vector<T>& source) {
        allocate(source.size());
        CUDA_CHECK(cudaMemcpy(data, source.data(), source.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    // Connection slot j occupies [j*N, (j+1)*N), so adjacent threads
    // read adjacent entries even though the mesh itself is unstructured.
    DeviceBuffer<uint32_t> neighbors;
    DeviceBuffer<val_t> connection_flux;
    DeviceBuffer<unsigned char> connection_count;
    DeviceBuffer<val_t> transfer_coeff;
    DeviceBuffer<val_t> external_flow;
    DeviceBuffer<val_t> energy_a;
    DeviceBuffer<val_t> energy_b;
    DeviceBuffer<val_t> accumulated_flux;
    DeviceBuffer<ElementDynamic> output;
    val_t* current_energy = nullptr;
    val_t* next_energy = nullptr;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Transpose the mesh once, outside the timed iteration loop. Indices fit
// uint32_t because the builder and command-line grid size are bounded by INT_MAX.
void prepareDevice(World& world) {
    const size_t count = world.elements_static.size();
    size_t max_connections = 0;
    for (const auto& elem : world.elements_static)
        max_connections = std::max(max_connections, size_t(elem.num_connections));
    // A one-element grid has no edges, but still uses valid device allocations.
    const size_t slots = std::max(max_connections, size_t(1));
    std::vector<uint32_t> neighbors(slots * count);
    std::vector<val_t> flux(slots * count);
    std::vector<unsigned char> connections(count);
    std::vector<val_t> coeff(count), external(count), energy(count), accumulated(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& elem = world.elements_static[i];
        const auto& mat = world.materials[elem.material_idx];
        connections[i] = static_cast<unsigned char>(elem.num_connections);
        coeff[i] = mat.transfer_coeff;
        external[i] = mat.external_flow;
        energy[i] = world.elements_dynamic[i].current_energy;
        accumulated[i] = world.elements_dynamic[i].total_flux;
        for (size_t j = 0; j < elem.num_connections; ++j) {
            neighbors[j * count + i] = static_cast<uint32_t>(elem.connected_idx[j]);
            flux[j * count + i] = elem.connected_flux[j];
        }
    }
    world.neighbors.upload(neighbors);
    world.connection_flux.upload(flux);
    world.connection_count.upload(connections);
    world.transfer_coeff.upload(coeff);
    world.external_flow.upload(external);
    world.energy_a.upload(energy);
    world.energy_b.allocate(count);
    world.accumulated_flux.upload(accumulated);
    world.output.allocate(count);
    world.current_energy = world.energy_a.data;
    world.next_energy = world.energy_b.data;
}

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

// One thread owns each output element. Only energy is double buffered:
// accumulated flux is never read by neighbors and can be updated in place.
__global__ void updateElements(size_t count,
                               const uint32_t* __restrict__ neighbors,
                               const val_t* __restrict__ connection_flux,
                               const unsigned char* __restrict__ connections,
                               const val_t* __restrict__ coeff,
                               const val_t* __restrict__ external,
                               const val_t* __restrict__ energy,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ accumulated_flux) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const val_t current = energy[i];
    const val_t transfer = coeff[i];
    val_t total_flux = external[i];
    // Preserve the original connection order and individual FP operations.
    for (unsigned int j = 0; j < connections[i]; ++j) {
        const size_t offset = size_t(j) * count + i;
        total_flux += (energy[neighbors[offset]] - current) * transfer *
                      connection_flux[offset] * 0.25;
    }
    next_energy[i] = current + total_flux;
    accumulated_flux[i] += fabs(total_flux);
}

__global__ void packResults(size_t count, const val_t* __restrict__ energy,
                            const val_t* __restrict__ flux,
                            ElementDynamic* __restrict__ output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = ElementDynamic{energy[i], flux[i]};
}

// Consecutive launches on the same stream provide the global iteration barrier.
// No host transfers, allocations, or device synchronizations occur per iteration.
void runSimulation(World& world, const int n_iters) {
    const size_t count = world.elements_static.size();
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    for (int iter = 0; iter < n_iters; ++iter) {
        updateElements<<<blocks, threads>>>(count, world.neighbors.data,
            world.connection_flux.data, world.connection_count.data,
            world.transfer_coeff.data, world.external_flow.data,
            world.current_energy, world.next_energy, world.accumulated_flux.data);
        std::swap(world.current_energy, world.next_energy);
    }
    CUDA_CHECK(cudaGetLastError());
    packResults<<<blocks, threads>>>(count, world.current_energy,
                                    world.accumulated_flux.data, world.output.data);
    CUDA_CHECK(cudaGetLastError());
    // This synchronous download also checks asynchronous execution errors and
    // ensures the benchmark timer includes all GPU work and final results.
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.output.data,
                         count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
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
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    if (n_elems_root <= 0 ||
        int64_t(n_elems_root) * n_elems_root > std::numeric_limits<int>::max() ||
        n_iters < 0) {
        fprintf(stderr, "Grid size must be positive with at most INT_MAX elements; iterations must be nonnegative.\n");
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize CUDA even for zero iterations; simulation always uses the GPU.
    CUDA_CHECK(cudaSetDevice(0));

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
    prepareDevice(world);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic);
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    const size_t device_mem = world.neighbors.size * sizeof(uint32_t) +
        world.connection_flux.size * sizeof(val_t) + world.connection_count.size +
        (world.transfer_coeff.size + world.external_flow.size + world.energy_a.size +
         world.energy_b.size + world.accumulated_flux.size) * sizeof(val_t) +
        world.output.size * sizeof(ElementDynamic);
    printf("GPU memory usage: %.2f MB\n", device_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double time_per_iter = elapsed_ms / std::max(n_iters, 1);
    const double giga_elems_per_sec = (double(n_iters) * n_elems) / (elapsed_ms * 1e6);
    
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
