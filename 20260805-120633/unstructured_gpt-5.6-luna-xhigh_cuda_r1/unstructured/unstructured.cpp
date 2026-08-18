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

// Device-side, structure-of-arrays representation.  The host representation
// remains unchanged for the benchmark's output and validation interfaces, but
// transposing the connectivity makes the reads for a warp contiguous for each
// connection slot.
struct GpuWorld {
    std::uint32_t n_elems = 0;
    std::uint8_t* material_idx = nullptr;
    std::uint8_t* num_connections = nullptr;
    std::uint32_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* current_energy[2] = {nullptr, nullptr};
    val_t* total_flux[2] = {nullptr, nullptr};
    int current_buffer = 0;

    GpuWorld() = default;
    GpuWorld(const GpuWorld&) = delete;
    GpuWorld& operator=(const GpuWorld&) = delete;

    ~GpuWorld() {
        release();
    }

    void release() {
        // cudaFree(nullptr) is valid, and ignoring cleanup errors here keeps a
        // prior CUDA error from being obscured during stack unwinding.
        cudaFree(material_idx);
        cudaFree(num_connections);
        cudaFree(connected_idx);
        cudaFree(connected_flux);
        cudaFree(current_energy[0]);
        cudaFree(current_energy[1]);
        cudaFree(total_flux[0]);
        cudaFree(total_flux[1]);
        material_idx = nullptr;
        num_connections = nullptr;
        connected_idx = nullptr;
        connected_flux = nullptr;
        current_energy[0] = nullptr;
        current_energy[1] = nullptr;
        total_flux[0] = nullptr;
        total_flux[1] = nullptr;
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

    if (n_elems == 0) {
        return;
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

// Material data is small and read by every thread, so constant memory avoids
// redundant global-memory traffic while retaining the original material model.
__constant__ val_t device_transfer_coeff[3];
__constant__ val_t device_external_flow[3];

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t error = (expression); \
        if (error != cudaSuccess) { \
            cudaFailure(error, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// One CUDA thread owns one element.  Each iteration is a Jacobi update: all
// reads use current_energy/total_flux and all writes use the other buffers.
__global__ void updateElementsKernel(
    const std::uint8_t* __restrict__ material_idx,
    const std::uint8_t* __restrict__ num_connections,
    const std::uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ total_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux,
    const std::uint32_t n_elems) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const std::uint8_t material = material_idx[i];
    const val_t this_energy = current_energy[i];
    val_t element_flux = device_external_flow[material];

    // Connectivity is stored as [connection_slot][element], so adjacent
    // threads in a warp read adjacent indices and fluxes.
    for (std::uint32_t j = 0; j < num_connections[i]; ++j) {
        const std::uint32_t connection_offset = j * n_elems + i;
        const std::uint32_t neighbor = connected_idx[connection_offset];
        element_flux += (current_energy[neighbor] - this_energy) *
                        device_transfer_coeff[material] *
                        connected_flux[connection_offset] * 0.25;
    }

    next_energy[i] = this_energy + element_flux;
    next_total_flux[i] = total_flux[i] + fabs(element_flux);
}

void copyDynamicStateToDevice(const std::vector<ElementDynamic>& source,
                              val_t* device_energy, val_t* device_flux,
                              const std::uint32_t n_elems) {
    if (n_elems == 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy2D(
        device_energy, sizeof(val_t), source.data(), sizeof(ElementDynamic),
        sizeof(val_t), n_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        device_flux, sizeof(val_t),
        reinterpret_cast<const char*>(source.data()) + offsetof(ElementDynamic, total_flux),
        sizeof(ElementDynamic), sizeof(val_t), n_elems, cudaMemcpyHostToDevice));
}

void copyDynamicStateToHost(const val_t* device_energy, const val_t* device_flux,
                            std::vector<ElementDynamic>& destination,
                            const std::uint32_t n_elems) {
    if (n_elems == 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy2D(
        destination.data(), sizeof(ElementDynamic), device_energy, sizeof(val_t),
        sizeof(val_t), n_elems, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(
        reinterpret_cast<char*>(destination.data()) + offsetof(ElementDynamic, total_flux),
        sizeof(ElementDynamic), device_flux, sizeof(val_t), sizeof(val_t), n_elems,
        cudaMemcpyDeviceToHost));
}

void initializeGpuWorld(const World& world, GpuWorld& gpu_world) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "Too many elements for the CUDA index representation\n");
        std::exit(EXIT_FAILURE);
    }
    gpu_world.n_elems = static_cast<std::uint32_t>(n_elems);

    val_t transfer_coeff[3];
    val_t external_flow[3];
    for (int i = 0; i < 3; ++i) {
        transfer_coeff[i] = world.materials[i].transfer_coeff;
        external_flow[i] = world.materials[i].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(device_transfer_coeff, transfer_coeff,
                                  sizeof(transfer_coeff)));
    CUDA_CHECK(cudaMemcpyToSymbol(device_external_flow, external_flow,
                                  sizeof(external_flow)));

    if (gpu_world.n_elems == 0) {
        return;
    }

    const size_t n_connections = n_elems * MAX_CONNECTIONS;
    std::vector<std::uint8_t> material_idx(n_elems);
    std::vector<std::uint8_t> num_connections(n_elems);
    std::vector<std::uint32_t> connected_idx(n_connections, 0);
    std::vector<val_t> connected_flux(n_connections, 0.0);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& element = world.elements_static[i];
        material_idx[i] = static_cast<std::uint8_t>(element.material_idx);
        num_connections[i] = static_cast<std::uint8_t>(element.num_connections);
        for (idx_t j = 0; j < element.num_connections; ++j) {
            const size_t connection_offset = static_cast<size_t>(j) * n_elems + i;
            connected_idx[connection_offset] =
                static_cast<std::uint32_t>(element.connected_idx[j]);
            connected_flux[connection_offset] = element.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&gpu_world.material_idx, n_elems * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&gpu_world.num_connections, n_elems * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&gpu_world.connected_idx,
                          n_connections * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_world.connected_flux,
                          n_connections * sizeof(val_t)));
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaMalloc(&gpu_world.current_energy[buffer],
                              n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&gpu_world.total_flux[buffer],
                              n_elems * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(gpu_world.material_idx, material_idx.data(),
                          n_elems * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_world.num_connections, num_connections.data(),
                          n_elems * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_world.connected_idx, connected_idx.data(),
                          n_connections * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_world.connected_flux, connected_flux.data(),
                          n_connections * sizeof(val_t), cudaMemcpyHostToDevice));

    copyDynamicStateToDevice(world.elements_dynamic, gpu_world.current_energy[0],
                             gpu_world.total_flux[0], gpu_world.n_elems);
    copyDynamicStateToDevice(world.elements_dynamic_swap, gpu_world.current_energy[1],
                             gpu_world.total_flux[1], gpu_world.n_elems);
}

// Run the simulation entirely on the GPU.  The default stream orders the
// dependent Jacobi iterations without a host synchronization between them.
void runSimulation(GpuWorld& gpu_world, const int n_iters) {
    constexpr int threads_per_block = 256;
    if (gpu_world.n_elems == 0) {
        return;
    }
    const int blocks = static_cast<int>(
        (gpu_world.n_elems + threads_per_block - 1) / threads_per_block);

    for (int iter = 0; iter < n_iters; ++iter) {
        const int next_buffer = gpu_world.current_buffer ^ 1;
        updateElementsKernel<<<blocks, threads_per_block>>>(
            gpu_world.material_idx, gpu_world.num_connections,
            gpu_world.connected_idx, gpu_world.connected_flux,
            gpu_world.current_energy[gpu_world.current_buffer],
            gpu_world.total_flux[gpu_world.current_buffer],
            gpu_world.current_energy[next_buffer],
            gpu_world.total_flux[next_buffer], gpu_world.n_elems);
        CUDA_CHECK(cudaGetLastError());
        gpu_world.current_buffer = next_buffer;
    }

    // Ensure the measured interval includes completed GPU work and surface
    // asynchronous execution errors before results are consumed.
    CUDA_CHECK(cudaDeviceSynchronize());
}

void downloadGpuWorld(const GpuWorld& gpu_world, World& world) {
    const int final_buffer = gpu_world.current_buffer;
    copyDynamicStateToHost(gpu_world.current_energy[final_buffer],
                           gpu_world.total_flux[final_buffer],
                           world.elements_dynamic, gpu_world.n_elems);
    copyDynamicStateToHost(gpu_world.current_energy[final_buffer ^ 1],
                           gpu_world.total_flux[final_buffer ^ 1],
                           world.elements_dynamic_swap, gpu_world.n_elems);
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
    GpuWorld gpu_world;
    CUDA_CHECK(cudaSetDevice(0));
    initializeGpuWorld(world, gpu_world);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(gpu_world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    const double duration_ms = std::max(elapsed.count(), 1e-9);
    
    printf("Computation time: %.4f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    downloadGpuWorld(gpu_world, world);
    
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
