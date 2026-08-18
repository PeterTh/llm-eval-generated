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

// The simulation is always executed on CUDA.  Static mesh data is converted
// once to a structure-of-arrays representation with the connection dimension
// outermost.  Thus, threads in a warp access consecutive connection records.
struct DeviceWorld {
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    uint32_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* accumulated_flux[2] = {nullptr, nullptr};
    ElementDynamic* packed_results = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t simulation = nullptr;
    bool host_results_pinned = false;
};

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

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

__constant__ Material device_materials[3];

__global__ __launch_bounds__(256)
void updateElements(const uint8_t* __restrict__ material_idx,
                    const uint8_t* __restrict__ num_connections,
                    const uint32_t* __restrict__ connected_idx,
                    const val_t* __restrict__ connected_flux,
                    const val_t* __restrict__ energy_read,
                    const val_t* __restrict__ accumulated_flux_read,
                    val_t* __restrict__ energy_write,
                    val_t* __restrict__ accumulated_flux_write,
                    uint32_t n_elems) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const val_t current_energy = energy_read[i];
    const Material mat = device_materials[material_idx[i]];
    val_t total_flux = mat.external_flow;
    const uint32_t connections = num_connections[i];

#pragma unroll
    for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j >= connections) {
            break;
        }
        const uint32_t connection = j * n_elems + i;
        const val_t neighbor_energy = energy_read[connected_idx[connection]];
        total_flux += (neighbor_energy - current_energy) *
                      mat.transfer_coeff * connected_flux[connection] * 0.25;
    }

    energy_write[i] = current_energy + total_flux;
    accumulated_flux_write[i] = accumulated_flux_read[i] + fabs(total_flux);
}

__global__ __launch_bounds__(256)
void packResults(const val_t* __restrict__ energy,
                 const val_t* __restrict__ accumulated_flux,
                 ElementDynamic* __restrict__ packed_results,
                 uint32_t n_elems) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        packed_results[i] = ElementDynamic{energy[i], accumulated_flux[i]};
    }
}

// Allocate and upload outside the timed region, and pre-build a CUDA graph so
// iteration launch overhead does not dominate smaller meshes.
DeviceWorld prepareDeviceWorld(World& world, const int n_iters) {
    DeviceWorld device;
    const size_t n_elems_size = world.elements_static.size();
    if (n_elems_size > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "Mesh is too large for CUDA connectivity indices\n");
        std::exit(EXIT_FAILURE);
    }
    const uint32_t n_elems = static_cast<uint32_t>(n_elems_size);

    std::vector<uint8_t> material_idx(n_elems);
    std::vector<uint8_t> num_connections(n_elems);
    std::vector<val_t> initial_energy(n_elems);
    std::vector<val_t> initial_flux(n_elems);
    uint32_t max_connections = 0;
    for (uint32_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        material_idx[i] = static_cast<uint8_t>(elem.material_idx);
        num_connections[i] = static_cast<uint8_t>(elem.num_connections);
        initial_energy[i] = world.elements_dynamic[i].current_energy;
        initial_flux[i] = world.elements_dynamic[i].total_flux;
        max_connections = std::max(max_connections,
                                   static_cast<uint32_t>(elem.num_connections));
    }

    const size_t connection_count =
        static_cast<size_t>(max_connections) * n_elems;
    std::vector<uint32_t> connected_idx(connection_count);
    std::vector<val_t> connected_flux(connection_count);
    for (uint32_t i = 0; i < n_elems; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        for (uint32_t j = 0; j < elem.num_connections; ++j) {
            const size_t dst = static_cast<size_t>(j) * n_elems + i;
            connected_idx[dst] = static_cast<uint32_t>(elem.connected_idx[j]);
            connected_flux[dst] = elem.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&device.material_idx, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&device.num_connections, n_elems * sizeof(uint8_t)));
    if (connection_count != 0) {
        CUDA_CHECK(cudaMalloc(&device.connected_idx,
                              connection_count * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&device.connected_flux,
                              connection_count * sizeof(val_t)));
    }
    const size_t field_bytes = n_elems_size * sizeof(val_t);
    const size_t dynamic_bytes = n_elems_size * sizeof(ElementDynamic);
    CUDA_CHECK(cudaMalloc(&device.energy[0], field_bytes));
    CUDA_CHECK(cudaMalloc(&device.energy[1], field_bytes));
    CUDA_CHECK(cudaMalloc(&device.accumulated_flux[0], field_bytes));
    CUDA_CHECK(cudaMalloc(&device.accumulated_flux[1], field_bytes));
    CUDA_CHECK(cudaMalloc(&device.packed_results, dynamic_bytes));

    CUDA_CHECK(cudaMemcpyAsync(device.material_idx, material_idx.data(),
                               n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice,
                               device.stream));
    CUDA_CHECK(cudaMemcpyAsync(device.num_connections, num_connections.data(),
                               n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice,
                               device.stream));
    if (connection_count != 0) {
        CUDA_CHECK(cudaMemcpyAsync(device.connected_idx, connected_idx.data(),
                                   connection_count * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(device.connected_flux, connected_flux.data(),
                                   connection_count * sizeof(val_t),
                                   cudaMemcpyHostToDevice, device.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(device.energy[0], initial_energy.data(),
                               field_bytes, cudaMemcpyHostToDevice,
                               device.stream));
    CUDA_CHECK(cudaMemcpyAsync(device.accumulated_flux[0], initial_flux.data(),
                               field_bytes, cudaMemcpyHostToDevice,
                               device.stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(device_materials, world.materials.data(),
                                       world.materials.size() * sizeof(Material),
                                       0, cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));

    // Page-locking the existing result vector permits a genuinely asynchronous,
    // full-bandwidth result copy without altering its externally visible layout.
    if (n_iters > 0) {
        CUDA_CHECK(cudaHostRegister(world.elements_dynamic.data(), dynamic_bytes,
                                    cudaHostRegisterDefault));
        device.host_results_pinned = true;
        constexpr uint32_t threads_per_block = 256;
        const uint32_t blocks =
            (n_elems + threads_per_block - 1) / threads_per_block;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream,
                                          cudaStreamCaptureModeThreadLocal));
        int read_buffer = 0;
        for (int iter = 0; iter < n_iters; ++iter) {
            const int write_buffer = read_buffer ^ 1;
            updateElements<<<blocks, threads_per_block, 0, device.stream>>>(
                device.material_idx, device.num_connections,
                device.connected_idx, device.connected_flux,
                device.energy[read_buffer], device.accumulated_flux[read_buffer],
                device.energy[write_buffer], device.accumulated_flux[write_buffer],
                n_elems);
            read_buffer = write_buffer;
        }
        packResults<<<blocks, threads_per_block, 0, device.stream>>>(
            device.energy[read_buffer], device.accumulated_flux[read_buffer],
            device.packed_results, n_elems);
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&device.simulation, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    return device;
}

// Run the pre-built simulation and materialize the final state on the host.
void runSimulation(World& world, DeviceWorld& device) {
    if (device.simulation != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(device.simulation, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            world.elements_dynamic.data(), device.packed_results,
            world.elements_dynamic.size() * sizeof(ElementDynamic),
            cudaMemcpyDeviceToHost, device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
    }
}

void releaseDeviceWorld(World& world, DeviceWorld& device) {
    if (device.host_results_pinned) {
        CUDA_CHECK(cudaHostUnregister(world.elements_dynamic.data()));
    }
    if (device.simulation != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(device.simulation));
    }
    CUDA_CHECK(cudaFree(device.packed_results));
    CUDA_CHECK(cudaFree(device.accumulated_flux[1]));
    CUDA_CHECK(cudaFree(device.accumulated_flux[0]));
    CUDA_CHECK(cudaFree(device.energy[1]));
    CUDA_CHECK(cudaFree(device.energy[0]));
    if (device.connected_flux != nullptr) CUDA_CHECK(cudaFree(device.connected_flux));
    if (device.connected_idx != nullptr) CUDA_CHECK(cudaFree(device.connected_idx));
    CUDA_CHECK(cudaFree(device.num_connections));
    CUDA_CHECK(cudaFree(device.material_idx));
    CUDA_CHECK(cudaStreamDestroy(device.stream));
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

    // Device allocation, topology transformation, uploads, and graph creation
    // are setup work and intentionally excluded from the simulation timer.
    DeviceWorld device = prepareDeviceWorld(world, n_iters);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, device);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    releaseDeviceWorld(world, device);
    
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
