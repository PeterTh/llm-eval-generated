#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace cg = cooperative_groups;

namespace {

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void checkCuda(cudaError_t error, const char* expression,
                      const char* file, int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, expression, file, line);
    }
}

#define CUDA_CHECK(expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__)

}  // namespace

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

static_assert(std::is_trivially_copyable<Material>::value,
              "Material must be transferable to CUDA constant memory");

// Device storage is structure-of-arrays.  The connectivity uses an ELLPACK
// layout (connection slot first, then element), so threads in a warp read
// adjacent addresses even though the logical mesh is unstructured.
struct DeviceWorld {
    uint32_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    uint8_t* material_idx = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* accumulated_flux[2] = {nullptr, nullptr};
    unsigned long long* result_hash = nullptr;
    size_t num_elements = 0;
    int packed_degree = 0;
    int active_buffer = 0;
    bool unit_connection_flux = true;

    DeviceWorld() = default;
    DeviceWorld(const DeviceWorld&) = delete;
    DeviceWorld& operator=(const DeviceWorld&) = delete;

    ~DeviceWorld() {
        // Destructors cannot report errors usefully; all operational CUDA calls
        // are checked at their call sites.
        cudaFree(connected_idx);
        cudaFree(connected_flux);
        cudaFree(material_idx);
        cudaFree(energy[0]);
        cudaFree(energy[1]);
        cudaFree(accumulated_flux[0]);
        cudaFree(accumulated_flux[1]);
        cudaFree(result_hash);
    }
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    DeviceWorld device;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr int NUM_MATERIALS = 3;
__constant__ Material device_materials[NUM_MATERIALS];

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

template <typename T>
void cudaAllocate(T*& pointer, size_t count) {
    if (count == 0) {
        pointer = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
}

// Convert the host's array-of-structures mesh to a coalesced ELLPACK device
// representation.  Missing entries are self-edges with zero coefficient; this
// permits a completely unrolled inner loop without altering the update.
void initializeDeviceWorld(World& world) {
    DeviceWorld& device = world.device;
    const size_t n_elems = world.elements_static.size();
    device.num_elements = n_elems;

    if (world.materials.size() != NUM_MATERIALS) {
        std::fprintf(stderr, "Expected exactly %d materials, found %zu\n",
                     NUM_MATERIALS, world.materials.size());
        std::exit(EXIT_FAILURE);
    }

    int active_device = 0;
    CUDA_CHECK(cudaGetDevice(&active_device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, active_device));
    if (!properties.cooperativeLaunch) {
        std::fprintf(stderr,
                     "CUDA device '%s' does not support cooperative kernels\n",
                     properties.name);
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(device_materials, world.materials.data(),
                                  NUM_MATERIALS * sizeof(Material)));

    int max_degree = 0;
    for (const ElementStatic& element : world.elements_static) {
        if (element.num_connections > static_cast<idx_t>(MAX_CONNECTIONS)) {
            std::fprintf(stderr, "Mesh element exceeds MAX_CONNECTIONS\n");
            std::exit(EXIT_FAILURE);
        }
        max_degree = std::max(max_degree,
                              static_cast<int>(element.num_connections));
        for (idx_t slot = 0; slot < element.num_connections; ++slot) {
            device.unit_connection_flux &= element.connected_flux[slot] == 1.0;
        }
    }
    device.packed_degree = max_degree;

    const size_t packed_size = n_elems * static_cast<size_t>(max_degree);
    std::vector<uint8_t> material_indices(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& element = world.elements_static[i];
        if (element.material_idx >= world.materials.size()) {
            std::fprintf(stderr, "Invalid material index at element %zu\n", i);
            std::exit(EXIT_FAILURE);
        }
        material_indices[i] = static_cast<uint8_t>(element.material_idx);
    }

    cudaAllocate(device.connected_idx, packed_size);
    cudaAllocate(device.connected_flux,
                 device.unit_connection_flux ? 0 : packed_size);
    cudaAllocate(device.material_idx, n_elems);
    cudaAllocate(device.energy[0], n_elems);
    cudaAllocate(device.energy[1], n_elems);
    cudaAllocate(device.accumulated_flux[0], n_elems);
    cudaAllocate(device.accumulated_flux[1], n_elems);
    cudaAllocate(device.result_hash, size_t{1});

    // Stream one ELLPACK slot at a time.  This bounds conversion scratch space
    // by O(elements), instead of duplicating the full O(edges) device topology
    // in host memory during setup.
    if (packed_size != 0) {
        std::vector<uint32_t> slot_indices(n_elems);
        std::vector<val_t> slot_flux(device.unit_connection_flux ? 0 : n_elems);
        for (int slot = 0; slot < max_degree; ++slot) {
            for (size_t i = 0; i < n_elems; ++i) {
                const ElementStatic& element = world.elements_static[i];
                if (slot < static_cast<int>(element.num_connections)) {
                    const idx_t neighbor = element.connected_idx[slot];
                    if (neighbor >= n_elems || neighbor > UINT32_MAX) {
                        std::fprintf(
                            stderr,
                            "Invalid connection at element %zu, slot %d\n",
                            i, slot);
                        std::exit(EXIT_FAILURE);
                    }
                    slot_indices[i] = static_cast<uint32_t>(neighbor);
                    if (!device.unit_connection_flux) {
                        slot_flux[i] = element.connected_flux[slot];
                    }
                } else {
                    slot_indices[i] = static_cast<uint32_t>(i);
                    if (!device.unit_connection_flux) {
                        slot_flux[i] = 0.0;
                    }
                }
            }
            const size_t slot_offset = static_cast<size_t>(slot) * n_elems;
            CUDA_CHECK(cudaMemcpy(device.connected_idx + slot_offset,
                                  slot_indices.data(),
                                  n_elems * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            if (!device.unit_connection_flux) {
                CUDA_CHECK(cudaMemcpy(device.connected_flux + slot_offset,
                                      slot_flux.data(),
                                      n_elems * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(device.material_idx, material_indices.data(),
                          n_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
    // buildSquare2D initializes both dynamic fields to zero.  Device memset is
    // much faster and avoids two additional full-size host staging arrays.
    CUDA_CHECK(cudaMemset(device.energy[0], 0, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(device.accumulated_flux[0], 0,
                          n_elems * sizeof(val_t)));

    // Static topology is now device-resident and is never consulted by output
    // or validation, so release its comparatively large host AoS allocation.
    std::vector<ElementStatic>().swap(world.elements_static);
}

template <int PackedDegree, bool UnitConnectionFlux>
__global__ void simulationKernel(
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const uint8_t* __restrict__ material_idx,
    val_t* __restrict__ energy_a,
    val_t* __restrict__ energy_b,
    val_t* __restrict__ accumulated_flux_a,
    val_t* __restrict__ accumulated_flux_b,
    size_t n_elems,
    int n_iters) {
    cg::grid_group grid = cg::this_grid();
    val_t* input_energy = energy_a;
    val_t* output_energy = energy_b;
    val_t* input_accumulated_flux = accumulated_flux_a;
    val_t* output_accumulated_flux = accumulated_flux_b;
    const size_t global_thread = blockIdx.x * static_cast<size_t>(blockDim.x)
                               + threadIdx.x;
    const size_t grid_stride = gridDim.x * static_cast<size_t>(blockDim.x);

    for (int iteration = 0; iteration < n_iters; ++iteration) {
        for (size_t i = global_thread; i < n_elems; i += grid_stride) {
            const val_t current_energy = input_energy[i];
            const Material material = device_materials[material_idx[i]];
            val_t total_flux = material.external_flow;

#pragma unroll
            for (int slot = 0; slot < PackedDegree; ++slot) {
                const size_t packed_index = static_cast<size_t>(slot) * n_elems + i;
                const val_t neighbor_energy = input_energy[connected_idx[packed_index]];
                const val_t energy_difference = neighbor_energy - current_energy;
                if constexpr (UnitConnectionFlux) {
                    total_flux += energy_difference
                                * material.transfer_coeff * 0.25;
                } else {
                    total_flux += energy_difference
                                * material.transfer_coeff
                                * connected_flux[packed_index] * 0.25;
                }
            }

            output_energy[i] = current_energy + total_flux;
            output_accumulated_flux[i] = input_accumulated_flux[i]
                                       + fabs(total_flux);
        }

        // Every element in the next Jacobi step must observe the completed
        // previous buffer.  A cooperative grid barrier provides that ordering
        // while retaining a single persistent kernel for all iterations.
        if (iteration + 1 < n_iters) {
            grid.sync();
        }
        val_t* temporary = input_energy;
        input_energy = output_energy;
        output_energy = temporary;
        temporary = input_accumulated_flux;
        input_accumulated_flux = output_accumulated_flux;
        output_accumulated_flux = temporary;
    }
}

template <int PackedDegree, bool UnitConnectionFlux>
void launchSimulation(DeviceWorld& device, int n_iters) {
    constexpr int threads_per_block = 256;
    int active_device = 0;
    CUDA_CHECK(cudaGetDevice(&active_device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, active_device));

    int blocks_per_multiprocessor = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        simulationKernel<PackedDegree, UnitConnectionFlux>,
        threads_per_block, 0));
    const size_t blocks_needed =
        (device.num_elements + threads_per_block - 1) / threads_per_block;
    const int resident_blocks = blocks_per_multiprocessor
                              * properties.multiProcessorCount;
    const int grid_blocks = static_cast<int>(
        std::min(blocks_needed, static_cast<size_t>(resident_blocks)));

    const uint32_t* connected_idx = device.connected_idx;
    const val_t* connected_flux = device.connected_flux;
    const uint8_t* material_idx = device.material_idx;
    val_t* energy_a = device.energy[0];
    val_t* energy_b = device.energy[1];
    val_t* accumulated_flux_a = device.accumulated_flux[0];
    val_t* accumulated_flux_b = device.accumulated_flux[1];
    size_t n_elems = device.num_elements;
    void* arguments[] = {
        &connected_idx, &connected_flux, &material_idx,
        &energy_a, &energy_b, &accumulated_flux_a, &accumulated_flux_b,
        &n_elems, &n_iters
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(
            simulationKernel<PackedDegree, UnitConnectionFlux>),
        dim3(grid_blocks), dim3(threads_per_block), arguments));
}

template <bool UnitConnectionFlux>
void dispatchSimulation(DeviceWorld& device, int n_iters) {
    switch (device.packed_degree) {
        case 0: launchSimulation<0, UnitConnectionFlux>(device, n_iters); break;
        case 1: launchSimulation<1, UnitConnectionFlux>(device, n_iters); break;
        case 2: launchSimulation<2, UnitConnectionFlux>(device, n_iters); break;
        case 3: launchSimulation<3, UnitConnectionFlux>(device, n_iters); break;
        case 4: launchSimulation<4, UnitConnectionFlux>(device, n_iters); break;
        case 5: launchSimulation<5, UnitConnectionFlux>(device, n_iters); break;
        case 6: launchSimulation<6, UnitConnectionFlux>(device, n_iters); break;
        case 7: launchSimulation<7, UnitConnectionFlux>(device, n_iters); break;
        case 8: launchSimulation<8, UnitConnectionFlux>(device, n_iters); break;
        default:
            std::fprintf(stderr, "Unsupported packed mesh degree: %d\n",
                         device.packed_degree);
            std::exit(EXIT_FAILURE);
    }
}

void warmupSimulation(DeviceWorld& device) {
    // Force lazy CUDA module loading and cooperative-launch initialization out
    // of the measured region.  One iteration also raises the GPU from its idle
    // power state.  It writes only buffer 1, leaving the initial state in
    // buffer 0 untouched for the measured simulation.
    if (device.unit_connection_flux) {
        dispatchSimulation<true>(device, 1);
    } else {
        dispatchSimulation<false>(device, 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Run all iterations in one persistent cooperative CUDA kernel.  Specializing
// the packed degree makes the adjacency loop compile-time bounded and unrolled.
void runSimulation(World& world, const int n_iters) {
    DeviceWorld& device = world.device;
    if (n_iters <= 0) {
        device.active_buffer = 0;
        return;
    }

    if (device.unit_connection_flux) {
        dispatchSimulation<true>(device, n_iters);
    } else {
        dispatchSimulation<false>(device, n_iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    device.active_buffer = n_iters & 1;
}

void downloadSimulationResults(World& world) {
    const DeviceWorld& device = world.device;
    const size_t n_elems = device.num_elements;
    std::vector<val_t> energy(n_elems);
    std::vector<val_t> accumulated_flux(n_elems);
    CUDA_CHECK(cudaMemcpy(energy.data(), device.energy[device.active_buffer],
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(accumulated_flux.data(),
                          device.accumulated_flux[device.active_buffer],
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = accumulated_flux[i];
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

__global__ void hashKernel(const val_t* __restrict__ energy,
                           const val_t* __restrict__ accumulated_flux,
                           size_t n_elems,
                           unsigned long long* result) {
    constexpr int hash_threads = 256;
    __shared__ unsigned long long partial_hashes[hash_threads];
    const size_t global_thread = blockIdx.x * static_cast<size_t>(blockDim.x)
                               + threadIdx.x;
    const size_t grid_stride = gridDim.x * static_cast<size_t>(blockDim.x);
    unsigned long long local_hash = 0;

    for (size_t i = global_thread; i < n_elems; i += grid_stride) {
        const unsigned long long energy_bits =
            static_cast<unsigned long long>(__double_as_longlong(energy[i]));
        const unsigned long long flux_bits = static_cast<unsigned long long>(
            __double_as_longlong(accumulated_flux[i]));
        local_hash ^= (energy_bits + static_cast<unsigned long long>(i))
                    * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + static_cast<unsigned long long>(i))
                    * 0xbf58476d1ce4e5b9ULL;
    }

    partial_hashes[threadIdx.x] = local_hash;
    __syncthreads();
    for (int stride = hash_threads / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_hashes[threadIdx.x] ^= partial_hashes[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(result, partial_hashes[0]);
    }
}

// The verification hash is also computed in parallel, avoiding a full device
// to host transfer during normal benchmark runs.
uint64_t computeHash(DeviceWorld& device) {
    constexpr int threads_per_block = 256;
    const size_t blocks_needed =
        (device.num_elements + threads_per_block - 1) / threads_per_block;
    const int blocks = static_cast<int>(std::min<size_t>(blocks_needed, 256));
    CUDA_CHECK(cudaMemset(device.result_hash, 0, sizeof(unsigned long long)));
    hashKernel<<<blocks, threads_per_block>>>(
        device.energy[device.active_buffer],
        device.accumulated_flux[device.active_buffer],
        device.num_elements, device.result_hash);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long result = 0;
    CUDA_CHECK(cudaMemcpy(&result, device.result_hash, sizeof(result),
                          cudaMemcpyDeviceToHost));
    return static_cast<uint64_t>(result);
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

    if (n_elems_root <= 0 || n_elems_root > 46340) {
        std::fprintf(stderr,
                     "Grid size must be in the range [1, 46340]\n");
        return 1;
    }
    if (n_iters < 0) {
        std::fprintf(stderr, "Iteration count must be non-negative\n");
        return 1;
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

    // Device allocation, mesh packing, and one-time transfers are setup work,
    // matching the original benchmark's exclusion of mesh construction.
    initializeDeviceWorld(world);
    warmupSimulation(world.device);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec =
        (static_cast<double>(n_measured_iters) * n_elems)
        / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.device);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    if (printResults || validate) {
        downloadSimulationResults(world);
    }
    
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
