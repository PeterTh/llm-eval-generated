#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
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

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(cudaError_t error, const char* expression,
                      const char* file, int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, expression, file, line);
    }
}

#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)

// The mesh is transposed to a structure-of-arrays representation before it is
// uploaded.  A warp therefore fetches consecutive material IDs, connection
// counts, neighbor IDs, and coefficients instead of 32 widely separated AoS
// records.  Device indices can be 32-bit because the host program's element
// count is an int, cutting the dominant connectivity traffic in half.
struct DeviceMesh {
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    uint32_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    Material* materials = nullptr;
    ElementDynamic* dynamic[2] = {nullptr, nullptr};
    size_t n_elems = 0;
    int connections_per_element = 0;

    DeviceMesh() = default;
    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;

    ~DeviceMesh() {
        cudaFree(dynamic[1]);
        cudaFree(dynamic[0]);
        cudaFree(materials);
        cudaFree(connected_flux);
        cudaFree(connected_idx);
        cudaFree(num_connections);
        cudaFree(material_idx);
    }
};

template <typename T>
void deviceAllocate(T*& pointer, size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                              count * sizeof(T)));
    }
}

template <typename T>
void copyToDevice(T* destination, const T* source, size_t count) {
    if (count != 0) {
        CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
}

void prepareDeviceMesh(const World& world, DeviceMesh& device) {
    device.n_elems = world.elements_static.size();

    if (device.n_elems > std::numeric_limits<uint32_t>::max()) {
        std::fprintf(stderr, "Mesh is too large for CUDA connectivity indices\n");
        std::exit(EXIT_FAILURE);
    }
    if (world.materials.size() > std::numeric_limits<uint8_t>::max()) {
        std::fprintf(stderr, "Too many material types for the CUDA mesh\n");
        std::exit(EXIT_FAILURE);
    }

    for (const ElementStatic& element : world.elements_static) {
        if (element.num_connections > MAX_CONNECTIONS) {
            std::fprintf(stderr, "Element has more than %d connections\n",
                         MAX_CONNECTIONS);
            std::exit(EXIT_FAILURE);
        }
        device.connections_per_element =
            std::max(device.connections_per_element,
                     static_cast<int>(element.num_connections));
    }

    std::vector<uint8_t> material_idx(device.n_elems);
    std::vector<uint8_t> num_connections(device.n_elems);
    const size_t connectivity_size =
        device.n_elems * static_cast<size_t>(device.connections_per_element);
    std::vector<uint32_t> connected_idx(connectivity_size);
    std::vector<val_t> connected_flux(connectivity_size);

    for (size_t i = 0; i < device.n_elems; ++i) {
        const ElementStatic& element = world.elements_static[i];
        if (element.material_idx >= world.materials.size()) {
            std::fprintf(stderr, "Element references an invalid material\n");
            std::exit(EXIT_FAILURE);
        }

        material_idx[i] = static_cast<uint8_t>(element.material_idx);
        num_connections[i] = static_cast<uint8_t>(element.num_connections);
        for (idx_t j = 0; j < element.num_connections; ++j) {
            if (element.connected_idx[j] >= device.n_elems) {
                std::fprintf(stderr, "Element references an invalid neighbor\n");
                std::exit(EXIT_FAILURE);
            }
            // Connection-major storage makes every load in a warp contiguous.
            const size_t offset = static_cast<size_t>(j) * device.n_elems + i;
            connected_idx[offset] =
                static_cast<uint32_t>(element.connected_idx[j]);
            connected_flux[offset] = element.connected_flux[j];
        }
    }

    deviceAllocate(device.material_idx, device.n_elems);
    deviceAllocate(device.num_connections, device.n_elems);
    deviceAllocate(device.connected_idx, connectivity_size);
    deviceAllocate(device.connected_flux, connectivity_size);
    deviceAllocate(device.materials, world.materials.size());
    deviceAllocate(device.dynamic[0], device.n_elems);
    deviceAllocate(device.dynamic[1], device.n_elems);

    copyToDevice(device.material_idx, material_idx.data(), device.n_elems);
    copyToDevice(device.num_connections, num_connections.data(), device.n_elems);
    copyToDevice(device.connected_idx, connected_idx.data(), connectivity_size);
    copyToDevice(device.connected_flux, connected_flux.data(), connectivity_size);
    copyToDevice(device.materials, world.materials.data(), world.materials.size());
    copyToDevice(device.dynamic[0], world.elements_dynamic.data(), device.n_elems);
}

template <int ConnectionsPerElement>
__global__ void simulationIterationKernel(
    const uint8_t* __restrict__ material_idx,
    const uint8_t* __restrict__ num_connections,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const Material* __restrict__ materials,
    const ElementDynamic* __restrict__ current,
    ElementDynamic* __restrict__ next,
    size_t n_elems) {
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = thread; i < n_elems; i += stride) {
        const ElementDynamic element = current[i];
        const Material material = materials[material_idx[i]];
        const int count = num_connections[i];
        val_t total_flux = material.external_flow;

        // Template specialization fully unrolls the small adjacency loop.  The
        // conditional retains the exact operation count and order for boundary
        // elements instead of padding them with arithmetic zeros.
#pragma unroll
        for (int j = 0; j < ConnectionsPerElement; ++j) {
            if (j < count) {
                const size_t offset = static_cast<size_t>(j) * n_elems + i;
                const ElementDynamic neighbor = current[connected_idx[offset]];
                total_flux +=
                    (neighbor.current_energy - element.current_energy) *
                    material.transfer_coeff * connected_flux[offset] * 0.25;
            }
        }

        next[i].current_energy = element.current_energy + total_flux;
        next[i].total_flux = element.total_flux + fabs(total_flux);
    }
}

template <int ConnectionsPerElement>
void launchSimulationIteration(DeviceMesh& device,
                               const ElementDynamic* current,
                               ElementDynamic* next,
                               cudaStream_t stream) {
    constexpr int threads_per_block = 256;
    const size_t blocks_for_mesh =
        (device.n_elems + threads_per_block - 1) / threads_per_block;
    simulationIterationKernel<ConnectionsPerElement>
        <<<static_cast<unsigned int>(blocks_for_mesh), threads_per_block, 0, stream>>>(
            device.material_idx, device.num_connections, device.connected_idx,
            device.connected_flux, device.materials, current, next, device.n_elems);
    CUDA_CHECK(cudaGetLastError());
}

void launchSimulationIteration(DeviceMesh& device,
                               const ElementDynamic* current,
                               ElementDynamic* next,
                               cudaStream_t stream) {
    // Every supported degree gets a compile-time-unrolled CUDA specialization.
    switch (device.connections_per_element) {
        case 0: launchSimulationIteration<0>(device, current, next, stream); break;
        case 1: launchSimulationIteration<1>(device, current, next, stream); break;
        case 2: launchSimulationIteration<2>(device, current, next, stream); break;
        case 3: launchSimulationIteration<3>(device, current, next, stream); break;
        case 4: launchSimulationIteration<4>(device, current, next, stream); break;
        case 5: launchSimulationIteration<5>(device, current, next, stream); break;
        case 6: launchSimulationIteration<6>(device, current, next, stream); break;
        case 7: launchSimulationIteration<7>(device, current, next, stream); break;
        case 8: launchSimulationIteration<8>(device, current, next, stream); break;
        default:
            std::fprintf(stderr, "Unsupported CUDA mesh degree\n");
            std::exit(EXIT_FAILURE);
    }
}

__global__ void emptySimulationKernel() {}

// Capture all iterations once so the timed path needs only one CUDA graph
// launch.  Consecutive graph nodes provide the required device-wide Jacobi
// barrier without constraining the grid to cooperatively resident blocks.
double runSimulation(DeviceMesh& device, const int n_iters) {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_instance = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    if (n_iters > 0) {
        for (int iter = 0; iter < n_iters; ++iter) {
            const int read_buffer = iter & 1;
            launchSimulationIteration(device, device.dynamic[read_buffer],
                                      device.dynamic[read_buffer ^ 1], stream);
        }
    } else {
        emptySimulationKernel<<<1, 1, 0, stream>>>();
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));
    // Upload performs graph setup before the measured region without executing
    // any node or changing the initial simulation state.
    CUDA_CHECK(cudaGraphUpload(graph_instance, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaGraphLaunch(graph_instance, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float duration_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaGraphExecDestroy(graph_instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return duration_ms;
}

void downloadSimulation(World& world, const DeviceMesh& device, int n_iters) {
    const int final_buffer = n_iters > 0 && (n_iters & 1) ? 1 : 0;
    if (device.n_elems != 0) {
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(),
                              device.dynamic[final_buffer],
                              device.n_elems * sizeof(ElementDynamic),
                              cudaMemcpyDeviceToHost));
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
    
    // Prepare CUDA-resident mesh and state outside the measured compute region.
    DeviceMesh device;
    prepareDeviceMesh(world, device);

    // Run simulation
    printf("Running simulation...\n");
    const double duration_ms = runSimulation(device, n_iters);
    downloadSimulation(world, device, n_iters);

    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec =
        (static_cast<double>(n_measured_iters) * n_elems) /
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
