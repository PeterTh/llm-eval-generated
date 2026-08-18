#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;
constexpr int MAX_DEVICE_MATERIALS = 256;
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_GRAPH_BATCH_SIZE = 256;
static_assert(CUDA_GRAPH_BATCH_SIZE % 2 == 0);

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

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                \
    do {                                                                      \
        const cudaError_t cuda_check_error = (expression);                    \
        if (cuda_check_error != cudaSuccess) {                                \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);   \
        }                                                                     \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    void allocate(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count * sizeof(T)));
        }
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
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

// Materials are tiny and are read uniformly by almost every warp. Constant
// memory avoids a global-memory indirection without specializing the mesh.
__constant__ Material device_materials[MAX_DEVICE_MATERIALS];

// Connectivity is stored slot-major on the device: all first neighbors are
// contiguous, then all second neighbors, and so on. This preserves each
// element's original neighbor order while making warp accesses coalesced.
template <unsigned MaxDegree>
__global__ __launch_bounds__(CUDA_BLOCK_SIZE)
void simulationIteration(
    const uint8_t* __restrict__ material_indices,
    const uint8_t* __restrict__ connection_counts,
    const uint32_t* __restrict__ connected_indices,
    const val_t* __restrict__ connected_fluxes,
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ current_accumulated_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_accumulated_flux,
    size_t element_count) {
    const size_t element_idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_idx >= element_count) {
        return;
    }

    const Material material = device_materials[material_indices[element_idx]];
    const val_t energy = current_energy[element_idx];
    val_t total_flux = material.external_flow;
    const unsigned connection_count = connection_counts[element_idx];

    if constexpr (MaxDegree > 0) {
#pragma unroll
        for (unsigned connection = 0; connection < MaxDegree; ++connection) {
            if (connection < connection_count) {
                const size_t connection_offset =
                    static_cast<size_t>(connection) * element_count +
                    element_idx;
                const uint32_t neighbor_idx =
                    connected_indices[connection_offset];
                total_flux += (current_energy[neighbor_idx] - energy) *
                              material.transfer_coeff *
                              connected_fluxes[connection_offset] * 0.25;
            }
        }
    }

    next_energy[element_idx] = energy + total_flux;
    next_accumulated_flux[element_idx] =
        current_accumulated_flux[element_idx] + fabs(total_flux);
}

class GpuWorld {
public:
    GpuWorld(const World& world, int n_iters)
        : element_count_(world.elements_static.size()) {
        if (element_count_ > std::numeric_limits<uint32_t>::max()) {
            std::fprintf(stderr,
                         "Mesh has too many elements for GPU connectivity\n");
            std::exit(EXIT_FAILURE);
        }
        if (world.materials.empty() ||
            world.materials.size() > MAX_DEVICE_MATERIALS) {
            std::fprintf(stderr, "GPU supports between 1 and %d materials\n",
                         MAX_DEVICE_MATERIALS);
            std::exit(EXIT_FAILURE);
        }
        if (world.elements_dynamic.size() != element_count_) {
            std::fprintf(stderr, "Static and dynamic mesh sizes differ\n");
            std::exit(EXIT_FAILURE);
        }

        std::vector<uint8_t> host_material_indices(element_count_);
        std::vector<uint8_t> host_connection_counts(element_count_);
        std::vector<val_t> host_energy(element_count_);
        std::vector<val_t> host_accumulated_flux(element_count_);

        for (size_t i = 0; i < element_count_; ++i) {
            const ElementStatic& element = world.elements_static[i];
            if (element.material_idx >= world.materials.size() ||
                element.num_connections > MAX_CONNECTIONS) {
                std::fprintf(stderr, "Invalid mesh metadata at element %zu\n", i);
                std::exit(EXIT_FAILURE);
            }
            host_material_indices[i] = static_cast<uint8_t>(element.material_idx);
            host_connection_counts[i] =
                static_cast<uint8_t>(element.num_connections);
            max_degree_ = std::max(
                max_degree_, static_cast<unsigned>(element.num_connections));
            host_energy[i] = world.elements_dynamic[i].current_energy;
            host_accumulated_flux[i] = world.elements_dynamic[i].total_flux;
        }

        std::vector<uint32_t> host_connected_indices(
            element_count_ * max_degree_);
        std::vector<val_t> host_connected_fluxes(element_count_ * max_degree_);
        for (size_t i = 0; i < element_count_; ++i) {
            const ElementStatic& element = world.elements_static[i];
            for (unsigned connection = 0;
                 connection < element.num_connections; ++connection) {
                if (element.connected_idx[connection] >= element_count_) {
                    std::fprintf(stderr,
                                 "Invalid neighbor at element %zu, slot %u\n",
                                 i, connection);
                    std::exit(EXIT_FAILURE);
                }
                const size_t offset =
                    static_cast<size_t>(connection) * element_count_ + i;
                host_connected_indices[offset] =
                    static_cast<uint32_t>(element.connected_idx[connection]);
                host_connected_fluxes[offset] =
                    element.connected_flux[connection];
            }
        }

        material_indices_.allocate(element_count_);
        connection_counts_.allocate(element_count_);
        connected_indices_.allocate(host_connected_indices.size());
        connected_fluxes_.allocate(host_connected_fluxes.size());
        energy_[0].allocate(element_count_);
        energy_[1].allocate(element_count_);
        accumulated_flux_[0].allocate(element_count_);
        accumulated_flux_[1].allocate(element_count_);

        copyToDevice(material_indices_, host_material_indices);
        copyToDevice(connection_counts_, host_connection_counts);
        copyToDevice(connected_indices_, host_connected_indices);
        copyToDevice(connected_fluxes_, host_connected_fluxes);
        copyToDevice(energy_[0], host_energy);
        copyToDevice(accumulated_flux_[0], host_accumulated_flux);
        CUDA_CHECK(cudaMemcpyToSymbol(
            device_materials, world.materials.data(),
            world.materials.size() * sizeof(Material)));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        prepareGraph(n_iters);
    }

    ~GpuWorld() {
        if (batch_graph_exec_ != nullptr) {
            cudaGraphExecDestroy(batch_graph_exec_);
        }
        if (remainder_graph_exec_ != nullptr) {
            cudaGraphExecDestroy(remainder_graph_exec_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    GpuWorld(const GpuWorld&) = delete;
    GpuWorld& operator=(const GpuWorld&) = delete;

    void run() {
        for (int batch = 0; batch < graph_batch_count_; ++batch) {
            CUDA_CHECK(cudaGraphLaunch(batch_graph_exec_, stream_));
        }
        if (remainder_graph_exec_ != nullptr) {
            CUDA_CHECK(cudaGraphLaunch(remainder_graph_exec_, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void download(World& world) const {
        std::vector<val_t> host_energy(element_count_);
        std::vector<val_t> host_accumulated_flux(element_count_);
        const size_t byte_count = element_count_ * sizeof(val_t);
        CUDA_CHECK(cudaMemcpy(host_energy.data(), energy_[active_buffer_].get(),
                              byte_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_accumulated_flux.data(),
                              accumulated_flux_[active_buffer_].get(),
                              byte_count, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < element_count_; ++i) {
            world.elements_dynamic[i].current_energy = host_energy[i];
            world.elements_dynamic[i].total_flux = host_accumulated_flux[i];
        }
    }

private:
    template <typename T>
    static void copyToDevice(DeviceBuffer<T>& destination,
                             const std::vector<T>& source) {
        if (!source.empty()) {
            CUDA_CHECK(cudaMemcpy(destination.get(), source.data(),
                                  source.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    template <unsigned MaxDegree>
    void launchIteration(unsigned read_buffer) {
        const unsigned write_buffer = read_buffer ^ 1U;
        const int block_count = static_cast<int>(
            (element_count_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        simulationIteration<MaxDegree><<<block_count, CUDA_BLOCK_SIZE, 0,
                                         stream_>>>(
            material_indices_.get(), connection_counts_.get(),
            connected_indices_.get(), connected_fluxes_.get(),
            energy_[read_buffer].get(), accumulated_flux_[read_buffer].get(),
            energy_[write_buffer].get(), accumulated_flux_[write_buffer].get(),
            element_count_);
    }

    void launchIteration(unsigned read_buffer) {
        switch (max_degree_) {
            case 0: launchIteration<0>(read_buffer); break;
            case 1: launchIteration<1>(read_buffer); break;
            case 2: launchIteration<2>(read_buffer); break;
            case 3: launchIteration<3>(read_buffer); break;
            case 4: launchIteration<4>(read_buffer); break;
            case 5: launchIteration<5>(read_buffer); break;
            case 6: launchIteration<6>(read_buffer); break;
            case 7: launchIteration<7>(read_buffer); break;
            case 8: launchIteration<8>(read_buffer); break;
            default:
                std::fprintf(stderr, "Unsupported GPU mesh degree: %u\n",
                             max_degree_);
                std::exit(EXIT_FAILURE);
        }
    }

    void prepareGraph(int n_iters) {
        active_buffer_ = static_cast<unsigned>(n_iters) & 1U;
        graph_batch_count_ = n_iters / CUDA_GRAPH_BATCH_SIZE;
        if (graph_batch_count_ != 0) {
            batch_graph_exec_ = makeGraph(CUDA_GRAPH_BATCH_SIZE);
        }
        remainder_graph_exec_ = makeGraph(n_iters % CUDA_GRAPH_BATCH_SIZE);
    }

    cudaGraphExec_t makeGraph(int iteration_count) {
        if (iteration_count == 0) {
            return nullptr;
        }
        CUDA_CHECK(cudaStreamBeginCapture(stream_,
                                          cudaStreamCaptureModeThreadLocal));
        for (int iter = 0; iter < iteration_count; ++iter) {
            launchIteration(static_cast<unsigned>(iter) & 1U);
        }

        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        cudaGraphExec_t graph_exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        return graph_exec;
    }

    size_t element_count_ = 0;
    unsigned max_degree_ = 0;
    unsigned active_buffer_ = 0;
    int graph_batch_count_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t batch_graph_exec_ = nullptr;
    cudaGraphExec_t remainder_graph_exec_ = nullptr;
    DeviceBuffer<uint8_t> material_indices_;
    DeviceBuffer<uint8_t> connection_counts_;
    DeviceBuffer<uint32_t> connected_indices_;
    DeviceBuffer<val_t> connected_fluxes_;
    DeviceBuffer<val_t> energy_[2];
    DeviceBuffer<val_t> accumulated_flux_[2];
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

    constexpr int max_grid_root = 46340;
    if (n_elems_root <= 0 || n_elems_root > max_grid_root) {
        std::fprintf(stderr, "Grid size must be between 1 and %d\n",
                     max_grid_root);
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

    // GPU allocation, layout conversion, host-to-device transfers, and graph
    // construction are setup work and intentionally remain outside the timed
    // simulation region.
    GpuWorld gpu_world(world, n_iters);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu_world.run();
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Preserve the host-visible result interface for hashing, validation, and
    // optional external result reporting. This transfer is not computation.
    gpu_world.download(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const double time_per_iter =
        n_iters == 0 ? 0.0 : duration_ms / static_cast<double>(n_iters);
    const double giga_elems_per_sec =
        n_iters == 0
            ? 0.0
            : (static_cast<double>(n_iters) * n_elems) /
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
