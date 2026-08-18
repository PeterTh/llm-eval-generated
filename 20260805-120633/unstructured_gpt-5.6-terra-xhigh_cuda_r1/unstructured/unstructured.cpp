#include <algorithm>
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
constexpr size_t NUM_MATERIALS = 3;

// The material table is tiny and read-only throughout the simulation.  Keeping
// it in constant memory avoids a global-memory load for every element update.
__constant__ Material device_materials[NUM_MATERIALS];

[[noreturn]] void cudaCheckFailure(cudaError_t error, const char* operation,
                                   const char* file, int line) {
    fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
            operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                     \
    do {                                                                          \
        const cudaError_t cuda_status = (operation);                              \
        if (cuda_status != cudaSuccess) {                                         \
            cudaCheckFailure(cuda_status, #operation, __FILE__, __LINE__);        \
        }                                                                         \
    } while (false)

// ELLPACK stores the jth connection for every element contiguously.  A warp
// therefore gathers its connectivity coalesced even though the mesh itself is
// unstructured.  The dynamic fields use a structure-of-arrays layout for the
// same reason.
__global__ void updateElementsKernel(const idx_t* __restrict__ material_idx,
                                     const idx_t* __restrict__ num_connections,
                                     const idx_t* __restrict__ connected_idx,
                                     const val_t* __restrict__ connected_flux,
                                     const val_t* __restrict__ energy_read,
                                     const val_t* __restrict__ total_flux_read,
                                     val_t* __restrict__ energy_write,
                                     val_t* __restrict__ total_flux_write,
                                     const size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const Material material = device_materials[material_idx[i]];
    const val_t current_energy = energy_read[i];
    val_t total_flux = material.external_flow;
    const idx_t connection_count = num_connections[i];

#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (static_cast<idx_t>(j) < connection_count) {
            const size_t offset = static_cast<size_t>(j) * n_elems + i;
            const idx_t neighbor_idx = connected_idx[offset];
            total_flux += (energy_read[neighbor_idx] - current_energy) *
                          material.transfer_coeff * connected_flux[offset] * 0.25;
        }
    }

    energy_write[i] = current_energy + total_flux;
    total_flux_write[i] = total_flux_read[i] + fabs(total_flux);
}

class CudaSimulation {
public:
    explicit CudaSimulation(const World& world) : n_elems_(world.elements_static.size()) {
        if (world.materials.size() != NUM_MATERIALS) {
            fprintf(stderr, "CUDA simulation requires exactly %zu materials\n", NUM_MATERIALS);
            std::exit(EXIT_FAILURE);
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
        CUDA_CHECK(cudaMemcpyToSymbol(device_materials, world.materials.data(),
                                      NUM_MATERIALS * sizeof(Material)));

        if (n_elems_ == 0) {
            return;
        }

        std::vector<idx_t> host_material_idx(n_elems_);
        std::vector<idx_t> host_num_connections(n_elems_);
        std::vector<idx_t> host_connected_idx(n_elems_ * MAX_CONNECTIONS);
        std::vector<val_t> host_connected_flux(n_elems_ * MAX_CONNECTIONS);

        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& element = world.elements_static[i];
            host_material_idx[i] = element.material_idx;
            host_num_connections[i] = element.num_connections;
            for (int j = 0; j < MAX_CONNECTIONS; ++j) {
                const size_t offset = static_cast<size_t>(j) * n_elems_ + i;
                host_connected_idx[offset] = element.connected_idx[j];
                host_connected_flux[offset] = element.connected_flux[j];
            }
        }

        const size_t element_bytes = n_elems_ * sizeof(val_t);
        const size_t index_bytes = n_elems_ * sizeof(idx_t);
        const size_t connections_bytes = n_elems_ * MAX_CONNECTIONS * sizeof(idx_t);
        const size_t flux_bytes = n_elems_ * MAX_CONNECTIONS * sizeof(val_t);

        CUDA_CHECK(cudaMalloc(&material_idx_, index_bytes));
        CUDA_CHECK(cudaMalloc(&num_connections_, index_bytes));
        CUDA_CHECK(cudaMalloc(&connected_idx_, connections_bytes));
        CUDA_CHECK(cudaMalloc(&connected_flux_, flux_bytes));
        CUDA_CHECK(cudaMalloc(&energy_[0], element_bytes));
        CUDA_CHECK(cudaMalloc(&energy_[1], element_bytes));
        CUDA_CHECK(cudaMalloc(&total_flux_[0], element_bytes));
        CUDA_CHECK(cudaMalloc(&total_flux_[1], element_bytes));

        CUDA_CHECK(cudaMemcpy(material_idx_, host_material_idx.data(), index_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(num_connections_, host_num_connections.data(), index_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_idx_, host_connected_idx.data(), connections_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(connected_flux_, host_connected_flux.data(), flux_bytes,
                              cudaMemcpyHostToDevice));
        // ElementDynamic is an array-of-structs on the host.  These 2-D
        // transfers extract each member into the GPU's contiguous SoA arrays.
        CUDA_CHECK(cudaMemcpy2D(energy_[0], sizeof(val_t),
                                reinterpret_cast<const unsigned char*>(world.elements_dynamic.data()) +
                                    offsetof(ElementDynamic, current_energy),
                                sizeof(ElementDynamic), sizeof(val_t), n_elems_,
                                cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy2D(total_flux_[0], sizeof(val_t),
                                reinterpret_cast<const unsigned char*>(world.elements_dynamic.data()) +
                                    offsetof(ElementDynamic, total_flux),
                                sizeof(ElementDynamic), sizeof(val_t), n_elems_,
                                cudaMemcpyHostToDevice));

        threads_per_block_ = 256;
        blocks_ = static_cast<unsigned int>((n_elems_ + threads_per_block_ - 1) /
                                            threads_per_block_);
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
        if (graph_ != nullptr) cudaGraphDestroy(graph_);
        if (start_event_ != nullptr) cudaEventDestroy(start_event_);
        if (stop_event_ != nullptr) cudaEventDestroy(stop_event_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
        cudaFree(material_idx_);
        cudaFree(num_connections_);
        cudaFree(connected_idx_);
        cudaFree(connected_flux_);
        cudaFree(energy_[0]);
        cudaFree(energy_[1]);
        cudaFree(total_flux_[0]);
        cudaFree(total_flux_[1]);
    }

    double run(const int n_iters) {
        if (n_elems_ == 0 || n_iters <= 0) {
            return 0.0;
        }

        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
        for (int iter = 0; iter < n_iters; ++iter) {
            const int read_buffer = iter & 1;
            const int write_buffer = read_buffer ^ 1;
            updateElementsKernel<<<blocks_, threads_per_block_, 0, stream_>>>(
                material_idx_, num_connections_, connected_idx_, connected_flux_,
                energy_[read_buffer], total_flux_[read_buffer], energy_[write_buffer],
                total_flux_[write_buffer], n_elems_);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

        CUDA_CHECK(cudaEventRecord(start_event_, stream_));
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
        CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
        return static_cast<double>(elapsed_ms);
    }

    void copyResultsToHost(World& world, const int n_iters) const {
        if (n_elems_ == 0) {
            return;
        }

        const int result_buffer = n_iters > 0 ? (n_iters & 1) : 0;
        CUDA_CHECK(cudaMemcpy2D(world.elements_dynamic.data(), sizeof(ElementDynamic),
                                energy_[result_buffer], sizeof(val_t), sizeof(val_t), n_elems_,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(
            reinterpret_cast<unsigned char*>(world.elements_dynamic.data()) +
                offsetof(ElementDynamic, total_flux),
            sizeof(ElementDynamic), total_flux_[result_buffer], sizeof(val_t), sizeof(val_t),
            n_elems_, cudaMemcpyDeviceToHost));
    }

private:
    size_t n_elems_ = 0;
    unsigned int blocks_ = 0;
    unsigned int threads_per_block_ = 0;
    idx_t* material_idx_ = nullptr;
    idx_t* num_connections_ = nullptr;
    idx_t* connected_idx_ = nullptr;
    val_t* connected_flux_ = nullptr;
    val_t* energy_[2] = {nullptr, nullptr};
    val_t* total_flux_[2] = {nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
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

// Device allocation and mesh upload are performed before timing.  The measured
// region is exclusively the CUDA simulation graph, matching the CPU version's
// intent of timing only iterative computation.
double runSimulation(CudaSimulation& simulation, World& world, const int n_iters) {
    const double elapsed_ms = simulation.run(n_iters);
    simulation.copyResultsToHost(world, n_iters);
    return elapsed_ms;
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
    
    // Upload the fixed mesh and initial state before timing the CUDA graph.
    CudaSimulation simulation(world);

    // Run simulation
    printf("Running simulation...\n");
    const double duration_ms = runSimulation(simulation, world, n_iters);

    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = duration_ms > 0.0
        ? (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9
        : 0.0;
    
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
