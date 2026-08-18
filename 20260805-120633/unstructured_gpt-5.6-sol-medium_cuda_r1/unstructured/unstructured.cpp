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

[[noreturn]] void cudaFailure(cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(error, operation);
}

// Connectivity is transposed on the device.  Threads in a warp consequently
// read adjacent entries for each connection instead of striding through the
// large host-side ElementStatic structure.
__global__ void updateElements(
    size_t n_elems,
    const uint8_t* __restrict__ material_idx,
    const uint8_t* __restrict__ num_connections,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ accumulated_flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ accumulated_flux_out) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < n_elems; i += stride) {
        const Material mat = device_materials[material_idx[i]];
        const val_t own_energy = energy_in[i];
        val_t total_flux = mat.external_flow;
        const int count = num_connections[i];

#pragma unroll
        for (int connection = 0; connection < MAX_CONNECTIONS; ++connection) {
            if (connection < count) {
                const size_t offset = static_cast<size_t>(connection) * n_elems + i;
                const val_t neighbor_energy = energy_in[connected_idx[offset]];
                total_flux += (neighbor_energy - own_energy) * mat.transfer_coeff *
                              connected_flux[offset] * 0.25;
            }
        }

        energy_out[i] = own_energy + total_flux;
        accumulated_flux_out[i] = accumulated_flux_in[i] + fabs(total_flux);
    }
}

class CudaSimulation {
public:
    explicit CudaSimulation(const World& world) : n_elems_(world.elements_static.size()) {
        size_t max_connections = 0;
        for (const ElementStatic& elem : world.elements_static) {
            max_connections = std::max(max_connections,
                                       static_cast<size_t>(elem.num_connections));
        }
        std::vector<uint8_t> material_idx(n_elems_);
        std::vector<uint8_t> num_connections(n_elems_);
        std::vector<uint32_t> connected_idx(max_connections * n_elems_);
        std::vector<val_t> connected_flux(max_connections * n_elems_);
        std::vector<val_t> energy(n_elems_);
        std::vector<val_t> accumulated_flux(n_elems_);

        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            material_idx[i] = static_cast<uint8_t>(elem.material_idx);
            num_connections[i] = static_cast<uint8_t>(elem.num_connections);
            energy[i] = world.elements_dynamic[i].current_energy;
            accumulated_flux[i] = world.elements_dynamic[i].total_flux;
            for (size_t connection = 0; connection < elem.num_connections; ++connection) {
                const size_t offset = connection * n_elems_ + i;
                connected_idx[offset] = static_cast<uint32_t>(elem.connected_idx[connection]);
                connected_flux[offset] = elem.connected_flux[connection];
            }
        }

        allocate(reinterpret_cast<void**>(&material_idx_), n_elems_ * sizeof(*material_idx_));
        allocate(reinterpret_cast<void**>(&num_connections_), n_elems_ * sizeof(*num_connections_));
        allocate(reinterpret_cast<void**>(&connected_idx_), connected_idx.size() * sizeof(*connected_idx_));
        allocate(reinterpret_cast<void**>(&connected_flux_), connected_flux.size() * sizeof(*connected_flux_));
        for (int buffer = 0; buffer < 2; ++buffer) {
            allocate(reinterpret_cast<void**>(&energy_[buffer]), n_elems_ * sizeof(*energy_[buffer]));
            allocate(reinterpret_cast<void**>(&accumulated_flux_[buffer]),
                     n_elems_ * sizeof(*accumulated_flux_[buffer]));
        }

        copyToDevice(material_idx_, material_idx.data(), material_idx.size() * sizeof(material_idx[0]));
        copyToDevice(num_connections_, num_connections.data(),
                     num_connections.size() * sizeof(num_connections[0]));
        copyToDevice(connected_idx_, connected_idx.data(),
                     connected_idx.size() * sizeof(connected_idx[0]));
        copyToDevice(connected_flux_, connected_flux.data(),
                     connected_flux.size() * sizeof(connected_flux[0]));
        copyToDevice(energy_[0], energy.data(), energy.size() * sizeof(energy[0]));
        copyToDevice(accumulated_flux_[0], accumulated_flux.data(),
                     accumulated_flux.size() * sizeof(accumulated_flux[0]));
        cudaCheck(cudaMemcpyToSymbol(device_materials, world.materials.data(),
                                    world.materials.size() * sizeof(Material)),
                  "copying materials");

        cudaDeviceProp properties{};
        int device = 0;
        cudaCheck(cudaGetDevice(&device), "querying the active device");
        cudaCheck(cudaGetDeviceProperties(&properties, device), "querying device properties");
        cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "creating simulation stream");
        constexpr int threads = 256;
        const size_t needed_blocks = (n_elems_ + threads - 1) / threads;
        const size_t resident_blocks = static_cast<size_t>(properties.multiProcessorCount) * 8;
        blocks_ = static_cast<unsigned>(std::min(needed_blocks, resident_blocks));
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        cudaStreamDestroy(stream_);
        cudaFree(material_idx_);
        cudaFree(num_connections_);
        cudaFree(connected_idx_);
        cudaFree(connected_flux_);
        for (int buffer = 0; buffer < 2; ++buffer) {
            cudaFree(energy_[buffer]);
            cudaFree(accumulated_flux_[buffer]);
        }
    }

    double run(int n_iters) {
        if (n_iters <= 0) return 0.0;

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
                  "starting CUDA graph capture");
        for (int iteration = 0; iteration < n_iters; ++iteration) {
            const int input = iteration & 1;
            const int output = input ^ 1;
            updateElements<<<blocks_, 256, 0, stream_>>>(
                n_elems_, material_idx_, num_connections_, connected_idx_, connected_flux_,
                energy_[input], accumulated_flux_[input], energy_[output],
                accumulated_flux_[output]);
        }
        cudaCheck(cudaStreamEndCapture(stream_, &graph), "ending CUDA graph capture");
        cudaCheck(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
                  "instantiating CUDA graph");
        cudaCheck(cudaEventCreate(&start), "creating start event");
        cudaCheck(cudaEventCreate(&stop), "creating stop event");
        cudaCheck(cudaEventRecord(start, stream_), "recording start event");
        cudaCheck(cudaGraphLaunch(graph_exec, stream_), "launching simulation graph");
        cudaCheck(cudaEventRecord(stop, stream_), "recording stop event");
        cudaCheck(cudaEventSynchronize(stop), "waiting for simulation");

        float elapsed_ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop), "measuring simulation");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        final_buffer_ = n_iters & 1;
        return elapsed_ms;
    }

    void copyResults(World& world) const {
        std::vector<val_t> energy(n_elems_);
        std::vector<val_t> accumulated_flux(n_elems_);
        cudaCheck(cudaMemcpy(energy.data(), energy_[final_buffer_],
                             energy.size() * sizeof(energy[0]), cudaMemcpyDeviceToHost),
                  "copying final energy");
        cudaCheck(cudaMemcpy(accumulated_flux.data(), accumulated_flux_[final_buffer_],
                             accumulated_flux.size() * sizeof(accumulated_flux[0]),
                             cudaMemcpyDeviceToHost),
                  "copying final flux");
        for (size_t i = 0; i < n_elems_; ++i) {
            world.elements_dynamic[i].current_energy = energy[i];
            world.elements_dynamic[i].total_flux = accumulated_flux[i];
        }
    }

private:
    static void allocate(void** pointer, size_t bytes) {
        cudaCheck(cudaMalloc(pointer, std::max<size_t>(bytes, 1)),
                  "allocating device memory");
    }

    static void copyToDevice(void* destination, const void* source, size_t bytes) {
        cudaCheck(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                  "copying mesh to device");
    }

    size_t n_elems_;
    unsigned blocks_ = 0;
    int final_buffer_ = 0;
    cudaStream_t stream_ = nullptr;
    uint8_t* material_idx_ = nullptr;
    uint8_t* num_connections_ = nullptr;
    uint32_t* connected_idx_ = nullptr;
    val_t* connected_flux_ = nullptr;
    val_t* energy_[2] = {nullptr, nullptr};
    val_t* accumulated_flux_[2] = {nullptr, nullptr};
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
    
    // Upload and reorganize the mesh before starting the computation timer.
    CudaSimulation simulation(world);

    // Run simulation
    printf("Running simulation...\n");
    const double duration_ms = simulation.run(n_iters);
    simulation.copyResults(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec =
        duration_ms > 0.0
            ? (static_cast<double>(n_measured_iters) * n_elems) /
                  (duration_ms / 1000.0) / 1e9
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
