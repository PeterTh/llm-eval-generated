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

[[noreturn]] void cudaFail(cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFail(error, operation);
}

// Materials are tiny, read-only, and shared by every thread. Constant memory
// avoids an extra global-memory indirection in the inner loop.
__constant__ Material device_materials[3];

// Connectivity is stored in ELLPACK order on the device: connection j for
// consecutive elements is consecutive in memory. This makes every load made
// by a warp coalesced while retaining the original per-element connection order.
__global__ __launch_bounds__(256)
void updateElements(const uint8_t* __restrict__ material_idx,
                    const uint8_t* __restrict__ num_connections,
                    const uint32_t* __restrict__ connected_idx,
                    const val_t* __restrict__ connected_flux,
                    const val_t* __restrict__ current_energy,
                    const val_t* __restrict__ total_flux_in,
                    val_t* __restrict__ next_energy,
                    val_t* __restrict__ total_flux_out,
                    size_t n_elems) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = device_materials[material_idx[i]];
    const val_t energy = current_energy[i];
    val_t flux = mat.external_flow;

#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j >= num_connections[i]) break;
        const size_t connection = static_cast<size_t>(j) * n_elems + i;
        const uint32_t neighbor = connected_idx[connection];
        flux += (current_energy[neighbor] - energy) * mat.transfer_coeff *
                connected_flux[connection] * 0.25;
    }

    next_energy[i] = energy + flux;
    total_flux_out[i] = total_flux_in[i] + fabs(flux);
}

class CudaSimulation {
public:
    explicit CudaSimulation(const World& world) : n_elems_(world.elements_static.size()) {
        // Initialize the CUDA context before benchmark timing begins.
        cudaCheck(cudaFree(nullptr), "CUDA context initialization");
        if (world.materials.size() != 3)
            cudaFail(cudaErrorInvalidValue, "material table size");
        cudaCheck(cudaMemcpyToSymbol(device_materials, world.materials.data(),
                                     3 * sizeof(Material)), "material upload");

        size_t connection_width = 0;
        for (const ElementStatic& elem : world.elements_static)
            connection_width = std::max(connection_width,
                                        static_cast<size_t>(elem.num_connections));

        std::vector<uint8_t> material_idx(n_elems_);
        std::vector<uint8_t> num_connections(n_elems_);
        std::vector<uint32_t> connected_idx(n_elems_ * connection_width);
        std::vector<val_t> connected_flux(n_elems_ * connection_width);
        std::vector<val_t> energy(n_elems_);
        std::vector<val_t> flux(n_elems_);

        for (size_t i = 0; i < n_elems_; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            if (elem.material_idx >= world.materials.size() ||
                elem.num_connections > MAX_CONNECTIONS)
                cudaFail(cudaErrorInvalidValue, "mesh connectivity");
            material_idx[i] = static_cast<uint8_t>(elem.material_idx);
            num_connections[i] = static_cast<uint8_t>(elem.num_connections);
            energy[i] = world.elements_dynamic[i].current_energy;
            flux[i] = world.elements_dynamic[i].total_flux;
            for (size_t j = 0; j < elem.num_connections; ++j) {
                const size_t dst = j * n_elems_ + i;
                connected_idx[dst] = static_cast<uint32_t>(elem.connected_idx[j]);
                connected_flux[dst] = elem.connected_flux[j];
            }
        }

        allocate(reinterpret_cast<void**>(&d_material_idx_), n_elems_ * sizeof(uint8_t));
        allocate(reinterpret_cast<void**>(&d_num_connections_), n_elems_ * sizeof(uint8_t));
        allocate(reinterpret_cast<void**>(&d_connected_idx_), connected_idx.size() * sizeof(uint32_t));
        allocate(reinterpret_cast<void**>(&d_connected_flux_), connected_flux.size() * sizeof(val_t));
        allocate(reinterpret_cast<void**>(&d_energy_[0]), n_elems_ * sizeof(val_t));
        allocate(reinterpret_cast<void**>(&d_energy_[1]), n_elems_ * sizeof(val_t));
        allocate(reinterpret_cast<void**>(&d_flux_[0]), n_elems_ * sizeof(val_t));
        allocate(reinterpret_cast<void**>(&d_flux_[1]), n_elems_ * sizeof(val_t));

        upload(d_material_idx_, material_idx.data(), material_idx.size() * sizeof(uint8_t));
        upload(d_num_connections_, num_connections.data(), num_connections.size() * sizeof(uint8_t));
        upload(d_connected_idx_, connected_idx.data(), connected_idx.size() * sizeof(uint32_t));
        upload(d_connected_flux_, connected_flux.data(), connected_flux.size() * sizeof(val_t));
        upload(d_energy_[0], energy.data(), energy.size() * sizeof(val_t));
        upload(d_flux_[0], flux.data(), flux.size() * sizeof(val_t));
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (stream_) cudaStreamDestroy(stream_);
        cudaFree(d_material_idx_);
        cudaFree(d_num_connections_);
        cudaFree(d_connected_idx_);
        cudaFree(d_connected_flux_);
        cudaFree(d_energy_[0]);
        cudaFree(d_energy_[1]);
        cudaFree(d_flux_[0]);
        cudaFree(d_flux_[1]);
    }

    // Build the complete dependent iteration chain before timing. A CUDA graph
    // reduces launch overhead to one submission, which matters for smaller
    // meshes while preserving a global synchronization between iterations.
    void prepare(int n_iters) {
        if (n_iters == 0) return;
        cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "simulation stream creation");
        cudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                  "CUDA graph capture start");
        constexpr int block_size = 256;
        const unsigned int blocks = static_cast<unsigned int>((n_elems_ + block_size - 1) / block_size);
        for (int iter = 0; iter < n_iters; ++iter) {
            updateElements<<<blocks, block_size, 0, stream_>>>(
                d_material_idx_, d_num_connections_, d_connected_idx_, d_connected_flux_,
                d_energy_[active_], d_flux_[active_], d_energy_[active_ ^ 1],
                d_flux_[active_ ^ 1], n_elems_);
            active_ ^= 1;
        }
        cudaCheck(cudaGetLastError(), "simulation graph kernel capture");
        cudaCheck(cudaStreamEndCapture(stream_, &graph_), "CUDA graph capture end");
        cudaCheck(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                  "CUDA graph instantiation");
    }

    void run() {
        if (!graph_exec_) return;
        cudaCheck(cudaGraphLaunch(graph_exec_, stream_), "simulation graph launch");
        cudaCheck(cudaStreamSynchronize(stream_), "simulation graph execution");
    }

    void download(World& world) const {
        std::vector<val_t> energy(n_elems_);
        std::vector<val_t> flux(n_elems_);
        cudaCheck(cudaMemcpy(energy.data(), d_energy_[active_], n_elems_ * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "energy download");
        cudaCheck(cudaMemcpy(flux.data(), d_flux_[active_], n_elems_ * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "flux download");
        for (size_t i = 0; i < n_elems_; ++i) {
            world.elements_dynamic[i].current_energy = energy[i];
            world.elements_dynamic[i].total_flux = flux[i];
        }
    }

private:
    void allocate(void** pointer, size_t bytes) {
        cudaCheck(cudaMalloc(pointer, bytes), "device allocation");
    }

    void upload(void* device, const void* host, size_t bytes) {
        cudaCheck(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "mesh upload");
    }

    size_t n_elems_;
    int active_ = 0;
    uint8_t* d_material_idx_ = nullptr;
    uint8_t* d_num_connections_ = nullptr;
    uint32_t* d_connected_idx_ = nullptr;
    val_t* d_connected_flux_ = nullptr;
    val_t* d_energy_[2] = {nullptr, nullptr};
    val_t* d_flux_[2] = {nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

// Run simulation using CUDA unconditionally.
void runSimulation(CudaSimulation& simulation) {
    simulation.run();
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

    // Element counts and connectivity use 32-bit indices on the GPU (and the
    // original builder uses int), so reject inputs that cannot be represented.
    if (n_elems_root <= 0 || n_elems_root > 46340 || n_iters < 0) {
        std::fprintf(stderr, "Grid size must be in [1, 46340] and iterations must be non-negative.\n");
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
    
    // Device allocation, layout conversion, and upload are setup costs rather
    // than simulation work, so keep them outside the measured interval.
    CudaSimulation simulation(world);
    simulation.prepare(n_iters);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(simulation);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Result transfer is also excluded from kernel timing.
    simulation.download(world);
    
    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) /
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
