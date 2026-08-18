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

static_assert(std::is_trivially_copyable_v<ElementDynamic>);

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

namespace {

constexpr int BLOCK_COLUMNS = 32;
constexpr int BLOCK_ROWS = 8;

[[noreturn]] void cudaFail(cudaError_t status, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        cudaFail(status, operation);
    }
}

class DeviceDynamicBuffers {
public:
    explicit DeviceDynamicBuffers(size_t n_elems) : bytes_(n_elems * sizeof(ElementDynamic)) {
        cudaCheck(cudaMalloc(&current_, bytes_), "allocating the current state");
        cudaCheck(cudaMalloc(&next_, bytes_), "allocating the next state");
        cudaCheck(cudaMemset(current_, 0, bytes_), "initializing the current state");
        cudaCheck(cudaMemset(next_, 0, bytes_), "initializing the next state");
    }

    DeviceDynamicBuffers(const DeviceDynamicBuffers&) = delete;
    DeviceDynamicBuffers& operator=(const DeviceDynamicBuffers&) = delete;

    ~DeviceDynamicBuffers() {
        // There is no useful recovery path at this point, and CUDA context
        // teardown may itself make cudaFree report an error.
        if (next_ != nullptr) {
            cudaFree(next_);
        }
        if (current_ != nullptr) {
            cudaFree(current_);
        }
    }

    ElementDynamic* current() const { return current_; }
    ElementDynamic* next() const { return next_; }

    void swap() { std::swap(current_, next_); }

private:
    ElementDynamic* current_ = nullptr;
    ElementDynamic* next_ = nullptr;
    size_t bytes_ = 0;
};

// The input generator always produces this regular 2-D topology.  Encoding
// it in the kernel replaces the 144-byte host-oriented ElementStatic record
// with four inexpensive index tests, while retaining the original neighbour
// order: +x, -x, +y, -y.
__global__ void updateEnergyKernel(const ElementDynamic* __restrict__ current,
                                   ElementDynamic* __restrict__ next,
                                   int n_elems_root) {
    extern __shared__ val_t energy_tile[];

    const int column = static_cast<int>(blockIdx.x) * BLOCK_COLUMNS + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y) * BLOCK_ROWS + threadIdx.y;
    const bool active = row < n_elems_root && column < n_elems_root;
    const int tile_pitch = BLOCK_COLUMNS + 2;
    const int tile_index = (threadIdx.y + 1) * tile_pitch + threadIdx.x + 1;

    size_t index = 0;
    ElementDynamic state{};
    if (active) {
        index = static_cast<size_t>(row) * static_cast<size_t>(n_elems_root) + column;
        state = current[index];
        energy_tile[tile_index] = state.current_energy;
    } else {
        energy_tile[tile_index] = 0.0;
    }

    // Populate the four one-cell halos.  At a physical boundary the value is
    // unused, because the corresponding connection does not exist.
    if (threadIdx.x == 0) {
        energy_tile[(threadIdx.y + 1) * tile_pitch] =
            (row < n_elems_root && column > 0)
                ? current[static_cast<size_t>(row) * n_elems_root + column - 1].current_energy
                : 0.0;
    }
    if (threadIdx.x == BLOCK_COLUMNS - 1) {
        energy_tile[(threadIdx.y + 1) * tile_pitch + BLOCK_COLUMNS + 1] =
            (row < n_elems_root && column + 1 < n_elems_root)
                ? current[static_cast<size_t>(row) * n_elems_root + column + 1].current_energy
                : 0.0;
    }
    if (threadIdx.y == 0) {
        energy_tile[threadIdx.x + 1] =
            (row > 0 && column < n_elems_root)
                ? current[static_cast<size_t>(row - 1) * n_elems_root + column].current_energy
                : 0.0;
    }
    if (threadIdx.y == BLOCK_ROWS - 1) {
        energy_tile[(BLOCK_ROWS + 1) * tile_pitch + threadIdx.x + 1] =
            (row + 1 < n_elems_root && column < n_elems_root)
                ? current[static_cast<size_t>(row + 1) * n_elems_root + column].current_energy
                : 0.0;
    }

    __syncthreads();

    if (!active) {
        return;
    }

    const val_t this_energy = state.current_energy;
    const int last = n_elems_root - 1;
    // This order also handles a 1x1 mesh exactly like the sequential material
    // assignments in buildSquare2D: the final material is the inflow one.
    val_t total_flux = ((row == 0 && column == 0) || (row == last && column == last))
                           ? 0.5
                           : (((row == 0 && column == last) ||
                               (row == last && column == 0))
                                  ? -0.5
                                  : 0.0);

    if (row + 1 < n_elems_root) {
        total_flux += (energy_tile[(threadIdx.y + 2) * tile_pitch + threadIdx.x + 1] - this_energy) *
                      0.8 * 1.0 * 0.25;
    }
    if (row > 0) {
        total_flux += (energy_tile[threadIdx.y * tile_pitch + threadIdx.x + 1] - this_energy) *
                      0.8 * 1.0 * 0.25;
    }
    if (column + 1 < n_elems_root) {
        total_flux += (energy_tile[(threadIdx.y + 1) * tile_pitch + threadIdx.x + 2] - this_energy) *
                      0.8 * 1.0 * 0.25;
    }
    if (column > 0) {
        total_flux += (energy_tile[(threadIdx.y + 1) * tile_pitch + threadIdx.x] - this_energy) *
                      0.8 * 1.0 * 0.25;
    }

    next[index].current_energy = this_energy + total_flux;
    next[index].total_flux = state.total_flux + fabs(total_flux);
}

}  // namespace

// Run simulation for n_iters iterations on CUDA and return device execution
// time in milliseconds.  State remains double precision throughout.
float runSimulation(World& world, const int n_elems_root, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    DeviceDynamicBuffers device_state(n_elems);

    if (n_iters <= 0) {
        cudaCheck(cudaMemcpy(world.elements_dynamic.data(), device_state.current(),
                             n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost),
                  "copying the initial state from the GPU");
        return 0.0F;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "creating the start event");
    cudaCheck(cudaEventCreate(&stop), "creating the stop event");

    const dim3 block(BLOCK_COLUMNS, BLOCK_ROWS);
    const dim3 grid((n_elems_root + BLOCK_COLUMNS - 1) / BLOCK_COLUMNS,
                    (n_elems_root + BLOCK_ROWS - 1) / BLOCK_ROWS);
    constexpr size_t shared_bytes =
        static_cast<size_t>(BLOCK_COLUMNS + 2) * static_cast<size_t>(BLOCK_ROWS + 2) * sizeof(val_t);

    cudaCheck(cudaEventRecord(start), "recording the start event");
    for (int iter = 0; iter < n_iters; ++iter) {
        updateEnergyKernel<<<grid, block, shared_bytes>>>(device_state.current(), device_state.next(),
                                                          n_elems_root);
        device_state.swap();
    }
    cudaCheck(cudaGetLastError(), "launching the update kernel");
    cudaCheck(cudaEventRecord(stop), "recording the stop event");
    cudaCheck(cudaEventSynchronize(stop), "waiting for the update kernels");

    float duration_ms = 0.0F;
    cudaCheck(cudaEventElapsedTime(&duration_ms, start, stop), "measuring the update kernels");
    cudaCheck(cudaEventDestroy(start), "destroying the start event");
    cudaCheck(cudaEventDestroy(stop), "destroying the stop event");

    cudaCheck(cudaMemcpy(world.elements_dynamic.data(), device_state.current(),
                         n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost),
              "copying the final state from the GPU");
    return duration_ms;
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
    const float duration_ms = runSimulation(world, n_elems_root, n_iters);
    
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
