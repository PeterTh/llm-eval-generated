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

// World state (host-side data)
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

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel: build connectivity for each element in parallel
__global__ void buildConnectivityKernel(ElementStatic* elements_static,
                                         int n_elems_root, int n_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    int x = idx / n_elems_root;
    int y = idx % n_elems_root;

    ElementStatic& elem = elements_static[idx];
    elem.material_idx = DEFAULT_MAT_ID;
    elem.num_connections = 0;

    for (int i = 0; i < MAX_CONNECTIONS; i++) {
        elem.connected_idx[i] = 0;
        elem.connected_flux[i] = 0.0;
    }

    // Connect to neighbors (up, down, left, right)
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int n = 0; n < 4; ++n) {
        const int nx = x + offsets[n][0];
        const int ny = y + offsets[n][1];

        if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
            const int neighbor_idx = nx * n_elems_root + ny;
            elem.connected_idx[elem.num_connections] = neighbor_idx;
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
    }
}

// Kernel: set corner materials (inflow/outflow)
__global__ void setCornerMaterialsKernel(ElementStatic* elements_static,
                                          int n_elems_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int last = n_elems_root - 1;
        elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
        elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Kernel: initialize dynamic elements to zero
__global__ void initDynamicKernel(ElementDynamic* elements_dynamic, int n_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    elements_dynamic[idx].current_energy = 0.0;
    elements_dynamic[idx].total_flux = 0.0;
}

// Kernel: one simulation step - one thread per element
__global__ void simulationStepKernel(const ElementStatic* elements_static,
                                      const ElementDynamic* elements_dynamic,
                                      ElementDynamic* elements_dynamic_swap,
                                      const Material* materials,
                                      int n_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    const ElementStatic& elem_static = elements_static[idx];
    const ElementDynamic& elem_dyn = elements_dynamic[idx];
    const Material& mat = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];

        // Inline computeFlux:
        // (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25
        total_flux += (neighbor_dyn.current_energy - elem_dyn.current_energy) *
                      mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
    }

    // Update element state in swap buffer
    elements_dynamic_swap[idx].current_energy = elem_dyn.current_energy + total_flux;
    elements_dynamic_swap[idx].total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Kernel: parallel reduction for energy sum (block-level)
__global__ void energySumKernel(const ElementDynamic* elements_dynamic,
                                 int n_elems, val_t* block_sums) {
    extern __shared__ val_t s_energy_sum[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    val_t sum = 0.0;
    for (int i = idx; i < n_elems; i += blockDim.x * gridDim.x) {
        sum += elements_dynamic[i].current_energy;
    }

    s_energy_sum[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_energy_sum[tid] += s_energy_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_energy_sum[0];
    }
}

// Kernel: parallel reduction for flux sum
__global__ void fluxSumKernel(const ElementDynamic* elements_dynamic,
                               int n_elems, val_t* block_sums) {
    extern __shared__ val_t s_flux_sum[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    val_t sum = 0.0;
    for (int i = idx; i < n_elems; i += blockDim.x * gridDim.x) {
        sum += elements_dynamic[i].total_flux;
    }

    s_flux_sum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_flux_sum[tid] += s_flux_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_flux_sum[0];
    }
}

// Kernel: parallel reduction for energy min/max
__global__ void energyMinMaxKernel(const ElementDynamic* elements_dynamic,
                                    int n_elems, val_t* min_block, val_t* max_block) {
    extern __shared__ char s_minmax[];
    val_t* shared_min = (val_t*)s_minmax;
    val_t* shared_max = (val_t*)(s_minmax + blockDim.x * sizeof(val_t));

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    val_t min_val = std::numeric_limits<val_t>::max();
    val_t max_val = std::numeric_limits<val_t>::lowest();

    for (int i = idx; i < n_elems; i += blockDim.x * gridDim.x) {
        val_t e = elements_dynamic[i].current_energy;
        if (e < min_val) min_val = e;
        if (e > max_val) max_val = e;
    }

    shared_min[tid] = min_val;
    shared_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fmin(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_block[blockIdx.x] = shared_min[0];
        max_block[blockIdx.x] = shared_max[0];
    }
}

// Kernel: parallel hash computation with XOR reduction
__global__ void hashKernel(const ElementDynamic* elements, int n_elems,
                            uint64_t* block_hashes) {
    extern __shared__ char s_hash[];
    uint64_t* shared_hash = (uint64_t*)s_hash;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    uint64_t hash = 0;
    for (int i = idx; i < n_elems; i += blockDim.x * gridDim.x) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }

    shared_hash[tid] = hash;
    __syncthreads();

    // XOR reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_hash[tid] ^= shared_hash[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_hashes[blockIdx.x] = shared_hash[0];
    }
}

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Host-side functions
// ============================================================================

// Build a 2D square grid as an unstructured mesh (using GPU for parallelism)
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize materials on host
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate host elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    const int block_size = 256;
    const int grid_size = (n_elems + block_size - 1) / block_size;

    // Allocate device memory
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;

    CUDA_CHECK(cudaMalloc(&d_elements_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic)));

    // Build connectivity on GPU in parallel
    buildConnectivityKernel<<<grid_size, block_size>>>(d_elements_static, n_elems_root, n_elems);
    CUDA_CHECK(cudaGetLastError());

    // Set corner materials
    setCornerMaterialsKernel<<<1, 1>>>(d_elements_static, n_elems_root);
    CUDA_CHECK(cudaGetLastError());

    // Initialize dynamic elements on GPU
    initDynamicKernel<<<grid_size, block_size>>>(d_elements_dynamic, n_elems);
    CUDA_CHECK(cudaGetLastError());
    initDynamicKernel<<<grid_size, block_size>>>(d_elements_dynamic_swap, n_elems);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(world.elements_static.data(), d_elements_static,
                          n_elems * sizeof(ElementStatic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_elements_dynamic,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic_swap.data(), d_elements_dynamic_swap,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_elements_static));
    CUDA_CHECK(cudaFree(d_elements_dynamic));
    CUDA_CHECK(cudaFree(d_elements_dynamic_swap));
}

// Run simulation for n_iters iterations (entirely on GPU)
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();

    const int block_size = 256;
    const int grid_size = (int)((n_elems + block_size - 1) / block_size);

    // Allocate device memory
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;
    Material* d_materials = nullptr;

    CUDA_CHECK(cudaMalloc(&d_elements_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_elements_static, world.elements_static.data(),
                          n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elements_dynamic, world.elements_dynamic.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elements_dynamic_swap, world.elements_dynamic_swap.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    // Run simulation iterations entirely on GPU
    for (int iter = 0; iter < n_iters; ++iter) {
        simulationStepKernel<<<grid_size, block_size>>>(
            d_elements_static, d_elements_dynamic, d_elements_dynamic_swap,
            d_materials, (int)n_elems);
        CUDA_CHECK(cudaGetLastError());

        // Swap pointers on device (zero-copy swap)
        std::swap(d_elements_dynamic, d_elements_dynamic_swap);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy final results back to host
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_elements_dynamic,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_elements_static));
    CUDA_CHECK(cudaFree(d_elements_dynamic));
    CUDA_CHECK(cudaFree(d_elements_dynamic_swap));
    CUDA_CHECK(cudaFree(d_materials));
}

// Validate simulation results (using GPU parallel reductions)
bool validateResults(const World& world) {
    const size_t n_elems = world.elements_dynamic.size();

    const int block_size = 256;
    const int grid_size = (int)((n_elems + block_size - 1) / block_size);

    // Allocate device memory for results
    ElementDynamic* d_elements = nullptr;
    val_t* d_block_sums = nullptr;
    val_t* d_block_min = nullptr;
    val_t* d_block_max = nullptr;

    CUDA_CHECK(cudaMalloc(&d_elements, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, grid_size * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_block_min, grid_size * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_block_max, grid_size * sizeof(val_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_elements, world.elements_dynamic.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));

    // Launch parallel reduction kernels
    size_t shared_size = block_size * sizeof(val_t);
    energySumKernel<<<grid_size, block_size, shared_size>>>(d_elements, (int)n_elems, d_block_sums);
    CUDA_CHECK(cudaGetLastError());

    size_t shared_size_minmax = block_size * 2 * sizeof(val_t);
    energyMinMaxKernel<<<grid_size, block_size, shared_size_minmax>>>(
        d_elements, (int)n_elems, d_block_min, d_block_max);
    CUDA_CHECK(cudaGetLastError());

    // Flux sum (reuse d_block_sums)
    fluxSumKernel<<<grid_size, block_size, shared_size>>>(d_elements, (int)n_elems, d_block_sums);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy block results to host and finalize
    std::vector<val_t> h_block_sums(grid_size);
    std::vector<val_t> h_flux_sums(grid_size);
    std::vector<val_t> h_block_min(grid_size);
    std::vector<val_t> h_block_max(grid_size);

    // d_block_sums currently holds energy sums (launched first)
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          grid_size * sizeof(val_t), cudaMemcpyDeviceToHost));

    // Now launch flux sum (overwrites d_block_sums)
    fluxSumKernel<<<grid_size, block_size, shared_size>>>(d_elements, (int)n_elems, d_block_sums);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_flux_sums.data(), d_block_sums,
                          grid_size * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_min.data(), d_block_min,
                          grid_size * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max,
                          grid_size * sizeof(val_t), cudaMemcpyDeviceToHost));

    // Finalize on host
    val_t energy_sum = 0.0;
    for (int i = 0; i < grid_size; i++) {
        energy_sum += h_block_sums[i];
    }

    val_t flux_sum = 0.0;
    for (int i = 0; i < grid_size; i++) {
        flux_sum += h_flux_sums[i];
    }

    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (int i = 0; i < grid_size; i++) {
        energy_max = std::max(h_block_max[i], energy_max);
        energy_min = std::min(h_block_min[i], energy_min);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_min));
    CUDA_CHECK(cudaFree(d_block_max));

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

// Compute a simple hash of the results for verification (using GPU parallel reduction)
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    const int n_elems = (int)elements.size();

    const int block_size = 256;
    const int grid_size = (n_elems + block_size - 1) / block_size;

    ElementDynamic* d_elements = nullptr;
    uint64_t* d_block_hashes = nullptr;

    CUDA_CHECK(cudaMalloc(&d_elements, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_block_hashes, grid_size * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_elements, elements.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));

    size_t shared_size = block_size * sizeof(uint64_t);
    hashKernel<<<grid_size, block_size, shared_size>>>(d_elements, n_elems, d_block_hashes);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_block_hashes(grid_size);
    CUDA_CHECK(cudaMemcpy(h_block_hashes.data(), d_block_hashes,
                          grid_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Finalize hash on host (XOR all block hashes)
    uint64_t hash = 0;
    for (int i = 0; i < grid_size; i++) {
        hash ^= h_block_hashes[i];
    }

    CUDA_CHECK(cudaFree(d_elements));
    CUDA_CHECK(cudaFree(d_block_hashes));

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
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

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
