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

// Device-side data pointers
struct DeviceWorld {
    Material* d_materials = nullptr;
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ============================================================
// Device world singleton (forward declarations)
// ============================================================
static DeviceWorld g_deviceWorld;
static bool g_deviceInitialized = false;

static DeviceWorld& getDeviceWorld() { return g_deviceWorld; }

// ============================================================
// CUDA Kernels
// ============================================================

// Kernel: build connectivity for one element
__global__ void buildConnectivityKernel(ElementStatic* elems, int n_elems_root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_elems = n_elems_root * n_elems_root;
    if (idx >= n_elems) return;

    int x = idx / n_elems_root;
    int y = idx % n_elems_root;

    elems[idx].material_idx = DEFAULT_MAT_ID;
    elems[idx].num_connections = 0;
    for (int c = 0; c < MAX_CONNECTIONS; ++c) {
        elems[idx].connected_idx[c] = 0;
        elems[idx].connected_flux[c] = 0.0;
    }

    // Connect to neighbors (right, left, up, down)
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    for (int n = 0; n < 4; ++n) {
        int nx = x + dx[n];
        int ny = y + dy[n];
        if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
            int nc = (int)elems[idx].num_connections;
            elems[idx].connected_idx[nc] = (idx_t)(nx * n_elems_root + ny);
            elems[idx].connected_flux[nc] = 1.0;
            elems[idx].num_connections++;
        }
    }
}

// Kernel: set corner materials
__global__ void setCornersKernel(ElementStatic* elems, int n_elems_root) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int last = n_elems_root - 1;
        elems[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        elems[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
        elems[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        elems[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Kernel: initialize dynamic elements to zero
__global__ void initDynamicKernel(ElementDynamic* dyn, size_t n_elems) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elems) {
        dyn[idx].current_energy = 0.0;
        dyn[idx].total_flux = 0.0;
    }
}

// Kernel: one simulation step — each thread updates one element
__global__ void simulateStepKernel(const ElementStatic* __restrict__ elems_static,
                                    const ElementDynamic* __restrict__ elems_read,
                                    ElementDynamic* __restrict__ elems_write,
                                    const Material* __restrict__ materials,
                                    size_t n_elems) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const ElementStatic& es = elems_static[i];
    const ElementDynamic& ed = elems_read[i];
    const Material& mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    for (idx_t j = 0; j < es.num_connections; ++j) {
        idx_t nidx = es.connected_idx[j];
        val_t n_energy = elems_read[nidx].current_energy;
        total_flux += (n_energy - ed.current_energy) * mat.transfer_coeff * es.connected_flux[j] * 0.25;
    }

    elems_write[i].current_energy = ed.current_energy + total_flux;
    elems_write[i].total_flux = ed.total_flux + fabs(total_flux);
}

// Kernel: compute hash per block, XOR-reduce within block
__global__ void hashKernel(const ElementDynamic* __restrict__ elems,
                           size_t n_elems,
                           uint64_t* block_hashes) {
    extern __shared__ unsigned char hash_smem_raw[];
    uint64_t* hash_smem = reinterpret_cast<uint64_t*>(hash_smem_raw);

    uint64_t local_hash = 0;

    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems;
         i += stride) {
        const uint64_t e_val = *reinterpret_cast<const uint64_t*>(&elems[i].current_energy);
        const uint64_t f_val = *reinterpret_cast<const uint64_t*>(&elems[i].total_flux);
        local_hash ^= (e_val + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f_val + i) * 0xbf58476d1ce4e5b9ULL;
    }

    hash_smem[threadIdx.x] = local_hash;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            hash_smem[threadIdx.x] ^= hash_smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_hashes[blockIdx.x] = hash_smem[0];
    }
}

// Kernel: validation reduction — sum energy, sum flux, min/max energy
__global__ void validateKernel(const ElementDynamic* __restrict__ elems,
                               size_t n_elems,
                               val_t* d_energy_sum, val_t* d_flux_sum,
                               val_t* d_energy_max, val_t* d_energy_min) {
    extern __shared__ unsigned char validate_smem_raw[];
    val_t* shared_mem = reinterpret_cast<val_t*>(validate_smem_raw);

    val_t l_energy_sum = 0.0;
    val_t l_flux_sum = 0.0;
    val_t l_energy_max = -HUGE_VAL;
    val_t l_energy_min = HUGE_VAL;

    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems;
         i += stride) {
        l_energy_sum += elems[i].current_energy;
        l_flux_sum += elems[i].total_flux;
        l_energy_max = fmax(l_energy_max, elems[i].current_energy);
        l_energy_min = fmin(l_energy_min, elems[i].current_energy);
    }

    // Store into shared memory
    shared_mem[threadIdx.x] = l_energy_sum;
    shared_mem[blockDim.x + threadIdx.x] = l_flux_sum;
    shared_mem[2 * blockDim.x + threadIdx.x] = l_energy_max;
    shared_mem[3 * blockDim.x + threadIdx.x] = l_energy_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
            shared_mem[blockDim.x + threadIdx.x] += shared_mem[blockDim.x + threadIdx.x + s];
            shared_mem[2 * blockDim.x + threadIdx.x] = fmax(
                shared_mem[2 * blockDim.x + threadIdx.x],
                shared_mem[2 * blockDim.x + threadIdx.x + s]);
            shared_mem[3 * blockDim.x + threadIdx.x] = fmin(
                shared_mem[3 * blockDim.x + threadIdx.x],
                shared_mem[3 * blockDim.x + threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_energy_sum, shared_mem[0]);
        atomicAdd(d_flux_sum, shared_mem[blockDim.x]);
        // atomicMax/Min for double via atomicCAS
        val_t old = *d_energy_max, val = shared_mem[2 * blockDim.x];
        while (val > old) {
            unsigned long long assumed = atomicCAS((unsigned long long*)d_energy_max,
                                                   __double_as_longlong(old),
                                                   __double_as_longlong(val));
            old = __longlong_as_double(assumed);
        }
        old = *d_energy_min;
        val = shared_mem[3 * blockDim.x];
        while (val < old) {
            unsigned long long assumed = atomicCAS((unsigned long long*)d_energy_min,
                                                   __double_as_longlong(old),
                                                   __double_as_longlong(val));
            old = __longlong_as_double(assumed);
        }
    }
}

// ============================================================
// Host functions
// ============================================================

void initDeviceWorld(World& world) {
    if (g_deviceInitialized) return;
    g_deviceInitialized = true;

    const size_t n_elems = world.elements_static.size();
    DeviceWorld& dw = g_deviceWorld;

    cudaMalloc(&dw.d_materials, world.materials.size() * sizeof(Material));
    cudaMemcpy(dw.d_materials, world.materials.data(),
               world.materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    cudaMalloc(&dw.d_elements_static, n_elems * sizeof(ElementStatic));
    cudaMemcpy(dw.d_elements_static, world.elements_static.data(),
               n_elems * sizeof(ElementStatic),
               cudaMemcpyHostToDevice);

    cudaMalloc(&dw.d_elements_dynamic, n_elems * sizeof(ElementDynamic));
    cudaMalloc(&dw.d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic));
}

void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize materials on host
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialize device memory
    initDeviceWorld(world);
    DeviceWorld& d_world = getDeviceWorld();

    int blockSize = 256;
    int numBlocks = (n_elems + blockSize - 1) / blockSize;

    // Build connectivity on GPU
    buildConnectivityKernel<<<numBlocks, blockSize>>>(
        d_world.d_elements_static, n_elems_root);

    setCornersKernel<<<1, 1>>>(d_world.d_elements_static, n_elems_root);

    // Copy static data back to host
    cudaMemcpy(world.elements_static.data(),
               d_world.d_elements_static,
               n_elems * sizeof(ElementStatic),
               cudaMemcpyDeviceToHost);

    // Initialize dynamic data on GPU
    initDynamicKernel<<<numBlocks, blockSize>>>(d_world.d_elements_dynamic, n_elems);
    initDynamicKernel<<<numBlocks, blockSize>>>(d_world.d_elements_dynamic_swap, n_elems);
}

void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    DeviceWorld& dw = getDeviceWorld();

    int blockSize = 256;
    int numBlocks = (int)((n_elems + blockSize - 1) / blockSize);

    for (int iter = 0; iter < n_iters; ++iter) {
        simulateStepKernel<<<numBlocks, blockSize>>>(
            dw.d_elements_static,
            dw.d_elements_dynamic,
            dw.d_elements_dynamic_swap,
            dw.d_materials,
            n_elems);

        // Swap device pointers
        std::swap(dw.d_elements_dynamic, dw.d_elements_dynamic_swap);
    }

    // Copy results back to host
    cudaMemcpy(world.elements_dynamic.data(),
               dw.d_elements_dynamic,
               n_elems * sizeof(ElementDynamic),
               cudaMemcpyDeviceToHost);
}

bool validateResults(const World& world) {
    const size_t n_elems = world.elements_dynamic.size();

    val_t h_energy_sum = 0.0;
    val_t h_flux_sum = 0.0;
    val_t h_energy_max = -HUGE_VAL;
    val_t h_energy_min = HUGE_VAL;

    val_t *d_energy_sum, *d_flux_sum, *d_energy_max, *d_energy_min;
    cudaMalloc(&d_energy_sum, sizeof(val_t));
    cudaMalloc(&d_flux_sum, sizeof(val_t));
    cudaMalloc(&d_energy_max, sizeof(val_t));
    cudaMalloc(&d_energy_min, sizeof(val_t));

    cudaMemcpy(d_energy_sum, &h_energy_sum, sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flux_sum, &h_flux_sum, sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy_max, &h_energy_max, sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy_min, &h_energy_min, sizeof(val_t), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = std::min((int)((n_elems + blockSize - 1) / blockSize), 256);
    size_t shared_mem_size = 4 * blockSize * sizeof(val_t);

    validateKernel<<<numBlocks, blockSize, shared_mem_size>>>(
        world.elements_dynamic.data(), n_elems,
        d_energy_sum, d_flux_sum, d_energy_max, d_energy_min);

    cudaMemcpy(&h_energy_sum, d_energy_sum, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_flux_sum, d_flux_sum, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_max, d_energy_max, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_min, d_energy_min, sizeof(val_t), cudaMemcpyDeviceToHost);

    cudaFree(d_energy_sum);
    cudaFree(d_flux_sum);
    cudaFree(d_energy_max);
    cudaFree(d_energy_min);

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", h_energy_sum);
    printf("  Flux sum: %.2f\n", h_flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", h_energy_min, h_energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(h_energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(h_energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }

    if (!std::isfinite(h_flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(h_energy_max) || !std::isfinite(h_energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    const size_t n_elems = elements.size();
    if (n_elems == 0) return 0;

    int blockSize = 256;
    int numBlocks = std::min((int)((n_elems + blockSize - 1) / blockSize), 512);

    uint64_t* d_block_hashes = nullptr;
    cudaMalloc(&d_block_hashes, numBlocks * sizeof(uint64_t));

    size_t shared_mem_size = blockSize * sizeof(uint64_t);
    hashKernel<<<numBlocks, blockSize, shared_mem_size>>>(
        elements.data(), n_elems, d_block_hashes);

    std::vector<uint64_t> h_block_hashes(numBlocks, 0);
    cudaMemcpy(h_block_hashes.data(), d_block_hashes,
               numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_block_hashes);

    uint64_t hash = 0;
    for (int i = 0; i < numBlocks; ++i) {
        hash ^= h_block_hashes[i];
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
