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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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

// Compute energy flux between two elements.
// Same operation order as the original host code. The explicit round-to-nearest
// intrinsics prevent nvcc from contracting these operations into FMA, so the
// results are bit-identical to the sequential reference.
__device__ __forceinline__ val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    const val_t diff = __dsub_rn(other_energy, this_energy);
    return __dmul_rn(__dmul_rn(__dmul_rn(diff, mat.transfer_coeff), connection_flux), 0.25);
}

// One simulation step: each element gathers flux from its neighbours.
// Element i reads only from `in` and writes only to `out`, so all elements
// are independent and run fully in parallel.
// Connectivity is structure-of-arrays with slot-major layout (index j * n_elems + i),
// so threads in a warp read consecutive addresses for each connection slot.
__global__ void stepKernel(const Material* __restrict__ materials,
                           const uint32_t* __restrict__ mat_idx,
                           const uint8_t* __restrict__ num_conn,
                           const uint32_t* __restrict__ conn_idx,
                           const val_t* __restrict__ conn_flux,
                           const size_t n_elems,
                           const ElementDynamic* __restrict__ in,
                           ElementDynamic* __restrict__ out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += stride) {
        const Material mat = materials[mat_idx[i]];
        const val_t self_energy = in[i].current_energy;
        const val_t self_flux = in[i].total_flux;

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        const size_t nc = num_conn[i];
        for (size_t j = 0; j < nc; ++j) {
            const size_t slot = j * n_elems + i;
            const uint32_t neighbor_idx = conn_idx[slot];
            total_flux = __dadd_rn(total_flux,
                                   computeFlux(mat, self_energy, conn_flux[slot],
                                               in[neighbor_idx].current_energy));
        }

        // Update element state
        out[i].current_energy = __dadd_rn(self_energy, total_flux);
        out[i].total_flux = __dadd_rn(self_flux, fabs(total_flux));
    }
}

// One-time conversion of the static mesh (verbatim ElementStatic array) into the
// structure-of-arrays, slot-major layout read by stepKernel
__global__ void packStaticKernel(const ElementStatic* __restrict__ src,
                                 const size_t n_elems,
                                 uint32_t* __restrict__ mat_idx,
                                 uint8_t* __restrict__ num_conn,
                                 uint32_t* __restrict__ conn_idx,
                                 val_t* __restrict__ conn_flux) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elems; i += stride) {
        const ElementStatic e = src[i];
        mat_idx[i] = static_cast<uint32_t>(e.material_idx);
        num_conn[i] = static_cast<uint8_t>(e.num_connections);
        for (size_t j = 0; j < MAX_CONNECTIONS; ++j) {
            const bool valid = j < e.num_connections;
            conn_idx[j * n_elems + i] = valid ? static_cast<uint32_t>(e.connected_idx[j]) : 0u;
            conn_flux[j * n_elems + i] = valid ? e.connected_flux[j] : 0.0;
        }
    }
}

// Force CUDA context creation so device initialization is not counted in the timed region
void initCudaDevice() {
    CUDA_CHECK(cudaFree(nullptr));
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_iters <= 0 || n_elems == 0) {
        return;
    }

    // Upload the static mesh exactly as built on the host (no host-side repacking),
    // then convert it once on the device into the SoA layout used by the step kernel.
    const size_t static_bytes = n_elems * sizeof(ElementStatic);
    const size_t dyn_bytes = n_elems * sizeof(ElementDynamic);
    const size_t mat_bytes = world.materials.size() * sizeof(Material);

    ElementStatic* d_static_raw = nullptr;
    Material* d_materials = nullptr;
    uint32_t* d_mat_idx = nullptr;
    uint8_t* d_num_conn = nullptr;
    uint32_t* d_conn_idx = nullptr;
    val_t* d_conn_flux = nullptr;
    ElementDynamic* d_dyn_a = nullptr;
    ElementDynamic* d_dyn_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_static_raw, static_bytes));
    CUDA_CHECK(cudaMalloc(&d_materials, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_mat_idx, n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_conn, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx, MAX_CONNECTIONS * n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux, MAX_CONNECTIONS * n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_dyn_a, dyn_bytes));
    CUDA_CHECK(cudaMalloc(&d_dyn_b, dyn_bytes));

    CUDA_CHECK(cudaMemcpy(d_static_raw, world.elements_static.data(), static_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(), mat_bytes, cudaMemcpyHostToDevice));
    // The swap buffer is fully written by the first kernel before it is read, so only the
    // current state needs to be uploaded
    CUDA_CHECK(cudaMemcpy(d_dyn_a, world.elements_dynamic.data(), dyn_bytes, cudaMemcpyHostToDevice));

    constexpr int BLOCK_SIZE = 256;
    const size_t max_blocks = size_t{1} << 20;
    const int n_blocks = static_cast<int>(
        std::min(max_blocks, (n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE));

    packStaticKernel<<<n_blocks, BLOCK_SIZE>>>(d_static_raw, n_elems, d_mat_idx, d_num_conn,
                                               d_conn_idx, d_conn_flux);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_static_raw));

    // Ping-pong buffers, mirroring the original std::swap of the dynamic vectors
    ElementDynamic* cur = d_dyn_a;
    ElementDynamic* nxt = d_dyn_b;
    for (int iter = 0; iter < n_iters; ++iter) {
        stepKernel<<<n_blocks, BLOCK_SIZE>>>(d_materials, d_mat_idx, d_num_conn,
                                             d_conn_idx, d_conn_flux, n_elems, cur, nxt);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cur, nxt);
    }

    // Blocking copies synchronize with the kernels and surface any execution errors
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), cur, dyn_bytes, cudaMemcpyDeviceToHost));
    // The swap buffer only holds kernel output after two or more iterations
    if (n_iters >= 2) {
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic_swap.data(), nxt, dyn_bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_mat_idx));
    CUDA_CHECK(cudaFree(d_num_conn));
    CUDA_CHECK(cudaFree(d_conn_idx));
    CUDA_CHECK(cudaFree(d_conn_flux));
    CUDA_CHECK(cudaFree(d_dyn_a));
    CUDA_CHECK(cudaFree(d_dyn_b));
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
    initCudaDevice();
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
