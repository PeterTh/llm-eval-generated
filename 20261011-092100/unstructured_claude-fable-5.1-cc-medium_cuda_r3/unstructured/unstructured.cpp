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

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Compact index type used on the device (element counts are bounded by int).
using didx_t = uint32_t;

constexpr int BLOCK_SIZE = 256;

// Structure-of-arrays view of the static mesh connectivity on the device.
// Connection arrays are stored connection-major ([max_connections][n_elems])
// so that threads of a warp handling consecutive elements access consecutive
// memory for each connection slot (fully coalesced loads). Only the number of
// connection slots actually used by the mesh (max_connections) is stored.
struct DeviceStatic {
    uint8_t* material_idx;     // [n]
    uint8_t* num_connections;  // [n]
    didx_t* connected_idx;     // [max_connections * n], padded with 0
    val_t* connected_flux;     // [max_connections * n], padded with 0.0
};

// Convert the host array-of-structures connectivity into the device SoA layout.
__global__ void convertStaticKernel(const ElementStatic* __restrict__ aos, DeviceStatic soa,
                                    const int max_connections, const didx_t n) {
    const didx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ElementStatic& e = aos[i];
    const int nc = static_cast<int>(min(e.num_connections, static_cast<idx_t>(MAX_CONNECTIONS)));
    soa.material_idx[i] = static_cast<uint8_t>(e.material_idx);
    soa.num_connections[i] = static_cast<uint8_t>(nc);
    for (int j = 0; j < max_connections; ++j) {
        const bool valid = j < nc;
        soa.connected_idx[static_cast<size_t>(j) * n + i] =
            valid ? static_cast<didx_t>(e.connected_idx[j]) : 0u;
        soa.connected_flux[static_cast<size_t>(j) * n + i] = valid ? e.connected_flux[j] : 0.0;
    }
}

// Split dynamic state (AoS) into separate energy / flux arrays.
__global__ void splitDynamicKernel(const ElementDynamic* __restrict__ aos,
                                   val_t* __restrict__ energy,
                                   val_t* __restrict__ total_flux, const didx_t n) {
    const didx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    energy[i] = aos[i].current_energy;
    total_flux[i] = aos[i].total_flux;
}

// Merge separate energy / flux arrays back into the AoS layout.
__global__ void mergeDynamicKernel(const val_t* __restrict__ energy,
                                   const val_t* __restrict__ total_flux,
                                   ElementDynamic* __restrict__ aos, const didx_t n) {
    const didx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ElementDynamic d;
    d.current_energy = energy[i];
    d.total_flux = total_flux[i];
    aos[i] = d;
}

// Compute energy flux between two elements.
// Uses explicitly rounded IEEE operations so that no FMA contraction happens:
// the result is bit-identical to the sequential reference evaluation
// ((other - this) * transfer_coeff) * connection_flux) * 0.25.
__device__ __forceinline__ val_t computeFlux(const val_t transfer_coeff,
                                             const val_t this_energy,
                                             const val_t connection_flux,
                                             const val_t other_energy) {
    const val_t d = __dsub_rn(other_energy, this_energy);
    return __dmul_rn(__dmul_rn(__dmul_rn(d, transfer_coeff), connection_flux), 0.25);
}

// One simulation step: every thread updates one element. MC is the number of
// connection slots stored for this mesh; all slots are loaded unconditionally
// (padding is harmless) so the loads are issued back-to-back, while the flux
// accumulation is predicated to keep the exact reference summation order.
template <int MC>
__global__ void __launch_bounds__(BLOCK_SIZE)
stepKernel(const DeviceStatic st, const Material* __restrict__ materials,
           const val_t* __restrict__ energy_in, const val_t* __restrict__ flux_in,
           val_t* __restrict__ energy_out, val_t* __restrict__ flux_out,
           const didx_t n) {
    const didx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Material mat = materials[st.material_idx[i]];
    const val_t this_energy = energy_in[i];
    const int nc = st.num_connections[i];

    didx_t neighbor_idx[MC > 0 ? MC : 1];
    val_t conn_flux[MC > 0 ? MC : 1];
    val_t other_energy[MC > 0 ? MC : 1];
#pragma unroll
    for (int j = 0; j < MC; ++j) {
        const size_t slot = static_cast<size_t>(j) * n + i;
        neighbor_idx[j] = st.connected_idx[slot];
        conn_flux[j] = st.connected_flux[slot];
    }
#pragma unroll
    for (int j = 0; j < MC; ++j) {
        other_energy[j] = __ldg(&energy_in[neighbor_idx[j]]);
    }

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements, in the same order as the reference
#pragma unroll
    for (int j = 0; j < MC; ++j) {
        if (j < nc) {
            total_flux = __dadd_rn(
                total_flux, computeFlux(mat.transfer_coeff, this_energy, conn_flux[j], other_energy[j]));
        }
    }

    // Update element state
    energy_out[i] = __dadd_rn(this_energy, total_flux);
    flux_out[i] = __dadd_rn(flux_in[i], fabs(total_flux));
}

using StepFn = void (*)(const DeviceStatic, const Material*, const val_t*, const val_t*, val_t*,
                        val_t*, const didx_t);

static StepFn selectStepKernel(const int max_connections) {
    static_assert(MAX_CONNECTIONS == 8, "update the kernel selection table");
    switch (max_connections) {
        case 0: return stepKernel<0>;
        case 1: return stepKernel<1>;
        case 2: return stepKernel<2>;
        case 3: return stepKernel<3>;
        case 4: return stepKernel<4>;
        case 5: return stepKernel<5>;
        case 6: return stepKernel<6>;
        case 7: return stepKernel<7>;
        default: return stepKernel<8>;
    }
}

// Initialize the CUDA runtime (context creation) and report the device used.
void initCuda() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));  // force context creation now, outside the timed region
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    // Make sure all kernels are loaded now (CUDA 12 loads modules lazily on
    // first launch, which would otherwise land inside the timed region).
    cudaFuncAttributes attr;
    for (int mc = 0; mc <= MAX_CONNECTIONS; ++mc) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(selectStepKernel(mc))));
    }
    CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(convertStaticKernel)));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(splitDynamicKernel)));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(mergeDynamicKernel)));
}

static inline unsigned int numBlocks(const size_t n) {
    return static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0 || n_iters <= 0) return;
    if (n_elems > std::numeric_limits<didx_t>::max()) {
        fprintf(stderr, "Too many elements for device index type\n");
        exit(EXIT_FAILURE);
    }
    const didx_t n = static_cast<didx_t>(n_elems);
    const unsigned int blocks = numBlocks(n_elems);

    // Number of connection slots actually used by this mesh
    idx_t max_nc = 0;
    for (const ElementStatic& e : world.elements_static) max_nc = std::max(max_nc, e.num_connections);
    const int max_connections = static_cast<int>(std::min<idx_t>(max_nc, MAX_CONNECTIONS));

    // --- One device allocation holding every buffer ---
    const size_t n_mat = world.materials.size();
    auto align = [](size_t x) { return (x + 255) & ~size_t(255); };
    const size_t off_static_aos = 0;
    const size_t off_dyn_aos = align(off_static_aos + n_elems * sizeof(ElementStatic));
    const size_t off_materials = align(off_dyn_aos + n_elems * sizeof(ElementDynamic));
    const size_t off_material_idx = align(off_materials + n_mat * sizeof(Material));
    const size_t off_nconn = align(off_material_idx + n_elems * sizeof(uint8_t));
    const size_t off_cidx = align(off_nconn + n_elems * sizeof(uint8_t));
    const size_t off_cflux = align(off_cidx + max_connections * n_elems * sizeof(didx_t));
    const size_t off_dyn = align(off_cflux + max_connections * n_elems * sizeof(val_t));
    const size_t dyn_bytes = align(n_elems * sizeof(val_t));
    const size_t total_bytes = off_dyn + 4 * dyn_bytes;

    unsigned char* d_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, total_bytes));

    ElementStatic* d_static_aos = reinterpret_cast<ElementStatic*>(d_base + off_static_aos);
    ElementDynamic* d_dyn_aos = reinterpret_cast<ElementDynamic*>(d_base + off_dyn_aos);
    Material* d_materials = reinterpret_cast<Material*>(d_base + off_materials);
    DeviceStatic st;
    st.material_idx = reinterpret_cast<uint8_t*>(d_base + off_material_idx);
    st.num_connections = reinterpret_cast<uint8_t*>(d_base + off_nconn);
    st.connected_idx = reinterpret_cast<didx_t*>(d_base + off_cidx);
    st.connected_flux = reinterpret_cast<val_t*>(d_base + off_cflux);
    val_t* d_energy[2] = {reinterpret_cast<val_t*>(d_base + off_dyn + 0 * dyn_bytes),
                          reinterpret_cast<val_t*>(d_base + off_dyn + 1 * dyn_bytes)};
    val_t* d_flux[2] = {reinterpret_cast<val_t*>(d_base + off_dyn + 2 * dyn_bytes),
                        reinterpret_cast<val_t*>(d_base + off_dyn + 3 * dyn_bytes)};

    // --- Upload mesh and convert to SoA on the device ---
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(), n_mat * sizeof(Material),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dyn_aos, world.elements_dynamic.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    splitDynamicKernel<<<blocks, BLOCK_SIZE>>>(d_dyn_aos, d_energy[0], d_flux[0], n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_static_aos, world.elements_static.data(),
                          n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    convertStaticKernel<<<blocks, BLOCK_SIZE>>>(d_static_aos, st, max_connections, n);
    CUDA_CHECK(cudaGetLastError());

    // --- Time stepping (ping-pong between the two dynamic state buffers) ---
    const StepFn step = selectStepKernel(max_connections);
    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        step<<<blocks, BLOCK_SIZE>>>(st, d_materials, d_energy[cur], d_flux[cur], d_energy[nxt],
                                     d_flux[nxt], n);
        cur = nxt;
    }
    CUDA_CHECK(cudaGetLastError());

    // --- Download final state ---
    mergeDynamicKernel<<<blocks, BLOCK_SIZE>>>(d_energy[cur], d_flux[cur], d_dyn_aos, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_dyn_aos,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_base));
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
    initCuda();
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
