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
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
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

// Packed per-element metadata layout (device side)
constexpr uint32_t META_NCONN_MASK = 0xF;      // bits 0-3: number of connections
constexpr uint32_t META_UNIT_FLUX = 1u << 4;   // bit 4: all connection fluxes == 1.0
constexpr int META_MAT_SHIFT = 5;              // bits 5+: material index

constexpr int BLOCK_SIZE = 256;

// Device-side material: transfer_coeff * 0.25 is precomputed when that product
// is exact, allowing (d * tc) * 0.25 to be evaluated as d * (tc * 0.25).
struct DeviceMaterial {
    val_t transfer_coeff;
    val_t external_flow;
    val_t transfer_coeff_q;  // transfer_coeff * 0.25
    int64_t q_exact;         // nonzero if transfer_coeff_q is usable
};

// Convert the AoS static element data into a compact SoA layout on the device
__global__ void packStaticKernel(const ElementStatic* __restrict__ elems, size_t n,
                                 uint32_t* __restrict__ meta,
                                 uint32_t* __restrict__ nbr,
                                 val_t* __restrict__ nbr_flux) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ElementStatic& e = elems[i];
    const uint32_t nconn = (uint32_t)e.num_connections;
    bool unit = true;
    for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nconn) {
            nbr[j * n + i] = (uint32_t)e.connected_idx[j];
            nbr_flux[j * n + i] = e.connected_flux[j];
            unit = unit && (e.connected_flux[j] == 1.0);
        }
    }
    meta[i] = nconn | (unit ? META_UNIT_FLUX : 0u) | ((uint32_t)e.material_idx << META_MAT_SHIFT);
}

// Split AoS dynamic state into SoA arrays
__global__ void unpackDynamicKernel(const ElementDynamic* __restrict__ in, size_t n,
                                    val_t* __restrict__ energy, val_t* __restrict__ tflux) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    energy[i] = in[i].current_energy;
    tflux[i] = in[i].total_flux;
}

// Merge SoA arrays back into AoS dynamic state
__global__ void packDynamicKernel(ElementDynamic* __restrict__ out, size_t n,
                                  const val_t* __restrict__ energy, const val_t* __restrict__ tflux) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i].current_energy = energy[i];
    out[i].total_flux = tflux[i];
}

// Bitwise-exact evaluation of total + ((other - mine) * tc) * 0.25 for a unit
// connection flux. If `fused`, the final scale-and-add is a single fused
// multiply-add, matching the host compiler's contraction of the scalar
// remainder iteration of the reference loop (only observable for subnormals).
// Fast path d * (tc * 0.25) is identical whenever no underflow/overflow occurs
// (scaling by a power of two commutes with rounding in the normal range).
__device__ __forceinline__ val_t addUnitFlux(val_t total, val_t d, const DeviceMaterial& mat,
                                             bool fused) {
    if (mat.q_exact) {
        const val_t y = __dmul_rn(d, mat.transfer_coeff_q);
        const uint64_t yb = (uint64_t)__double_as_longlong(y);
        const uint32_t e = (uint32_t)(yb >> 52) & 0x7FF;
        const uint64_t db = (uint64_t)__double_as_longlong(d) << 1;
        if (db == 0 || (e >= 2 && e <= 2043)) return __dadd_rn(total, y);
    }
    const val_t p = __dmul_rn(d, mat.transfer_coeff);
    return fused ? __fma_rn(p, 0.25, total) : __dadd_rn(total, __dmul_rn(p, 0.25));
}

__global__ void __launch_bounds__(BLOCK_SIZE)
updateKernel(size_t n,
             const uint32_t* __restrict__ meta,
             const uint32_t* __restrict__ nbr,
             const val_t* __restrict__ nbr_flux,
             const DeviceMaterial* __restrict__ materials,
             const val_t* __restrict__ energy_in,
             const val_t* __restrict__ tflux_in,
             val_t* __restrict__ energy_out,
             val_t* __restrict__ tflux_out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    const uint32_t m = __ldg(&meta[i]);
    const uint32_t nconn = m & META_NCONN_MASK;
    const DeviceMaterial mat = materials[m >> META_MAT_SHIFT];
    const val_t my_energy = __ldg(&energy_in[i]);

    // Start with external flow
    val_t total_flux = mat.external_flow;

    if (m & META_UNIT_FLUX) {
        // Multiplication by a connection flux of 1.0 is exact, so it is skipped
#pragma unroll
        for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j < nconn) {
                const val_t other = __ldg(&energy_in[__ldg(&nbr[j * n + i])]);
                total_flux = addUnitFlux(total_flux, __dsub_rn(other, my_energy), mat,
                                         (nconn & 1) && j == nconn - 1);
            }
        }
    } else {
#pragma unroll
        for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j < nconn) {
                const val_t other = __ldg(&energy_in[__ldg(&nbr[j * n + i])]);
                const val_t p = __dmul_rn(__dmul_rn(__dsub_rn(other, my_energy), mat.transfer_coeff),
                                          __ldg(&nbr_flux[j * n + i]));
                total_flux = ((nconn & 1) && j == nconn - 1)
                                 ? __fma_rn(p, 0.25, total_flux)
                                 : __dadd_rn(total_flux, __dmul_rn(p, 0.25));
            }
        }
    }

    energy_out[i] = __dadd_rn(my_energy, total_flux);
    tflux_out[i] = __dadd_rn(__ldg(&tflux_in[i]), fabs(total_flux));
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0) return;
    const size_t n_mats = world.materials.size();

    // Device materials with precomputed quarter transfer coefficients
    std::vector<DeviceMaterial> h_mats(n_mats);
    for (size_t m = 0; m < n_mats; ++m) {
        const val_t tc = world.materials[m].transfer_coeff;
        const val_t q = tc * 0.25;
        h_mats[m].transfer_coeff = tc;
        h_mats[m].external_flow = world.materials[m].external_flow;
        h_mats[m].transfer_coeff_q = q;
        // Exact iff q is a normal number (no subnormal rounding) and tc is finite
        h_mats[m].q_exact = std::isfinite(tc) && std::fpclassify(q) == FP_NORMAL;
    }

    ElementStatic* d_static = nullptr;
    uint32_t* d_meta = nullptr;
    uint32_t* d_nbr = nullptr;
    val_t* d_nbr_flux = nullptr;
    DeviceMaterial* d_mats = nullptr;
    ElementDynamic* d_aos = nullptr;
    val_t* d_dyn = nullptr;  // energy[2][n], tflux[2][n]

    CUDA_CHECK(cudaMalloc(&d_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_meta, n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nbr, n_elems * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nbr_flux, n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mats, n_mats * sizeof(DeviceMaterial)));
    CUDA_CHECK(cudaMalloc(&d_aos, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_dyn, 4 * n_elems * sizeof(val_t)));

    val_t* energy[2] = {d_dyn, d_dyn + n_elems};
    val_t* tflux[2] = {d_dyn + 2 * n_elems, d_dyn + 3 * n_elems};
    const unsigned int grid = (unsigned int)((n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Upload state and convert to SoA on the device
    CUDA_CHECK(cudaMemcpy(d_mats, h_mats.data(), n_mats * sizeof(DeviceMaterial),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aos, world.elements_dynamic.data(),
                          n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    unpackDynamicKernel<<<grid, BLOCK_SIZE>>>(d_aos, n_elems, energy[0], tflux[0]);
    CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data(),
                          n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    packStaticKernel<<<grid, BLOCK_SIZE>>>(d_static, n_elems, d_meta, d_nbr, d_nbr_flux);
    CUDA_CHECK(cudaGetLastError());

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        updateKernel<<<grid, BLOCK_SIZE>>>(n_elems, d_meta, d_nbr, d_nbr_flux, d_mats,
                                           energy[cur], tflux[cur],
                                           energy[cur ^ 1], tflux[cur ^ 1]);
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    // Convert back to AoS and download final state
    packDynamicKernel<<<grid, BLOCK_SIZE>>>(d_aos, n_elems, energy[cur], tflux[cur]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_aos,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_meta));
    CUDA_CHECK(cudaFree(d_nbr));
    CUDA_CHECK(cudaFree(d_nbr_flux));
    CUDA_CHECK(cudaFree(d_mats));
    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_dyn));
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
    
    // Initialize the CUDA context up front (outside the timed region)
    CUDA_CHECK(cudaFree(0));

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
