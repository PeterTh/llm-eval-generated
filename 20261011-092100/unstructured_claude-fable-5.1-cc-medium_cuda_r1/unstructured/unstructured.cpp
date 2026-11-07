// Unstructured mesh energy transfer benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt). The mesh is built on the
// host exactly as in the reference implementation, then uploaded to the GPU in
// a coalescing-friendly structure-of-arrays / ELL layout. Every iteration runs
// one kernel with one thread per element; the double buffers are swapped by
// pointer on the device so no data moves between host and device during the
// timed region.

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

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Maximum number of materials that fit into constant memory. The host code
// falls back to global memory if a world ever defines more.
constexpr int MAX_CONST_MATERIALS = 64;
__constant__ Material c_materials[MAX_CONST_MATERIALS];

// Device-side mesh in structure-of-arrays form.
//
// Connectivity is stored "connection-major" (ELL layout): entry j of element i
// is at [j * n_elems + i], so consecutive threads read consecutive addresses.
// Indices are stored as 32-bit (element counts come from an int on the host).
struct DeviceMesh {
    int n_elems = 0;
    uint8_t* material_idx = nullptr;     // [n]
    uint8_t* num_connections = nullptr;  // [n]
    int32_t* connected_idx = nullptr;    // [MAX_CONNECTIONS * n]
    val_t* connected_flux = nullptr;     // [MAX_CONNECTIONS * n]
    val_t* energy_a = nullptr;           // [n]
    val_t* flux_a = nullptr;             // [n]
    val_t* energy_b = nullptr;           // [n]
    val_t* flux_b = nullptr;             // [n]
    Material* materials = nullptr;       // [n_materials] (global fallback)
    bool materials_in_const = true;
    bool unit_flux = false;              // true if every connected_flux == 1.0
};

// Compute energy flux between two elements (same expression as the reference).
__device__ __forceinline__ val_t computeFluxDev(val_t transfer_coeff, val_t this_energy,
                                                val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// MAT_CONST: materials live in __constant__ memory.
// UNIT_FLUX: every connection flux is exactly 1.0, so the (exact) multiply by
//            1.0 and its load are skipped. Results are bit-identical either way.
template <bool MAT_CONST, bool UNIT_FLUX>
__global__ void __launch_bounds__(256)
simulationStepKernel(const int n_elems,
                     const uint8_t* __restrict__ material_idx,
                     const uint8_t* __restrict__ num_connections,
                     const int32_t* __restrict__ connected_idx,
                     const val_t* __restrict__ connected_flux,
                     const Material* __restrict__ materials_global,
                     const val_t* __restrict__ energy_in,
                     const val_t* __restrict__ flux_in,
                     val_t* __restrict__ energy_out,
                     val_t* __restrict__ flux_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const int mat_id = material_idx[i];
    Material mat;
    if constexpr (MAT_CONST) {
        mat = c_materials[mat_id];
    } else {
        mat = materials_global[mat_id];
    }

    const val_t this_energy = energy_in[i];
    const int n_conn = num_connections[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements, in the same order as the reference
    // so that floating-point results are bit-identical.
#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < n_conn) {
            const size_t slot = static_cast<size_t>(j) * n_elems + i;
            const int32_t neighbor_idx = connected_idx[slot];
            const val_t other_energy = __ldg(energy_in + neighbor_idx);
            if constexpr (UNIT_FLUX) {
                total_flux += (other_energy - this_energy) * mat.transfer_coeff * 0.25;
            } else {
                const val_t cflux = connected_flux[slot];
                total_flux += computeFluxDev(mat.transfer_coeff, this_energy, cflux, other_energy);
            }
        }
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

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

// Upload the host world to the device in SoA / ELL layout.
void uploadWorld(const World& world, DeviceMesh& dev) {
    const size_t n = world.elements_static.size();
    dev.n_elems = static_cast<int>(n);
    if (n == 0) return;
    if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        fprintf(stderr, "Too many elements for 32-bit device indices\n");
        exit(1);
    }

    // Pack static data into the device layout on the host first.
    std::vector<uint8_t> h_mat(n), h_nconn(n);
    std::vector<int32_t> h_cidx(MAX_CONNECTIONS * n, 0);
    std::vector<val_t> h_cflux(MAX_CONNECTIONS * n, 0.0);
    std::vector<val_t> h_energy(n), h_flux(n);
    bool unit_flux = true;

    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_mat[i] = static_cast<uint8_t>(es.material_idx);
        h_nconn[i] = static_cast<uint8_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_cidx[j * n + i] = static_cast<int32_t>(es.connected_idx[j]);
            h_cflux[j * n + i] = es.connected_flux[j];
            unit_flux = unit_flux && (es.connected_flux[j] == 1.0);
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMalloc(&dev.material_idx, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dev.num_connections, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dev.connected_idx, MAX_CONNECTIONS * n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dev.connected_flux, MAX_CONNECTIONS * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.energy_a, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.flux_a, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.energy_b, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.flux_b, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(dev.material_idx, h_mat.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.num_connections, h_nconn.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.connected_idx, h_cidx.data(), MAX_CONNECTIONS * n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.connected_flux, h_cflux.data(), MAX_CONNECTIONS * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.energy_a, h_energy.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.flux_a, h_flux.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));

    dev.unit_flux = unit_flux;

    const size_t n_mat = world.materials.size();
    if (n_mat <= static_cast<size_t>(MAX_CONST_MATERIALS)) {
        dev.materials_in_const = true;
        CUDA_CHECK(cudaMemcpyToSymbol(c_materials, world.materials.data(), n_mat * sizeof(Material)));
    } else {
        dev.materials_in_const = false;
        CUDA_CHECK(cudaMalloc(&dev.materials, n_mat * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(dev.materials, world.materials.data(), n_mat * sizeof(Material), cudaMemcpyHostToDevice));
    }
}

// Download the current dynamic state (buffer "a" holds the latest state after
// runSimulation) back into the host world.
void downloadWorld(const DeviceMesh& dev, World& world) {
    const size_t n = world.elements_dynamic.size();
    if (n == 0) return;
    std::vector<val_t> h_energy(n), h_flux(n);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), dev.energy_a, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), dev.flux_a, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

void freeDeviceMesh(DeviceMesh& dev) {
    cudaFree(dev.material_idx);
    cudaFree(dev.num_connections);
    cudaFree(dev.connected_idx);
    cudaFree(dev.connected_flux);
    cudaFree(dev.energy_a);
    cudaFree(dev.flux_a);
    cudaFree(dev.energy_b);
    cudaFree(dev.flux_b);
    cudaFree(dev.materials);
    dev = DeviceMesh{};
}

// Run simulation for n_iters iterations on the GPU. On return, energy_a/flux_a
// hold the final state.
void runSimulation(DeviceMesh& dev, const int n_iters) {
    const int n_elems = dev.n_elems;
    if (n_elems == 0) return;

    constexpr int block = 256;
    const int grid = (n_elems + block - 1) / block;

    // Select the kernel specialization once; all iterations use it.
    using KernelFn = void (*)(int, const uint8_t*, const uint8_t*, const int32_t*,
                              const val_t*, const Material*, const val_t*,
                              const val_t*, val_t*, val_t*);
    KernelFn kernel;
    if (dev.materials_in_const) {
        kernel = dev.unit_flux ? simulationStepKernel<true, true>
                               : simulationStepKernel<true, false>;
    } else {
        kernel = dev.unit_flux ? simulationStepKernel<false, true>
                               : simulationStepKernel<false, false>;
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        kernel<<<grid, block>>>(
            n_elems, dev.material_idx, dev.num_connections, dev.connected_idx,
            dev.connected_flux, dev.materials, dev.energy_a, dev.flux_a,
            dev.energy_b, dev.flux_b);
        // Swap buffers (pointer swap, no data movement)
        std::swap(dev.energy_a, dev.energy_b);
        std::swap(dev.flux_a, dev.flux_b);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &elements[i].current_energy, sizeof(uint64_t));
        memcpy(&f_bits, &elements[i].total_flux, sizeof(uint64_t));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    
    // Upload mesh to the GPU
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // Force context creation outside the timed region
    DeviceMesh dev;
    uploadWorld(world, dev);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(dev, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Bring results back to the host
    downloadWorld(dev, world);
    freeDeviceMesh(dev);
    
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
