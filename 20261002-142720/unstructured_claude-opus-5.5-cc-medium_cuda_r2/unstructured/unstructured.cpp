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

// Abort with a message on any CUDA error
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int BLOCK_SIZE = 256;
// Elements per chunk when uploading the AoS static data for conversion
constexpr size_t UPLOAD_CHUNK = size_t(1) << 20;

// Compute energy flux between two elements.
// Explicitly rounded operations keep the evaluation order and rounding identical
// to the sequential reference (no FMA contraction).
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    val_t r = __dsub_rn(other_energy, this_energy);
    r = __dmul_rn(r, transfer_coeff);
    r = __dmul_rn(r, connection_flux);
    return __dmul_rn(r, 0.25);
}

// Device-side mesh representation (structure-of-arrays, connection-major)
struct DeviceMesh {
    size_t n_elems = 0;
    uint32_t* meta = nullptr;       // (material_idx << 4) | num_connections
    uint32_t* conn_idx = nullptr;   // [MAX_CONNECTIONS][n_elems]
    double* conn_flux_d = nullptr;  // [MAX_CONNECTIONS][n_elems]
    float* conn_flux_f = nullptr;   // compact copy when lossless
    Material* materials = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* total_flux = nullptr;
    ElementDynamic* packed = nullptr;
    bool flux_is_float = false;
};

// Convert a chunk of AoS static elements into the SoA device layout
__global__ void convertStaticKernel(const ElementStatic* __restrict__ chunk, size_t offset,
                                    size_t count, size_t n_elems, uint32_t* __restrict__ meta,
                                    uint32_t* __restrict__ conn_idx,
                                    double* __restrict__ conn_flux,
                                    int* __restrict__ not_float) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= count) return;
    const ElementStatic& e = chunk[k];
    const size_t i = offset + k;
    const uint32_t nc = (uint32_t)e.num_connections;
    meta[i] = ((uint32_t)e.material_idx << 4) | nc;
    for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
        uint32_t idx = (uint32_t)i;
        double f = 0.0;
        if (j < nc) {
            idx = (uint32_t)e.connected_idx[j];
            f = e.connected_flux[j];
            if ((double)(float)f != f) *not_float = 1;
        }
        conn_idx[j * n_elems + i] = idx;
        conn_flux[j * n_elems + i] = f;
    }
}

__global__ void fluxToFloatKernel(const double* __restrict__ in, float* __restrict__ out,
                                  size_t n) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k < n) out[k] = (float)in[k];
}

__global__ void unpackDynamicKernel(const ElementDynamic* __restrict__ in,
                                    val_t* __restrict__ energy, val_t* __restrict__ total_flux,
                                    size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        energy[i] = in[i].current_energy;
        total_flux[i] = in[i].total_flux;
    }
}

__global__ void packDynamicKernel(const val_t* __restrict__ energy,
                                  const val_t* __restrict__ total_flux,
                                  ElementDynamic* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].current_energy = energy[i];
        out[i].total_flux = total_flux[i];
    }
}

// One simulation step: one thread per element
template <typename FluxT>
__global__ void __launch_bounds__(BLOCK_SIZE)
stepKernel(const uint32_t* __restrict__ meta, const uint32_t* __restrict__ conn_idx,
           const FluxT* __restrict__ conn_flux, const Material* __restrict__ materials,
           const val_t* __restrict__ energy_in, val_t* __restrict__ energy_out,
           val_t* __restrict__ total_flux_arr, uint32_t n_elems) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const uint32_t m = __ldg(&meta[i]);
    const uint32_t nc = m & 0xFu;
    const Material mat = materials[m >> 4];
    const val_t this_energy = energy_in[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (in original order)
#pragma unroll
    for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const size_t off = (size_t)j * n_elems + i;
            const uint32_t nb = __ldg(&conn_idx[off]);
            const val_t cf = (val_t)__ldg(&conn_flux[off]);
            total_flux = __dadd_rn(total_flux,
                                   computeFlux(mat.transfer_coeff, this_energy, cf, energy_in[nb]));
        }
    }

    // Update element state
    energy_out[i] = __dadd_rn(this_energy, total_flux);
    total_flux_arr[i] = __dadd_rn(total_flux_arr[i], fabs(total_flux));
}

// Upload the world to the GPU (mesh conversion to the device layout)
void setupDevice(const World& world, DeviceMesh& dm) {
    const size_t n = world.elements_static.size();
    dm.n_elems = n;
    if (n == 0) return;

    CUDA_CHECK(cudaMalloc(&dm.meta, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dm.conn_idx, n * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dm.conn_flux_d, n * MAX_CONNECTIONS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dm.materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&dm.energy[0], n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dm.energy[1], n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dm.total_flux, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dm.packed, n * sizeof(ElementDynamic)));

    CUDA_CHECK(cudaMemcpy(dm.materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    int* d_not_float = nullptr;
    CUDA_CHECK(cudaMalloc(&d_not_float, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_not_float, 0, sizeof(int)));

    // Upload AoS static data in chunks and convert to SoA on the device
    const size_t chunk = std::min(n, UPLOAD_CHUNK);
    ElementStatic* d_stage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stage, chunk * sizeof(ElementStatic)));
    for (size_t off = 0; off < n; off += chunk) {
        const size_t cnt = std::min(chunk, n - off);
        CUDA_CHECK(cudaMemcpy(d_stage, world.elements_static.data() + off,
                              cnt * sizeof(ElementStatic), cudaMemcpyHostToDevice));
        convertStaticKernel<<<(cnt + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_stage, off, cnt, n, dm.meta, dm.conn_idx, dm.conn_flux_d, d_not_float);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFree(d_stage));

    int not_float = 0;
    CUDA_CHECK(cudaMemcpy(&not_float, d_not_float, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_not_float));

    // If every flux coefficient is exactly representable in float, store it compactly
    if (!not_float) {
        const size_t total = n * MAX_CONNECTIONS;
        CUDA_CHECK(cudaMalloc(&dm.conn_flux_f, total * sizeof(float)));
        fluxToFloatKernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            dm.conn_flux_d, dm.conn_flux_f, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(dm.conn_flux_d));
        dm.conn_flux_d = nullptr;
        dm.flux_is_float = true;
    }

    // Upload initial dynamic state
    CUDA_CHECK(cudaMemcpy(dm.packed, world.elements_dynamic.data(), n * sizeof(ElementDynamic),
                          cudaMemcpyHostToDevice));
    unpackDynamicKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        dm.packed, dm.energy[0], dm.total_flux, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeDevice(DeviceMesh& dm) {
    cudaFree(dm.meta);
    cudaFree(dm.conn_idx);
    cudaFree(dm.conn_flux_d);
    cudaFree(dm.conn_flux_f);
    cudaFree(dm.materials);
    cudaFree(dm.energy[0]);
    cudaFree(dm.energy[1]);
    cudaFree(dm.total_flux);
    cudaFree(dm.packed);
    dm = DeviceMesh{};
}

// Run simulation for n_iters iterations on the GPU, writing the final state back to world
void runSimulation(World& world, DeviceMesh& dm, const int n_iters) {
    const size_t n_elems = dm.n_elems;
    if (n_elems == 0) return;

    const unsigned grid = (unsigned)((n_elems + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (dm.flux_is_float) {
            stepKernel<float><<<grid, BLOCK_SIZE>>>(dm.meta, dm.conn_idx, dm.conn_flux_f,
                                                    dm.materials, dm.energy[cur],
                                                    dm.energy[cur ^ 1], dm.total_flux,
                                                    (uint32_t)n_elems);
        } else {
            stepKernel<double><<<grid, BLOCK_SIZE>>>(dm.meta, dm.conn_idx, dm.conn_flux_d,
                                                     dm.materials, dm.energy[cur],
                                                     dm.energy[cur ^ 1], dm.total_flux,
                                                     (uint32_t)n_elems);
        }
        // Swap buffers
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    // Gather final state back into the host world
    packDynamicKernel<<<grid, BLOCK_SIZE>>>(dm.energy[cur], dm.total_flux, dm.packed, n_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), dm.packed,
                          n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
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
    
    // Upload mesh to the GPU
    DeviceMesh dm;
    setupDevice(world, dm);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, dm, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    freeDevice(dm);

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
