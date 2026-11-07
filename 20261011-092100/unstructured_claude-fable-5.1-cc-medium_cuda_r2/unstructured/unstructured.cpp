// Unstructured mesh energy transfer benchmark - CUDA implementation.
//
// This translation unit is compiled as CUDA (see CMakeLists.txt, which marks
// it with LANGUAGE CUDA). The mesh is built on the host exactly as in the
// original code; the simulation itself runs unconditionally on the GPU.

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Device-side index type. Element counts are ints on the host, so 32-bit
// indices are sufficient and halve the connectivity bandwidth on the GPU.
using dev_idx_t = uint32_t;

// Maximum number of materials that fit in constant memory (the benchmark uses 3).
constexpr int MAX_DEVICE_MATERIALS = 64;
__constant__ Material c_materials[MAX_DEVICE_MATERIALS];

// Device mesh in structure-of-arrays layout. Connectivity is stored
// "connection-major" (conn_idx[j * n + i]) so that, for a fixed j, a warp
// reads consecutive elements' entries -> fully coalesced accesses.
struct DeviceWorld {
    int n_elems = 0;
    int n_materials = 0;
    dev_idx_t* material_idx = nullptr;   // [n]
    dev_idx_t* num_connections = nullptr;// [n]
    dev_idx_t* conn_idx = nullptr;       // [MAX_CONNECTIONS][n]
    val_t* conn_flux = nullptr;          // [MAX_CONNECTIONS][n]
    val_t* energy = nullptr;             // [n]
    val_t* total_flux = nullptr;         // [n]
    val_t* energy_swap = nullptr;        // [n]
    val_t* total_flux_swap = nullptr;    // [n]
};

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
// Host <-> device transfer
// ---------------------------------------------------------------------------

// Upload the host world to the device in SoA layout.
void uploadWorld(const World& world, DeviceWorld& dw) {
    const int n = static_cast<int>(world.elements_static.size());
    dw.n_elems = n;
    dw.n_materials = static_cast<int>(world.materials.size());
    if (dw.n_materials > MAX_DEVICE_MATERIALS) {
        fprintf(stderr, "Too many materials for constant memory (%d > %d)\n",
                dw.n_materials, MAX_DEVICE_MATERIALS);
        exit(1);
    }

    // Pack into SoA host staging buffers.
    std::vector<dev_idx_t> h_material_idx(n);
    std::vector<dev_idx_t> h_num_connections(n);
    std::vector<dev_idx_t> h_conn_idx(static_cast<size_t>(MAX_CONNECTIONS) * n, 0);
    std::vector<val_t> h_conn_flux(static_cast<size_t>(MAX_CONNECTIONS) * n, 0.0);
    std::vector<val_t> h_energy(n);
    std::vector<val_t> h_total_flux(n);

    for (int i = 0; i < n; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_material_idx[i] = static_cast<dev_idx_t>(es.material_idx);
        h_num_connections[i] = static_cast<dev_idx_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_conn_idx[j * n + i] = static_cast<dev_idx_t>(es.connected_idx[j]);
            h_conn_flux[j * n + i] = es.connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_total_flux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, world.materials.data(),
                                  sizeof(Material) * dw.n_materials));

    CUDA_CHECK(cudaMalloc(&dw.material_idx, sizeof(dev_idx_t) * n));
    CUDA_CHECK(cudaMalloc(&dw.num_connections, sizeof(dev_idx_t) * n));
    CUDA_CHECK(cudaMalloc(&dw.conn_idx, sizeof(dev_idx_t) * MAX_CONNECTIONS * n));
    CUDA_CHECK(cudaMalloc(&dw.conn_flux, sizeof(val_t) * MAX_CONNECTIONS * n));
    CUDA_CHECK(cudaMalloc(&dw.energy, sizeof(val_t) * n));
    CUDA_CHECK(cudaMalloc(&dw.total_flux, sizeof(val_t) * n));
    CUDA_CHECK(cudaMalloc(&dw.energy_swap, sizeof(val_t) * n));
    CUDA_CHECK(cudaMalloc(&dw.total_flux_swap, sizeof(val_t) * n));

    CUDA_CHECK(cudaMemcpy(dw.material_idx, h_material_idx.data(), sizeof(dev_idx_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.num_connections, h_num_connections.data(), sizeof(dev_idx_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.conn_idx, h_conn_idx.data(), sizeof(dev_idx_t) * MAX_CONNECTIONS * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.conn_flux, h_conn_flux.data(), sizeof(val_t) * MAX_CONNECTIONS * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.energy, h_energy.data(), sizeof(val_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.total_flux, h_total_flux.data(), sizeof(val_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Download the current dynamic state back into world.elements_dynamic.
void downloadWorld(const DeviceWorld& dw, World& world) {
    const int n = dw.n_elems;
    std::vector<val_t> h_energy(n);
    std::vector<val_t> h_total_flux(n);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), dw.energy, sizeof(val_t) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_total_flux.data(), dw.total_flux, sizeof(val_t) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_total_flux[i];
    }
}

void freeWorld(DeviceWorld& dw) {
    cudaFree(dw.material_idx);
    cudaFree(dw.num_connections);
    cudaFree(dw.conn_idx);
    cudaFree(dw.conn_flux);
    cudaFree(dw.energy);
    cudaFree(dw.total_flux);
    cudaFree(dw.energy_swap);
    cudaFree(dw.total_flux_swap);
    dw = DeviceWorld{};
}

// ---------------------------------------------------------------------------
// Simulation kernel
// ---------------------------------------------------------------------------

// Compute energy flux between two elements
__device__ __forceinline__ val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// One thread per element. Summation order matches the sequential reference:
// external_flow first, then connections in index order.
__global__ void __launch_bounds__(256)
simulationStep(const dev_idx_t* __restrict__ material_idx,
               const dev_idx_t* __restrict__ num_connections,
               const dev_idx_t* __restrict__ conn_idx,
               const val_t* __restrict__ conn_flux,
               const val_t* __restrict__ energy_in,
               const val_t* __restrict__ total_flux_in,
               val_t* __restrict__ energy_out,
               val_t* __restrict__ total_flux_out,
               const int n_elems) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = c_materials[material_idx[i]];
    const val_t this_energy = energy_in[i];
    const dev_idx_t n_conn = num_connections[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    #pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < static_cast<int>(n_conn)) {
            const size_t off = static_cast<size_t>(j) * n_elems + i;
            const dev_idx_t neighbor_idx = __ldg(&conn_idx[off]);
            const val_t cflux = __ldg(&conn_flux[off]);
            const val_t neighbor_energy = __ldg(&energy_in[neighbor_idx]);
            total_flux += computeFlux(mat, this_energy, cflux, neighbor_energy);
        }
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    total_flux_out[i] = total_flux_in[i] + fabs(total_flux);
}

// Run simulation for n_iters iterations
void runSimulation(DeviceWorld& dw, const int n_iters) {
    const int n_elems = dw.n_elems;
    constexpr int block = 256;
    const int grid = (n_elems + block - 1) / block;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationStep<<<grid, block>>>(dw.material_idx, dw.num_connections,
                                        dw.conn_idx, dw.conn_flux,
                                        dw.energy, dw.total_flux,
                                        dw.energy_swap, dw.total_flux_swap,
                                        n_elems);
        // Swap buffers
        std::swap(dw.energy, dw.energy_swap);
        std::swap(dw.total_flux, dw.total_flux_swap);
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
        memcpy(&e_bits, &elements[i].current_energy, sizeof(e_bits));
        memcpy(&f_bits, &elements[i].total_flux, sizeof(f_bits));
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
    
    // Upload mesh to the GPU (this also initializes the CUDA context)
    CUDA_CHECK(cudaSetDevice(0));
    DeviceWorld dw;
    uploadWorld(world, dw);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(dw, n_iters);
    
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
    
    // Copy results back to the host
    downloadWorld(dw, world);
    freeWorld(dw);
    
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
