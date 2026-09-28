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

// Abort on any CUDA error
#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        const cudaError_t err_ = (expr);                                            \
        if (err_ != cudaSuccess) {                                                  \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),          \
                   __FILE__, __LINE__, cudaGetErrorString(err_));                   \
            exit(1);                                                                \
        }                                                                           \
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

// Maximum number of distinct materials (kept in GPU constant memory)
constexpr int MAX_MATERIALS = 16;

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

// Materials live in constant memory: tiny, read by every thread, broadcast-friendly
__constant__ Material c_materials[MAX_MATERIALS];

// Device-side mesh in structure-of-arrays form.  The per-connection arrays are
// stored transposed (connection-major, element-minor) so that neighbouring
// threads touch neighbouring addresses -> fully coalesced loads.
struct DeviceMesh {
    uint8_t*  material_idx  = nullptr;   // one byte per element (index into c_materials)
    uint8_t*  num_conn      = nullptr;   // connections per element (<= MAX_CONNECTIONS)
    uint32_t* conn_idx      = nullptr;   // [j * n_elems + i]
    val_t*    conn_flux     = nullptr;   // [j * n_elems + i]
    val_t*    energy[2]     = {nullptr, nullptr};  // ping-pong energy buffers
    val_t*    flux_accum    = nullptr;   // accumulated |total_flux| per element
    uint32_t  n_elems       = 0;
    int       block_size    = 256;
    int       grid_size     = 0;
};

static DeviceMesh g_mesh;

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// One thread per element: gather flux from the element's connections, write the
// updated energy to the output buffer and accumulate the absolute flux in place.
__global__ __launch_bounds__(256) void simulationKernel(
        const uint8_t*  __restrict__ material_idx,
        const uint8_t*  __restrict__ num_conn,
        const uint32_t* __restrict__ conn_idx,
        const val_t*    __restrict__ conn_flux,
        const val_t*    __restrict__ energy_in,
        val_t*          __restrict__ energy_out,
        val_t*          __restrict__ flux_accum,
        const uint32_t n_elems) {
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems; i += stride) {
        const Material mat = c_materials[material_idx[i]];
        const val_t this_energy = energy_in[i];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements (same order as the reference,
        // so the floating-point result is bit-identical)
        const int n = num_conn[i];
        for (int j = 0; j < n; ++j) {
            const uint32_t neighbor_idx = conn_idx[static_cast<size_t>(j) * n_elems + i];
            total_flux += computeFlux(mat, this_energy,
                                      conn_flux[static_cast<size_t>(j) * n_elems + i],
                                      __ldg(&energy_in[neighbor_idx]));
        }

        // Update element state
        energy_out[i] = this_energy + total_flux;
        flux_accum[i] += fabs(total_flux);
    }
}

// Upload the mesh to the GPU (part of setup, not of the timed simulation).
void initDevice(const World& world) {
    const size_t n_elems = world.elements_static.size();
    if (world.materials.size() > MAX_MATERIALS) {
        printf("ERROR: too many materials (%zu > %d)\n", world.materials.size(), MAX_MATERIALS);
        exit(1);
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation before timing

    DeviceMesh& m = g_mesh;
    m.n_elems = static_cast<uint32_t>(n_elems);

    // Flatten the array-of-structs mesh into transposed SoA host staging buffers
    std::vector<uint8_t>  h_mat(n_elems);
    std::vector<uint8_t>  h_nc(n_elems);
    std::vector<uint32_t> h_idx(n_elems * MAX_CONNECTIONS);
    std::vector<val_t>    h_flux(n_elems * MAX_CONNECTIONS);
    std::vector<val_t>    h_energy(n_elems);
    std::vector<val_t>    h_accum(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_mat[i] = static_cast<uint8_t>(es.material_idx);
        h_nc[i]  = static_cast<uint8_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_idx[j * n_elems + i]  = static_cast<uint32_t>(es.connected_idx[j]);
            h_flux[j * n_elems + i] = es.connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_accum[i]  = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMalloc(&m.material_idx, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&m.num_conn,     n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&m.conn_idx,     n_elems * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m.conn_flux,    n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&m.energy[0],    n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&m.energy[1],    n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&m.flux_accum,   n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(m.material_idx, h_mat.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.num_conn, h_nc.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.conn_idx, h_idx.data(), h_idx.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.conn_flux, h_flux.data(), h_flux.size() * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.energy[0], h_energy.data(), n_elems * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.flux_accum, h_accum.data(), n_elems * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, world.materials.data(),
                                  world.materials.size() * sizeof(Material)));

    // Size the grid for full occupancy; the grid-stride loop covers any excess
    int min_grid = 0, block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, simulationKernel, 0,
                                                  m.block_size));
    m.grid_size = static_cast<int>((n_elems + m.block_size - 1) / m.block_size);
    if (min_grid > 0 && m.grid_size > min_grid) {
        m.grid_size = min_grid;
    }
    if (m.grid_size < 1) {
        m.grid_size = 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeDevice() {
    DeviceMesh& m = g_mesh;
    cudaFree(m.material_idx);
    cudaFree(m.num_conn);
    cudaFree(m.conn_idx);
    cudaFree(m.conn_flux);
    cudaFree(m.energy[0]);
    cudaFree(m.energy[1]);
    cudaFree(m.flux_accum);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    DeviceMesh& m = g_mesh;
    const size_t n_elems = m.n_elems;
    int cur = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernel<<<m.grid_size, m.block_size>>>(
            m.material_idx, m.num_conn, m.conn_idx, m.conn_flux,
            m.energy[cur], m.energy[cur ^ 1], m.flux_accum, m.n_elems);

        // Swap buffers
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());

    // Bring the final state back into the host-side element array
    std::vector<val_t> h_energy(n_elems), h_accum(n_elems);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), m.energy[cur], n_elems * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accum.data(), m.flux_accum, n_elems * sizeof(val_t),
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_accum[i];
    }
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

    // Upload the mesh to the GPU
    initDevice(world);

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
            freeDevice();
            return 1;
        }
    }

    freeDevice();
    return 0;
}
