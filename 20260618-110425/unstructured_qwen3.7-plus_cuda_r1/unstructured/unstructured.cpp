#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"
#include <cuda_runtime.h>

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

// Device-side SoA (Structure of Arrays) storage for GPU
struct DeviceWorld {
    idx_t* d_material_idx;
    idx_t* d_num_connections;
    idx_t* d_connected_idx;
    val_t* d_connected_flux;
    val_t* d_transfer_coeff;
    val_t* d_external_flow;
    val_t* d_current_energy[2];
    val_t* d_total_flux[2];
    idx_t n_elems;
    idx_t n_mats;
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

// CUDA kernel: compute one simulation step using SoA layout for coalesced access
__global__ void simulation_step_kernel(
    const idx_t* __restrict__ material_idx,
    const idx_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const val_t* __restrict__ transfer_coeff,
    const val_t* __restrict__ external_flow,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ flux_out,
    const idx_t n_elems)
{
    const idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const idx_t mat_id = material_idx[i];
    const val_t tc = transfer_coeff[mat_id];
    const val_t ef = external_flow[mat_id];
    const val_t my_energy = energy_in[i];

    val_t total_flux = ef;
    const idx_t nc = num_connections[i];
    const idx_t base = i * MAX_CONNECTIONS;

    #pragma unroll 4
    for (idx_t j = 0; j < nc; ++j) {
        const idx_t neighbor = connected_idx[base + j];
        const val_t flux_coeff = connected_flux[base + j];
        const val_t neighbor_energy = energy_in[neighbor];
        total_flux += (neighbor_energy - my_energy) * tc * flux_coeff * (val_t)0.25;
    }

    energy_out[i] = my_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Upload world data from CPU AoS to GPU SoA layout
void uploadWorldToDevice(DeviceWorld& dw, const World& world) {
    dw.n_elems = world.elements_static.size();
    dw.n_mats = world.materials.size();

    // Prepare SoA host arrays
    std::vector<idx_t> h_mat_idx(dw.n_elems);
    std::vector<idx_t> h_num_conn(dw.n_elems);
    std::vector<idx_t> h_conn_idx(dw.n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_conn_flux(dw.n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_transfer_coeff(dw.n_mats);
    std::vector<val_t> h_external_flow(dw.n_mats);
    std::vector<val_t> h_energy(dw.n_elems);
    std::vector<val_t> h_flux(dw.n_elems);

    for (size_t i = 0; i < dw.n_mats; ++i) {
        h_transfer_coeff[i] = world.materials[i].transfer_coeff;
        h_external_flow[i] = world.materials[i].external_flow;
    }

    for (size_t i = 0; i < dw.n_elems; ++i) {
        h_mat_idx[i] = world.elements_static[i].material_idx;
        h_num_conn[i] = world.elements_static[i].num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_conn_idx[i * MAX_CONNECTIONS + j] = world.elements_static[i].connected_idx[j];
            h_conn_flux[i * MAX_CONNECTIONS + j] = world.elements_static[i].connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dw.d_material_idx, dw.n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_num_connections, dw.n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_connected_idx, dw.n_elems * MAX_CONNECTIONS * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_connected_flux, dw.n_elems * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_transfer_coeff, dw.n_mats * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_external_flow, dw.n_mats * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_current_energy[0], dw.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_current_energy[1], dw.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_total_flux[0], dw.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.d_total_flux[1], dw.n_elems * sizeof(val_t)));

    // Copy static data and initial state to device
    CUDA_CHECK(cudaMemcpy(dw.d_material_idx, h_mat_idx.data(), dw.n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_num_connections, h_num_conn.data(), dw.n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_connected_idx, h_conn_idx.data(), dw.n_elems * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_connected_flux, h_conn_flux.data(), dw.n_elems * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_transfer_coeff, h_transfer_coeff.data(), dw.n_mats * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_external_flow, h_external_flow.data(), dw.n_mats * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_current_energy[0], h_energy.data(), dw.n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.d_total_flux[0], h_flux.data(), dw.n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dw.d_current_energy[1], 0, dw.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(dw.d_total_flux[1], 0, dw.n_elems * sizeof(val_t)));
}

// Download final state from GPU back to CPU world
void downloadWorldFromDevice(DeviceWorld& dw, World& world, int n_iters) {
    // Determine which buffer has the final result
    int final_buf = n_iters % 2;

    std::vector<val_t> h_energy(dw.n_elems);
    std::vector<val_t> h_flux(dw.n_elems);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), dw.d_current_energy[final_buf], dw.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), dw.d_total_flux[final_buf], dw.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < dw.n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

// Free device memory
void freeDeviceWorld(DeviceWorld& dw) {
    cudaFree(dw.d_material_idx);
    cudaFree(dw.d_num_connections);
    cudaFree(dw.d_connected_idx);
    cudaFree(dw.d_connected_flux);
    cudaFree(dw.d_transfer_coeff);
    cudaFree(dw.d_external_flow);
    cudaFree(dw.d_current_energy[0]);
    cudaFree(dw.d_current_energy[1]);
    cudaFree(dw.d_total_flux[0]);
    cudaFree(dw.d_total_flux[1]);
}

// Run simulation for n_iters iterations on GPU
void runSimulation(DeviceWorld& dw, World& world, const int n_iters) {
    const int block_size = 256;
    const int grid_size = (int)((dw.n_elems + block_size - 1) / block_size);

    int cur_buf = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        int next_buf = 1 - cur_buf;
        simulation_step_kernel<<<grid_size, block_size>>>(
            dw.d_material_idx,
            dw.d_num_connections,
            dw.d_connected_idx,
            dw.d_connected_flux,
            dw.d_transfer_coeff,
            dw.d_external_flow,
            dw.d_current_energy[cur_buf],
            dw.d_total_flux[cur_buf],
            dw.d_current_energy[next_buf],
            dw.d_total_flux[next_buf],
            dw.n_elems);
        cur_buf = next_buf;
    }

    // Synchronize to ensure all kernels complete before timing stops
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results back to CPU world
    downloadWorldFromDevice(dw, world, n_iters);
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
    
    // Upload world to GPU (SoA layout)
    DeviceWorld dw;
    uploadWorldToDevice(dw, world);
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(dw, world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Free device memory
    freeDeviceWorld(dw);
    
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
