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

// Compute energy flux between two elements (host fallback)
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA error check
static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

// GPU kernel: one thread per element
__global__ void stepKernel(
    const idx_t* __restrict__ d_material_idx,
    const val_t* __restrict__ d_transfer_coeff,
    const val_t* __restrict__ d_external_flow,
    const idx_t* __restrict__ d_num_connections,
    const idx_t* __restrict__ d_connected_idx,
    const val_t* __restrict__ d_connected_flux,
    const val_t* __restrict__ d_curr_energy,
    const val_t* __restrict__ d_curr_total_flux,
    val_t* __restrict__ d_next_energy,
    val_t* __restrict__ d_next_total_flux,
    const int n_elems)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const idx_t mat_id = d_material_idx[i];
    const val_t tcoeff = d_transfer_coeff[mat_id];
    val_t total_flux = d_external_flow[mat_id];
    const val_t this_energy = d_curr_energy[i];
    const val_t this_total_flux = d_curr_total_flux[i];

    const int base = i * MAX_CONNECTIONS;
    const int nconn = static_cast<int>(d_num_connections[i]);
    for (int j = 0; j < nconn; ++j) {
        const idx_t neighbor_idx = d_connected_idx[base + j];
        const val_t other_energy = d_curr_energy[neighbor_idx];
        total_flux += (other_energy - this_energy) * tcoeff * d_connected_flux[base + j] * 0.25;
    }

    d_next_energy[i] = this_energy + total_flux;
    d_next_total_flux[i] = this_total_flux + fabs(total_flux);
}

// Run simulation for n_iters iterations using CUDA
void runSimulation(World& world, const int n_iters) {
    const int n_elems = static_cast<int>(world.elements_static.size());

    // Prepare flattened host arrays for device
    std::vector<idx_t> h_material_idx(n_elems);
    std::vector<idx_t> h_num_conn(n_elems);
    std::vector<idx_t> h_connected_idx(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_connected_flux(n_elems * MAX_CONNECTIONS);

    for (int i = 0; i < n_elems; ++i) {
        const auto& s = world.elements_static[i];
        h_material_idx[i] = s.material_idx;
        h_num_conn[i] = s.num_connections;
        const int base = i * MAX_CONNECTIONS;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_connected_idx[base + j] = s.connected_idx[j];
            h_connected_flux[base + j] = s.connected_flux[j];
        }
    }

    // Materials small - copy to contiguous arrays
    const int n_materials = static_cast<int>(world.materials.size());
    std::vector<val_t> h_transfer_coeff(n_materials), h_external_flow(n_materials);
    for (int m = 0; m < n_materials; ++m) {
        h_transfer_coeff[m] = world.materials[m].transfer_coeff;
        h_external_flow[m] = world.materials[m].external_flow;
    }

    // Dynamic arrays
    std::vector<val_t> h_curr_energy(n_elems), h_curr_total(n_elems);
    for (int i = 0; i < n_elems; ++i) {
        h_curr_energy[i] = world.elements_dynamic[i].current_energy;
        h_curr_total[i] = world.elements_dynamic[i].total_flux;
    }

    // Allocate device memory
    idx_t *d_material_idx = nullptr, *d_num_conn = nullptr, *d_connected_idx = nullptr;
    val_t *d_connected_flux = nullptr, *d_transfer_coeff = nullptr, *d_external_flow = nullptr;
    val_t *d_curr_energy = nullptr, *d_curr_total = nullptr, *d_next_energy = nullptr, *d_next_total = nullptr;

    cudaCheck(cudaMalloc(&d_material_idx, n_elems * sizeof(idx_t)), "malloc material_idx");
    cudaCheck(cudaMalloc(&d_num_conn, n_elems * sizeof(idx_t)), "malloc num_conn");
    cudaCheck(cudaMalloc(&d_connected_idx, n_elems * MAX_CONNECTIONS * sizeof(idx_t)), "malloc connected_idx");
    cudaCheck(cudaMalloc(&d_connected_flux, n_elems * MAX_CONNECTIONS * sizeof(val_t)), "malloc connected_flux");

    cudaCheck(cudaMalloc(&d_transfer_coeff, n_materials * sizeof(val_t)), "malloc transfer_coeff");
    cudaCheck(cudaMalloc(&d_external_flow, n_materials * sizeof(val_t)), "malloc external_flow");

    cudaCheck(cudaMalloc(&d_curr_energy, n_elems * sizeof(val_t)), "malloc curr_energy");
    cudaCheck(cudaMalloc(&d_curr_total, n_elems * sizeof(val_t)), "malloc curr_total");
    cudaCheck(cudaMalloc(&d_next_energy, n_elems * sizeof(val_t)), "malloc next_energy");
    cudaCheck(cudaMalloc(&d_next_total, n_elems * sizeof(val_t)), "malloc next_total");

    // Copy host to device
    cudaCheck(cudaMemcpy(d_material_idx, h_material_idx.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice), "copy material_idx");
    cudaCheck(cudaMemcpy(d_num_conn, h_num_conn.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice), "copy num_conn");
    cudaCheck(cudaMemcpy(d_connected_idx, h_connected_idx.data(), n_elems * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice), "copy connected_idx");
    cudaCheck(cudaMemcpy(d_connected_flux, h_connected_flux.data(), n_elems * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice), "copy connected_flux");

    cudaCheck(cudaMemcpy(d_transfer_coeff, h_transfer_coeff.data(), n_materials * sizeof(val_t), cudaMemcpyHostToDevice), "copy transfer_coeff");
    cudaCheck(cudaMemcpy(d_external_flow, h_external_flow.data(), n_materials * sizeof(val_t), cudaMemcpyHostToDevice), "copy external_flow");

    cudaCheck(cudaMemcpy(d_curr_energy, h_curr_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "copy curr_energy");
    cudaCheck(cudaMemcpy(d_curr_total, h_curr_total.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "copy curr_total");

    // Kernel launch parameters
    const int threads = 256;
    const int blocks = (n_elems + threads - 1) / threads;

    // Run iterations, swapping pointers each step
    for (int iter = 0; iter < n_iters; ++iter) {
        stepKernel<<<blocks, threads>>>(
            d_material_idx,
            d_transfer_coeff,
            d_external_flow,
            d_num_conn,
            d_connected_idx,
            d_connected_flux,
            d_curr_energy,
            d_curr_total,
            d_next_energy,
            d_next_total,
            n_elems);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");

        // swap current and next
        std::swap(d_curr_energy, d_next_energy);
        std::swap(d_curr_total, d_next_total);
    }

    // Copy results back to host
    cudaCheck(cudaMemcpy(h_curr_energy.data(), d_curr_energy, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "copy back energy");
    cudaCheck(cudaMemcpy(h_curr_total.data(), d_curr_total, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "copy back total");

    // Update world dynamic arrays
    for (int i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_curr_energy[i];
        world.elements_dynamic[i].total_flux = h_curr_total[i];
    }

    // Free device memory
    cudaFree(d_material_idx);
    cudaFree(d_num_conn);
    cudaFree(d_connected_idx);
    cudaFree(d_connected_flux);
    cudaFree(d_transfer_coeff);
    cudaFree(d_external_flow);
    cudaFree(d_curr_energy);
    cudaFree(d_curr_total);
    cudaFree(d_next_energy);
    cudaFree(d_next_total);
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
