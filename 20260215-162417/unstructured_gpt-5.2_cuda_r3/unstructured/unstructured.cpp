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

static inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

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

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

namespace {

__global__ void sim_step_kernel(const size_t n_elems,
                               const idx_t* __restrict__ material_idx,
                               const idx_t* __restrict__ num_connections,
                               const idx_t* __restrict__ connected_idx,
                               const val_t* __restrict__ connected_flux,
                               const val_t* __restrict__ mat_transfer_coeff,
                               const val_t* __restrict__ mat_external_flow,
                               const val_t* __restrict__ energy_in,
                               const val_t* __restrict__ total_flux_in,
                               val_t* __restrict__ energy_out,
                               val_t* __restrict__ total_flux_out) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = tid; i < n_elems; i += stride) {
        const idx_t mid = material_idx[i];
        const val_t transfer = mat_transfer_coeff[mid];
        val_t total = mat_external_flow[mid];

        const val_t this_energy = energy_in[i];
        const val_t this_total_flux = total_flux_in[i];
        const idx_t n_conn = num_connections[i];

        const size_t base = i * static_cast<size_t>(MAX_CONNECTIONS);

#pragma unroll
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (static_cast<idx_t>(j) < n_conn) {
                const idx_t nidx = connected_idx[base + static_cast<size_t>(j)];
                const val_t other_energy = energy_in[static_cast<size_t>(nidx)];
                const val_t cflux = connected_flux[base + static_cast<size_t>(j)];

                // Mirror computeFlux() exactly: (other - this) * transfer * connection_flux * 0.25
                const val_t flux = (other_energy - this_energy) * transfer * cflux * 0.25;
                total += flux;
            }
        }

        energy_out[i] = this_energy + total;
        total_flux_out[i] = this_total_flux + fabs(total);
    }
}

void runSimulationCUDA(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0 || n_iters <= 0) {
        return;
    }

    // Build host-side SoA copies for better device access.
    std::vector<idx_t> h_material_idx(n_elems);
    std::vector<idx_t> h_num_conn(n_elems);
    std::vector<idx_t> h_connected_idx(n_elems * static_cast<size_t>(MAX_CONNECTIONS));
    std::vector<val_t> h_connected_flux(n_elems * static_cast<size_t>(MAX_CONNECTIONS));

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_material_idx[i] = es.material_idx;
        h_num_conn[i] = es.num_connections;

        const size_t base = i * static_cast<size_t>(MAX_CONNECTIONS);
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_connected_idx[base + static_cast<size_t>(j)] = es.connected_idx[j];
            h_connected_flux[base + static_cast<size_t>(j)] = es.connected_flux[j];
        }
    }

    std::vector<val_t> h_mat_transfer(world.materials.size());
    std::vector<val_t> h_mat_external(world.materials.size());
    for (size_t i = 0; i < world.materials.size(); ++i) {
        h_mat_transfer[i] = world.materials[i].transfer_coeff;
        h_mat_external[i] = world.materials[i].external_flow;
    }

    std::vector<val_t> h_energy(n_elems);
    std::vector<val_t> h_total_flux(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_total_flux[i] = world.elements_dynamic[i].total_flux;
    }

    idx_t* d_material_idx = nullptr;
    idx_t* d_num_conn = nullptr;
    idx_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;
    val_t* d_mat_transfer = nullptr;
    val_t* d_mat_external = nullptr;
    val_t* d_energy_a = nullptr;
    val_t* d_energy_b = nullptr;
    val_t* d_flux_a = nullptr;
    val_t* d_flux_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_material_idx, n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_num_conn, n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_connected_idx, n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&d_connected_flux, n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_transfer, world.materials.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_external, world.materials.size() * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&d_energy_a, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_b, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_a, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_b, n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_material_idx, h_material_idx.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_conn, h_num_conn.data(), n_elems * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_idx, h_connected_idx.data(), n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_flux, h_connected_flux.data(), n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_transfer, h_mat_transfer.data(), world.materials.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_external, h_mat_external.data(), world.materials.size() * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_energy_a, h_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux_a, h_total_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    constexpr int block = 256;
    int grid = static_cast<int>((n_elems + block - 1) / block);
    if (grid < 1) grid = 1;
    // Clamp grid to avoid excessive launch overhead.
    if (grid > 65535) grid = 65535;

    const idx_t* d_mat = d_material_idx;
    const idx_t* d_nc = d_num_conn;
    const idx_t* d_cidx = d_connected_idx;
    const val_t* d_cflux = d_connected_flux;

    val_t* energy_in = d_energy_a;
    val_t* flux_in = d_flux_a;
    val_t* energy_out = d_energy_b;
    val_t* flux_out = d_flux_b;

    for (int iter = 0; iter < n_iters; ++iter) {
        sim_step_kernel<<<grid, block>>>(n_elems, d_mat, d_nc, d_cidx, d_cflux, d_mat_transfer, d_mat_external,
                                         energy_in, flux_in, energy_out, flux_out);
        CUDA_CHECK(cudaGetLastError());

        // Ping-pong.
        std::swap(energy_in, energy_out);
        std::swap(flux_in, flux_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_energy.data(), energy_in, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_total_flux.data(), flux_in, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_total_flux[i];
    }

    cudaFree(d_material_idx);
    cudaFree(d_num_conn);
    cudaFree(d_connected_idx);
    cudaFree(d_connected_flux);
    cudaFree(d_mat_transfer);
    cudaFree(d_mat_external);
    cudaFree(d_energy_a);
    cudaFree(d_energy_b);
    cudaFree(d_flux_a);
    cudaFree(d_flux_b);
}

}  // namespace

// Run simulation for n_iters iterations (unconditionally CUDA/GPU).
void runSimulation(World& world, const int n_iters) {
    runSimulationCUDA(world, n_iters);
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

    // Ensure CUDA context is initialized outside the timed region.
    CUDA_CHECK(cudaFree(0));

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
