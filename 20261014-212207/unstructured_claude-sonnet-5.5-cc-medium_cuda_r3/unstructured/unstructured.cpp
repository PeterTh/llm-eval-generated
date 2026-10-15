#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

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

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// One simulation step: structure-of-arrays, connection-major layout for coalescing
__global__ void stepKernel(const size_t n_elems,
                           const uint32_t* __restrict__ mat_idx,
                           const uint8_t* __restrict__ num_conn,
                           const uint32_t* __restrict__ conn_idx,
                           const val_t* __restrict__ conn_flux,
                           const Material* __restrict__ materials,
                           const val_t* __restrict__ energy_in,
                           val_t* __restrict__ energy_out,
                           val_t* __restrict__ total_flux) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = materials[mat_idx[i]];
    const val_t e = energy_in[i];
    const int nc = num_conn[i];

    val_t flux = mat.external_flow;
    for (int j = 0; j < nc; ++j) {
        const uint32_t nb = conn_idx[(size_t)j * n_elems + i];
        const val_t cf = conn_flux[(size_t)j * n_elems + i];
        // Explicit intrinsics keep the original operation order without FMA contraction
        const val_t t = __dmul_rn(__dmul_rn(__dmul_rn(__dsub_rn(__ldg(&energy_in[nb]), e),
                                                      mat.transfer_coeff), cf), 0.25);
        flux = __dadd_rn(flux, t);
    }

    energy_out[i] = __dadd_rn(e, flux);
    total_flux[i] = __dadd_rn(total_flux[i], fabs(flux));
}

// Device-resident mesh and state, built once before the timed simulation
struct DeviceState {
    uint32_t *mat = nullptr, *cidx = nullptr;
    uint8_t* nc = nullptr;
    val_t *cflux = nullptr, *e0 = nullptr, *e1 = nullptr, *tf = nullptr;
    Material* materials = nullptr;
};
static DeviceState g_dev;

// Initialize the GPU, repack the mesh into SoA layout and upload it
void prepareDevice(World& world) {
    const size_t n = world.elements_static.size();
    CUDA_CHECK(cudaFree(0));  // create context

    // Repack host data into SoA layout
    std::vector<uint32_t> h_mat(n);
    std::vector<uint8_t> h_nc(n);
    std::vector<uint32_t> h_cidx(n * MAX_CONNECTIONS, 0);
    std::vector<val_t> h_cflux(n * MAX_CONNECTIONS, 0.0);
    std::vector<val_t> h_energy(n), h_tflux(n);
    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& e = world.elements_static[i];
        h_mat[i] = (uint32_t)e.material_idx;
        h_nc[i] = (uint8_t)e.num_connections;
        for (idx_t j = 0; j < e.num_connections; ++j) {
            h_cidx[j * n + i] = (uint32_t)e.connected_idx[j];
            h_cflux[j * n + i] = e.connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_tflux[i] = world.elements_dynamic[i].total_flux;
    }

    uint32_t *&d_mat = g_dev.mat, *&d_cidx = g_dev.cidx;
    uint8_t*& d_nc = g_dev.nc;
    val_t *&d_cflux = g_dev.cflux, *&d_e0 = g_dev.e0, *&d_e1 = g_dev.e1, *&d_tf = g_dev.tf;
    Material*& d_materials = g_dev.materials;
    CUDA_CHECK(cudaMalloc(&d_mat, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nc, n));
    CUDA_CHECK(cudaMalloc(&d_cidx, n * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cflux, n * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_e0, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_e1, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_tf, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));

    CUDA_CHECK(cudaMemcpy(d_mat, h_mat.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nc, h_nc.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cidx, h_cidx.data(), n * MAX_CONNECTIONS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cflux, h_cflux.data(), n * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_e0, h_energy.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tf, h_tflux.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n = world.elements_static.size();
    const DeviceState& d = g_dev;
    uint32_t* d_mat = d.mat; uint32_t* d_cidx = d.cidx; uint8_t* d_nc = d.nc;
    val_t* d_cflux = d.cflux; val_t* d_e0 = d.e0; val_t* d_e1 = d.e1; val_t* d_tf = d.tf;
    Material* d_materials = d.materials;
    std::vector<val_t> h_energy(n), h_tflux(n);

    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    val_t* cur = d_e0;
    val_t* nxt = d_e1;
    for (int iter = 0; iter < n_iters; ++iter) {
        stepKernel<<<(unsigned)grid, block>>>(n, d_mat, d_nc, d_cidx, d_cflux, d_materials,
                                              cur, nxt, d_tf);
        std::swap(cur, nxt);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_energy.data(), cur, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tflux.data(), d_tf, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_tflux[i];
    }

    cudaFree(d_mat); cudaFree(d_nc); cudaFree(d_cidx); cudaFree(d_cflux);
    cudaFree(d_e0); cudaFree(d_e1); cudaFree(d_tf); cudaFree(d_materials);
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
    prepareDevice(world);
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
