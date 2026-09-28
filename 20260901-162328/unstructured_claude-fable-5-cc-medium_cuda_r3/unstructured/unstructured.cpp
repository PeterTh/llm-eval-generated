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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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

// Materials live in constant memory: every thread of a warp reads the same
// entry almost always, so accesses are broadcast.
constexpr int MAX_MATERIALS = 16;
__constant__ Material c_materials[MAX_MATERIALS];

// One simulation step: each thread updates one element. The connectivity is
// stored transposed (connection-major) so that consecutive threads read
// consecutive addresses. The flux accumulation order matches the original
// scalar code exactly (external flow first, then connections in order).
__global__ void stepKernel(const int n_elems,
                           const uint8_t* __restrict__ mat_id,
                           const uint8_t* __restrict__ n_conn,
                           const int* __restrict__ conn_idx,
                           const double* __restrict__ conn_flux,
                           const double* __restrict__ energy_in,
                           const double* __restrict__ flux_in,
                           double* __restrict__ energy_out,
                           double* __restrict__ flux_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = c_materials[mat_id[i]];
    const double energy = energy_in[i];
    const int nc = n_conn[i];

    double total_flux = mat.external_flow;
    for (int j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_elems + i;
        const double neighbor_energy = __ldg(&energy_in[conn_idx[slot]]);
        total_flux += (neighbor_energy - energy) * mat.transfer_coeff *
                      conn_flux[slot] * 0.25;
    }

    energy_out[i] = energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Device-side mirror of the world state, in SoA layout
struct GpuSim {
    int n_elems = 0;
    uint8_t* d_mat_id = nullptr;
    uint8_t* d_n_conn = nullptr;
    int* d_conn_idx = nullptr;      // transposed: [connection][element]
    double* d_conn_flux = nullptr;  // transposed: [connection][element]
    double* d_energy[2] = {nullptr, nullptr};
    double* d_flux[2] = {nullptr, nullptr};
    int cur = 0;  // which buffer holds the current state
};

// Upload the world to the GPU, converting to SoA with transposed connectivity
void gpuInit(GpuSim& sim, const World& world) {
    const int n_elems = static_cast<int>(world.elements_static.size());
    sim.n_elems = n_elems;
    sim.cur = 0;

    if (world.materials.size() > MAX_MATERIALS) {
        fprintf(stderr, "Too many materials for constant memory\n");
        exit(1);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, world.materials.data(),
                                  world.materials.size() * sizeof(Material)));

    std::vector<uint8_t> mat_id(n_elems);
    std::vector<uint8_t> n_conn(n_elems);
    std::vector<int> conn_idx(static_cast<size_t>(n_elems) * MAX_CONNECTIONS, 0);
    std::vector<double> conn_flux(static_cast<size_t>(n_elems) * MAX_CONNECTIONS, 0.0);
    std::vector<double> energy(n_elems);
    std::vector<double> flux(n_elems);

    for (int i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        mat_id[i] = static_cast<uint8_t>(es.material_idx);
        n_conn[i] = static_cast<uint8_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            conn_idx[j * n_elems + i] = static_cast<int>(es.connected_idx[j]);
            conn_flux[j * n_elems + i] = es.connected_flux[j];
        }
        energy[i] = world.elements_dynamic[i].current_energy;
        flux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMalloc(&sim.d_mat_id, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_n_conn, n_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_conn_idx, conn_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_conn_flux, conn_flux.size() * sizeof(double)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&sim.d_energy[b], n_elems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sim.d_flux[b], n_elems * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(sim.d_mat_id, mat_id.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_n_conn, n_conn.data(), n_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_conn_idx, conn_idx.data(),
                          conn_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_conn_flux, conn_flux.data(),
                          conn_flux.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_energy[0], energy.data(), n_elems * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_flux[0], flux.data(), n_elems * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy the current dynamic state back into the world
void gpuDownload(const GpuSim& sim, World& world) {
    const int n_elems = sim.n_elems;
    std::vector<double> energy(n_elems);
    std::vector<double> flux(n_elems);
    CUDA_CHECK(cudaMemcpy(energy.data(), sim.d_energy[sim.cur],
                          n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), sim.d_flux[sim.cur],
                          n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
    }
}

void gpuFree(GpuSim& sim) {
    cudaFree(sim.d_mat_id);
    cudaFree(sim.d_n_conn);
    cudaFree(sim.d_conn_idx);
    cudaFree(sim.d_conn_flux);
    for (int b = 0; b < 2; ++b) {
        cudaFree(sim.d_energy[b]);
        cudaFree(sim.d_flux[b]);
    }
}

// Run simulation for n_iters iterations
void runSimulation(GpuSim& sim, const int n_iters) {
    constexpr int block_size = 256;
    const int grid_size = (sim.n_elems + block_size - 1) / block_size;

    for (int iter = 0; iter < n_iters; ++iter) {
        const int src = sim.cur;
        const int dst = 1 - src;
        stepKernel<<<grid_size, block_size>>>(
            sim.n_elems, sim.d_mat_id, sim.d_n_conn, sim.d_conn_idx,
            sim.d_conn_flux, sim.d_energy[src], sim.d_flux[src],
            sim.d_energy[dst], sim.d_flux[dst]);
        sim.cur = dst;
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

    // Upload the mesh to the GPU
    GpuSim sim;
    gpuInit(sim, world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    gpuDownload(sim, world);
    gpuFree(sim);

    printf("Computation time: %ld ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec =
        (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

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
