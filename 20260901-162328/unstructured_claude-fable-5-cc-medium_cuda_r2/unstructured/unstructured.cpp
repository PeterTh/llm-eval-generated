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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

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

// Compute energy flux between two elements (device-side, same math as reference)
__device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                    val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// One simulation step: each thread updates one element.
// Static connectivity is stored in struct-of-arrays layout, with connection
// slot j for element i at [j * n_elems + i], so that consecutive threads
// access consecutive memory (coalesced loads).
__global__ void simulationStepKernel(const val_t* __restrict__ transfer_coeff,
                                     const val_t* __restrict__ external_flow,
                                     const uint32_t* __restrict__ num_connections,
                                     const uint32_t* __restrict__ conn_idx,
                                     const val_t* __restrict__ conn_flux,
                                     const val_t* __restrict__ energy_in,
                                     const val_t* __restrict__ flux_in,
                                     val_t* __restrict__ energy_out,
                                     val_t* __restrict__ flux_out,
                                     const size_t n_elems) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const val_t coeff = transfer_coeff[i];
    const val_t this_energy = energy_in[i];
    const uint32_t n_conn = num_connections[i];

    // Start with external flow
    val_t total_flux = external_flow[i];

    // Add flux from all connected elements, in the same order as the
    // reference implementation so floating-point results are identical
    for (uint32_t j = 0; j < n_conn; ++j) {
        const size_t slot = (size_t)j * n_elems + i;
        const val_t neighbor_energy = energy_in[conn_idx[slot]];
        total_flux += computeFlux(coeff, this_energy, conn_flux[slot], neighbor_energy);
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (n_elems == 0 || n_iters <= 0) return;

    // Determine the maximum number of connections actually used
    idx_t max_conn = 0;
    for (size_t i = 0; i < n_elems; ++i) {
        max_conn = std::max(max_conn, world.elements_static[i].num_connections);
    }

    // Build struct-of-arrays host staging buffers.
    // Per-element material properties are pre-resolved so the kernel avoids
    // an indirect material lookup.
    std::vector<val_t> h_transfer_coeff(n_elems);
    std::vector<val_t> h_external_flow(n_elems);
    std::vector<uint32_t> h_num_connections(n_elems);
    std::vector<uint32_t> h_conn_idx(n_elems * max_conn, 0);
    std::vector<val_t> h_conn_flux(n_elems * max_conn, 0.0);
    std::vector<val_t> h_energy(n_elems);
    std::vector<val_t> h_flux(n_elems);

    for (size_t i = 0; i < n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        const Material& mat = world.materials[es.material_idx];
        h_transfer_coeff[i] = mat.transfer_coeff;
        h_external_flow[i] = mat.external_flow;
        h_num_connections[i] = (uint32_t)es.num_connections;
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_conn_idx[j * n_elems + i] = (uint32_t)es.connected_idx[j];
            h_conn_flux[j * n_elems + i] = es.connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // Allocate device buffers
    val_t *d_transfer_coeff, *d_external_flow, *d_conn_flux;
    uint32_t *d_num_connections, *d_conn_idx;
    val_t *d_energy_a, *d_flux_a, *d_energy_b, *d_flux_b;

    CUDA_CHECK(cudaMalloc(&d_transfer_coeff, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_external_flow, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_num_connections, n_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx, n_elems * max_conn * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux, n_elems * max_conn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_a, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_a, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_b, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_b, n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_transfer_coeff, h_transfer_coeff.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_external_flow, h_external_flow.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_connections, h_num_connections.data(),
                          n_elems * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_idx, h_conn_idx.data(),
                          n_elems * max_conn * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux.data(),
                          n_elems * max_conn * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_energy_a, h_energy.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux_a, h_flux.data(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (int)((n_elems + block_size - 1) / block_size);

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationStepKernel<<<grid_size, block_size>>>(
            d_transfer_coeff, d_external_flow, d_num_connections,
            d_conn_idx, d_conn_flux,
            d_energy_a, d_flux_a, d_energy_b, d_flux_b, n_elems);

        // Swap buffers
        std::swap(d_energy_a, d_energy_b);
        std::swap(d_flux_a, d_flux_b);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back into the host world state
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy_a,
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux_a,
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }

    CUDA_CHECK(cudaFree(d_transfer_coeff));
    CUDA_CHECK(cudaFree(d_external_flow));
    CUDA_CHECK(cudaFree(d_num_connections));
    CUDA_CHECK(cudaFree(d_conn_idx));
    CUDA_CHECK(cudaFree(d_conn_flux));
    CUDA_CHECK(cudaFree(d_energy_a));
    CUDA_CHECK(cudaFree(d_flux_a));
    CUDA_CHECK(cudaFree(d_energy_b));
    CUDA_CHECK(cudaFree(d_flux_b));
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

    // Initialize the CUDA context up front so it is not counted in the
    // measured computation time
    CUDA_CHECK(cudaFree(nullptr));

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
