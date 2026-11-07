// Unstructured mesh energy transfer benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt). The per-element update
// is executed on the GPU, one thread per mesh element. The mesh connectivity is
// stored in a transposed (structure-of-arrays) layout so that consecutive
// threads read consecutive memory locations (coalesced access).

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

// Compact index type used for device connectivity (n_elems always fits in an int)
using dev_idx_t = uint32_t;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state.
//
// Static connectivity is stored structure-of-arrays, transposed so that the
// j-th connection of all elements is contiguous: connected_idx[j * n + i].
struct World {
    std::vector<Material> materials;

    // Static element data (host copies)
    size_t n_elems = 0;
    std::vector<uint8_t> material_idx;          // per element
    std::vector<uint8_t> num_connections;       // per element
    std::vector<dev_idx_t> connected_idx;       // [MAX_CONNECTIONS][n_elems]
    std::vector<val_t> connected_flux;          // [MAX_CONNECTIONS][n_elems]

    // Dynamic element data (host copy, valid after download)
    std::vector<ElementDynamic> elements_dynamic;

    // Device buffers
    Material* d_materials = nullptr;
    uint8_t* d_material_idx = nullptr;
    uint8_t* d_num_connections = nullptr;
    dev_idx_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;
    val_t* d_energy = nullptr;        // current energy (read)
    val_t* d_flux = nullptr;          // accumulated flux (read)
    val_t* d_energy_swap = nullptr;   // write buffers
    val_t* d_flux_swap = nullptr;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                           \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    world.n_elems = n_elems;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements; default material, no connections, zero energy
    world.material_idx.assign(n_elems, static_cast<uint8_t>(DEFAULT_MAT_ID));
    world.num_connections.assign(n_elems, 0);
    world.connected_idx.assign(static_cast<size_t>(MAX_CONNECTIONS) * n_elems, 0);
    world.connected_flux.assign(static_cast<size_t>(MAX_CONNECTIONS) * n_elems, 0.0);
    world.elements_dynamic.assign(n_elems, ElementDynamic{0.0, 0.0});

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(x) * n_elems_root + y;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            uint8_t nc = 0;
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const size_t neighbor_idx = static_cast<size_t>(nx) * n_elems_root + ny;
                    world.connected_idx[nc * n_elems + idx] = static_cast<dev_idx_t>(neighbor_idx);
                    world.connected_flux[nc * n_elems + idx] = 1.0;
                    nc++;
                }
            }
            world.num_connections[idx] = nc;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const size_t last = n_elems_root - 1;
    world.material_idx[0 * n_elems_root + 0] = INFLOW_MAT_ID;
    world.material_idx[0 * n_elems_root + last] = OUTFLOW_MAT_ID;
    world.material_idx[last * n_elems_root + 0] = OUTFLOW_MAT_ID;
    world.material_idx[last * n_elems_root + last] = INFLOW_MAT_ID;
}

// Upload static mesh data and initial dynamic state to the GPU
void uploadWorld(World& world) {
    const size_t n = world.n_elems;
    const size_t conn = static_cast<size_t>(MAX_CONNECTIONS) * n;

    CUDA_CHECK(cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&world.d_material_idx, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&world.d_num_connections, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_idx, conn * sizeof(dev_idx_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_flux, conn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_energy, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_flux, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_energy_swap, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_flux_swap, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_material_idx, world.material_idx.data(),
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_num_connections, world.num_connections.data(),
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_connected_idx, world.connected_idx.data(),
                          conn * sizeof(dev_idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_connected_flux, world.connected_flux.data(),
                          conn * sizeof(val_t), cudaMemcpyHostToDevice));

    std::vector<val_t> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = world.elements_dynamic[i].current_energy;
    CUDA_CHECK(cudaMemcpy(world.d_energy, tmp.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < n; ++i) tmp[i] = world.elements_dynamic[i].total_flux;
    CUDA_CHECK(cudaMemcpy(world.d_flux, tmp.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
}

// Download the current dynamic state from the GPU into world.elements_dynamic
void downloadWorld(World& world) {
    const size_t n = world.n_elems;
    std::vector<val_t> energy(n), flux(n);
    CUDA_CHECK(cudaMemcpy(energy.data(), world.d_energy, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), world.d_flux, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
    }
}

void freeWorld(World& world) {
    cudaFree(world.d_materials);
    cudaFree(world.d_material_idx);
    cudaFree(world.d_num_connections);
    cudaFree(world.d_connected_idx);
    cudaFree(world.d_connected_flux);
    cudaFree(world.d_energy);
    cudaFree(world.d_flux);
    cudaFree(world.d_energy_swap);
    cudaFree(world.d_flux_swap);
}

// Compute energy flux between two elements
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// One simulation step: each thread updates one element.
// Connections are evaluated in the same order as the sequential reference so
// the floating-point results are bit-identical.
__global__ void __launch_bounds__(256)
simulationStepKernel(const Material* __restrict__ materials,
                     const uint8_t* __restrict__ material_idx,
                     const uint8_t* __restrict__ num_connections,
                     const dev_idx_t* __restrict__ connected_idx,
                     const val_t* __restrict__ connected_flux,
                     const val_t* __restrict__ energy_in,
                     const val_t* __restrict__ flux_in,
                     val_t* __restrict__ energy_out,
                     val_t* __restrict__ flux_out,
                     const unsigned int n_elems) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const Material mat = materials[material_idx[i]];
    const val_t this_energy = energy_in[i];
    const unsigned int nc = num_connections[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    #pragma unroll
    for (unsigned int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const size_t slot = static_cast<size_t>(j) * n_elems + i;
            const dev_idx_t neighbor_idx = __ldg(&connected_idx[slot]);
            const val_t cflux = __ldg(&connected_flux[slot]);
            const val_t neighbor_energy = __ldg(&energy_in[neighbor_idx]);
            total_flux += computeFlux(mat.transfer_coeff, this_energy, cflux, neighbor_energy);
        }
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const unsigned int n_elems = static_cast<unsigned int>(world.n_elems);
    constexpr int block = 256;
    const unsigned int grid = (n_elems + block - 1) / block;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        simulationStepKernel<<<grid, block>>>(
            world.d_materials, world.d_material_idx, world.d_num_connections,
            world.d_connected_idx, world.d_connected_flux,
            world.d_energy, world.d_flux, world.d_energy_swap, world.d_flux_swap, n_elems);

        // Swap buffers
        std::swap(world.d_energy, world.d_energy_swap);
        std::swap(world.d_flux, world.d_flux_swap);
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
    uploadWorld(world);

    // Calculate memory usage (device-side layout)
    const size_t static_mem = world.n_elems * (2 * sizeof(uint8_t)) +
                              world.connected_idx.size() * sizeof(dev_idx_t) +
                              world.connected_flux.size() * sizeof(val_t);
    const size_t dynamic_mem = world.n_elems * sizeof(ElementDynamic) * 2;
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

    downloadWorld(world);

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

    int ret = 0;
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            ret = 1;
        }
    }

    freeWorld(world);
    return ret;
}
