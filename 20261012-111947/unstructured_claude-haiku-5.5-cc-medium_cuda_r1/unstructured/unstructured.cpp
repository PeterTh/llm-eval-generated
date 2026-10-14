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

// Threads per block for the element update kernel
constexpr int THREADS_PER_BLOCK = 256;

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

// Device-resident copy of the world. Stored as structure-of-arrays so that
// consecutive threads read consecutive memory, and so the neighbor gather
// only touches the energy arrays.
struct DeviceWorld {
    size_t n_elems = 0;
    Material* materials = nullptr;
    uint32_t* material_idx = nullptr;                    // [n_elems]
    uint32_t* num_connections = nullptr;                 // [n_elems]
    uint32_t* connected_idx = nullptr;                   // [MAX_CONNECTIONS][n_elems]
    val_t* connected_flux = nullptr;                     // [MAX_CONNECTIONS][n_elems]
    val_t* energy[2] = {nullptr, nullptr};               // Double-buffered energy
    val_t* flux[2] = {nullptr, nullptr};                 // Double-buffered total flux
    int cur = 0;                                         // Buffer holding the latest state
};

// Abort on any CUDA runtime error
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update every element for one iteration: reads the energy/flux of the
// current state and writes the next state. One thread per element.
__global__ void updateElements(size_t n_elems, const Material* __restrict__ materials,
                               const uint32_t* __restrict__ material_idx,
                               const uint32_t* __restrict__ num_connections,
                               const uint32_t* __restrict__ connected_idx,
                               const val_t* __restrict__ connected_flux,
                               const val_t* __restrict__ energy_in,
                               const val_t* __restrict__ flux_in,
                               val_t* __restrict__ energy_out,
                               val_t* __restrict__ flux_out) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) {
        return;
    }

    const Material& mat = materials[material_idx[i]];
    const val_t this_energy = energy_in[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (same order as the serial reference)
    const uint32_t n_conn = num_connections[i];
    for (uint32_t j = 0; j < n_conn; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_elems + i;
        const uint32_t neighbor_idx = connected_idx[slot];
        total_flux += computeFlux(mat, this_energy, connected_flux[slot], energy_in[neighbor_idx]);
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Allocate device buffers and upload the mesh and initial state
void uploadWorld(const World& world, DeviceWorld& dw) {
    const size_t n = world.elements_static.size();
    if (n > UINT32_MAX) {
        fprintf(stderr, "Mesh has too many elements for 32-bit indices\n");
        exit(1);
    }
    dw.n_elems = n;

    // Repack the array-of-structs host data into device structure-of-arrays
    std::vector<uint32_t> h_material_idx(n);
    std::vector<uint32_t> h_num_connections(n);
    std::vector<uint32_t> h_connected_idx(MAX_CONNECTIONS * n, 0);
    std::vector<val_t> h_connected_flux(MAX_CONNECTIONS * n, 0.0);
    std::vector<val_t> h_energy(n);
    std::vector<val_t> h_flux(n);
    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        h_material_idx[i] = static_cast<uint32_t>(elem.material_idx);
        h_num_connections[i] = static_cast<uint32_t>(elem.num_connections);
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            h_connected_idx[j * n + i] = static_cast<uint32_t>(elem.connected_idx[j]);
            h_connected_flux[j * n + i] = elem.connected_flux[j];
        }
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMalloc(&dw.materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&dw.material_idx, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dw.num_connections, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dw.connected_idx, MAX_CONNECTIONS * n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dw.connected_flux, MAX_CONNECTIONS * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.energy[0], n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.energy[1], n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.flux[0], n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dw.flux[1], n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(dw.materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.material_idx, h_material_idx.data(),
                          n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.num_connections, h_num_connections.data(),
                          n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.connected_idx, h_connected_idx.data(),
                          MAX_CONNECTIONS * n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.connected_flux, h_connected_flux.data(),
                          MAX_CONNECTIONS * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.energy[0], h_energy.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw.flux[0], h_flux.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    // Buffers [1] are fully overwritten by the first iteration, so they need no upload
    dw.cur = 0;
}

// Copy the final state back into the host world
void downloadWorld(const DeviceWorld& dw, World& world) {
    const size_t n = dw.n_elems;
    std::vector<val_t> h_energy(n);
    std::vector<val_t> h_flux(n);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), dw.energy[dw.cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.data(), dw.flux[dw.cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

void freeWorld(DeviceWorld& dw) {
    cudaFree(dw.materials);
    cudaFree(dw.material_idx);
    cudaFree(dw.num_connections);
    cudaFree(dw.connected_idx);
    cudaFree(dw.connected_flux);
    cudaFree(dw.energy[0]);
    cudaFree(dw.energy[1]);
    cudaFree(dw.flux[0]);
    cudaFree(dw.flux[1]);
    dw = DeviceWorld{};
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(DeviceWorld& dw, const int n_iters) {
    if (dw.n_elems == 0) {
        return;
    }

    const size_t n_elems = dw.n_elems;
    const unsigned int blocks = static_cast<unsigned int>((n_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    for (int iter = 0; iter < n_iters; ++iter) {
        const int in = dw.cur;
        const int out = 1 - dw.cur;

        updateElements<<<blocks, THREADS_PER_BLOCK>>>(n_elems, dw.materials, dw.material_idx,
                                                      dw.num_connections, dw.connected_idx,
                                                      dw.connected_flux, dw.energy[in], dw.flux[in],
                                                      dw.energy[out], dw.flux[out]);

        // Swap buffers
        dw.cur = out;
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

    // Upload the mesh to the GPU (setup, not included in the timed computation)
    DeviceWorld device_world;
    uploadWorld(world, device_world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(device_world, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Bring the final state back to the host for verification and output
    downloadWorld(device_world, world);
    freeWorld(device_world);

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
