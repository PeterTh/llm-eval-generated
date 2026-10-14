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

// Device-side mesh and state. Structure-of-arrays layout so that consecutive
// threads touch consecutive addresses. Element indices are 32-bit on the device:
// n_elems_root * n_elems_root is held in an int, so the element count fits.
struct DeviceWorld {
    size_t n_elems = 0;
    Material* materials = nullptr;
    uint32_t* material_idx = nullptr;
    uint32_t* num_connections = nullptr;
    uint32_t* connected_idx = nullptr;      // [MAX_CONNECTIONS][n_elems], plane-major
    val_t* connected_flux = nullptr;        // [MAX_CONNECTIONS][n_elems], plane-major
    val_t* energy[2] = {nullptr, nullptr};  // Ping-pong buffers
    val_t* flux[2] = {nullptr, nullptr};    // Ping-pong buffers
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(1);                                                         \
        }                                                                    \
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

// Compute energy flux between two elements
__device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                    val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// One thread per element: reads the current energies, writes the next state.
// Floating-point operations run in the same order as the sequential reference.
__global__ void stepKernel(const DeviceWorld w, const int src) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= w.n_elems) {
        return;
    }

    const int dst = 1 - src;
    const val_t* energy_in = w.energy[src];
    const val_t* flux_in = w.flux[src];

    const Material mat = w.materials[w.material_idx[i]];
    const val_t self_energy = energy_in[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const uint32_t n_conn = w.num_connections[i];
    for (uint32_t j = 0; j < n_conn; ++j) {
        const size_t k = static_cast<size_t>(j) * w.n_elems + i;
        const uint32_t neighbor_idx = w.connected_idx[k];
        total_flux += computeFlux(mat, self_energy, w.connected_flux[k], energy_in[neighbor_idx]);
    }

    // Update element state
    w.energy[dst][i] = self_energy + total_flux;
    w.flux[dst][i] = flux_in[i] + fabs(total_flux);
}

// Allocate a device array of `count` elements
template <typename T>
T* allocDevice(size_t count) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, count * sizeof(T)));
    return d;
}

// Copy a host vector to a new device array
template <typename T>
T* toDevice(const std::vector<T>& host) {
    T* d = allocDevice<T>(host.size());
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Repack the mesh into device arrays and upload it along with the initial state
DeviceWorld uploadWorld(const World& world) {
    DeviceWorld dw;
    const size_t n = world.elements_static.size();
    dw.n_elems = n;

    std::vector<uint32_t> material_idx(n);
    std::vector<uint32_t> num_connections(n);
    std::vector<uint32_t> connected_idx(MAX_CONNECTIONS * n, 0);
    std::vector<val_t> connected_flux(MAX_CONNECTIONS * n, 0.0);
    std::vector<val_t> energy(n);
    std::vector<val_t> flux(n);

    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        material_idx[i] = static_cast<uint32_t>(elem.material_idx);
        num_connections[i] = static_cast<uint32_t>(elem.num_connections);
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            connected_idx[j * n + i] = static_cast<uint32_t>(elem.connected_idx[j]);
            connected_flux[j * n + i] = elem.connected_flux[j];
        }
        energy[i] = world.elements_dynamic[i].current_energy;
        flux[i] = world.elements_dynamic[i].total_flux;
    }

    dw.materials = toDevice(world.materials);
    dw.material_idx = toDevice(material_idx);
    dw.num_connections = toDevice(num_connections);
    dw.connected_idx = toDevice(connected_idx);
    dw.connected_flux = toDevice(connected_flux);
    dw.energy[0] = toDevice(energy);
    dw.energy[1] = allocDevice<val_t>(n);
    dw.flux[0] = toDevice(flux);
    dw.flux[1] = allocDevice<val_t>(n);
    return dw;
}

// Release all device memory held by the world
void freeWorld(DeviceWorld& dw) {
    CUDA_CHECK(cudaFree(dw.materials));
    CUDA_CHECK(cudaFree(dw.material_idx));
    CUDA_CHECK(cudaFree(dw.num_connections));
    CUDA_CHECK(cudaFree(dw.connected_idx));
    CUDA_CHECK(cudaFree(dw.connected_flux));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(dw.energy[b]));
        CUDA_CHECK(cudaFree(dw.flux[b]));
    }
}

// Run simulation for n_iters iterations. State stays on the device; each
// iteration is one kernel launch on the default stream, so launches are
// queued back-to-back. Returns the buffer index holding the final state.
int runSimulation(const DeviceWorld& dw, const int n_iters) {
    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((dw.n_elems + threads_per_block - 1) / threads_per_block);

    int cur = 0;
    if (dw.n_elems > 0) {
        for (int iter = 0; iter < n_iters; ++iter) {
            stepKernel<<<blocks, threads_per_block>>>(dw, cur);
            CUDA_CHECK(cudaGetLastError());
            cur = 1 - cur;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return cur;
}

// Copy the final state back into the host world
void downloadResults(const DeviceWorld& dw, const int cur, World& world) {
    const size_t n = dw.n_elems;
    std::vector<val_t> energy(n);
    std::vector<val_t> flux(n);
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(energy.data(), dw.energy[cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flux.data(), dw.flux[cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < n; ++i) {
        world.elements_dynamic[i].current_energy = energy[i];
        world.elements_dynamic[i].total_flux = flux[i];
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

    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");

    // Upload the mesh and initial state to the GPU (setup, excluded from timing)
    DeviceWorld dw = uploadWorld(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const int final_buf = runSimulation(dw, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    const double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    downloadResults(dw, final_buf, world);
    freeWorld(dw);

    printf("Computation time: %.3f ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

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
