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

// CUDA error checking
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t _err = (call);                                                     \
        if (_err != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                                    \
        }                                                                                   \
    } while (0)

constexpr int N_MATERIALS = 3;
static_assert(N_MATERIALS >= 3, "Expected at least 3 materials");

__constant__ Material d_materials[N_MATERIALS];

__global__ void update_elements_kernel(const size_t n_elems,
                                      const idx_t* __restrict__ d_material_idx,
                                      const idx_t* __restrict__ d_num_connections,
                                      const idx_t* __restrict__ d_connected_idx,
                                      const val_t* __restrict__ d_connected_flux,
                                      const val_t* __restrict__ d_energy_in,
                                      const val_t* __restrict__ d_flux_in,
                                      val_t* __restrict__ d_energy_out,
                                      val_t* __restrict__ d_flux_out) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = tid; i < n_elems; i += stride) {
        const idx_t mat_id = d_material_idx[i];
        const Material mat = d_materials[mat_id];

        const val_t e_i = d_energy_in[i];
        val_t total_flux = mat.external_flow;

        const idx_t nconn = d_num_connections[i];
        const size_t base = i * static_cast<size_t>(MAX_CONNECTIONS);

        #pragma unroll
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (static_cast<idx_t>(j) >= nconn) break;
            const idx_t nidx = d_connected_idx[base + static_cast<size_t>(j)];
            const val_t cflux = d_connected_flux[base + static_cast<size_t>(j)];
            total_flux += (d_energy_in[nidx] - e_i) * mat.transfer_coeff * cflux * 0.25;
        }

        d_energy_out[i] = e_i + total_flux;
        d_flux_out[i] = d_flux_in[i] + fabs(total_flux);
    }
}

struct DeviceWorld {
    size_t n_elems = 0;
    idx_t* d_material_idx = nullptr;
    idx_t* d_num_connections = nullptr;
    idx_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;

    val_t* d_energy_a = nullptr;
    val_t* d_energy_b = nullptr;
    val_t* d_flux_a = nullptr;
    val_t* d_flux_b = nullptr;

    cudaStream_t stream = nullptr;
};

static DeviceWorld initDeviceWorld(const World& world) {
    DeviceWorld dev;
    dev.n_elems = world.elements_static.size();

    CUDA_CHECK(cudaStreamCreateWithFlags(&dev.stream, cudaStreamNonBlocking));

    // Marshal static connectivity into structure-of-arrays for better GPU memory behavior
    std::vector<idx_t> h_material_idx(dev.n_elems);
    std::vector<idx_t> h_num_connections(dev.n_elems);
    std::vector<idx_t> h_connected_idx(dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS));
    std::vector<val_t> h_connected_flux(dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS));

    for (size_t i = 0; i < dev.n_elems; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_material_idx[i] = es.material_idx;
        h_num_connections[i] = es.num_connections;
        const size_t base = i * static_cast<size_t>(MAX_CONNECTIONS);
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_connected_idx[base + static_cast<size_t>(j)] = es.connected_idx[j];
            h_connected_flux[base + static_cast<size_t>(j)] = es.connected_flux[j];
        }
    }

    // Dynamic state as SoA
    std::vector<val_t> h_energy(dev.n_elems);
    std::vector<val_t> h_flux(dev.n_elems);
    for (size_t i = 0; i < dev.n_elems; ++i) {
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_materials, world.materials.data(),
                                      sizeof(Material) * world.materials.size(), 0,
                                      cudaMemcpyHostToDevice, dev.stream));

    CUDA_CHECK(cudaMalloc(&dev.d_material_idx, dev.n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_num_connections, dev.n_elems * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_connected_idx, dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_connected_flux, dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&dev.d_energy_a, dev.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_energy_b, dev.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_flux_a, dev.n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.d_flux_b, dev.n_elems * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpyAsync(dev.d_material_idx, h_material_idx.data(), dev.n_elems * sizeof(idx_t),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_num_connections, h_num_connections.data(), dev.n_elems * sizeof(idx_t),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_connected_idx, h_connected_idx.data(),
                               dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(idx_t),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_connected_flux, h_connected_flux.data(),
                               dev.n_elems * static_cast<size_t>(MAX_CONNECTIONS) * sizeof(val_t),
                               cudaMemcpyHostToDevice, dev.stream));

    CUDA_CHECK(cudaMemcpyAsync(dev.d_energy_a, h_energy.data(), dev.n_elems * sizeof(val_t),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_flux_a, h_flux.data(), dev.n_elems * sizeof(val_t),
                               cudaMemcpyHostToDevice, dev.stream));

    CUDA_CHECK(cudaStreamSynchronize(dev.stream));
    return dev;
}

static void freeDeviceWorld(DeviceWorld& dev) {
    if (dev.d_material_idx) CUDA_CHECK(cudaFree(dev.d_material_idx));
    if (dev.d_num_connections) CUDA_CHECK(cudaFree(dev.d_num_connections));
    if (dev.d_connected_idx) CUDA_CHECK(cudaFree(dev.d_connected_idx));
    if (dev.d_connected_flux) CUDA_CHECK(cudaFree(dev.d_connected_flux));
    if (dev.d_energy_a) CUDA_CHECK(cudaFree(dev.d_energy_a));
    if (dev.d_energy_b) CUDA_CHECK(cudaFree(dev.d_energy_b));
    if (dev.d_flux_a) CUDA_CHECK(cudaFree(dev.d_flux_a));
    if (dev.d_flux_b) CUDA_CHECK(cudaFree(dev.d_flux_b));
    dev.d_material_idx = nullptr;
    dev.d_num_connections = nullptr;
    dev.d_connected_idx = nullptr;
    dev.d_connected_flux = nullptr;
    dev.d_energy_a = nullptr;
    dev.d_energy_b = nullptr;
    dev.d_flux_a = nullptr;
    dev.d_flux_b = nullptr;
    if (dev.stream) CUDA_CHECK(cudaStreamDestroy(dev.stream));
    dev.stream = nullptr;
}

static void downloadResults(World& world, const DeviceWorld& dev, const val_t* d_energy, const val_t* d_flux) {
    std::vector<val_t> h_energy(dev.n_elems);
    std::vector<val_t> h_flux(dev.n_elems);
    CUDA_CHECK(cudaMemcpyAsync(h_energy.data(), d_energy, dev.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_flux.data(), d_flux, dev.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost, dev.stream));
    CUDA_CHECK(cudaStreamSynchronize(dev.stream));

    world.elements_dynamic.resize(dev.n_elems);
    for (size_t i = 0; i < dev.n_elems; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

static void runSimulationDevice(const DeviceWorld& dev, const int n_iters,
                                val_t*& d_energy_final, val_t*& d_flux_final) {
    val_t* d_energy_in = dev.d_energy_a;
    val_t* d_energy_out = dev.d_energy_b;
    val_t* d_flux_in = dev.d_flux_a;
    val_t* d_flux_out = dev.d_flux_b;

    const int block = 256;
    int grid = static_cast<int>((dev.n_elems + block - 1) / block);
    grid = std::min(grid, 65535);

    for (int iter = 0; iter < n_iters; ++iter) {
        update_elements_kernel<<<grid, block, 0, dev.stream>>>(
            dev.n_elems,
            dev.d_material_idx,
            dev.d_num_connections,
            dev.d_connected_idx,
            dev.d_connected_flux,
            d_energy_in,
            d_flux_in,
            d_energy_out,
            d_flux_out);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_energy_in, d_energy_out);
        std::swap(d_flux_in, d_flux_out);
    }

    d_energy_final = d_energy_in;
    d_flux_final = d_flux_in;
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
    
    // Initialize GPU resources (excluded from timed region)
    DeviceWorld dev = initDeviceWorld(world);

    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    val_t* d_energy_final = nullptr;
    val_t* d_flux_final = nullptr;
    runSimulationDevice(dev, n_iters, d_energy_final, d_flux_final);
    CUDA_CHECK(cudaStreamSynchronize(dev.stream));

    auto end = std::chrono::high_resolution_clock::now();
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double duration_ms = static_cast<double>(duration_us) / 1000.0;

    // Download results (excluded from timed region)
    downloadResults(world, dev, d_energy_final, d_flux_final);
    freeDeviceWorld(dev);

    printf("Computation time: %.3f ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
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
