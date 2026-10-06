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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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
__host__ __device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Device-side mesh representation (structure of arrays for coalesced access)
struct DeviceMesh {
    size_t n_elems = 0;
    size_t n_materials = 0;
    Material* materials = nullptr;   // [n_materials]
    uint32_t* meta = nullptr;        // [n_elems]: num_connections | (material_idx << 8)
    uint32_t* conn_idx = nullptr;    // [MAX_CONNECTIONS * n_elems], column-major (j * n + i)
    val_t* conn_flux = nullptr;      // [MAX_CONNECTIONS * n_elems], column-major (j * n + i)
    val_t* energy[2] = {nullptr, nullptr};
    val_t* tflux[2] = {nullptr, nullptr};
    ElementDynamic* packed = nullptr; // AoS output buffer [n_elems]
    int cur = 0;
};

static DeviceMesh g_dev;

// Upload the mesh (static connectivity + initial state) to the GPU
void uploadWorld(const World& world) {
    const size_t n = world.elements_static.size();
    DeviceMesh& d = g_dev;
    d.n_elems = n;
    d.n_materials = world.materials.size();

    std::vector<uint32_t> meta(n);
    std::vector<uint32_t> conn_idx(MAX_CONNECTIONS * n, 0);
    std::vector<val_t> conn_flux(MAX_CONNECTIONS * n, 0.0);
    std::vector<val_t> energy(n), tflux(n);
    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& e = world.elements_static[i];
        meta[i] = static_cast<uint32_t>(e.num_connections) |
                  (static_cast<uint32_t>(e.material_idx) << 8);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            conn_idx[j * n + i] = static_cast<uint32_t>(e.connected_idx[j]);
            conn_flux[j * n + i] = e.connected_flux[j];
        }
        energy[i] = world.elements_dynamic[i].current_energy;
        tflux[i] = world.elements_dynamic[i].total_flux;
    }

    CUDA_CHECK(cudaMalloc(&d.materials, std::max<size_t>(d.n_materials, 1) * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d.meta, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.conn_idx, MAX_CONNECTIONS * n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.conn_flux, MAX_CONNECTIONS * n * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d.energy[b], n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.tflux[b], n * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&d.packed, n * sizeof(ElementDynamic)));

    CUDA_CHECK(cudaMemcpy(d.materials, world.materials.data(),
                          d.n_materials * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.meta, meta.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.conn_idx, conn_idx.data(), MAX_CONNECTIONS * n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.conn_flux, conn_flux.data(), MAX_CONNECTIONS * n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.energy[0], energy.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.tflux[0], tflux.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    d.cur = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeDevice() {
    DeviceMesh& d = g_dev;
    cudaFree(d.materials);
    cudaFree(d.meta);
    cudaFree(d.conn_idx);
    cudaFree(d.conn_flux);
    for (int b = 0; b < 2; ++b) {
        cudaFree(d.energy[b]);
        cudaFree(d.tflux[b]);
    }
    cudaFree(d.packed);
    d = DeviceMesh{};
}

// One simulation step: each thread updates one element
__global__ void __launch_bounds__(256)
stepKernel(const size_t n, const Material* __restrict__ materials,
           const uint32_t* __restrict__ meta, const uint32_t* __restrict__ conn_idx,
           const val_t* __restrict__ conn_flux, const val_t* __restrict__ energy_in,
           const val_t* __restrict__ tflux_in, val_t* __restrict__ energy_out,
           val_t* __restrict__ tflux_out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const uint32_t m = __ldg(&meta[i]);
        const uint32_t nconn = m & 0xFFu;
        const uint32_t mat_idx = m >> 8;
        const val_t transfer_coeff = __ldg(&materials[mat_idx].transfer_coeff);
        const val_t external_flow = __ldg(&materials[mat_idx].external_flow);
        const val_t e = __ldg(&energy_in[i]);

        // Start with external flow
        val_t total_flux = external_flow;

        // Add flux from all connected elements (same order as the reference)
#pragma unroll
        for (uint32_t j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j < nconn) {
                const uint32_t nb = __ldg(&conn_idx[j * n + i]);
                const val_t cf = __ldg(&conn_flux[j * n + i]);
                total_flux += computeFlux(transfer_coeff, e, cf, __ldg(&energy_in[nb]));
            }
        }

        energy_out[i] = e + total_flux;
        tflux_out[i] = __ldg(&tflux_in[i]) + fabs(total_flux);
    }
}

// Pack SoA device state into the AoS host layout
__global__ void packKernel(const size_t n, const val_t* __restrict__ energy,
                           const val_t* __restrict__ tflux, ElementDynamic* __restrict__ out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i].current_energy = energy[i];
        out[i].total_flux = tflux[i];
    }
}

static unsigned int gridFor(size_t n, int block) {
    size_t blocks = (n + block - 1) / block;
    const size_t max_blocks = 1u << 30;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks == 0) blocks = 1;
    return static_cast<unsigned int>(blocks);
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    DeviceMesh& d = g_dev;
    const size_t n = d.n_elems;
    constexpr int BLOCK = 256;
    const unsigned int grid = gridFor(n, BLOCK);

    if (n > 0) {
        for (int iter = 0; iter < n_iters; ++iter) {
            const int nxt = d.cur ^ 1;
            stepKernel<<<grid, BLOCK>>>(n, d.materials, d.meta, d.conn_idx, d.conn_flux,
                                        d.energy[d.cur], d.tflux[d.cur], d.energy[nxt],
                                        d.tflux[nxt]);
            // Swap buffers
            d.cur = nxt;
        }
        CUDA_CHECK(cudaGetLastError());

        packKernel<<<grid, BLOCK>>>(n, d.energy[d.cur], d.tflux[d.cur], d.packed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d.packed,
                              n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaFree(0));  // initialize CUDA context outside the timed region
    uploadWorld(world);
    
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
    freeDevice();

    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            return 1;
        }
    }
    
    return 0;
}
