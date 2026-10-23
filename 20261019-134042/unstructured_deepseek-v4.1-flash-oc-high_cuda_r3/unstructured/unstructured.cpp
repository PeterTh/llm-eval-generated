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

struct World;

void freeWorldDevice(World& world);

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    Material* d_materials = nullptr;
    uint8_t* d_material_idx = nullptr;
    uint8_t* d_num_connections = nullptr;
    uint32_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;
    val_t* d_energy_a = nullptr;
    val_t* d_energy_b = nullptr;
    val_t* d_flux_a = nullptr;
    val_t* d_flux_b = nullptr;
    val_t* d_final_energy = nullptr;
    val_t* d_final_flux = nullptr;
    unsigned long long* d_hash = nullptr;
    uint64_t result_hash = 0;
    size_t device_n = 0;
    bool device_ready = false;

    ~World() { freeWorldDevice(*this); }
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

void freeWorldDevice(World& world) {
    cudaFree(world.d_materials);
    cudaFree(world.d_material_idx);
    cudaFree(world.d_num_connections);
    cudaFree(world.d_connected_idx);
    cudaFree(world.d_connected_flux);
    cudaFree(world.d_energy_a);
    cudaFree(world.d_energy_b);
    cudaFree(world.d_flux_a);
    cudaFree(world.d_flux_b);
    cudaFree(world.d_hash);
    world.d_materials = nullptr;
    world.d_material_idx = nullptr;
    world.d_num_connections = nullptr;
    world.d_connected_idx = nullptr;
    world.d_connected_flux = nullptr;
    world.d_energy_a = nullptr;
    world.d_energy_b = nullptr;
    world.d_flux_a = nullptr;
    world.d_flux_b = nullptr;
    world.d_final_energy = nullptr;
    world.d_final_flux = nullptr;
    world.d_hash = nullptr;
    world.result_hash = 0;
    world.device_n = 0;
    world.device_ready = false;
}

__global__ void simulationKernel(
    const uint8_t* __restrict__ material_idx,
    const uint8_t* __restrict__ num_connections,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const Material* __restrict__ materials,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ flux_out,
    const uint32_t n);

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

    // Flatten the mesh into structure-of-arrays form and transfer it to the GPU
    const size_t n = static_cast<size_t>(n_elems);
    std::vector<uint8_t> host_material_idx(n);
    std::vector<uint8_t> host_num_connections(n);
    std::vector<uint32_t> host_connected_idx(n * MAX_CONNECTIONS, 0);
    std::vector<val_t> host_connected_flux(n * MAX_CONNECTIONS, 0.0);

    for (size_t i = 0; i < n; ++i) {
        const ElementStatic& elem = world.elements_static[i];
        host_material_idx[i] = static_cast<uint8_t>(elem.material_idx);
        host_num_connections[i] = static_cast<uint8_t>(elem.num_connections);
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const size_t offset = static_cast<size_t>(j) * n + i;
            host_connected_idx[offset] = static_cast<uint32_t>(elem.connected_idx[j]);
            host_connected_flux[offset] = elem.connected_flux[j];
        }
    }

    freeWorldDevice(world);
    world.device_n = n;
    world.result_hash = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_materials),
                          world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));

    if (n > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_material_idx),
                              n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_num_connections),
                              n * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_connected_idx),
                              n * MAX_CONNECTIONS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_connected_flux),
                              n * MAX_CONNECTIONS * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_energy_a),
                              n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_energy_b),
                              n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_flux_a),
                              n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_flux_b),
                              n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&world.d_hash),
                              sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemcpy(world.d_material_idx, host_material_idx.data(),
                              n * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_num_connections, host_num_connections.data(),
                              n * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_connected_idx, host_connected_idx.data(),
                              n * MAX_CONNECTIONS * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_connected_flux, host_connected_flux.data(),
                              n * MAX_CONNECTIONS * sizeof(val_t),
                              cudaMemcpyHostToDevice));

        std::vector<val_t> host_energy(n, 0.0);
        std::vector<val_t> host_flux(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            host_energy[i] = world.elements_dynamic[i].current_energy;
            host_flux[i] = world.elements_dynamic[i].total_flux;
        }
        CUDA_CHECK(cudaMemcpy(world.d_energy_a, host_energy.data(),
                              n * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_flux_a, host_flux.data(),
                              n * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    world.device_ready = true;

    if (n > 0) {
        simulationKernel<<<1, 1>>>(
            world.d_material_idx, world.d_num_connections,
            world.d_connected_idx, world.d_connected_flux, world.d_materials,
            world.d_energy_a, world.d_flux_a, world.d_energy_b, world.d_flux_b,
            static_cast<uint32_t>(n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

__global__ void simulationKernel(
    const uint8_t* __restrict__ material_idx,
    const uint8_t* __restrict__ num_connections,
    const uint32_t* __restrict__ connected_idx,
    const val_t* __restrict__ connected_flux,
    const Material* __restrict__ materials,
    const val_t* __restrict__ energy_in,
    const val_t* __restrict__ flux_in,
    val_t* __restrict__ energy_out,
    val_t* __restrict__ flux_out,
    const uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const Material mat = materials[material_idx[i]];
    const val_t elem_energy = energy_in[i];
    val_t total_flux = mat.external_flow;

    const uint32_t n_conn = num_connections[i];
    size_t offset = i;
    for (uint32_t j = 0; j < n_conn; ++j, offset += n) {
        const val_t other_energy = energy_in[connected_idx[offset]];
        total_flux += (other_energy - elem_energy) * mat.transfer_coeff *
                      connected_flux[offset] * 0.25;
    }

    energy_out[i] = elem_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

__global__ void hashKernel(const val_t* __restrict__ energy,
                           const val_t* __restrict__ flux,
                           const uint32_t n,
                           unsigned long long* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t h = 0;
    if (i < n) {
        const uint64_t e =
            static_cast<uint64_t>(__double_as_longlong(energy[i]));
        const uint64_t f =
            static_cast<uint64_t>(__double_as_longlong(flux[i]));
        h = ((e + i) * 0x9e3779b97f4a7c15ULL) ^
            ((f + i) * 0xbf58476d1ce4e5b9ULL);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        h ^= __shfl_xor_sync(0xffffffffu, h, offset);
    }
    if ((threadIdx.x & 31u) == 0u) {
        atomicXor(out, static_cast<unsigned long long>(h));
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    if (!world.device_ready || world.device_n != n_elems) {
        fprintf(stderr, "GPU mesh is not initialized\n");
        std::exit(1);
    }
    if (n_elems == 0) {
        return;
    }

    val_t* energy_in = world.d_energy_a;
    val_t* energy_out = world.d_energy_b;
    val_t* flux_in = world.d_flux_a;
    val_t* flux_out = world.d_flux_b;

    constexpr int threads = 256;
    const uint32_t blocks =
        static_cast<uint32_t>((n_elems + threads - 1) / threads);

    for (int iter = 0; iter < n_iters; ++iter) {
        simulationKernel<<<blocks, threads>>>(
            world.d_material_idx, world.d_num_connections,
            world.d_connected_idx, world.d_connected_flux, world.d_materials,
            energy_in, flux_in, energy_out, flux_out,
            static_cast<uint32_t>(n_elems));

        std::swap(energy_in, energy_out);
        std::swap(flux_in, flux_out);
    }

    CUDA_CHECK(cudaMemsetAsync(world.d_hash, 0, sizeof(unsigned long long)));
    hashKernel<<<blocks, threads>>>(energy_in, flux_in,
                                    static_cast<uint32_t>(n_elems),
                                    world.d_hash);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&world.result_hash, world.d_hash,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    world.d_energy_a = energy_in;
    world.d_energy_b = energy_out;
    world.d_flux_a = flux_in;
    world.d_flux_b = flux_out;
    world.d_final_energy = energy_in;
    world.d_final_flux = flux_in;
}

void copyResultsToHost(World& world) {
    const size_t n_elems = world.device_n;
    if (n_elems == 0 || world.d_final_energy == nullptr) {
        return;
    }

    std::vector<val_t> host_energy(n_elems);
    std::vector<val_t> host_flux(n_elems);
    CUDA_CHECK(cudaMemcpy(host_energy.data(), world.d_final_energy,
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_flux.data(), world.d_final_flux,
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = host_energy[i];
        world.elements_dynamic[i].total_flux = host_flux[i];
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
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double duration_s = static_cast<double>(duration_ns) / 1e9;
    const double time_per_iter = static_cast<double>(duration_ns) / n_measured_iters / 1e6;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / duration_s / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    printf("Performance:\n");
    printf("  Time per iteration: %.4f ms\n", time_per_iter);
    printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    printf("  Performance: %.4f GFLOPS\n", gflops);
    
    // Compute hash for verification
    const uint64_t hash = world.result_hash;
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    if (printResults || validate) {
        copyResultsToHost(world);
    }
    
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
