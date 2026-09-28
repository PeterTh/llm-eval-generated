#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(err_));                      \
            exit(1);                                                          \
        }                                                                     \
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
__device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                    val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           transfer_coeff * connection_flux * 0.25;
}

// One simulation step: one thread per element. Connectivity is stored in a
// transposed (structure-of-arrays) layout so that connection j of consecutive
// elements is contiguous in memory, giving fully coalesced accesses.
__global__ void stepKernel(const int n_elems,
                           const val_t* __restrict__ energy_in,
                           const val_t* __restrict__ flux_in,
                           val_t* __restrict__ energy_out,
                           val_t* __restrict__ flux_out,
                           const unsigned int* __restrict__ conn_idx,
                           const val_t* __restrict__ conn_flux,
                           const unsigned int* __restrict__ num_conn,
                           const val_t* __restrict__ transfer_coeff,
                           const val_t* __restrict__ external_flow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const val_t this_energy = energy_in[i];
    const val_t coeff = transfer_coeff[i];
    const unsigned int nc = num_conn[i];

    // Start with external flow
    val_t total_flux = external_flow[i];

    // Add flux from all connected elements
    for (unsigned int j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_elems + i;
        const val_t other_energy = __ldg(&energy_in[conn_idx[slot]]);
        total_flux += computeFlux(coeff, this_energy, conn_flux[slot], other_energy);
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Run simulation for n_iters iterations on the GPU
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    const int n = static_cast<int>(n_elems);

    // Partition the elements across host worker threads (fewer threads for
    // small problems to keep thread-spawn overhead negligible)
    const size_t hw_threads = std::max<size_t>(std::thread::hardware_concurrency(), 1);
    const size_t n_workers = std::max<size_t>(
        std::min<size_t>(hw_threads, n_elems / 16384), 1);
    const size_t chunk = (n_elems + n_workers - 1) / n_workers;
    auto parallel_for = [&](auto&& body) {
        std::vector<std::thread> workers;
        workers.reserve(n_workers);
        for (size_t w = 0; w < n_workers; ++w) {
            const size_t begin = w * chunk;
            const size_t end = std::min(begin + chunk, n_elems);
            if (begin >= end) break;
            workers.emplace_back([&body, w, begin, end]() { body(w, begin, end); });
        }
        for (auto& t : workers) t.join();
    };

    // Determine the maximum connection count to size the transposed layout
    std::vector<unsigned int> worker_max(n_workers, 0);
    parallel_for([&](size_t w, size_t begin, size_t end) {
        unsigned int local_max = 0;
        for (size_t i = begin; i < end; ++i) {
            local_max = std::max(local_max,
                                 static_cast<unsigned int>(world.elements_static[i].num_connections));
        }
        worker_max[w] = local_max;
    });
    const unsigned int max_conn =
        *std::max_element(worker_max.begin(), worker_max.end());

    // Flatten static data into structure-of-arrays host buffers. The buffers
    // are allocated uninitialized and fully written by the flatten pass below
    // (connection slots past num_connections are never read by the kernel,
    // same as the original AoS layout).
    const size_t conn_slots = static_cast<size_t>(max_conn) * n_elems;
    std::unique_ptr<unsigned int[]> h_conn_idx(new unsigned int[conn_slots]);
    std::unique_ptr<val_t[]> h_conn_flux(new val_t[conn_slots]);
    std::unique_ptr<unsigned int[]> h_num_conn(new unsigned int[n_elems]);
    std::unique_ptr<val_t[]> h_coeff(new val_t[n_elems]);
    std::unique_ptr<val_t[]> h_extflow(new val_t[n_elems]);
    std::unique_ptr<val_t[]> h_energy(new val_t[n_elems]);
    std::unique_ptr<val_t[]> h_flux(new val_t[n_elems]);

    // Flatten in parallel: each worker handles a disjoint range of elements
    parallel_for([&](size_t, size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const ElementStatic& es = world.elements_static[i];
            const Material& mat = world.materials[es.material_idx];
            h_num_conn[i] = static_cast<unsigned int>(es.num_connections);
            h_coeff[i] = mat.transfer_coeff;
            h_extflow[i] = mat.external_flow;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                h_conn_idx[j * n_elems + i] = static_cast<unsigned int>(es.connected_idx[j]);
                h_conn_flux[j * n_elems + i] = es.connected_flux[j];
            }
            h_energy[i] = world.elements_dynamic[i].current_energy;
            h_flux[i] = world.elements_dynamic[i].total_flux;
        }
    });

    // Allocate device buffers
    unsigned int* d_conn_idx = nullptr;
    val_t* d_conn_flux = nullptr;
    unsigned int* d_num_conn = nullptr;
    val_t* d_coeff = nullptr;
    val_t* d_extflow = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};

    CUDA_CHECK(cudaMalloc(&d_conn_idx, conn_slots * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux, conn_slots * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_num_conn, n_elems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_coeff, n_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_extflow, n_elems * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_energy[b], n_elems * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_flux[b], n_elems * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(d_conn_idx, h_conn_idx.get(),
                          conn_slots * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux.get(),
                          conn_slots * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_conn, h_num_conn.get(),
                          n_elems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeff, h_coeff.get(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_extflow, h_extflow.get(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_energy[0], h_energy.get(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux[0], h_flux.get(),
                          n_elems * sizeof(val_t), cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = 1 - cur;
        stepKernel<<<grid_size, block_size>>>(n, d_energy[cur], d_flux[cur],
                                              d_energy[nxt], d_flux[nxt],
                                              d_conn_idx, d_conn_flux, d_num_conn,
                                              d_coeff, d_extflow);
        // Swap buffers
        cur = nxt;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to the host world state
    CUDA_CHECK(cudaMemcpy(h_energy.get(), d_energy[cur],
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_flux.get(), d_flux[cur],
                          n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    parallel_for([&](size_t, size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            world.elements_dynamic[i].current_energy = h_energy[i];
            world.elements_dynamic[i].total_flux = h_flux[i];
        }
    });

    CUDA_CHECK(cudaFree(d_conn_idx));
    CUDA_CHECK(cudaFree(d_conn_flux));
    CUDA_CHECK(cudaFree(d_num_conn));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_extflow));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_energy[b]));
        CUDA_CHECK(cudaFree(d_flux[b]));
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
    
    // Initialize the CUDA context before timing
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

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
