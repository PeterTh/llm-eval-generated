#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
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
    
    // MPI domain decomposition
    int mpi_rank;
    int mpi_size;
    idx_t local_start;
    idx_t local_count;
    idx_t global_count;
    
    // GPU device pointers
    ElementStatic* d_elements_static;
    ElementDynamic* d_elements_dynamic;
    ElementDynamic* d_elements_dynamic_swap;
    Material* d_materials;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    world.global_count = n_elems;
    
    // Distribute elements across MPI ranks (row-wise decomposition)
    idx_t elems_per_rank = (n_elems + world.mpi_size - 1) / world.mpi_size;
    world.local_start = world.mpi_rank * elems_per_rank;
    world.local_count = std::min(elems_per_rank, (idx_t)(n_elems - world.local_start));
    
    if (world.local_start >= n_elems) {
        world.local_count = 0;
    }
    
    // Initialize materials (all ranks have all materials)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements for local partition
    world.elements_static.resize(world.local_count);
    world.elements_dynamic.resize(world.local_count);
    world.elements_dynamic_swap.resize(world.local_count);
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for
    for (idx_t local_i = 0; local_i < world.local_count; ++local_i) {
        const idx_t global_i = world.local_start + local_i;
        const int x = global_i / n_elems_root;
        const int y = global_i % n_elems_root;
        
        ElementStatic& elem = world.elements_static[local_i];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        
        world.elements_dynamic[local_i].current_energy = 0.0;
        world.elements_dynamic[local_i].total_flux = 0.0;
        
        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            
            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = nx * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
        
        // Set corner elements as inflow/outflow
        const int last = n_elems_root - 1;
        if ((x == 0 && y == 0) || (x == last && y == last)) {
            elem.material_idx = INFLOW_MAT_ID;
        } else if ((x == 0 && y == last) || (x == last && y == 0)) {
            elem.material_idx = OUTFLOW_MAT_ID;
        }
    }
}

// Compute energy flux between two elements
__device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel for element update
__global__ void updateElementsKernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_out,
    const Material* __restrict__ materials,
    const idx_t n_elems,
    const ElementDynamic* __restrict__ global_dynamic_data,
    const idx_t global_offset
) {
    const idx_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_elems) return;
    
    const ElementStatic& elem_static = elements_static[idx];
    const ElementDynamic& elem_dyn = elements_dynamic[idx];
    const Material& mat = materials[elem_static.material_idx];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_global_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = global_dynamic_data[neighbor_global_idx];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }
    
    // Update element state
    elements_dynamic_out[idx].current_energy = elem_dyn.current_energy + total_flux;
    elements_dynamic_out[idx].total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Allocate GPU memory for world data
void allocateGPUMemory(World& world) {
    const size_t static_size = world.local_count * sizeof(ElementStatic);
    const size_t dynamic_size = world.local_count * sizeof(ElementDynamic);
    const size_t material_size = world.materials.size() * sizeof(Material);
    
    CUDA_CHECK(cudaMalloc(&world.d_elements_static, static_size));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic, dynamic_size));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic_swap, dynamic_size));
    CUDA_CHECK(cudaMalloc(&world.d_materials, material_size));
    
    // Copy static data to GPU (only once)
    CUDA_CHECK(cudaMemcpy(world.d_elements_static, world.elements_static.data(), 
                          static_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(), 
                          material_size, cudaMemcpyHostToDevice));
    
    // Copy initial dynamic data
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(), 
                          dynamic_size, cudaMemcpyHostToDevice));
}

// Free GPU memory
void freeGPUMemory(World& world) {
    if (world.d_elements_static) CUDA_CHECK(cudaFree(world.d_elements_static));
    if (world.d_elements_dynamic) CUDA_CHECK(cudaFree(world.d_elements_dynamic));
    if (world.d_elements_dynamic_swap) CUDA_CHECK(cudaFree(world.d_elements_dynamic_swap));
    if (world.d_materials) CUDA_CHECK(cudaFree(world.d_materials));
}

// Copy results from GPU back to host
void copyFromGPU(World& world) {
    const size_t dynamic_size = world.local_count * sizeof(ElementDynamic);
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic, 
                          dynamic_size, cudaMemcpyDeviceToHost));
}

// Run simulation for n_iters iterations with hybrid MPI+CUDA+OpenMP
void runSimulation(World& world, const int n_iters) {
    // Allocate buffer for gathering all dynamic data across MPI ranks
    std::vector<ElementDynamic> global_dynamic_buffer(world.global_count);
    
    // Allocate GPU memory for global dynamic data
    ElementDynamic* d_global_dynamic;
    CUDA_CHECK(cudaMalloc(&d_global_dynamic, world.global_count * sizeof(ElementDynamic)));
    
    // Gather counts and displacements for MPI_Allgatherv
    std::vector<int> recvcounts(world.mpi_size);
    std::vector<int> displs(world.mpi_size);
    
    int local_count_int = (int)world.local_count;
    MPI_Allgather(&local_count_int, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    displs[0] = 0;
    for (int i = 1; i < world.mpi_size; ++i) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    // Launch configuration for CUDA kernel
    const int blockSize = 256;
    const int numBlocks = (world.local_count + blockSize - 1) / blockSize;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Step 1: Gather all dynamic data from all MPI ranks
        MPI_Allgatherv(world.elements_dynamic.data(), local_count_int, 
                       MPI_BYTE, global_dynamic_buffer.data(), 
                       recvcounts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
        
        // Step 2: Copy global dynamic data to GPU
        CUDA_CHECK(cudaMemcpy(d_global_dynamic, global_dynamic_buffer.data(), 
                              world.global_count * sizeof(ElementDynamic), 
                              cudaMemcpyHostToDevice));
        
        // Step 3: Update local elements on GPU using global data
        if (world.local_count > 0) {
            updateElementsKernel<<<numBlocks, blockSize>>>(
                world.d_elements_static,
                world.d_elements_dynamic,
                world.d_elements_dynamic_swap,
                world.d_materials,
                world.local_count,
                d_global_dynamic,
                world.local_start
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Step 4: Swap device buffers
        ElementDynamic* temp = world.d_elements_dynamic;
        world.d_elements_dynamic = world.d_elements_dynamic_swap;
        world.d_elements_dynamic_swap = temp;
        
        // Step 5: Copy updated data back to host for next iteration's gather
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic, 
                              world.local_count * sizeof(ElementDynamic), 
                              cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_global_dynamic));
}

// Validate simulation results (MPI-aware)
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    #pragma omp parallel for reduction(+:local_energy_sum,local_flux_sum) \
                             reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }
    
    // Reduce across MPI ranks
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.mpi_rank == 0) {
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
    }
    
    int valid = 1;
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

// Compute a simple hash of the results for verification (MPI-aware)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    
    #pragma omp parallel for reduction(^:local_hash)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const idx_t global_i = world.local_start + i;
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    
    uint64_t global_hash;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_hash, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    
    return global_hash;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        int device_id = mpi_rank % num_devices;
        CUDA_CHECK(cudaSetDevice(device_id));
    }
    
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("========================================================================\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", num_devices);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (mpi_rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    
    World world;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    world.d_elements_static = nullptr;
    world.d_elements_dynamic = nullptr;
    world.d_elements_dynamic_swap = nullptr;
    world.d_materials = nullptr;
    
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    size_t global_total_mem;
    MPI_Reduce(&total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Memory usage per rank (avg): %.2f MB\n", 
               global_total_mem / (1024.0 * 1024.0 * mpi_size));
        printf("Total memory usage: %.2f MB\n", global_total_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Allocate GPU memory
    allocateGPUMemory(world);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (mpi_rank == 0) {
        printf("Running simulation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Copy final results from GPU
    copyFromGPU(world);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        // Gather all results to rank 0
        std::vector<ElementDynamic> all_elements;
        if (mpi_rank == 0) {
            all_elements.resize(n_elems);
        }
        
        std::vector<int> recvcounts(mpi_size);
        std::vector<int> displs(mpi_size);
        int local_count = (int)world.local_count;
        
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(world.elements_dynamic.data(), local_count, MPI_BYTE,
                    all_elements.data(), recvcounts.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            std::vector<double> energyData;
            energyData.reserve(all_elements.size());
            for (const auto& elem : all_elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            freeGPUMemory(world);
            MPI_Finalize();
            return 1;
        }
    }
    
    // Clean up
    freeGPUMemory(world);
    MPI_Finalize();
    
    return 0;
}
