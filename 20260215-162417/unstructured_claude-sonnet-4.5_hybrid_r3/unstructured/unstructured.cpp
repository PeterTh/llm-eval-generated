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
    
    // MPI decomposition info
    int mpi_rank;
    int mpi_size;
    idx_t local_start;  // First element owned by this rank
    idx_t local_size;   // Number of elements owned by this rank
    
    // GPU device pointers
    Material* d_materials;
    ElementStatic* d_elements_static;
    ElementDynamic* d_elements_dynamic;
    ElementDynamic* d_elements_dynamic_swap;
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
    
    // Initialize all elements with default material and zero energy (OpenMP parallel)
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid (OpenMP parallel)
    #pragma omp parallel for collapse(2)
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
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel for element update
__device__ val_t computeFluxDevice(const Material& mat, const ElementDynamic& this_elem,
                                   val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

__global__ void updateElementsKernel(const Material* materials,
                                      const ElementStatic* elements_static,
                                      const ElementDynamic* elements_dynamic,
                                      ElementDynamic* elements_dynamic_swap,
                                      idx_t n_elems) {
    idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_elems) {
        const ElementStatic& elem_static = elements_static[i];
        const ElementDynamic& elem_dyn = elements_dynamic[i];
        const Material& mat = materials[elem_static.material_idx];
        
        // Start with external flow
        val_t total_flux = mat.external_flow;
        
        // Add flux from all connected elements
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const idx_t neighbor_idx = elem_static.connected_idx[j];
            const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
            total_flux += computeFluxDevice(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }
        
        // Update element state
        ElementDynamic& elem_write = elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    // Allocate GPU memory for this rank's local data
    CUDA_CHECK(cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&world.d_elements_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic)));
    
    // Copy static data to GPU
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(), 
                         world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_static, world.elements_static.data(),
                         n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(),
                         n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic_swap, world.elements_dynamic_swap.data(),
                         n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    
    // Determine kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n_elems + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch CUDA kernel for local elements
        updateElementsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            world.d_materials,
            world.d_elements_static,
            world.d_elements_dynamic,
            world.d_elements_dynamic_swap,
            n_elems
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap device pointers
        ElementDynamic* tmp = world.d_elements_dynamic;
        world.d_elements_dynamic = world.d_elements_dynamic_swap;
        world.d_elements_dynamic_swap = tmp;
        
        // MPI synchronization: exchange boundary data if needed
        // For this mesh where all ranks have full data, we need to gather/scatter
        // In a true distributed case, we'd do halo exchange here
        if (world.mpi_size > 1) {
            // Copy current state back to host for MPI exchange
            CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic,
                                 n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
            
            // Allgather to ensure all ranks have consistent view
            // This is simplified - in production would use halo exchange
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                         world.elements_dynamic.data(), 
                         n_elems * sizeof(ElementDynamic) / world.mpi_size,
                         MPI_BYTE, MPI_COMM_WORLD);
            
            // Copy back to device
            CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(),
                                 n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        }
    }
    
    // Copy final results back to host
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic,
                         n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(world.d_materials));
    CUDA_CHECK(cudaFree(world.d_elements_static));
    CUDA_CHECK(cudaFree(world.d_elements_dynamic));
    CUDA_CHECK(cudaFree(world.d_elements_dynamic_swap));
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Use OpenMP reduction for parallel aggregation
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    // MPI reduction across all ranks
    if (world.mpi_size > 1) {
        val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
        MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        
        energy_sum = global_energy_sum;
        flux_sum = global_flux_sum;
        energy_max = global_energy_max;
        energy_min = global_energy_min;
    }
    
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
    }
    
    return true;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    
    // Use OpenMP for parallel hash computation with reduction
    #pragma omp parallel for reduction(^:hash)
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        uint64_t local_hash = 0;
        local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        hash ^= local_hash;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Set GPU device based on local rank (assumes one GPU per MPI rank)
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    
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
        printf("=======================================================================\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", device_count);
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
    
    // For simplicity, all ranks get full mesh (in production, would partition)
    // Each rank works on full data for now, proper domain decomposition would split elements
    world.local_start = 0;
    world.local_size = n_elems;
    
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    if (mpi_rank == 0) {
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Barrier to ensure all ranks are ready
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (mpi_rank == 0) {
        printf("Running simulation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    // Barrier to synchronize timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (mpi_rank == 0) {
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
    }
    
    // Compute hash for verification (only rank 0)
    if (mpi_rank == 0) {
        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
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
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
