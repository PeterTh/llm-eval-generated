#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cstddef>

#include <mpi.h>
#include <omp.h>

#ifndef CUDA_AVAILABLE
#define CUDA_AVAILABLE 1
#endif

#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

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

// World state (distributed across MPI processes)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    
    // Distributed state
    size_t local_elem_start;  // Start index of local elements in global mesh
    size_t local_elem_count;  // Number of local elements
    size_t global_elem_count; // Total number of elements
    int mpi_rank;
    int mpi_size;
    int gpu_id;
    
    // GPU device pointers (null if CUDA not available)
    #if CUDA_AVAILABLE
    ElementStatic* d_elements_static;
    ElementDynamic* d_elements_dynamic;
    ElementDynamic* d_elements_dynamic_swap;
    Material* d_materials;
    #endif
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// CUDA kernel for flux computation and state update
#if CUDA_AVAILABLE
__global__ void computeStateUpdateKernel(
    const ElementStatic* elements_static,
    const ElementDynamic* elements_dynamic,
    const Material* materials,
    ElementDynamic* elements_dynamic_new,
    size_t num_elements) {
    
    idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    
    const ElementStatic& elem_static = elements_static[i];
    const ElementDynamic& elem_dyn = elements_dynamic[i];
    const Material& mat = materials[elem_static.material_idx];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        
        // computeFlux inline
        val_t flux = (neighbor_dyn.current_energy - elem_dyn.current_energy) * 
                     mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
        total_flux += flux;
    }
    
    // Update element state
    elements_dynamic_new[i].current_energy = elem_dyn.current_energy + total_flux;
    elements_dynamic_new[i].total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Check and handle CUDA errors
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", msg, cudaGetErrorString(err));
    }
}

// Initialize GPU and allocate device memory
void initializeGPU(World& world) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        world.gpu_id = world.mpi_rank % device_count;
        cudaSetDevice(world.gpu_id);
        checkCudaError("initializeGPU: cudaSetDevice");
        
        // Enable peer access for multi-GPU clusters
        for (int i = 0; i < device_count; ++i) {
            if (i != world.gpu_id) {
                cudaDeviceEnablePeerAccess(i, 0);
            }
        }
    }
}

// Allocate device memory for element arrays
void allocateDeviceMemory(World& world) {
    size_t static_size = world.elements_static.size() * sizeof(ElementStatic);
    size_t dynamic_size = world.elements_dynamic.size() * sizeof(ElementDynamic);
    size_t material_size = world.materials.size() * sizeof(Material);
    
    cudaMalloc(&world.d_elements_static, static_size);
    cudaMalloc(&world.d_elements_dynamic, dynamic_size);
    cudaMalloc(&world.d_elements_dynamic_swap, dynamic_size);
    cudaMalloc(&world.d_materials, material_size);
    
    checkCudaError("allocateDeviceMemory: cudaMalloc");
    
    // Copy static data to device
    cudaMemcpy(world.d_elements_static, world.elements_static.data(), static_size, cudaMemcpyHostToDevice);
    cudaMemcpy(world.d_materials, world.materials.data(), material_size, cudaMemcpyHostToDevice);
    checkCudaError("allocateDeviceMemory: cudaMemcpy static");
}

// Free device memory
void freeDeviceMemory(World& world) {
    if (world.d_elements_static) cudaFree(world.d_elements_static);
    if (world.d_elements_dynamic) cudaFree(world.d_elements_dynamic);
    if (world.d_elements_dynamic_swap) cudaFree(world.d_elements_dynamic_swap);
    if (world.d_materials) cudaFree(world.d_materials);
    checkCudaError("freeDeviceMemory");
}
#else
void initializeGPU(World& world) {
    world.gpu_id = -1;
}

void allocateDeviceMemory(World& /* world */) {}
void freeDeviceMemory(World& /* world */) {}
#endif

// Build a 2D square grid as an unstructured mesh (distributed across MPI processes)
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials (same on all processes)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    world.global_elem_count = n_elems;
    
    // Distribute elements across MPI processes
    size_t elems_per_process = n_elems / world.mpi_size;
    size_t remainder = n_elems % world.mpi_size;
    
    world.local_elem_start = (world.mpi_rank < (int)remainder) ?
        world.mpi_rank * (elems_per_process + 1) :
        remainder * (elems_per_process + 1) + (world.mpi_rank - remainder) * elems_per_process;
    
    world.local_elem_count = (world.mpi_rank < (int)remainder) ?
        elems_per_process + 1 :
        elems_per_process;
    
    // Allocate local elements
    world.elements_static.resize(world.local_elem_count);
    world.elements_dynamic.resize(world.local_elem_count);
    world.elements_dynamic_swap.resize(world.local_elem_count);
    
    // Initialize all local elements with default material and zero energy
    #pragma omp parallel for collapse(1) schedule(static)
    for (size_t i = 0; i < world.local_elem_count; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int global_idx = x * n_elems_root + y;
            
            // Check if this element belongs to this process
            if (global_idx < (int)world.local_elem_start || 
                global_idx >= (int)(world.local_elem_start + world.local_elem_count)) {
                continue;
            }
            
            const size_t local_idx = global_idx - world.local_elem_start;
            ElementStatic& elem = world.elements_static[local_idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_global_idx = nx * n_elems_root + ny;
                    
                    // Adjust neighbor index to local space (for local neighbors)
                    // Global indices used for connectivity to support all-to-all access
                    elem.connected_idx[elem.num_connections] = neighbor_global_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow (only on process that owns them)
    const int last = n_elems_root - 1;
    
    if (0 >= (int)world.local_elem_start && 0 < (int)(world.local_elem_start + world.local_elem_count)) {
        size_t local_idx = 0 - world.local_elem_start;
        world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
    }
    
    int idx = 0 * n_elems_root + last;
    if (idx >= (int)world.local_elem_start && idx < (int)(world.local_elem_start + world.local_elem_count)) {
        size_t local_idx = idx - world.local_elem_start;
        world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
    }
    
    idx = last * n_elems_root + 0;
    if (idx >= (int)world.local_elem_start && idx < (int)(world.local_elem_start + world.local_elem_count)) {
        size_t local_idx = idx - world.local_elem_start;
        world.elements_static[local_idx].material_idx = OUTFLOW_MAT_ID;
    }
    
    idx = last * n_elems_root + last;
    if (idx >= (int)world.local_elem_start && idx < (int)(world.local_elem_start + world.local_elem_count)) {
        size_t local_idx = idx - world.local_elem_start;
        world.elements_static[local_idx].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations with hybrid parallelization
void runSimulation(World& world, const int n_iters) {
    bool use_gpu = false;
    #if CUDA_AVAILABLE
    int device_count;
    cudaGetDeviceCount(&device_count);
    use_gpu = device_count > 0;
    #endif
    
    // Allocate GPU memory if available
    if (use_gpu) {
        allocateDeviceMemory(world);
    }
    
    // Create a replicated copy of all elements for computation (needed for neighbor access)
    std::vector<ElementDynamic> all_elements;
    std::vector<ElementStatic> all_elements_static;
    std::vector<ElementDynamic> all_elements_swap;
    
    if (world.mpi_size == 1) {
        // Single process: use local arrays directly
        all_elements = world.elements_dynamic;
        all_elements_static = world.elements_static;
        all_elements_swap = world.elements_dynamic_swap;
    } else {
        // Multi-process: gather all data on all processes
        all_elements.resize(world.global_elem_count);
        all_elements_static.resize(world.global_elem_count);
        all_elements_swap.resize(world.global_elem_count);
        
        // Create MPI datatypes for structs
        MPI_Datatype mpi_element_static;
        MPI_Datatype mpi_element_dynamic;
        
        // ElementStatic: 2 idx_t + 8 idx_t + 8 val_t
        int blocklens_static[] = {1, 1, MAX_CONNECTIONS, MAX_CONNECTIONS};
        MPI_Aint displs_static[] = {
            offsetof(ElementStatic, material_idx),
            offsetof(ElementStatic, num_connections),
            offsetof(ElementStatic, connected_idx),
            offsetof(ElementStatic, connected_flux)
        };
        MPI_Datatype types_static[] = {MPI_UINT64_T, MPI_UINT64_T, MPI_UINT64_T, MPI_DOUBLE};
        MPI_Type_create_struct(4, blocklens_static, displs_static, types_static, &mpi_element_static);
        MPI_Type_commit(&mpi_element_static);
        
        // ElementDynamic: 2 val_t
        int blocklens_dynamic[] = {2};
        MPI_Aint displs_dynamic[] = {0};
        MPI_Datatype types_dynamic[] = {MPI_DOUBLE};
        MPI_Type_create_struct(1, blocklens_dynamic, displs_dynamic, types_dynamic, &mpi_element_dynamic);
        MPI_Type_commit(&mpi_element_dynamic);
        
        // Gather phase
        std::vector<int> recv_counts(world.mpi_size);
        std::vector<int> recv_displs(world.mpi_size);
        
        int local_count = world.local_elem_count;
        MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        recv_displs[0] = 0;
        for (int i = 1; i < world.mpi_size; ++i) {
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        }
        
        MPI_Allgatherv(world.elements_dynamic.data(), local_count,
                       mpi_element_dynamic, all_elements.data(), recv_counts.data(),
                       recv_displs.data(), mpi_element_dynamic, MPI_COMM_WORLD);
        
        MPI_Allgatherv(world.elements_static.data(), local_count,
                       mpi_element_static, all_elements_static.data(), recv_counts.data(),
                       recv_displs.data(), mpi_element_static, MPI_COMM_WORLD);
        
        MPI_Type_free(&mpi_element_static);
        MPI_Type_free(&mpi_element_dynamic);
    }
    
    // Copy to GPU if available
    if (use_gpu) {
        #if CUDA_AVAILABLE
        cudaMemcpy(world.d_elements_dynamic, all_elements.data(),
                   all_elements.size() * sizeof(ElementDynamic), cudaMemcpyHostToDevice);
        checkCudaError("runSimulation: copy initial state to GPU");
        #endif
    }
    
    for (int iter = 0; iter < n_iters; ++iter) {
        if (use_gpu) {
            #if CUDA_AVAILABLE
            // GPU computation
            int block_size = 256;
            int grid_size = (all_elements.size() + block_size - 1) / block_size;
            computeStateUpdateKernel<<<grid_size, block_size>>>(
                world.d_elements_static,
                world.d_elements_dynamic,
                world.d_materials,
                world.d_elements_dynamic_swap,
                all_elements.size());
            checkCudaError("runSimulation: computeStateUpdateKernel");
            
            // Swap device pointers
            std::swap(world.d_elements_dynamic, world.d_elements_dynamic_swap);
            
            // Copy back to host for next iteration (if multi-GPU or sync needed)
            cudaMemcpy(all_elements.data(), world.d_elements_dynamic,
                       all_elements.size() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);
            checkCudaError("runSimulation: copy from GPU");
            #endif
        } else {
            // CPU computation with OpenMP parallelization
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < all_elements.size(); ++i) {
                const ElementStatic& elem_static = all_elements_static[i];
                const ElementDynamic& elem_dyn = all_elements[i];
                const Material& mat = world.materials[elem_static.material_idx];
                
                // Start with external flow
                val_t total_flux = mat.external_flow;
                
                // Add flux from all connected elements
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = all_elements[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }
                
                // Update element state
                ElementDynamic& elem_write = all_elements_swap[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
            
            // Swap buffers
            std::swap(all_elements, all_elements_swap);
        }
    }
    
    // Copy results back to local elements
    if (world.mpi_size == 1) {
        world.elements_dynamic = all_elements;
    } else {
        // Scatter local portion back
        int local_count = world.local_elem_count;
        size_t start_offset = world.local_elem_start;
        std::copy(all_elements.begin() + start_offset,
                  all_elements.begin() + start_offset + local_count,
                  world.elements_dynamic.begin());
        
        // Broadcast full results to all processes for validation
        MPI_Bcast(all_elements.data(), all_elements.size(), MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
        world.elements_dynamic = all_elements;
    }
    
    // Free GPU memory
    if (use_gpu) {
        freeDeviceMemory(world);
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
    return true;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
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
    // Initialize MPI
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("=====================================================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        
        #if CUDA_AVAILABLE
        int device_count;
        cudaGetDeviceCount(&device_count);
        printf("CUDA devices available: %d\n", device_count);
        #else
        printf("CUDA: Not available\n");
        #endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    
    World world;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    initializeGPU(world);
    buildSquare2D(world, n_elems_root);
    
    if (mpi_rank == 0) {
        const size_t static_mem = world.global_elem_count * sizeof(ElementStatic);
        const size_t dynamic_mem = world.global_elem_count * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
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
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
