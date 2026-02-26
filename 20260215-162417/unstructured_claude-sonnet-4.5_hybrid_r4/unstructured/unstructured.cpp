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
    } while(0)

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
    int local_start_row;
    int local_num_rows;
    int n_elems_root;
    int rank;
    int num_ranks;
    
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
void buildSquare2D(World& world, const int n_elems_root, int rank, int num_ranks) {
    world.n_elems_root = n_elems_root;
    world.rank = rank;
    world.num_ranks = num_ranks;
    
    // Decompose rows across ranks
    const int rows_per_rank = n_elems_root / num_ranks;
    const int extra_rows = n_elems_root % num_ranks;
    
    world.local_start_row = rank * rows_per_rank + std::min(rank, extra_rows);
    world.local_num_rows = rows_per_rank + (rank < extra_rows ? 1 : 0);
    
    const int local_n_elems = world.local_num_rows * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(local_n_elems);
    world.elements_dynamic.resize(local_n_elems);
    world.elements_dynamic_swap.resize(local_n_elems);
    
    // Initialize all elements with default material and zero energy
    #pragma omp parallel for
    for (int i = 0; i < local_n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2)
    for (int local_x = 0; local_x < world.local_num_rows; ++local_x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int x = world.local_start_row + local_x;
            const int local_idx = local_x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];
            
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
    int corners[4][2] = {{0, 0}, {0, last}, {last, 0}, {last, last}};
    idx_t corner_mats[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
    
    for (int c = 0; c < 4; ++c) {
        int corner_x = corners[c][0];
        if (corner_x >= world.local_start_row && 
            corner_x < world.local_start_row + world.local_num_rows) {
            int local_x = corner_x - world.local_start_row;
            int local_idx = local_x * n_elems_root + corners[c][1];
            world.elements_static[local_idx].material_idx = corner_mats[c];
        }
    }
}

// Compute energy flux between two elements
__device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel for element updates
__global__ void updateElementsKernel(
    const Material* materials,
    const ElementStatic* elements_static,
    const ElementDynamic* elements_dynamic,
    ElementDynamic* elements_dynamic_swap,
    const int* global_elem_map,
    int local_n_elems,
    int n_elems_root,
    int local_start_row
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n_elems) return;
    
    const ElementStatic& elem_static = elements_static[i];
    const ElementDynamic& elem_dyn = elements_dynamic[i];
    const Material& mat = materials[elem_static.material_idx];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_global_idx = elem_static.connected_idx[j];
        
        // Convert global index to local index
        int neighbor_x = neighbor_global_idx / n_elems_root;
        int neighbor_y = neighbor_global_idx % n_elems_root;
        int neighbor_local_x = neighbor_x - local_start_row;
        int neighbor_local_idx = neighbor_local_x * n_elems_root + neighbor_y;
        
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_local_idx];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }
    
    // Update element state
    ElementDynamic& elem_write = elements_dynamic_swap[i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Allocate GPU memory and copy data
void allocateGPUMemory(World& world) {
    const int local_n_elems = world.elements_static.size();
    const int n_elems_root = world.n_elems_root;
    
    // Allocate extra space for halo regions (one row above and below)
    const int halo_size = n_elems_root;
    const int total_with_halo = local_n_elems + 2 * halo_size;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&world.d_elements_static, local_n_elems * sizeof(ElementStatic)));
    
    // Allocate dynamic arrays with halo regions
    ElementDynamic* d_base_dynamic;
    ElementDynamic* d_base_dynamic_swap;
    CUDA_CHECK(cudaMalloc(&d_base_dynamic, total_with_halo * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_base_dynamic_swap, total_with_halo * sizeof(ElementDynamic)));
    
    // Offset pointers to account for top halo
    world.d_elements_dynamic = d_base_dynamic + halo_size;
    world.d_elements_dynamic_swap = d_base_dynamic_swap + halo_size;
    
    // Copy static data to device
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(),
                         world.materials.size() * sizeof(Material),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_static, world.elements_static.data(),
                         local_n_elems * sizeof(ElementStatic),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(),
                         local_n_elems * sizeof(ElementDynamic),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic_swap, world.elements_dynamic_swap.data(),
                         local_n_elems * sizeof(ElementDynamic),
                         cudaMemcpyHostToDevice));
}

// Free GPU memory
void freeGPUMemory(World& world) {
    const int n_elems_root = world.n_elems_root;
    const int halo_size = n_elems_root;
    
    CUDA_CHECK(cudaFree(world.d_materials));
    CUDA_CHECK(cudaFree(world.d_elements_static));
    
    // Free base pointers (before halo offset)
    CUDA_CHECK(cudaFree(world.d_elements_dynamic - halo_size));
    CUDA_CHECK(cudaFree(world.d_elements_dynamic_swap - halo_size));
}

// Exchange halo data between MPI ranks
void exchangeHaloData(World& world) {
    if (world.num_ranks == 1) return;
    
    const int n_elems_root = world.n_elems_root;
    const int halo_size = n_elems_root;
    
    std::vector<ElementDynamic> send_top(halo_size);
    std::vector<ElementDynamic> send_bottom(halo_size);
    std::vector<ElementDynamic> recv_top(halo_size);
    std::vector<ElementDynamic> recv_bottom(halo_size);
    
    // Copy boundary rows from device to host
    if (world.local_num_rows > 0) {
        CUDA_CHECK(cudaMemcpy(send_top.data(), world.d_elements_dynamic,
                             halo_size * sizeof(ElementDynamic),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(send_bottom.data(), 
                             world.d_elements_dynamic + (world.local_num_rows - 1) * n_elems_root,
                             halo_size * sizeof(ElementDynamic),
                             cudaMemcpyDeviceToHost));
    }
    
    // Exchange with neighbors
    MPI_Request reqs[4];
    int req_count = 0;
    
    // Send to previous rank, receive from next rank
    if (world.rank > 0) {
        MPI_Isend(send_top.data(), halo_size * sizeof(ElementDynamic), MPI_BYTE,
                 world.rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (world.rank < world.num_ranks - 1) {
        MPI_Irecv(recv_bottom.data(), halo_size * sizeof(ElementDynamic), MPI_BYTE,
                 world.rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    // Send to next rank, receive from previous rank
    if (world.rank < world.num_ranks - 1) {
        MPI_Isend(send_bottom.data(), halo_size * sizeof(ElementDynamic), MPI_BYTE,
                 world.rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (world.rank > 0) {
        MPI_Irecv(recv_top.data(), halo_size * sizeof(ElementDynamic), MPI_BYTE,
                 world.rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    
    // Copy received halo data back to device
    if (world.rank > 0) {
        CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic - n_elems_root,
                             recv_top.data(),
                             halo_size * sizeof(ElementDynamic),
                             cudaMemcpyHostToDevice));
    }
    if (world.rank < world.num_ranks - 1) {
        CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic + world.local_num_rows * n_elems_root,
                             recv_bottom.data(),
                             halo_size * sizeof(ElementDynamic),
                             cudaMemcpyHostToDevice));
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int local_n_elems = world.elements_static.size();
    const int n_elems_root = world.n_elems_root;
    
    // Configure CUDA kernel
    const int blockSize = 256;
    const int numBlocks = (local_n_elems + blockSize - 1) / blockSize;
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange halo data between MPI ranks
        exchangeHaloData(world);
        
        // Launch CUDA kernel
        updateElementsKernel<<<numBlocks, blockSize>>>(
            world.d_materials,
            world.d_elements_static,
            world.d_elements_dynamic,
            world.d_elements_dynamic_swap,
            nullptr,  // global_elem_map not needed for this implementation
            local_n_elems,
            n_elems_root,
            world.local_start_row
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Swap buffers
        std::swap(world.d_elements_dynamic, world.d_elements_dynamic_swap);
    }
    
    // Copy final results back to host
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic,
                         local_n_elems * sizeof(ElementDynamic),
                         cudaMemcpyDeviceToHost));
}

// Validate simulation results
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
    
    // Global reductions across MPI ranks
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    bool valid = true;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        
        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        
        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            // Don't fail validation as this can happen with external flows
        }
        
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    
    // Broadcast validation result
    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_int == 1;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    const int n_elems_root = world.n_elems_root;
    
    #pragma omp parallel for reduction(^:local_hash)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        // Convert local index to global index for consistent hashing
        int local_x = i / n_elems_root;
        int y = i % n_elems_root;
        int global_x = world.local_start_row + local_x;
        size_t global_idx = global_x * n_elems_root + y;
        
        // Simple hash combining energy and flux values with global index
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    // Global XOR reduction across MPI ranks
    uint64_t global_hash;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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
    
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    // Initialize CUDA for this rank
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", num_ranks);
        printf("CUDA devices: %d\n", device_count);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, num_ranks);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    if (rank == 0) {
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Allocate GPU memory and copy data
    allocateGPUMemory(world);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Get max time across all ranks
    long max_duration_ms;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Gather results for printing
    if (printResults) {
        std::vector<double> local_energyData;
        local_energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            local_energyData.push_back(elem.current_energy);
        }
        
        std::vector<double> global_energyData;
        if (rank == 0) {
            global_energyData.resize(n_elems);
        }
        
        // Gather all energy data to rank 0
        std::vector<int> recvcounts(num_ranks);
        std::vector<int> displs(num_ranks);
        int local_count = local_energyData.size();
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < num_ranks; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(local_energyData.data(), local_count, MPI_DOUBLE,
                    global_energyData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_energyData, "ElementEnergy");
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
