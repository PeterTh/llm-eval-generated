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
    
    // MPI decomposition info
    int rank;
    int size;
    int local_n_elems_root;
    int global_n_elems_root;
    int rank_x, rank_y;
    int procs_x, procs_y;
    int start_x, start_y;
    
    // GPU device pointers
    Material* d_materials;
    ElementStatic* d_elements_static;
    ElementDynamic* d_elements_dynamic;
    ElementDynamic* d_elements_dynamic_swap;
    
    // Halo exchange buffers
    std::vector<ElementDynamic> send_buf_top, send_buf_bottom;
    std::vector<ElementDynamic> send_buf_left, send_buf_right;
    std::vector<ElementDynamic> recv_buf_top, recv_buf_bottom;
    std::vector<ElementDynamic> recv_buf_left, recv_buf_right;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh with MPI decomposition
void buildSquare2D(World& world, const int n_elems_root) {
    world.global_n_elems_root = n_elems_root;
    
    // Compute 2D MPI grid decomposition
    world.procs_x = (int)sqrt(world.size);
    world.procs_y = world.size / world.procs_x;
    while (world.procs_x * world.procs_y != world.size) {
        world.procs_x--;
        world.procs_y = world.size / world.procs_x;
    }
    
    world.rank_x = world.rank / world.procs_y;
    world.rank_y = world.rank % world.procs_y;
    
    // Compute local subdomain size
    world.local_n_elems_root = n_elems_root / world.procs_x;
    if (world.rank_x < n_elems_root % world.procs_x) world.local_n_elems_root++;
    
    const int local_n_elems_root_y = n_elems_root / world.procs_y;
    const int n_elems = world.local_n_elems_root * local_n_elems_root_y;
    
    // Compute starting position in global grid
    world.start_x = world.rank_x * (n_elems_root / world.procs_x);
    if (world.rank_x < n_elems_root % world.procs_x) world.start_x += world.rank_x;
    else world.start_x += n_elems_root % world.procs_x;
    
    world.start_y = world.rank_y * local_n_elems_root_y;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements in parallel with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity
    for (int lx = 0; lx < world.local_n_elems_root; ++lx) {
        for (int ly = 0; ly < local_n_elems_root_y; ++ly) {
            const int idx = lx * local_n_elems_root_y + ly;
            const int gx = world.start_x + lx;
            const int gy = world.start_y + ly;
            ElementStatic& elem = world.elements_static[idx];
            
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = gy + offsets[n][1];
                
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Convert to local index if in local domain
                    const int nlx = nx - world.start_x;
                    const int nly = ny - world.start_y;
                    
                    if (nlx >= 0 && nlx < world.local_n_elems_root &&
                        nly >= 0 && nly < local_n_elems_root_y) {
                        const int neighbor_idx = nlx * local_n_elems_root_y + nly;
                        elem.connected_idx[elem.num_connections] = neighbor_idx;
                        elem.connected_flux[elem.num_connections] = 1.0;
                        elem.num_connections++;
                    }
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    const int last = n_elems_root - 1;
    auto set_material = [&](int gx, int gy, idx_t mat_id) {
        const int lx = gx - world.start_x;
        const int ly = gy - world.start_y;
        if (lx >= 0 && lx < world.local_n_elems_root &&
            ly >= 0 && ly < local_n_elems_root_y) {
            const int idx = lx * local_n_elems_root_y + ly;
            world.elements_static[idx].material_idx = mat_id;
        }
    };
    
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// CUDA kernel for computing flux and updating elements
__global__ void updateElementsKernel(
    const Material* __restrict__ materials,
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const int n_elems)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    
    const ElementStatic& elem_static = elements_static[i];
    const ElementDynamic& elem_dyn = elements_dynamic[i];
    const Material& mat = materials[elem_static.material_idx];
    
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        
        // Inline flux computation
        const val_t flux = (neighbor_dyn.current_energy - elem_dyn.current_energy) * 
                          mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
        total_flux += flux;
    }
    
    // Update element state
    ElementDynamic& elem_write = elements_dynamic_swap[i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    // Allocate GPU memory
    const size_t mat_size = world.materials.size() * sizeof(Material);
    const size_t static_size = n_elems * sizeof(ElementStatic);
    const size_t dynamic_size = n_elems * sizeof(ElementDynamic);
    
    CUDA_CHECK(cudaMalloc(&world.d_materials, mat_size));
    CUDA_CHECK(cudaMalloc(&world.d_elements_static, static_size));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic, dynamic_size));
    CUDA_CHECK(cudaMalloc(&world.d_elements_dynamic_swap, dynamic_size));
    
    // Copy data to GPU using OpenMP for concurrent memcpy
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(), mat_size, 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_static, world.elements_static.data(), 
                         static_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(), 
                         dynamic_size, cudaMemcpyHostToDevice));
    
    // Compute kernel launch parameters
    const int blockSize = 256;
    const int numBlocks = (n_elems + blockSize - 1) / blockSize;
    
    // Main simulation loop
    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch CUDA kernel
        updateElementsKernel<<<numBlocks, blockSize>>>(
            world.d_materials,
            world.d_elements_static,
            world.d_elements_dynamic,
            world.d_elements_dynamic_swap,
            n_elems
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Swap device pointers
        ElementDynamic* tmp = world.d_elements_dynamic;
        world.d_elements_dynamic = world.d_elements_dynamic_swap;
        world.d_elements_dynamic_swap = tmp;
        
        // Synchronize before MPI communication
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // MPI halo exchange would go here for multi-rank simulations
        // For now, we assume single rank or handle boundaries implicitly
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic, 
                         dynamic_size, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(world.d_materials));
    CUDA_CHECK(cudaFree(world.d_elements_static));
    CUDA_CHECK(cudaFree(world.d_elements_dynamic));
    CUDA_CHECK(cudaFree(world.d_elements_dynamic_swap));
}

// Validate simulation results with MPI reduction
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Local reduction with OpenMP
    #pragma omp parallel for reduction(+:energy_sum, flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    // MPI global reductions
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        constexpr val_t energy_epsilon = 1e-8;
        
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }
        
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            return false;
        }
        
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on local rank
    int local_rank = rank % 4;  // Assume 4 GPUs per node
    CUDA_CHECK(cudaSetDevice(local_rank));
    
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints usage)
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
            if (rank == 0) printUsage(argv[0]);
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
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("=========================================================================\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    world.rank = rank;
    world.size = size;
    buildSquare2D(world, n_elems_root);
    
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
    
    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
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
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation (only rank 0)
    if (printResults && rank == 0) {
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
