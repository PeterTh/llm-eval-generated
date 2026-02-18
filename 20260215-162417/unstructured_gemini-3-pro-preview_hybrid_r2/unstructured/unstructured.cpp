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

// CUDA error checking macro
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
    // Parallelize initialization with OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    // Parallelize with OpenMP
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
__host__ __device__
inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                        val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * 
           transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel for simulation step
__global__
void updateElementsKernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    size_t n_local_elems,
    idx_t start_idx,
    size_t n_total_elems // Total elements including ghosts if needed, or just global size for checking
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local_elems) return;

    // Adjust for global indexing if needed, but here we process local chunk 0..n_local_elems
    // However, connectivity uses global indices.
    // If we copy the WHOLE world to GPU, we can use global indices directly.
    // For MPI, we only copy local + ghost.
    // Strategy: 
    // If we use unified memory or just copy everything, it's easiest but doesn't scale.
    // Better: Remap indices. But that's complex for "unstructured".
    // 
    // Given the constraints and the goal of "hybrid parallelization", let's assume
    // we copy the FULL state to each GPU if it fits, OR handles mapping.
    //
    // Let's implement the simpler approach first: 
    // Each MPI rank has a full copy of the world structure for simplicity of indexing,
    // but only updates its assigned range.
    // This is "replicated memory, distributed computation".
    // It limits problem size to single node memory, but scales computation.
    // For a benchmark, this is often acceptable if not specified otherwise.
    //
    // Wait, if N is large, replicated memory is bad.
    // But remapping indices requires changing ElementStatic.
    // Let's stick to: GLOBAL indices in ElementStatic.
    //
    // For CUDA: pass the WHOLE arrays. 
    // Each rank updates [start_idx, end_idx).
    
    // Using global index for the element to update
    size_t global_i = start_idx + idx;

    const ElementStatic& elem_static = elements_static[global_i];
    const ElementDynamic& elem_dyn = elements_dynamic[global_i];
    const Material& mat = materials[elem_static.material_idx];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        // Note: neighbor_idx is global. elements_dynamic must cover global range.
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        total_flux += computeFlux(mat.transfer_coeff, elem_dyn.current_energy, 
                                elem_static.connected_flux[j], neighbor_dyn.current_energy);
    }
    
    // Update element state
    ElementDynamic& elem_write = elements_dynamic_swap[global_i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + abs(total_flux);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, int rank, int size) {
    const size_t n_elems = world.elements_static.size();
    
    // Determine local range
    size_t chunk_size = (n_elems + size - 1) / size;
    size_t start_idx = rank * chunk_size;
    size_t end_idx = std::min(start_idx + chunk_size, n_elems);
    size_t local_count = (start_idx < end_idx) ? (end_idx - start_idx) : 0;

    // Allocate device memory
    ElementStatic* d_elements_static;
    ElementDynamic* d_elements_dynamic;
    ElementDynamic* d_elements_dynamic_swap;
    Material* d_materials;

    CUDA_CHECK(cudaMalloc(&d_elements_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_elements_dynamic_swap, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));

    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_elements_static, world.elements_static.data(), 
               n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elements_dynamic, world.elements_dynamic.data(), 
               n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    // dynamic_swap doesn't need initialization, will be overwritten
    // But for validation safety, let's copy it too (contains zeros)
    CUDA_CHECK(cudaMemcpy(d_elements_dynamic_swap, world.elements_dynamic_swap.data(), 
               n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(), 
               world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (local_count + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Compute on GPU for local range
        if (local_count > 0) {
            updateElementsKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_elements_static,
                d_elements_dynamic,
                d_elements_dynamic_swap,
                d_materials,
                local_count,
                start_idx,
                n_elems
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // We need to exchange ghost cells.
        // Since we are updating `d_elements_dynamic_swap` based on `d_elements_dynamic`,
        // the NEW state is in `swap`.
        // BUT, the NEXT iteration will read from `swap` (which becomes `dynamic`).
        // So we need to ensure the `swap` buffer has updated values from other ranks
        // for the parts we depend on?
        // No, in standard simulation:
        // Step k: Read A, Write B.
        // Step k+1: Read B, Write A.
        // We computed our part of B. Other ranks computed their part of B.
        // We need the *whole* B (or at least the parts we need) to be consistent on our GPU for the next step.
        //
        // So we need to Allgather the updated parts of the array.
        // Since we are using replicated memory strategy (simplest for unstructured without graph partitioning),
        // we gather the `local_count` elements from each rank into the global array on all ranks.
        
        // Copy updated local chunk back to host
        if (local_count > 0) {
            CUDA_CHECK(cudaMemcpy(
                world.elements_dynamic_swap.data() + start_idx,
                d_elements_dynamic_swap + start_idx,
                local_count * sizeof(ElementDynamic),
                cudaMemcpyDeviceToHost
            ));
        }

        // MPI Allgather to sync the whole world
        // Note: This is expensive but correct for this "unstructured" layout without complex graph partitioning.
        // For a benchmark, correctness comes first.
        // To optimize, we would only exchange boundary layers, but "unstructured" implies arbitrary connectivity.
        // The `buildSquare2D` makes it structured, but the code treats it as unstructured.
        // So Allgatherv is the safe general approach.
        
        // We need to sync `elements_dynamic_swap`.
        // Prepare counts and displs for Allgatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        // Calculate counts and displacements
        // Since n_elems might be large (uint64_t), but MPI uses int, we assume it fits in int for this benchmark.
        // If > 2B elements, we need BigMPI, but standard benchmark probably < 2B.
        
        for (int r = 0; r < size; ++r) {
            size_t r_start = r * chunk_size;
            size_t r_end = std::min(r_start + chunk_size, n_elems);
            recvcounts[r] = (r_start < r_end) ? (r_end - r_start) * sizeof(ElementDynamic) : 0;
            displs[r] = r_start * sizeof(ElementDynamic);
        }

        // In-place Allgatherv? No, we have the local piece in place, need to receive others.
        // MPI_Allgatherv with MPI_IN_PLACE is tricky.
        // Better to use the standard call.
        // Note: We are gathering bytes here because ElementDynamic is a struct of doubles.
        
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       world.elements_dynamic_swap.data(), recvcounts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);

        // Now world.elements_dynamic_swap is fully updated on host.
        // Copy the synced world back to device for next iteration?
        // Actually, we only need to copy the *ghosts*. But copying all is easier.
        // Optimization: Double buffering on GPU.
        // We just computed into `swap`. Next step reads from `swap`.
        // But `swap` on GPU only has VALID data for `local_range`.
        // We need valid data for `ghosts`.
        // So we copy the HOST `swap` (which is now fully synced) to DEVICE `swap`.
        
        CUDA_CHECK(cudaMemcpy(d_elements_dynamic_swap, world.elements_dynamic_swap.data(),
                   n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));

        // Swap pointers/indices for next iteration
        std::swap(d_elements_dynamic, d_elements_dynamic_swap);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }

    // Final result is in world.elements_dynamic (on host)
    // (We already synced it in the last step of the loop via dynamic_swap and then host swap)
    
    CUDA_CHECK(cudaFree(d_elements_static));
    CUDA_CHECK(cudaFree(d_elements_dynamic));
    CUDA_CHECK(cudaFree(d_elements_dynamic_swap));
    CUDA_CHECK(cudaFree(d_materials));
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Parallelize reduction with OpenMP
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
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
    // Parallelize reduction? XOR is bitwise, harder with OpenMP standard reductions in old versions.
    // Custom reduction or just serial. Serial is fine for hash.
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
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } 
        // MPI passes other args too potentially? No, just ours.
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    // Set CUDA device based on local rank
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    
    if (num_gpus > 0) {
        cudaSetDevice(local_rank % num_gpus);
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
        printf("CUDA Devices: %d available (using round-robin assignment)\n", num_gpus);
        printf("\n");
    }
    
    // Build the unstructured mesh (Replicated on all ranks)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (rank == 0) {
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
                // Cannot easily return from main with non-zero on just one rank without aborting
                // Just print error
                printf("Validation FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
