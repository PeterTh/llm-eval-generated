#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <map>
#include <set>

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
            exit(EXIT_FAILURE); \
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
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
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
__device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

__global__ void updateElementsKernel(
    const idx_t start_idx, const idx_t end_idx,
    const ElementStatic* d_static,
    const ElementDynamic* d_dynamic,
    ElementDynamic* d_dynamic_swap,
    const Material* d_materials,
    const idx_t num_materials) {

    idx_t i = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end_idx) return;

    const ElementStatic& elem_static = d_static[i];
    const ElementDynamic& elem_dyn = d_dynamic[i];
    const Material& mat = d_materials[elem_static.material_idx];
    
    // Start with external flow
    val_t total_flux = mat.external_flow;
    
    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        // Note: neighbor_idx is global. Since we copy the WHOLE dynamic array to GPU,
        // we can access any neighbor directly.
        // Optimization: For huge meshes exceeding GPU memory, we'd need more complex halo management.
        // But here we assume it fits (or we use unified memory).
        const ElementDynamic& neighbor_dyn = d_dynamic[neighbor_idx];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }
    
    // Update element state
    ElementDynamic& elem_write = d_dynamic_swap[i];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + abs(total_flux);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, int rank, int size) {
    const size_t n_elems = world.elements_static.size();
    
    // Simple 1D block decomposition
    const idx_t local_count = n_elems / size;
    const idx_t start_idx = rank * local_count;
    const idx_t end_idx = (rank == size - 1) ? n_elems : (rank + 1) * local_count;
    const idx_t my_count = end_idx - start_idx;

    // Identify ghost elements (needed neighbors not owned by this rank)
    std::set<idx_t> ghosts;
    for (idx_t i = start_idx; i < end_idx; ++i) {
        for (idx_t j = 0; j < world.elements_static[i].num_connections; ++j) {
            idx_t neighbor = world.elements_static[i].connected_idx[j];
            if (neighbor < start_idx || neighbor >= end_idx) {
                ghosts.insert(neighbor);
            }
        }
    }
    
    // Map owners to requested indices
    std::map<int, std::vector<idx_t>> send_requests; // rank -> indices I need to send to them
    std::map<int, std::vector<idx_t>> recv_requests; // rank -> indices I need to receive from them

    // In a real unstructured application, we would do a complex handshake here.
    // For simplicity, we'll gather all ghost requests.
    // Each rank broadcasts what it needs? No, too expensive.
    // Better: Each rank knows who owns what (block decomposition is simple).
    
    for (idx_t g : ghosts) {
        int owner = std::min((int)(g / local_count), size - 1);
        recv_requests[owner].push_back(g);
    }

    // Now tell owners what we need from them
    // We use MPI_Alltoallv or a series of Isends/Irecvs to exchange request lists.
    // Since the connectivity is static, we only do this ONCE.
    
    // Simplified exchange of requests:
    // 1. Everyone sends count of requests to everyone
    std::vector<int> send_counts(size, 0);
    std::vector<int> recv_counts(size, 0);
    for (const auto& pair : recv_requests) {
        send_counts[pair.first] = pair.second.size();
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 2. Exchange actual indices
    // recv_requests maps Owner -> [List of IDs I want].
    // I need to send this list to Owner.
    // Owner will store this as [Rank who wants] -> [List of IDs to send].
    
    std::vector<std::vector<idx_t>> incoming_requests(size);
    std::vector<MPI_Request> reqs;
    
    // Post receives for incoming requests (what others want from me)
    for (int r = 0; r < size; ++r) {
        if (recv_counts[r] > 0) {
            incoming_requests[r].resize(recv_counts[r]);
            reqs.emplace_back();
            MPI_Irecv(incoming_requests[r].data(), recv_counts[r] * sizeof(idx_t), MPI_BYTE, 
                      r, 0, MPI_COMM_WORLD, &reqs.back());
        }
    }
    
    // Send my requests (what I want from others)
    for (int r = 0; r < size; ++r) {
        if (send_counts[r] > 0) {
            reqs.emplace_back();
            MPI_Isend(recv_requests[r].data(), send_counts[r] * sizeof(idx_t), MPI_BYTE,
                      r, 0, MPI_COMM_WORLD, &reqs.back());
        }
    }
    
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();
    
    // Store what I need to send to others
    for (int r = 0; r < size; ++r) {
        if (!incoming_requests[r].empty()) {
            send_requests[r] = incoming_requests[r];
        }
    }

    // CUDA Setup
    ElementStatic* d_static;
    ElementDynamic* d_dynamic;
    ElementDynamic* d_dynamic_swap;
    Material* d_materials;
    
    CUDA_CHECK(cudaMalloc(&d_static, n_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_dynamic, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_dynamic_swap, n_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));
    
    CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data(), n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dynamic, world.elements_dynamic.data(), n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    // Prepare buffers for ghost exchange
    std::map<int, std::vector<ElementDynamic>> send_buffers;
    std::map<int, std::vector<ElementDynamic>> recv_buffers;
    
    for (const auto& pair : send_requests) send_buffers[pair.first].resize(pair.second.size());
    for (const auto& pair : recv_requests) recv_buffers[pair.first].resize(pair.second.size());

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (my_count + threadsPerBlock - 1) / threadsPerBlock;

    // Synchronize before starting timing loop if called from main
    // But main handles timing around this function.
    // To be fair to original code which didn't have data transfer overhead inside the loop,
    // we should consider if this is appropriate.
    // The original code passed `world` which was already in memory.
    // Here we move to GPU.
    // I will leave it as is because moving data to device is part of "running on accelerator".
    // But I will add a barrier at the start of the function? No, main has barrier.
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // 1. Compute on GPU for LOCAL elements
        updateElementsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            start_idx, end_idx, d_static, d_dynamic, d_dynamic_swap, d_materials, world.materials.size());
        CUDA_CHECK(cudaGetLastError());
        
        // 2. We need to exchange ghost data for the NEXT iteration.
        // Wait, current loop uses `d_dynamic` and writes to `d_dynamic_swap`.
        // The NEXT iteration will read from `d_dynamic_swap` (which becomes `d_dynamic`).
        // So we need to update the GHOSTS in `d_dynamic_swap` before the next iteration starts.
        
        // BUT, we only compute local elements. Ghosts are not computed by us.
        // They are computed by their owners.
        // So after computation, `d_dynamic_swap` has valid data for [start_idx, end_idx).
        // It has INVALID data for ghosts.
        // We need to fetch valid data for ghosts from their owners and put it into `d_dynamic_swap`.
        
        // This requires:
        // a) Copy updated local data from GPU to Host (only the parts needed by others).
        // b) Exchange with MPI.
        // c) Copy received ghost data from Host to GPU.
        
        // Optimization: We can overlap this with computation if we split the kernel into "inner" and "boundary" elements.
        // For simplicity here, we do it synchronously after kernel launch (but before next iter).
        
        CUDA_CHECK(cudaDeviceSynchronize()); // Finish computation
        
        // --- Halo Exchange ---
        
        // Copy ONLY owned elements back to CPU
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic_swap.data() + start_idx, 
                              d_dynamic_swap + start_idx, 
                              my_count * sizeof(ElementDynamic), 
                              cudaMemcpyDeviceToHost));
        
        // Pack buffers
        for (auto& pair : send_requests) {
            int target_rank = pair.first;
            const auto& indices = pair.second;
            auto& buffer = send_buffers[target_rank];
            for (size_t k = 0; k < indices.size(); ++k) {
                buffer[k] = world.elements_dynamic_swap[indices[k]];
            }
            
            reqs.emplace_back();
            MPI_Isend(buffer.data(), buffer.size() * sizeof(ElementDynamic), MPI_BYTE,
                      target_rank, 1, MPI_COMM_WORLD, &reqs.back());
        }
        
        // Receive buffers
        for (auto& pair : recv_requests) {
            int source_rank = pair.first;
            auto& buffer = recv_buffers[source_rank];
            
            reqs.emplace_back();
            MPI_Irecv(buffer.data(), buffer.size() * sizeof(ElementDynamic), MPI_BYTE,
                      source_rank, 1, MPI_COMM_WORLD, &reqs.back());
        }
        
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        reqs.clear();
        
        // Unpack ghosts into CPU world
        for (auto& pair : recv_requests) {
            int source_rank = pair.first;
            const auto& indices = pair.second;
            auto& buffer = recv_buffers[source_rank];
            for (size_t k = 0; k < indices.size(); ++k) {
                world.elements_dynamic_swap[indices[k]] = buffer[k];
            }
        }
        
        // Copy ghosts to GPU
        // To avoid many small copies, we could copy everything.
        // But copying only ghosts is hard because they are scattered.
        // Copying the whole array (host->device) is safe because we have the latest local data (just copied from GPU)
        // AND the latest ghost data (just received).
        // Total size ~4MB. Acceptable.
        CUDA_CHECK(cudaMemcpy(d_dynamic_swap, world.elements_dynamic_swap.data(), 
                              n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        
        // Swap pointers
        std::swap(d_dynamic, d_dynamic_swap);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap); // Keep CPU in sync-ish (fully sync at end)
    }
    
    // Final copy back (already done in loop, but ensure consistency)
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_dynamic, n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_dynamic));
    CUDA_CHECK(cudaFree(d_dynamic_swap));
    CUDA_CHECK(cudaFree(d_materials));
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
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
        
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        printf("GPUs per node: %d\n", num_gpus);
        printf("\n");
    }

    // Assign GPU based on local rank (assuming 1 rank per GPU if possible)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        // Simple assignment: rank % num_gpus
        // In a real cluster, we'd use local rank.
        cudaSetDevice(rank % num_gpus);
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
            total_mem / (1024.0 * 1024.0),
            static_mem / (1024.0 * 1024.0),
            dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
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
    }

    // Gather results for validation
    // Every rank has its computed chunk correct in 'world'.
    // We need to gather everything to rank 0 (or all ranks) to compute the hash/validate correctly
    // Since we did full copies, `world` on each rank has correct data for its chunk + ghosts.
    // But other parts are stale.
    // We need to Allgather the chunks.
    
    const idx_t local_count = n_elems / size;
    // Handle remainder for last rank if n_elems not divisible (but here we used simple logic)
    // Actually our decomposition logic was:
    // const idx_t end_idx = (rank == size - 1) ? n_elems : (rank + 1) * local_count;
    // So sizes vary. MPI_Allgatherv is needed.
    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
        idx_t r_start = r * local_count;
        idx_t r_end = (r == size - 1) ? n_elems : (r + 1) * local_count;
        recvcounts[r] = (r_end - r_start) * sizeof(ElementDynamic);
        displs[r] = r_start * sizeof(ElementDynamic);
    }
    
    // In place gather? No, we have separate chunks.
    // We want to update world.elements_dynamic with data from everyone.
    // Since we have the whole vector allocated, we can point to the right place.
    // But MPI_Allgatherv expects a contiguous receive buffer.
    // world.elements_dynamic is contiguous.
    
    // We send OUR chunk.
    // idx_t my_start = rank * local_count;
    // idx_t my_end = (rank == size - 1) ? n_elems : (rank + 1) * local_count;
    
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   world.elements_dynamic.data(), recvcounts.data(), displs.data(), MPI_BYTE,
                   MPI_COMM_WORLD);

    if (rank == 0) {
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
