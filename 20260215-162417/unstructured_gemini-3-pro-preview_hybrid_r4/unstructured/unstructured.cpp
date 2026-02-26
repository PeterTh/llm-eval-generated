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

#include "../common/results_output.hpp"

// CUDA headers and macros
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}
#endif

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
    
    // Parallel decomposition info
    int rank;
    int size;
    size_t global_n_elems;
    size_t local_start;
    size_t local_end;
    size_t local_count;
    
    // Ghost exchange info
    std::vector<int> send_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_counts;
    std::vector<int> recv_displs;
    
    std::vector<idx_t> send_indices; // Global indices to send
    std::vector<idx_t> recv_indices; // Global indices we need (ghosts)
    
    std::vector<ElementDynamic> send_buffer;
    std::vector<ElementDynamic> recv_buffer;

    // CUDA pointers
    Material* d_materials = nullptr;
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;
    
    // Ghost mapping for GPU
    idx_t* d_ghost_indices = nullptr;   // Mapping from global index to ghost buffer index
    int* d_ghost_ranks = nullptr;       // Rank of owner for each ghost
    
    // Arrays for packed communication on GPU
    ElementDynamic* d_send_buffer = nullptr;
    ElementDynamic* d_recv_buffer = nullptr;
    idx_t* d_send_indices = nullptr;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#ifdef __CUDACC__
// CUDA Kernels
__global__ void computeFluxKernel(
    const int n_local,
    const size_t local_start,
    const Material* materials,
    const ElementStatic* elements_static,
    const ElementDynamic* elements_dynamic,
    const ElementDynamic* ghost_elements, 
    const idx_t* recv_indices,
    const int n_ghosts,
    ElementDynamic* elements_dynamic_swap) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const ElementStatic& elem_static = elements_static[i];
    const ElementDynamic& elem_dyn = elements_dynamic[i];
    const Material& mat = materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;

    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        idx_t neighbor_idx = elem_static.connected_idx[j]; // Remapped index
        
        ElementDynamic neighbor_dyn;
        
        if (neighbor_idx < n_local) {
             neighbor_dyn = elements_dynamic[neighbor_idx];
        } else {
             // It's a ghost, index is relative to n_local
             neighbor_dyn = ghost_elements[neighbor_idx - n_local];
        }

        // Compute flux
        val_t flux = (neighbor_dyn.current_energy - elem_dyn.current_energy) * 
                     mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
        total_flux += flux;
    }

    // Update element state
    elements_dynamic_swap[i].current_energy = elem_dyn.current_energy + total_flux;
    elements_dynamic_swap[i].total_flux = elem_dyn.total_flux + abs(total_flux);
}

// Kernel to pack send buffer
__global__ void packSendBufferKernel(
    const int n_send,
    const idx_t* send_indices,
    const size_t local_start,
    const ElementDynamic* elements_dynamic,
    ElementDynamic* send_buffer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_send) return;
    
    idx_t global_idx = send_indices[i];
    idx_t local_idx = global_idx - local_start;
    
    send_buffer[i] = elements_dynamic[local_idx];
}
#endif

// Helper to remap connections for local processing
void remapConnections(World& world) {
    // Map from global index to local index (0 to n_local-1) or ghost index (n_local to n_local+n_ghosts-1)
    
    // Pre-calculate ghost map for faster lookup
    // Since recv_indices is sorted (and unique), we can use binary search or map
    // recv_indices[k] maps to local index n_local + k
    
    for (size_t i = 0; i < world.local_count; ++i) {
        ElementStatic& elem = world.elements_static[i];
        for (int c = 0; c < elem.num_connections; ++c) {
            idx_t global_neighbor = elem.connected_idx[c];
            
            if (global_neighbor >= world.local_start && global_neighbor < world.local_end) {
                // Local
                elem.connected_idx[c] = global_neighbor - world.local_start;
            } else {
                // Ghost
                // Find in recv_indices
                auto it = std::lower_bound(world.recv_indices.begin(), world.recv_indices.end(), global_neighbor);
                if (it != world.recv_indices.end() && *it == global_neighbor) {
                    size_t ghost_idx = std::distance(world.recv_indices.begin(), it);
                    elem.connected_idx[c] = world.local_count + ghost_idx;
                } else {
                    // Should not happen if ghost exchange setup is correct
                    printf("Rank %d: Error - Neighbor %lu not found in ghost list for element %lu\n", 
                           world.rank, global_neighbor, world.local_start + i);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
    }
}

// Build a 2D square grid as an unstructured mesh
void buildSquare2D(World& world, const int n_elems_root) {
    const idx_t n_elems = (idx_t)n_elems_root * n_elems_root;
    world.global_n_elems = n_elems;
    
    // Initialize materials (same on all ranks)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Determine local range
    size_t elems_per_rank = n_elems / world.size;
    size_t remainder = n_elems % world.size;
    
    if (world.rank < (int)remainder) {
        world.local_start = world.rank * (elems_per_rank + 1);
        world.local_count = elems_per_rank + 1;
    } else {
        world.local_start = world.rank * elems_per_rank + remainder;
        world.local_count = elems_per_rank;
    }
    world.local_end = world.local_start + world.local_count;
    
    // Allocate local elements
    world.elements_static.resize(world.local_count);
    world.elements_dynamic.resize(world.local_count);
    world.elements_dynamic_swap.resize(world.local_count);
    
    // We need to identify ghosts
    std::vector<idx_t> needed_ghosts;
    
    // Build connectivity locally
    // We iterate global indices that belong to us, calculate their coordinates, find neighbors
    for (size_t i = 0; i < world.local_count; ++i) {
        idx_t global_idx = world.local_start + i;
        int x = global_idx / n_elems_root;
        int y = global_idx % n_elems_root;
        
        ElementStatic& elem = world.elements_static[i];
        
        // Initialize basic properties
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        
        // Setup corners (Global logic)
        const int last = n_elems_root - 1;
        if (global_idx == 0) elem.material_idx = INFLOW_MAT_ID;
        if (global_idx == (idx_t)last) elem.material_idx = OUTFLOW_MAT_ID; // 0, last
        if (global_idx == (idx_t)(last * n_elems_root)) elem.material_idx = OUTFLOW_MAT_ID; // last, 0
        if (global_idx == (idx_t)(n_elems - 1)) elem.material_idx = INFLOW_MAT_ID; // last, last
        
        // Initial state
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
        
        // Connectivity
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            int nx = x + offsets[n][0];
            int ny = y + offsets[n][1];
            
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                idx_t neighbor_idx = (idx_t)nx * n_elems_root + ny;
                
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
                
                // If neighbor is remote, add to needed ghosts
                if (neighbor_idx < world.local_start || neighbor_idx >= world.local_end) {
                    needed_ghosts.push_back(neighbor_idx);
                }
            }
        }
    }
    
    // Sort and remove duplicates from ghosts
    std::sort(needed_ghosts.begin(), needed_ghosts.end());
    needed_ghosts.erase(std::unique(needed_ghosts.begin(), needed_ghosts.end()), needed_ghosts.end());
    world.recv_indices = needed_ghosts;
    
    // Setup communication pattern
    // Determine who owns the ghosts we need
    int p = world.size;
    std::vector<int> request_counts(p, 0);
    std::vector<std::vector<idx_t>> requests(p);
    
    for (idx_t ghost_idx : world.recv_indices) {
        // Find owner
        // Simple inverse mapping of the logic above
        int owner = -1;
        size_t split = remainder * (elems_per_rank + 1);
        if (ghost_idx < split) {
             owner = ghost_idx / (elems_per_rank + 1);
        } else {
             owner = remainder + (ghost_idx - split) / elems_per_rank;
        }
        
        if (owner >= 0 && owner < p) {
            requests[owner].push_back(ghost_idx);
            request_counts[owner]++;
        }
    }
    
    // Exchange counts of requests
    std::vector<int> incoming_request_counts(p);
    MPI_Alltoall(request_counts.data(), 1, MPI_INT, incoming_request_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare data structures for Alltoallv
    std::vector<int> sdispls(p), rdispls(p);
    int s_total = 0, r_total = 0;
    
    std::vector<idx_t> send_req_buf;
    std::vector<idx_t> recv_req_buf;
    
    for (int i = 0; i < p; ++i) {
        sdispls[i] = s_total;
        s_total += request_counts[i];
        send_req_buf.insert(send_req_buf.end(), requests[i].begin(), requests[i].end());
        
        rdispls[i] = r_total;
        r_total += incoming_request_counts[i];
    }
    recv_req_buf.resize(r_total);
    
    // Send requests to owners
    MPI_Alltoallv(send_req_buf.data(), request_counts.data(), sdispls.data(), MPI_UINT64_T,
                  recv_req_buf.data(), incoming_request_counts.data(), rdispls.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    
    // Process incoming requests: these are the elements I need to send
    world.send_indices = recv_req_buf; // These are global indices I own
    
    // Now setup the persistent communication pattern for values
    // We send 'recv_req_buf' elements to the requesters.
    // The requesters are the 'incoming_request_counts' sources.
    // NOTE: Alltoallv sends to destination i, receives from source i.
    // In request phase: I sent request to OWNER. OWNER received request from ME.
    // Now OWNER sends value to ME. I receive value from OWNER.
    // So for value exchange:
    // My SEND count to rank i is incoming_request_counts[i] (what they asked for).
    // My RECV count from rank i is request_counts[i] (what I asked for).
    
    world.send_counts = incoming_request_counts;
    world.send_displs = rdispls; 
    
    world.recv_counts = request_counts;
    world.recv_displs = sdispls;
    
    world.send_buffer.resize(world.send_indices.size());
    world.recv_buffer.resize(world.recv_indices.size());
    
    // Remap connections to use local/ghost indices
    remapConnections(world);
}

// Compute energy flux between two elements (Host version)
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.local_count;
    size_t n_send = world.send_buffer.size();
    size_t n_ghosts = world.recv_buffer.size();
    
#ifdef __CUDACC__
    // Setup CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int device = world.rank % deviceCount;
        cudaSetDevice(device);
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material)));
    CHECK_CUDA(cudaMalloc(&world.d_elements_static, world.elements_static.size() * sizeof(ElementStatic)));
    CHECK_CUDA(cudaMalloc(&world.d_elements_dynamic, world.elements_dynamic.size() * sizeof(ElementDynamic)));
    CHECK_CUDA(cudaMalloc(&world.d_elements_dynamic_swap, world.elements_dynamic_swap.size() * sizeof(ElementDynamic)));
    
    // Ghost buffers on GPU
    ElementDynamic* d_ghost_elements = nullptr;
    if (n_ghosts > 0) {
        CHECK_CUDA(cudaMalloc(&d_ghost_elements, n_ghosts * sizeof(ElementDynamic)));
    }
    
    // Communication buffers on GPU
    if (n_send > 0) {
        CHECK_CUDA(cudaMalloc(&world.d_send_buffer, n_send * sizeof(ElementDynamic)));
        CHECK_CUDA(cudaMalloc(&world.d_send_indices, n_send * sizeof(idx_t)));
        CHECK_CUDA(cudaMemcpy(world.d_send_indices, world.send_indices.data(), n_send * sizeof(idx_t), cudaMemcpyHostToDevice));
    }
    
    // Copy initial data
    CHECK_CUDA(cudaMemcpy(world.d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(world.d_elements_static, world.elements_static.data(), n_local * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(), n_local * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    
    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (n_local + blockSize - 1) / blockSize;
    int sendPackBlocks = (n_send + blockSize - 1) / blockSize;
#endif

    for (int iter = 0; iter < n_iters; ++iter) {
        // Exchange ghost data
        // 1. Pack send buffer
#ifdef __CUDACC__
        if (n_send > 0) {
            packSendBufferKernel<<<sendPackBlocks, blockSize>>>(
                n_send, world.d_send_indices, world.local_start, 
                world.d_elements_dynamic, world.d_send_buffer);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpy(world.send_buffer.data(), world.d_send_buffer, n_send * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
        }
#else
        #pragma omp parallel for
        for (size_t i = 0; i < world.send_indices.size(); ++i) {
            idx_t global_idx = world.send_indices[i];
            world.send_buffer[i] = world.elements_dynamic[global_idx - world.local_start];
        }
#endif

        // 2. MPI Exchange
        // MPI uses byte-wise transfer for structs, ElementDynamic is POD.
        // We need to use MPI_BYTE or create a type. sizeof(ElementDynamic) is 16 bytes.
        MPI_Alltoallv(world.send_buffer.data(), world.send_counts.data(), world.send_displs.data(), 
                      MPI_DOUBLE_COMPLEX, // Hack: size 16 bytes matches double complex (usually)
                      world.recv_buffer.data(), world.recv_counts.data(), world.recv_displs.data(), 
                      MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
        // Note: MPI_DOUBLE_COMPLEX is 16 bytes (2 doubles). ElementDynamic is 2 doubles. Correct.

        // 3. Copy received ghosts to device (if CUDA)
#ifdef __CUDACC__
        if (n_ghosts > 0) {
            CHECK_CUDA(cudaMemcpy(d_ghost_elements, world.recv_buffer.data(), n_ghosts * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        }
#endif

        // 4. Update elements
#ifdef __CUDACC__
        computeFluxKernel<<<numBlocks, blockSize>>>(
            n_local, world.local_start, 
            world.d_materials, world.d_elements_static, 
            world.d_elements_dynamic, d_ghost_elements, 
            nullptr, n_ghosts, 
            world.d_elements_dynamic_swap);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Swap pointers
        std::swap(world.d_elements_dynamic, world.d_elements_dynamic_swap);
#else
        #pragma omp parallel for
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            val_t total_flux = mat.external_flow;
            
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                idx_t connected_idx = elem_static.connected_idx[j]; // Remapped index
                ElementDynamic neighbor_dyn;
                
                if (connected_idx < n_local) {
                    neighbor_dyn = world.elements_dynamic[connected_idx];
                } else {
                    neighbor_dyn = world.recv_buffer[connected_idx - n_local];
                }
                
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
#endif
    }
    
#ifdef __CUDACC__
    // Copy back result
    CHECK_CUDA(cudaMemcpy(world.elements_dynamic.data(), world.d_elements_dynamic, n_local * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(world.d_materials);
    cudaFree(world.d_elements_static);
    cudaFree(world.d_elements_dynamic);
    cudaFree(world.d_elements_dynamic_swap);
    if (d_ghost_elements) cudaFree(d_ghost_elements);
    if (world.d_send_buffer) cudaFree(world.d_send_buffer);
    if (world.d_send_indices) cudaFree(world.d_send_indices);
#endif
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
    
    // Reduce across all ranks
    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Reduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);
        
        // Check for numerical issues
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
        
        printf("  Validation: PASSED\n");
    }
    
    return true;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        // Simple hash combining energy and flux values
        // Use GLOBAL index for consistency
        idx_t global_idx = world.local_start + i;
        
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }
    
    uint64_t global_hash;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
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
    
    const idx_t n_elems = (idx_t)n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("====================================================================\n");
        printf("Grid size: %d x %d = %lu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI Ranks: %d\n", size);
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
#ifdef __CUDACC__
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA Devices: %d\n", deviceCount);
#else
        printf("CUDA: Disabled\n");
#endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    
    World world;
    world.rank = rank;
    world.size = size;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    // Just print rank 0 memory for brevity, or sum it up
    if (rank == 0) {
         printf("Memory usage (Rank 0): %.2f MB\n", total_mem / (1024.0 * 1024.0));
         printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (double)(n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration
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
    
    // Print results for external validation
    if (printResults) {
        // Gather all data to rank 0 (Not scalable, but required for this flag)
        std::vector<double> localEnergy;
        localEnergy.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            localEnergy.push_back(elem.current_energy);
        }
        
        // Gather counts first
        std::vector<int> recv_counts(size);
        int local_count_val = (int)localEnergy.size();
        MPI_Gather(&local_count_val, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displs(size);
        std::vector<double> globalEnergy;
        
        if (rank == 0) {
            int total = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = total;
                total += recv_counts[i];
            }
            globalEnergy.resize(total);
        }
        
        MPI_Gatherv(localEnergy.data(), local_count_val, MPI_DOUBLE, 
                    globalEnergy.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(globalEnergy, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (rank == 0 && !valid) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
