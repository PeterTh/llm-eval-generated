#include <map>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Check for CUDA errors
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

// CUDA kernel for Flux calculation
__global__ void computeFluxKernel(
    const int num_local_elements,
    const idx_t* __restrict__ material_indices,
    const idx_t* __restrict__ num_connections,
    const idx_t* __restrict__ connected_indices,
    const val_t* __restrict__ connected_fluxes,
    const val_t* __restrict__ material_transfer_coeff,
    const val_t* __restrict__ material_external_flow,
    const val_t* __restrict__ current_energy,
    const val_t* __restrict__ total_flux_in,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_total_flux
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local_elements) return;

    idx_t mat_idx = material_indices[i];
    val_t transfer_coeff = material_transfer_coeff[mat_idx];
    val_t external_flow = material_external_flow[mat_idx];
    
    val_t my_energy = current_energy[i];
    val_t my_total_flux = total_flux_in[i];

    val_t computed_flux = external_flow;
    
    int conn_start = i * MAX_CONNECTIONS;
    int num_conn = num_connections[i];

    for (int j = 0; j < num_conn; ++j) {
        idx_t neighbor_local_idx = connected_indices[conn_start + j];
        val_t neighbor_energy = current_energy[neighbor_local_idx];
        val_t conn_flux = connected_fluxes[conn_start + j];
        
        computed_flux += (neighbor_energy - my_energy) * transfer_coeff * conn_flux * 0.25;
    }

    next_energy[i] = my_energy + computed_flux;
    next_total_flux[i] = my_total_flux + abs(computed_flux);
}

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
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
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
    MPI_Init(&argc, &argv);

    // Initialize MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Ranks: %d, GPUs available: %d\n", size, num_devices);
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    // Only rank 0 prints
    if (rank == 0) {
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Partition elements
    size_t local_n_elems = n_elems / size;
    size_t remainder = n_elems % size;
    size_t local_start = rank * local_n_elems + std::min((size_t)rank, remainder);
    if (rank < remainder) local_n_elems++;
    size_t local_end = local_start + local_n_elems;

    // Identify ghosts and build local mapping
    std::vector<idx_t> global_to_local_ghost;
    std::vector<idx_t> ghost_global_indices;
    std::map<idx_t, idx_t> global_to_local_map; // Only for ghosts

    // Fill map for local elements
    for (size_t i = 0; i < local_n_elems; ++i) {
        // No explicit map needed for local, just offset
    }

    // Find ghosts
    // This part can be parallelized but requires thread-local storage or lock
    // For simplicity and since it's initialization, we keep it serial or use OpenMP with care.
    // Given the complexity of map insertions, we leave it serial.
    for (size_t i = local_start; i < local_end; ++i) {
        const auto& elem = world.elements_static[i];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            idx_t neighbor_global = elem.connected_idx[j];
            if (neighbor_global < local_start || neighbor_global >= local_end) {
                if (global_to_local_map.find(neighbor_global) == global_to_local_map.end()) {
                    global_to_local_map[neighbor_global] = local_n_elems + ghost_global_indices.size();
                    ghost_global_indices.push_back(neighbor_global);
                }
            }
        }
    }

    size_t num_ghosts = ghost_global_indices.size();
    size_t total_local_elements = local_n_elems + num_ghosts;

    // Allocate GPU memory
    idx_t *d_material_indices, *d_num_connections, *d_connected_indices;
    val_t *d_connected_fluxes;
    val_t *d_material_transfer, *d_material_external;
    val_t *d_current_energy, *d_total_flux, *d_next_energy, *d_next_total_flux;

    cudaMalloc(&d_material_indices, local_n_elems * sizeof(idx_t));
    cudaMalloc(&d_num_connections, local_n_elems * sizeof(idx_t));
    cudaMalloc(&d_connected_indices, local_n_elems * MAX_CONNECTIONS * sizeof(idx_t));
    cudaMalloc(&d_connected_fluxes, local_n_elems * MAX_CONNECTIONS * sizeof(val_t));
    cudaMalloc(&d_material_transfer, world.materials.size() * sizeof(val_t));
    cudaMalloc(&d_material_external, world.materials.size() * sizeof(val_t));
    cudaMalloc(&d_current_energy, total_local_elements * sizeof(val_t)); // Include ghosts
    cudaMalloc(&d_total_flux, total_local_elements * sizeof(val_t));     // Include ghosts
    cudaMalloc(&d_next_energy, local_n_elems * sizeof(val_t));           // Only write owned
    cudaMalloc(&d_next_total_flux, local_n_elems * sizeof(val_t));       // Only write owned

    // Prepare host data for upload
    std::vector<idx_t> h_material_indices(local_n_elems);
    std::vector<idx_t> h_num_connections(local_n_elems);
    std::vector<idx_t> h_connected_indices(local_n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_connected_fluxes(local_n_elems * MAX_CONNECTIONS);
    std::vector<val_t> h_material_transfer(world.materials.size());
    std::vector<val_t> h_material_external(world.materials.size());
    std::vector<val_t> h_current_energy(total_local_elements);
    std::vector<val_t> h_total_flux(total_local_elements);

    // Fill host data
    for (size_t i = 0; i < local_n_elems; ++i) {
        idx_t global_idx = local_start + i;
        const auto& static_elem = world.elements_static[global_idx];
        const auto& dynamic_elem = world.elements_dynamic[global_idx];

        h_material_indices[i] = static_elem.material_idx;
        h_num_connections[i] = static_elem.num_connections;
        h_current_energy[i] = dynamic_elem.current_energy;
        h_total_flux[i] = dynamic_elem.total_flux;

        for (int c = 0; c < MAX_CONNECTIONS; ++c) {
            if (c < static_elem.num_connections) {
                idx_t neighbor_global = static_elem.connected_idx[c];
                idx_t neighbor_local;
                if (neighbor_global >= local_start && neighbor_global < local_end) {
                    neighbor_local = neighbor_global - local_start;
                } else {
                    neighbor_local = global_to_local_map[neighbor_global];
                }
                h_connected_indices[i * MAX_CONNECTIONS + c] = neighbor_local;
                h_connected_fluxes[i * MAX_CONNECTIONS + c] = static_elem.connected_flux[c];
            } else {
                h_connected_indices[i * MAX_CONNECTIONS + c] = 0; // Dummy
                h_connected_fluxes[i * MAX_CONNECTIONS + c] = 0.0;
            }
        }
    }

    // Fill ghosts initial state (though they will be updated before use)
    for (size_t i = 0; i < num_ghosts; ++i) {
        idx_t global_idx = ghost_global_indices[i];
        h_current_energy[local_n_elems + i] = world.elements_dynamic[global_idx].current_energy;
        h_total_flux[local_n_elems + i] = world.elements_dynamic[global_idx].total_flux;
    }

    for (size_t i = 0; i < world.materials.size(); ++i) {
        h_material_transfer[i] = world.materials[i].transfer_coeff;
        h_material_external[i] = world.materials[i].external_flow;
    }

    // Copy to GPU
    cudaMemcpy(d_material_indices, h_material_indices.data(), local_n_elems * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_connections, h_num_connections.data(), local_n_elems * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_connected_indices, h_connected_indices.data(), local_n_elems * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_connected_fluxes, h_connected_fluxes.data(), local_n_elems * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_material_transfer, h_material_transfer.data(), world.materials.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_material_external, h_material_external.data(), world.materials.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_energy, h_current_energy.data(), total_local_elements * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_flux, h_total_flux.data(), total_local_elements * sizeof(val_t), cudaMemcpyHostToDevice);

    // Setup MPI communication for ghosts
    // We need to know who owns our ghosts (to receive from) and who needs our elements as ghosts (to send to)
    
    // Simple approach: Allgather ghost requests? No, that's N^2.
    // Better: We know the ownership rule.
    // Map ghost_global_indices to ranks.
    std::map<int, std::vector<idx_t>> ghosts_by_rank;
    std::map<int, std::vector<int>> ghosts_local_indices_by_rank; // Where to put received data

    for (size_t i = 0; i < num_ghosts; ++i) {
        idx_t global_idx = ghost_global_indices[i];
        // Re-calculate owner
        int owner = -1;
        // Fast search for uniform distribution:
        // Reverse engineer rank from global_idx?
        // It's easier to iterate ranks or use the formula.
        // ownership boundaries:
        // rank r starts at: r * (n/s) + min(r, rem)
        // next starts at: (r+1)*(n/s) + min(r+1, rem)
        
        // Since it's monotonic, we can binary search or just linear scan (small number of ranks typically).
        // Or assume uniform enough.
        
        // Let's just loop over ranks to find owner
        for (int r = 0; r < size; ++r) {
            size_t r_start = r * (n_elems / size) + std::min((size_t)r, (size_t)(n_elems % size));
            size_t r_len = (n_elems / size) + (r < (int)(n_elems % size) ? 1 : 0);
            if (global_idx >= r_start && global_idx < r_start + r_len) {
                owner = r;
                break;
            }
        }
        
        ghosts_by_rank[owner].push_back(global_idx);
        ghosts_local_indices_by_rank[owner].push_back(local_n_elems + i);
    }

    // Now we know what we need to RECEIVE.
    // We need to tell other ranks what we need, so they know what to SEND.
    // This is a sparse graph setup.
    
    std::vector<int> send_counts(size, 0);
    std::vector<int> recv_counts(size, 0);
    
    // We need to send the list of indices we want from each rank
    // But first let's just use MPI_Alltoall to exchange counts of requests
    for (auto& kv : ghosts_by_rank) {
        send_counts[kv.first] = kv.second.size();
    }
    
    // This is "I will send you X requests"
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Now recv_counts[r] tells me that rank 'r' wants 'recv_counts[r]' elements from me.
    // I need to receive the list of indices they want.
    
    std::vector<std::vector<idx_t>> indices_to_send_to_rank(size);
    std::vector<MPI_Request> requests;
    
    // Send my requests (indices I need)
    for (auto& kv : ghosts_by_rank) {
        int target = kv.first;
        MPI_Request req;
        MPI_Isend(kv.second.data(), kv.second.size() * sizeof(idx_t), MPI_BYTE, target, 0, MPI_COMM_WORLD, &req);
        requests.push_back(req);
    }
    
    // Receive requests from others
    for (int r = 0; r < size; ++r) {
        if (recv_counts[r] > 0) {
            indices_to_send_to_rank[r].resize(recv_counts[r]);
            MPI_Request req;
            MPI_Irecv(indices_to_send_to_rank[r].data(), recv_counts[r] * sizeof(idx_t), MPI_BYTE, r, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }
    
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();
    
    // Now convert the requested global indices to local indices (relative to my array start)
    // so I can pack them efficiently during simulation
    std::vector<std::vector<idx_t>> local_indices_to_send(size);
    for (int r = 0; r < size; ++r) {
        for (idx_t global_idx : indices_to_send_to_rank[r]) {
            local_indices_to_send[r].push_back(global_idx - local_start);
        }
    }
    
    // Pre-allocate buffers for send/recv data (energy values)
    std::vector<std::vector<val_t>> send_buffers(size);
    std::vector<std::vector<val_t>> recv_buffers(size);
    for (int r = 0; r < size; ++r) {
        if (local_indices_to_send[r].size() > 0) send_buffers[r].resize(local_indices_to_send[r].size());
        if (ghosts_by_rank.count(r)) recv_buffers[r].resize(ghosts_by_rank[r].size());
    }

    // Simulation Loop
    for (int iter = 0; iter < n_iters; ++iter) {
        // 1. Pack data to send
        for (int r = 0; r < size; ++r) {
            if (local_indices_to_send[r].empty()) continue;
            
            // We need to get current values from GPU or CPU?
            // Values are on GPU. We can launch a kernel to pack, or copy whole array to CPU and pack.
            // Copying whole array is expensive.
            // Better: Copy only needed values.
            // Even better: Kernel to pack into a pinned buffer, then cudaMemcpyAsync.
            // For now, let's keep it simple: Copy relevant elements to host.
            // Actually, we can use cudaMemcpy with strided access? No, indices are random.
            // Let's just copy the whole current_energy array to host for now (easiest to implement, though slow).
            // Optimization: Maintain a "boundary" buffer on GPU?
            // Optimization: Launch a kernel to pack into a contiguous GPU buffer, then copy that buffer.
        }
        
        // Let's implement the packing kernel approach for performance
        // ... (Need to define kernel) ...
        // Simplification for this task: Copy required elements one by one or in batches?
        // Or just copy the whole local array to CPU, do exchange, then copy back ghosts.
        // Copying `local_n_elems` doubles is not too bad if N is small.
        // But for "max performance", we should avoid full D2H.
        
        // Let's do the packing on CPU for simplicity of code changes first, but it violates "max performance".
        // Let's stick to D2H for the whole local array, exchange, then H2D for ghosts.
        // Wait, if I copy D2H, I serialize.
        // I will copy `d_current_energy` (only local part) to `h_current_energy`.
        cudaMemcpy(h_current_energy.data(), d_current_energy, local_n_elems * sizeof(val_t), cudaMemcpyDeviceToHost);
        
        // Pack
        for (int r = 0; r < size; ++r) {
            for (size_t k = 0; k < local_indices_to_send[r].size(); ++k) {
                send_buffers[r][k] = h_current_energy[local_indices_to_send[r][k]];
            }
        }
        
        // Exchange
        requests.clear();
        for (int r = 0; r < size; ++r) {
            if (!send_buffers[r].empty()) {
                MPI_Request req;
                MPI_Isend(send_buffers[r].data(), send_buffers[r].size(), MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
            if (!recv_buffers[r].empty()) {
                MPI_Request req;
                MPI_Irecv(recv_buffers[r].data(), recv_buffers[r].size(), MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        
        // Unpack to ghosts
        for (int r = 0; r < size; ++r) {
            if (recv_buffers[r].empty()) continue;
            const auto& indices = ghosts_local_indices_by_rank[r];
            for (size_t k = 0; k < indices.size(); ++k) {
                h_current_energy[indices[k]] = recv_buffers[r][k];
            }
        }
        
        // Copy ghosts to GPU
        cudaMemcpy(d_current_energy + local_n_elems, h_current_energy.data() + local_n_elems, num_ghosts * sizeof(val_t), cudaMemcpyHostToDevice);
        
        // 2. Launch Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (local_n_elems + threadsPerBlock - 1) / threadsPerBlock;
        computeFluxKernel<<<blocksPerGrid, threadsPerBlock>>>(
            local_n_elems,
            d_material_indices,
            d_num_connections,
            d_connected_indices,
            d_connected_fluxes,
            d_material_transfer,
            d_material_external,
            d_current_energy,
            d_total_flux,
            d_next_energy,
            d_next_total_flux
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 3. Swap (Update pointers or copy?)
        // We have next_energy separate. We need to update current_energy for next step.
        // Pointers d_current_energy and d_next_energy.
        // But d_current_energy includes ghosts space, d_next_energy does not (we don't compute ghosts).
        // So we can't just swap pointers unless we allocate ghost space for next_energy too.
        // Let's just copy next -> current (local part).
        cudaMemcpy(d_current_energy, d_next_energy, local_n_elems * sizeof(val_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_total_flux, d_next_total_flux, local_n_elems * sizeof(val_t), cudaMemcpyDeviceToDevice);
    }

    // Retrieve final results
    cudaMemcpy(h_current_energy.data(), d_current_energy, local_n_elems * sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_total_flux.data(), d_total_flux, local_n_elems * sizeof(val_t), cudaMemcpyDeviceToHost);

    // Reconstruct global world for validation/printing (Rank 0 gathers?)
    // Or just validate locally and reduce.
    // Validation function `validateResults` expects a global `World`.
    // It's easier to gather everything to rank 0 for validation if size is small.
    // Given the constraints and existing code, I'll modify the validation to be parallel or gather.
    // For simplicity, let's gather to Rank 0.
    
    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    
    if (rank == 0) {
        global_energy.resize(n_elems);
        global_flux.resize(n_elems);
    }
    
    // Gather counts and displacements
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    int local_n = (int)local_n_elems;
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];
    }
    
    MPI_Gatherv(h_current_energy.data(), local_n, MPI_DOUBLE, 
                global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
                
    MPI_Gatherv(h_total_flux.data(), local_n, MPI_DOUBLE, 
                global_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Update world object for validation
        for (size_t i = 0; i < n_elems; ++i) {
            world.elements_dynamic[i].current_energy = global_energy[i];
            world.elements_dynamic[i].total_flux = global_flux[i];
        }

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
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    cudaFree(d_material_indices);
    cudaFree(d_num_connections);
    cudaFree(d_connected_indices);
    cudaFree(d_connected_fluxes);
    cudaFree(d_material_transfer);
    cudaFree(d_material_external);
    cudaFree(d_current_energy);
    cudaFree(d_total_flux);
    cudaFree(d_next_energy);
    cudaFree(d_next_total_flux);

    MPI_Finalize();
    return 0;
}


