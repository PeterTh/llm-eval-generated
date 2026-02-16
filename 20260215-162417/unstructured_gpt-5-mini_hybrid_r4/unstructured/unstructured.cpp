#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cassert>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

// CUDA error check
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while(0)

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
    #pragma omp parallel for schedule(static)
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

// Compute energy flux between two elements (CPU fallback if needed)
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel: compute updates for a contiguous local range
extern "C" __global__ void compute_updates_kernel(
    int n_elems,
    int start_idx,
    int local_count,
    const unsigned long long* d_material_idx,
    const double* d_material_transfer_coeff,
    const double* d_material_external_flow,
    const unsigned int* d_num_connections,
    const unsigned long long* d_connected_idx,
    const double* d_connected_flux,
    const double* d_current_energy,
    double* d_swap_current_local,
    double* d_swap_total_flux_local
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= local_count) return;
    int i = start_idx + t;

    unsigned long long mat_id = d_material_idx[i];
    double transfer = d_material_transfer_coeff[mat_id];
    double external = d_material_external_flow[mat_id];

    double this_energy = d_current_energy[i];
    double total_flux = external;

    unsigned int nconn = d_num_connections[i];
    const unsigned long long* conn_base = d_connected_idx + (unsigned long long)i * (unsigned long long)MAX_CONNECTIONS;
    const double* flux_base = d_connected_flux + (unsigned long long)i * (unsigned long long)MAX_CONNECTIONS;

    for (unsigned int j = 0; j < nconn; ++j) {
        unsigned long long nb = conn_base[j];
        double other_energy = d_current_energy[nb];
        double conn_flux = flux_base[j];
        total_flux += (other_energy - this_energy) * transfer * conn_flux * 0.25;
    }

    d_swap_current_local[t] = this_energy + total_flux;
    // d_swap_total_flux_local stores cumulative abs flux increment only for this iteration
    d_swap_total_flux_local[t] = fabs(total_flux);
}

// Run simulation for n_iters iterations using MPI + OpenMP + CUDA
void runSimulationHybrid(World& world, const int n_iters, int mpi_rank, int mpi_size) {
    const int n_elems = static_cast<int>(world.elements_static.size());

    // Partition elements across ranks (contiguous blocks)
    std::vector<int> counts(mpi_size, 0);
    std::vector<int> displs(mpi_size, 0);
    int base = n_elems / mpi_size;
    int rem = n_elems % mpi_size;
    for (int r = 0; r < mpi_size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < mpi_size; ++r) displs[r] = displs[r-1] + counts[r-1];

    const int local_start = displs[mpi_rank];
    const int local_count = counts[mpi_rank];

    // Flatten static arrays for device
    std::vector<unsigned long long> h_material_idx(n_elems);
    std::vector<double> h_material_transfer_coeff(world.materials.size());
    std::vector<double> h_material_external_flow(world.materials.size());
    std::vector<unsigned int> h_num_connections(n_elems);
    std::vector<unsigned long long> h_connected_idx((size_t)n_elems * MAX_CONNECTIONS);
    std::vector<double> h_connected_flux((size_t)n_elems * MAX_CONNECTIONS);

    for (int i = 0; i < n_elems; ++i) {
        h_material_idx[i] = static_cast<unsigned long long>(world.elements_static[i].material_idx);
        h_num_connections[i] = static_cast<unsigned int>(world.elements_static[i].num_connections);
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            size_t off = (size_t)i * MAX_CONNECTIONS + j;
            if (j < static_cast<int>(world.elements_static[i].num_connections)) {
                h_connected_idx[off] = static_cast<unsigned long long>(world.elements_static[i].connected_idx[j]);
                h_connected_flux[off] = world.elements_static[i].connected_flux[j];
            } else {
                h_connected_idx[off] = 0ULL;
                h_connected_flux[off] = 0.0;
            }
        }
    }
    for (size_t m = 0; m < world.materials.size(); ++m) {
        h_material_transfer_coeff[m] = world.materials[m].transfer_coeff;
        h_material_external_flow[m] = world.materials[m].external_flow;
    }

    // Host-side arrays for energies
    std::vector<double> h_current_energy(n_elems);
    std::vector<double> h_total_flux(n_elems);
    std::vector<double> h_swap_current_local(local_count);
    std::vector<double> h_swap_total_flux_local(local_count);

    // Initialize from world
    for (int i = 0; i < n_elems; ++i) {
        h_current_energy[i] = world.elements_dynamic[i].current_energy;
        h_total_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // Initialize CUDA device for this rank (spread ranks across GPUs)
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device = 0;
    if (device_count > 0) device = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    // Allocate device memory for static arrays (once)
    unsigned long long* d_material_idx = nullptr;
    double* d_material_transfer_coeff = nullptr;
    double* d_material_external_flow = nullptr;
    unsigned int* d_num_connections = nullptr;
    unsigned long long* d_connected_idx = nullptr;
    double* d_connected_flux = nullptr;
    double* d_current_energy = nullptr;
    double* d_swap_current_local = nullptr;
    double* d_swap_total_flux_local = nullptr;

    CUDA_CHECK(cudaMalloc(&d_material_idx, sizeof(unsigned long long) * (size_t)n_elems));
    CUDA_CHECK(cudaMalloc(&d_num_connections, sizeof(unsigned int) * (size_t)n_elems));
    CUDA_CHECK(cudaMalloc(&d_connected_idx, sizeof(unsigned long long) * (size_t)n_elems * MAX_CONNECTIONS));
    CUDA_CHECK(cudaMalloc(&d_connected_flux, sizeof(double) * (size_t)n_elems * MAX_CONNECTIONS));
    CUDA_CHECK(cudaMalloc(&d_material_transfer_coeff, sizeof(double) * world.materials.size()));
    CUDA_CHECK(cudaMalloc(&d_material_external_flow, sizeof(double) * world.materials.size()));
    CUDA_CHECK(cudaMalloc(&d_current_energy, sizeof(double) * (size_t)n_elems));
    CUDA_CHECK(cudaMalloc(&d_swap_current_local, sizeof(double) * (size_t)local_count));
    CUDA_CHECK(cudaMalloc(&d_swap_total_flux_local, sizeof(double) * (size_t)local_count));

    CUDA_CHECK(cudaMemcpy(d_material_idx, h_material_idx.data(), sizeof(unsigned long long) * (size_t)n_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_connections, h_num_connections.data(), sizeof(unsigned int) * (size_t)n_elems, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_idx, h_connected_idx.data(), sizeof(unsigned long long) * (size_t)n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_connected_flux, h_connected_flux.data(), sizeof(double) * (size_t)n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_material_transfer_coeff, h_material_transfer_coeff.data(), sizeof(double) * world.materials.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_material_external_flow, h_material_external_flow.data(), sizeof(double) * world.materials.size(), cudaMemcpyHostToDevice));

    // Prepare MPI allgatherv parameters
    std::vector<int> sendcounts(mpi_size), recvcounts(mpi_size), rdispls(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        sendcounts[r] = counts[r];
        recvcounts[r] = counts[r];
        rdispls[r] = displs[r];
    }

    // Main iterations
    const int threads = 256;
    const int blocks = (local_count + threads - 1) / threads;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Copy global current energy to device
        CUDA_CHECK(cudaMemcpy(d_current_energy, h_current_energy.data(), sizeof(double) * (size_t)n_elems, cudaMemcpyHostToDevice));

        // Launch kernel to compute updates for local range
        compute_updates_kernel<<<blocks, threads>>>(
            n_elems,
            local_start,
            local_count,
            d_material_idx,
            d_material_transfer_coeff,
            d_material_external_flow,
            d_num_connections,
            d_connected_idx,
            d_connected_flux,
            d_current_energy,
            d_swap_current_local,
            d_swap_total_flux_local
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy back local swap results
        CUDA_CHECK(cudaMemcpy(h_swap_current_local.data(), d_swap_current_local, sizeof(double) * (size_t)local_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_swap_total_flux_local.data(), d_swap_total_flux_local, sizeof(double) * (size_t)local_count, cudaMemcpyDeviceToHost));

        // Fill local portion of swap arrays
        for (int t = 0; t < local_count; ++t) {
            int idx = local_start + t;
            world.elements_dynamic_swap[idx].current_energy = h_swap_current_local[t];
            // accumulate previous total_flux + iteration's abs flux
            world.elements_dynamic_swap[idx].total_flux = world.elements_dynamic[idx].total_flux + h_swap_total_flux_local[t];
        }

        // Prepare send buffers for MPI_Allgatherv: send local swap current_energy and total_flux
        std::vector<double> send_curr(local_count), send_flux(local_count);
        for (int t = 0; t < local_count; ++t) {
            int idx = local_start + t;
            send_curr[t] = world.elements_dynamic_swap[idx].current_energy;
            send_flux[t] = world.elements_dynamic_swap[idx].total_flux;
        }

        // Receive full arrays
        std::vector<double> recv_curr(n_elems);
        std::vector<double> recv_flux(n_elems);

        MPI_Allgatherv(send_curr.data(), local_count, MPI_DOUBLE, recv_curr.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(send_flux.data(), local_count, MPI_DOUBLE, recv_flux.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Update world.elements_dynamic with gathered swap results
        for (int i = 0; i < n_elems; ++i) {
            world.elements_dynamic[i].current_energy = recv_curr[i];
            world.elements_dynamic[i].total_flux = recv_flux[i];
            h_current_energy[i] = recv_curr[i];
            h_total_flux[i] = recv_flux[i];
        }
    }

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_material_idx));
    CUDA_CHECK(cudaFree(d_num_connections));
    CUDA_CHECK(cudaFree(d_connected_idx));
    CUDA_CHECK(cudaFree(d_connected_flux));
    CUDA_CHECK(cudaFree(d_material_transfer_coeff));
    CUDA_CHECK(cudaFree(d_material_external_flow));
    CUDA_CHECK(cudaFree(d_current_energy));
    CUDA_CHECK(cudaFree(d_swap_current_local));
    CUDA_CHECK(cudaFree(d_swap_total_flux_local));
}

// Validate simulation results (parallel-aware: only rank 0 prints)
bool validateResultsParallel(const World& world, int mpi_rank) {
    if (mpi_rank != 0) return true; // only root validates/prints
    return validateResults(world);
}

// Compute a simple hash of the results for verification (only root prints)
uint64_t computeHashParallel(const std::vector<ElementDynamic>& elements, int mpi_rank) {
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
    if (MPI::COMM_WORLD.Get_rank() != 0) return;
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
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse same args)
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (i > 0) {
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
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh (all ranks build identical structure)
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage (only root prints)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (mpi_rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run hybrid simulation
    if (mpi_rank == 0) printf("Running hybrid MPI+OpenMP+CUDA simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulationHybrid(world, n_iters, mpi_rank, mpi_size);

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
        const uint64_t hash = computeHashParallel(world.elements_dynamic, mpi_rank);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (only root)
    if (printResults && mpi_rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation (root only prints)
    if (validate) {
        bool valid = validateResultsParallel(world, mpi_rank);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
