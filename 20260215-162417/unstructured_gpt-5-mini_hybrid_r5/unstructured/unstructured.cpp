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
#include <mpi.h>
#include <omp.h>
#ifndef USE_CUDA
#define USE_CUDA 0
#endif
#if USE_CUDA
#include <cuda_runtime.h>
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

// Run simulation for n_iters iterations using MPI + OpenMP + CUDA hybrid
void runSimulation(World& world, const int n_iters) {
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const int n_elems = static_cast<int>(world.elements_static.size());

    // Prepare host-side flat arrays for GPU
    std::vector<uint64_t> h_material_idx(n_elems);
    std::vector<int> h_num_connections(n_elems);
    std::vector<uint64_t> h_connected_idx(static_cast<size_t>(n_elems) * MAX_CONNECTIONS);
    std::vector<double> h_connected_flux(static_cast<size_t>(n_elems) * MAX_CONNECTIONS);
    std::vector<double> h_transfer_coeff(world.materials.size());
    std::vector<double> h_external_flow(world.materials.size());
    std::vector<double> h_current_energy(n_elems);
    std::vector<double> h_current_total(n_elems);
    std::vector<double> h_next_energy(n_elems);
    std::vector<double> h_next_total(n_elems);

    for (int i = 0; i < n_elems; ++i) {
        const auto &es = world.elements_static[i];
        h_material_idx[i] = es.material_idx;
        h_num_connections[i] = static_cast<int>(es.num_connections);
        int base = i * MAX_CONNECTIONS;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_connected_idx[base + j] = es.connected_idx[j];
            h_connected_flux[base + j] = es.connected_flux[j];
        }
        h_current_energy[i] = world.elements_dynamic[i].current_energy;
        h_current_total[i] = world.elements_dynamic[i].total_flux;
    }

    for (size_t m = 0; m < world.materials.size(); ++m) {
        h_transfer_coeff[m] = world.materials[m].transfer_coeff;
        h_external_flow[m] = world.materials[m].external_flow;
    }

    // Determine local range for this MPI rank
    int base_count = n_elems / mpi_size;
    int remainder = n_elems % mpi_size;
    int local_start = mpi_rank * base_count + std::min(mpi_rank, remainder);
    int local_count = base_count + (mpi_rank < remainder ? 1 : 0);
    int local_end = local_start + local_count;

    // Setup MPI gather counts and displacements
    std::vector<int> sendcounts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        int sc = base_count + (r < remainder ? 1 : 0);
        sendcounts[r] = sc;
        displs[r] = (r == 0) ? 0 : (displs[r-1] + sendcounts[r-1]);
    }

#if USE_CUDA
    // CUDA device buffers
    uint64_t *d_material_idx = nullptr;
    int *d_num_connections = nullptr;
    uint64_t *d_connected_idx = nullptr;
    double *d_connected_flux = nullptr;
    double *d_transfer_coeff = nullptr;
    double *d_external_flow = nullptr;
    double *d_current_energy = nullptr;
    double *d_current_total = nullptr;
    double *d_next_energy = nullptr;
    double *d_next_total = nullptr;

    auto checkCuda = [](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    };

    // Allocate and copy
    checkCuda(cudaMalloc(&d_material_idx, sizeof(uint64_t) * n_elems), "malloc mat idx");
    checkCuda(cudaMalloc(&d_num_connections, sizeof(int) * n_elems), "malloc num conn");
    checkCuda(cudaMalloc(&d_connected_idx, sizeof(uint64_t) * n_elems * MAX_CONNECTIONS), "malloc conn idx");
    checkCuda(cudaMalloc(&d_connected_flux, sizeof(double) * n_elems * MAX_CONNECTIONS), "malloc conn flux");
    checkCuda(cudaMalloc(&d_transfer_coeff, sizeof(double) * world.materials.size()), "malloc transfer coeff");
    checkCuda(cudaMalloc(&d_external_flow, sizeof(double) * world.materials.size()), "malloc external flow");
    checkCuda(cudaMalloc(&d_current_energy, sizeof(double) * n_elems), "malloc cur energy");
    checkCuda(cudaMalloc(&d_current_total, sizeof(double) * n_elems), "malloc cur total");
    checkCuda(cudaMalloc(&d_next_energy, sizeof(double) * n_elems), "malloc next energy");
    checkCuda(cudaMalloc(&d_next_total, sizeof(double) * n_elems), "malloc next total");

    checkCuda(cudaMemcpy(d_material_idx, h_material_idx.data(), sizeof(uint64_t) * n_elems, cudaMemcpyHostToDevice), "cpy mat idx");
    checkCuda(cudaMemcpy(d_num_connections, h_num_connections.data(), sizeof(int) * n_elems, cudaMemcpyHostToDevice), "cpy num conn");
    checkCuda(cudaMemcpy(d_connected_idx, h_connected_idx.data(), sizeof(uint64_t) * n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice), "cpy conn idx");
    checkCuda(cudaMemcpy(d_connected_flux, h_connected_flux.data(), sizeof(double) * n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice), "cpy conn flux");
    checkCuda(cudaMemcpy(d_transfer_coeff, h_transfer_coeff.data(), sizeof(double) * world.materials.size(), cudaMemcpyHostToDevice), "cpy transfer coeff");
    checkCuda(cudaMemcpy(d_external_flow, h_external_flow.data(), sizeof(double) * world.materials.size(), cudaMemcpyHostToDevice), "cpy external flow");
    checkCuda(cudaMemcpy(d_current_energy, h_current_energy.data(), sizeof(double) * n_elems, cudaMemcpyHostToDevice), "cpy cur energy");
    checkCuda(cudaMemcpy(d_current_total, h_current_total.data(), sizeof(double) * n_elems, cudaMemcpyHostToDevice), "cpy cur total");

    // Kernel definition
    auto threadsPerBlock = 256;
    auto blocksPerGrid = (local_count + threadsPerBlock - 1) / threadsPerBlock;

    // Main iteration loop: each rank computes its local slice on GPU, then allgathers to synchronize
    for (int iter = 0; iter < n_iters; ++iter) {
        // Launch kernel to update local slice
        update_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_material_idx, d_num_connections, d_connected_idx, d_connected_flux,
            d_transfer_coeff, d_external_flow,
            d_current_energy, d_current_total,
            d_next_energy, d_next_total,
            n_elems, local_start, local_end);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel sync");

        // Copy local updated segment back to host
        if (local_count > 0) {
            checkCuda(cudaMemcpy(h_next_energy.data() + local_start, d_next_energy + local_start, sizeof(double) * local_count, cudaMemcpyDeviceToHost), "cpy next energy to host");
            checkCuda(cudaMemcpy(h_next_total.data() + local_start, d_next_total + local_start, sizeof(double) * local_count, cudaMemcpyDeviceToHost), "cpy next total to host");
        }

        // Allgather to form full arrays on every rank
        MPI_Allgatherv(h_next_energy.data() + local_start, local_count, MPI_DOUBLE,
                       h_current_energy.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        MPI_Allgatherv(h_next_total.data() + local_start, local_count, MPI_DOUBLE,
                       h_current_total.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Copy full arrays back to device for next iteration
        checkCuda(cudaMemcpy(d_current_energy, h_current_energy.data(), sizeof(double) * n_elems, cudaMemcpyHostToDevice), "cpy full cur energy to device");
        checkCuda(cudaMemcpy(d_current_total, h_current_total.data(), sizeof(double) * n_elems, cudaMemcpyHostToDevice), "cpy full cur total to device");

        // Swap device buffers: make next the current for next iter
        std::swap(d_current_energy, d_next_energy);
        std::swap(d_current_total, d_next_total);
    }

    // Final gather: ensure rank 0 has full arrays (already synchronized)
    if (mpi_rank == 0) {
        // Copy device arrays back to host for final results
        checkCuda(cudaMemcpy(h_current_energy.data(), d_current_energy, sizeof(double) * n_elems, cudaMemcpyDeviceToHost), "final cpy energy");
        checkCuda(cudaMemcpy(h_current_total.data(), d_current_total, sizeof(double) * n_elems, cudaMemcpyDeviceToHost), "final cpy total");

        // Update world dynamic arrays
        for (int i = 0; i < n_elems; ++i) {
            world.elements_dynamic[i].current_energy = h_current_energy[i];
            world.elements_dynamic[i].total_flux = h_current_total[i];
        }
    }

    // Free GPU memory
    cudaFree(d_material_idx);
    cudaFree(d_num_connections);
    cudaFree(d_connected_idx);
    cudaFree(d_connected_flux);
    cudaFree(d_transfer_coeff);
    cudaFree(d_external_flow);
    cudaFree(d_current_energy);
    cudaFree(d_current_total);
    cudaFree(d_next_energy);
    cudaFree(d_next_total);

    // Ensure all ranks reach here
    MPI_Barrier(MPI_COMM_WORLD);
#else
    // CPU-only fallback using OpenMP for local slice and MPI_Allgatherv for synchronization
    // Main iteration loop
    for (int iter = 0; iter < n_iters; ++iter) {
        // Compute local slice in parallel with OpenMP
        #pragma omp parallel for schedule(static)
        for (int gid = local_start; gid < local_end; ++gid) {
            const ElementStatic &es = world.elements_static[gid];
            const ElementDynamic &ed = world.elements_dynamic[gid];
            const Material &mat = world.materials[es.material_idx];
            double total_flux = mat.external_flow;
            for (int j = 0; j < static_cast<int>(es.num_connections); ++j) {
                const int neighbor = static_cast<int>(es.connected_idx[j]);
                const double other = world.elements_dynamic[neighbor].current_energy;
                total_flux += (other - ed.current_energy) * mat.transfer_coeff * es.connected_flux[j] * 0.25;
            }
            h_next_energy[gid] = ed.current_energy + total_flux;
            h_next_total[gid] = ed.total_flux + fabs(total_flux);
        }

        // Allgather local results to form full arrays on every rank
        MPI_Allgatherv(h_next_energy.data() + local_start, local_count, MPI_DOUBLE,
                       h_current_energy.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_next_total.data() + local_start, local_count, MPI_DOUBLE,
                       h_current_total.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Update world dynamic arrays on every rank for next iteration
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_elems; ++i) {
            world.elements_dynamic[i].current_energy = h_current_energy[i];
            world.elements_dynamic[i].total_flux = h_current_total[i];
        }
    }

    // Ensure rank 0 has final results
    if (mpi_rank == 0) {
        for (int i = 0; i < n_elems; ++i) {
            // world already updated on rank 0 in loop
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

#if USE_CUDA
// CUDA kernel for element updates
extern "C" __global__ void update_kernel(const uint64_t* material_idx,
                                           const int* num_connections,
                                           const uint64_t* connected_idx,
                                           const double* connected_flux,
                                           const double* transfer_coeff,
                                           const double* external_flow,
                                           const double* current_energy,
                                           const double* current_total,
                                           double* next_energy,
                                           double* next_total,
                                           const int n_elems,
                                           const int start_idx,
                                           const int end_idx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (gid >= end_idx || gid >= n_elems) return;

    int mat = static_cast<int>(material_idx[gid]);
    double my_energy = current_energy[gid];
    double total_flux = external_flow[mat];
    int nc = num_connections[gid];
    int base = gid * MAX_CONNECTIONS;
    for (int j = 0; j < nc; ++j) {
        uint64_t neighbor = connected_idx[base + j];
        double other = current_energy[neighbor];
        double conn_flux = connected_flux[base + j];
        total_flux += (other - my_energy) * transfer_coeff[mat] * conn_flux * 0.25;
    }
    next_energy[gid] = my_energy + total_flux;
    next_total[gid] = current_total[gid] + fabs(total_flux);
}
#endif

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
    
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("============================================\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    // Build the unstructured mesh
    printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    printf("\n");
    
    // Run simulation
    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
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
            return 1;
        }
    }
    
    return 0;
}
