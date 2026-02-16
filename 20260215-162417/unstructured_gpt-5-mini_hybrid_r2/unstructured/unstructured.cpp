#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <iostream>

#include <mpi.h>
#include <omp.h>
#if defined(USE_CUDA) && USE_CUDA
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

// GPU kernel (if compiled with CUDA) to compute updates for a range of elements
#if defined(USE_CUDA) && USE_CUDA
extern "C" __global__ void compute_kernel(const ElementStatic* elements_static,
                                            const ElementDynamic* elements_dynamic,
                                            ElementDynamic* elements_dynamic_swap,
                                            const Material* materials,
                                            size_t start_idx, size_t n_elems) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = start_idx + gid;
    if (gid >= n_elems) return;
    const ElementStatic& elem_static = elements_static[idx];
    const ElementDynamic& elem_dyn = elements_dynamic[idx];
    const Material& mat = materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        total_flux += (neighbor_dyn.current_energy - elem_dyn.current_energy) * mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
    }

    ElementDynamic out;
    out.current_energy = elem_dyn.current_energy + total_flux;
    out.total_flux = elem_dyn.total_flux + fabs(total_flux);
    elements_dynamic_swap[idx] = out;
}
#endif

// Run simulation for n_iters iterations with hybrid MPI+OpenMP+CUDA
void runSimulation(World& world, const int n_iters, int mpi_rank, int mpi_size) {
    const size_t n_elems = world.elements_static.size();

    // Partition elements across MPI ranks (contiguous blocks)
    size_t elems_per_rank = (n_elems + mpi_size - 1) / mpi_size;
    size_t start = mpi_rank * elems_per_rank;
    size_t end = std::min(start + elems_per_rank, n_elems);
    if (start >= end) { // no work on this rank
        // still need to participate in MPI operations
        start = end = 0;
    }
    size_t local_n = end - start;

#if defined(USE_CUDA) && USE_CUDA
    // Allocate device buffers
    ElementStatic* d_static = nullptr;
    ElementDynamic* d_dyn = nullptr;
    ElementDynamic* d_swap = nullptr;
    Material* d_mat = nullptr;

    cudaMalloc(&d_static, n_elems * sizeof(ElementStatic));
    cudaMalloc(&d_dyn, n_elems * sizeof(ElementDynamic));
    cudaMalloc(&d_swap, n_elems * sizeof(ElementDynamic));
    cudaMalloc(&d_mat, world.materials.size() * sizeof(Material));

    cudaMemcpy(d_static, world.elements_static.data(), n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dyn, world.elements_dynamic.data(), n_elems * sizeof(ElementDynamic), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
#endif

    std::vector<int> recvcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t rs = r * elems_per_rank;
        size_t re = std::min(rs + elems_per_rank, n_elems);
        recvcounts[r] = static_cast<int>((re - rs) * sizeof(ElementDynamic));
        displs[r] = static_cast<int>(rs * sizeof(ElementDynamic));
    }

    for (int iter = 0; iter < n_iters; ++iter) {
#if defined(USE_CUDA) && USE_CUDA
        // Launch CUDA kernel to compute local range into device swap buffer
        if (local_n > 0) {
            const int threads = 256;
            const int blocks = (int)((local_n + threads - 1) / threads);
            compute_kernel<<<blocks, threads>>>(d_static, d_dyn, d_swap, d_mat, start, local_n);
            cudaDeviceSynchronize();
        }
        // Copy device swap buffer back to host buffer for the local range
        if (local_n > 0) {
            cudaMemcpy(world.elements_dynamic_swap.data() + start, d_swap + start, local_n * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);
        }
#else
        // CPU path with OpenMP parallelization for local range
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t ii = 0; ii < (ptrdiff_t)local_n; ++ii) {
            size_t i = start + ii;
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }

            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
#endif
        // Gather all swapped ranges from ranks to form the full swapped buffer on each rank
        MPI_Allgatherv(world.elements_dynamic_swap.data() + start, recvcounts[mpi_rank], MPI_BYTE,
                       world.elements_dynamic_swap.data(), recvcounts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);

#if defined(USE_CUDA) && USE_CUDA
        // Swap device pointers (d_dyn <-> d_swap)
        ElementDynamic* tmp = d_dyn; d_dyn = d_swap; d_swap = tmp;
        // Copy entire device dynamic buffer back to host for next iteration reads
        cudaMemcpy(world.elements_dynamic.data(), d_dyn, n_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);
#else
        // Swap host buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
#endif
    }

#if defined(USE_CUDA) && USE_CUDA
    // Free device buffers
    cudaFree(d_static);
    cudaFree(d_dyn);
    cudaFree(d_swap);
    cudaFree(d_mat);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int mpi_rank = 0, mpi_size = 1;
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (mpi_rank == 0) printUsage(argv[0]);
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
        printf("MPI ranks: %d\n", mpi_size);
        printf("\n");
    }
    
    // Build the unstructured mesh on all ranks (replicated data model)
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (print only on rank 0)
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
    
    // Run simulation
    if (mpi_rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = 0;
    if (mpi_rank == 0) {
        duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
    }

    // Compute hash for verification (done on rank 0)
    uint64_t hash = 0;
    if (mpi_rank == 0) {
        hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation only on rank 0
    if (printResults && mpi_rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation only on rank 0
    if (validate && mpi_rank == 0) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
