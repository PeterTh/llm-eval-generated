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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2) schedule(static)
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

__global__ void updateKernel(const ElementStatic* elements, const Material* materials,
                             const val_t* current_energy, const val_t* current_flux,
                             val_t* next_energy, val_t* next_flux,
                             size_t begin, size_t end) {
    const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= end) return;
    const ElementStatic elem = elements[i];
    const Material mat = materials[elem.material_idx];
    const val_t this_energy = current_energy[i];
    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem.num_connections; ++j) {
        total_flux += (current_energy[elem.connected_idx[j]] - this_energy) *
                      mat.transfer_coeff * elem.connected_flux[j] * 0.25;
    }
    next_energy[i] = this_energy + total_flux;
    next_flux[i] = current_flux[i] + fabs(total_flux);
}

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

// Run simulation for n_iters iterations using MPI decomposition and CUDA updates.
void runSimulation(World& world, const int n_iters, int rank, int ranks) {
    const size_t n_elems = world.elements_static.size();
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "enumerating GPUs");
    if (device_count == 0) {
        fprintf(stderr, "No CUDA accelerator is available for MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int local_rank = 0;
    const char* local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!local_rank_text) local_rank_text = std::getenv("SLURM_LOCALID");
    if (!local_rank_text) local_rank_text = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (local_rank_text) local_rank = std::atoi(local_rank_text);
    cudaCheck(cudaSetDevice(local_rank % device_count), "selecting rank GPU");
    const size_t begin = n_elems * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
    const size_t end = n_elems * static_cast<size_t>(rank + 1) / static_cast<size_t>(ranks);
    const int local_count = static_cast<int>(end - begin);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t b = n_elems * static_cast<size_t>(r) / static_cast<size_t>(ranks);
        const size_t e = n_elems * static_cast<size_t>(r + 1) / static_cast<size_t>(ranks);
        counts[r] = static_cast<int>(e - b);
        displacements[r] = static_cast<int>(b);
    }

    ElementStatic* d_static = nullptr;
    Material* d_materials = nullptr;
    val_t *d_current_energy = nullptr, *d_current_flux = nullptr;
    val_t *d_next_energy = nullptr, *d_next_flux = nullptr;
    cudaCheck(cudaMalloc(&d_static, n_elems * sizeof(ElementStatic)), "allocating static elements");
    cudaCheck(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)), "allocating materials");
    cudaCheck(cudaMalloc(&d_current_energy, n_elems * sizeof(val_t)), "allocating current energies");
    cudaCheck(cudaMalloc(&d_current_flux, n_elems * sizeof(val_t)), "allocating current fluxes");
    cudaCheck(cudaMalloc(&d_next_energy, n_elems * sizeof(val_t)), "allocating next energies");
    cudaCheck(cudaMalloc(&d_next_flux, n_elems * sizeof(val_t)), "allocating next fluxes");
    cudaCheck(cudaMemcpy(d_static, world.elements_static.data(), n_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice), "copying static elements");
    cudaCheck(cudaMemcpy(d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "copying materials");
    std::vector<val_t> initial_energy(n_elems), initial_flux(n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        initial_energy[i] = world.elements_dynamic[i].current_energy;
        initial_flux[i] = world.elements_dynamic[i].total_flux;
    }
    cudaCheck(cudaMemcpy(d_current_energy, initial_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "copying initial energies");
    cudaCheck(cudaMemcpy(d_current_flux, initial_flux.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "copying initial fluxes");

    std::vector<val_t> local_energy(static_cast<size_t>(local_count));
    std::vector<val_t> all_energy(n_elems);

    for (int iter = 0; iter < n_iters; ++iter) {
        const size_t local_size = end - begin;
        const int threads = 256;
        const unsigned blocks = static_cast<unsigned>(std::max<size_t>(1, (local_size + threads - 1) / threads));
        updateKernel<<<blocks, threads>>>(
            d_static, d_materials, d_current_energy, d_current_flux,
            d_next_energy, d_next_flux, begin, end);
        cudaCheck(cudaGetLastError(), "launching update kernel");
        cudaCheck(cudaDeviceSynchronize(), "completing update kernel");
        cudaCheck(cudaMemcpy(local_energy.data(), d_next_energy + begin, local_size * sizeof(val_t), cudaMemcpyDeviceToHost), "copying local energies");
        MPI_Allgatherv(local_energy.data(), local_count, MPI_DOUBLE, all_energy.data(),
                       counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(d_next_energy, all_energy.data(), n_elems * sizeof(val_t), cudaMemcpyHostToDevice), "exchanging energies");
        std::swap(d_current_energy, d_next_energy);
        std::swap(d_current_flux, d_next_flux);
    }

    cudaCheck(cudaMemcpy(initial_energy.data(), d_current_energy, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "copying final energies");
    cudaCheck(cudaMemcpy(initial_flux.data(), d_current_flux, n_elems * sizeof(val_t), cudaMemcpyDeviceToHost), "copying final fluxes");
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = initial_energy[i];
        world.elements_dynamic[i].total_flux = initial_flux[i];
    }
    std::vector<val_t> local_flux(static_cast<size_t>(local_count));
    for (int i = 0; i < local_count; ++i) local_flux[static_cast<size_t>(i)] = world.elements_dynamic[begin + static_cast<size_t>(i)].total_flux;
    std::vector<val_t> gathered_flux(rank == 0 ? n_elems : 0);
    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE,
                rank == 0 ? gathered_flux.data() : nullptr, counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < n_elems; ++i) world.elements_dynamic[i].total_flux = gathered_flux[i];
    }
    cudaFree(d_static); cudaFree(d_materials); cudaFree(d_current_energy); cudaFree(d_current_flux);
    cudaFree(d_next_energy); cudaFree(d_next_flux);
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
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", ranks, omp_get_max_threads());
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
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
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters, rank, ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", max_duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (max_duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    if (rank == 0) {
        printf("  Result hash: %016lX\n\n", hash);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid = rank == 0 ? validateResults(world) : true;
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Finalize();
    return 0;
}
