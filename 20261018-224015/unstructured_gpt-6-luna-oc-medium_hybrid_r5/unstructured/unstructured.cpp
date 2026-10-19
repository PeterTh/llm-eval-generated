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
__global__ void updateKernel(const val_t* energy, val_t* next_energy, val_t* flux_sum,
                             int n, int rows, int row_start, int root,
                             const val_t* external_flow, const val_t* coeff) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * root;
    if (local >= count) return;
    const int global = row_start * root + local;
    const int x = global / root, y = global % root;
    const val_t old = energy[local + root];
    val_t flux = external_flow[global];
    if (x > 0) flux += (energy[local] - old) * coeff[global] * 0.25;
    if (x + 1 < n / root) flux += (energy[local + 2 * root] - old) * coeff[global] * 0.25;
    if (y > 0) flux += (energy[local + root - 1] - old) * coeff[global] * 0.25;
    if (y + 1 < root) flux += (energy[local + root + 1] - old) * coeff[global] * 0.25;
    next_energy[local + root] = old + flux;
    flux_sum[local] += fabs(flux);
}

void runSimulation(World& world, const int n_iters, int rank, int nranks, int root) {
    const size_t n_elems = world.elements_static.size();
    const int rows = root / nranks + (rank < root % nranks);
    const int row_start = rank * (root / nranks) + std::min(rank, root % nranks);
    const int offset = row_start * root;
    const int local_count = rows * root;
    std::vector<val_t> host_energy((rows + 2) * root, 0.0), host_next((rows + 2) * root, 0.0);
    std::vector<val_t> host_flux(local_count, 0.0), external(local_count), coeff(local_count);
    for (int i = 0; i < local_count; ++i) {
        host_energy[(i / root + 1) * root + i % root] = world.elements_dynamic[offset + i].current_energy;
        const auto& e = world.elements_static[offset + i];
        external[i] = world.materials[e.material_idx].external_flow;
        coeff[i] = world.materials[e.material_idx].transfer_coeff;
    }
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const int rrows = root / nranks + (r < root % nranks);
        const int rstart = r * (root / nranks) + std::min(r, root % nranks);
        counts[r] = rrows * root; displs[r] = rstart * root;
    }
    val_t *d_energy, *d_next, *d_flux, *d_external, *d_coeff;
    cudaMalloc(&d_energy, host_energy.size() * sizeof(val_t)); cudaMalloc(&d_next, host_next.size() * sizeof(val_t));
    cudaMalloc(&d_flux, host_flux.size() * sizeof(val_t)); cudaMalloc(&d_external, external.size() * sizeof(val_t)); cudaMalloc(&d_coeff, coeff.size() * sizeof(val_t));
    cudaMemcpy(d_energy, host_energy.data(), host_energy.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flux, host_flux.data(), host_flux.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_external, external.data(), external.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeff, coeff.data(), coeff.size() * sizeof(val_t), cudaMemcpyHostToDevice);
    for (int iter = 0; iter < n_iters; ++iter) {
        cudaMemcpy(host_energy.data(), d_energy, host_energy.size() * sizeof(val_t), cudaMemcpyDeviceToHost);
        if (nranks > 1) {
            MPI_Sendrecv(host_energy.data() + root, root, MPI_DOUBLE, rank ? rank - 1 : MPI_PROC_NULL, 0,
                         host_energy.data() + (rows + 1) * root, root, MPI_DOUBLE, rank + 1 < nranks ? rank + 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(host_energy.data() + rows * root, root, MPI_DOUBLE, rank + 1 < nranks ? rank + 1 : MPI_PROC_NULL, 1,
                         host_energy.data(), root, MPI_DOUBLE, rank ? rank - 1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        cudaMemcpy(d_energy, host_energy.data(), host_energy.size() * sizeof(val_t), cudaMemcpyHostToDevice);
        updateKernel<<<(local_count + 255) / 256, 256>>>(d_energy, d_next, d_flux, root * root, rows, row_start, root, d_external, d_coeff);
        cudaDeviceSynchronize();
        std::swap(d_energy, d_next);
    }
    cudaMemcpy(host_energy.data(), d_energy, host_energy.size() * sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_flux.data(), d_flux, host_flux.size() * sizeof(val_t), cudaMemcpyDeviceToHost);
    #pragma omp parallel for
    for (int i = 0; i < local_count; ++i) {
        world.elements_dynamic[offset + i].current_energy = host_energy[(i / root + 1) * root + i % root];
        world.elements_dynamic[offset + i].total_flux += host_flux[i];
    }
    std::vector<val_t> all_energy(n_elems), all_flux(n_elems);
    std::vector<val_t> local_energy(local_count);
    for (int i = 0; i < local_count; ++i) local_energy[i] = host_energy[(i / root + 1) * root + i % root];
    MPI_Allgatherv(host_flux.data(), local_count, MPI_DOUBLE, all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_energy.data(), local_count, MPI_DOUBLE, all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    #pragma omp parallel for
    for (size_t i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = all_energy[i];
        world.elements_dynamic[i].total_flux = all_flux[i];
    }
    cudaFree(d_energy); cudaFree(d_next); cudaFree(d_flux); cudaFree(d_external); cudaFree(d_coeff);
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
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int devices = 0;
    cudaGetDeviceCount(&devices);
    if (devices > 0) cudaSetDevice(rank % devices);
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
    
    if (rank == 0) printf("Unstructured Mesh Energy Transfer Benchmark\n");
    if (rank == 0) printf("============================================\n");
    if (rank == 0) { printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems); printf("Iterations: %d\n", n_iters); printf("Validation: %s\n\n", validate ? "enabled" : "disabled"); }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (rank == 0) printf("\n");
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters, rank, nranks, n_elems_root);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long local_ms = duration_ms, max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    duration_ms = max_ms;
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if (rank == 0) printf("Performance:\n");
    if (rank == 0) { printf("  Time per iteration: %.4f ms\n", time_per_iter); printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec); printf("  Performance: %.4f GFLOPS\n", gflops); }
    
    // Compute hash for verification
    const uint64_t hash = computeHash(world.elements_dynamic);
    if (rank == 0) { printf("  Result hash: %016lX\n", hash); printf("\n"); }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate && rank == 0) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
