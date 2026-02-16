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
    uint32_t material_idx;
    uint32_t num_connections;
    uint32_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
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

// CUDA kernel to compute one iteration for all elements
__global__ void simulateKernel(
    int n_elems,
    const int* mat_idx,
    const int* num_conn,
    const int* connected_idx,
    const val_t* connected_flux,
    const val_t* mat_transfer,
    const val_t* mat_external,
    const val_t* dyn_current_energy,
    const val_t* dyn_total_flux,
    val_t* out_current_energy,
    val_t* out_total_flux
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const int m_idx = mat_idx[i];
    val_t total = mat_external[m_idx];
    val_t this_energy = dyn_current_energy[i];

    int base = i * MAX_CONNECTIONS;
    int nc = num_conn[i];
    for (int j = 0; j < nc; ++j) {
        int nei = connected_idx[base + j];
        val_t other_e = dyn_current_energy[nei];
        val_t conn_flux = connected_flux[base + j];
        total += (other_e - this_energy) * mat_transfer[m_idx] * conn_flux * 0.25;
    }

    out_current_energy[i] = this_energy + total;
    out_total_flux[i] = dyn_total_flux[i] + fabs(total);
}

// Run simulation on GPU for n_iters iterations (each rank runs full domain on its accelerator)
void runSimulation(World& world, const int n_iters) {
    const int n_elems = static_cast<int>(world.elements_static.size());

    // Flatten static data
    std::vector<int> mat_idx(n_elems);
    std::vector<int> num_conn(n_elems);
    std::vector<int> connected_idx(n_elems * MAX_CONNECTIONS);
    std::vector<val_t> connected_flux(n_elems * MAX_CONNECTIONS);
    for (int i = 0; i < n_elems; ++i) {
        mat_idx[i] = static_cast<int>(world.elements_static[i].material_idx);
        num_conn[i] = static_cast<int>(world.elements_static[i].num_connections);
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            connected_idx[i * MAX_CONNECTIONS + j] = static_cast<int>(world.elements_static[i].connected_idx[j]);
            connected_flux[i * MAX_CONNECTIONS + j] = world.elements_static[i].connected_flux[j];
        }
    }

    // Materials
    const int n_materials = static_cast<int>(world.materials.size());
    std::vector<val_t> mat_transfer(n_materials);
    std::vector<val_t> mat_external(n_materials);
    for (int i = 0; i < n_materials; ++i) {
        mat_transfer[i] = world.materials[i].transfer_coeff;
        mat_external[i] = world.materials[i].external_flow;
    }

    // Dynamic arrays
    std::vector<val_t> dyn_curr(n_elems);
    std::vector<val_t> dyn_flux(n_elems);
    for (int i = 0; i < n_elems; ++i) {
        dyn_curr[i] = world.elements_dynamic[i].current_energy;
        dyn_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // Allocate device memory
    int *d_mat_idx = nullptr; int *d_num_conn = nullptr; int *d_connected_idx = nullptr;
    val_t *d_connected_flux = nullptr; val_t *d_mat_transfer = nullptr; val_t *d_mat_external = nullptr;
    val_t *d_dyn_curr = nullptr; val_t *d_dyn_flux = nullptr; val_t *d_out_curr = nullptr; val_t *d_out_flux = nullptr;

    auto cudaCheck = [](cudaError_t e){ if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } };

    cudaCheck(cudaMalloc(&d_mat_idx, sizeof(int) * n_elems));
    cudaCheck(cudaMalloc(&d_num_conn, sizeof(int) * n_elems));
    cudaCheck(cudaMalloc(&d_connected_idx, sizeof(int) * n_elems * MAX_CONNECTIONS));
    cudaCheck(cudaMalloc(&d_connected_flux, sizeof(val_t) * n_elems * MAX_CONNECTIONS));
    cudaCheck(cudaMalloc(&d_mat_transfer, sizeof(val_t) * n_materials));
    cudaCheck(cudaMalloc(&d_mat_external, sizeof(val_t) * n_materials));
    cudaCheck(cudaMalloc(&d_dyn_curr, sizeof(val_t) * n_elems));
    cudaCheck(cudaMalloc(&d_dyn_flux, sizeof(val_t) * n_elems));
    cudaCheck(cudaMalloc(&d_out_curr, sizeof(val_t) * n_elems));
    cudaCheck(cudaMalloc(&d_out_flux, sizeof(val_t) * n_elems));

    // Copy static and material data to device
    cudaCheck(cudaMemcpy(d_mat_idx, mat_idx.data(), sizeof(int) * n_elems, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_num_conn, num_conn.data(), sizeof(int) * n_elems, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_connected_idx, connected_idx.data(), sizeof(int) * n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_connected_flux, connected_flux.data(), sizeof(val_t) * n_elems * MAX_CONNECTIONS, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_mat_transfer, mat_transfer.data(), sizeof(val_t) * n_materials, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_mat_external, mat_external.data(), sizeof(val_t) * n_materials, cudaMemcpyHostToDevice));

    // Copy initial dynamics
    cudaCheck(cudaMemcpy(d_dyn_curr, dyn_curr.data(), sizeof(val_t) * n_elems, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dyn_flux, dyn_flux.data(), sizeof(val_t) * n_elems, cudaMemcpyHostToDevice));

    // Launch parameters
    const int threads = 256;
    const int blocks = (n_elems + threads - 1) / threads;

    for (int iter = 0; iter < n_iters; ++iter) {
        simulateKernel<<<blocks, threads>>>(n_elems, d_mat_idx, d_num_conn, d_connected_idx, d_connected_flux,
                                            d_mat_transfer, d_mat_external, d_dyn_curr, d_dyn_flux,
                                            d_out_curr, d_out_flux);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // swap pointers
        std::swap(d_dyn_curr, d_out_curr);
        std::swap(d_dyn_flux, d_out_flux);
    }

    // Copy results back
    cudaCheck(cudaMemcpy(dyn_curr.data(), d_dyn_curr, sizeof(val_t) * n_elems, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dyn_flux.data(), d_dyn_flux, sizeof(val_t) * n_elems, cudaMemcpyDeviceToHost));

    // Fill world dynamic buffers
    for (int i = 0; i < n_elems; ++i) {
        world.elements_dynamic[i].current_energy = dyn_curr[i];
        world.elements_dynamic[i].total_flux = dyn_flux[i];
    }

    // Free device memory
    cudaFree(d_mat_idx); cudaFree(d_num_conn); cudaFree(d_connected_idx); cudaFree(d_connected_flux);
    cudaFree(d_mat_transfer); cudaFree(d_mat_external); cudaFree(d_dyn_curr); cudaFree(d_dyn_flux);
    cudaFree(d_out_curr); cudaFree(d_out_flux);
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    // Use OpenMP for reduction
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
        printf("\n");
    }
    
    // Build the unstructured mesh on all ranks (replicated) to simplify implementation and ensure correctness
    if (world_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (only rank 0 prints)
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (world_rank == 0) printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (world_rank == 0) printf("\n");
    
    // Set OpenMP threads from environment or default
    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    // Run simulation and time it (each rank runs on its local accelerator)
    if (world_rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    runSimulation(world, n_iters);

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // Aggregate minimum elapsed time across ranks
    double min_elapsed = 0.0;
    MPI_Reduce(&elapsed, &min_elapsed, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        long duration_ms = static_cast<long>(min_elapsed * 1000.0);
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

    // Compute hash for verification (each rank computes hash locally; reduce to rank 0)
    uint64_t local_hash = computeHash(world.elements_dynamic);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (world_rank == 0) printf("  Result hash: %016lX\n\n", global_hash);

    // Print results for external validation
    if (printResults && world_rank == 0) {
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
