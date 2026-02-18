#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

constexpr int MATERIAL_COUNT = 3;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

__constant__ val_t d_transfer_coeff[MATERIAL_COUNT];
__constant__ val_t d_external_flow[MATERIAL_COUNT];

struct RowPartition {
    int start_row;
    int rows;
};

RowPartition computeRowPartition(const int rank, const int size, const int n_rows) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    RowPartition part{};
    if (rank < rem) {
        part.rows = base + 1;
        part.start_row = rank * (base + 1);
    } else {
        part.rows = base;
        part.start_row = rem * (base + 1) + (rank - rem) * base;
    }
    return part;
}

__global__ void update_kernel(const val_t* energy, const val_t* flux, val_t* energy_out,
                              val_t* flux_out, const int n_cols, const int local_rows,
                              const int global_row_start, const int n_rows_total) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n_cols || local_row >= local_rows) {
        return;
    }

    const int global_row = global_row_start + local_row;
    const int row_with_halo = local_row + 1;
    const size_t idx = static_cast<size_t>(row_with_halo) * n_cols + col;

    int mat_idx = DEFAULT_MAT_ID;
    if ((global_row == 0 && col == 0) || (global_row == n_rows_total - 1 && col == n_cols - 1)) {
        mat_idx = INFLOW_MAT_ID;
    } else if ((global_row == 0 && col == n_cols - 1) ||
               (global_row == n_rows_total - 1 && col == 0)) {
        mat_idx = OUTFLOW_MAT_ID;
    }

    const val_t transfer_coeff = d_transfer_coeff[mat_idx];
    val_t total_flux = d_external_flow[mat_idx];
    const val_t current_energy = energy[idx];

    if (global_row > 0) {
        const val_t neighbor = energy[idx - n_cols];
        total_flux += (neighbor - current_energy) * transfer_coeff * 0.25;
    }
    if (global_row + 1 < n_rows_total) {
        const val_t neighbor = energy[idx + n_cols];
        total_flux += (neighbor - current_energy) * transfer_coeff * 0.25;
    }
    if (col > 0) {
        const val_t neighbor = energy[idx - 1];
        total_flux += (neighbor - current_energy) * transfer_coeff * 0.25;
    }
    if (col + 1 < n_cols) {
        const val_t neighbor = energy[idx + 1];
        total_flux += (neighbor - current_energy) * transfer_coeff * 0.25;
    }

    energy_out[idx] = current_energy + total_flux;
    flux_out[idx] = flux[idx] + fabs(total_flux);
}

// Build a 2D square grid for local rows
void buildSquare2D(World& world, const int n_elems_root, const int local_rows) {
    // Initialize materials
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const size_t local_elems = static_cast<size_t>(local_rows) * n_elems_root;
    world.elements_static.clear();
    world.elements_dynamic.resize(local_elems);
    world.elements_dynamic_swap.resize(local_elems);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_elems; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations using CUDA + MPI
void runSimulation(World& world, const int n_iters, const int n_elems_root, const int local_rows,
                   const int global_row_start, const int rank, const int size) {
    const int n_cols = n_elems_root;
    const int rows_with_halo = local_rows + 2;
    const size_t elems_with_halo = static_cast<size_t>(rows_with_halo) * n_cols;

    val_t* energy_d = nullptr;
    val_t* energy_next_d = nullptr;
    val_t* flux_d = nullptr;
    val_t* flux_next_d = nullptr;
    CUDA_CHECK(cudaMalloc(&energy_d, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&energy_next_d, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&flux_d, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&flux_next_d, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(energy_d, 0, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(energy_next_d, 0, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(flux_d, 0, elems_with_halo * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(flux_next_d, 0, elems_with_halo * sizeof(val_t)));

    std::vector<val_t> host_energy(static_cast<size_t>(local_rows) * n_cols);
    std::vector<val_t> host_flux(static_cast<size_t>(local_rows) * n_cols);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < host_energy.size(); ++i) {
        host_energy[i] = 0.0;
        host_flux[i] = 0.0;
    }

    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy2D(energy_d + n_cols, n_cols * sizeof(val_t), host_energy.data(),
                                n_cols * sizeof(val_t), n_cols * sizeof(val_t), local_rows,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(flux_d + n_cols, n_cols * sizeof(val_t), host_flux.data(),
                                n_cols * sizeof(val_t), n_cols * sizeof(val_t), local_rows,
                                cudaMemcpyHostToDevice));
    }

    val_t transfer[MATERIAL_COUNT];
    val_t external[MATERIAL_COUNT];
    for (int i = 0; i < MATERIAL_COUNT; ++i) {
        transfer[i] = world.materials[i].transfer_coeff;
        external[i] = world.materials[i].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_transfer_coeff, transfer, sizeof(transfer)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_external_flow, external, sizeof(external)));

    const RowPartition top_part = (rank > 0) ? computeRowPartition(rank - 1, size, n_elems_root)
                                             : RowPartition{0, 0};
    const RowPartition bottom_part = (rank + 1 < size)
                                         ? computeRowPartition(rank + 1, size, n_elems_root)
                                         : RowPartition{0, 0};
    const int top_rank = (rank > 0 && top_part.rows > 0) ? rank - 1 : MPI_PROC_NULL;
    const int bottom_rank = (rank + 1 < size && bottom_part.rows > 0) ? rank + 1 : MPI_PROC_NULL;

    std::vector<val_t> send_top(n_cols);
    std::vector<val_t> send_bottom(n_cols);
    std::vector<val_t> recv_top(n_cols);
    std::vector<val_t> recv_bottom(n_cols);

    const dim3 block(16, 16);
    const dim3 grid((n_cols + block.x - 1) / block.x,
                    (local_rows + block.y - 1) / block.y);

    for (int iter = 0; iter < n_iters; ++iter) {
        if (size > 1 && local_rows > 0) {
            if (top_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(send_top.data(), energy_d + n_cols,
                                      n_cols * sizeof(val_t), cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_top.data(), n_cols, MPI_DOUBLE, top_rank, 0,
                             recv_top.data(), n_cols, MPI_DOUBLE, top_rank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CUDA_CHECK(cudaMemcpy(energy_d, recv_top.data(), n_cols * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
            }
            if (bottom_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(send_bottom.data(),
                                      energy_d + static_cast<size_t>(local_rows) * n_cols,
                                      n_cols * sizeof(val_t), cudaMemcpyDeviceToHost));
                MPI_Sendrecv(send_bottom.data(), n_cols, MPI_DOUBLE, bottom_rank, 1,
                             recv_bottom.data(), n_cols, MPI_DOUBLE, bottom_rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CUDA_CHECK(cudaMemcpy(energy_d + static_cast<size_t>(local_rows + 1) * n_cols,
                                      recv_bottom.data(), n_cols * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
            }
        }

        if (local_rows > 0) {
            update_kernel<<<grid, block>>>(energy_d, flux_d, energy_next_d, flux_next_d,
                                           n_cols, local_rows, global_row_start,
                                           n_elems_root);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(energy_d, energy_next_d);
            std::swap(flux_d, flux_next_d);
        }
    }

    if (local_rows > 0) {
        for (int row = 0; row < local_rows; ++row) {
            CUDA_CHECK(cudaMemcpy(host_energy.data() + static_cast<size_t>(row) * n_cols,
                                  energy_d + static_cast<size_t>(row + 1) * n_cols,
                                  n_cols * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_flux.data() + static_cast<size_t>(row) * n_cols,
                                  flux_d + static_cast<size_t>(row + 1) * n_cols,
                                  n_cols * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
    }

    world.elements_dynamic.resize(host_energy.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < host_energy.size(); ++i) {
        world.elements_dynamic[i].current_energy = host_energy[i];
        world.elements_dynamic[i].total_flux = host_flux[i];
    }

    CUDA_CHECK(cudaFree(energy_d));
    CUDA_CHECK(cudaFree(energy_next_d));
    CUDA_CHECK(cudaFree(flux_d));
    CUDA_CHECK(cudaFree(flux_next_d));
}

// Validate simulation results
bool validateResults(const World& world, const int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum, flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const val_t energy = world.elements_dynamic[i].current_energy;
        const val_t flux = world.elements_dynamic[i].total_flux;
        energy_sum += energy;
        flux_sum += flux;
        energy_max = std::max(energy, energy_max);
        energy_min = std::min(energy, energy_min);
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = std::numeric_limits<val_t>::lowest();
    val_t global_energy_min = std::numeric_limits<val_t>::max();
    MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    bool valid = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }

        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }

        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }

        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    int valid_flag = valid ? 1 : 0;
    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_flag == 1;
}

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const size_t global_offset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const size_t global_idx = global_offset + i;
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            printf("ERROR: No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % device_count));

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;

    const RowPartition part = computeRowPartition(rank, size, n_elems_root);
    const int local_rows = part.rows;
    const int global_row_start = part.start_row;
    const size_t local_elems = static_cast<size_t>(local_rows) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, local_rows);
    
    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    runSimulation(world, n_iters, n_elems_root, local_rows, global_row_start, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double local_duration_ms = (end - start) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
    }
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    if (rank == 0) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    const size_t global_offset = static_cast<size_t>(global_row_start) * n_elems_root;
    const uint64_t local_hash = computeHash(world.elements_dynamic, global_offset);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> local_energy(local_elems);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local_elems; ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }

        std::vector<int> recvcounts;
        std::vector<int> displs;
        std::vector<double> energyData;
        if (rank == 0) {
            recvcounts.resize(size);
            displs.resize(size);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                const RowPartition rpart = computeRowPartition(r, size, n_elems_root);
                recvcounts[r] = rpart.rows * n_elems_root;
                displs[r] = offset;
                offset += recvcounts[r];
            }
            energyData.resize(static_cast<size_t>(n_elems));
        }

        MPI_Gatherv(local_energy.data(), static_cast<int>(local_elems), MPI_DOUBLE,
                    energyData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world, rank);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
