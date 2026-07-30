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

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// CUDA kernel for simulation step
// Grid is decomposed by rows. Each rank owns a contiguous block of rows.
// The local array includes ghost rows above (index 0) and below (index local_n_rows+1).
// Owned rows are at indices 1..local_n_rows.
__global__ void simulation_step_kernel(
    const double* __restrict__ energy,
    double* __restrict__ energy_new,
    const double* __restrict__ flux,
    double* __restrict__ flux_new,
    const double* __restrict__ transfer_coeff,
    const double* __restrict__ external_flow,
    const int n_elems_root,
    const int local_n_rows,
    const int has_ghost_above,
    const int has_ghost_below)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= n_elems_root || local_row >= local_n_rows) return;

    // Index in the local array (with ghost rows)
    // Ghost row above is at row 0, owned rows start at row 1
    const int idx = (local_row + 1) * n_elems_root + col;

    const double tc = transfer_coeff[idx];
    const double ef = external_flow[idx];
    const double e = energy[idx];

    double total_flux = ef;

    // Up neighbor (row - 1)
    if (local_row > 0 || has_ghost_above) {
        total_flux += (energy[idx - n_elems_root] - e) * tc * 0.25;
    }

    // Down neighbor (row + 1)
    if (local_row < local_n_rows - 1 || has_ghost_below) {
        total_flux += (energy[idx + n_elems_root] - e) * tc * 0.25;
    }

    // Left neighbor (col - 1)
    if (col > 0) {
        total_flux += (energy[idx - 1] - e) * tc * 0.25;
    }

    // Right neighbor (col + 1)
    if (col < n_elems_root - 1) {
        total_flux += (energy[idx + 1] - e) * tc * 0.25;
    }

    energy_new[idx] = e + total_flux;
    flux_new[idx] = flux[idx] + fabs(total_flux);
}

// Build a 2D square grid as an unstructured mesh (original serial version, kept for reference)
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

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

// Run simulation for n_iters iterations (serial reference version)
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();

    for (int iter = 0; iter < n_iters; ++iter) {
        for (size_t i = 0; i < n_elems; ++i) {
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize OpenMP threading
    int omp_num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        omp_num_threads = omp_get_num_threads();
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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

    // Check that we have enough rows for all ranks
    if (nprocs > n_elems_root) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid dimension (%d)\n", nprocs, n_elems_root);
        }
        MPI_Finalize();
        return 1;
    }

    // Domain decomposition: split rows among ranks
    int base_rows = n_elems_root / nprocs;
    int remainder = n_elems_root % nprocs;
    int local_n_rows, row_start;
    if (rank < remainder) {
        local_n_rows = base_rows + 1;
        row_start = rank * (base_rows + 1);
    } else {
        local_n_rows = base_rows;
        row_start = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }

    const int has_ghost_above = (rank > 0) ? 1 : 0;
    const int has_ghost_below = (rank < nprocs - 1) ? 1 : 0;

    // Print header (rank 0 only)
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA: enabled\n", nprocs, omp_num_threads);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build local mesh data (SoA layout for GPU efficiency)
    // Local array includes ghost rows: row 0 = ghost above, rows 1..local_n_rows = owned,
    // row local_n_rows+1 = ghost below
    const int local_total = (local_n_rows + 2) * n_elems_root;
    const int local_owned = local_n_rows * n_elems_root;

    std::vector<double> h_energy(local_total, 0.0);
    std::vector<double> h_flux(local_total, 0.0);
    std::vector<double> h_transfer_coeff(local_total, 0.0);
    std::vector<double> h_external_flow(local_total, 0.0);

    // Initialize material properties for owned rows using OpenMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int local_row = 0; local_row < local_n_rows; ++local_row) {
        for (int col = 0; col < n_elems_root; ++col) {
            const int global_row = row_start + local_row;
            const int idx = (local_row + 1) * n_elems_root + col;

            h_transfer_coeff[idx] = 0.8;
            h_external_flow[idx] = 0.0;

            // Corner elements: same assignment as original buildSquare2D
            // (0,0) and (last,last) are INFLOW; (0,last) and (last,0) are OUTFLOW
            if ((global_row == 0 && col == 0) || (global_row == n_elems_root - 1 && col == n_elems_root - 1)) {
                h_external_flow[idx] = 0.5;   // Inflow material
            }
            if ((global_row == 0 && col == n_elems_root - 1) || (global_row == n_elems_root - 1 && col == 0)) {
                h_external_flow[idx] = -0.5;  // Outflow material
            }
        }
    }

    // Calculate memory usage (aggregate across ranks)
    size_t local_mem = (size_t)local_total * sizeof(double) * 4 +
                       (size_t)local_owned * (sizeof(double) * 2 + sizeof(int));
    size_t total_mem_global = 0;
    MPI_Reduce(&local_mem, &total_mem_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Building unstructured mesh...\n");
        printf("Memory usage: %.2f MB\n", total_mem_global / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Set CUDA device (use local rank for multi-GPU nodes)
    int local_rank = 0;
    {
        // Determine local rank via MPI_COMM_WORLD split by node
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));
    }

    // Allocate GPU memory
    double *d_energy, *d_energy_new, *d_flux, *d_flux_new;
    double *d_transfer_coeff, *d_external_flow;

    CUDA_CHECK(cudaMalloc(&d_energy, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy_new, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux_new, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_transfer_coeff, local_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_external_flow, local_total * sizeof(double)));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), local_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flux, h_flux.data(), local_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transfer_coeff, h_transfer_coeff.data(), local_total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_external_flow, h_external_flow.data(), local_total * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate pinned host buffers for halo exchange (faster transfers)
    double *h_send_top, *h_send_bottom, *h_recv_top, *h_recv_bottom;
    CUDA_CHECK(cudaMallocHost(&h_send_top, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, n_elems_root * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, n_elems_root * sizeof(double)));

    // CUDA kernel launch configuration
    dim3 block(32, 4);  // 128 threads per block
    dim3 grid((n_elems_root + block.x - 1) / block.x,
              (local_n_rows + block.y - 1) / block.y);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Simulation loop
    for (int iter = 0; iter < n_iters; ++iter) {
        // === Halo exchange ===
        // Copy boundary rows from GPU to pinned host buffers
        // Top owned row is at index 1 * n_elems_root
        CUDA_CHECK(cudaMemcpy(h_send_top, d_energy + 1 * n_elems_root,
                              n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
        // Bottom owned row is at index local_n_rows * n_elems_root
        CUDA_CHECK(cudaMemcpy(h_send_bottom, d_energy + local_n_rows * n_elems_root,
                              n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));

        // MPI exchange with rank-1 (upward): send top row, receive ghost above
        if (rank > 0) {
            MPI_Sendrecv(h_send_top, n_elems_root, MPI_DOUBLE, rank - 1, 0,
                         h_recv_top, n_elems_root, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy received ghost above to GPU (row 0)
            CUDA_CHECK(cudaMemcpy(d_energy + 0 * n_elems_root, h_recv_top,
                                  n_elems_root * sizeof(double), cudaMemcpyHostToDevice));
        }

        // MPI exchange with rank+1 (downward): send bottom row, receive ghost below
        if (rank < nprocs - 1) {
            MPI_Sendrecv(h_send_bottom, n_elems_root, MPI_DOUBLE, rank + 1, 1,
                         h_recv_bottom, n_elems_root, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Copy received ghost below to GPU (row local_n_rows+1)
            CUDA_CHECK(cudaMemcpy(d_energy + (local_n_rows + 1) * n_elems_root, h_recv_bottom,
                                  n_elems_root * sizeof(double), cudaMemcpyHostToDevice));
        }

        // === Launch CUDA kernel ===
        simulation_step_kernel<<<grid, block>>>(
            d_energy, d_energy_new, d_flux, d_flux_new,
            d_transfer_coeff, d_external_flow,
            n_elems_root, local_n_rows, has_ghost_above, has_ghost_below);

        // Swap pointers (O(1) operation)
        std::swap(d_energy, d_energy_new);
        std::swap(d_flux, d_flux_new);
    }

    // Synchronize GPU and all ranks before stopping timer
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;

    // === Copy results back to host ===
    std::vector<double> h_local_energy(local_owned);
    std::vector<double> h_local_flux(local_owned);

    // Copy owned rows from GPU to host (skip ghost rows)
    for (int local_row = 0; local_row < local_n_rows; ++local_row) {
        CUDA_CHECK(cudaMemcpy(h_local_energy.data() + local_row * n_elems_root,
                              d_energy + (local_row + 1) * n_elems_root,
                              n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_local_flux.data() + local_row * n_elems_root,
                              d_flux + (local_row + 1) * n_elems_root,
                              n_elems_root * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // === Gather results to rank 0 ===
    std::vector<int> all_local_n_rows(nprocs);
    MPI_Allgather(&local_n_rows, 1, MPI_INT, all_local_n_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> recv_counts(nprocs);
    std::vector<int> displacements(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        recv_counts[r] = all_local_n_rows[r] * n_elems_root;
        displacements[r] = (r > 0) ? displacements[r - 1] + recv_counts[r - 1] : 0;
    }

    std::vector<double> h_result_energy, h_result_flux;
    if (rank == 0) {
        h_result_energy.resize(n_elems);
        h_result_flux.resize(n_elems);
    }

    MPI_Gatherv(h_local_energy.data(), local_owned, MPI_DOUBLE,
                h_result_energy.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_local_flux.data(), local_owned, MPI_DOUBLE,
                h_result_flux.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // === Output (rank 0 only) ===
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration_ms);

        // Performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash using OpenMP parallel reduction (XOR is associative/commutative)
        uint64_t result_hash = 0;
        #pragma omp parallel for reduction(^:result_hash) schedule(static)
        for (int i = 0; i < n_elems; ++i) {
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&h_result_energy[i]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&h_result_flux[i]);
            result_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            result_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }

        printf("  Result hash: %016lX\n", result_hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(h_result_energy, "ElementEnergy");
        }

        // Validation using OpenMP parallel reductions
        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min) schedule(static)
            for (int i = 0; i < n_elems; ++i) {
                energy_sum += h_result_energy[i];
                flux_sum += h_result_flux[i];
                energy_max = std::max(energy_max, h_result_energy[i]);
                energy_min = std::min(energy_min, h_result_energy[i]);
            }

            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            constexpr val_t energy_epsilon = 1e-8;
            bool valid = true;

            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            if (valid) {
                printf("  Validation: PASSED\n");
            }

            if (!valid) {
                // Cleanup before exit
                cudaFree(d_energy); cudaFree(d_energy_new);
                cudaFree(d_flux); cudaFree(d_flux_new);
                cudaFree(d_transfer_coeff); cudaFree(d_external_flow);
                cudaFreeHost(h_send_top); cudaFreeHost(h_send_bottom);
                cudaFreeHost(h_recv_top); cudaFreeHost(h_recv_bottom);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // === Cleanup ===
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_energy_new));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_flux_new));
    CUDA_CHECK(cudaFree(d_transfer_coeff));
    CUDA_CHECK(cudaFree(d_external_flow));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));

    MPI_Finalize();
    return 0;
}
