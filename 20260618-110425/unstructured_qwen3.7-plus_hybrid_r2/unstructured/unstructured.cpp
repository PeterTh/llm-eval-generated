#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <cuda_runtime.h>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local indexing)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: compute one simulation step for all real (non-ghost) elements
__global__ void simulation_kernel(
    const ElementStatic* __restrict__ elem_static,
    const ElementDynamic* __restrict__ elem_dynamic,
    ElementDynamic* __restrict__ elem_swap,
    const Material* __restrict__ materials,
    int first_real_idx,
    int last_real_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + first_real_idx;
    if (i > last_real_idx) return;

    const ElementStatic& es = elem_static[i];
    const ElementDynamic& ed = elem_dynamic[i];
    const Material& mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    const idx_t nc = es.num_connections;
    #pragma unroll 4
    for (idx_t j = 0; j < nc; ++j) {
        const idx_t ni = es.connected_idx[j];
        const ElementDynamic& nd = elem_dynamic[ni];
        total_flux += (nd.current_energy - ed.current_energy) *
                      mat.transfer_coeff * es.connected_flux[j] * 0.25;
    }

    ElementDynamic& ew = elem_swap[i];
    ew.current_energy = ed.current_energy + total_flux;
    ew.total_flux = ed.total_flux + fabs(total_flux);
}

// Local world state for one MPI rank
struct LocalWorld {
    int n_elems_root;         // Grid dimension
    int local_rows;           // Number of real rows this rank owns
    int global_row_start;     // First global row owned by this rank
    int total_local_elems;    // Total elements including ghost rows
    int first_real_idx;       // First real element index in local array
    int last_real_idx;        // Last real element index in local array (inclusive)

    int mpi_rank, mpi_size;
    int mpi_neighbor_up;      // Rank with row above (-1 if none)
    int mpi_neighbor_down;    // Rank with row below (-1 if none)

    // Host arrays
    std::vector<Material> materials;
    std::vector<ElementStatic> h_static;
    std::vector<ElementDynamic> h_dynamic[2];

    // Device arrays
    ElementStatic* d_static = nullptr;
    ElementDynamic* d_dynamic[2] = {nullptr, nullptr};
    Material* d_materials = nullptr;

    int current_buffer = 0;

    // Host buffers for halo exchange
    std::vector<ElementDynamic> send_up_buf;
    std::vector<ElementDynamic> send_down_buf;
    std::vector<ElementDynamic> recv_up_buf;
    std::vector<ElementDynamic> recv_down_buf;
};

// Build local portion of the 2D grid for this MPI rank
void buildLocalMesh(LocalWorld& world, int n_elems_root, int local_rows, int global_row_start,
                    int mpi_rank, int mpi_size) {
    world.n_elems_root = n_elems_root;
    world.local_rows = local_rows;
    world.global_row_start = global_row_start;
    world.mpi_rank = mpi_rank;
    world.mpi_size = mpi_size;
    world.mpi_neighbor_up = (mpi_rank > 0) ? mpi_rank - 1 : -1;
    world.mpi_neighbor_down = (mpi_rank < mpi_size - 1) ? mpi_rank + 1 : -1;

    const int n_cols = n_elems_root;
    const int total_rows = local_rows + 2;  // ghost row above + real rows + ghost row below
    world.total_local_elems = total_rows * n_cols;
    world.first_real_idx = n_cols;  // Row 1 (first real row) starts at index n_cols
    world.last_real_idx = (local_rows + 1) * n_cols - 1;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate host arrays
    world.h_static.resize(world.total_local_elems);
    world.h_dynamic[0].resize(world.total_local_elems);
    world.h_dynamic[1].resize(world.total_local_elems);

    // Initialize all static elements
    for (int i = 0; i < world.total_local_elems; ++i) {
        world.h_static[i].material_idx = DEFAULT_MAT_ID;
        world.h_static[i].num_connections = 0;
    }

    // Initialize dynamic to zero
    for (int i = 0; i < world.total_local_elems; ++i) {
        world.h_dynamic[0][i].current_energy = 0.0;
        world.h_dynamic[0][i].total_flux = 0.0;
        world.h_dynamic[1][i].current_energy = 0.0;
        world.h_dynamic[1][i].total_flux = 0.0;
    }

    // Build connectivity for real elements only (local rows 1..local_rows)
    for (int lr = 1; lr <= local_rows; ++lr) {
        const int global_x = global_row_start + (lr - 1);
        for (int y = 0; y < n_cols; ++y) {
            const int local_idx = lr * n_cols + y;
            ElementStatic& elem = world.h_static[local_idx];

            // Set material for corner elements (matching original)
            const int last = n_elems_root - 1;
            if (global_x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            else if (global_x == 0 && y == last) elem.material_idx = OUTFLOW_MAT_ID;
            else if (global_x == last && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            else if (global_x == last && y == last) elem.material_idx = INFLOW_MAT_ID;

            // Neighbors: up (x+1), down (x-1), right (y+1), left (y-1)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = global_x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_cols) {
                    // Convert global neighbor to local index
                    const int local_nx = (nx - global_row_start) + 1; // +1 for ghost offset
                    const int neighbor_local_idx = local_nx * n_cols + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_local_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Allocate halo exchange buffers
    world.send_up_buf.resize(n_cols);
    world.send_down_buf.resize(n_cols);
    world.recv_up_buf.resize(n_cols);
    world.recv_down_buf.resize(n_cols);
}

// Allocate GPU memory and copy initial data
void initGPU(LocalWorld& world) {
    const int n = world.total_local_elems;

    CUDA_CHECK(cudaMalloc(&world.d_static, n * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&world.d_dynamic[0], n * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&world.d_dynamic[1], n * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&world.d_materials, 3 * sizeof(Material)));

    CUDA_CHECK(cudaMemcpy(world.d_static, world.h_static.data(),
                          n * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_materials, world.materials.data(),
                          3 * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_dynamic[0], world.h_dynamic[0].data(),
                          n * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_dynamic[1], world.h_dynamic[1].data(),
                          n * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
}

// Exchange halo rows between MPI ranks
void exchangeHalos(LocalWorld& world) {
    const int n_cols = world.n_elems_root;
    const int local_rows = world.local_rows;
    const int buf_bytes = n_cols * sizeof(ElementDynamic);
    const int cur = world.current_buffer;

    MPI_Request requests[4];
    int n_requests = 0;

    // Copy first real row to send_up buffer (to rank-1)
    if (world.mpi_neighbor_up >= 0) {
        CUDA_CHECK(cudaMemcpy(world.send_up_buf.data(),
                              world.d_dynamic[cur] + world.first_real_idx,
                              buf_bytes, cudaMemcpyDeviceToHost));
    }

    // Copy last real row to send_down buffer (to rank+1)
    if (world.mpi_neighbor_down >= 0) {
        const int last_real_start = local_rows * n_cols;
        CUDA_CHECK(cudaMemcpy(world.send_down_buf.data(),
                              world.d_dynamic[cur] + last_real_start,
                              buf_bytes, cudaMemcpyDeviceToHost));
    }

    // Post receives first
    if (world.mpi_neighbor_up >= 0) {
        // Receive from rank-1 into ghost row 0 (their last real row)
        MPI_Irecv(world.recv_up_buf.data(), n_cols * 2, MPI_DOUBLE,
                  world.mpi_neighbor_up, 1, MPI_COMM_WORLD, &requests[n_requests++]);
    }
    if (world.mpi_neighbor_down >= 0) {
        // Receive from rank+1 into ghost row (local_rows+1) (their first real row)
        MPI_Irecv(world.recv_down_buf.data(), n_cols * 2, MPI_DOUBLE,
                  world.mpi_neighbor_down, 0, MPI_COMM_WORLD, &requests[n_requests++]);
    }

    // Post sends
    if (world.mpi_neighbor_up >= 0) {
        // Send first real row to rank-1 (they receive as their ghost row 0)
        MPI_Isend(world.send_up_buf.data(), n_cols * 2, MPI_DOUBLE,
                  world.mpi_neighbor_up, 0, MPI_COMM_WORLD, &requests[n_requests++]);
    }
    if (world.mpi_neighbor_down >= 0) {
        // Send last real row to rank+1 (they receive as their ghost row local_rows+1)
        MPI_Isend(world.send_down_buf.data(), n_cols * 2, MPI_DOUBLE,
                  world.mpi_neighbor_down, 1, MPI_COMM_WORLD, &requests[n_requests++]);
    }

    // Wait for all communications
    if (n_requests > 0) {
        MPI_Status statuses[4];
        MPI_Waitall(n_requests, requests, statuses);
    }

    // Copy received halo data to GPU
    if (world.mpi_neighbor_up >= 0) {
        CUDA_CHECK(cudaMemcpy(world.d_dynamic[cur],
                              world.recv_up_buf.data(),
                              buf_bytes, cudaMemcpyHostToDevice));
    }
    if (world.mpi_neighbor_down >= 0) {
        const int ghost_down_start = (local_rows + 1) * n_cols;
        CUDA_CHECK(cudaMemcpy(world.d_dynamic[cur] + ghost_down_start,
                              world.recv_down_buf.data(),
                              buf_bytes, cudaMemcpyHostToDevice));
    }
}

// Run simulation using CUDA + MPI + OpenMP hybrid
void runSimulation(LocalWorld& world, const int n_iters) {
    const int n_real = world.local_rows * world.n_elems_root;
    const int block_size = 256;
    const int grid_size = (n_real + block_size - 1) / block_size;

    for (int iter = 0; iter < n_iters; ++iter) {
        const int cur = world.current_buffer;
        const int nxt = 1 - cur;

        // Exchange halo data for current buffer
        exchangeHalos(world);

        // Launch CUDA kernel: compute real elements, read from cur, write to nxt
        simulation_kernel<<<grid_size, block_size>>>(
            world.d_static,
            world.d_dynamic[cur],
            world.d_dynamic[nxt],
            world.d_materials,
            world.first_real_idx,
            world.last_real_idx);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap buffers
        world.current_buffer = nxt;
    }
}

// Gather results to rank 0 for validation and output
void gatherResults(LocalWorld& world, std::vector<ElementDynamic>& global_result) {
    const int n_cols = world.n_elems_root;
    const int local_rows = world.local_rows;
    const int cur = world.current_buffer;
    const int local_real_count = local_rows * n_cols;

    // Copy real elements from GPU to host
    std::vector<ElementDynamic> h_result(local_real_count);
    CUDA_CHECK(cudaMemcpy(h_result.data(),
                          world.d_dynamic[cur] + world.first_real_idx,
                          local_real_count * sizeof(ElementDynamic),
                          cudaMemcpyDeviceToHost));

    // Gather counts for rank 0
    std::vector<int> recv_counts(world.mpi_size);
    MPI_Gather(&local_real_count, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ElementDynamic = 2 doubles, send as 2*N doubles per rank
    const int local_doubles = local_real_count * 2;
    std::vector<int> recv_counts_dbl(world.mpi_size);
    std::vector<int> displacements_dbl(world.mpi_size);

    MPI_Gather(&local_doubles, 1, MPI_INT,
               recv_counts_dbl.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world.mpi_rank == 0) {
        displacements_dbl[0] = 0;
        for (int i = 1; i < world.mpi_size; ++i) {
            displacements_dbl[i] = displacements_dbl[i-1] + recv_counts_dbl[i-1];
        }
        const int total_elems = world.n_elems_root * world.n_elems_root;
        global_result.resize(total_elems);
    }

    MPI_Gatherv(h_result.data(), local_doubles, MPI_DOUBLE,
                global_result.data(), recv_counts_dbl.data(), displacements_dbl.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Validate simulation results (runs on rank 0 with global data)
bool validateResults(const std::vector<ElementDynamic>& elements, int mpi_rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < elements.size(); ++i) {
        energy_sum += elements[i].current_energy;
        flux_sum += elements[i].total_flux;
        energy_max = std::max(elements[i].current_energy, energy_max);
        energy_min = std::min(elements[i].current_energy, energy_min);
    }

    if (mpi_rank == 0) {
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
    }
    return true;
}

// Compute hash of results (must match original serial hash)
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        uint64_t local_h = 0;
        local_h ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        local_h ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        hash ^= local_h;
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

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Assign one GPU per MPI rank (round-robin if more ranks than GPUs)
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus > 0) {
        int gpu_id = mpi_rank % n_gpus;
        CUDA_CHECK(cudaSetDevice(gpu_id));
    }

    // Set OpenMP threads
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);
    }
    int total_cores = omp_get_num_procs();
    int omp_threads = std::max(1, total_cores / local_size);
    omp_set_num_threads(omp_threads);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 broadcasts)
    if (mpi_rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validate_int = validate ? 1 : 0;
    int printResults_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_int != 0;
    printResults = printResults_int != 0;

    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs: %d\n", mpi_size, omp_threads, n_gpus);
        printf("\n");
    }

    // Compute domain decomposition (1D along rows)
    int local_rows = n_elems_root / mpi_size;
    int remainder = n_elems_root % mpi_size;
    int global_row_start;
    if (mpi_rank < remainder) {
        local_rows += 1;
        global_row_start = mpi_rank * local_rows;
    } else {
        global_row_start = remainder * (local_rows + 1) + (mpi_rank - remainder) * local_rows;
    }

    // Build local mesh
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    LocalWorld world;
    buildLocalMesh(world, n_elems_root, local_rows, global_row_start, mpi_rank, mpi_size);

    // Calculate memory usage
    const size_t static_mem = world.total_local_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = world.total_local_elems * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;

    if (mpi_rank == 0) {
        size_t global_total_mem = 0;
        MPI_Reduce(&total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Memory usage: %.2f MB total across all ranks (%.2f MB per rank avg)\n",
               global_total_mem / (1024.0 * 1024.0),
               total_mem / (1024.0 * 1024.0));
    } else {
        MPI_Reduce(&total_mem, nullptr, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (mpi_rank == 0) printf("\n");

    // Initialize GPU
    initGPU(world);

    // Run simulation
    if (mpi_rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Get max time across all ranks
    double local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double max_ms;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double duration_ms = max_ms;

    // Gather results to rank 0
    std::vector<ElementDynamic> global_result;
    gatherResults(world, global_result);

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(global_result);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_result.size());
            for (const auto& elem : global_result) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResults(global_result, mpi_rank);
            if (!valid) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    } else {
        if (validate) {
            validateResults(global_result, mpi_rank);
        }
    }

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(world.d_static));
    CUDA_CHECK(cudaFree(world.d_dynamic[0]));
    CUDA_CHECK(cudaFree(world.d_dynamic[1]));
    CUDA_CHECK(cudaFree(world.d_materials));

    MPI_Finalize();
    return 0;
}
