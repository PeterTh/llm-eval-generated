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
constexpr int MAX_MATERIALS = 4;

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Dynamic state for each element (host-side, used for I/O and validation only)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Constant memory copy of the (very small) materials table, used by the GPU kernel
__constant__ Material c_materials[MAX_MATERIALS];

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// World state: a row-decomposed (MPI) shard of the global 2D unstructured mesh,
// mirrored on the GPU (CUDA) for the hot compute loop, built with OpenMP on the host.
struct World {
    int n_elems_root = 0;   // global grid width/height
    int rank = 0;
    int num_ranks = 1;
    int local_rank = 0;     // node-local rank, used to pick a GPU
    int num_gpus = 1;

    int row_start = 0;      // first globally-owned row for this rank
    int row_count = 0;      // number of globally-owned rows for this rank
    bool has_top_ghost = false;
    bool has_bottom_ghost = false;
    int total_local_rows = 0;

    size_t n_local = 0;      // total_local_rows * n_elems_root (owned + ghost)
    size_t n_owned = 0;      // row_count * n_elems_root
    size_t owned_offset = 0; // flat index of first owned element within local arrays

    std::vector<Material> materials;

    // Host-side static connectivity, flattened Structure-of-Arrays, layout [conn][elem]
    std::vector<idx_t> h_material_idx;
    std::vector<idx_t> h_num_connections;
    std::vector<idx_t> h_connected_idx;
    std::vector<val_t> h_connected_flux;

    // Pinned host staging buffers for halo exchange (one row of width n_elems_root)
    val_t* h_send_top = nullptr;
    val_t* h_send_bottom = nullptr;
    val_t* h_recv_top = nullptr;
    val_t* h_recv_bottom = nullptr;

    // Device-side static connectivity
    idx_t* d_material_idx = nullptr;
    idx_t* d_num_connections = nullptr;
    idx_t* d_connected_idx = nullptr;
    val_t* d_connected_flux = nullptr;

    // Device-side dynamic state (ping-pong buffers)
    val_t* d_energy = nullptr;
    val_t* d_energy_swap = nullptr;
    val_t* d_flux = nullptr;
    val_t* d_flux_swap = nullptr;

    // Final host copies of owned-element results (for validation/hash/print)
    std::vector<val_t> h_energy_owned;
    std::vector<val_t> h_flux_owned;
};

// Compute how the n_elems_root rows are split across MPI ranks (balanced, contiguous blocks)
static void computePartition(int n_elems_root, int rank, int num_ranks, int& row_start, int& row_count) {
    const int base = n_elems_root / num_ranks;
    const int rem = n_elems_root % num_ranks;
    row_start = rank * base + std::min(rank, rem);
    row_count = base + (rank < rem ? 1 : 0);
}

// Build the local shard (with halo rows) of a 2D square grid unstructured mesh
void buildSquare2D(World& world, const int n_elems_root) {
    world.n_elems_root = n_elems_root;
    computePartition(n_elems_root, world.rank, world.num_ranks, world.row_start, world.row_count);

    // A rank that owns no rows (more ranks than grid rows) needs no halo at all
    world.has_top_ghost = (world.row_count > 0) && (world.row_start > 0);
    world.has_bottom_ghost = (world.row_count > 0) && (world.row_start + world.row_count < n_elems_root);
    world.total_local_rows = world.row_count + (world.has_top_ghost ? 1 : 0) + (world.has_bottom_ghost ? 1 : 0);

    const int width = n_elems_root;
    world.n_local = static_cast<size_t>(world.total_local_rows) * width;
    world.n_owned = static_cast<size_t>(world.row_count) * width;
    world.owned_offset = (world.has_top_ghost ? 1 : 0) * static_cast<size_t>(width);

    // Initialize materials (same as reference)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.h_material_idx.assign(world.n_local, DEFAULT_MAT_ID);
    world.h_num_connections.assign(world.n_local, 0);
    world.h_connected_idx.assign(world.n_local * MAX_CONNECTIONS, 0);
    world.h_connected_flux.assign(world.n_local * MAX_CONNECTIONS, 0.0);

    const int local_row_base = world.row_start - (world.has_top_ghost ? 1 : 0);
    const int last = n_elems_root - 1;

    // Build connectivity for owned rows only; ghost rows stay at defaults (unused for compute)
    #pragma omp parallel for schedule(static)
    for (int lr = (world.has_top_ghost ? 1 : 0);
         lr < (world.has_top_ghost ? 1 : 0) + world.row_count; ++lr) {
        const int gx = local_row_base + lr;
        for (int y = 0; y < width; ++y) {
            const size_t idx = static_cast<size_t>(lr) * width + y;

            idx_t mat_id = DEFAULT_MAT_ID;
            if (gx == 0 && y == 0) mat_id = INFLOW_MAT_ID;
            else if (gx == 0 && y == last) mat_id = OUTFLOW_MAT_ID;
            else if (gx == last && y == 0) mat_id = OUTFLOW_MAT_ID;
            else if (gx == last && y == last) mat_id = INFLOW_MAT_ID;
            world.h_material_idx[idx] = mat_id;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            idx_t num_connections = 0;
            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < width) {
                    const int local_lr = lr + offsets[n][0];
                    const size_t neighbor_idx = static_cast<size_t>(local_lr) * width + ny;
                    world.h_connected_idx[num_connections * world.n_local + idx] = neighbor_idx;
                    world.h_connected_flux[num_connections * world.n_local + idx] = 1.0;
                    num_connections++;
                }
            }
            world.h_num_connections[idx] = num_connections;
        }
    }

    // Pick a GPU: node-local rank modulo the number of visible devices
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world.rank, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &world.local_rank);
    MPI_Comm_free(&node_comm);

    CUDA_CHECK(cudaGetDeviceCount(&world.num_gpus));
    if (world.num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(world.local_rank % world.num_gpus));
    }

    // Upload materials table to constant memory
    std::vector<Material> mat_table(MAX_MATERIALS, Material{0.0, 0.0});
    for (size_t i = 0; i < world.materials.size(); ++i) mat_table[i] = world.materials[i];
    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, mat_table.data(), sizeof(Material) * MAX_MATERIALS));

    // Allocate and upload static connectivity
    CUDA_CHECK(cudaMalloc(&world.d_material_idx, world.n_local * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&world.d_num_connections, world.n_local * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_idx, world.n_local * MAX_CONNECTIONS * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(&world.d_connected_flux, world.n_local * MAX_CONNECTIONS * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(world.d_material_idx, world.h_material_idx.data(),
                          world.n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_num_connections, world.h_num_connections.data(),
                          world.n_local * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_connected_idx, world.h_connected_idx.data(),
                          world.n_local * MAX_CONNECTIONS * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_connected_flux, world.h_connected_flux.data(),
                          world.n_local * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));

    // Allocate dynamic state (ping-pong), zero-initialized like the reference
    CUDA_CHECK(cudaMalloc(&world.d_energy, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_energy_swap, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_flux, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&world.d_flux_swap, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_energy, 0, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_energy_swap, 0, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_flux, 0, world.n_local * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(world.d_flux_swap, 0, world.n_local * sizeof(val_t)));

    // Pinned host buffers for fast halo staging
    CUDA_CHECK(cudaMallocHost(&world.h_send_top, width * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_send_bottom, width * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_top, width * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&world.h_recv_bottom, width * sizeof(val_t)));

    world.h_energy_owned.resize(world.n_owned);
    world.h_flux_owned.resize(world.n_owned);
}

// GPU kernel: compute one energy-transfer update step for all owned elements
__global__ void updateElementsKernel(const idx_t* __restrict__ material_idx,
                                     const idx_t* __restrict__ num_connections,
                                     const idx_t* __restrict__ connected_idx,
                                     const val_t* __restrict__ connected_flux,
                                     const val_t* __restrict__ energy_in,
                                     val_t* __restrict__ energy_out,
                                     const val_t* __restrict__ flux_in,
                                     val_t* __restrict__ flux_out,
                                     size_t n_local, size_t owned_offset, size_t n_owned) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_owned) return;
    const size_t i = owned_offset + tid;

    const Material mat = c_materials[material_idx[i]];
    const val_t this_energy = energy_in[i];
    val_t total_flux = mat.external_flow;

    const idx_t nconn = num_connections[i];
    for (idx_t j = 0; j < nconn; ++j) {
        const idx_t nb = connected_idx[j * n_local + i];
        const val_t flux_coef = connected_flux[j * n_local + i];
        const val_t nb_energy = energy_in[nb];
        total_flux += (nb_energy - this_energy) * mat.transfer_coeff * flux_coef * 0.25;
    }

    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Exchange the single-row halo of "current energy" with row-neighbor MPI ranks
static void haloExchange(World& world) {
    const int width = world.n_elems_root;
    const int top_neighbor = world.rank - 1;
    const int bottom_neighbor = world.rank + 1;

    MPI_Request reqs[4];
    int nreqs = 0;

    if (world.has_top_ghost) {
        MPI_Irecv(world.h_recv_top, width, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (world.has_bottom_ghost) {
        MPI_Irecv(world.h_recv_bottom, width, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    // Stage the boundary rows to send: first/last owned rows, contiguous in device memory
    if (world.has_top_ghost) {
        CUDA_CHECK(cudaMemcpy(world.h_send_top, world.d_energy + world.owned_offset,
                              width * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Isend(world.h_send_top, width, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
    }
    if (world.has_bottom_ghost) {
        const size_t last_owned_row_offset = world.owned_offset + static_cast<size_t>(world.row_count - 1) * width;
        CUDA_CHECK(cudaMemcpy(world.h_send_bottom, world.d_energy + last_owned_row_offset,
                              width * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Isend(world.h_send_bottom, width, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    if (world.has_top_ghost) {
        CUDA_CHECK(cudaMemcpy(world.d_energy, world.h_recv_top, width * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (world.has_bottom_ghost) {
        const size_t bottom_ghost_offset = world.owned_offset + static_cast<size_t>(world.row_count) * width;
        CUDA_CHECK(cudaMemcpy(world.d_energy + bottom_ghost_offset, world.h_recv_bottom,
                              width * sizeof(val_t), cudaMemcpyHostToDevice));
    }
}

// Run simulation for n_iters iterations: MPI halo exchange + CUDA compute per step
void runSimulation(World& world, const int n_iters) {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((world.n_owned + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int iter = 0; iter < n_iters; ++iter) {
        haloExchange(world);

        if (world.n_owned > 0) {
            updateElementsKernel<<<grid_size, BLOCK_SIZE>>>(
                world.d_material_idx, world.d_num_connections, world.d_connected_idx, world.d_connected_flux,
                world.d_energy, world.d_energy_swap, world.d_flux, world.d_flux_swap,
                world.n_local, world.owned_offset, world.n_owned);
            CUDA_CHECK(cudaGetLastError());
        }

        std::swap(world.d_energy, world.d_energy_swap);
        std::swap(world.d_flux, world.d_flux_swap);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy owned-element results back to the host for validation / hashing / output
    CUDA_CHECK(cudaMemcpy(world.h_energy_owned.data(), world.d_energy + world.owned_offset,
                          world.n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.h_flux_owned.data(), world.d_flux + world.owned_offset,
                          world.n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
}

// Validate simulation results (reduced across all MPI ranks)
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
                             reduction(max:local_energy_max) reduction(min:local_energy_min) schedule(static)
    for (size_t i = 0; i < world.n_owned; ++i) {
        local_energy_sum += world.h_energy_owned[i];
        local_flux_sum += world.h_flux_owned[i];
        local_energy_max = std::max(world.h_energy_owned[i], local_energy_max);
        local_energy_min = std::min(world.h_energy_owned[i], local_energy_min);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    bool valid = true;

    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }

        if (valid && std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (valid && !std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }

        if (valid && (!std::isfinite(energy_max) || !std::isfinite(energy_min))) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }

        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return valid;
}

// Compute a simple hash of the results for verification (order-independent XOR combine,
// so per-rank partial hashes computed with global element indices reduce to the same value)
uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;

    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (size_t li = 0; li < world.n_owned; ++li) {
        const size_t global_i = static_cast<size_t>(world.row_start) * world.n_elems_root + li;
        const val_t energy = world.h_energy_owned[li];
        const val_t flux = world.h_flux_owned[li];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t hash = 0;
    MPI_Allreduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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

    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.num_ranks);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            if (world.rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world.num_ranks);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build the (distributed) unstructured mesh and upload it to the GPU
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage (aggregate across all ranks/GPUs)
    const size_t local_static_mem = world.n_local * (2 * sizeof(idx_t) + MAX_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t)));
    const size_t local_dynamic_mem = world.n_local * sizeof(val_t) * 4;
    size_t static_mem = 0, dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world.rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (reduced across ranks)
    const uint64_t hash = computeHash(world);
    if (world.rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather full global array on rank 0)
    if (printResults) {
        const int width = n_elems_root;
        std::vector<int> counts(world.num_ranks), displs(world.num_ranks);
        for (int r = 0; r < world.num_ranks; ++r) {
            int rs, rc;
            computePartition(n_elems_root, r, world.num_ranks, rs, rc);
            counts[r] = rc * width;
            displs[r] = rs * width;
        }

        std::vector<double> energyData;
        if (world.rank == 0) energyData.resize(static_cast<size_t>(n_elems));

        MPI_Gatherv(world.h_energy_owned.data(), static_cast<int>(world.n_owned), MPI_DOUBLE,
                    world.rank == 0 ? energyData.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (world.rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    bool valid = true;
    if (validate) {
        valid = validateResults(world);
    }

    // Free CUDA resources
    cudaFree(world.d_material_idx);
    cudaFree(world.d_num_connections);
    cudaFree(world.d_connected_idx);
    cudaFree(world.d_connected_flux);
    cudaFree(world.d_energy);
    cudaFree(world.d_energy_swap);
    cudaFree(world.d_flux);
    cudaFree(world.d_flux_swap);
    cudaFreeHost(world.h_send_top);
    cudaFreeHost(world.h_send_bottom);
    cudaFreeHost(world.h_recv_top);
    cudaFreeHost(world.h_recv_bottom);

    MPI_Finalize();

    if (validate && !valid) {
        return 1;
    }
    return 0;
}
