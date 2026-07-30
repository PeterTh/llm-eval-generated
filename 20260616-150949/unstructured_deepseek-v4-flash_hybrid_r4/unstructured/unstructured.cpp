#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// =========================================================================
// Types to represent unstructured mesh elements
// =========================================================================
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices in combined local+ghost array
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t),
              "ElementDynamic must be tightly packed for MPI exchange");

// Distributed world state (each MPI rank owns a sub-domain)
struct World {
    // MPI info
    int rank;
    int n_ranks;

    // Global dimensions
    int global_n_elems_root;

    // Local partition (row-based: each rank owns contiguous rows)
    int local_start_x;
    int local_end_x;
    int n_local;
    int n_ghost_above;
    int n_ghost_below;
    int n_total;  // n_ghost_above + n_local + n_ghost_below

    // Materials (replicated on all ranks)
    std::vector<Material> materials;

    // Static connectivity: n_local entries (only local elements)
    std::vector<ElementStatic> elements_static;

    // Dynamic state: n_total entries [ghost_above | local | ghost_below]
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;

    // GPU device pointers
    ElementStatic*  d_static;
    ElementDynamic* d_dynamic;
    ElementDynamic* d_dynamic_swap;
    Material*       d_materials;

    // Pinned host buffers for MPI ghost exchange
    char* h_send_buf;
    char* h_recv_buf;
    char* d_send_buf;
    char* d_recv_buf;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID  = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Number of bytes to exchange for one boundary row
#define ROW_BYTES(g, stride)  ((size_t)(g) * (stride))

// =========================================================================
// CUDA error checking macro
// =========================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = call;                                               \
        if (err_ != cudaSuccess) {                                             \
            int mpi_rank_;                                                     \
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);                         \
            fprintf(stderr, "[MPI rank %d] CUDA error at %s:%d: %s\n",         \
                    mpi_rank_, __FILE__, __LINE__, cudaGetErrorString(err_));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// =========================================================================
// CUDA kernel: compute all local element updates in one launch
// =========================================================================
// static_data   : n_local entries (indexed by local element id)
// dyn_in        : n_total [ghost_above | local | ghost_below]
// dyn_out       : same layout, written for local elements only
// materials     : array of Material structs
// ghost_above   : number of ghost elements before local data
// n_local       : number of local elements
// =========================================================================
__global__ void computeStepKernel(
    const ElementStatic* __restrict__ static_data,
    const ElementDynamic* __restrict__ dyn_in,
    ElementDynamic* __restrict__ dyn_out,
    const Material* __restrict__ materials,
    int ghost_above,
    int n_local)
{
    int lidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lidx >= n_local) return;

    int combined_self = ghost_above + lidx;

    const ElementStatic& es = static_data[lidx];
    const ElementDynamic& ed = dyn_in[combined_self];
    const Material& mat = materials[es.material_idx];

    val_t flux = mat.external_flow;
    idx_t nc = es.num_connections;

    #pragma unroll
    for (idx_t j = 0; j < nc; ++j) {
        const ElementDynamic& nd = dyn_in[es.connected_idx[j]];
        flux += (nd.current_energy - ed.current_energy) *
                mat.transfer_coeff * es.connected_flux[j] * static_cast<val_t>(0.25);
    }

    dyn_out[combined_self].current_energy = ed.current_energy + flux;
    dyn_out[combined_self].total_flux = ed.total_flux + fabs(flux);
}

// =========================================================================
// Build distributed grid: each rank creates its local portion
// =========================================================================
void buildDistributedGrid(World& w, int n_elems_root_global)
{
    int rank = w.rank;
    int n_ranks = w.n_ranks;

    w.global_n_elems_root = n_elems_root_global;

    // ----- 1D row-based partition -----
    int rows_per = n_elems_root_global / n_ranks;
    int rem     = n_elems_root_global % n_ranks;

    w.local_start_x = rank * rows_per + (rank < rem ? rank : rem);
    w.local_end_x   = w.local_start_x + rows_per + (rank < rem ? 1 : 0);

    int n_local_rows   = w.local_end_x - w.local_start_x;
    w.n_local          = n_local_rows * n_elems_root_global;
    w.n_ghost_above    = (rank > 0)           ? n_elems_root_global : 0;
    w.n_ghost_below    = (rank < n_ranks - 1) ? n_elems_root_global : 0;
    w.n_total          = w.n_ghost_above + w.n_local + w.n_ghost_below;

    // ----- Materials (replicated) -----
    w.materials.clear();
    w.materials.reserve(3);
    w.materials.emplace_back(Material{0.8,  0.0});
    w.materials.emplace_back(Material{0.8,  0.5});
    w.materials.emplace_back(Material{0.8, -0.5});

    // ----- Allocate host data -----
    w.elements_static.resize(w.n_local);
    w.elements_dynamic.resize(w.n_total);
    w.elements_dynamic_swap.resize(w.n_total);

    // Zero initial dynamic state
    #pragma omp parallel for
    for (int i = 0; i < w.n_total; ++i) {
        w.elements_dynamic[i]     = ElementDynamic{0.0, 0.0};
        w.elements_dynamic_swap[i]= ElementDynamic{0.0, 0.0};
    }

    const int g = n_elems_root_global;

    // ----- Build local static connectivity (OpenMP parallel) -----
    #pragma omp parallel for
    for (int lidx = 0; lidx < w.n_local; ++lidx) {
        int gx = w.local_start_x + lidx / g;
        int gy = lidx % g;

        ElementStatic& es = w.elements_static[lidx];
        es.material_idx    = DEFAULT_MAT_ID;
        es.num_connections = 0;

        // Neighbours: (dx, dy) = (1,0), (-1,0), (0,1), (0,-1)
        static const int dx[4] = { 1, -1,  0,  0};
        static const int dy[4] = { 0,  0,  1, -1};

        for (int n = 0; n < 4; ++n) {
            int nx = gx + dx[n];
            int ny = gy + dy[n];
            if (nx < 0 || nx >= g || ny < 0 || ny >= g) continue;

            idx_t cidx;
            if (nx >= w.local_start_x && nx < w.local_end_x) {
                // Local neighbour
                cidx = w.n_ghost_above + (nx - w.local_start_x) * g + ny;
            } else if (nx == w.local_start_x - 1) {
                // Ghost above
                cidx = static_cast<idx_t>(ny);
            } else {
                // Ghost below (nx == w.local_end_x)
                cidx = w.n_ghost_above + w.n_local + static_cast<idx_t>(ny);
            }
            es.connected_idx[es.num_connections] = cidx;
            es.connected_flux[es.num_connections] = 1.0;
            es.num_connections++;
        }
    }

    // ----- Set corner materials -----
    const int last = g - 1;
    // (0, 0)        -- INFLOW
    if (w.local_start_x <= 0 && w.local_end_x > 0) {
        w.elements_static[0].material_idx = INFLOW_MAT_ID;
    }
    // (0, last)     -- OUTFLOW
    if (w.local_start_x <= 0 && w.local_end_x > 0) {
        w.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    }
    // (last, 0)     -- OUTFLOW
    if (w.local_start_x <= last && w.local_end_x > last) {
        int local_row = last - w.local_start_x;
        w.elements_static[local_row * g + 0].material_idx = OUTFLOW_MAT_ID;
    }
    // (last, last)  -- INFLOW
    if (w.local_start_x <= last && w.local_end_x > last) {
        int local_row = last - w.local_start_x;
        w.elements_static[local_row * g + last].material_idx = INFLOW_MAT_ID;
    }

    // ----- Allocate GPU memory and pinned host buffers -----
    // Distribute ranks round-robin among available GPUs
    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % ngpus));
    }

    size_t sz_static  = w.n_local * sizeof(ElementStatic);
    size_t sz_dyn     = w.n_total * sizeof(ElementDynamic);
    size_t sz_mat     = w.materials.size() * sizeof(Material);

    CUDA_CHECK(cudaMalloc(&w.d_static,  sz_static));
    CUDA_CHECK(cudaMalloc(&w.d_dynamic, sz_dyn));
    CUDA_CHECK(cudaMalloc(&w.d_dynamic_swap, sz_dyn));
    CUDA_CHECK(cudaMalloc(&w.d_materials, sz_mat));

    // Copy static data and materials to GPU (read-only)
    CUDA_CHECK(cudaMemcpy(w.d_static,   w.elements_static.data(), sz_static,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.d_materials, w.materials.data(), sz_mat,
                          cudaMemcpyHostToDevice));

    // Initialise dynamic buffers on device (all zeros)
    CUDA_CHECK(cudaMemset(w.d_dynamic,      0, sz_dyn));
    CUDA_CHECK(cudaMemset(w.d_dynamic_swap, 0, sz_dyn));

    // Pinned host buffers + device-side exchange buffers for ghost transfer
    size_t row_sz = static_cast<size_t>(g) * sizeof(ElementDynamic);

    CUDA_CHECK(cudaMallocHost(&w.h_send_buf, row_sz));
    CUDA_CHECK(cudaMallocHost(&w.h_recv_buf, row_sz));
    CUDA_CHECK(cudaMalloc(&w.d_send_buf, row_sz));
    CUDA_CHECK(cudaMalloc(&w.d_recv_buf, row_sz));
}


// =========================================================================
// Ghost exchange: update ghost regions with neighbour data via MPI
// =========================================================================
void exchangeGhosts(World& w)
{
    int rank = w.rank;
    int n_ranks = w.n_ranks;
    int g = w.global_n_elems_root;
    size_t row_bytes = static_cast<size_t>(g) * sizeof(ElementDynamic);
    int tag = 0;

    // Combined buffer layout: [ghost_above | local | ghost_below]

    // --- Exchange with rank above (send first local row up, recv ghost above) ---
    if (rank > 0) {
        {
            // Copy first local row from d_dynamic into d_send_buf
            char* dyn_bytes = reinterpret_cast<char*>(w.d_dynamic);
            size_t first_off = w.n_ghost_above * sizeof(ElementDynamic);
            CUDA_CHECK(cudaMemcpy(w.d_send_buf, dyn_bytes + first_off,
                                  row_bytes, cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaMemcpy(w.h_send_buf, w.d_send_buf, row_bytes,
                              cudaMemcpyDeviceToHost));

        MPI_Sendrecv(w.h_send_buf, static_cast<int>(row_bytes), MPI_BYTE,
                     rank - 1, tag,
                     w.h_recv_buf, static_cast<int>(row_bytes), MPI_BYTE,
                     rank - 1, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // h_recv_buf -> d_recv_buf -> ghost_above region in d_dynamic
        CUDA_CHECK(cudaMemcpy(w.d_recv_buf, w.h_recv_buf, row_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.d_dynamic, w.d_recv_buf, row_bytes,
                              cudaMemcpyDeviceToDevice));
    }

    // --- Exchange with rank below (send last local row down, recv ghost below) ---
    if (rank < n_ranks - 1) {
        {
            // Copy last local row from d_dynamic into d_send_buf
            char* dyn_bytes = reinterpret_cast<char*>(w.d_dynamic);
            size_t last_off = (w.n_ghost_above + w.n_local - g) * sizeof(ElementDynamic);
            CUDA_CHECK(cudaMemcpy(w.d_send_buf, dyn_bytes + last_off,
                                  row_bytes, cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaMemcpy(w.h_send_buf, w.d_send_buf, row_bytes,
                              cudaMemcpyDeviceToHost));

        MPI_Sendrecv(w.h_send_buf, static_cast<int>(row_bytes), MPI_BYTE,
                     rank + 1, tag,
                     w.h_recv_buf, static_cast<int>(row_bytes), MPI_BYTE,
                     rank + 1, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        CUDA_CHECK(cudaMemcpy(w.d_recv_buf, w.h_recv_buf, row_bytes,
                              cudaMemcpyHostToDevice));
        {
            char* dyn_bytes = reinterpret_cast<char*>(w.d_dynamic);
            size_t ghost_off = (w.n_ghost_above + w.n_local) * sizeof(ElementDynamic);
            CUDA_CHECK(cudaMemcpy(dyn_bytes + ghost_off,
                                  w.d_recv_buf, row_bytes, cudaMemcpyDeviceToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// =========================================================================
// Run simulation using hybrid MPI + CUDA + OpenMP
// =========================================================================
void runSimulationHybrid(World& w, const int n_iters)
{
    const int threads_per_block = 256;
    int blocks = (w.n_local + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < n_iters; ++iter) {
        // 1. Exchange ghost cells via MPI (updates ghost regions on GPU)
        exchangeGhosts(w);

        // 2. Launch CUDA kernel for local elements
        computeStepKernel<<<blocks, threads_per_block>>>(
            w.d_static, w.d_dynamic, w.d_dynamic_swap,
            w.d_materials, w.n_ghost_above, w.n_local);

        CUDA_CHECK(cudaGetLastError());

        // 3. Swap GPU dynamic buffers
        ElementDynamic* tmp = w.d_dynamic;
        w.d_dynamic = w.d_dynamic_swap;
        w.d_dynamic_swap = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// =========================================================================
// Copy GPU dynamic data back to host
// =========================================================================
void downloadDynamicData(World& w)
{
    size_t sz = w.n_total * sizeof(ElementDynamic);
    CUDA_CHECK(cudaMemcpy(w.elements_dynamic.data(), w.d_dynamic,
                          sz, cudaMemcpyDeviceToHost));
}

// =========================================================================
// Global reduction for validation (MPI)
// =========================================================================
bool validateResultsDistributed(const World& w)
{
    int rank = w.rank;

    // Local reductions (OpenMP parallel)
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
        reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (int i = 0; i < w.n_local; ++i) {
        const auto& e = w.elements_dynamic[w.n_ghost_above + i];
        local_energy_sum += e.current_energy;
        local_flux_sum   += e.total_flux;
        if (e.current_energy > local_energy_max) local_energy_max = e.current_energy;
        if (e.current_energy < local_energy_min) local_energy_min = e.current_energy;
    }

    // Global MPI reductions
    val_t global_energy_sum = 0.0;
    val_t global_flux_sum   = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;

    MPI_Allreduce(&local_energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum,   &global_flux_sum,   1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }
        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            return false;
        }
        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            return false;
        }

        printf("  Validation: PASSED\n");
    }

    // Broadcast validation result to all ranks
    int passed_int = 1;
    if (rank == 0) {
        passed_int = (std::isfinite(global_energy_sum) &&
                      std::isfinite(global_flux_sum) &&
                      std::isfinite(global_energy_max) &&
                      std::isfinite(global_energy_min)) ? 1 : 0;
    }
    MPI_Bcast(&passed_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return passed_int != 0;
}

// =========================================================================
// Distributed hash: each rank hashes its local elements; XOR across ranks
// =========================================================================
uint64_t computeHashDistributed(const World& w)
{
    uint64_t local_hash = 0;

    #pragma omp parallel for reduction(^:local_hash)
    for (int i = 0; i < w.n_local; ++i) {
        const auto& e = w.elements_dynamic[w.n_ghost_above + i];
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &e.current_energy, sizeof(e_bits));
        memcpy(&f_bits, &e.total_flux,      sizeof(f_bits));

        int64_t global_index =
            static_cast<int64_t>((w.local_start_x + i / w.global_n_elems_root)) *
                w.global_n_elems_root + (i % w.global_n_elems_root);

        local_hash ^= (e_bits + static_cast<uint64_t>(global_index)) *
                      0x9e3779b97f4a7c15ULL;
        local_hash ^= (f_bits + static_cast<uint64_t>(global_index)) *
                      0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);
    return global_hash;
}

// =========================================================================
// Clean up GPU resources
// =========================================================================
void cleanupGPU(World& w)
{
    CUDA_CHECK(cudaFree(w.d_static));
    CUDA_CHECK(cudaFree(w.d_dynamic));
    CUDA_CHECK(cudaFree(w.d_dynamic_swap));
    CUDA_CHECK(cudaFree(w.d_materials));
    CUDA_CHECK(cudaFree(w.d_send_buf));
    CUDA_CHECK(cudaFree(w.d_recv_buf));
    CUDA_CHECK(cudaFreeHost(w.h_send_buf));
    CUDA_CHECK(cudaFreeHost(w.h_recv_buf));
}

// =========================================================================
// Usage
// =========================================================================
void printUsage(const char* progName)
{
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv)
{
    // ----- MPI initialisation -----
    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // ----- Parse arguments (all ranks parse) -----
    int n_elems_root = 512;
    int n_iters      = 10;
    bool validate    = false;
    bool printResults = false;

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

    const int n_elems_global = n_elems_root * n_elems_root;

    // ----- Print banner (rank 0 only) -----
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n",
               n_elems_root, n_elems_root, n_elems_global);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // GPU configuration
    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % ngpus));
    }

    if (rank == 0) {
        printf("MPI ranks: %d\n", n_ranks);
        printf("GPUs per node: %d\n", ngpus);
        printf("\n");
    }

    // ----- Build the distributed unstructured mesh -----
    if (rank == 0) printf("Building unstructured mesh...\n");

    World w;
    w.rank    = rank;
    w.n_ranks = n_ranks;

    buildDistributedGrid(w, n_elems_root);

    // ----- Memory statistics (rank 0 prints) -----
    if (rank == 0) {
        const size_t static_mem  = w.n_local * sizeof(ElementStatic);
        const size_t dynamic_mem = w.n_total * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        // Scale up to approximate total across all ranks
        printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // ----- Run simulation (hybrid MPI + CUDA + OpenMP) -----
    if (rank == 0) printf("Running simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    runSimulationHybrid(w, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double elapsed_local = t_end - t_start;  // seconds
    double elapsed_global = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed_global, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // ----- Download results from GPU -----
    downloadDynamicData(w);

    // ----- Compute performance metrics -----
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double total_ms      = elapsed_global * 1000.0;
    const double time_per_iter = total_ms / static_cast<double>(n_measured_iters);

    // Global throughput
    const double elems_per_sec = (static_cast<double>(n_measured_iters) *
                                  n_elems_global) / elapsed_global;
    const double giga_elems_per_sec = elems_per_sec / 1.0e9;
    const double gflops = giga_elems_per_sec * 22.0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", total_ms);
        printf("  (wall-clock, max across MPI ranks)\n");
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // ----- Result hash (global, computed with MPI_BXOR) -----
    const uint64_t global_hash = computeHashDistributed(w);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // ----- Print results for external validation (rank 0 only) -----
    if (printResults && rank == 0) {
        std::vector<double> energyData;
        energyData.reserve(w.n_local);
        for (int i = 0; i < w.n_local; ++i) {
            energyData.push_back(
                w.elements_dynamic[w.n_ghost_above + i].current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // ----- Validation (distributed) -----
    bool valid = true;
    if (validate) {
        valid = validateResultsDistributed(w);
    }

    // ----- Cleanup -----
    cleanupGPU(w);
    MPI_Finalize();

    return valid ? 0 : 1;
}
