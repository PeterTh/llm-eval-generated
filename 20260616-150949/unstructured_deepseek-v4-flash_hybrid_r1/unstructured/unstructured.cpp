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

// World state (local per-MPI-rank portion with ghost cells)
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

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans)  do { cudaAssert((ans), __FILE__, __LINE__); } while (0)
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %d: %s  (%s:%d)\n",
                static_cast<int>(code), cudaGetErrorString(code), file, line);
        int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ---------------------------------------------------------------------------
// CUDA device functions and kernel
// ---------------------------------------------------------------------------
__device__ inline val_t device_compute_flux(
    val_t transfer_coeff,
    val_t this_energy,
    val_t connection_flux,
    val_t other_energy)
{
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

__global__ void flux_kernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    const size_t     n_owned,
    const size_t     owned_offset)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_owned) return;

    const size_t          local_idx = owned_offset + i;
    const ElementStatic&  es = elements_static[i];
    const ElementDynamic& ed = elements_dynamic[local_idx];
    const Material&       mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    for (idx_t j = 0; j < es.num_connections; ++j) {
        total_flux += device_compute_flux(
            mat.transfer_coeff,
            ed.current_energy,
            es.connected_flux[j],
            elements_dynamic[es.connected_idx[j]].current_energy);
    }

    elements_dynamic_swap[local_idx].current_energy = ed.current_energy + total_flux;
    elements_dynamic_swap[local_idx].total_flux     = ed.total_flux + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Build a square 2D grid as an unstructured mesh, distributed across MPI ranks
// Each rank owns a contiguous block of columns.  The dynamic arrays include
// ghost cells at the left and right boundaries (one column each) so that
// neighbor data is available without index remapping inside the kernel.
// ---------------------------------------------------------------------------
void buildDistributedSquare2D(
    World& world,
    const int n_elems_root,
    const int rank,
    const int size,
    int&      out_col_start,
    int&      out_col_end)
{
    // ---- 1D decomposition along x (columns) ----
    int cols_per_rank = n_elems_root / size;
    int remainder     = n_elems_root % size;

    int col_start = rank * cols_per_rank + (rank < remainder ? rank : remainder);
    int col_end   = col_start + cols_per_rank + (rank < remainder ? 1 : 0);
    int local_nx  = col_end - col_start;                     // owned columns
    int total_nx  = local_nx + 2;                            // + 2 ghost columns

    out_col_start = col_start;
    out_col_end   = col_end;

    const int n_owned = local_nx * n_elems_root;
    const int n_total = total_nx * n_elems_root;

    // ---- Materials (identical on all ranks) ----
    world.materials.clear();
    world.materials.reserve(3);
    world.materials.emplace_back(Material{0.8,  0.0});   // Default
    world.materials.emplace_back(Material{0.8,  0.5});   // Inflow
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow

    // ---- Allocate local data ----
    world.elements_static.resize(n_owned);
    world.elements_dynamic.resize(n_total);
    world.elements_dynamic_swap.resize(n_total);

    // ---- Initialise all elements ----
    #pragma omp parallel for
    for (int i = 0; i < n_owned; ++i) {
        world.elements_static[i].material_idx   = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    #pragma omp parallel for
    for (int i = 0; i < n_total; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux     = 0.0;
        world.elements_dynamic_swap[i].current_energy = 0.0;
        world.elements_dynamic_swap[i].total_flux     = 0.0;
    }

    // ---- Build connectivity (only owned elements) ----
    // Each owned element stores *ghost-extended* local indices so that the
    // CUDA kernel can simply read  elements_dynamic[connected_idx[j]].
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int x = col_start; x < col_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int owned_idx  = (x - col_start) * n_elems_root + y;
            ElementStatic& elem  = world.elements_static[owned_idx];

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                if (nx >= 0 && nx < n_elems_root &&
                    ny >= 0 && ny < n_elems_root) {
                    // Ghost-extended index of the neighbour
                    const idx_t neigh_idx =
                        static_cast<idx_t>(nx - col_start + 1) * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neigh_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // ---- Set corner materials ----
    const int last_idx = n_elems_root - 1;

    // (0, 0)  → INFLOW  — owned by rank 0
    if (col_start <= 0 && 0 < col_end) {
        const int oi = (0 - col_start) * n_elems_root + 0;
        world.elements_static[oi].material_idx = INFLOW_MAT_ID;
    }
    // (0, last_idx) → OUTFLOW  — owned by rank 0
    if (col_start <= 0 && 0 < col_end) {
        const int oi = (0 - col_start) * n_elems_root + last_idx;
        world.elements_static[oi].material_idx = OUTFLOW_MAT_ID;
    }
    // (last_idx, 0) → OUTFLOW  — owned by last rank
    if (col_start <= last_idx && last_idx < col_end) {
        const int oi = (last_idx - col_start) * n_elems_root + 0;
        world.elements_static[oi].material_idx = OUTFLOW_MAT_ID;
    }
    // (last_idx, last_idx) → INFLOW  — owned by last rank
    if (col_start <= last_idx && last_idx < col_end) {
        const int oi = (last_idx - col_start) * n_elems_root + last_idx;
        world.elements_static[oi].material_idx = INFLOW_MAT_ID;
    }
}

// ---------------------------------------------------------------------------
// Run the simulation with hybrid MPI + CUDA + OpenMP parallelism
//
//  Domain:        1-D column decomposition (x-axis) across MPI ranks.
//  Accelerator:   CUDA kernel launched on each rank's GPU.
//  Ghost cells:   One column of padding at each x-boundary, exchanged every
//                 iteration via non-blocking MPI point-to-point.
//  Host work:     OpenMP for packing / unpacking MPI buffers.
// ---------------------------------------------------------------------------
void runSimulationHybrid(
    World&    world,
    const int n_iters,
    const int n_elems_root,
    const int rank,
    const int size,
    const int col_start,
    const int col_end,
    MPI_Comm  comm)
{
    const int  local_nx   = col_end - col_start;
    const int  total_nx   = local_nx + 2;
    const size_t n_owned     = static_cast<size_t>(local_nx) * n_elems_root;
    const size_t total_elems = static_cast<size_t>(total_nx) * n_elems_root;
    const size_t ghost_span  = static_cast<size_t>(n_elems_root);

    // Offsets inside the ghost-extended array
    const size_t owned_offset     = ghost_span;                       // first owned element
    const size_t right_ghost_off  = ghost_span + n_owned;            // first right ghost

    // ---- MPI neighbour ranks (MPI_PROC_NULL means no neighbour) ----
    const int left_rank  = (rank > 0)       ? rank - 1 : MPI_PROC_NULL;
    const int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // ---- Allocate device memory ----
    ElementStatic*  d_static         = nullptr;
    ElementDynamic* d_dynamic         = nullptr;
    ElementDynamic* d_dynamic_swap    = nullptr;
    Material*       d_materials       = nullptr;

    CUDA_CHECK(cudaMalloc(&d_static,   world.elements_static.size()  * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_dynamic,  total_elems                   * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_dynamic_swap, total_elems               * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size()       * sizeof(Material)));

    // ---- Upload data to device ----
    CUDA_CHECK(cudaMemcpy(d_static,  world.elements_static.data(),
                          world.elements_static.size() * sizeof(ElementStatic),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dynamic, world.elements_dynamic.data(),
                          total_elems * sizeof(ElementDynamic),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dynamic_swap, world.elements_dynamic_swap.data(),
                          total_elems * sizeof(ElementDynamic),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));

    // ---- Launch configuration ----
    const int block_size = 256;
    const int num_blocks = static_cast<int>((n_owned + block_size - 1) / block_size);

    // ---- Host buffers for ghost exchange ----
    // We copy full ElementDynamic from device, then extract current_energy
    // into double vectors for MPI (half the MPI traffic).
    std::vector<ElementDynamic> left_boundary(ghost_span);
    std::vector<ElementDynamic> right_boundary(ghost_span);
    std::vector<val_t> left_send_energy(ghost_span);
    std::vector<val_t> right_send_energy(ghost_span);
    std::vector<val_t> left_recv_energy(ghost_span);
    std::vector<val_t> right_recv_energy(ghost_span);
    std::vector<ElementDynamic> ghost_left(ghost_span);
    std::vector<ElementDynamic> ghost_right(ghost_span);

    // ---- Main simulation loop ----
    for (int iter = 0; iter < n_iters; ++iter) {

        // ---- 1. Copy boundary owned data from device → host ----
        // Leftmost owned column: ghost-extended indices [owned_offset, owned_offset + ghost_span)
        CUDA_CHECK(cudaMemcpy(left_boundary.data(),
                              d_dynamic + owned_offset,
                              ghost_span * sizeof(ElementDynamic),
                              cudaMemcpyDeviceToHost));
        // Rightmost owned column: ghost-extended indices [local_nx * ghost_span, (local_nx + 1) * ghost_span)
        CUDA_CHECK(cudaMemcpy(right_boundary.data(),
                              d_dynamic + right_ghost_off - ghost_span,
                              ghost_span * sizeof(ElementDynamic),
                              cudaMemcpyDeviceToHost));

        // Extract current_energy values for MPI (only energy is needed for neighbours)
        #pragma omp parallel for
        for (size_t y = 0; y < ghost_span; ++y) {
            left_send_energy[y]  = left_boundary[y].current_energy;
            right_send_energy[y] = right_boundary[y].current_energy;
        }

        // ---- 2. MPI ghost exchange (non-blocking, current_energy only) ----
        MPI_Request reqs[4];
        MPI_Irecv( left_recv_energy.data(), static_cast<int>(ghost_span), MPI_DOUBLE,
                   left_rank,  0, comm, &reqs[0]);
        MPI_Irecv( right_recv_energy.data(), static_cast<int>(ghost_span), MPI_DOUBLE,
                   right_rank, 1, comm, &reqs[1]);
        MPI_Isend( left_send_energy.data(),  static_cast<int>(ghost_span), MPI_DOUBLE,
                   left_rank,  1, comm, &reqs[2]);
        MPI_Isend( right_send_energy.data(), static_cast<int>(ghost_span), MPI_DOUBLE,
                   right_rank, 0, comm, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // ---- 3. Build ghost-element structs from received energy and upload to device ----
        #pragma omp parallel for
        for (size_t y = 0; y < ghost_span; ++y) {
            ghost_left[y]  = ElementDynamic{left_recv_energy[y],  0.0};
            ghost_right[y] = ElementDynamic{right_recv_energy[y], 0.0};
        }

        CUDA_CHECK(cudaMemcpy(d_dynamic,
                              ghost_left.data(),
                              ghost_span * sizeof(ElementDynamic),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dynamic + right_ghost_off,
                              ghost_right.data(),
                              ghost_span * sizeof(ElementDynamic),
                              cudaMemcpyHostToDevice));

        // ---- 4. Launch CUDA kernel (owned elements only) ----
        flux_kernel<<<num_blocks, block_size>>>(
            d_static, d_dynamic, d_dynamic_swap, d_materials,
            n_owned, owned_offset);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- 5. Swap device buffers ----
        std::swap(d_dynamic, d_dynamic_swap);
    }

    // ---- Copy final state to host (owned portion) ----
    // After the loop d_dynamic points to the buffer that holds the most recent
    // results (swapped at the end of the last iteration).
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data() + owned_offset,
                          d_dynamic + owned_offset,
                          n_owned * sizeof(ElementDynamic),
                          cudaMemcpyDeviceToHost));

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_dynamic));
    CUDA_CHECK(cudaFree(d_dynamic_swap));
    CUDA_CHECK(cudaFree(d_materials));
}

// ---------------------------------------------------------------------------
// Distributed validation (reductions across all ranks)
// ---------------------------------------------------------------------------
bool validateResultsDistributed(
    const World& world,
    const int    n_elems_root,
    const int    col_start,
    const int    col_end,
    const int    rank,
    MPI_Comm     comm)
{
    const int  local_nx = col_end - col_start;
    const size_t n_owned     = static_cast<size_t>(local_nx) * n_elems_root;
    const size_t owned_offset = static_cast<size_t>(n_elems_root);

    // Per-rank partial sums
    val_t energy_sum = 0.0;
    val_t flux_sum   = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum, flux_sum) \
        reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < n_owned; ++i) {
        const auto& e = world.elements_dynamic[owned_offset + i];
        energy_sum += e.current_energy;
        flux_sum   += e.total_flux;
        if (e.current_energy > energy_max) energy_max = e.current_energy;
        if (e.current_energy < energy_min) energy_min = e.current_energy;
    }

    // Global reductions
    val_t g_energy_sum, g_flux_sum, g_energy_max, g_energy_min;
    MPI_Reduce(&energy_sum, &g_energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&flux_sum,   &g_flux_sum,   1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&energy_max, &g_energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&energy_min, &g_energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    if (rank != 0) return true;

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", g_energy_sum);
    printf("  Flux sum: %.2f\n",   g_flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", g_energy_min, g_energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(g_energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(g_energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(g_flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(g_energy_max) || !std::isfinite(g_energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

// ---------------------------------------------------------------------------
// Distributed hash (XOR-reduced across ranks)
// ---------------------------------------------------------------------------
uint64_t computeHashDistributed(
    const World& world,
    const int    n_elems_root,
    const int    col_start,
    const int    col_end)
{
    const int  local_nx = col_end - col_start;
    const size_t n_owned     = static_cast<size_t>(local_nx) * n_elems_root;
    const size_t owned_offset = static_cast<size_t>(n_elems_root);

    uint64_t hash = 0;
    for (size_t i = 0; i < n_owned; ++i) {
        // Global element index for this owned element
        const size_t row    = static_cast<size_t>(col_start) + i / n_elems_root;
        const size_t col    = i % n_elems_root;
        const size_t global = row * n_elems_root + col;

        const auto& e = world.elements_dynamic[owned_offset + i];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&e.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&e.total_flux);
        hash ^= (*e_ptr + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// ---------------------------------------------------------------------------
// Print results for external validation (gathered to rank 0)
// ---------------------------------------------------------------------------
void printResultsDistributed(
    const World& world,
    const int    n_elems_root,
    const int    n_elems,
    const int    col_start,
    const int    col_end,
    const int    rank,
    const int    size,
    MPI_Comm     comm)
{
    const int  local_nx = col_end - col_start;
    const int  n_owned  = local_nx * n_elems_root;
    const size_t owned_offset = static_cast<size_t>(n_elems_root);

    // Each rank's count of owned elements
    int local_count = n_owned;
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i-1] + counts[i-1];
    }

    // Extract local energy values
    std::vector<double> local_energy(n_owned);
    #pragma omp parallel for
    for (int i = 0; i < n_owned; ++i) {
        local_energy[i] = world.elements_dynamic[owned_offset + i].current_energy;
    }

    // Gather to rank 0
    std::vector<double> all_energy;
    if (rank == 0) all_energy.resize(static_cast<size_t>(n_elems));

    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                all_energy.data(),   counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        print_results(all_energy, "ElementEnergy");
    }
}

// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]  (run with mpirun -np <N> %s [options])\n", progName, progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Assign GPU to MPI rank (round‑robin) ----
    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    if (ngpus == 0) {
        fprintf(stderr, "No CUDA-capable device found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % ngpus));

    // ---- Parse arguments on rank 0 and broadcast ----
    int n_elems_root = 512;
    int n_iters      = 10;
    bool validate    = false;
    bool printRes    = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printRes = true;
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printRes,     1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // ---- Build distributed mesh ----
    if (rank == 0) printf("Building distributed unstructured mesh...\n");
    World world;
    int col_start, col_end;
    buildDistributedSquare2D(world, n_elems_root, rank, size,
                             col_start, col_end);

    const int local_nx = col_end - col_start;

    // ---- Print memory usage (rank 0 only) ----
    if (rank == 0) {
        const size_t static_mem  = static_cast<size_t>(local_nx) * n_elems_root * sizeof(ElementStatic) * size;
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // ---- Run hybrid simulation ----
    if (rank == 0) printf("Running hybrid MPI+CUDA+OpenMP simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);   // synchronise before timing
    double t_start = MPI_Wtime();

    runSimulationHybrid(world, n_iters, n_elems_root, rank, size,
                        col_start, col_end, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = max_elapsed * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / max_elapsed / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // ---- Distributed hash ----
    const uint64_t local_hash = computeHashDistributed(
        world, n_elems_root, col_start, col_end);
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    // ---- Print results for external validation ----
    if (printRes) {
        printResultsDistributed(world, n_elems_root, n_elems,
                                col_start, col_end, rank, size,
                                MPI_COMM_WORLD);
    }

    // ---- Distributed validation ----
    bool valid = true;
    if (validate) {
        valid = validateResultsDistributed(world, n_elems_root, col_start, col_end,
                                           rank, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return (valid) ? 0 : 1;
}
