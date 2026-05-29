#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

// ---------------------------------------------------------------------------
// CUDA kernel – computes one simulation step on the GPU
// ---------------------------------------------------------------------------
__global__ void simulationKernel(
    const ElementStatic* __restrict__ elem_static,
    const Material*      __restrict__ materials,
    const ElementDynamic* __restrict__ elem_dyn,
    ElementDynamic*       __restrict__ elem_write,
    const size_t n_elems)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;

    const ElementStatic& es = elem_static[i];
    const ElementDynamic& ed = elem_dyn[i];
    const Material& mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    for (idx_t j = 0; j < es.num_connections; ++j) {
        const idx_t nb = es.connected_idx[j];
        const val_t ce = elem_dyn[nb].current_energy;
        total_flux += (ce - ed.current_energy) * mat.transfer_coeff *
                      es.connected_flux[j] * 0.25;
    }

    elem_write[i].current_energy = ed.current_energy + total_flux;
    elem_write[i].total_flux     = ed.total_flux + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Build the full 2D square grid mesh (OpenMP parallel)
// ---------------------------------------------------------------------------
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
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

// ---------------------------------------------------------------------------
// Domain decomposition info (row-wise strip with ghost rows)
// ---------------------------------------------------------------------------
struct PartitionInfo {
    int global_n_root;
    int local_start_row;   // first owned row (global)
    int local_end_row;     // last owned row + 1 (global)
    int ghost_start_row;   // first local row (global, may be ghost)
    int ghost_end_row;     // last local row + 1 (global)
    int local_n_rows;      // owned rows
    int local_n_total;     // total rows (owned + ghost)
    int local_n_elems;     // total elements
};

static PartitionInfo computePartition(int n_root, int rank, int nprocs) {
    PartitionInfo info;
    info.global_n_root = n_root;
    int rows_per = n_root / nprocs;
    int rem = n_root % nprocs;
    info.local_start_row = rank * rows_per + std::min(rank, rem);
    info.local_end_row = info.local_start_row + rows_per + (rank < rem ? 1 : 0);
    info.local_n_rows = info.local_end_row - info.local_start_row;
    info.ghost_start_row = std::max(0, info.local_start_row - 1);
    info.ghost_end_row   = std::min(n_root, info.local_end_row + 1);
    info.local_n_total   = info.ghost_end_row - info.ghost_start_row;
    info.local_n_elems   = info.local_n_total * n_root;
    return info;
}

// Build local partition from global mesh; remaps connectivity to local indices.
static World buildLocalPartition(const World& gw, const PartitionInfo& p) {
    World local;
    local.materials = gw.materials;
    const int N = p.global_n_root;
    const int goff = p.ghost_start_row;

    local.elements_static.resize(p.local_n_elems);
    local.elements_dynamic.resize(p.local_n_elems);
    local.elements_dynamic_swap.resize(p.local_n_elems);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = p.ghost_start_row; x < p.ghost_end_row; ++x) {
        for (int y = 0; y < N; ++y) {
            const idx_t gidx = static_cast<idx_t>(x * N + y);
            const idx_t lidx = static_cast<idx_t>((x - goff) * N + y);

            local.elements_static[lidx] = gw.elements_static[gidx];
            for (idx_t j = 0; j < local.elements_static[lidx].num_connections; ++j) {
                const idx_t gn = gw.elements_static[gidx].connected_idx[j];
                const int gnx = static_cast<int>(gn / N);
                const int gny = static_cast<int>(gn % N);
                local.elements_static[lidx].connected_idx[j] =
                    static_cast<idx_t>((gnx - goff) * N + gny);
            }
            local.elements_dynamic[lidx].current_energy = gw.elements_dynamic[gidx].current_energy;
            local.elements_dynamic[lidx].total_flux     = gw.elements_dynamic[gidx].total_flux;
        }
    }
    return local;
}

// ---------------------------------------------------------------------------
// Simulation: CUDA kernel + MPI halo exchange per iteration
// ---------------------------------------------------------------------------
static void runSimulation(World& world, const int n_iters,
                          const PartitionInfo& part, int rank, int nprocs) {
    const size_t n_local = world.elements_static.size();
    if (n_local == 0) return;

    // Device memory
    ElementStatic*      d_static  = nullptr;
    Material*           d_mats    = nullptr;
    ElementDynamic*     d_dyn     = nullptr;
    ElementDynamic*     d_dyn_sw  = nullptr;
    ElementDynamic*     d_halo    = nullptr;

    cudaMalloc(&d_static,  n_local * sizeof(ElementStatic));
    cudaMalloc(&d_mats,    world.materials.size() * sizeof(Material));
    cudaMalloc(&d_dyn,     n_local * sizeof(ElementDynamic));
    cudaMalloc(&d_dyn_sw,  n_local * sizeof(ElementDynamic));
    cudaMalloc(&d_halo,    2 * part.global_n_root * sizeof(ElementDynamic));

    cudaMemcpy(d_static, world.elements_static.data(),
               n_local * sizeof(ElementStatic), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mats,   world.materials.data(),
               world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dyn,    world.elements_dynamic.data(),
               n_local * sizeof(ElementDynamic), cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid  = static_cast<int>((n_local + block - 1) / block);
    const size_t row_sz = part.global_n_root * sizeof(ElementDynamic);

    // Host buffers for halo exchange
    ElementDynamic* hs[2] = {new ElementDynamic[part.global_n_root],
                             new ElementDynamic[part.global_n_root]};
    ElementDynamic* hr[2] = {new ElementDynamic[part.global_n_root],
                             new ElementDynamic[part.global_n_root]};

    const int north = (part.ghost_start_row > 0)    ? rank - 1 : MPI_PROC_NULL;
    const int south = (part.ghost_end_row < part.global_n_root) ? rank + 1 : MPI_PROC_NULL;

    MPI_Request reqs[4];

    // Determine whether we actually have ghost rows
    const bool has_top_ghost  = (part.ghost_start_row < part.local_start_row);
    const bool has_bot_ghost  = (part.ghost_end_row   > part.local_end_row);

    for (int iter = 0; iter < n_iters; ++iter) {
        // --- Halo exchange ---
        if (has_top_ghost || has_bot_ghost) {
            if (part.local_n_rows > 0) {
                const int top_lr  = part.local_start_row - part.ghost_start_row;
                const int bot_lr  = part.local_end_row   - part.ghost_start_row - 1;

                // Copy owned boundary rows to halo buffer on GPU
                if (has_top_ghost)
                    cudaMemcpy(d_halo, d_dyn + top_lr * part.global_n_root,
                               row_sz, cudaMemcpyDeviceToDevice);
                if (has_bot_ghost)
                    cudaMemcpy(d_halo + part.global_n_root,
                               d_dyn + bot_lr * part.global_n_root,
                               row_sz, cudaMemcpyDeviceToDevice);
            }

            // GPU -> host
            if (has_top_ghost)
                cudaMemcpyAsync(hs[0], d_halo, row_sz, cudaMemcpyDeviceToHost);
            if (has_bot_ghost)
                cudaMemcpyAsync(hs[1], d_halo + part.global_n_root, row_sz, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);

            // Non-blocking MPI – track request count
            int nreqs = 0;
            if (has_top_ghost) {
                MPI_Isend(hs[0], row_sz, MPI_BYTE, north, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
                MPI_Irecv(hr[0], row_sz, MPI_BYTE, north, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            }
            if (has_bot_ghost) {
                MPI_Isend(hs[1], row_sz, MPI_BYTE, south, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
                MPI_Irecv(hr[1], row_sz, MPI_BYTE, south, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            }
            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

            // host -> GPU -> ghost rows
            if (has_top_ghost) {
                cudaMemcpyAsync(d_halo, hr[0], row_sz, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(0);
                cudaMemcpy(d_dyn, d_halo, row_sz, cudaMemcpyDeviceToDevice);
            }
            if (has_bot_ghost) {
                cudaMemcpyAsync(d_halo + part.global_n_root, hr[1], row_sz, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(0);
                cudaMemcpy(d_dyn + (part.local_n_total - 1) * part.global_n_root,
                           d_halo + part.global_n_root, row_sz, cudaMemcpyDeviceToDevice);
            }
        }

        // --- CUDA kernel ---
        simulationKernel<<<grid, block>>>(d_static, d_mats, d_dyn, d_dyn_sw, n_local);
        std::swap(d_dyn, d_dyn_sw);
    }

    cudaMemcpy(world.elements_dynamic.data(), d_dyn,
               n_local * sizeof(ElementDynamic), cudaMemcpyDeviceToHost);

    cudaFree(d_static); cudaFree(d_mats); cudaFree(d_dyn);
    cudaFree(d_dyn_sw); cudaFree(d_halo);
    delete[] hs[0]; delete[] hs[1]; delete[] hr[0]; delete[] hr[1];
}

// ---------------------------------------------------------------------------
// Validate simulation results (OpenMP reduction)
// ---------------------------------------------------------------------------
bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum,flux_sum) \
                             reduction(max:energy_max) reduction(min:energy_min) \
                             schedule(static)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        energy_sum += world.elements_dynamic[i].current_energy;
        flux_sum   += world.elements_dynamic[i].total_flux;
        energy_max  = std::max(world.elements_dynamic[i].current_energy, energy_max);
        energy_min  = std::min(world.elements_dynamic[i].current_energy, energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum))      { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > energy_epsilon)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum))        { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min))
        { printf("  ERROR: Energy extrema are not finite\n"); return false; }

    printf("  Validation: PASSED\n");
    return true;
}

// ---------------------------------------------------------------------------
// Compute a simple hash of the results for verification
// ---------------------------------------------------------------------------
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
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_elems_root = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", nprocs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Build full mesh on rank 0 ----
    World global_world;
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
        buildSquare2D(global_world, n_elems_root);
    }

    // Broadcast materials
    int n_mats = static_cast<int>(global_world.materials.size());
    MPI_Bcast(&n_mats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        global_world.materials.resize(n_mats);
    }
    MPI_Bcast(global_world.materials.data(), n_mats * sizeof(Material),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    // ---- Compute partition ----
    PartitionInfo part = computePartition(n_elems_root, rank, nprocs);

    // ---- Broadcast full mesh, build local partition ----
    if (rank == 0) {
        // Broadcast static + dynamic data to all ranks
        MPI_Bcast(global_world.elements_static.data(),
                  n_elems * sizeof(ElementStatic), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_world.elements_dynamic.data(),
                  n_elems * sizeof(ElementDynamic), MPI_BYTE, 0, MPI_COMM_WORLD);

        World local = buildLocalPartition(global_world, part);

        const size_t static_mem  = part.local_n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = part.local_n_elems * sizeof(ElementDynamic) * 2;
        printf("Memory usage (local): %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (static_mem + dynamic_mem) / (1024.0*1024.0),
               static_mem / (1024.0*1024.0),
               dynamic_mem / (1024.0*1024.0));
        printf("\n");

        MPI_Barrier(MPI_COMM_WORLD);
        printf("Running simulation...\n");
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        runSimulation(local, n_iters, part, rank, nprocs);
        auto end   = std::chrono::high_resolution_clock::now();

        // ---- Gather owned results to rank 0 ----
        // Each rank sends its owned (non-ghost) rows
        std::vector<int> counts(nprocs);
        counts[rank] = part.local_n_rows * n_elems_root;
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                      counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs);
        displs[0] = 0;
        for (int r = 1; r < nprocs; ++r) displs[r] = displs[r-1] + counts[r-1];

        // Extract owned elements (no ghost)
        std::vector<ElementDynamic> my_owned(part.local_n_rows * n_elems_root);
        for (int row = 0; row < part.local_n_rows; ++row) {
            const int lr = (part.local_start_row + row) - part.ghost_start_row;
            const int global_base = (part.local_start_row + row) * n_elems_root;
            const int local_base  = lr * n_elems_root;
            for (int col = 0; col < n_elems_root; ++col) {
                my_owned[global_base + col - displs[rank]] =
                    local.elements_dynamic[local_base + col];
            }
        }

        std::vector<ElementDynamic> full_results(n_elems);
        // Convert counts/displs from element counts to byte counts for MPI_BYTE
        std::vector<int> byte_counts(nprocs), byte_displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            byte_counts[r] = counts[r] * static_cast<int>(sizeof(ElementDynamic));
            byte_displs[r] = displs[r] * static_cast<int>(sizeof(ElementDynamic));
        }
        MPI_Gatherv(my_owned.data(),
                    static_cast<int>(my_owned.size() * sizeof(ElementDynamic)), MPI_BYTE,
                    full_results.data(),
                    byte_counts.data(), byte_displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(full_results);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(full_results.size());
            for (const auto& elem : full_results)
                energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            World val_world;
            val_world.elements_dynamic = full_results;
            if (!validateResults(val_world)) { MPI_Finalize(); return 1; }
        }
    } else {
        // Non-zero ranks: receive full mesh, build local partition
        std::vector<ElementStatic>   recv_static(n_elems);
        std::vector<ElementDynamic>  recv_dynamic(n_elems);
        MPI_Bcast(recv_static.data(),  n_elems * sizeof(ElementStatic),  MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(recv_dynamic.data(), n_elems * sizeof(ElementDynamic), MPI_BYTE, 0, MPI_COMM_WORLD);

        World recv_world;
        recv_world.materials       = global_world.materials;
        recv_world.elements_static = std::move(recv_static);
        recv_world.elements_dynamic = std::move(recv_dynamic);

        World local = buildLocalPartition(recv_world, part);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        runSimulation(local, n_iters, part, rank, nprocs);
        auto end   = std::chrono::high_resolution_clock::now();

        // Gather owned results to rank 0
        std::vector<int> counts(nprocs);
        counts[rank] = part.local_n_rows * n_elems_root;
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                      counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs);
        displs[0] = 0;
        for (int r = 1; r < nprocs; ++r) displs[r] = displs[r-1] + counts[r-1];

        std::vector<ElementDynamic> my_owned(part.local_n_rows * n_elems_root);
        for (int row = 0; row < part.local_n_rows; ++row) {
            const int lr = (part.local_start_row + row) - part.ghost_start_row;
            const int global_base = (part.local_start_row + row) * n_elems_root;
            const int local_base  = lr * n_elems_root;
            for (int col = 0; col < n_elems_root; ++col) {
                my_owned[global_base + col - displs[rank]] =
                    local.elements_dynamic[local_base + col];
            }
        }

        std::vector<int> byte_counts(nprocs), byte_displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            byte_counts[r] = counts[r] * static_cast<int>(sizeof(ElementDynamic));
            byte_displs[r] = displs[r] * static_cast<int>(sizeof(ElementDynamic));
        }
        MPI_Gatherv(my_owned.data(),
                    static_cast<int>(my_owned.size() * sizeof(ElementDynamic)), MPI_BYTE,
                    nullptr, byte_counts.data(), byte_displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
