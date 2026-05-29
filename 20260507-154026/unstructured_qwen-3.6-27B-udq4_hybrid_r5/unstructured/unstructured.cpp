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

// ---------------------------------------------------------------------------
// Material properties for energy transfer
// ---------------------------------------------------------------------------
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// ---------------------------------------------------------------------------
// Static connectivity information for each element
// ---------------------------------------------------------------------------
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// ---------------------------------------------------------------------------
// Dynamic state for each element (packed for GPU transfer)
// ---------------------------------------------------------------------------
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// ---------------------------------------------------------------------------
// World state
// ---------------------------------------------------------------------------
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// ---------------------------------------------------------------------------
// Material type IDs
// ---------------------------------------------------------------------------
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// MPI state
// ---------------------------------------------------------------------------
static int mpi_rank = 0, mpi_size = 1;

// ---------------------------------------------------------------------------
// Build a 2D square grid as an unstructured mesh (OpenMP parallelized)
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

// ---------------------------------------------------------------------------
// CUDA kernel – flux computation
// ---------------------------------------------------------------------------
__global__ void fluxKernel(
    const ElementStatic*  d_static,
    const ElementDynamic* d_dynamic,
    ElementDynamic*       d_swap,
    const Material*       d_materials,
    const int             n_local,
    const int             local_offset)
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const int ei = local_offset + i;
    const ElementStatic&  es = d_static[i];
    const ElementDynamic& ed = d_dynamic[ei];
    const Material&       mat = d_materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    for (int j = 0; j < es.num_connections; ++j) {
        const idx_t  nid = es.connected_idx[j];
        const val_t  cf  = es.connected_flux[j];
        total_flux += (d_dynamic[nid].current_energy - ed.current_energy)
                     * mat.transfer_coeff * cf * 0.25;
    }

    d_swap[ei].current_energy = ed.current_energy + total_flux;
    d_swap[ei].total_flux     = ed.total_flux     + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Run simulation – hybrid MPI + CUDA + OpenMP
// ---------------------------------------------------------------------------
void runSimulation(World& world, const int n_iters, const int n_elems_root) {
    // Domain decomposition: distribute rows among MPI ranks
    const int rows_per_rank = n_elems_root / mpi_size;
    const int local_start_row = mpi_rank * rows_per_rank;
    const int local_end_row   = (mpi_rank == mpi_size - 1)
                                   ? n_elems_root
                                   : (mpi_rank + 1) * rows_per_rank;
    const int local_rows = local_end_row - local_start_row;
    const int n_local    = local_rows * n_elems_root;
    const int first_local_idx = local_start_row * n_elems_root;

    // Extended buffer: [upper_halo][local][lower_halo]
    const int halo_rows   = 1;
    const int ext_rows    = local_rows + 2 * halo_rows;
    const int n_extended  = ext_rows * n_elems_root;
    const int halo_offset = n_elems_root;  // upper halo = 1 row

    // Allocate buffers
    ElementStatic*  ext_static  = new ElementStatic[n_local];
    ElementDynamic* ext_dynamic = new ElementDynamic[n_extended];
    ElementDynamic* ext_swap    = new ElementDynamic[n_extended];

    // Copy local data from global world
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n_local; ++j) {
        const int g = first_local_idx + j;
        ext_static[j]    = world.elements_static[g];
        ext_dynamic[j + halo_offset] = world.elements_dynamic[g];
    }

    // Zero-initialize halo rows
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n_extended; ++j) {
        if (j < halo_offset || j >= halo_offset + n_local) {
            ext_dynamic[j].current_energy = 0.0;
            ext_dynamic[j].total_flux     = 0.0;
            ext_swap[j].current_energy    = 0.0;
            ext_swap[j].total_flux        = 0.0;
        }
    }

    // Remap connectivity indices to extended buffer indices
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n_local; ++j) {
        for (int c = 0; c < static_cast<int>(ext_static[j].num_connections); ++c) {
            idx_t& nid = ext_static[j].connected_idx[c];
            const int g_row = static_cast<int>(nid) / n_elems_root;
            const int g_col = static_cast<int>(nid) % n_elems_root;
            const int ext_row = g_row - local_start_row + halo_rows;
            nid = static_cast<idx_t>(ext_row * n_elems_root + g_col);
        }
    }

    // Halo exchange buffers
    const int halo_count = n_elems_root;
    const int halo_dbl   = halo_count * 2;  // ElementDynamic has 2 doubles
    double* send_upper   = new double[halo_dbl];
    double* send_lower   = new double[halo_dbl];
    double* recv_upper   = new double[halo_dbl];
    double* recv_lower   = new double[halo_dbl];

    // GPU buffers
    ElementStatic*  d_static    = nullptr;
    ElementDynamic* d_dynamic   = nullptr;
    ElementDynamic* d_swap      = nullptr;
    Material*       d_materials = nullptr;

    cudaMalloc(&d_static,    n_local    * sizeof(ElementStatic));
    cudaMalloc(&d_dynamic,   n_extended * sizeof(ElementDynamic));
    cudaMalloc(&d_swap,      n_extended * sizeof(ElementDynamic));
    cudaMalloc(&d_materials, world.materials.size() * sizeof(Material));

    cudaMemcpy(d_static,    ext_static, n_local * sizeof(ElementStatic),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, world.materials.data(),
               world.materials.size() * sizeof(Material),
               cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = (n_local + blockSize - 1) / blockSize;

    // -----------------------------------------------------------------------
    // Simulation loop
    // -----------------------------------------------------------------------
    for (int iter = 0; iter < n_iters; ++iter) {
        // --- Prepare send buffers ---
        // Upper send: last row of local data
        // Lower send: first row of local data
        const int first_local_ext = halo_offset;
        const int last_local_ext  = halo_offset + n_local - 1;

       double* dyn_raw = reinterpret_cast<double*>(ext_dynamic);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < halo_dbl; ++k) {
            send_upper[k] = dyn_raw[(last_local_ext - halo_count + 1) * 2 + k];
            send_lower[k] = dyn_raw[first_local_ext * 2 + k];
        }

        // --- Halo exchange using MPI_Sendrecv ---
        // Tag convention (consistent between communicating pairs):
        //   TAG_SEND = 10, TAG_RECV = 11
        // Rank i sends to rank i-1 with TAG_SEND, receives from rank i-1 with TAG_RECV
        // Rank i sends to rank i+1 with TAG_SEND, receives from rank i+1 with TAG_RECV

        if (mpi_rank > 0) {
            MPI_Sendrecv(send_upper, halo_dbl, MPI_DOUBLE, mpi_rank - 1, 10,
                         recv_upper, halo_dbl, MPI_DOUBLE, mpi_rank - 1, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (mpi_rank < mpi_size - 1) {
            MPI_Sendrecv(send_lower, halo_dbl, MPI_DOUBLE, mpi_rank + 1, 10,
                         recv_lower, halo_dbl, MPI_DOUBLE, mpi_rank + 1, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // --- Copy halo data into extended buffer ---
        if (mpi_rank > 0) {
#pragma omp parallel for schedule(static)
            for (int k = 0; k < halo_dbl; ++k)
                dyn_raw[k] = recv_upper[k];
        }
        if (mpi_rank < mpi_size - 1) {
            const int lower_halo_start = (halo_offset + n_local) * 2;
#pragma omp parallel for schedule(static)
            for (int k = 0; k < halo_dbl; ++k)
                dyn_raw[lower_halo_start + k] = recv_lower[k];
        }

        // --- Upload dynamic data to GPU ---
        cudaMemcpy(d_dynamic, ext_dynamic, n_extended * sizeof(ElementDynamic),
                   cudaMemcpyHostToDevice);

        // --- Launch CUDA kernel ---
        fluxKernel<<<gridSize, blockSize>>>(
            d_static, d_dynamic, d_swap, d_materials, n_local, halo_offset);
        cudaDeviceSynchronize();

        // --- Download swap buffer from GPU ---
        cudaMemcpy(ext_swap, d_swap, n_extended * sizeof(ElementDynamic),
                   cudaMemcpyDeviceToHost);

        // --- Copy local results back into ext_dynamic ---
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n_local; ++j)
            ext_dynamic[halo_offset + j] = ext_swap[halo_offset + j];
    }

    // -----------------------------------------------------------------------
    // Gather results to rank 0 using Gatherv
    // -----------------------------------------------------------------------
    {
        std::vector<int> counts(mpi_size);
        std::vector<int> displs(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            const int r_start = (r == mpi_size - 1)
                                    ? ((mpi_size - 1) * rows_per_rank)
                                    : r * rows_per_rank;
            const int r_end   = (r == mpi_size - 1)
                                    ? n_elems_root
                                    : (r + 1) * rows_per_rank;
            const int r_rows  = r_end - r_start;
            counts[r] = r_rows * n_elems_root * 2;
            displs[r] = r_start * n_elems_root * 2;
        }
        if (mpi_rank == 0) {
            MPI_Gatherv(reinterpret_cast<double*>(ext_dynamic) + halo_offset * 2,
                        n_local * 2, MPI_DOUBLE,
                        reinterpret_cast<double*>(world.elements_dynamic.data()),
                        counts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(reinterpret_cast<double*>(ext_dynamic) + halo_offset * 2,
                        n_local * 2, MPI_DOUBLE,
                        nullptr,
                        counts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        }
    }

    // Cleanup
    cudaFree(d_static);
    cudaFree(d_dynamic);
    cudaFree(d_swap);
    cudaFree(d_materials);
    delete[] ext_static;
    delete[] ext_dynamic;
    delete[] ext_swap;
    delete[] send_upper;
    delete[] send_lower;
    delete[] recv_upper;
    delete[] recv_lower;
}

// ---------------------------------------------------------------------------
// Validate simulation results (on rank 0)
// ---------------------------------------------------------------------------
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    // Assign each MPI rank to a GPU device
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int gpu_device = mpi_rank % std::max(num_gpus, 1);
    cudaSetDevice(gpu_device);

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d, GPUs: %d\n", mpi_size, num_gpus);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build mesh on rank 0, then broadcast
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");

    World world;
    buildSquare2D(world, n_elems_root);

    // Broadcast materials
    const int n_mats = 3;
    if (mpi_rank == 0) {
        MPI_Bcast(world.materials.data(), n_mats * sizeof(Material), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    } else {
        world.materials.resize(n_mats);
        MPI_Bcast(world.materials.data(), n_mats * sizeof(Material), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    }

    // Broadcast static and initial dynamic data
    MPI_Bcast(world.elements_static.data(), n_elems * sizeof(ElementStatic),
              MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(world.elements_dynamic.data(), n_elems * sizeof(ElementDynamic),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        const size_t static_mem  = world.elements_static.size()  * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem   = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, n_elems_root);

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Only rank 0 prints results
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems)
                                          / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world.elements_dynamic.size());
            for (const auto& elem : world.elements_dynamic)
                energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            if (!validateResults(world)) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
