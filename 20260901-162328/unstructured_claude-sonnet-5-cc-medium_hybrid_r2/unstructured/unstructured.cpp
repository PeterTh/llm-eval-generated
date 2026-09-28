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

// Static connectivity information for each element.
// connected_idx values are indices into the *local extended* dynamic array,
// which includes one ghost row on each side of the rank's owned rows.
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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Distributed (MPI row-decomposed) world state. The global n x n grid is
// split into contiguous row ranges ("stripes") across MPI ranks. Each rank
// keeps one ghost row above and below its own rows so that up/down neighbor
// data (which may live on a different rank) can be exchanged via halo swaps.
// Left/right neighbors are always within the same row and therefore always
// local to a rank.
struct LocalWorld {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;         // size n_owned
    std::vector<ElementDynamic> elements_dynamic_host;  // size n_extended (owned + 2 ghost rows)

    int n_cols = 0;         // = n_elems_root
    int n_local_rows = 0;   // rows owned by this rank
    int row_start = 0;      // global row index of first owned row
    int rank_down = MPI_PROC_NULL;  // rank owning row_start - 1
    int rank_up = MPI_PROC_NULL;    // rank owning row_start + n_local_rows
};

// Compute the contiguous row range [row_start, row_start + n_local_rows) owned by `rank`.
void computeRowRange(int rank, int nranks, int n, int& row_start, int& n_local_rows) {
    const int base = n / nranks;
    const int rem = n % nranks;
    if (rank < rem) {
        n_local_rows = base + 1;
        row_start = rank * (base + 1);
    } else {
        n_local_rows = base;
        row_start = rem * (base + 1) + (rank - rem) * base;
    }
}

// Build the local (owned + ghost) portion of a 2D square grid unstructured mesh
// for this rank's row range. This represents computation on arbitrarily-shaped
// geometries; the local piece is built independently and only communicates via
// halo exchange of boundary-row dynamic state.
void buildSquare2DLocal(LocalWorld& world, const int n_elems_root, int rank, int nranks) {
    world.n_cols = n_elems_root;
    computeRowRange(rank, nranks, n_elems_root, world.row_start, world.n_local_rows);
    world.rank_down = (world.row_start > 0) ? rank - 1 : MPI_PROC_NULL;
    world.rank_up = (world.row_start + world.n_local_rows < n_elems_root) ? rank + 1 : MPI_PROC_NULL;

    // Initialize materials (identical on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    const int n_local_rows = world.n_local_rows;
    const int n_cols = world.n_cols;
    const int n_owned = n_local_rows * n_cols;
    const int n_extended = (n_local_rows + 2) * n_cols;

    world.elements_static.resize(n_owned);
    world.elements_dynamic_host.assign(n_extended, ElementDynamic{0.0, 0.0});

    const int last = n_elems_root - 1;

    // Build connectivity independently per owned row: fully data-parallel across rows.
    #pragma omp parallel for schedule(static)
    for (int li = 0; li < n_local_rows; ++li) {
        const int x = world.row_start + li;
        for (int y = 0; y < n_cols; ++y) {
            const int owned_idx = li * n_cols + y;
            ElementStatic elem{};
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_cols) {
                    idx_t neighbor_dyn_idx;
                    if (ny != y) {
                        // left/right neighbor: same row, always local
                        neighbor_dyn_idx = static_cast<idx_t>((li + 1) * n_cols + ny);
                    } else if (nx == x + 1) {
                        // "down" neighbor row (nx = x+1)
                        if (nx < world.row_start + n_local_rows) {
                            neighbor_dyn_idx = static_cast<idx_t>((li + 2) * n_cols + ny);
                        } else {
                            neighbor_dyn_idx = static_cast<idx_t>((n_local_rows + 1) * n_cols + ny); // top ghost
                        }
                    } else {
                        // "up" neighbor row (nx = x-1)
                        if (nx >= world.row_start) {
                            neighbor_dyn_idx = static_cast<idx_t>(li * n_cols + ny);
                        } else {
                            neighbor_dyn_idx = static_cast<idx_t>(ny); // bottom ghost (row 0)
                        }
                    }
                    elem.connected_idx[elem.num_connections] = neighbor_dyn_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }

            // Corner elements: inflow/outflow to create interesting dynamics
            if (x == 0 && y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (x == 0 && y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == last) {
                elem.material_idx = INFLOW_MAT_ID;
            }

            world.elements_static[owned_idx] = elem;
        }
    }
}

// CUDA kernel: computes one flux-update iteration for all owned elements.
// `cur` and `next` are extended arrays of size (n_local_rows + 2) * n_cols;
// index 0..n_cols-1 is the "down" ghost row, n_cols..(n_local_rows+1)*n_cols-1
// are the owned rows, and the last n_cols entries are the "up" ghost row.
__global__ void simulateKernel(const ElementStatic* __restrict__ static_arr,
                                const ElementDynamic* __restrict__ cur,
                                ElementDynamic* __restrict__ next,
                                const Material* __restrict__ materials,
                                int n_owned, int n_cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_owned) return;

    const ElementStatic& es = static_arr[i];
    const Material& mat = materials[es.material_idx];
    const int self_idx = i + n_cols;  // shift by one ghost row
    const ElementDynamic self = cur[self_idx];

    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < es.num_connections; ++j) {
        const ElementDynamic nb = cur[es.connected_idx[j]];
        total_flux += (nb.current_energy - self.current_energy) *
                      mat.transfer_coeff * es.connected_flux[j] * 0.25;
    }

    ElementDynamic out;
    out.current_energy = self.current_energy + total_flux;
    out.total_flux = self.total_flux + fabs(total_flux);
    next[self_idx] = out;
}

// Run simulation for n_iters iterations using CUDA for per-element compute,
// with MPI halo exchange of boundary rows between iterations.
void runSimulation(LocalWorld& world, const int n_iters) {
    const int n_cols = world.n_cols;
    const int n_local_rows = world.n_local_rows;
    const int n_owned = n_local_rows * n_cols;
    const int n_extended = (n_local_rows + 2) * n_cols;

    ElementStatic* d_static = nullptr;
    ElementDynamic* d_cur = nullptr;
    ElementDynamic* d_next = nullptr;
    Material* d_materials = nullptr;

    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                           world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    if (n_owned > 0) {
        CUDA_CHECK(cudaMalloc(&d_static, static_cast<size_t>(n_owned) * sizeof(ElementStatic)));
        CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data(),
                               static_cast<size_t>(n_owned) * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_cur, static_cast<size_t>(n_extended) * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_next, static_cast<size_t>(n_extended) * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemcpy(d_cur, world.elements_dynamic_host.data(),
                           static_cast<size_t>(n_extended) * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next, 0, static_cast<size_t>(n_extended) * sizeof(ElementDynamic)));

    // Host-side staging buffers for halo exchange (device <-> host <-> MPI <-> host <-> device)
    std::vector<ElementDynamic> send_down(n_cols), send_up(n_cols);
    std::vector<ElementDynamic> recv_down(n_cols), recv_up(n_cols);

    const int threads = 256;
    const int blocks = (n_owned > 0) ? (n_owned + threads - 1) / threads : 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (n_owned > 0) {
            simulateKernel<<<blocks, threads>>>(d_static, d_cur, d_next, d_materials, n_owned, n_cols);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_cur, d_next);

        // Pack this rank's boundary rows (first/last owned rows) from device to host.
        if (n_owned > 0) {
            CUDA_CHECK(cudaMemcpy(send_down.data(), d_cur + n_cols,
                                   static_cast<size_t>(n_cols) * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(send_up.data(), d_cur + static_cast<size_t>(n_local_rows) * n_cols,
                                   static_cast<size_t>(n_cols) * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
        }

        // Exchange boundary rows with row-neighbor ranks (halo exchange).
        MPI_Sendrecv(send_down.data(), n_cols * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, world.rank_down, 0,
                     recv_down.data(), n_cols * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, world.rank_down, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_up.data(), n_cols * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, world.rank_up, 1,
                     recv_up.data(), n_cols * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, world.rank_up, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Unpack received ghost rows back to device.
        if (world.rank_down != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_cur, recv_down.data(),
                                   static_cast<size_t>(n_cols) * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        }
        if (world.rank_up != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_cur + static_cast<size_t>(n_local_rows + 1) * n_cols, recv_up.data(),
                                   static_cast<size_t>(n_cols) * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaMemcpy(world.elements_dynamic_host.data(), d_cur,
                           static_cast<size_t>(n_extended) * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    if (d_static) CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_materials));
}

// Validate simulation results (globally, across all MPI ranks)
bool validateResults(const LocalWorld& world) {
    const int n_cols = world.n_cols;
    const int n_local_rows = world.n_local_rows;

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for reduction(+:energy_sum, flux_sum) reduction(max:energy_max) reduction(min:energy_min) schedule(static)
    for (int i = 0; i < n_local_rows * n_cols; ++i) {
        const ElementDynamic& elem = world.elements_dynamic_host[i + n_cols];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    val_t global_energy_sum, global_flux_sum, global_energy_max, global_energy_min;
    MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        return std::isfinite(global_energy_sum) && std::isfinite(global_flux_sum) &&
               std::isfinite(global_energy_max) && std::isfinite(global_energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", global_energy_sum);
    printf("  Flux sum: %.2f\n", global_flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(global_energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(global_energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
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
    return true;
}

// Compute a simple hash of the results for verification (global, across all MPI ranks).
// Uses each element's *global* index so the combined result is identical to the
// hash computed by the original single-process implementation (XOR-combine is
// order independent, so per-rank partial hashes can simply be XORed together).
uint64_t computeHash(const LocalWorld& world) {
    const int n_cols = world.n_cols;
    const int n_local_rows = world.n_local_rows;
    const int row_start = world.row_start;

    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (int li = 0; li < n_local_rows; ++li) {
        for (int y = 0; y < n_cols; ++y) {
            const uint64_t i = static_cast<uint64_t>(row_start + li) * n_cols + y;
            const ElementDynamic& elem = world.elements_dynamic_host[(li + 1) * n_cols + y];
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash;
    MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    return global_hash;
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind each rank to a GPU (round-robin across the GPUs visible to this process/node).
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(rank % device_count));
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int exit_status = -1;  // -1: continue running, >=0: exit immediately with this code

    // Parse command line arguments (rank 0 only, then broadcast the outcome)
    if (rank == 0) {
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
                exit_status = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_status = 1;
                break;
            }
        }
    }

    {
        int params[5] = {n_elems_root, n_iters, validate ? 1 : 0, printResults ? 1 : 0, exit_status};
        MPI_Bcast(params, 5, MPI_INT, 0, MPI_COMM_WORLD);
        n_elems_root = params[0];
        n_iters = params[1];
        validate = params[2] != 0;
        printResults = params[3] != 0;
        exit_status = params[4];
    }
    if (exit_status >= 0) {
        MPI_Finalize();
        return exit_status;
    }

    if (n_elems_root < nranks) {
        if (rank == 0) {
            fprintf(stderr, "Grid size (%d) must be >= number of MPI ranks (%d)\n", n_elems_root, nranks);
        }
        MPI_Finalize();
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs visible per rank: %d, OpenMP threads per rank: %d\n",
               nranks, device_count, omp_get_max_threads());
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    // Build this rank's local (row-decomposed) piece of the unstructured mesh
    LocalWorld world;
    buildSquare2DLocal(world, n_elems_root, rank, nranks);

    // Calculate memory usage (aggregate across ranks)
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic_host.size() * sizeof(ElementDynamic) * 2;
    size_t static_mem = 0, dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation (GPU-resident per rank, MPI halo exchange between ranks)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long duration_ms;
    MPI_Allreduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Compute hash for verification (needed before/independent of printing)
    const uint64_t hash = computeHash(world);

    if (rank == 0) {
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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather global energy array on rank 0)
    if (printResults) {
        const int n_cols = world.n_cols;
        const int n_owned = world.n_local_rows * n_cols;
        std::vector<double> localEnergy(n_owned);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_owned; ++i) {
            localEnergy[i] = world.elements_dynamic_host[i + n_cols].current_energy;
        }

        std::vector<int> counts, displs;
        if (rank == 0) {
            counts.resize(nranks);
            displs.resize(nranks);
        }
        int local_count = n_owned;
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int offset = 0;
            for (int r = 0; r < nranks; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
        }

        std::vector<double> energyData;
        if (rank == 0) energyData.resize(n_elems);
        MPI_Gatherv(localEnergy.data(), local_count, MPI_DOUBLE,
                    energyData.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    int exit_code = 0;
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            exit_code = 1;
        }
    }

    MPI_Finalize();
    return exit_code;
}
