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

// Static connectivity information for each element. Connection indices are
// expressed in the "extended" local index space of the owning rank (see
// World below), so this struct is fully self-contained for the GPU kernel.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (extended-local)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Distributed (per-MPI-rank) world state.
//
// The global n_root x n_root grid is decomposed into contiguous row blocks
// across MPI ranks (one block per rank). Each rank only stores its own
// elements. Element connectivity indices are expressed in an "extended"
// local index space that reserves one ghost row on each side of the local
// block for values received from neighboring ranks:
//
//   [0, n_root)                          -> ghost row below (row local_x_start-1)
//   [n_root, n_root + local_n)           -> locally owned rows
//   [n_root + local_n, n_root*2+local_n) -> ghost row above (row local_x_start+local_x_count)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;    // size local_n
    std::vector<ElementDynamic> elements_dynamic;  // size local_n

    int n_elems_root = 0;
    int local_x_start = 0;
    int local_x_count = 0;
    size_t local_n = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Compute the contiguous row block [start, start+count) owned by `rank` when
// splitting `n_root` grid rows across `nranks` MPI ranks as evenly as possible.
void computeRowPartition(int n_root, int nranks, int rank, int& start, int& count) {
    const int base = n_root / nranks;
    const int rem = n_root % nranks;
    if (rank < rem) {
        count = base + 1;
        start = rank * (base + 1);
    } else {
        count = base;
        start = rem * (base + 1) + (rank - rem) * base;
    }
}

// Build the local slice of a 2D square-grid unstructured mesh owned by this rank.
// This represents computation on arbitrarily-shaped geometries, decomposed by
// contiguous row blocks for distributed-memory (MPI) parallelism.
void buildSquare2D(World& world, const int n_elems_root, int rank, int nranks) {
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_elems_root = n_elems_root;
    computeRowPartition(n_elems_root, nranks, rank, world.local_x_start, world.local_x_count);
    world.local_n = static_cast<size_t>(world.local_x_count) * n_elems_root;

    world.elements_static.resize(world.local_n);
    world.elements_dynamic.resize(world.local_n);

    const int last = n_elems_root - 1;
    const int local_x_start = world.local_x_start;
    const int local_x_count = world.local_x_count;

    // Build connectivity: each element connects to its neighbors in the 2D grid.
    // Neighbor offset order matches the original single-rank implementation
    // ({+x, -x, +y, -y}) so per-element flux accumulation order (and thus the
    // resulting floating-point values) is identical to the sequential version.
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < local_x_count; ++r) {
        const int x = local_x_start + r;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t idx = static_cast<size_t>(r) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            world.elements_dynamic[idx].current_energy = 0.0;
            world.elements_dynamic[idx].total_flux = 0.0;

            // Neighbor at (x+1, y)
            if (x + 1 <= last) {
                idx_t ext = (r + 1 < local_x_count)
                                ? static_cast<idx_t>(r + 2) * n_elems_root + y
                                : static_cast<idx_t>(local_x_count + 1) * n_elems_root + y;
                elem.connected_idx[elem.num_connections] = ext;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            // Neighbor at (x-1, y)
            if (x - 1 >= 0) {
                idx_t ext = (r > 0) ? static_cast<idx_t>(r) * n_elems_root + y
                                    : static_cast<idx_t>(y);
                elem.connected_idx[elem.num_connections] = ext;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            // Neighbor at (x, y+1)
            if (y + 1 <= last) {
                idx_t ext = static_cast<idx_t>(r + 1) * n_elems_root + (y + 1);
                elem.connected_idx[elem.num_connections] = ext;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
            // Neighbor at (x, y-1)
            if (y - 1 >= 0) {
                idx_t ext = static_cast<idx_t>(r + 1) * n_elems_root + (y - 1);
                elem.connected_idx[elem.num_connections] = ext;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }

            // Set corner elements as inflow/outflow to create interesting dynamics
            if (x == 0 && y == 0) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if (x == 0 && y == last) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == 0) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else if (x == last && y == last) {
                elem.material_idx = INFLOW_MAT_ID;
            }
        }
    }
}

// GPU kernel: update all locally-owned elements for one iteration.
// `energy_ext` holds the extended-local current_energy values (ghosts + local),
// `energy_next_mid` points at the local (non-ghost) portion of the next buffer.
__global__ void updateElementsKernel(const ElementStatic* __restrict__ static_local,
                                      const val_t* __restrict__ energy_ext,
                                      const val_t* __restrict__ total_flux_cur,
                                      val_t* __restrict__ energy_next_mid,
                                      val_t* __restrict__ total_flux_next,
                                      const Material* __restrict__ materials, size_t local_n,
                                      uint64_t n_root) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= local_n) return;

    const ElementStatic& es = static_local[i];
    const Material& mat = materials[es.material_idx];
    const val_t self_energy = energy_ext[i + n_root];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (same accumulation order as the
    // sequential reference implementation)
    for (idx_t j = 0; j < es.num_connections; ++j) {
        const val_t neighbor_energy = energy_ext[static_cast<size_t>(es.connected_idx[j])];
        total_flux += (neighbor_energy - self_energy) * mat.transfer_coeff * es.connected_flux[j] * 0.25;
    }

    energy_next_mid[i] = self_energy + total_flux;
    total_flux_next[i] = total_flux_cur[i] + fabs(total_flux);
}

// Run the distributed simulation for n_iters iterations.
//
// Parallelization strategy (hybrid MPI + OpenMP + CUDA):
//  - MPI decomposes the mesh by contiguous row blocks across ranks/nodes and
//    exchanges only the (small) boundary rows between neighboring ranks each
//    iteration (halo exchange), giving near-linear scalability across an
//    accelerator cluster.
//  - Each rank offloads its local element update to its own GPU via a CUDA
//    kernel, double-buffered so no per-iteration allocation is needed.
//  - OpenMP parallelizes host-side data preparation/reduction work (mesh
//    construction, hashing, validation, and packing) across CPU cores.
void runSimulation(World& world, const int n_iters, MPI_Comm comm, int rank) {
    const size_t local_n = world.local_n;
    const int n_root = world.n_elems_root;

    if (local_n == 0) {
        // This rank owns no rows (only possible if nranks > n_root); nothing to do.
        return;
    }

    const size_t ext_n = local_n + 2 * static_cast<size_t>(n_root);

    ElementStatic* d_static = nullptr;
    Material* d_materials = nullptr;
    val_t* d_energy_a = nullptr;
    val_t* d_energy_b = nullptr;
    val_t* d_flux_a = nullptr;
    val_t* d_flux_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_static, local_n * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_energy_a, ext_n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_energy_b, ext_n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_a, local_n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_flux_b, local_n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data(), local_n * sizeof(ElementStatic),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                           world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_energy_a, 0, ext_n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_energy_b, 0, ext_n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_flux_a, 0, local_n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_flux_b, 0, local_n * sizeof(val_t)));

    const bool has_left = world.local_x_start > 0;
    const bool has_right = (world.local_x_start + world.local_x_count) < n_root;
    const int left_rank = rank - 1;
    const int right_rank = rank + 1;
    constexpr int TAG_TO_TOP = 1;     // a rank sending its first row up to a lower-index neighbor
    constexpr int TAG_TO_BOTTOM = 2;  // a rank sending its last row down to a higher-index neighbor

    std::vector<val_t> send_first(n_root), send_last(n_root);
    std::vector<val_t> recv_below(n_root), recv_above(n_root);

    val_t* d_energy_cur = d_energy_a;
    val_t* d_energy_next = d_energy_b;
    val_t* d_flux_cur = d_flux_a;
    val_t* d_flux_next = d_flux_b;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int threads = 256;
    const int blocks = static_cast<int>((local_n + threads - 1) / threads);
    const size_t last_row_offset = n_root + (static_cast<size_t>(world.local_x_count) - 1) * n_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        // --- Halo exchange: refresh ghost rows of d_energy_cur from neighbors ---
        if (has_left) {
            CUDA_CHECK(cudaMemcpyAsync(send_first.data(), d_energy_cur + n_root,
                                        n_root * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
        }
        if (has_right) {
            CUDA_CHECK(cudaMemcpyAsync(send_last.data(), d_energy_cur + last_row_offset,
                                        n_root * sizeof(val_t), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Request reqs[4];
        int nreq = 0;
        if (has_left) {
            MPI_Isend(send_first.data(), n_root, MPI_DOUBLE, left_rank, TAG_TO_BOTTOM, comm, &reqs[nreq++]);
            MPI_Irecv(recv_below.data(), n_root, MPI_DOUBLE, left_rank, TAG_TO_TOP, comm, &reqs[nreq++]);
        }
        if (has_right) {
            MPI_Isend(send_last.data(), n_root, MPI_DOUBLE, right_rank, TAG_TO_TOP, comm, &reqs[nreq++]);
            MPI_Irecv(recv_above.data(), n_root, MPI_DOUBLE, right_rank, TAG_TO_BOTTOM, comm, &reqs[nreq++]);
        }
        if (nreq > 0) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        if (has_left) {
            CUDA_CHECK(cudaMemcpyAsync(d_energy_cur, recv_below.data(), n_root * sizeof(val_t),
                                        cudaMemcpyHostToDevice, stream));
        }
        if (has_right) {
            CUDA_CHECK(cudaMemcpyAsync(d_energy_cur + n_root + local_n, recv_above.data(),
                                        n_root * sizeof(val_t), cudaMemcpyHostToDevice, stream));
        }

        // --- Compute new state for locally owned elements on the GPU ---
        updateElementsKernel<<<blocks, threads, 0, stream>>>(
            d_static, d_energy_cur, d_flux_cur, d_energy_next + n_root, d_flux_next, d_materials,
            local_n, static_cast<uint64_t>(n_root));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::swap(d_energy_cur, d_energy_next);
        std::swap(d_flux_cur, d_flux_next);
    }

    std::vector<val_t> host_energy(local_n), host_flux(local_n);
    CUDA_CHECK(cudaMemcpy(host_energy.data(), d_energy_cur + n_root, local_n * sizeof(val_t),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_flux.data(), d_flux_cur, local_n * sizeof(val_t), cudaMemcpyDeviceToHost));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_n; ++i) {
        world.elements_dynamic[i].current_energy = host_energy[i];
        world.elements_dynamic[i].total_flux = host_flux[i];
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_energy_a));
    CUDA_CHECK(cudaFree(d_energy_b));
    CUDA_CHECK(cudaFree(d_flux_a));
    CUDA_CHECK(cudaFree(d_flux_b));
}

// Validate simulation results (reduced across all MPI ranks).
bool validateResults(const World& world, MPI_Comm comm, int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for schedule(static) reduction(+:energy_sum, flux_sum) \
        reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const ElementDynamic& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    val_t g_energy_sum = 0.0, g_flux_sum = 0.0, g_energy_max = 0.0, g_energy_min = 0.0;
    MPI_Allreduce(&energy_sum, &g_energy_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&flux_sum, &g_flux_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&energy_max, &g_energy_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&energy_min, &g_energy_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", g_energy_sum);
        printf("  Flux sum: %.2f\n", g_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", g_energy_min, g_energy_max);
    }

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    bool valid = true;

    if (!std::isfinite(g_energy_sum)) {
        if (rank == 0) printf("  ERROR: Energy sum is not finite\n");
        valid = false;
    }

    if (rank == 0 && std::abs(g_energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }

    if (!std::isfinite(g_flux_sum)) {
        if (rank == 0) printf("  ERROR: Flux sum is not finite\n");
        valid = false;
    }

    if (!std::isfinite(g_energy_max) || !std::isfinite(g_energy_min)) {
        if (rank == 0) printf("  ERROR: Energy extrema are not finite\n");
        valid = false;
    }

    if (rank == 0 && valid) {
        printf("  Validation: PASSED\n");
    }
    return valid;
}

// Compute a simple hash of the results for verification, reduced across all
// MPI ranks. Each per-element term is XORed independently (using its global
// index), so combining partial per-rank/per-thread hashes with XOR yields
// the same result as the original sequential computation regardless of order.
uint64_t computeHash(const World& world, MPI_Comm comm) {
    uint64_t local_hash = 0;
    const size_t global_offset = static_cast<size_t>(world.local_x_start) * world.n_elems_root;
    const auto& elements = world.elements_dynamic;

    #pragma omp parallel reduction(^:local_hash)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < elements.size(); ++i) {
            const size_t gi = global_offset + i;
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
            local_hash ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, comm);
    return global_hash;
}

// Gather the full, globally-ordered current_energy array onto rank 0.
std::vector<val_t> gatherEnergyToRoot(const World& world, MPI_Comm comm, int rank, int nranks) {
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        int s = 0, c = 0;
        computeRowPartition(world.n_elems_root, nranks, r, s, c);
        counts[r] = c * world.n_elems_root;
        displs[r] = s * world.n_elems_root;
    }

    std::vector<val_t> local_energy(world.elements_dynamic.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_energy.size(); ++i) {
        local_energy[i] = world.elements_dynamic[i].current_energy;
    }

    std::vector<val_t> global_energy;
    if (rank == 0) {
        global_energy.resize(static_cast<size_t>(world.n_elems_root) * world.n_elems_root);
    }

    MPI_Gatherv(local_energy.data(), static_cast<int>(local_energy.size()), MPI_DOUBLE,
                rank == 0 ? global_energy.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, comm);
    return global_energy;
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

    // Map each MPI rank to a GPU on its node (round-robin over local ranks).
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(node_rank % device_count));
    }

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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    // Build the local (per-rank) slice of the unstructured mesh
    World world;
    buildSquare2D(world, n_elems_root, rank, nranks);

    if (rank == 0) {
        // Calculate memory usage (aggregate across all ranks)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, MPI_COMM_WORLD, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long global_duration_ms = 0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(global_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (global_duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (reduced across ranks)
    const uint64_t hash = computeHash(world, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData = gatherEnergyToRoot(world, MPI_COMM_WORLD, rank, nranks);
        if (rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    bool valid = true;
    if (validate) {
        valid = validateResults(world, MPI_COMM_WORLD, rank);
    }

    MPI_Finalize();

    if (validate && !valid) {
        return 1;
    }
    return 0;
}
