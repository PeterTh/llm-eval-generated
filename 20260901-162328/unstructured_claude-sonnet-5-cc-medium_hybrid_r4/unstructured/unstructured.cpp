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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cuda_err__ = (call);                                        \
        if (cuda_err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(cuda_err__));                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices (into local incl. halo array)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Distributed world state: this rank owns a contiguous slab of grid rows,
// plus (up to) one halo row on each side shared with neighboring ranks.
struct World {
    std::vector<Material> materials;

    // Host-side data: elements_static holds only the OWNED elements (row-major,
    // n_owned = local_rows * n_elems_root). elements_dynamic (when populated)
    // holds only the OWNED elements too (used for initial upload / final download).
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;

    // Device buffers
    Material* d_materials = nullptr;
    ElementStatic* d_elements_static = nullptr;   // size n_owned
    ElementDynamic* d_current = nullptr;          // size total_local_elems (incl. halo)
    ElementDynamic* d_swap = nullptr;              // size total_local_elems (incl. halo)

    // Pinned host staging buffers for halo exchange (one row = n_elems_root elements)
    ElementDynamic* h_send_up = nullptr;
    ElementDynamic* h_recv_up = nullptr;
    ElementDynamic* h_send_down = nullptr;
    ElementDynamic* h_recv_down = nullptr;

    // Domain decomposition metadata
    int n_elems_root = 0;
    int row_start = 0;         // first global row owned by this rank
    int row_end = 0;           // one past last global row owned
    bool halo_top = false;     // true if a top halo row exists (row_start > 0)
    bool halo_bottom = false;  // true if a bottom halo row exists (row_end < n_elems_root)
    int local_rows = 0;        // row_end - row_start
    int total_local_rows = 0;  // local_rows + halo_top + halo_bottom
    int n_owned = 0;           // local_rows * n_elems_root
    int owned_offset = 0;      // (halo_top ? 1 : 0) * n_elems_root, offset of first owned elem
    int total_local_elems = 0; // total_local_rows * n_elems_root

    int rank = 0, num_ranks = 1;
    int up_rank = MPI_PROC_NULL;
    int down_rank = MPI_PROC_NULL;

    cudaStream_t stream = nullptr;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute this rank's row range decomposition of the global n_elems_root x n_elems_root grid.
void computeDecomposition(World& world, int n_elems_root) {
    if (n_elems_root < world.num_ranks) {
        if (world.rank == 0) {
            fprintf(stderr,
                    "Error: grid size (%d) must be >= number of MPI ranks (%d)\n",
                    n_elems_root, world.num_ranks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int base = n_elems_root / world.num_ranks;
    const int rem = n_elems_root % world.num_ranks;

    world.n_elems_root = n_elems_root;
    world.row_start = world.rank * base + std::min(world.rank, rem);
    const int my_rows = base + (world.rank < rem ? 1 : 0);
    world.row_end = world.row_start + my_rows;
    world.local_rows = my_rows;

    world.halo_top = (world.row_start > 0);
    world.halo_bottom = (world.row_end < n_elems_root);
    world.total_local_rows = world.local_rows + (world.halo_top ? 1 : 0) + (world.halo_bottom ? 1 : 0);

    world.n_owned = world.local_rows * n_elems_root;
    world.owned_offset = (world.halo_top ? 1 : 0) * n_elems_root;
    world.total_local_elems = world.total_local_rows * n_elems_root;

    world.up_rank = world.halo_top ? (world.rank - 1) : MPI_PROC_NULL;
    world.down_rank = world.halo_bottom ? (world.rank + 1) : MPI_PROC_NULL;
}

// Build a 2D square grid as an unstructured mesh (this rank's local slab only).
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root) {
    // Initialize materials (replicated on every rank)
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local (owned-only) static connectivity and local (incl. halo) dynamic state
    world.elements_static.resize(world.n_owned);
    world.elements_dynamic.assign(world.total_local_elems, ElementDynamic{0.0, 0.0});

    const int row_lo = world.row_start - (world.halo_top ? 1 : 0);  // first global row represented locally

    // Initialize all owned elements with default material and zero energy
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < world.n_owned; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (int x = world.row_start; x < world.row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int owned_i = (x - world.row_start) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[owned_i];

            // Connect to neighbors (down, up, right, left) -- same order as reference
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            idx_t nconn = 0;
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within global bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int local_row = nx - row_lo;
                    const idx_t neighbor_local_idx = static_cast<idx_t>(local_row) * n_elems_root + ny;
                    elem.connected_idx[nconn] = neighbor_local_idx;
                    elem.connected_flux[nconn] = 1.0;
                    ++nconn;
                }
            }
            elem.num_connections = nconn;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto setCornerMaterial = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= world.row_start && gx < world.row_end) {
            const int owned_i = (gx - world.row_start) * n_elems_root + gy;
            world.elements_static[owned_i].material_idx = mat_id;
        }
    };
    setCornerMaterial(0, 0, INFLOW_MAT_ID);
    setCornerMaterial(0, last, OUTFLOW_MAT_ID);
    setCornerMaterial(last, 0, OUTFLOW_MAT_ID);
    setCornerMaterial(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements (device-side, used by the CUDA kernel)
__device__ inline val_t computeFluxDevice(const Material& mat, const ElementDynamic& this_elem,
                                           val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel: update all OWNED elements for one simulation iteration.
__global__ void computeStepKernel(const ElementStatic* __restrict__ elem_static,
                                   const ElementDynamic* __restrict__ current,
                                   ElementDynamic* __restrict__ next,
                                   const Material* __restrict__ materials,
                                   int owned_offset, int n_owned) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_owned) return;

    const int local_idx = owned_offset + i;
    const ElementStatic es = elem_static[i];
    const ElementDynamic elem_dyn = current[local_idx];
    const Material mat = materials[es.material_idx];

    val_t total_flux = mat.external_flow;

    #pragma unroll
    for (idx_t j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j >= es.num_connections) break;
        const ElementDynamic neighbor_dyn = current[es.connected_idx[j]];
        total_flux += computeFluxDevice(mat, elem_dyn, es.connected_flux[j], neighbor_dyn);
    }

    ElementDynamic out;
    out.current_energy = elem_dyn.current_energy + total_flux;
    out.total_flux = elem_dyn.total_flux + fabs(total_flux);
    next[local_idx] = out;
}

// Allocate device memory and upload static connectivity + initial state.
void setupDevice(World& world) {
    CUDA_CHECK(cudaStreamCreate(&world.stream));

    CUDA_CHECK(cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpyAsync(world.d_materials, world.materials.data(),
                                world.materials.size() * sizeof(Material),
                                cudaMemcpyHostToDevice, world.stream));

    CUDA_CHECK(cudaMalloc(&world.d_elements_static, world.n_owned * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMemcpyAsync(world.d_elements_static, world.elements_static.data(),
                                world.n_owned * sizeof(ElementStatic),
                                cudaMemcpyHostToDevice, world.stream));

    CUDA_CHECK(cudaMalloc(&world.d_current, world.total_local_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&world.d_swap, world.total_local_elems * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemcpyAsync(world.d_current, world.elements_dynamic.data(),
                                world.total_local_elems * sizeof(ElementDynamic),
                                cudaMemcpyHostToDevice, world.stream));
    CUDA_CHECK(cudaMemcpyAsync(world.d_swap, world.elements_dynamic.data(),
                                world.total_local_elems * sizeof(ElementDynamic),
                                cudaMemcpyHostToDevice, world.stream));

    const size_t row_bytes = static_cast<size_t>(world.n_elems_root) * sizeof(ElementDynamic);
    if (world.halo_top) {
        CUDA_CHECK(cudaHostAlloc(&world.h_send_up, row_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&world.h_recv_up, row_bytes, cudaHostAllocDefault));
    }
    if (world.halo_bottom) {
        CUDA_CHECK(cudaHostAlloc(&world.h_send_down, row_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&world.h_recv_down, row_bytes, cudaHostAllocDefault));
    }

    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

void teardownDevice(World& world) {
    cudaFree(world.d_materials);
    cudaFree(world.d_elements_static);
    cudaFree(world.d_current);
    cudaFree(world.d_swap);
    if (world.h_send_up) cudaFreeHost(world.h_send_up);
    if (world.h_recv_up) cudaFreeHost(world.h_recv_up);
    if (world.h_send_down) cudaFreeHost(world.h_send_down);
    if (world.h_recv_down) cudaFreeHost(world.h_recv_down);
    cudaStreamDestroy(world.stream);
}

// Element offset (into the local incl.-halo array) of the given owned row index (0-based
// within this rank's owned rows).
inline size_t owned_row_offset(const World& world, int owned_row) {
    const int local_row = (world.halo_top ? 1 : 0) + owned_row;
    return static_cast<size_t>(local_row) * world.n_elems_root;
}

// Exchange halo rows with neighboring MPI ranks (updates world.d_current's halo rows).
void exchangeHalo(World& world) {
    const size_t row_elems = static_cast<size_t>(world.n_elems_root);
    const size_t row_bytes = row_elems * sizeof(ElementDynamic);

    // Stage boundary rows from device to pinned host buffers
    if (world.halo_top) {
        ElementDynamic* src = world.d_current + owned_row_offset(world, 0);
        CUDA_CHECK(cudaMemcpyAsync(world.h_send_up, src, row_bytes, cudaMemcpyDeviceToHost, world.stream));
    }
    if (world.halo_bottom) {
        ElementDynamic* src = world.d_current + owned_row_offset(world, world.local_rows - 1);
        CUDA_CHECK(cudaMemcpyAsync(world.h_send_down, src, row_bytes, cudaMemcpyDeviceToHost, world.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(world.stream));

    MPI_Request reqs[4];
    int nreq = 0;
    if (world.halo_top) {
        MPI_Irecv(world.h_recv_up, static_cast<int>(row_elems * 2), MPI_DOUBLE, world.up_rank, 1,
                  MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(world.h_send_up, static_cast<int>(row_elems * 2), MPI_DOUBLE, world.up_rank, 0,
                  MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (world.halo_bottom) {
        MPI_Irecv(world.h_recv_down, static_cast<int>(row_elems * 2), MPI_DOUBLE, world.down_rank, 0,
                  MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(world.h_send_down, static_cast<int>(row_elems * 2), MPI_DOUBLE, world.down_rank, 1,
                  MPI_COMM_WORLD, &reqs[nreq++]);
    }
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Copy received halo rows back to device
    if (world.halo_top) {
        ElementDynamic* dst = world.d_current;  // local row 0 is the top halo row
        CUDA_CHECK(cudaMemcpyAsync(dst, world.h_recv_up, row_bytes, cudaMemcpyHostToDevice, world.stream));
    }
    if (world.halo_bottom) {
        ElementDynamic* dst = world.d_current + static_cast<size_t>(world.total_local_rows - 1) * row_elems;
        CUDA_CHECK(cudaMemcpyAsync(dst, world.h_recv_down, row_bytes, cudaMemcpyHostToDevice, world.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(world.stream));
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    constexpr int threads = 256;
    const int blocks = (world.n_owned + threads - 1) / threads;

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalo(world);

        if (world.n_owned > 0) {
            computeStepKernel<<<blocks, threads, 0, world.stream>>>(
                world.d_elements_static, world.d_current, world.d_swap,
                world.d_materials, world.owned_offset, world.n_owned);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(world.stream));

        std::swap(world.d_current, world.d_swap);
    }
}

// Validate simulation results (global, reduced across all MPI ranks)
bool validateResults(World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    #pragma omp parallel for schedule(static) reduction(+:local_energy_sum, local_flux_sum) \
        reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (int i = 0; i < world.n_owned; ++i) {
        const ElementDynamic& elem = world.elements_dynamic[i];
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world.rank != 0) {
        return true;  // Only rank 0 reports/validates; result is broadcast-equivalent since it aborts on rank 0.
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
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

// Compute a simple hash of the results for verification (global, reduced across all MPI ranks).
// Combines contributions using XOR, so per-rank partial results can be combined in any order.
uint64_t computeHash(World& world) {
    uint64_t local_hash = 0;
    const uint64_t global_offset = static_cast<uint64_t>(world.row_start) * world.n_elems_root;

    #pragma omp parallel for schedule(static) reduction(^:local_hash)
    for (int i = 0; i < world.n_owned; ++i) {
        const uint64_t global_i = global_offset + static_cast<uint64_t>(i);
        const ElementDynamic& elem = world.elements_dynamic[i];
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elem.current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elem.total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.num_ranks);

    // Bind this rank to a GPU (round-robin across available devices on the node)
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        if (world.rank == 0) fprintf(stderr, "Error: no CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(world.rank % device_count));

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
            if (world.rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;

    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs visible per node: %d, OpenMP threads/rank: %d\n",
               world.num_ranks, device_count, omp_get_max_threads());
        printf("\n");
    }

    computeDecomposition(world, n_elems_root);

    // Build the unstructured mesh (this rank's local slab)
    if (world.rank == 0) printf("Building unstructured mesh...\n");
    buildSquare2D(world, n_elems_root);
    setupDevice(world);

    // Calculate memory usage (aggregated across all ranks)
    const size_t local_static_mem = static_cast<size_t>(world.n_owned) * sizeof(ElementStatic);
    const size_t local_dynamic_mem = static_cast<size_t>(world.total_local_elems) * sizeof(ElementDynamic) * 2;
    size_t static_mem = 0, dynamic_mem = 0;
    MPI_Reduce(&local_static_mem, &static_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world.rank == 0) {
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (world.rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Allreduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Download owned elements back to host for validation/hashing/printing
    world.elements_dynamic.resize(world.n_owned);
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), world.d_current + world.owned_offset,
                           static_cast<size_t>(world.n_owned) * sizeof(ElementDynamic),
                           cudaMemcpyDeviceToHost));

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

    // Compute hash for verification (reduced to rank 0)
    const uint64_t hash = computeHash(world);
    if (world.rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather full global array on rank 0)
    if (printResults) {
        std::vector<double> localEnergy(world.n_owned);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < world.n_owned; ++i) {
            localEnergy[i] = world.elements_dynamic[i].current_energy;
        }

        std::vector<int> counts, displs;
        if (world.rank == 0) {
            counts.resize(world.num_ranks);
            displs.resize(world.num_ranks);
        }
        MPI_Gather(&world.n_owned, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> energyData;
        if (world.rank == 0) {
            int offset = 0;
            for (int r = 0; r < world.num_ranks; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
            energyData.resize(offset);
        }
        MPI_Gatherv(localEnergy.data(), world.n_owned, MPI_DOUBLE,
                    world.rank == 0 ? energyData.data() : nullptr,
                    world.rank == 0 ? counts.data() : nullptr,
                    world.rank == 0 ? displs.data() : nullptr,
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

    teardownDevice(world);

    int exit_code = 0;
    if (world.rank == 0 && validate && !valid) {
        exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
