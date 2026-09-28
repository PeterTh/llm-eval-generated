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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialize all elements with default material and zero energy
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
__device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                                    val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update one element per thread: gather flux from connected elements and
// integrate. The device only holds this rank's slice: elements_static covers
// the owned range [elem_begin, elem_end), elements_in/out cover it plus one
// halo row on each side starting at global index alloc_begin. Connectivity
// indices are global and are rebased onto the slice.
__global__ void updateKernel(const ElementStatic* __restrict__ elements_static,
                             const ElementDynamic* __restrict__ elements_in,
                             ElementDynamic* __restrict__ elements_out,
                             const Material* __restrict__ materials,
                             const size_t elem_begin, const size_t elem_end,
                             const size_t alloc_begin) {
    const size_t i = elem_begin + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= elem_end) return;

    const ElementStatic& elem_static = elements_static[i - elem_begin];
    const ElementDynamic elem_dyn = elements_in[i - alloc_begin];
    const Material mat = materials[elem_static.material_idx];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        const idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic neighbor_dyn = elements_in[neighbor_idx - alloc_begin];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }

    // Update element state
    ElementDynamic& elem_write = elements_out[i - alloc_begin];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Row-block domain decomposition: rank owns grid rows [row_begin, row_end).
struct Decomposition {
    int rank = 0;
    int size = 1;
    int row_begin = 0;
    int row_end = 0;
    size_t elem_begin = 0;
    size_t elem_end = 0;
};

Decomposition decompose(const int n_elems_root) {
    Decomposition d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    const int rows_per_rank = n_elems_root / d.size;
    const int remainder = n_elems_root % d.size;
    d.row_begin = d.rank * rows_per_rank + std::min(d.rank, remainder);
    d.row_end = d.row_begin + rows_per_rank + (d.rank < remainder ? 1 : 0);
    d.elem_begin = (size_t)d.row_begin * n_elems_root;
    d.elem_end = (size_t)d.row_end * n_elems_root;
    return d;
}

// Run simulation for n_iters iterations. Each rank advances its row block on
// its GPU; boundary rows are exchanged with neighboring ranks each iteration.
void runSimulation(World& world, const int n_iters, const Decomposition& d,
                   const int n_elems_root) {
    const size_t row_bytes = (size_t)n_elems_root * sizeof(ElementDynamic);
    const size_t n_owned = d.elem_end - d.elem_begin;

    // Each rank only stores its owned slice on the device, plus one halo row
    // on each side of the dynamic state, so memory and transfer volume scale
    // with 1/n_ranks.
    const size_t alloc_begin = d.elem_begin - (d.row_begin > 0 ? (size_t)n_elems_root : 0);
    const size_t alloc_end = d.elem_end + (d.row_end < n_elems_root && n_owned > 0
                                               ? (size_t)n_elems_root : 0);
    const size_t n_alloc = alloc_end - alloc_begin;

    ElementStatic* d_static = nullptr;
    Material* d_materials = nullptr;
    ElementDynamic* d_dyn = nullptr;
    ElementDynamic* d_dyn_swap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_static, std::max(n_owned, (size_t)1) * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&d_dyn, std::max(n_alloc, (size_t)1) * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_dyn_swap, std::max(n_alloc, (size_t)1) * sizeof(ElementDynamic)));

    if (n_owned > 0) {
        CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data() + d.elem_begin,
                              n_owned * sizeof(ElementStatic), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dyn, world.elements_dynamic.data() + alloc_begin,
                              n_alloc * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    // Pinned staging buffers for halo rows: [0] = row sent down/received from
    // below (row_begin - 1), [1] = row sent up/received from above (row_end).
    ElementDynamic* h_send = nullptr;
    ElementDynamic* h_recv = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send, 2 * row_bytes));
    CUDA_CHECK(cudaMallocHost(&h_recv, 2 * row_bytes));

    // Ranks with no rows (size > n_elems_root) sit out the halo exchange;
    // empty blocks only occur at the tail of the rank range, so a non-empty
    // rank's index neighbors are always its rank neighbors.
    const bool has_work = n_owned > 0;
    const int prev_rank = (has_work && d.row_begin > 0) ? d.rank - 1 : MPI_PROC_NULL;
    const int next_rank = (has_work && d.row_end < n_elems_root) ? d.rank + 1 : MPI_PROC_NULL;

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = has_work ? (int)((n_owned + BLOCK_SIZE - 1) / BLOCK_SIZE) : 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Refresh halo rows with neighbors' results from the previous
        // iteration (initial state is globally consistent on every rank).
        if (iter > 0 && d.size > 1) {
            if (has_work) {
                CUDA_CHECK(cudaMemcpy(h_send, d_dyn + (d.elem_begin - alloc_begin),
                                      row_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_send + n_elems_root,
                                      d_dyn + (d.elem_end - n_elems_root - alloc_begin),
                                      row_bytes, cudaMemcpyDeviceToHost));
            }
            MPI_Request reqs[4];
            MPI_Irecv(h_recv, (int)row_bytes, MPI_BYTE, prev_rank, 1, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(h_recv + n_elems_root, (int)row_bytes, MPI_BYTE, next_rank, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(h_send, (int)row_bytes, MPI_BYTE, prev_rank, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Isend(h_send + n_elems_root, (int)row_bytes, MPI_BYTE, next_rank, 1, MPI_COMM_WORLD, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
            if (prev_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_dyn, h_recv, row_bytes, cudaMemcpyHostToDevice));
            }
            if (next_rank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_dyn + (d.elem_end - alloc_begin),
                                      h_recv + n_elems_root, row_bytes,
                                      cudaMemcpyHostToDevice));
            }
        }

        // Update all owned elements
        if (has_work) {
            updateKernel<<<grid_size, BLOCK_SIZE>>>(d_static, d_dyn, d_dyn_swap,
                                                    d_materials, d.elem_begin, d.elem_end,
                                                    alloc_begin);
            CUDA_CHECK(cudaGetLastError());
        }

        // Swap buffers
        std::swap(d_dyn, d_dyn_swap);
    }

    // Gather all owned blocks onto every rank so the final state matches the
    // serial code's elements_dynamic everywhere.
    if (has_work) {
        CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data() + d.elem_begin,
                              d_dyn + (d.elem_begin - alloc_begin),
                              n_owned * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    }
    if (d.size > 1) {
        std::vector<int> counts(d.size), displs(d.size);
        const int rows_per_rank = n_elems_root / d.size;
        const int remainder = n_elems_root % d.size;
        for (int r = 0; r < d.size; ++r) {
            const int rb = r * rows_per_rank + std::min(r, remainder);
            const int re = rb + rows_per_rank + (r < remainder ? 1 : 0);
            counts[r] = (int)((size_t)(re - rb) * n_elems_root * sizeof(ElementDynamic));
            displs[r] = (int)((size_t)rb * n_elems_root * sizeof(ElementDynamic));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, world.elements_dynamic.data(),
                       counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_materials));
    CUDA_CHECK(cudaFree(d_dyn));
    CUDA_CHECK(cudaFree(d_dyn_swap));
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

// Compute a simple hash of the results for verification.
// XOR-combining is order-independent, so the loop parallelizes bit-exactly.
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    const size_t n = elements.size();
    #pragma omp parallel for schedule(static) reduction(^:hash)
    for (size_t i = 0; i < n; ++i) {
        // Simple hash combining energy and flux values
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
            int rank0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank0);
            if (rank0 == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            int rank0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank0);
            if (rank0 == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const Decomposition d = decompose(n_elems_root);
    const bool is_root = (d.rank == 0);

    // Bind each rank to a GPU by node-local rank
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank,
                            MPI_INFO_NULL, &local_comm);
        int local_rank;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
        int n_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_devices));
        if (n_devices == 0) {
            fprintf(stderr, "No CUDA devices available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA GPUs\n",
               d.size, omp_get_max_threads());
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root);

    if (is_root) {
        // Calculate memory usage
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, d, n_elems_root);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (is_root) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world.elements_dynamic.size());
            for (const auto& elem : world.elements_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    int exit_code = 0;
    if (validate && is_root) {
        bool valid = validateResults(world);
        if (!valid) {
            exit_code = 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
