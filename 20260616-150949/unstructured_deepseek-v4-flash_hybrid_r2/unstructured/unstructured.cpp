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

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (local)
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

// MPI decomposition globals
static int mpi_rank = 0;
static int mpi_size = 1;
static int local_nrows = 0;
static int first_global_row = 0;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Build a 2D square grid as an unstructured mesh
// MPI-aware: each rank builds its local portion of the grid
// The dynamic array includes halo cells for boundary exchange:
//   Row 0: top halo (n_elems_root elements)
//   Rows 1..local_nrows: local elements
//   Row local_nrows+1: bottom halo (n_elems_root elements)
// ElementStatic uses local indices pointing into this layout
void buildSquare2D(World& world, const int n_elems_root) {
    // Compute domain decomposition via 1D row partitioning
    local_nrows = n_elems_root / mpi_size;
    const int remainder = n_elems_root % mpi_size;
    if (mpi_rank < remainder) local_nrows++;

    first_global_row = 0;
    for (int r = 0; r < mpi_rank; ++r) {
        int rows_r = n_elems_root / mpi_size;
        if (r < remainder) rows_r++;
        first_global_row += rows_r;
    }
    const int last_global_row = first_global_row + local_nrows;

    const int n_local_elems = local_nrows * n_elems_root;
    const int total_with_halo = (local_nrows + 2) * n_elems_root;

    // Initialize materials (replicated on all ranks)
    world.materials = {Material{0.8, 0.0}, Material{0.8, 0.5}, Material{0.8, -0.5}};

    // Allocate elements
    world.elements_static.resize(n_local_elems);
    world.elements_dynamic.resize(total_with_halo);
    world.elements_dynamic_swap.resize(total_with_halo);

    // Initialize all local elements with default material and zero energy
    #pragma omp parallel for
    for (int i = 0; i < n_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }

    // Initialize dynamic data (all entries including halo cells)
    #pragma omp parallel for
    for (int i = 0; i < total_with_halo; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity with local indices that include halo cells
    for (int lr = 0; lr < local_nrows; ++lr) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = lr * n_elems_root + y;
            const int global_row = first_global_row + lr;
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            // Dynamic array index = (local_row + 1) * n_elems_root + y for local element
            // Up: (global_row - 1, y)
            if (global_row > 0) {
                elem.connected_idx[elem.num_connections] =
                    (global_row - 1 >= first_global_row)
                    ? (lr * n_elems_root + y)          // local: row above
                    : y;                                // top halo (row 0 of dynamic array)
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }

            // Down: (global_row + 1, y)
            if (global_row < n_elems_root - 1) {
                elem.connected_idx[elem.num_connections] =
                    (global_row + 1 < last_global_row)
                    ? ((lr + 2) * n_elems_root + y)    // local: row below
                    : ((local_nrows + 1) * n_elems_root + y);  // bottom halo
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }

            // Left: (global_row, y - 1)
            if (y > 0) {
                elem.connected_idx[elem.num_connections] = (lr + 1) * n_elems_root + (y - 1);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }

            // Right: (global_row, y + 1)
            if (y < n_elems_root - 1) {
                elem.connected_idx[elem.num_connections] = (lr + 1) * n_elems_root + (y + 1);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow (on the rank that owns them)
    const int last = n_elems_root - 1;
    if (first_global_row <= 0 && first_global_row + local_nrows > 0) {
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    if (first_global_row <= last && first_global_row + local_nrows > last) {
        const int cr = last - first_global_row;
        world.elements_static[cr * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[cr * n_elems_root + last].material_idx = INFLOW_MAT_ID;
    }
}

// Compute energy flux between two elements (host and device)
__host__ __device__ inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                                             val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// CUDA kernel: update all local elements on GPU
__global__ void updateElementsKernel(
    const ElementStatic* __restrict__ elements_static,
    const ElementDynamic* __restrict__ elements_dynamic,
    ElementDynamic* __restrict__ elements_dynamic_swap,
    const Material* __restrict__ materials,
    int n_local_elems,
    int local_offset)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= n_local_elems) return;

    const ElementStatic& elem_static = elements_static[elem_idx];
    int dyn_idx = local_offset + elem_idx;
    const ElementDynamic& elem_dyn = elements_dynamic[dyn_idx];
    const Material& mat = materials[elem_static.material_idx];

    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = elements_dynamic[neighbor_idx];
        total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
    }

    // Update element state in swap buffer
    ElementDynamic& elem_write = elements_dynamic_swap[dyn_idx];
    elem_write.current_energy = elem_dyn.current_energy + total_flux;
    elem_write.total_flux = elem_dyn.total_flux + fabs(total_flux);
}

// Run simulation using MPI + CUDA + OpenMP hybrid approach
void runSimulation(World& world, const int n_iters, int n_elems_root) {
    const int n_local_elems = local_nrows * n_elems_root;
    const int total_with_halo = (local_nrows + 2) * n_elems_root;
    const int local_offset = n_elems_root;  // first local element in dynamic array

    if (n_local_elems == 0) return;

    // Allocate device memory
    ElementStatic *d_static = nullptr;
    ElementDynamic *d_dyn[2] = {nullptr, nullptr};
    Material *d_materials = nullptr;

    CUDA_CHECK(cudaMalloc(&d_static, n_local_elems * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&d_dyn[0], total_with_halo * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_dyn[1], total_with_halo * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&d_materials, 3 * sizeof(Material)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_static, world.elements_static.data(),
                          n_local_elems * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dyn[0], world.elements_dynamic.data(),
                          total_with_halo * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    // Both buffers start from same initial state (swap buffer is overwritten before first read)
    CUDA_CHECK(cudaMemcpy(d_dyn[1], world.elements_dynamic.data(),
                          total_with_halo * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, world.materials.data(),
                          3 * sizeof(Material), cudaMemcpyHostToDevice));

    // Host buffers for MPI halo exchange
    std::vector<ElementDynamic> send_above(n_elems_root);
    std::vector<ElementDynamic> send_below(n_elems_root);
    std::vector<ElementDynamic> recv_above(n_elems_root);
    std::vector<ElementDynamic> recv_below(n_elems_root);

    // CUDA kernel launch configuration
    const int block_size = 256;
    const int grid_size = (n_local_elems + block_size - 1) / block_size;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int cur = 0;  // index of current (read) buffer
    const size_t row_bytes = n_elems_root * sizeof(ElementDynamic);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Step 1: Copy halo boundary rows from device to host asynchronously
        // First local row -> send_above (to rank-1, becomes their bottom halo)
        CUDA_CHECK(cudaMemcpyAsync(send_above.data(), d_dyn[cur] + local_offset,
                                   row_bytes, cudaMemcpyDeviceToHost, stream));
        // Last local row -> send_below (to rank+1, becomes their top halo)
        CUDA_CHECK(cudaMemcpyAsync(send_below.data(), d_dyn[cur] + local_nrows * n_elems_root,
                                   row_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Step 2: Non-blocking MPI halo exchange
        MPI_Request req[4];
        int nreq = 0;

        if (mpi_rank > 0) {
            // Send first local row to rank-1, receive top halo from rank-1
            MPI_Isend(send_above.data(), row_bytes, MPI_BYTE,
                      mpi_rank - 1, 0, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Irecv(recv_above.data(), row_bytes, MPI_BYTE,
                      mpi_rank - 1, 1, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (mpi_rank < mpi_size - 1) {
            // Send last local row to rank+1, receive bottom halo from rank+1
            MPI_Isend(send_below.data(), row_bytes, MPI_BYTE,
                      mpi_rank + 1, 1, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Irecv(recv_below.data(), row_bytes, MPI_BYTE,
                      mpi_rank + 1, 0, MPI_COMM_WORLD, &req[nreq++]);
        }

        if (nreq > 0) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

        // Step 3: Copy received halo data from host to device
        if (mpi_rank > 0) {
            // Received from rank-1: their last row -> our top halo
            CUDA_CHECK(cudaMemcpyAsync(d_dyn[cur], recv_above.data(), row_bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (mpi_rank < mpi_size - 1) {
            // Received from rank+1: their first row -> our bottom halo
            CUDA_CHECK(cudaMemcpyAsync(d_dyn[cur] + (local_nrows + 1) * n_elems_root,
                                       recv_below.data(), row_bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Step 4: Launch CUDA kernel (read from d_dyn[cur], write to d_dyn[1-cur])
        updateElementsKernel<<<grid_size, block_size>>>(
            d_static, d_dyn[cur], d_dyn[1 - cur], d_materials, n_local_elems, local_offset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 5: Swap buffers for next iteration
        cur = 1 - cur;
    }

    // Copy results back to host (local elements only, compacted)
    CUDA_CHECK(cudaMemcpy(world.elements_dynamic.data(), d_dyn[cur] + local_offset,
                          n_local_elems * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));

    // Cleanup device resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_static));
    CUDA_CHECK(cudaFree(d_dyn[0]));
    CUDA_CHECK(cudaFree(d_dyn[1]));
    CUDA_CHECK(cudaFree(d_materials));
}

// Validate simulation results using MPI reductions for global aggregates
bool validateResults(const World& world, int n_elems_root) {
    const int n_local = local_nrows * n_elems_root;

    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    // OpenMP parallel reduction
    #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
                                     reduction(max:local_energy_max) \
                                     reduction(min:local_energy_min)
    for (int i = 0; i < n_local; ++i) {
        local_energy_sum += world.elements_dynamic[i].current_energy;
        local_flux_sum += world.elements_dynamic[i].total_flux;
        if (world.elements_dynamic[i].current_energy > local_energy_max)
            local_energy_max = world.elements_dynamic[i].current_energy;
        if (world.elements_dynamic[i].current_energy < local_energy_min)
            local_energy_min = world.elements_dynamic[i].current_energy;
    }

    // MPI reductions for global results
    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    }

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    bool valid = true;

    if (!std::isfinite(energy_sum)) {
        if (mpi_rank == 0) printf("  ERROR: Energy sum is not finite\n");
        valid = false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        if (mpi_rank == 0)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }

    if (!std::isfinite(flux_sum)) {
        if (mpi_rank == 0) printf("  ERROR: Flux sum is not finite\n");
        valid = false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        if (mpi_rank == 0) printf("  ERROR: Energy extrema are not finite\n");
        valid = false;
    }

    if (mpi_rank == 0) printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

// Compute a simple hash of the results for verification (MPI-aware)
uint64_t computeHash(const std::vector<ElementDynamic>& elements, int n_elems_root) {
    const int n_local = local_nrows * n_elems_root;
    uint64_t local_hash = 0;

    #pragma omp parallel for reduction(^:local_hash)
    for (int i = 0; i < n_local; ++i) {
        // Map local index to global index for deterministic hash
        int local_row = i / n_elems_root;
        int col = i % n_elems_root;
        uint64_t global_i = (uint64_t)(first_global_row + local_row) * n_elems_root + col;

        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        local_hash ^= (*e_ptr + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_i) * 0xbf58476d1ce4e5b9ULL;
    }

    // MPI XOR reduction across all ranks (XOR is associative and commutative)
    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse independently)
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

    // Setup CUDA device - round-robin assignment across MPI ranks
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices == 0) {
        fprintf(stderr, "ERROR: No CUDA devices available on rank %d\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(mpi_rank % n_devices);

    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh (each rank builds its local portion)
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // Calculate per-rank memory usage
    const int n_local_elems = local_nrows * n_elems_root;
    const size_t static_mem = (size_t)n_local_elems * sizeof(ElementStatic);
    const size_t dynamic_mem = (size_t)(local_nrows + 2) * n_elems_root * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;

    if (mpi_rank == 0) {
        printf("Memory usage per rank: %.2f MB\n", total_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation with MPI + CUDA
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    runSimulation(world, n_iters, n_elems_root);

    double end_time = MPI_Wtime();
    double local_duration = (end_time - start_time) * 1000.0;

    // Report max computation time across all ranks (wall-clock time)
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Computation time: %.2f ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (MPI XOR reduction)
    // First compact the vector to local elements only
    world.elements_dynamic.resize(n_local_elems);
    const uint64_t hash = computeHash(world.elements_dynamic, n_elems_root);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation (gather all data to rank 0)
    if (printResults) {
        // Compute per-rank element counts for Gatherv
        int base_rows = n_elems_root / mpi_size;
        int rem = n_elems_root % mpi_size;
        std::vector<int> counts(mpi_size), displs(mpi_size);
        int total = 0;
        for (int r = 0; r < mpi_size; ++r) {
            int n = ((r < rem) ? base_rows + 1 : base_rows) * n_elems_root;
            counts[r] = n;
            displs[r] = total;
            total += n;
        }

        std::vector<double> global_energy;
        if (mpi_rank == 0) global_energy.resize(total);

        std::vector<double> local_energy(n_local_elems);
        #pragma omp parallel for
        for (int i = 0; i < n_local_elems; ++i) {
            local_energy[i] = world.elements_dynamic[i].current_energy;
        }

        MPI_Gatherv(local_energy.data(), n_local_elems, MPI_DOUBLE,
                    global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            print_results(global_energy, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        bool valid = validateResults(world, n_elems_root);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
