// Hybrid MPI + OpenMP + CUDA parallelization.
//
// Decomposition: the global element index space is split into contiguous
// row-slabs of the 2D grid (elements are numbered idx = x * n_root + y, so a
// slab of rows is a contiguous index range). Each MPI rank owns one slab and
// drives one GPU (round-robin over the node's devices). Per iteration the
// boundary rows' energies are exchanged with neighbor ranks (halo exchange)
// while the CUDA kernel updates the slab interior; boundary rows are updated
// once the halos have arrived. OpenMP parallelizes host-side mesh
// construction and (de)serialization of results.
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Local slab of the 2D square grid, built as an unstructured mesh.
// Connectivity indices are stored as local indices into the halo-padded
// energy array: local = global - x0 * n_root + n_root (one ghost row of
// padding below and above the owned rows).
struct LocalWorld {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    long long n_owned;    // owned elements (my_rows * n_root)
    long long ghost;      // elements per ghost row (= n_root)
    int x0;               // first owned grid row
    int my_rows;          // number of owned grid rows
};

// Build the owned slab of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(LocalWorld& world, const int n_elems_root,
                   const int x0, const int my_rows) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.x0 = x0;
    world.my_rows = my_rows;
    world.ghost = n_elems_root;
    world.n_owned = static_cast<long long>(my_rows) * n_elems_root;
    world.elements_static.resize(world.n_owned);

    const long long local_base =
        static_cast<long long>(x0) * n_elems_root - n_elems_root;

    // Build connectivity: each element connects to its neighbors in 2D grid
#pragma omp parallel for schedule(static)
    for (long long li = 0; li < world.n_owned; ++li) {
        const int x = x0 + static_cast<int>(li / n_elems_root);
        const int y = static_cast<int>(li % n_elems_root);
        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const long long neighbor_global =
                    static_cast<long long>(nx) * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] =
                    static_cast<idx_t>(neighbor_global - local_base);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto set_material = [&](int gx, int gy, idx_t mat) {
        if (gx >= x0 && gx < x0 + my_rows) {
            world.elements_static[static_cast<long long>(gx - x0) * n_elems_root + gy]
                .material_idx = mat;
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Device-side element update. Connectivity is stored SoA / column-major so
// that neighboring threads access consecutive memory. The per-connection
// summation order matches the original scalar loop.
__global__ void stepKernel(const long long offset, const long long count,
                           const long long n_owned, const long long ghost,
                           const uint8_t* __restrict__ num_conn,
                           const uint32_t* __restrict__ conn_idx,
                           const val_t* __restrict__ conn_flux,
                           const val_t* __restrict__ transfer_coeff,
                           const val_t* __restrict__ external_flow,
                           const val_t* __restrict__ energy_in,
                           val_t* __restrict__ energy_out,
                           const val_t* __restrict__ flux_in,
                           val_t* __restrict__ flux_out) {
    const long long i =
        offset + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= offset + count) return;

    const val_t self_energy = energy_in[i + ghost];
    const val_t coeff = transfer_coeff[i];

    // Start with external flow
    val_t total_flux = external_flow[i];

    // Add flux from all connected elements
    const int nc = num_conn[i];
    for (int j = 0; j < nc; ++j) {
        const uint32_t nb = conn_idx[static_cast<long long>(j) * n_owned + i];
        const val_t cf = conn_flux[static_cast<long long>(j) * n_owned + i];
        total_flux += (energy_in[nb] - self_energy) * coeff * cf * 0.25;
    }

    // Update element state
    energy_out[i + ghost] = self_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Per-rank GPU simulation state
struct DeviceSim {
    long long n_owned = 0;
    long long ghost = 0;
    long long n_padded = 0;
    uint8_t* d_num_conn = nullptr;
    uint32_t* d_conn_idx = nullptr;
    val_t* d_conn_flux = nullptr;
    val_t* d_coeff = nullptr;
    val_t* d_ext = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    val_t* h_send_lo = nullptr;   // pinned halo staging buffers
    val_t* h_send_hi = nullptr;
    val_t* h_recv_lo = nullptr;
    val_t* h_recv_hi = nullptr;
    cudaStream_t s_comp = nullptr;
    cudaStream_t s_halo = nullptr;
};

// Upload the static mesh to the GPU in SoA layout and zero-init dynamic state
void setupDevice(const LocalWorld& world, DeviceSim& sim) {
    sim.n_owned = world.n_owned;
    sim.ghost = world.ghost;
    sim.n_padded = sim.n_owned + 2 * sim.ghost;
    if (sim.n_owned == 0) return;

    CUDA_CHECK(cudaStreamCreate(&sim.s_comp));
    CUDA_CHECK(cudaStreamCreate(&sim.s_halo));

    const long long n = sim.n_owned;
    std::vector<uint8_t> h_num_conn(n);
    std::vector<uint32_t> h_conn_idx(n * MAX_CONNECTIONS);
    std::vector<val_t> h_conn_flux(n * MAX_CONNECTIONS, 0.0);
    std::vector<val_t> h_coeff(n);
    std::vector<val_t> h_ext(n);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        const ElementStatic& es = world.elements_static[i];
        const Material& mat = world.materials[es.material_idx];
        h_num_conn[i] = static_cast<uint8_t>(es.num_connections);
        h_coeff[i] = mat.transfer_coeff;
        h_ext[i] = mat.external_flow;
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_conn_idx[j * n + i] = static_cast<uint32_t>(es.connected_idx[j]);
            h_conn_flux[j * n + i] = es.connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&sim.d_num_conn, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_conn_idx, n * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_conn_flux, n * MAX_CONNECTIONS * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_coeff, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_ext, n * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&sim.d_energy[b], sim.n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&sim.d_flux[b], n * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(sim.d_energy[b], 0, sim.n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(sim.d_flux[b], 0, n * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(sim.d_num_conn, h_num_conn.data(),
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_conn_idx, h_conn_idx.data(),
                          n * MAX_CONNECTIONS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_conn_flux, h_conn_flux.data(),
                          n * MAX_CONNECTIONS * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_coeff, h_coeff.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_ext, h_ext.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMallocHost(&sim.h_send_lo, sim.ghost * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_send_hi, sim.ghost * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recv_lo, sim.ghost * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recv_hi, sim.ghost * sizeof(val_t)));
}

constexpr int BLOCK_SIZE = 256;

static inline void launchStep(const DeviceSim& sim, long long offset,
                              long long count, const val_t* e_in,
                              val_t* e_out, const val_t* f_in, val_t* f_out,
                              cudaStream_t stream) {
    if (count <= 0) return;
    const int n_blocks = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stepKernel<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        offset, count, sim.n_owned, sim.ghost, sim.d_num_conn, sim.d_conn_idx,
        sim.d_conn_flux, sim.d_coeff, sim.d_ext, e_in, e_out, f_in, f_out);
}

// Run simulation for n_iters iterations. Halo exchange of boundary-row
// energies overlaps with the interior kernel each iteration.
void runSimulation(DeviceSim& sim, const int n_iters, const int rank_prev,
                   const int rank_next, MPI_Comm comm) {
    val_t* e_in = sim.d_energy[0];
    val_t* e_out = sim.d_energy[1];
    val_t* f_in = sim.d_flux[0];
    val_t* f_out = sim.d_flux[1];
    const long long g = sim.ghost;
    const long long n = sim.n_owned;
    // Boundary segments: first and last owned row (they read ghost cells)
    const long long lo_end = std::min(g, n);
    const long long hi_start = std::max(n - g, lo_end);

    for (int iter = 0; iter < n_iters; ++iter) {
        if (n > 0) {
            // Stage boundary-row energies for the halo exchange
            if (rank_prev != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sim.h_send_lo, e_in + g,
                                           g * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, sim.s_halo));
            }
            if (rank_next != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(sim.h_send_hi, e_in + n,
                                           g * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, sim.s_halo));
            }
            // Interior update does not touch ghost cells: overlap with comm
            launchStep(sim, lo_end, hi_start - lo_end, e_in, e_out, f_in,
                       f_out, sim.s_comp);
            CUDA_CHECK(cudaStreamSynchronize(sim.s_halo));
        }

        if (n > 0) {
            MPI_Sendrecv(sim.h_send_lo, static_cast<int>(g), MPI_DOUBLE,
                         rank_prev, 0, sim.h_recv_hi, static_cast<int>(g),
                         MPI_DOUBLE, rank_next, 0, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(sim.h_send_hi, static_cast<int>(g), MPI_DOUBLE,
                         rank_next, 1, sim.h_recv_lo, static_cast<int>(g),
                         MPI_DOUBLE, rank_prev, 1, comm, MPI_STATUS_IGNORE);
        }

        if (n > 0) {
            if (rank_prev != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_in, sim.h_recv_lo,
                                           g * sizeof(val_t),
                                           cudaMemcpyHostToDevice, sim.s_halo));
            }
            if (rank_next != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_in + g + n, sim.h_recv_hi,
                                           g * sizeof(val_t),
                                           cudaMemcpyHostToDevice, sim.s_halo));
            }
            // Boundary rows update after the ghost cells arrive
            launchStep(sim, 0, lo_end, e_in, e_out, f_in, f_out, sim.s_halo);
            launchStep(sim, hi_start, n - hi_start, e_in, e_out, f_in, f_out,
                       sim.s_halo);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Swap buffers
        std::swap(e_in, e_out);
        std::swap(f_in, f_out);
    }

    // Leave the final state in buffer 0
    sim.d_energy[0] = e_in;
    sim.d_energy[1] = e_out;
    sim.d_flux[0] = f_in;
    sim.d_flux[1] = f_out;
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

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

    // Bind this rank to a GPU (round-robin over the node's devices)
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

    const long long n_elems =
        static_cast<long long>(n_elems_root) * n_elems_root;

    // Row-slab decomposition of the grid across ranks
    const int base_rows = n_elems_root / n_ranks;
    const int rem_rows = n_elems_root % n_ranks;
    const int my_rows = base_rows + (rank < rem_rows ? 1 : 0);
    const int x0 = rank * base_rows + std::min(rank, rem_rows);
    const int rank_prev =
        (my_rows > 0 && x0 > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_next =
        (my_rows > 0 && x0 + my_rows < n_elems_root) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root,
               n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks x %d OpenMP threads, CUDA (%d GPUs/node)\n",
               n_ranks, omp_get_max_threads(), n_devices);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    LocalWorld world;
    buildSquare2D(world, n_elems_root, x0, my_rows);

    DeviceSim sim;
    setupDevice(world, sim);

    if (rank == 0) {
        // Calculate memory usage (global, as in the reference layout)
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
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

    runSimulation(sim, n_iters, rank_prev, rank_next, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Fetch owned results from the GPU
    std::vector<val_t> local_energy(std::max(sim.n_owned, 1LL));
    std::vector<val_t> local_flux(std::max(sim.n_owned, 1LL));
    if (sim.n_owned > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), sim.d_energy[0] + sim.ghost,
                              sim.n_owned * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), sim.d_flux[0],
                              sim.n_owned * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    // Gather the global result on rank 0 (slabs are contiguous and ordered)
    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const int r_rows = base_rows + (r < rem_rows ? 1 : 0);
        const int r_x0 = r * base_rows + std::min(r, rem_rows);
        counts[r] = r_rows * n_elems_root;
        displs[r] = r_x0 * n_elems_root;
    }
    std::vector<val_t> global_energy, global_flux;
    if (rank == 0) {
        global_energy.resize(n_elems);
        global_flux.resize(n_elems);
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(sim.n_owned), MPI_DOUBLE,
                global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(sim.n_owned), MPI_DOUBLE,
                global_flux.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exit_code = 0;
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

        std::vector<ElementDynamic> elements_dynamic(n_elems);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < n_elems; ++i) {
            elements_dynamic[i].current_energy = global_energy[i];
            elements_dynamic[i].total_flux = global_flux[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(global_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(elements_dynamic);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
