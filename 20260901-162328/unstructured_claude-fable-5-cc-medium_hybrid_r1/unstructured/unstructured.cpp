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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
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

// CUDA kernel: one thread per local element. Connectivity is stored transposed
// (connection-major) so neighbor index/flux loads are coalesced. Energy is read
// from an extended array that includes halo elements received via MPI. The
// per-connection accumulation order matches the serial reference exactly.
__global__ void updateKernel(const int n_local, const int local_off,
                             const int* __restrict__ num_connections,
                             const val_t* __restrict__ transfer_coeff,
                             const val_t* __restrict__ external_flow,
                             const int* __restrict__ conn_idx,
                             const val_t* __restrict__ conn_flux,
                             const val_t* __restrict__ energy_in,
                             const val_t* __restrict__ tflux_in,
                             val_t* __restrict__ energy_out,
                             val_t* __restrict__ tflux_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const val_t energy = energy_in[local_off + i];
    const val_t coeff = transfer_coeff[i];
    val_t total_flux = external_flow[i];

    const int n_conn = num_connections[i];
    for (int j = 0; j < n_conn; ++j) {
        const val_t neighbor_energy = energy_in[conn_idx[j * n_local + i]];
        total_flux += (neighbor_energy - energy) * coeff * conn_flux[j * n_local + i] * 0.25;
    }

    energy_out[local_off + i] = energy + total_flux;
    tflux_out[i] = tflux_in[i] + fabs(total_flux);
}

// A contiguous range of global element indices exchanged with one peer rank
struct ExchangeRange {
    int peer;
    idx_t global_start;
    int count;
    int tag;
};

// Run the simulation distributed across MPI ranks, with the per-element update
// on the GPU. Each rank owns the contiguous global range [own_lo, own_hi) and
// keeps an extended energy array covering [ext_lo, ext_hi) that includes all
// halo elements referenced by its local connectivity.
void runSimulation(World& world, const int n_iters, const int rank, const int n_ranks,
                   const idx_t own_lo, const idx_t own_hi) {
    const idx_t n_elems = world.elements_static.size();
    const int n_local = static_cast<int>(own_hi - own_lo);

    // Determine the extent of the halo needed by local connectivity
    idx_t ext_lo = own_lo;
    idx_t ext_hi = own_hi;
    #pragma omp parallel for schedule(static) reduction(min:ext_lo) reduction(max:ext_hi)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& elem = world.elements_static[own_lo + i];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            ext_lo = std::min(ext_lo, g);
            ext_hi = std::max(ext_hi, g + 1);
        }
    }
    if (n_local == 0) { ext_lo = own_lo; ext_hi = own_hi; }
    const int n_ext = static_cast<int>(ext_hi - ext_lo);
    const int local_off = static_cast<int>(own_lo - ext_lo);

    // Share every rank's owned and needed ranges to build the exchange plan
    long long my_bounds[4] = {(long long)own_lo, (long long)own_hi,
                              (long long)ext_lo, (long long)ext_hi};
    std::vector<long long> all_bounds(4 * n_ranks);
    MPI_Allgather(my_bounds, 4, MPI_LONG_LONG, all_bounds.data(), 4, MPI_LONG_LONG,
                  MPI_COMM_WORLD);

    // Receives: intersections of my halo intervals with each peer's owned range.
    // Sends: intersections of my owned range with each peer's halo intervals.
    // Tags identify which halo interval (0 = below owned, 1 = above) at the receiver.
    std::vector<ExchangeRange> recvs, sends;
    for (int r = 0; r < n_ranks; ++r) {
        if (r == rank) continue;
        const long long r_own_lo = all_bounds[4 * r + 0];
        const long long r_own_hi = all_bounds[4 * r + 1];
        const long long r_ext_lo = all_bounds[4 * r + 2];
        const long long r_ext_hi = all_bounds[4 * r + 3];

        const long long my_halo[2][2] = {{(long long)ext_lo, (long long)own_lo},
                                         {(long long)own_hi, (long long)ext_hi}};
        const long long peer_halo[2][2] = {{r_ext_lo, r_own_lo}, {r_own_hi, r_ext_hi}};

        for (int k = 0; k < 2; ++k) {
            const long long rlo = std::max(my_halo[k][0], r_own_lo);
            const long long rhi = std::min(my_halo[k][1], r_own_hi);
            if (rlo < rhi) recvs.push_back({r, (idx_t)rlo, (int)(rhi - rlo), k});

            const long long slo = std::max(peer_halo[k][0], (long long)own_lo);
            const long long shi = std::min(peer_halo[k][1], (long long)own_hi);
            if (slo < shi) sends.push_back({r, (idx_t)slo, (int)(shi - slo), k});
        }
    }

    // Pack static element data into structure-of-arrays device layout.
    // Connection indices are remapped into the extended-local index space.
    std::vector<int> h_ncon(n_local);
    std::vector<val_t> h_coeff(n_local), h_extflow(n_local);
    std::vector<int> h_conn((size_t)n_local * MAX_CONNECTIONS);
    std::vector<val_t> h_cflux((size_t)n_local * MAX_CONNECTIONS);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& elem = world.elements_static[own_lo + i];
        const Material& mat = world.materials[elem.material_idx];
        h_ncon[i] = static_cast<int>(elem.num_connections);
        h_coeff[i] = mat.transfer_coeff;
        h_extflow[i] = mat.external_flow;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            h_conn[j * n_local + i] = static_cast<int>(elem.connected_idx[j] - ext_lo);
            h_cflux[j * n_local + i] = elem.connected_flux[j];
        }
    }

    // Device buffers
    int *d_ncon = nullptr, *d_conn = nullptr;
    val_t *d_coeff = nullptr, *d_extflow = nullptr, *d_cflux = nullptr;
    val_t *d_energy[2] = {nullptr, nullptr};
    val_t *d_tflux[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_ncon, std::max(n_local, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coeff, std::max(n_local, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_extflow, std::max(n_local, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_conn, std::max<size_t>((size_t)n_local * MAX_CONNECTIONS, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cflux, std::max<size_t>((size_t)n_local * MAX_CONNECTIONS, 1) * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_energy[b], std::max(n_ext, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_tflux[b], std::max(n_local, 1) * sizeof(val_t)));
    }

    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(d_ncon, h_ncon.data(), n_local * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coeff, h_coeff.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_extflow, h_extflow.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conn, h_conn.data(), (size_t)n_local * MAX_CONNECTIONS * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cflux, h_cflux.data(), (size_t)n_local * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    // Pinned host staging buffer for halo exchange, indexed like the extended array
    val_t* h_energy_ext = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_energy_ext, std::max(n_ext, 1) * sizeof(val_t)));

    // Seed initial state (energy including halo, total flux for local elements)
    std::vector<val_t> h_tflux_init(std::max(n_local, 1));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_ext; ++i) {
        h_energy_ext[i] = world.elements_dynamic[ext_lo + i].current_energy;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        h_tflux_init[i] = world.elements_dynamic[own_lo + i].total_flux;
    }
    if (n_ext > 0) {
        CUDA_CHECK(cudaMemcpy(d_energy[0], h_energy_ext, n_ext * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(d_tflux[0], h_tflux_init.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
    }

    std::vector<MPI_Request> requests;
    requests.reserve(sends.size() + recvs.size());

    int cur = 0;
    const int block_size = 256;
    const int grid_size = (n_local + block_size - 1) / block_size;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Refresh halos: stage owned boundary energies to the host, exchange
        // with peers, and upload the received halo values to the device.
        for (const auto& s : sends) {
            const size_t off = s.global_start - ext_lo;
            CUDA_CHECK(cudaMemcpy(h_energy_ext + off, d_energy[cur] + off,
                                  s.count * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        requests.clear();
        for (const auto& r : recvs) {
            MPI_Request req;
            MPI_Irecv(h_energy_ext + (r.global_start - ext_lo), r.count, MPI_DOUBLE,
                      r.peer, r.tag, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        for (const auto& s : sends) {
            MPI_Request req;
            MPI_Isend(h_energy_ext + (s.global_start - ext_lo), s.count, MPI_DOUBLE,
                      s.peer, s.tag, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
        for (const auto& r : recvs) {
            const size_t off = r.global_start - ext_lo;
            CUDA_CHECK(cudaMemcpy(d_energy[cur] + off, h_energy_ext + off,
                                  r.count * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        // Update all local elements on the GPU
        if (n_local > 0) {
            updateKernel<<<grid_size, block_size>>>(
                n_local, local_off, d_ncon, d_coeff, d_extflow, d_conn, d_cflux,
                d_energy[cur], d_tflux[cur], d_energy[1 - cur], d_tflux[1 - cur]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Swap buffers
        cur = 1 - cur;
    }

    // Bring final local state back to the host and gather the full result on
    // rank 0 so that output, hashing, and validation match the serial code.
    std::vector<val_t> h_energy_out(std::max(n_local, 1)), h_tflux_out(std::max(n_local, 1));
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(h_energy_out.data(), d_energy[cur] + local_off,
                              n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tflux_out.data(), d_tflux[cur],
                              n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const idx_t lo = n_elems * r / n_ranks;
        const idx_t hi = n_elems * (r + 1) / n_ranks;
        counts[r] = static_cast<int>(hi - lo);
        displs[r] = static_cast<int>(lo);
    }
    std::vector<val_t> full_energy, full_tflux;
    if (rank == 0) {
        full_energy.resize(n_elems);
        full_tflux.resize(n_elems);
    }
    MPI_Gatherv(h_energy_out.data(), n_local, MPI_DOUBLE, full_energy.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(h_tflux_out.data(), n_local, MPI_DOUBLE, full_tflux.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < n_elems; ++i) {
            world.elements_dynamic[i].current_energy = full_energy[i];
            world.elements_dynamic[i].total_flux = full_tflux[i];
        }
    }

    CUDA_CHECK(cudaFreeHost(h_energy_ext));
    CUDA_CHECK(cudaFree(d_ncon));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_extflow));
    CUDA_CHECK(cudaFree(d_conn));
    CUDA_CHECK(cudaFree(d_cflux));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_energy[b]));
        CUDA_CHECK(cudaFree(d_tflux[b]));
    }
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

// Compute a simple hash of the results for verification (XOR combination is
// order-independent, so the loop parallelizes safely)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Bind each rank to a GPU round-robin by node-local rank
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
    CUDA_CHECK(cudaFree(nullptr));  // Establish CUDA context up front

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
        printf("Parallelization: %d MPI ranks x %d OpenMP threads + CUDA (%d GPUs/node)\n",
               n_ranks, omp_get_max_threads(), n_devices);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    // Every rank builds the full (cheap) static mesh, then works on its
    // contiguous partition [own_lo, own_hi) of the global element range.
    World world;
    buildSquare2D(world, n_elems_root);

    const idx_t own_lo = static_cast<idx_t>(n_elems) * rank / n_ranks;
    const idx_t own_hi = static_cast<idx_t>(n_elems) * (rank + 1) / n_ranks;

    if (rank == 0) {
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

    runSimulation(world, n_iters, rank, n_ranks, own_lo, own_hi);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

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

        // Validation
        if (validate) {
            bool valid = validateResults(world);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
