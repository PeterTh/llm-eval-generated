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

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;
constexpr int NUM_MATERIALS = 3;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// -----------------------------------------------------------------------------
// Distributed world state
//
// The mesh is decomposed across MPI ranks along the first grid dimension (x):
// every rank owns a contiguous band of rows, which in the original global
// indexing (idx = x * n_root + y) is a contiguous index range. Each rank keeps
// one halo row on each side so that the connectivity of its owned elements can
// be resolved purely locally.
//
// Element data is kept on the device in SoA form for coalesced access, while
// the connectivity is stored in its fully general (per-element index/flux list)
// form, so the kernel performs the same unstructured gather as the reference.
// -----------------------------------------------------------------------------
struct World {
    int n_elems_root = 0;   // global grid edge length
    int x_begin = 0;        // first owned row
    int rows_local = 0;     // number of owned rows
    size_t n_owned = 0;     // owned elements (rows_local * n_elems_root)
    size_t n_local = 0;     // owned + 2 halo rows

    std::vector<Material> materials;

    // Device state (SoA)
    double* d_energy[2] = {nullptr, nullptr};  // size n_local, owned block offset by n_elems_root
    double* d_flux[2] = {nullptr, nullptr};    // size n_owned
    int32_t* d_conn_idx = nullptr;             // MAX_CONNECTIONS * n_owned, stride n_owned
    double* d_conn_flux = nullptr;             // MAX_CONNECTIONS * n_owned, stride n_owned
    uint8_t* d_num_conn = nullptr;             // n_owned
    uint8_t* d_mat_idx = nullptr;              // n_owned

    // Pinned staging buffers for halo exchange
    double* h_send_lo = nullptr;
    double* h_send_hi = nullptr;
    double* h_recv_lo = nullptr;
    double* h_recv_hi = nullptr;

    // Communicator containing only the ranks that actually own elements
    MPI_Comm comm_active = MPI_COMM_NULL;
    int rank_active = 0;
    int size_active = 1;

    int cur = 0;  // index of the buffer holding the current state
};

// Material table in constant memory (tiny, read by every thread)
__constant__ double c_transfer_coeff[NUM_MATERIALS];
__constant__ double c_external_flow[NUM_MATERIALS];

// Build the local part of the 2D square grid mesh.
// Produces exactly the connectivity (and connection ordering) of the reference
// implementation, restricted to the elements owned by this rank.
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    world.n_elems_root = n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Row-wise decomposition of the global grid
    const int base = n_elems_root / n_ranks;
    const int rem = n_elems_root % n_ranks;
    world.rows_local = base + (rank < rem ? 1 : 0);
    world.x_begin = rank * base + std::min(rank, rem);

    const size_t n_root = static_cast<size_t>(n_elems_root);
    world.n_owned = static_cast<size_t>(world.rows_local) * n_root;
    world.n_local = world.n_owned + 2 * n_root;

    double mat_transfer[NUM_MATERIALS];
    double mat_external[NUM_MATERIALS];
    for (int m = 0; m < NUM_MATERIALS; ++m) {
        mat_transfer[m] = world.materials[m].transfer_coeff;
        mat_external[m] = world.materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, mat_transfer, sizeof(mat_transfer)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, mat_external, sizeof(mat_external)));

    // Host-side assembly of the local connectivity (OpenMP parallel over elements)
    const size_t n_owned = world.n_owned;
    std::vector<int32_t> conn_idx(MAX_CONNECTIONS * n_owned);
    std::vector<double> conn_flux(MAX_CONNECTIONS * n_owned);
    std::vector<uint8_t> num_conn(n_owned);
    std::vector<uint8_t> mat_idx(n_owned);

    const int last = n_elems_root - 1;
    const int x_begin = world.x_begin;

    #pragma omp parallel for schedule(static)
    for (int lx = 0; lx < world.rows_local; ++lx) {
        const int x = x_begin + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t i = static_cast<size_t>(lx) * n_root + static_cast<size_t>(y);

            // Connect to neighbors (up, down, left, right) - same order as reference
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            int nc = 0;
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // Local index: owned block starts one halo row into the buffer
                    const int32_t neighbor_local =
                        static_cast<int32_t>((nx - x_begin + 1) * static_cast<int64_t>(n_root) + ny);
                    conn_idx[static_cast<size_t>(nc) * n_owned + i] = neighbor_local;
                    conn_flux[static_cast<size_t>(nc) * n_owned + i] = 1.0;
                    ++nc;
                }
            }
            num_conn[i] = static_cast<uint8_t>(nc);

            // Corner elements are inflow/outflow to create interesting dynamics
            uint8_t mat = static_cast<uint8_t>(DEFAULT_MAT_ID);
            if ((x == 0 && y == 0) || (x == last && y == last)) {
                mat = static_cast<uint8_t>(INFLOW_MAT_ID);
            } else if ((x == 0 && y == last) || (x == last && y == 0)) {
                mat = static_cast<uint8_t>(OUTFLOW_MAT_ID);
            }
            mat_idx[i] = mat;
        }
    }

    // Upload to device
    CUDA_CHECK(cudaMalloc(&world.d_energy[0], world.n_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&world.d_energy[1], world.n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(world.d_energy[0], 0, world.n_local * sizeof(double)));
    CUDA_CHECK(cudaMemset(world.d_energy[1], 0, world.n_local * sizeof(double)));

    if (n_owned > 0) {
        CUDA_CHECK(cudaMalloc(&world.d_flux[0], n_owned * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&world.d_flux[1], n_owned * sizeof(double)));
        CUDA_CHECK(cudaMemset(world.d_flux[0], 0, n_owned * sizeof(double)));
        CUDA_CHECK(cudaMemset(world.d_flux[1], 0, n_owned * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&world.d_conn_idx, conn_idx.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&world.d_conn_flux, conn_flux.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&world.d_num_conn, n_owned * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&world.d_mat_idx, n_owned * sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpy(world.d_conn_idx, conn_idx.data(),
                              conn_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_conn_flux, conn_flux.data(),
                              conn_flux.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_num_conn, num_conn.data(), n_owned * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(world.d_mat_idx, mat_idx.data(), n_owned * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    }

    // Ranks without elements (more ranks than grid rows) drop out of the
    // neighbour exchange entirely.
    MPI_Comm_split(MPI_COMM_WORLD, (n_owned > 0) ? 1 : MPI_UNDEFINED, rank, &world.comm_active);
    if (world.comm_active != MPI_COMM_NULL) {
        MPI_Comm_rank(world.comm_active, &world.rank_active);
        MPI_Comm_size(world.comm_active, &world.size_active);
    } else {
        world.rank_active = 0;
        world.size_active = 0;
    }

    if (world.size_active > 1) {
        CUDA_CHECK(cudaMallocHost(&world.h_send_lo, n_root * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&world.h_send_hi, n_root * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&world.h_recv_lo, n_root * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&world.h_recv_hi, n_root * sizeof(double)));
    }
}

// Compute energy flux between two elements
__device__ __forceinline__ double computeFlux(double transfer_coeff, double this_energy,
                                              double connection_flux, double other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update the owned elements in [begin, end) of the local element range
__global__ void stepKernel(const double* __restrict__ energy_in, double* __restrict__ energy_out,
                           const double* __restrict__ flux_in, double* __restrict__ flux_out,
                           const int32_t* __restrict__ conn_idx,
                           const double* __restrict__ conn_flux,
                           const uint8_t* __restrict__ num_conn,
                           const uint8_t* __restrict__ mat_idx, size_t stride, size_t halo_offset,
                           size_t begin, size_t end) {
    const size_t total = end - begin;
    for (size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; t < total;
         t += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = begin + t;
        const size_t li = i + halo_offset;

        const int mat = mat_idx[i];
        const double coeff = c_transfer_coeff[mat];
        const double this_energy = energy_in[li];

        // Start with external flow
        double total_flux = c_external_flow[mat];

        // Add flux from all connected elements (same order as the reference)
        const int nc = num_conn[i];
        for (int j = 0; j < nc; ++j) {
            const size_t off = static_cast<size_t>(j) * stride + i;
            const int32_t neighbor = conn_idx[off];
            total_flux += computeFlux(coeff, this_energy, conn_flux[off],
                                      __ldg(&energy_in[neighbor]));
        }

        // Update element state
        energy_out[li] = this_energy + total_flux;
        flux_out[i] = flux_in[i] + fabs(total_flux);
    }
}

namespace {
constexpr int BLOCK_SIZE = 256;

inline int gridFor(size_t n, int max_blocks) {
    const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return static_cast<int>(std::min<size_t>(blocks, static_cast<size_t>(max_blocks)));
}
}  // namespace

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, cudaStream_t stream_bnd,
                   cudaStream_t stream_int, int max_blocks) {
    const size_t n_root = static_cast<size_t>(world.n_elems_root);
    const size_t n_owned = world.n_owned;
    const size_t stride = n_owned;
    if (n_owned == 0) return;

    const int rank = world.rank_active;
    const int n_ranks = world.size_active;
    const int rank_lo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_hi = (rank + 1 < n_ranks) ? rank + 1 : MPI_PROC_NULL;

    // Rows on the rank boundary are computed first so their halo exchange can
    // overlap with the interior update.
    const bool split = (n_ranks > 1) && (world.rows_local >= 3);
    const size_t bnd_lo_end = n_root;
    const size_t bnd_hi_begin = n_owned - n_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        const double* e_in = world.d_energy[world.cur];
        double* e_out = world.d_energy[1 - world.cur];
        const double* f_in = world.d_flux[world.cur];
        double* f_out = world.d_flux[1 - world.cur];

        auto launch = [&](size_t begin, size_t end, cudaStream_t s) {
            if (end <= begin) return;
            stepKernel<<<gridFor(end - begin, max_blocks), BLOCK_SIZE, 0, s>>>(
                e_in, e_out, f_in, f_out, world.d_conn_idx, world.d_conn_flux, world.d_num_conn,
                world.d_mat_idx, stride, n_root, begin, end);
        };

        if (split) {
            launch(0, bnd_lo_end, stream_bnd);
            launch(bnd_hi_begin, n_owned, stream_bnd);
            CUDA_CHECK(cudaMemcpyAsync(world.h_send_lo, e_out + n_root, n_root * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_bnd));
            CUDA_CHECK(cudaMemcpyAsync(world.h_send_hi, e_out + n_owned, n_root * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_bnd));
            launch(bnd_lo_end, bnd_hi_begin, stream_int);
            CUDA_CHECK(cudaStreamSynchronize(stream_bnd));
        } else {
            launch(0, n_owned, stream_bnd);
            if (n_ranks > 1) {
                CUDA_CHECK(cudaMemcpyAsync(world.h_send_lo, e_out + n_root, n_root * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream_bnd));
                CUDA_CHECK(cudaMemcpyAsync(world.h_send_hi, e_out + n_owned, n_root * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream_bnd));
                CUDA_CHECK(cudaStreamSynchronize(stream_bnd));
            }
        }

        if (n_ranks > 1) {
            // Send my first owned row down, my last owned row up.
            MPI_Sendrecv(world.h_send_lo, static_cast<int>(n_root), MPI_DOUBLE, rank_lo, 0,
                         world.h_recv_hi, static_cast<int>(n_root), MPI_DOUBLE, rank_hi, 0,
                         world.comm_active, MPI_STATUS_IGNORE);
            MPI_Sendrecv(world.h_send_hi, static_cast<int>(n_root), MPI_DOUBLE, rank_hi, 1,
                         world.h_recv_lo, static_cast<int>(n_root), MPI_DOUBLE, rank_lo, 1,
                         world.comm_active, MPI_STATUS_IGNORE);

            if (rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_out, world.h_recv_lo, n_root * sizeof(double),
                                           cudaMemcpyHostToDevice, stream_bnd));
            }
            if (rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_out + n_owned + n_root, world.h_recv_hi,
                                           n_root * sizeof(double), cudaMemcpyHostToDevice,
                                           stream_bnd));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_bnd));
        if (split) {
            CUDA_CHECK(cudaStreamSynchronize(stream_int));
        }

        // Swap buffers
        world.cur = 1 - world.cur;
    }
}

// Copy the owned dynamic state back to the host
void downloadState(const World& world, std::vector<double>& energy, std::vector<double>& flux) {
    energy.resize(world.n_owned);
    flux.resize(world.n_owned);
    if (world.n_owned == 0) return;
    CUDA_CHECK(cudaMemcpy(energy.data(), world.d_energy[world.cur] + world.n_elems_root,
                          world.n_owned * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), world.d_flux[world.cur], world.n_owned * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements_dynamic) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements_dynamic) {
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
// The combination is a plain XOR, so it is computed with an OpenMP reduction.
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    const size_t n = elements.size();
    #pragma omp parallel for reduction(^ : hash) schedule(static)
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

    int rank = 0;
    int n_ranks = 1;
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

    const int n_elems = n_elems_root * n_elems_root;

    // Bind each rank to one of the local GPUs
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    CUDA_CHECK(cudaSetDevice(node_rank % n_devices));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context up front

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, node_rank % n_devices));
    const int max_blocks = prop.multiProcessorCount * 32;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI rank(s), %d OpenMP thread(s)/rank, CUDA\n", n_ranks,
               omp_get_max_threads());
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    cudaStream_t stream_bnd, stream_int;
    CUDA_CHECK(cudaStreamCreate(&stream_bnd));
    CUDA_CHECK(cudaStreamCreate(&stream_int));

    if (rank == 0) {
        // Calculate memory usage (of the equivalent global mesh)
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

    runSimulation(world, n_iters, stream_bnd, stream_int, max_blocks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Collect the final state on rank 0 (untimed, as in the reference)
    std::vector<double> local_energy, local_flux;
    downloadState(world, local_energy, local_flux);

    std::vector<int> counts(n_ranks), displs(n_ranks);
    const int local_count = static_cast<int>(world.n_owned);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> all_energy, all_flux;
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < n_ranks; ++r) {
            displs[r] = off;
            off += counts[r];
        }
        all_energy.resize(static_cast<size_t>(off));
        all_flux.resize(static_cast<size_t>(off));
    }
    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, all_energy.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE, all_flux.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(max_duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / (max_duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        std::vector<ElementDynamic> elements_dynamic(all_energy.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < all_energy.size(); ++i) {
            elements_dynamic[i].current_energy = all_energy[i];
            elements_dynamic[i].total_flux = all_flux[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(elements_dynamic)) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream_bnd));
    CUDA_CHECK(cudaStreamDestroy(stream_int));
    if (world.comm_active != MPI_COMM_NULL) MPI_Comm_free(&world.comm_active);
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return exit_code;
}
