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

// Local (per-rank) index type used for on-device connectivity. Local element
// counts stay far below 2^31 for any problem that fits into device memory.
using lidx_t = int32_t;

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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),        \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Distributed mesh representation
//
// The mesh is partitioned across MPI ranks by contiguous blocks of grid rows
// (the leading index of the element numbering, i.e. `x`). Every rank stores the
// static connectivity of the elements it owns, plus one halo row of dynamic
// state on each side so that all connections of owned elements resolve to a
// local index.
//
// Dynamic state is kept in structure-of-arrays form for coalesced device
// accesses; connectivity is stored connection-major (stride = owned count) for
// the same reason.
// ---------------------------------------------------------------------------
struct World {
    std::vector<Material> materials;

    int n_elems_root = 0;   // Grid edge length
    int row_begin = 0;      // First owned grid row
    int row_end = 0;        // One past last owned grid row
    int halo_lo = 0;        // 1 if a halo row exists below the owned block
    int halo_hi = 0;        // 1 if a halo row exists above the owned block

    size_t owned = 0;       // Number of owned elements
    size_t local_total = 0; // Owned elements plus halo rows
    size_t base = 0;        // Offset of the first owned element in local arrays
    size_t conn_stride = 0; // Number of connection slots stored per element

    // Host-side static description of the owned elements
    std::vector<uint32_t> material_idx;
    std::vector<uint32_t> num_connections;
    std::vector<lidx_t> connected_idx;   // [j * owned + i]
    std::vector<val_t> connected_flux;   // [j * owned + i]
};

// Build the owned part of a 2D square grid as an unstructured mesh.
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root, const int row_begin, const int row_end) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_elems_root = n_elems_root;
    world.row_begin = row_begin;
    world.row_end = row_end;
    world.halo_lo = (row_begin > 0) ? 1 : 0;
    world.halo_hi = (row_end < n_elems_root) ? 1 : 0;

    const size_t rows_owned = static_cast<size_t>(row_end - row_begin);
    world.owned = rows_owned * static_cast<size_t>(n_elems_root);
    world.local_total = (rows_owned + world.halo_lo + world.halo_hi) * static_cast<size_t>(n_elems_root);
    world.base = static_cast<size_t>(world.halo_lo) * static_cast<size_t>(n_elems_root);

    world.material_idx.assign(world.owned, static_cast<uint32_t>(DEFAULT_MAT_ID));
    world.num_connections.assign(world.owned, 0);
    // Connect to neighbors (up, down, left, right): at most four per element
    constexpr int GRID_CONNECTIONS = 4;
    static_assert(GRID_CONNECTIONS <= MAX_CONNECTIONS);
    world.conn_stride = GRID_CONNECTIONS;
    world.connected_idx.assign(world.owned * GRID_CONNECTIONS, 0);
    world.connected_flux.assign(world.owned * GRID_CONNECTIONS, 0.0);

    if (world.owned == 0) {
        return;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid.
    // Neighbor indices are stored as local indices into the halo-padded arrays.
    const size_t owned = world.owned;
    const long long local_offset =
        static_cast<long long>(row_begin - world.halo_lo) * static_cast<long long>(n_elems_root);

    #pragma omp parallel for schedule(static)
    for (int x = row_begin; x < row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t i = static_cast<size_t>(x - row_begin) * n_elems_root + y;

            const int offsets[GRID_CONNECTIONS][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            uint32_t n_conn = 0;
            for (int n = 0; n < GRID_CONNECTIONS; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const long long neighbor_idx =
                        static_cast<long long>(nx) * n_elems_root + ny - local_offset;
                    world.connected_idx[static_cast<size_t>(n_conn) * owned + i] =
                        static_cast<lidx_t>(neighbor_idx);
                    world.connected_flux[static_cast<size_t>(n_conn) * owned + i] = 1.0;
                    n_conn++;
                }
            }
            world.num_connections[i] = n_conn;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    const auto set_material = [&](int x, int y, idx_t mat) {
        if (x >= row_begin && x < row_end) {
            world.material_idx[static_cast<size_t>(x - row_begin) * n_elems_root + y] =
                static_cast<uint32_t>(mat);
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Compute energy flux between two elements
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update the elements in [start, start + count) of the owned range
__global__ void simulationStepKernel(const uint32_t* __restrict__ material_idx,
                                     const uint32_t* __restrict__ num_connections,
                                     const lidx_t* __restrict__ connected_idx,
                                     const val_t* __restrict__ connected_flux,
                                     const val_t* __restrict__ mat_transfer_coeff,
                                     const val_t* __restrict__ mat_external_flow,
                                     const val_t* __restrict__ energy_in,
                                     const val_t* __restrict__ flux_in,
                                     val_t* __restrict__ energy_out,
                                     val_t* __restrict__ flux_out,
                                     size_t owned, size_t base, size_t start, size_t count) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    const size_t i = start + tid;

    const uint32_t mat = material_idx[i];
    const uint32_t n_conn = num_connections[i];
    const val_t transfer_coeff = mat_transfer_coeff[mat];
    const val_t this_energy = energy_in[base + i];

    // Start with external flow
    val_t total_flux = mat_external_flow[mat];

    // Add flux from all connected elements
    for (uint32_t j = 0; j < n_conn; ++j) {
        const size_t slot = static_cast<size_t>(j) * owned + i;
        const val_t other_energy = energy_in[connected_idx[slot]];
        total_flux += computeFlux(transfer_coeff, this_energy, connected_flux[slot], other_energy);
    }

    // Update element state
    energy_out[base + i] = this_energy + total_flux;
    flux_out[base + i] = flux_in[base + i] + fabs(total_flux);
}

// Device-side state for one rank
struct DeviceState {
    uint32_t* material_idx = nullptr;
    uint32_t* num_connections = nullptr;
    lidx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* mat_transfer_coeff = nullptr;
    val_t* mat_external_flow = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux[2] = {nullptr, nullptr};

    // Pinned staging buffers for halo exchange (one grid row each)
    val_t* send_lo = nullptr;
    val_t* send_hi = nullptr;
    val_t* recv_lo = nullptr;
    val_t* recv_hi = nullptr;

    cudaStream_t stream_boundary = nullptr;
    cudaStream_t stream_interior = nullptr;
};

void uploadWorld(const World& world, DeviceState& dev) {
    const size_t owned = world.owned;
    const size_t local_total = world.local_total;
    const size_t n_mat = world.materials.size();
    const size_t row = static_cast<size_t>(world.n_elems_root);

    CUDA_CHECK(cudaStreamCreate(&dev.stream_boundary));
    CUDA_CHECK(cudaStreamCreate(&dev.stream_interior));

    CUDA_CHECK(cudaMalloc(&dev.mat_transfer_coeff, n_mat * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dev.mat_external_flow, n_mat * sizeof(val_t)));
    std::vector<val_t> coeff(n_mat), ext(n_mat);
    for (size_t m = 0; m < n_mat; ++m) {
        coeff[m] = world.materials[m].transfer_coeff;
        ext[m] = world.materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpy(dev.mat_transfer_coeff, coeff.data(), n_mat * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.mat_external_flow, ext.data(), n_mat * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    if (owned > 0) {
        CUDA_CHECK(cudaMalloc(&dev.material_idx, owned * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dev.num_connections, owned * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dev.connected_idx, owned * world.conn_stride * sizeof(lidx_t)));
        CUDA_CHECK(cudaMalloc(&dev.connected_flux, owned * world.conn_stride * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(dev.material_idx, world.material_idx.data(),
                              owned * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.num_connections, world.num_connections.data(),
                              owned * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.connected_idx, world.connected_idx.data(),
                              owned * world.conn_stride * sizeof(lidx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.connected_flux, world.connected_flux.data(),
                              owned * world.conn_stride * sizeof(val_t), cudaMemcpyHostToDevice));

        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dev.energy[b], local_total * sizeof(val_t)));
            CUDA_CHECK(cudaMalloc(&dev.flux[b], local_total * sizeof(val_t)));
            // All elements start with zero energy and zero accumulated flux
            CUDA_CHECK(cudaMemset(dev.energy[b], 0, local_total * sizeof(val_t)));
            CUDA_CHECK(cudaMemset(dev.flux[b], 0, local_total * sizeof(val_t)));
        }

        CUDA_CHECK(cudaHostAlloc(&dev.send_lo, row * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&dev.send_hi, row * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&dev.recv_lo, row * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&dev.recv_hi, row * sizeof(val_t), cudaHostAllocDefault));
    }
}

void freeWorld(DeviceState& dev) {
    cudaFree(dev.material_idx);
    cudaFree(dev.num_connections);
    cudaFree(dev.connected_idx);
    cudaFree(dev.connected_flux);
    cudaFree(dev.mat_transfer_coeff);
    cudaFree(dev.mat_external_flow);
    for (int b = 0; b < 2; ++b) {
        cudaFree(dev.energy[b]);
        cudaFree(dev.flux[b]);
    }
    cudaFreeHost(dev.send_lo);
    cudaFreeHost(dev.send_hi);
    cudaFreeHost(dev.recv_lo);
    cudaFreeHost(dev.recv_hi);
    if (dev.stream_boundary) cudaStreamDestroy(dev.stream_boundary);
    if (dev.stream_interior) cudaStreamDestroy(dev.stream_interior);
}

// Run simulation for n_iters iterations.
// Returns the buffer index holding the final state.
int runSimulation(const World& world, DeviceState& dev, const int n_iters, MPI_Comm sim_comm) {
    const size_t owned = world.owned;
    const size_t row = static_cast<size_t>(world.n_elems_root);
    const size_t rows_owned = (row > 0) ? owned / row : 0;

    int sim_rank = MPI_PROC_NULL, sim_size = 0;
    if (sim_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(sim_comm, &sim_rank);
        MPI_Comm_size(sim_comm, &sim_size);
    }
    const int rank_lo = (world.halo_lo && sim_comm != MPI_COMM_NULL) ? sim_rank - 1 : MPI_PROC_NULL;
    const int rank_hi = (world.halo_hi && sim_comm != MPI_COMM_NULL && sim_rank + 1 < sim_size)
                            ? sim_rank + 1
                            : MPI_PROC_NULL;

    int cur = 0;
    if (owned == 0) {
        return cur;
    }

    constexpr int block = 256;
    const auto launch = [&](size_t start, size_t count, cudaStream_t stream, int buf) {
        if (count == 0) {
            return;
        }
        const size_t grid = (count + block - 1) / block;
        simulationStepKernel<<<grid, block, 0, stream>>>(
            dev.material_idx, dev.num_connections, dev.connected_idx, dev.connected_flux,
            dev.mat_transfer_coeff, dev.mat_external_flow, dev.energy[buf], dev.flux[buf],
            dev.energy[buf ^ 1], dev.flux[buf ^ 1], owned, world.base, start, count);
    };

    // The first and last owned rows feed the neighbors' halos, so they are
    // computed first and exchanged while the interior is still being updated.
    const bool has_interior = rows_owned > 2;
    const size_t interior_start = has_interior ? row : 0;
    const size_t interior_count = has_interior ? owned - 2 * row : 0;
    const size_t boundary_count = has_interior ? row : owned;

    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;

        // Boundary rows
        launch(0, boundary_count, dev.stream_boundary, cur);
        if (has_interior) {
            launch(owned - row, row, dev.stream_boundary, cur);
        }

        // Interior rows overlap with the halo exchange below
        launch(interior_start, interior_count, dev.stream_interior, cur);

        if (rank_lo != MPI_PROC_NULL || rank_hi != MPI_PROC_NULL) {
            if (rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dev.send_lo, dev.energy[nxt] + world.base,
                                           row * sizeof(val_t), cudaMemcpyDeviceToHost,
                                           dev.stream_boundary));
            }
            if (rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dev.send_hi, dev.energy[nxt] + world.base + owned - row,
                                           row * sizeof(val_t), cudaMemcpyDeviceToHost,
                                           dev.stream_boundary));
            }
            CUDA_CHECK(cudaStreamSynchronize(dev.stream_boundary));

            MPI_Request reqs[4];
            int n_req = 0;
            if (rank_lo != MPI_PROC_NULL) {
                MPI_Irecv(dev.recv_lo, static_cast<int>(row), MPI_DOUBLE, rank_lo, 0, sim_comm,
                          &reqs[n_req++]);
                MPI_Isend(dev.send_lo, static_cast<int>(row), MPI_DOUBLE, rank_lo, 1, sim_comm,
                          &reqs[n_req++]);
            }
            if (rank_hi != MPI_PROC_NULL) {
                MPI_Irecv(dev.recv_hi, static_cast<int>(row), MPI_DOUBLE, rank_hi, 1, sim_comm,
                          &reqs[n_req++]);
                MPI_Isend(dev.send_hi, static_cast<int>(row), MPI_DOUBLE, rank_hi, 0, sim_comm,
                          &reqs[n_req++]);
            }
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

            if (rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dev.energy[nxt], dev.recv_lo, row * sizeof(val_t),
                                           cudaMemcpyHostToDevice, dev.stream_boundary));
            }
            if (rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dev.energy[nxt] + world.base + owned, dev.recv_hi,
                                           row * sizeof(val_t), cudaMemcpyHostToDevice,
                                           dev.stream_boundary));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(dev.stream_boundary));
        CUDA_CHECK(cudaStreamSynchronize(dev.stream_interior));

        // Swap buffers
        cur = nxt;
    }

    return cur;
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
    }
    #pragma omp parallel for schedule(static) reduction(max : energy_max) reduction(min : energy_min)
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energy.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[i]);
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
    int mpi_thread_provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_provided);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("\n");
    }

    // Pin each rank to one GPU, round-robin within the node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % n_devices));

    // Row-block decomposition of the mesh across ranks
    const int row_begin = static_cast<int>(static_cast<int64_t>(n_elems_root) * rank / n_ranks);
    const int row_end = static_cast<int>(static_cast<int64_t>(n_elems_root) * (rank + 1) / n_ranks);

    // Ranks without any owned rows sit out the halo exchange
    MPI_Comm sim_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (row_end > row_begin) ? 0 : MPI_UNDEFINED, rank, &sim_comm);

    // Build the unstructured mesh
    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, row_begin, row_end);

    if (world.local_total > static_cast<size_t>(std::numeric_limits<lidx_t>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Local partition too large for 32-bit indexing\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    DeviceState dev;
    uploadWorld(world, dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate memory usage (global mesh footprint, as in the reference)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int final_buf = runSimulation(world, dev, n_iters, sim_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

    // Collect the final state on rank 0 in global element order
    std::vector<val_t> local_energy(world.owned), local_flux(world.owned);
    if (world.owned > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), dev.energy[final_buf] + world.base,
                              world.owned * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), dev.flux[final_buf] + world.base,
                              world.owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const int rb = static_cast<int>(static_cast<int64_t>(n_elems_root) * r / n_ranks);
        const int re = static_cast<int>(static_cast<int64_t>(n_elems_root) * (r + 1) / n_ranks);
        counts[r] = (re - rb) * n_elems_root;
        displs[r] = rb * n_elems_root;
    }

    std::vector<val_t> energy, flux;
    if (rank == 0) {
        energy.resize(static_cast<size_t>(n_elems));
        flux.resize(static_cast<size_t>(n_elems));
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(world.owned), MPI_DOUBLE, energy.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(world.owned), MPI_DOUBLE, flux.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        const uint64_t hash = computeHash(energy, flux);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(energy, flux);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    freeWorld(dev);
    if (sim_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&sim_comm);
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
