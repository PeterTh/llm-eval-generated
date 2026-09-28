// Unstructured mesh energy transfer benchmark.
//
// Hybrid parallelization:
//   * MPI   - the mesh is distributed over the ranks as contiguous slabs of
//             elements; one layer of halo elements is exchanged per iteration.
//   * CUDA  - the per-iteration element update runs on the GPU owned by the
//             rank, using an ELLPACK-style (structure of arrays) connectivity
//             layout for fully coalesced accesses.  Halo exchange is
//             overlapped with the interior update.
//   * OpenMP - mesh construction, host side packing and the final reductions
//             (hash / validation) are threaded.

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

constexpr int MAX_MATERIALS = 8;

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Material properties live in constant memory (tiny, broadcast to all threads)
__constant__ val_t c_transfer_coeff[MAX_MATERIALS];
__constant__ val_t c_external_flow[MAX_MATERIALS];

// ---------------------------------------------------------------------------
// Distributed mesh
// ---------------------------------------------------------------------------

// Local part of the mesh owned by this rank.
//
// Elements owned by the rank are numbered 0 .. n_owned-1 and correspond to the
// global elements [elem_offset, elem_offset + n_owned).  The energy array is
// padded with one halo layer of `halo` elements on each side, so the energy of
// owned element `o` lives at energy[o + halo].  Connectivity indices are stored
// as indices into that padded array, so halo neighbors need no special casing.
struct LocalMesh {
    // geometry / decomposition
    int n_elems_root = 0;
    int64_t row_begin = 0;   // first mesh row (x) owned by this rank
    int64_t n_rows = 0;      // number of mesh rows owned by this rank
    int64_t n_owned = 0;     // n_rows * n_elems_root
    int64_t halo = 0;        // halo elements on each side (n_elems_root, or 0)
    int64_t n_padded = 0;    // n_owned + 2 * halo
    int64_t elem_offset = 0; // global index of local element 0

    // materials (replicated, tiny)
    std::vector<Material> materials;

    // host side connectivity (ELLPACK / SoA)
    int max_conn = 0;
    int64_t stride = 0;               // column stride of the ELLPACK arrays
    std::vector<int32_t> h_neighbor;  // [max_conn * stride] padded local indices
    std::vector<val_t> h_conn_flux;   // [max_conn * stride]
    std::vector<uint8_t> h_num_conn;  // [n_owned]
    std::vector<uint8_t> h_material;  // [n_owned]

    // device state
    val_t* d_energy = nullptr;      // [n_padded]
    val_t* d_energy_new = nullptr;  // [n_padded]
    val_t* d_flux = nullptr;        // [n_owned]
    int32_t* d_neighbor = nullptr;
    val_t* d_conn_flux = nullptr;
    uint8_t* d_num_conn = nullptr;
    uint8_t* d_material = nullptr;
};

// Build the local slab of a 2D square grid represented as an unstructured mesh.
// Semantically identical to building the full mesh and keeping the local rows.
void buildSquare2D(LocalMesh& mesh, const int n_elems_root, const int64_t row_begin,
                   const int64_t n_rows) {
    mesh.n_elems_root = n_elems_root;
    mesh.row_begin = row_begin;
    mesh.n_rows = n_rows;
    mesh.n_owned = n_rows * n_elems_root;
    mesh.halo = (n_rows > 0) ? n_elems_root : 0;
    mesh.n_padded = mesh.n_owned + 2 * mesh.halo;
    mesh.elem_offset = row_begin * static_cast<int64_t>(n_elems_root);

    // Initialize materials
    mesh.materials.emplace_back(Material{0.8, 0.0});    // Default material
    mesh.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    mesh.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    if (mesh.n_owned == 0) {
        return;
    }

    if (mesh.n_padded > static_cast<int64_t>(INT32_MAX)) {
        fprintf(stderr, "Local mesh too large for 32 bit connectivity indices\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // A 2D grid element has at most 4 neighbors
    mesh.max_conn = 4;
    // Round the column stride up so every ELLPACK column starts 1 KiB aligned
    constexpr int64_t align = 128;
    mesh.stride = ((mesh.n_owned + align - 1) / align) * align;

    mesh.h_neighbor.resize(mesh.max_conn * mesh.stride, 0);
    mesh.h_conn_flux.resize(mesh.max_conn * mesh.stride, 0.0);
    mesh.h_num_conn.resize(mesh.n_owned, 0);
    mesh.h_material.resize(mesh.n_owned, static_cast<uint8_t>(DEFAULT_MAT_ID));

    const int64_t stride = mesh.stride;
    const int64_t halo = mesh.halo;

    // Build connectivity: each element connects to its neighbors in 2D grid
#pragma omp parallel for schedule(static)
    for (int64_t lx = 0; lx < n_rows; ++lx) {
        const int64_t x = row_begin + lx;
        for (int64_t y = 0; y < n_elems_root; ++y) {
            const int64_t o = lx * n_elems_root + y;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            int num_connections = 0;
            for (int n = 0; n < 4; ++n) {
                const int64_t nx = x + offsets[n][0];
                const int64_t ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    // index into the padded local energy array
                    const int64_t neighbor_idx = (nx - row_begin) * n_elems_root + ny + halo;
                    mesh.h_neighbor[num_connections * stride + o] =
                        static_cast<int32_t>(neighbor_idx);
                    mesh.h_conn_flux[num_connections * stride + o] = 1.0;
                    num_connections++;
                }
            }
            mesh.h_num_conn[o] = static_cast<uint8_t>(num_connections);
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int64_t last = n_elems_root - 1;
    const auto set_material = [&](int64_t x, int64_t y, idx_t mat) {
        if (x >= row_begin && x < row_begin + n_rows) {
            mesh.h_material[(x - row_begin) * n_elems_root + y] = static_cast<uint8_t>(mat);
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// Upload the local mesh (and the zero initial state) to the GPU
void uploadMesh(LocalMesh& mesh) {
    if (mesh.n_owned == 0) {
        return;
    }

    const size_t conn_elems = static_cast<size_t>(mesh.max_conn) * mesh.stride;

    CUDA_CHECK(cudaMalloc(&mesh.d_energy, mesh.n_padded * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_energy_new, mesh.n_padded * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_flux, mesh.n_owned * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_neighbor, conn_elems * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_conn_flux, conn_elems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_num_conn, mesh.n_owned * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_material, mesh.n_owned * sizeof(uint8_t)));

    // Initial state: zero energy and zero accumulated flux (incl. halos)
    CUDA_CHECK(cudaMemset(mesh.d_energy, 0, mesh.n_padded * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(mesh.d_energy_new, 0, mesh.n_padded * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(mesh.d_flux, 0, mesh.n_owned * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(mesh.d_neighbor, mesh.h_neighbor.data(), conn_elems * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.d_conn_flux, mesh.h_conn_flux.data(), conn_elems * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.d_num_conn, mesh.h_num_conn.data(), mesh.n_owned * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mesh.d_material, mesh.h_material.data(), mesh.n_owned * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    val_t tc[MAX_MATERIALS] = {};
    val_t ef[MAX_MATERIALS] = {};
    for (size_t i = 0; i < mesh.materials.size() && i < MAX_MATERIALS; ++i) {
        tc[i] = mesh.materials[i].transfer_coeff;
        ef[i] = mesh.materials[i].external_flow;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, tc, sizeof(tc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, ef, sizeof(ef)));

    // Connectivity is only needed on the device from here on
    mesh.h_neighbor.clear();
    mesh.h_neighbor.shrink_to_fit();
    mesh.h_conn_flux.clear();
    mesh.h_conn_flux.shrink_to_fit();
}

void freeMesh(LocalMesh& mesh) {
    if (mesh.d_energy) CUDA_CHECK(cudaFree(mesh.d_energy));
    if (mesh.d_energy_new) CUDA_CHECK(cudaFree(mesh.d_energy_new));
    if (mesh.d_flux) CUDA_CHECK(cudaFree(mesh.d_flux));
    if (mesh.d_neighbor) CUDA_CHECK(cudaFree(mesh.d_neighbor));
    if (mesh.d_conn_flux) CUDA_CHECK(cudaFree(mesh.d_conn_flux));
    if (mesh.d_num_conn) CUDA_CHECK(cudaFree(mesh.d_num_conn));
    if (mesh.d_material) CUDA_CHECK(cudaFree(mesh.d_material));
}

// ---------------------------------------------------------------------------
// Simulation kernel
// ---------------------------------------------------------------------------

constexpr int BLOCK_SIZE = 256;

// Update the owned elements [begin, begin + count) for one iteration.
__global__ __launch_bounds__(BLOCK_SIZE) void updateElementsKernel(
    const val_t* __restrict__ energy, val_t* __restrict__ energy_new,
    val_t* __restrict__ flux, const int32_t* __restrict__ neighbor,
    const val_t* __restrict__ conn_flux, const uint8_t* __restrict__ num_conn,
    const uint8_t* __restrict__ material, const int64_t stride, const int64_t halo,
    const int64_t begin, const int64_t count) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t i = tid; i < count; i += step) {
        const int64_t o = begin + i;

        const int mat = material[o];
        const val_t transfer_coeff = c_transfer_coeff[mat];

        const val_t self_energy = energy[o + halo];

        // Start with external flow
        val_t total_flux = c_external_flow[mat];

        // Add flux from all connected elements
        const int nc = num_conn[o];
        for (int j = 0; j < nc; ++j) {
            const int64_t off = static_cast<int64_t>(j) * stride + o;
            const val_t neighbor_energy = energy[neighbor[off]];
            total_flux += (neighbor_energy - self_energy) * transfer_coeff * conn_flux[off] * 0.25;
        }

        // Update element state
        energy_new[o + halo] = self_energy + total_flux;
        flux[o] += fabs(total_flux);
    }
}

static inline void launchUpdate(const LocalMesh& mesh, int64_t begin, int64_t count,
                                cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    int64_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min<int64_t>(blocks, 65535 * 16);
    updateElementsKernel<<<static_cast<unsigned>(blocks), BLOCK_SIZE, 0, stream>>>(
        mesh.d_energy, mesh.d_energy_new, mesh.d_flux, mesh.d_neighbor, mesh.d_conn_flux,
        mesh.d_num_conn, mesh.d_material, mesh.stride, mesh.halo, begin, count);
}

// Run simulation for n_iters iterations.
// `comm` contains exactly the ranks that own at least one mesh row, in
// increasing order of their global element offset.
void runSimulation(LocalMesh& mesh, const int n_iters, MPI_Comm comm) {
    if (mesh.n_owned == 0) {
        return;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int rank_lo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_hi = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    const bool exchange = (rank_lo != MPI_PROC_NULL) || (rank_hi != MPI_PROC_NULL);

    const int64_t n_root = mesh.n_elems_root;
    const int64_t halo = mesh.halo;
    const int64_t n_rows = mesh.n_rows;

    cudaStream_t stream_compute, stream_halo;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_halo));

    // Pinned staging buffers for the halo exchange (MPI is not CUDA aware)
    val_t *send_lo = nullptr, *send_hi = nullptr, *recv_lo = nullptr, *recv_hi = nullptr;
    if (exchange) {
        CUDA_CHECK(cudaMallocHost(&send_lo, n_root * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&send_hi, n_root * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&recv_lo, n_root * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&recv_hi, n_root * sizeof(val_t)));
    }

    // Interior rows (independent of the halo) vs. boundary rows
    const int64_t interior_begin = n_root;                         // skip first owned row
    const int64_t interior_count = std::max<int64_t>(n_rows - 2, 0) * n_root;
    const int64_t last_row_begin = (n_rows - 1) * n_root;

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int n_reqs = 0;

        if (exchange) {
            // Ship the outermost owned rows of the current energy state
            CUDA_CHECK(cudaMemcpyAsync(send_lo, mesh.d_energy + halo, n_root * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, stream_halo));
            CUDA_CHECK(cudaMemcpyAsync(send_hi, mesh.d_energy + halo + last_row_begin,
                                       n_root * sizeof(val_t), cudaMemcpyDeviceToHost,
                                       stream_halo));
            CUDA_CHECK(cudaStreamSynchronize(stream_halo));

            MPI_Irecv(recv_lo, n_root, MPI_DOUBLE, rank_lo, 0, comm, &reqs[n_reqs++]);
            MPI_Irecv(recv_hi, n_root, MPI_DOUBLE, rank_hi, 1, comm, &reqs[n_reqs++]);
            MPI_Isend(send_lo, n_root, MPI_DOUBLE, rank_lo, 1, comm, &reqs[n_reqs++]);
            MPI_Isend(send_hi, n_root, MPI_DOUBLE, rank_hi, 0, comm, &reqs[n_reqs++]);
        }

        // Interior update overlaps with the halo exchange
        launchUpdate(mesh, interior_begin, interior_count, stream_compute);

        if (exchange) {
            MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
            if (rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(mesh.d_energy, recv_lo, n_root * sizeof(val_t),
                                           cudaMemcpyHostToDevice, stream_halo));
            }
            if (rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(mesh.d_energy + halo + mesh.n_owned, recv_hi,
                                           n_root * sizeof(val_t), cudaMemcpyHostToDevice,
                                           stream_halo));
            }
        }

        // Boundary rows (need the freshly received halos)
        launchUpdate(mesh, 0, n_root, stream_halo);
        if (n_rows > 1) {
            launchUpdate(mesh, last_row_begin, n_root, stream_halo);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_halo));

        // Swap buffers
        std::swap(mesh.d_energy, mesh.d_energy_new);
    }

    if (exchange) {
        CUDA_CHECK(cudaFreeHost(send_lo));
        CUDA_CHECK(cudaFreeHost(send_hi));
        CUDA_CHECK(cudaFreeHost(recv_lo));
        CUDA_CHECK(cudaFreeHost(recv_hi));
    }
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_halo));
}

// ---------------------------------------------------------------------------
// Post processing
// ---------------------------------------------------------------------------

// Validate simulation results (rank 0, on the gathered global state)
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
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

// Compute a simple hash of the local results.  The per-element contributions
// are XOR combined and use the global element index, so the local hashes can be
// XOR reduced across ranks to yield the hash of the full mesh.
uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                     const int64_t elem_offset) {
    uint64_t hash = 0;
    const int64_t n = static_cast<int64_t>(energy.size());
#pragma omp parallel for schedule(static) reduction(^ : hash)
    for (int64_t k = 0; k < n; ++k) {
        const uint64_t i = static_cast<uint64_t>(elem_offset + k);
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energy[k]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&flux[k]);
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
    int mpi_provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0) {
        if (world_rank == 0) {
            printf("Grid size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Pick the GPU for this rank (round robin over the devices of its node)
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,
                        &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % n_devices));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context before timing

    const int64_t n_elems = static_cast<int64_t>(n_elems_root) * n_elems_root;

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %ld elements\n", n_elems_root, n_elems_root,
               static_cast<long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n", world_size,
               omp_get_max_threads(), n_devices);
        printf("\n");
    }

    // Decompose the mesh into contiguous slabs of rows
    const int64_t rows_per_rank = n_elems_root / world_size;
    const int64_t rows_rem = n_elems_root % world_size;
    const int64_t row_begin =
        world_rank * rows_per_rank + std::min<int64_t>(world_rank, rows_rem);
    const int64_t n_rows = rows_per_rank + (world_rank < rows_rem ? 1 : 0);

    // Build the unstructured mesh
    if (world_rank == 0) {
        printf("Building unstructured mesh...\n");
    }
    LocalMesh mesh;
    buildSquare2D(mesh, n_elems_root, row_begin, n_rows);
    uploadMesh(mesh);

    // Calculate memory usage (of the full, distributed mesh)
    if (world_rank == 0) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Communicator holding only the ranks that actually own mesh rows
    MPI_Comm sim_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (n_rows > 0) ? 0 : MPI_UNDEFINED, world_rank, &sim_comm);

    // Run simulation
    if (world_rank == 0) {
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (sim_comm != MPI_COMM_NULL) {
        runSimulation(mesh, n_iters, sim_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    // Use the slowest rank as the wall clock time of the simulation.  The
    // elapsed time is kept in double precision so the derived metrics stay
    // meaningful even when the GPUs finish in less than a millisecond.
    double duration_s = std::chrono::duration<double>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(duration_s * 1000.0);

    if (sim_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&sim_comm);
    }

    // Fetch the local results back to the host
    std::vector<val_t> energy(mesh.n_owned);
    std::vector<val_t> flux(mesh.n_owned);
    if (mesh.n_owned > 0) {
        CUDA_CHECK(cudaMemcpy(energy.data(), mesh.d_energy + mesh.halo,
                              mesh.n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flux.data(), mesh.d_flux, mesh.n_owned * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    freeMesh(mesh);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
    }

    // Calculate performance metrics
    if (world_rank == 0) {
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_s * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / duration_s / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification (XOR reduction over the distributed mesh)
    uint64_t hash = computeHash(energy, flux, mesh.elem_offset);
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");
    }

    // Gather the global state on rank 0 for output / validation
    if (printResults || validate) {
        std::vector<int> counts(world_size), displs(world_size);
        const int local_count = static_cast<int>(mesh.n_owned);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<val_t> all_energy, all_flux;
        if (world_rank == 0) {
            int64_t total = 0;
            for (int r = 0; r < world_size; ++r) {
                displs[r] = static_cast<int>(total);
                total += counts[r];
            }
            all_energy.resize(total);
            all_flux.resize(validate ? total : 0);
        }

        MPI_Gatherv(energy.data(), local_count, MPI_DOUBLE, all_energy.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            MPI_Gatherv(flux.data(), local_count, MPI_DOUBLE, all_flux.data(), counts.data(),
                        displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        int valid = 1;
        if (world_rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(all_energy, "ElementEnergy");
            }

            // Validation
            if (validate) {
                valid = validateResults(all_energy, all_flux) ? 1 : 0;
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
