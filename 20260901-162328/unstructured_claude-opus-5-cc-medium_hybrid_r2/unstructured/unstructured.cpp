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

// Local (per-rank) index type used for on-device connectivity. A rank's
// sub-mesh always fits comfortably into 32 bits, and the narrower indices
// halve the connectivity bandwidth of the (memory bound) update kernel.
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
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Distributed mesh description
// ---------------------------------------------------------------------------
//
// The mesh is a 2D grid stored in row-major order (global index = x * R + y).
// It is decomposed across MPI ranks in contiguous blocks of rows, so each rank
// owns a contiguous slice of the global element range. Only the energy of the
// two rows adjacent to a rank's slice is needed by its neighbours, so a single
// ghost row on each side is enough.
//
// Per-rank device storage keeps the element state in structure-of-arrays form:
//   energy[]  has (n_own + 2 * R) entries, owned element i lives at i + R,
//             the low ghost row at [0, R) and the high ghost row at the end.
//   flux[]    has n_own entries (never read by other elements).
// Connectivity is stored connection-major (conn[j * n_own + i]) so that the
// per-connection gathers are fully coalesced.

struct MeshPartition {
    int n_elems_root = 0;  // R
    int row_begin = 0;     // first owned row
    int n_rows = 0;        // number of owned rows
    int n_own = 0;         // n_rows * R
    int rank_lo = MPI_PROC_NULL;  // neighbour owning the row above
    int rank_hi = MPI_PROC_NULL;  // neighbour owning the row below

    // Host-side static data (structure of arrays)
    std::vector<lidx_t> conn;      // MAX_CONNECTIONS * n_own
    std::vector<val_t> conn_flux;  // MAX_CONNECTIONS * n_own
    std::vector<uint8_t> nconn;    // n_own
    std::vector<uint8_t> mat;      // n_own
};

// Materials of the benchmark's world, identical on every rank.
static std::vector<Material> buildMaterials() {
    std::vector<Material> materials;
    materials.emplace_back(Material{0.8, 0.0});    // Default material
    materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    return materials;
}

// Material of element (x, y), replicating the assignment order of the original
// corner setup (later assignments to the same element win).
static inline uint8_t materialFor(const int x, const int y, const int n_elems_root) {
    const int last = n_elems_root - 1;
    idx_t m = DEFAULT_MAT_ID;
    if (x == 0 && y == 0) m = INFLOW_MAT_ID;
    if (x == 0 && y == last) m = OUTFLOW_MAT_ID;
    if (x == last && y == 0) m = OUTFLOW_MAT_ID;
    if (x == last && y == last) m = INFLOW_MAT_ID;
    return static_cast<uint8_t>(m);
}

static inline int rowBegin(const int rank, const int size, const int n_elems_root) {
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    return rank * base + std::min(rank, rem);
}

// Build this rank's slice of the 2D square grid. The connectivity is generated
// in exactly the same order as the original serial build so that the per
// element flux summation order - and therefore the result - is unchanged.
static void buildSquare2DPartition(MeshPartition& part, const int n_elems_root, const int rank,
                                   const int size) {
    part.n_elems_root = n_elems_root;
    part.row_begin = rowBegin(rank, size, n_elems_root);
    part.n_rows = rowBegin(rank + 1, size, n_elems_root) - part.row_begin;
    part.n_own = part.n_rows * n_elems_root;

    // Neighbouring ranks: nearest ranks that actually own rows (ranks can be
    // empty when there are more ranks than grid rows).
    for (int r = rank - 1; r >= 0; --r) {
        if (rowBegin(r + 1, size, n_elems_root) - rowBegin(r, size, n_elems_root) > 0) {
            part.rank_lo = r;
            break;
        }
    }
    for (int r = rank + 1; r < size; ++r) {
        if (rowBegin(r + 1, size, n_elems_root) - rowBegin(r, size, n_elems_root) > 0) {
            part.rank_hi = r;
            break;
        }
    }
    if (part.n_own == 0) {
        part.rank_lo = MPI_PROC_NULL;
        part.rank_hi = MPI_PROC_NULL;
        return;
    }

    const size_t n_own = static_cast<size_t>(part.n_own);
    part.conn.resize(n_own * MAX_CONNECTIONS);
    part.conn_flux.resize(n_own * MAX_CONNECTIONS);
    part.nconn.resize(n_own);
    part.mat.resize(n_own);

    const int R = n_elems_root;
    const int x0 = part.row_begin;

#pragma omp parallel for schedule(static)
    for (int xl = 0; xl < part.n_rows; ++xl) {
        const int x = x0 + xl;
        for (int y = 0; y < R; ++y) {
            const int i = xl * R + y;
            part.mat[i] = materialFor(x, y, R);

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            int nc = 0;
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < R && ny >= 0 && ny < R) {
                    // Padded local index: owned rows start at offset R.
                    part.conn[static_cast<size_t>(nc) * n_own + i] =
                        static_cast<lidx_t>((nx - x0 + 1) * static_cast<int64_t>(R) + ny);
                    part.conn_flux[static_cast<size_t>(nc) * n_own + i] = 1.0;
                    ++nc;
                }
            }
            part.nconn[i] = static_cast<uint8_t>(nc);
        }
    }
}

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------

// Compute energy flux between two elements
__device__ __forceinline__ val_t computeFlux(const Material& mat, const val_t this_energy,
                                             const val_t connection_flux,
                                             const val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update elements [elem_begin, elem_begin + count) of this rank's slice.
__global__ void updateElementsKernel(const int elem_begin, const int count, const int n_own,
                                     const int halo, const lidx_t* __restrict__ conn,
                                     const val_t* __restrict__ conn_flux,
                                     const uint8_t* __restrict__ nconn,
                                     const uint8_t* __restrict__ mat_idx,
                                     const Material* __restrict__ materials,
                                     const val_t* __restrict__ energy_in,
                                     val_t* __restrict__ energy_out, val_t* __restrict__ flux) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = elem_begin + t;

    const Material mat = materials[mat_idx[i]];
    const val_t self_energy = energy_in[i + halo];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const int nc = nconn[i];
    for (int j = 0; j < nc; ++j) {
        const size_t off = static_cast<size_t>(j) * n_own + i;
        total_flux += computeFlux(mat, self_energy, conn_flux[off], energy_in[conn[off]]);
    }

    // Update element state
    energy_out[i + halo] = self_energy + total_flux;
    flux[i] += fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Distributed simulation
// ---------------------------------------------------------------------------

struct DeviceState {
    lidx_t* conn = nullptr;
    val_t* conn_flux = nullptr;
    uint8_t* nconn = nullptr;
    uint8_t* mat = nullptr;
    Material* materials = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux = nullptr;
};

// Runs the simulation on this rank's slice; on return d.energy[cur] / d.flux
// hold the final state (the returned index selects the current energy buffer).
static int runSimulation(const MeshPartition& part, const int n_iters, DeviceState& d,
                         cudaStream_t compute_stream,
                         cudaStream_t halo_stream, val_t* h_send_lo, val_t* h_send_hi,
                         val_t* h_recv_lo, val_t* h_recv_hi) {
    const int R = part.n_elems_root;
    const int n_own = part.n_own;
    const bool has_halo = (part.rank_lo != MPI_PROC_NULL) || (part.rank_hi != MPI_PROC_NULL);
    const size_t row_bytes = static_cast<size_t>(R) * sizeof(val_t);

    constexpr int block = 256;
    const auto grid = [](const int n) { return (n + block - 1) / block; };

    cudaEvent_t ev_compute, ev_halo;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_halo, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev_compute, compute_stream));

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        if (n_own == 0) continue;
        const val_t* e_in = d.energy[cur];
        val_t* e_out = d.energy[1 - cur];

        if (!has_halo) {
            updateElementsKernel<<<grid(n_own), block, 0, compute_stream>>>(
                0, n_own, n_own, R, d.conn, d.conn_flux, d.nconn, d.mat, d.materials, e_in, e_out,
                d.flux);
        } else {
            // Stage the boundary rows for exchange as soon as the previous
            // iteration's update has produced them...
            CUDA_CHECK(cudaStreamWaitEvent(halo_stream, ev_compute, 0));
            if (part.rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_lo, e_in + R, row_bytes, cudaMemcpyDeviceToHost,
                                           halo_stream));
            }
            if (part.rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_hi, e_in + part.n_rows * R, row_bytes,
                                           cudaMemcpyDeviceToHost, halo_stream));
            }

            // ...and overlap the communication with the interior update, which
            // does not touch the ghost rows.
            const int interior_begin = R;
            const int interior_count = std::max(n_own - 2 * R, 0);
            if (interior_count > 0) {
                updateElementsKernel<<<grid(interior_count), block, 0, compute_stream>>>(
                    interior_begin, interior_count, n_own, R, d.conn, d.conn_flux, d.nconn, d.mat,
                    d.materials, e_in, e_out, d.flux);
            }

            CUDA_CHECK(cudaStreamSynchronize(halo_stream));
            MPI_Request reqs[4];
            int n_req = 0;
            if (part.rank_lo != MPI_PROC_NULL) {
                MPI_Irecv(h_recv_lo, R, MPI_DOUBLE, part.rank_lo, 0, MPI_COMM_WORLD,
                          &reqs[n_req++]);
                MPI_Isend(h_send_lo, R, MPI_DOUBLE, part.rank_lo, 1, MPI_COMM_WORLD,
                          &reqs[n_req++]);
            }
            if (part.rank_hi != MPI_PROC_NULL) {
                MPI_Irecv(h_recv_hi, R, MPI_DOUBLE, part.rank_hi, 1, MPI_COMM_WORLD,
                          &reqs[n_req++]);
                MPI_Isend(h_send_hi, R, MPI_DOUBLE, part.rank_hi, 0, MPI_COMM_WORLD,
                          &reqs[n_req++]);
            }
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

            // The ghost rows are part of the input buffer, so they must be
            // written into the buffer the boundary update reads from.
            val_t* e_in_rw = d.energy[cur];
            if (part.rank_lo != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_in_rw, h_recv_lo, row_bytes, cudaMemcpyHostToDevice,
                                           halo_stream));
            }
            if (part.rank_hi != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(e_in_rw + (part.n_rows + 1) * R, h_recv_hi, row_bytes,
                                           cudaMemcpyHostToDevice, halo_stream));
            }
            CUDA_CHECK(cudaEventRecord(ev_halo, halo_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ev_halo, 0));

            // Boundary rows (first and last owned row).
            updateElementsKernel<<<grid(R), block, 0, compute_stream>>>(
                0, R, n_own, R, d.conn, d.conn_flux, d.nconn, d.mat, d.materials, e_in, e_out,
                d.flux);
            if (part.n_rows > 1) {
                updateElementsKernel<<<grid(R), block, 0, compute_stream>>>(
                    n_own - R, R, n_own, R, d.conn, d.conn_flux, d.nconn, d.mat, d.materials, e_in,
                    e_out, d.flux);
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_compute, compute_stream));

        // Swap buffers
        cur = 1 - cur;
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(ev_compute));
    CUDA_CHECK(cudaEventDestroy(ev_halo));
    return cur;
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
    int mpi_provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided);
    (void)mpi_provided;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind each rank to one GPU, distributing the ranks of a node over the
    // devices that node provides.
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation outside the timed region

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Parallelization: %d MPI rank(s) x %d OpenMP thread(s), %d CUDA device(s)/node\n",
               size, omp_get_max_threads(), device_count);
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }

    const std::vector<Material> materials = buildMaterials();
    MeshPartition part;
    buildSquare2DPartition(part, n_elems_root, rank, size);

    if (rank == 0) {
        // Calculate memory usage
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // ---- Upload this rank's slice to its device -------------------------
    const int R = n_elems_root;
    const size_t n_own = static_cast<size_t>(part.n_own);
    const size_t n_padded = n_own + 2 * static_cast<size_t>(R);

    DeviceState d;
    cudaStream_t compute_stream, halo_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&halo_stream));

    CUDA_CHECK(cudaMalloc(&d.materials, materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(d.materials, materials.data(), materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));

    if (n_own > 0) {
        CUDA_CHECK(cudaMalloc(&d.conn, n_own * MAX_CONNECTIONS * sizeof(lidx_t)));
        CUDA_CHECK(cudaMalloc(&d.conn_flux, n_own * MAX_CONNECTIONS * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.nconn, n_own * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d.mat, n_own * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d.energy[0], n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.energy[1], n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.flux, n_own * sizeof(val_t)));

        CUDA_CHECK(cudaMemcpy(d.conn, part.conn.data(), n_own * MAX_CONNECTIONS * sizeof(lidx_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.conn_flux, part.conn_flux.data(),
                              n_own * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.nconn, part.nconn.data(), n_own * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.mat, part.mat.data(), n_own * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        // Initialize all elements with zero energy and zero flux
        CUDA_CHECK(cudaMemset(d.energy[0], 0, n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.energy[1], 0, n_padded * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.flux, 0, n_own * sizeof(val_t)));
    }

    val_t* h_send_lo = nullptr;
    val_t* h_send_hi = nullptr;
    val_t* h_recv_lo = nullptr;
    val_t* h_recv_hi = nullptr;
    if (n_own > 0 && (part.rank_lo != MPI_PROC_NULL || part.rank_hi != MPI_PROC_NULL)) {
        CUDA_CHECK(cudaHostAlloc(&h_send_lo, R * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_send_hi, R * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv_lo, R * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv_hi, R * sizeof(val_t), cudaHostAllocDefault));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int cur = runSimulation(part, n_iters, d, compute_stream, halo_stream, h_send_lo,
                                  h_send_hi, h_recv_lo, h_recv_hi);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // ---- Collect the distributed state on rank 0 ------------------------
    std::vector<val_t> local_energy(n_own);
    std::vector<val_t> local_flux(n_own);
    if (n_own > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), d.energy[cur] + R, n_own * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), d.flux, n_own * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const int rb = rowBegin(r, size, R);
        counts[r] = (rowBegin(r + 1, size, R) - rb) * R;
        displs[r] = rb * R;
    }
    std::vector<val_t> all_energy, all_flux;
    if (rank == 0) {
        all_energy.resize(static_cast<size_t>(n_elems));
        all_flux.resize(static_cast<size_t>(n_elems));
    }
    MPI_Gatherv(local_energy.data(), part.n_own, MPI_DOUBLE, all_energy.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), part.n_own, MPI_DOUBLE, all_flux.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        std::vector<ElementDynamic> elements_dynamic(static_cast<size_t>(n_elems));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_elems; ++i) {
            elements_dynamic[i].current_energy = all_energy[i];
            elements_dynamic[i].total_flux = all_flux[i];
        }

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
        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
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

    // ---- Cleanup --------------------------------------------------------
    if (h_send_lo) CUDA_CHECK(cudaFreeHost(h_send_lo));
    if (h_send_hi) CUDA_CHECK(cudaFreeHost(h_send_hi));
    if (h_recv_lo) CUDA_CHECK(cudaFreeHost(h_recv_lo));
    if (h_recv_hi) CUDA_CHECK(cudaFreeHost(h_recv_hi));
    if (d.conn) CUDA_CHECK(cudaFree(d.conn));
    if (d.conn_flux) CUDA_CHECK(cudaFree(d.conn_flux));
    if (d.nconn) CUDA_CHECK(cudaFree(d.nconn));
    if (d.mat) CUDA_CHECK(cudaFree(d.mat));
    if (d.energy[0]) CUDA_CHECK(cudaFree(d.energy[0]));
    if (d.energy[1]) CUDA_CHECK(cudaFree(d.energy[1]));
    if (d.flux) CUDA_CHECK(cudaFree(d.flux));
    if (d.materials) CUDA_CHECK(cudaFree(d.materials));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(halo_stream));

    MPI_Finalize();
    return exit_code;
}
