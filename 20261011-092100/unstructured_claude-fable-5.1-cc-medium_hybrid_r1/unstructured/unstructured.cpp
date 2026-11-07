// Unstructured mesh energy transfer benchmark -- hybrid MPI + OpenMP + CUDA.
//
// Parallelization strategy
//  * MPI:    the global element set is decomposed into contiguous row blocks
//            (1D domain decomposition).  Each rank owns a block of rows plus
//            one halo row on each side.  Only the halo rows are exchanged per
//            iteration (via pinned host staging buffers so that the code works
//            with any MPI implementation, CUDA-aware or not).
//  * CUDA:   every rank drives one GPU.  The per-element update is executed by
//            one CUDA thread per element on a structure-of-arrays copy of the
//            connectivity so that all accesses are coalesced.  The interior
//            update runs on one stream while the halo exchange and the
//            boundary-row update run on a second stream, overlapping
//            communication with computation.
//  * OpenMP: host side work (mesh construction, SoA packing, result
//            post-processing and hashing) is multi-threaded on every rank.
//
// Numerical semantics are identical to the original serial code: the same
// floating point expressions are evaluated in the same order for every element.

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
using lidx_t = uint32_t;  // local (per-rank) element index used on the device

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

// Distributed world state. Element indices stored in the connectivity are
// *local* indices into an array laid out as
//   [ lower halo row | owned rows (n_local_rows) | upper halo row ]
// Each row contains `n_elems_root` elements.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;   // owned elements only
    std::vector<ElementDynamic> elements_dynamic; // owned elements only (host copy)

    // Decomposition information
    int n_elems_root = 0;   // global grid edge length
    int row_begin = 0;      // first owned global row
    int row_end = 0;        // one past last owned global row
    int rank_prev = MPI_PROC_NULL;
    int rank_next = MPI_PROC_NULL;

    int n_local_rows() const { return row_end - row_begin; }
    size_t n_owned() const { return static_cast<size_t>(n_local_rows()) * n_elems_root; }
    size_t n_local() const { return static_cast<size_t>(n_local_rows() + 2) * n_elems_root; }
    // Global element index -> local element index (halo offset of one row)
    size_t to_local(idx_t global) const {
        return static_cast<size_t>(global) - static_cast<size_t>(row_begin) * n_elems_root +
               static_cast<size_t>(n_elems_root);
    }
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// Row block decomposition helpers: rank r owns rows [rowStart(r), rowStart(r+1))
static inline int rowStart(int rank, int n_rows, int n_ranks) {
    return static_cast<int>((static_cast<long long>(n_rows) * rank) / n_ranks);
}

static int rowOwner(int row, int n_rows, int n_ranks) {
    for (int r = 0; r < n_ranks; ++r) {
        if (rowStart(r, n_rows, n_ranks) <= row && row < rowStart(r + 1, n_rows, n_ranks)) {
            return r;
        }
    }
    return MPI_PROC_NULL;
}

// Build a 2D square grid as an unstructured mesh (local part of this rank)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int rank, const int n_ranks) {
    world.n_elems_root = n_elems_root;
    world.row_begin = rowStart(rank, n_elems_root, n_ranks);
    world.row_end = rowStart(rank + 1, n_elems_root, n_ranks);
    world.rank_prev = (world.row_begin > 0 && world.n_local_rows() > 0)
                          ? rowOwner(world.row_begin - 1, n_elems_root, n_ranks)
                          : MPI_PROC_NULL;
    world.rank_next = (world.row_end < n_elems_root && world.n_local_rows() > 0)
                          ? rowOwner(world.row_end, n_elems_root, n_ranks)
                          : MPI_PROC_NULL;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate owned elements
    const size_t n_owned = world.n_owned();
    world.elements_static.resize(n_owned);
    world.elements_dynamic.resize(n_owned);

    const int row_begin = world.row_begin;
    const int row_end = world.row_end;

    // Initialize all elements with default material and zero energy, and build
    // connectivity: each element connects to its neighbors in the 2D grid
#pragma omp parallel for schedule(static)
    for (int x = row_begin; x < row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const size_t owned_idx = static_cast<size_t>(x - row_begin) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[owned_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            world.elements_dynamic[owned_idx].current_energy = 0.0;
            world.elements_dynamic[owned_idx].total_flux = 0.0;

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const idx_t neighbor_idx =
                        static_cast<idx_t>(nx) * n_elems_root + static_cast<idx_t>(ny);
                    elem.connected_idx[elem.num_connections] = world.to_local(neighbor_idx);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto set_material = [&](int x, int y, idx_t mat) {
        if (x >= row_begin && x < row_end) {
            world.elements_static[static_cast<size_t>(x - row_begin) * n_elems_root + y].material_idx = mat;
        }
    };
    set_material(0, 0, INFLOW_MAT_ID);
    set_material(0, last, OUTFLOW_MAT_ID);
    set_material(last, 0, OUTFLOW_MAT_ID);
    set_material(last, last, INFLOW_MAT_ID);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute energy flux between two elements (same expression/order as original)
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update owned elements with local indices in [first, first + count).
// Connectivity is stored structure-of-arrays: conn_idx[j * n_owned + k] where k
// is the owned-element index (local index minus one halo row).
__global__ void __launch_bounds__(256)
updateElementsKernel(const val_t* __restrict__ energy_in,
                     const val_t* __restrict__ tflux_in,
                     val_t* __restrict__ energy_out,
                     val_t* __restrict__ tflux_out,
                     const Material* __restrict__ materials,
                     const int32_t* __restrict__ material_idx,
                     const int32_t* __restrict__ num_connections,
                     const lidx_t* __restrict__ conn_idx,
                     const val_t* __restrict__ conn_flux,
                     const size_t n_owned, const size_t halo_offset,
                     const size_t first, const size_t count) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;

    const size_t li = first + t;            // local index (incl. halo offset)
    const size_t k = li - halo_offset;      // owned-element index

    const Material mat = materials[material_idx[k]];
    const val_t e = energy_in[li];
    const int32_t nconn = num_connections[k];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (int32_t j = 0; j < nconn; ++j) {
        const lidx_t neighbor_idx = conn_idx[static_cast<size_t>(j) * n_owned + k];
        const val_t cf = conn_flux[static_cast<size_t>(j) * n_owned + k];
        total_flux += computeFlux(mat.transfer_coeff, e, cf, energy_in[neighbor_idx]);
    }

    // Update element state
    energy_out[li] = e + total_flux;
    tflux_out[li] = tflux_in[li] + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Device-side simulation state
// ---------------------------------------------------------------------------
struct DeviceWorld {
    val_t* energy[2] = {nullptr, nullptr};
    val_t* tflux[2] = {nullptr, nullptr};
    Material* materials = nullptr;
    int32_t* material_idx = nullptr;
    int32_t* num_connections = nullptr;
    lidx_t* conn_idx = nullptr;
    val_t* conn_flux = nullptr;

    // Pinned staging buffers for halo exchange
    val_t* h_send = nullptr;  // [0,N): row sent to prev, [N,2N): row sent to next
    val_t* h_recv = nullptr;  // [0,N): row received from prev, [N,2N): from next

    cudaStream_t stream_interior = nullptr;
    cudaStream_t stream_boundary = nullptr;
    cudaEvent_t ev_interior = nullptr;
    cudaEvent_t ev_boundary = nullptr;

    int cur = 0;  // index of the buffer holding the current state
};

static void uploadWorld(const World& world, DeviceWorld& dev) {
    const size_t n_owned = world.n_owned();
    const size_t n_local = world.n_local();
    const size_t N = world.n_elems_root;

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dev.energy[b], n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dev.tflux[b], n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.energy[b], 0, n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.tflux[b], 0, n_local * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&dev.materials, world.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(dev.materials, world.materials.data(),
                          world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev.material_idx, std::max<size_t>(n_owned, 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dev.num_connections, std::max<size_t>(n_owned, 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dev.conn_idx, std::max<size_t>(n_owned, 1) * MAX_CONNECTIONS * sizeof(lidx_t)));
    CUDA_CHECK(cudaMalloc(&dev.conn_flux, std::max<size_t>(n_owned, 1) * MAX_CONNECTIONS * sizeof(val_t)));

    CUDA_CHECK(cudaMallocHost(&dev.h_send, std::max<size_t>(2 * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&dev.h_recv, std::max<size_t>(2 * N, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&dev.stream_interior, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&dev.stream_boundary, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_interior, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&dev.ev_boundary, cudaEventDisableTiming));

    if (n_owned == 0) return;

    // Pack AoS host connectivity into SoA device layout (OpenMP parallel)
    std::vector<int32_t> h_mat(n_owned), h_nconn(n_owned);
    std::vector<lidx_t> h_cidx(n_owned * MAX_CONNECTIONS, 0);
    std::vector<val_t> h_cflux(n_owned * MAX_CONNECTIONS, 0.0);
    std::vector<val_t> h_energy(n_local, 0.0), h_tflux(n_local, 0.0);

#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n_owned; ++k) {
        const ElementStatic& es = world.elements_static[k];
        h_mat[k] = static_cast<int32_t>(es.material_idx);
        h_nconn[k] = static_cast<int32_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            h_cidx[j * n_owned + k] = static_cast<lidx_t>(es.connected_idx[j]);
            h_cflux[j * n_owned + k] = es.connected_flux[j];
        }
        h_energy[N + k] = world.elements_dynamic[k].current_energy;
        h_tflux[N + k] = world.elements_dynamic[k].total_flux;
    }

    CUDA_CHECK(cudaMemcpy(dev.material_idx, h_mat.data(), n_owned * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.num_connections, h_nconn.data(), n_owned * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.conn_idx, h_cidx.data(), h_cidx.size() * sizeof(lidx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.conn_flux, h_cflux.data(), h_cflux.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.energy[0], h_energy.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.tflux[0], h_tflux.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void downloadWorld(World& world, DeviceWorld& dev) {
    const size_t n_owned = world.n_owned();
    if (n_owned == 0) return;
    const size_t N = world.n_elems_root;
    std::vector<val_t> h_energy(n_owned), h_tflux(n_owned);
    CUDA_CHECK(cudaMemcpy(h_energy.data(), dev.energy[dev.cur] + N, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tflux.data(), dev.tflux[dev.cur] + N, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n_owned; ++k) {
        world.elements_dynamic[k].current_energy = h_energy[k];
        world.elements_dynamic[k].total_flux = h_tflux[k];
    }
}

static void freeDeviceWorld(DeviceWorld& dev) {
    for (int b = 0; b < 2; ++b) {
        cudaFree(dev.energy[b]);
        cudaFree(dev.tflux[b]);
    }
    cudaFree(dev.materials);
    cudaFree(dev.material_idx);
    cudaFree(dev.num_connections);
    cudaFree(dev.conn_idx);
    cudaFree(dev.conn_flux);
    cudaFreeHost(dev.h_send);
    cudaFreeHost(dev.h_recv);
    cudaEventDestroy(dev.ev_interior);
    cudaEventDestroy(dev.ev_boundary);
    cudaStreamDestroy(dev.stream_interior);
    cudaStreamDestroy(dev.stream_boundary);
}

static inline void launchUpdate(const DeviceWorld& dev, const World& world, cudaStream_t stream,
                                size_t first, size_t count) {
    if (count == 0) return;
    constexpr int BLOCK = 256;
    const unsigned grid = static_cast<unsigned>((count + BLOCK - 1) / BLOCK);
    updateElementsKernel<<<grid, BLOCK, 0, stream>>>(
        dev.energy[dev.cur], dev.tflux[dev.cur], dev.energy[dev.cur ^ 1], dev.tflux[dev.cur ^ 1],
        dev.materials, dev.material_idx, dev.num_connections, dev.conn_idx, dev.conn_flux,
        world.n_owned(), static_cast<size_t>(world.n_elems_root), first, count);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, DeviceWorld& dev, const int n_iters) {
    const size_t N = world.n_elems_root;
    const int n_rows = world.n_local_rows();
    if (n_rows == 0) return;  // this rank owns no elements

    const bool has_prev = world.rank_prev != MPI_PROC_NULL;
    const bool has_next = world.rank_next != MPI_PROC_NULL;

    // Local row r (1-based owned rows 1..n_rows) starts at local index r*N.
    // Boundary rows: 1 and n_rows (which coincide when n_rows == 1).
    const size_t first_row_begin = N;                 // local row 1
    const size_t last_row_begin = static_cast<size_t>(n_rows) * N;  // local row n_rows
    const size_t interior_begin = 2 * N;              // local row 2
    const size_t interior_count = (n_rows > 2) ? static_cast<size_t>(n_rows - 2) * N : 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* energy_cur = dev.energy[dev.cur];

        // Both streams must see the complete state produced by the previous
        // iteration (interior and boundary kernels wrote to the same buffer).
        if (iter > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(dev.stream_interior, dev.ev_boundary, 0));
            CUDA_CHECK(cudaStreamWaitEvent(dev.stream_boundary, dev.ev_interior, 0));
        }

        // 1) Interior update (does not depend on halo data) -- overlaps with 2)
        launchUpdate(dev, world, dev.stream_interior, interior_begin, interior_count);

        // 2) Halo exchange of the current state's boundary rows
        if (has_prev || has_next) {
            if (has_prev) {
                CUDA_CHECK(cudaMemcpyAsync(dev.h_send, energy_cur + first_row_begin, N * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, dev.stream_boundary));
            }
            if (has_next) {
                CUDA_CHECK(cudaMemcpyAsync(dev.h_send + N, energy_cur + last_row_begin, N * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, dev.stream_boundary));
            }
            CUDA_CHECK(cudaStreamSynchronize(dev.stream_boundary));

            MPI_Request reqs[4];
            int n_req = 0;
            if (has_prev) {
                MPI_Irecv(dev.h_recv, static_cast<int>(N), MPI_DOUBLE, world.rank_prev, 0, MPI_COMM_WORLD, &reqs[n_req++]);
                MPI_Isend(dev.h_send, static_cast<int>(N), MPI_DOUBLE, world.rank_prev, 1, MPI_COMM_WORLD, &reqs[n_req++]);
            }
            if (has_next) {
                MPI_Irecv(dev.h_recv + N, static_cast<int>(N), MPI_DOUBLE, world.rank_next, 1, MPI_COMM_WORLD, &reqs[n_req++]);
                MPI_Isend(dev.h_send + N, static_cast<int>(N), MPI_DOUBLE, world.rank_next, 0, MPI_COMM_WORLD, &reqs[n_req++]);
            }
            MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);

            if (has_prev) {
                CUDA_CHECK(cudaMemcpyAsync(energy_cur, dev.h_recv, N * sizeof(val_t),
                                           cudaMemcpyHostToDevice, dev.stream_boundary));
            }
            if (has_next) {
                CUDA_CHECK(cudaMemcpyAsync(energy_cur + static_cast<size_t>(n_rows + 1) * N, dev.h_recv + N,
                                           N * sizeof(val_t), cudaMemcpyHostToDevice, dev.stream_boundary));
            }
        }

        // 3) Boundary row update (needs the halo)
        launchUpdate(dev, world, dev.stream_boundary, first_row_begin, N);
        if (n_rows > 1) {
            launchUpdate(dev, world, dev.stream_boundary, last_row_begin, N);
        }

        CUDA_CHECK(cudaEventRecord(dev.ev_interior, dev.stream_interior));
        CUDA_CHECK(cudaEventRecord(dev.ev_boundary, dev.stream_boundary));

        // Swap buffers
        dev.cur ^= 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(dev.stream_interior));
    CUDA_CHECK(cudaStreamSynchronize(dev.stream_boundary));
    CUDA_CHECK(cudaGetLastError());
}

// Validate simulation results (global state, executed on rank 0)
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

// Compute a simple hash of the results for verification.
// XOR is associative and commutative, so the OpenMP reduction yields exactly
// the same value as the original sequential loop.
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    const size_t n = elements.size();
#pragma omp parallel for schedule(static) reduction(^ : hash)
    for (size_t i = 0; i < n; ++i) {
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &elements[i].current_energy, sizeof(uint64_t));
        memcpy(&f_bits, &elements[i].total_flux, sizeof(uint64_t));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    const bool is_root = (rank == 0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int exit_code = -1;  // < 0: continue, otherwise exit with this code

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
            if (is_root) printUsage(argv[0]);
            exit_code = 0;
            break;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exit_code = 1;
            break;
        }
    }
    if (exit_code >= 0) {
        MPI_Finalize();
        return exit_code;
    }

    // Select the GPU: one device per rank, round-robin over the node's devices
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int node_rank = 0;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_free(&node_comm);
        int n_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_devices));
        if (n_devices <= 0) {
            fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(node_rank % n_devices));
        CUDA_CHECK(cudaFree(nullptr));  // establish context early
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (is_root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/rank: 1\n", n_ranks, omp_get_max_threads());
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, n_ranks);

    DeviceWorld dev;
    uploadWorld(world, dev);

    if (is_root) {
        // Calculate memory usage (global problem size)
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, dev, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Collect the distributed dynamic state on rank 0 (rows are contiguous and
    // ordered by rank, so the gathered array has the original global ordering)
    downloadWorld(world, dev);
    freeDeviceWorld(dev);

    MPI_Datatype dyn_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dyn_type);
    MPI_Type_commit(&dyn_type);

    std::vector<ElementDynamic> global_dynamic;
    std::vector<int> counts, displs;
    if (is_root) {
        global_dynamic.resize(static_cast<size_t>(n_elems));
        counts.resize(n_ranks);
        displs.resize(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            const int rb = rowStart(r, n_elems_root, n_ranks);
            const int re = rowStart(r + 1, n_elems_root, n_ranks);
            counts[r] = (re - rb) * n_elems_root;
            displs[r] = rb * n_elems_root;
        }
    }
    MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(world.n_owned()), dyn_type,
                global_dynamic.data(), counts.data(), displs.data(), dyn_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&dyn_type);

    int result = 0;
    if (is_root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        const uint64_t hash = computeHash(global_dynamic);
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(global_dynamic.size());
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < global_dynamic.size(); ++i) {
                energyData[i] = global_dynamic[i].current_energy;
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(global_dynamic);
            if (!valid) {
                result = 1;
            }
        }
        fflush(stdout);
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
