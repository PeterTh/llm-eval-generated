// Unstructured mesh energy transfer benchmark.
//
// Hybrid parallelization:
//   * MPI    - the global element range is block-partitioned across ranks; every
//              rank builds only its own part of the mesh and exchanges a general
//              (index-list based) halo of neighbor energies each iteration.
//   * CUDA   - the per-iteration element update runs on the GPU (one GPU per
//              rank, selected by node-local rank). Interior elements are
//              updated while the halo exchange is in flight; boundary elements
//              are updated once the halo has arrived.
//   * OpenMP - host-side work (mesh construction, index remapping, halo set
//              construction, hashing, result conversion) is multi-threaded.
//
// The numerical semantics of the original serial code are preserved exactly:
// every element accumulates its flux contributions in the original connection
// order, the hash is an XOR reduction (order independent), and all printed
// statistics are computed on rank 0 in the original sequential order.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Global indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (rank-local portion of the mesh; connectivity uses global indices)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    idx_t n_global = 0;   // Total number of elements in the mesh
    idx_t local_begin = 0;  // First global index owned by this rank
    idx_t local_end = 0;    // One past the last global index owned by this rank
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Block partition of [0, n) across `size` ranks: rank r owns [begin(r), begin(r+1))
static inline idx_t partitionBegin(idx_t n, int size, int rank) {
    return (n * static_cast<idx_t>(rank)) / static_cast<idx_t>(size);
}

// Owner rank of global element index g under the block partition
static inline int ownerOf(idx_t g, idx_t n, int size) {
    // Initial guess from the inverse of partitionBegin, corrected by bounds checks.
    int r = static_cast<int>((static_cast<__uint128_t>(g) * size + size - 1) / n);
    if (r >= size) r = size - 1;
    while (r > 0 && partitionBegin(n, size, r) > g) --r;
    while (r + 1 < size && partitionBegin(n, size, r + 1) <= g) ++r;
    return r;
}

// Build a 2D square grid as an unstructured mesh (rank-local portion)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int mpi_rank, int mpi_size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    world.n_global = n_elems;
    world.local_begin = partitionBegin(n_elems, mpi_size, mpi_rank);
    world.local_end = partitionBegin(n_elems, mpi_size, mpi_rank + 1);
    const idx_t n_local = world.local_end - world.local_begin;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    const idx_t last = static_cast<idx_t>(n_elems_root - 1);
    const idx_t root = static_cast<idx_t>(n_elems_root);

    // Initialize elements and build connectivity: each element connects to its
    // neighbors in the 2D grid (same neighbor order as the serial reference).
#pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = world.local_begin + li;
        const idx_t x = idx / root;
        const idx_t y = idx % root;

        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        world.elements_dynamic[li].current_energy = 0.0;
        world.elements_dynamic[li].total_flux = 0.0;

        // Connect to neighbors (up, down, left, right)
        const int64_t offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = static_cast<int64_t>(x) + offsets[n][0];
            const int64_t ny = static_cast<int64_t>(y) + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * root + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        if (idx == 0 * root + 0) elem.material_idx = INFLOW_MAT_ID;
        if (idx == 0 * root + last) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * root + 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * root + last) elem.material_idx = INFLOW_MAT_ID;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute energy flux between two elements (identical arithmetic to the reference)
__device__ __forceinline__ val_t computeFluxDev(val_t transfer_coeff, val_t this_energy,
                                                val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update a list of elements. Connectivity is stored in a transposed ELL layout
// (connection-major) so that consecutive threads access consecutive memory.
// Neighbor indices are local indices into `energy` (owned elements followed by
// halo elements).
__global__ void updateElementsKernel(const idx_t* __restrict__ list, idx_t list_count,
                                     const uint32_t* __restrict__ material_idx,
                                     const uint32_t* __restrict__ num_connections,
                                     const uint32_t* __restrict__ conn_idx,
                                     const val_t* __restrict__ conn_flux,
                                     const Material* __restrict__ materials,
                                     const val_t* __restrict__ energy,
                                     const val_t* __restrict__ flux_in,
                                     val_t* __restrict__ energy_out,
                                     val_t* __restrict__ flux_out,
                                     idx_t n_local) {
    const idx_t k = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= list_count) return;
    const idx_t i = list ? list[k] : k;

    const Material mat = materials[material_idx[i]];
    const val_t e = energy[i];
    const uint32_t nc = num_connections[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (original connection order)
    for (uint32_t j = 0; j < nc; ++j) {
        const idx_t slot = static_cast<idx_t>(j) * n_local + i;
        const val_t other = energy[conn_idx[slot]];
        total_flux += computeFluxDev(mat.transfer_coeff, e, conn_flux[slot], other);
    }

    // Update element state
    energy_out[i] = e + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Gather energies of elements that must be sent to other ranks
__global__ void packHaloKernel(const idx_t* __restrict__ send_idx, idx_t count,
                               const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const idx_t k = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) out[k] = energy[send_idx[k]];
}

// ---------------------------------------------------------------------------
// Distributed GPU simulation state
// ---------------------------------------------------------------------------
struct GpuSim {
    idx_t n_local = 0;
    idx_t n_halo = 0;
    uint32_t max_conn = 0;

    // Device static data
    uint32_t* d_material_idx = nullptr;
    uint32_t* d_num_conn = nullptr;
    uint32_t* d_conn_idx = nullptr;   // [max_conn][n_local]
    val_t* d_conn_flux = nullptr;     // [max_conn][n_local]
    Material* d_materials = nullptr;
    idx_t* d_interior = nullptr;
    idx_t* d_boundary = nullptr;
    idx_t n_interior = 0;
    idx_t n_boundary = 0;

    // Device dynamic data (energy arrays have n_local + n_halo entries)
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};

    // Halo exchange
    idx_t* d_send_idx = nullptr;
    val_t* d_send_buf = nullptr;
    val_t* h_send_buf = nullptr;  // pinned
    val_t* h_recv_buf = nullptr;  // pinned
    idx_t n_send = 0;
    std::vector<int> send_ranks, recv_ranks;
    std::vector<idx_t> send_offsets, send_counts;  // per send_ranks entry
    std::vector<idx_t> recv_offsets, recv_counts;  // per recv_ranks entry

    cudaStream_t s_comp = nullptr;
    cudaStream_t s_comm = nullptr;
    cudaEvent_t ev_halo_ready = nullptr;
    cudaEvent_t ev_iter_done = nullptr;
    cudaEvent_t ev_pack_done = nullptr;
};

// Build the halo lists, remap connectivity to local indices, and upload to the GPU
static void setupGpuSim(const World& world, GpuSim& sim, MPI_Comm comm) {
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &size));

    const idx_t n_local = world.elements_static.size();
    const idx_t n_global = world.n_global;
    const idx_t lo = world.local_begin;
    const idx_t hi = world.local_end;
    sim.n_local = n_local;

    // --- Collect the set of remote global indices referenced by local elements
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<idx_t>> remote_per_thread(n_threads);
    uint32_t max_conn = 0;
#pragma omp parallel reduction(max : max_conn)
    {
        std::vector<idx_t>& mine = remote_per_thread[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (idx_t i = 0; i < n_local; ++i) {
            const ElementStatic& es = world.elements_static[i];
            max_conn = std::max<uint32_t>(max_conn, static_cast<uint32_t>(es.num_connections));
            for (idx_t j = 0; j < es.num_connections; ++j) {
                const idx_t g = es.connected_idx[j];
                if (g < lo || g >= hi) mine.push_back(g);
            }
        }
        std::sort(mine.begin(), mine.end());
        mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
    }
    std::vector<idx_t> halo;  // sorted unique remote global indices
    for (auto& v : remote_per_thread) {
        std::vector<idx_t> merged(halo.size() + v.size());
        std::merge(halo.begin(), halo.end(), v.begin(), v.end(), merged.begin());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
        halo.swap(merged);
    }
    sim.n_halo = halo.size();
    sim.max_conn = max_conn;

    // Because ownership is a block partition, the sorted halo list is grouped by
    // owner rank in increasing rank order, so the receive buffer can be used
    // directly as the halo section of the energy array.
    std::vector<idx_t> recv_count_per_rank(size, 0);
    for (idx_t g : halo) recv_count_per_rank[ownerOf(g, n_global, size)]++;

    // --- Exchange halo requests: tell each owner which of its elements we need
    std::vector<int> recv_counts_int(size), send_counts_int(size);
    for (int r = 0; r < size; ++r) recv_counts_int[r] = static_cast<int>(recv_count_per_rank[r]);
    MPI_CHECK(MPI_Alltoall(recv_counts_int.data(), 1, MPI_INT, send_counts_int.data(), 1, MPI_INT, comm));

    std::vector<int> recv_displs(size + 1, 0), send_displs(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts_int[r];
        send_displs[r + 1] = send_displs[r] + send_counts_int[r];
    }
    sim.n_send = static_cast<idx_t>(send_displs[size]);
    std::vector<idx_t> send_global(sim.n_send);
    MPI_CHECK(MPI_Alltoallv(halo.data(), recv_counts_int.data(), recv_displs.data(), MPI_UINT64_T,
                            send_global.data(), send_counts_int.data(), send_displs.data(),
                            MPI_UINT64_T, comm));

    std::vector<idx_t> send_idx(sim.n_send);
#pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < sim.n_send; ++k) send_idx[k] = send_global[k] - lo;

    for (int r = 0; r < size; ++r) {
        if (send_counts_int[r] > 0) {
            sim.send_ranks.push_back(r);
            sim.send_offsets.push_back(send_displs[r]);
            sim.send_counts.push_back(send_counts_int[r]);
        }
        if (recv_counts_int[r] > 0) {
            sim.recv_ranks.push_back(r);
            sim.recv_offsets.push_back(recv_displs[r]);
            sim.recv_counts.push_back(recv_counts_int[r]);
        }
    }

    // --- Remap connectivity to local indices in a transposed ELL layout and
    //     classify elements as interior (no remote neighbors) or boundary.
    const idx_t ell_size = static_cast<idx_t>(max_conn) * n_local;
    std::vector<uint32_t> h_material_idx(n_local), h_num_conn(n_local);
    std::vector<uint32_t> h_conn_idx(ell_size, 0);
    std::vector<val_t> h_conn_flux(ell_size, 0.0);
    std::vector<uint8_t> is_boundary(n_local, 0);
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        h_material_idx[i] = static_cast<uint32_t>(es.material_idx);
        h_num_conn[i] = static_cast<uint32_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            idx_t local;
            if (g >= lo && g < hi) {
                local = g - lo;
            } else {
                const auto it = std::lower_bound(halo.begin(), halo.end(), g);
                local = n_local + static_cast<idx_t>(it - halo.begin());
                is_boundary[i] = 1;
            }
            h_conn_idx[j * n_local + i] = static_cast<uint32_t>(local);
            h_conn_flux[j * n_local + i] = es.connected_flux[j];
        }
    }

    std::vector<idx_t> interior, boundary;
    interior.reserve(n_local);
    for (idx_t i = 0; i < n_local; ++i) {
        if (is_boundary[i]) boundary.push_back(i);
        else interior.push_back(i);
    }
    sim.n_interior = interior.size();
    sim.n_boundary = boundary.size();

    // --- Initial dynamic state (SoA)
    std::vector<val_t> h_energy(n_local + sim.n_halo, 0.0), h_flux(n_local, 0.0);
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_local; ++i) {
        h_energy[i] = world.elements_dynamic[i].current_energy;
        h_flux[i] = world.elements_dynamic[i].total_flux;
    }

    // --- Device allocation and upload
    auto upload = [](auto*& dptr, const auto& host) {
        using T = std::remove_reference_t<decltype(host[0])>;
        const size_t bytes = std::max<size_t>(host.size(), 1) * sizeof(T);
        CUDA_CHECK(cudaMalloc(&dptr, bytes));
        if (!host.empty()) CUDA_CHECK(cudaMemcpy(dptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    };
    upload(sim.d_material_idx, h_material_idx);
    upload(sim.d_num_conn, h_num_conn);
    upload(sim.d_conn_idx, h_conn_idx);
    upload(sim.d_conn_flux, h_conn_flux);
    upload(sim.d_materials, world.materials);
    upload(sim.d_interior, interior);
    upload(sim.d_boundary, boundary);
    upload(sim.d_send_idx, send_idx);
    upload(sim.d_energy[0], h_energy);
    upload(sim.d_energy[1], h_energy);
    upload(sim.d_flux[0], h_flux);
    upload(sim.d_flux[1], h_flux);
    CUDA_CHECK(cudaMalloc(&sim.d_send_buf, std::max<idx_t>(sim.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_send_buf, std::max<idx_t>(sim.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recv_buf, std::max<idx_t>(sim.n_halo, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_halo_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_iter_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_pack_done, cudaEventDisableTiming));

    // Warm up: load kernel modules and touch the streams outside the timed region.
    packHaloKernel<<<1, 32, 0, sim.s_comm>>>(sim.d_send_idx, 0, sim.d_energy[0], sim.d_send_buf);
    updateElementsKernel<<<1, 32, 0, sim.s_comp>>>(sim.d_interior, 0, sim.d_material_idx, sim.d_num_conn,
                                                  sim.d_conn_idx, sim.d_conn_flux, sim.d_materials,
                                                  sim.d_energy[0], sim.d_flux[0], sim.d_energy[1],
                                                  sim.d_flux[1], sim.n_local);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(sim.ev_iter_done, sim.s_comp));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void destroyGpuSim(GpuSim& sim) {
    cudaFree(sim.d_material_idx);
    cudaFree(sim.d_num_conn);
    cudaFree(sim.d_conn_idx);
    cudaFree(sim.d_conn_flux);
    cudaFree(sim.d_materials);
    cudaFree(sim.d_interior);
    cudaFree(sim.d_boundary);
    cudaFree(sim.d_send_idx);
    cudaFree(sim.d_send_buf);
    for (int b = 0; b < 2; ++b) {
        cudaFree(sim.d_energy[b]);
        cudaFree(sim.d_flux[b]);
    }
    cudaFreeHost(sim.h_send_buf);
    cudaFreeHost(sim.h_recv_buf);
    cudaStreamDestroy(sim.s_comp);
    cudaStreamDestroy(sim.s_comm);
    cudaEventDestroy(sim.ev_halo_ready);
    cudaEventDestroy(sim.ev_iter_done);
    cudaEventDestroy(sim.ev_pack_done);
}

// Run simulation for n_iters iterations (distributed, GPU accelerated)
void runSimulation(GpuSim& sim, const int n_iters, MPI_Comm comm) {
    constexpr int BLOCK = 256;
    auto grid = [](idx_t n) { return static_cast<unsigned>((n + BLOCK - 1) / BLOCK); };

    const bool has_halo = !sim.send_ranks.empty() || !sim.recv_ranks.empty();
    std::vector<MPI_Request> reqs;
    reqs.reserve(sim.send_ranks.size() + sim.recv_ranks.size());

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* e_cur = sim.d_energy[cur];
        val_t* e_new = sim.d_energy[cur ^ 1];
        val_t* f_cur = sim.d_flux[cur];
        val_t* f_new = sim.d_flux[cur ^ 1];

        if (has_halo) {
            // Post receives first, then pack and ship the halo on the comm stream.
            reqs.clear();
            for (size_t k = 0; k < sim.recv_ranks.size(); ++k) {
                reqs.emplace_back();
                MPI_CHECK(MPI_Irecv(sim.h_recv_buf + sim.recv_offsets[k], static_cast<int>(sim.recv_counts[k]),
                                    MPI_DOUBLE, sim.recv_ranks[k], 0, comm, &reqs.back()));
            }
            // The comm stream must see the energies produced by the previous iteration.
            CUDA_CHECK(cudaStreamWaitEvent(sim.s_comm, sim.ev_iter_done, 0));
            if (sim.n_send > 0) {
                packHaloKernel<<<grid(sim.n_send), BLOCK, 0, sim.s_comm>>>(sim.d_send_idx, sim.n_send, e_cur,
                                                                           sim.d_send_buf);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(sim.h_send_buf, sim.d_send_buf, sim.n_send * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, sim.s_comm));
            }
            CUDA_CHECK(cudaEventRecord(sim.ev_pack_done, sim.s_comm));
        }

        // Interior elements do not depend on the halo: overlap with communication.
        if (sim.n_interior > 0) {
            updateElementsKernel<<<grid(sim.n_interior), BLOCK, 0, sim.s_comp>>>(
                sim.d_interior, sim.n_interior, sim.d_material_idx, sim.d_num_conn, sim.d_conn_idx,
                sim.d_conn_flux, sim.d_materials, e_cur, f_cur, e_new, f_new, sim.n_local);
            CUDA_CHECK(cudaGetLastError());
        }

        if (has_halo) {
            CUDA_CHECK(cudaEventSynchronize(sim.ev_pack_done));
            for (size_t k = 0; k < sim.send_ranks.size(); ++k) {
                reqs.emplace_back();
                MPI_CHECK(MPI_Isend(sim.h_send_buf + sim.send_offsets[k], static_cast<int>(sim.send_counts[k]),
                                    MPI_DOUBLE, sim.send_ranks[k], 0, comm, &reqs.back()));
            }
            MPI_CHECK(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));
            if (sim.n_halo > 0) {
                CUDA_CHECK(cudaMemcpyAsync(e_cur + sim.n_local, sim.h_recv_buf, sim.n_halo * sizeof(val_t),
                                           cudaMemcpyHostToDevice, sim.s_comm));
            }
            CUDA_CHECK(cudaEventRecord(sim.ev_halo_ready, sim.s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(sim.s_comp, sim.ev_halo_ready, 0));
        }

        if (sim.n_boundary > 0) {
            updateElementsKernel<<<grid(sim.n_boundary), BLOCK, 0, sim.s_comp>>>(
                sim.d_boundary, sim.n_boundary, sim.d_material_idx, sim.d_num_conn, sim.d_conn_idx,
                sim.d_conn_flux, sim.d_materials, e_cur, f_cur, e_new, f_new, sim.n_local);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(sim.ev_iter_done, sim.s_comp));

        // Swap buffers
        cur ^= 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(sim.s_comp));
    CUDA_CHECK(cudaStreamSynchronize(sim.s_comm));

    // Make sure the "current" buffers are at index 0 for the download step.
    if (cur != 0) {
        std::swap(sim.d_energy[0], sim.d_energy[1]);
        std::swap(sim.d_flux[0], sim.d_flux[1]);
    }
}

// Download the final local state from the GPU back into the World
static void downloadResults(const GpuSim& sim, World& world) {
    const idx_t n_local = sim.n_local;
    std::vector<val_t> h_energy(n_local), h_flux(n_local);
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(h_energy.data(), sim.d_energy[0], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), sim.d_flux[0], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_local; ++i) {
        world.elements_dynamic[i].current_energy = h_energy[i];
        world.elements_dynamic[i].total_flux = h_flux[i];
    }
}

// Validate simulation results (rank 0, over the gathered global state)
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
// The hash is an XOR of per-element terms (keyed by global index), so it can be
// computed as a parallel/distributed XOR reduction with an identical result.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, idx_t global_offset, MPI_Comm comm) {
    uint64_t hash = 0;
    const idx_t n = elements.size();
#pragma omp parallel for schedule(static) reduction(^ : hash)
    for (idx_t li = 0; li < n; ++li) {
        const uint64_t i = global_offset + li;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[li].current_energy, sizeof(uint64_t));
        std::memcpy(&f_bits, &elements[li].total_flux, sizeof(uint64_t));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_CHECK(MPI_Allreduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, comm));
    return global_hash;
}

// Gather the full dynamic state onto rank 0 (in global index order)
static std::vector<ElementDynamic> gatherDynamic(const World& world, MPI_Comm comm) {
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &size));

    MPI_Datatype dyn_type;
    MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &dyn_type));
    MPI_CHECK(MPI_Type_commit(&dyn_type));

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(partitionBegin(world.n_global, size, r + 1) - partitionBegin(world.n_global, size, r));
        displs[r] = static_cast<int>(partitionBegin(world.n_global, size, r));
    }
    std::vector<ElementDynamic> all;
    if (rank == 0) all.resize(world.n_global);
    MPI_CHECK(MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(world.elements_dynamic.size()), dyn_type,
                          all.data(), counts.data(), displs.data(), dyn_type, 0, comm));
    MPI_CHECK(MPI_Type_free(&dyn_type));
    return all;
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int mpi_rank = 0, mpi_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

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

    // Select one GPU per rank based on the node-local rank
    {
        MPI_Comm node_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &node_comm));
        int node_rank = 0;
        MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
        MPI_CHECK(MPI_Comm_free(&node_comm));
        int n_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_devices));
        if (n_devices <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(node_rank % n_devices));
        CUDA_CHECK(cudaFree(0));  // establish the context now, outside the timed region
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mpi_size, omp_get_max_threads());
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, mpi_rank, mpi_size);

    // Calculate memory usage (global totals)
    if (mpi_rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Set up halo exchange and upload the mesh to the GPU
    GpuSim sim;
    setupGpuSim(world, sim, MPI_COMM_WORLD);

    // Run simulation
    if (mpi_rank == 0) printf("Running simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters, MPI_COMM_WORLD);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    downloadResults(sim, world);
    destroyGpuSim(sim);

    // Compute hash for verification (distributed XOR reduction)
    const uint64_t hash = computeHash(world.elements_dynamic, world.local_begin, MPI_COMM_WORLD);

    // Gather the global state on rank 0 only when it is needed for output
    std::vector<ElementDynamic> all_dynamic;
    if (validate || printResults) all_dynamic = gatherDynamic(world, MPI_COMM_WORLD);

    int exit_code = 0;
    if (mpi_rank == 0) {
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
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(all_dynamic.size());
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < all_dynamic.size(); ++i) energyData[i] = all_dynamic[i].current_energy;
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(all_dynamic);
            if (!valid) exit_code = 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}
