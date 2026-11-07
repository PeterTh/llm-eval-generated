// Unstructured mesh energy transfer benchmark
// Hybrid MPI + OpenMP + CUDA implementation.
//
//  * MPI    : the global element index space is split into contiguous blocks,
//             one per rank. Each rank owns one GPU. Off-rank neighbours
//             ("ghost" elements) are discovered from the connectivity itself,
//             so the exchange pattern works for arbitrary unstructured meshes.
//  * CUDA   : the per-iteration element update runs on the GPU in a
//             structure-of-arrays layout. Interior elements are updated while
//             the halo is in flight; boundary elements are updated afterwards.
//  * OpenMP : all host-side setup (mesh construction, renumbering, packing of
//             the gathered result) and the final hash use OpenMP threads.
//
// Floating point operation order inside the element update is identical to
// the original scalar code so results are bit-for-bit reproducible.

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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (global)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (per rank: the locally owned contiguous block of elements)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;     // local elements [start, end)
    std::vector<ElementDynamic> elements_dynamic;   // local elements (global copy on rank 0 at the end)
    std::vector<ElementDynamic> elements_dynamic_swap;
    idx_t n_global = 0;   // total number of elements
    idx_t start = 0;      // first global index owned by this rank
    idx_t end = 0;        // one past last global index owned by this rank
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int err__ = (call);                                                    \
        if (err__ != MPI_SUCCESS) {                                            \
            fprintf(stderr, "MPI error %s at %s:%d\n", #call, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Contiguous block partitioning of the global index space
static inline idx_t partitionStart(idx_t n_global, int rank, int size) {
    return (n_global * static_cast<idx_t>(rank)) / static_cast<idx_t>(size);
}

static inline int ownerOf(idx_t g, idx_t n_global, int size) {
    // Initial guess from the partition formula, then correct for rounding
    int r = static_cast<int>((static_cast<__uint128_t>(g) * size) / n_global);
    if (r >= size) r = size - 1;
    while (r > 0 && partitionStart(n_global, r, size) > g) --r;
    while (r + 1 < size && partitionStart(n_global, r + 1, size) <= g) ++r;
    return r;
}

// Build a 2D square grid as an unstructured mesh (only the local block)
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    world.n_global = n_elems;
    world.start = partitionStart(n_elems, rank, size);
    world.end = partitionStart(n_elems, rank + 1, size);
    const idx_t n_local = world.end - world.start;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    // (the swap buffer lives on the GPU; see DeviceSim)

    const int last = n_elems_root - 1;

    // Initialize elements and build connectivity: each element connects to its
    // neighbors in the 2D grid (up, down, left, right)
#pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = world.start + li;
        const int x = static_cast<int>(idx / n_elems_root);
        const int y = static_cast<int>(idx % n_elems_root);

        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        world.elements_dynamic[li].current_energy = 0.0;
        world.elements_dynamic[li].total_flux = 0.0;

        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
        if (x == 0 && y == last) elem.material_idx = OUTFLOW_MAT_ID;
        if (x == last && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (x == last && y == last) elem.material_idx = INFLOW_MAT_ID;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute energy flux between two elements (same operation order as the
// original computeFlux: (((other - this) * coeff) * connection_flux) * 0.25)
__device__ __forceinline__ val_t computeFluxDev(val_t transfer_coeff, val_t this_energy,
                                                val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update elements [begin, end) of the (renumbered) local element set.
// Connectivity is stored transposed (connection-major) for coalesced access.
__global__ void __launch_bounds__(256)
updateElementsKernel(const int begin, const int end, const int stride,
                     const int* __restrict__ num_conn,
                     const int* __restrict__ conn_idx,
                     const val_t* __restrict__ conn_flux,
                     const int* __restrict__ mat_idx,
                     const Material* __restrict__ materials,
                     const val_t* __restrict__ energy_in,
                     const val_t* __restrict__ tflux_in,
                     val_t* __restrict__ energy_out,
                     val_t* __restrict__ tflux_out) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const Material mat = materials[mat_idx[i]];
    const val_t e_this = energy_in[i];
    const int nc = num_conn[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
#pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const int nb = conn_idx[j * stride + i];
            const val_t cf = conn_flux[j * stride + i];
            const val_t e_nb = __ldg(&energy_in[nb]);
            total_flux += computeFluxDev(mat.transfer_coeff, e_this, cf, e_nb);
        }
    }

    // Update element state
    energy_out[i] = e_this + total_flux;
    tflux_out[i] = tflux_in[i] + fabs(total_flux);
}

// Gather energy values of the elements requested by neighbouring ranks
__global__ void packHaloKernel(const int n, const int* __restrict__ send_idx,
                               const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = energy[send_idx[i]];
}

// ---------------------------------------------------------------------------
// Distributed GPU simulation state
// ---------------------------------------------------------------------------
struct DeviceSim {
    int n_local = 0;      // locally owned elements
    int n_ghost = 0;      // ghost (off-rank) elements appended after the local ones
    int n_boundary = 0;   // local elements [0, n_boundary) have at least one ghost neighbour
    int max_conn = 0;     // global maximum number of connections per element (array slots)
    int final_buf = 0;    // which of the double buffers holds the final state

    // Device arrays
    int* d_num_conn = nullptr;
    int* d_conn_idx = nullptr;      // [max_conn][n_local]
    val_t* d_conn_flux = nullptr;   // [max_conn][n_local]
    int* d_mat_idx = nullptr;
    Material* d_materials = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};   // n_local + n_ghost
    val_t* d_tflux[2] = {nullptr, nullptr};    // n_local
    int* d_send_idx = nullptr;
    val_t* d_sendbuf = nullptr;

    // Host staging (pinned)
    val_t* h_sendbuf = nullptr;
    val_t* h_recvbuf = nullptr;

    // Communication pattern
    std::vector<int> send_counts, send_displs, recv_counts, recv_displs;
    std::vector<int> comm_ranks;   // ranks with non-zero traffic in either direction
    int n_send = 0;

    // Renumbering: perm[new_local] = original local index
    std::vector<int> perm;

    cudaStream_t s_comp = nullptr, s_comm = nullptr;
    cudaEvent_t ev_comp = nullptr, ev_comm = nullptr;
};

// Build the local SoA representation, the ghost layer and the MPI exchange
// pattern from the (global-index) connectivity in world.elements_static.
void setupDeviceSim(const World& world, DeviceSim& sim, int rank, int size) {
    const idx_t n_local_u = world.end - world.start;
    const int n_local = static_cast<int>(n_local_u);
    sim.n_local = n_local;
    const int nthreads = omp_get_max_threads();

    // ---- 1. Discover ghost elements (off-rank neighbours), sorted & unique
    std::vector<std::vector<idx_t>> ghost_tl(nthreads);
#pragma omp parallel
    {
        std::vector<idx_t>& mine = ghost_tl[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (int li = 0; li < n_local; ++li) {
            const ElementStatic& e = world.elements_static[li];
            for (idx_t j = 0; j < e.num_connections; ++j) {
                const idx_t g = e.connected_idx[j];
                if (g < world.start || g >= world.end) mine.push_back(g);
            }
        }
        std::sort(mine.begin(), mine.end());
        mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
    }
    std::vector<idx_t> ghosts;
    for (auto& v : ghost_tl) ghosts.insert(ghosts.end(), v.begin(), v.end());
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    const int n_ghost = static_cast<int>(ghosts.size());
    sim.n_ghost = n_ghost;

    // ---- 2. Exchange pattern. Ghosts are sorted by global index and ranks own
    //         contiguous blocks, so ghosts from one owner are contiguous.
    sim.recv_counts.assign(size, 0);
    sim.recv_displs.assign(size + 1, 0);
    sim.send_counts.assign(size, 0);
    sim.send_displs.assign(size + 1, 0);
    for (int k = 0; k < n_ghost; ++k) sim.recv_counts[ownerOf(ghosts[k], world.n_global, size)]++;
    for (int r = 0; r < size; ++r) sim.recv_displs[r + 1] = sim.recv_displs[r] + sim.recv_counts[r];

    MPI_CHECK(MPI_Alltoall(sim.recv_counts.data(), 1, MPI_INT, sim.send_counts.data(), 1, MPI_INT,
                           MPI_COMM_WORLD));
    for (int r = 0; r < size; ++r) sim.send_displs[r + 1] = sim.send_displs[r] + sim.send_counts[r];
    sim.n_send = sim.send_displs[size];

    // Tell each owner which of its elements we need (global indices)
    std::vector<idx_t> send_global(sim.n_send);
    MPI_CHECK(MPI_Alltoallv(ghosts.data(), sim.recv_counts.data(), sim.recv_displs.data(),
                            MPI_UINT64_T, send_global.data(), sim.send_counts.data(),
                            sim.send_displs.data(), MPI_UINT64_T, MPI_COMM_WORLD));
    for (int r = 0; r < size; ++r)
        if (sim.send_counts[r] > 0 || sim.recv_counts[r] > 0) sim.comm_ranks.push_back(r);

    // ---- 3. Classify local elements: boundary (touches a ghost) first, then interior
    std::vector<unsigned char> is_boundary(n_local, 0);
#pragma omp parallel for schedule(static)
    for (int li = 0; li < n_local; ++li) {
        const ElementStatic& e = world.elements_static[li];
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const idx_t g = e.connected_idx[j];
            if (g < world.start || g >= world.end) { is_boundary[li] = 1; break; }
        }
    }

    sim.perm.resize(n_local);
    std::vector<int> inv_perm(n_local);
    {
        // Stable partition: boundary elements keep relative order, then interior
        std::vector<int> cnt_b(nthreads + 1, 0), cnt_i(nthreads + 1, 0);
#pragma omp parallel
        {
            const int t = omp_get_thread_num();
            int cb = 0, ci = 0;
#pragma omp for schedule(static)
            for (int li = 0; li < n_local; ++li) { if (is_boundary[li]) ++cb; else ++ci; }
            cnt_b[t + 1] = cb;
            cnt_i[t + 1] = ci;
#pragma omp barrier
#pragma omp single
            {
                for (int k = 0; k < nthreads; ++k) {
                    cnt_b[k + 1] += cnt_b[k];
                    cnt_i[k + 1] += cnt_i[k];
                }
                sim.n_boundary = cnt_b[nthreads];
            }
            int pb = cnt_b[t], pi = sim.n_boundary + cnt_i[t];
#pragma omp for schedule(static)
            for (int li = 0; li < n_local; ++li) {
                const int ni = is_boundary[li] ? pb++ : pi++;
                sim.perm[ni] = li;
                inv_perm[li] = ni;
            }
        }
    }

    // ---- 4. Build SoA host arrays in the new numbering. Only as many connection
    //         slots as actually used anywhere in the mesh are stored.
    int max_conn_local = 0;
#pragma omp parallel for schedule(static) reduction(max : max_conn_local)
    for (int li = 0; li < n_local; ++li)
        max_conn_local = std::max(max_conn_local, static_cast<int>(world.elements_static[li].num_connections));
    MPI_CHECK(MPI_Allreduce(&max_conn_local, &sim.max_conn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    const int max_conn = sim.max_conn;

    const int n_ext = n_local + n_ghost;
    std::vector<int> h_num_conn(n_local), h_mat_idx(n_local);
    std::vector<int> h_conn_idx(static_cast<size_t>(max_conn) * n_local, 0);
    std::vector<val_t> h_conn_flux(static_cast<size_t>(max_conn) * n_local, 0.0);
    std::vector<val_t> h_energy(n_ext, 0.0), h_tflux(n_local);

#pragma omp parallel for schedule(static)
    for (int ni = 0; ni < n_local; ++ni) {
        const int li = sim.perm[ni];
        const ElementStatic& e = world.elements_static[li];
        h_num_conn[ni] = static_cast<int>(e.num_connections);
        h_mat_idx[ni] = static_cast<int>(e.material_idx);
        h_energy[ni] = world.elements_dynamic[li].current_energy;
        h_tflux[ni] = world.elements_dynamic[li].total_flux;
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const idx_t g = e.connected_idx[j];
            int target;
            if (g >= world.start && g < world.end) {
                target = inv_perm[static_cast<int>(g - world.start)];
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), g);
                target = n_local + static_cast<int>(it - ghosts.begin());
            }
            h_conn_idx[static_cast<size_t>(j) * n_local + ni] = target;
            h_conn_flux[static_cast<size_t>(j) * n_local + ni] = e.connected_flux[j];
        }
    }

    std::vector<int> h_send_idx(sim.n_send);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < sim.n_send; ++k)
        h_send_idx[k] = inv_perm[static_cast<int>(send_global[k] - world.start)];

    // ---- 5. Device allocation and upload
    auto dmalloc = [](void** p, size_t bytes) {
        if (bytes == 0) { *p = nullptr; return; }
        CUDA_CHECK(cudaMalloc(p, bytes));
    };
    dmalloc((void**)&sim.d_num_conn, sizeof(int) * n_local);
    dmalloc((void**)&sim.d_mat_idx, sizeof(int) * n_local);
    dmalloc((void**)&sim.d_conn_idx, sizeof(int) * h_conn_idx.size());
    dmalloc((void**)&sim.d_conn_flux, sizeof(val_t) * h_conn_flux.size());
    dmalloc((void**)&sim.d_materials, sizeof(Material) * world.materials.size());
    for (int b = 0; b < 2; ++b) {
        dmalloc((void**)&sim.d_energy[b], sizeof(val_t) * n_ext);
        dmalloc((void**)&sim.d_tflux[b], sizeof(val_t) * n_local);
    }
    dmalloc((void**)&sim.d_send_idx, sizeof(int) * sim.n_send);
    dmalloc((void**)&sim.d_sendbuf, sizeof(val_t) * sim.n_send);
    if (sim.n_send > 0) CUDA_CHECK(cudaMallocHost((void**)&sim.h_sendbuf, sizeof(val_t) * sim.n_send));
    if (n_ghost > 0) CUDA_CHECK(cudaMallocHost((void**)&sim.h_recvbuf, sizeof(val_t) * n_ghost));

    auto upload = [](void* d, const void* h, size_t bytes) {
        if (bytes > 0) CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    };
    upload(sim.d_num_conn, h_num_conn.data(), sizeof(int) * n_local);
    upload(sim.d_mat_idx, h_mat_idx.data(), sizeof(int) * n_local);
    upload(sim.d_conn_idx, h_conn_idx.data(), sizeof(int) * h_conn_idx.size());
    upload(sim.d_conn_flux, h_conn_flux.data(), sizeof(val_t) * h_conn_flux.size());
    upload(sim.d_materials, world.materials.data(), sizeof(Material) * world.materials.size());
    for (int b = 0; b < 2; ++b) {
        upload(sim.d_energy[b], h_energy.data(), sizeof(val_t) * n_ext);
        upload(sim.d_tflux[b], h_tflux.data(), sizeof(val_t) * n_local);
    }
    upload(sim.d_send_idx, h_send_idx.data(), sizeof(int) * sim.n_send);

    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_comp, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_comm, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());
    (void)rank;
}

// Run simulation for n_iters iterations (distributed, GPU-resident)
void runSimulation(DeviceSim& sim, const int n_iters) {
    constexpr int BLOCK = 256;
    const int n_local = sim.n_local;
    const int n_boundary = sim.n_boundary;
    const int n_interior = n_local - n_boundary;
    const int n_comm = static_cast<int>(sim.comm_ranks.size());
    std::vector<MPI_Request> reqs(2 * static_cast<size_t>(n_comm));

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* e_in = sim.d_energy[cur];
        val_t* e_out = sim.d_energy[cur ^ 1];
        val_t* f_in = sim.d_tflux[cur];
        val_t* f_out = sim.d_tflux[cur ^ 1];

        // (a) Pack halo values and start the device->host copy on the comm stream
        if (sim.n_send > 0) {
            packHaloKernel<<<(sim.n_send + BLOCK - 1) / BLOCK, BLOCK, 0, sim.s_comm>>>(
                sim.n_send, sim.d_send_idx, e_in, sim.d_sendbuf);
            CUDA_CHECK(cudaMemcpyAsync(sim.h_sendbuf, sim.d_sendbuf, sizeof(val_t) * sim.n_send,
                                       cudaMemcpyDeviceToHost, sim.s_comm));
        }

        // (b) Interior elements do not depend on ghost values: overlap with communication
        if (n_interior > 0) {
            updateElementsKernel<<<(n_interior + BLOCK - 1) / BLOCK, BLOCK, 0, sim.s_comp>>>(
                n_boundary, n_local, n_local, sim.d_num_conn, sim.d_conn_idx, sim.d_conn_flux,
                sim.d_mat_idx, sim.d_materials, e_in, f_in, e_out, f_out);
        }

        // (c) Halo exchange through pinned host buffers
        if (n_comm > 0) {
            int nr = 0;
            for (int r : sim.comm_ranks) {
                if (sim.recv_counts[r] > 0) {
                    MPI_CHECK(MPI_Irecv(sim.h_recvbuf + sim.recv_displs[r], sim.recv_counts[r],
                                        MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &reqs[nr++]));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(sim.s_comm));
            for (int r : sim.comm_ranks) {
                if (sim.send_counts[r] > 0) {
                    MPI_CHECK(MPI_Isend(sim.h_sendbuf + sim.send_displs[r], sim.send_counts[r],
                                        MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &reqs[nr++]));
                }
            }
            MPI_CHECK(MPI_Waitall(nr, reqs.data(), MPI_STATUSES_IGNORE));
            if (sim.n_ghost > 0) {
                CUDA_CHECK(cudaMemcpyAsync(e_in + n_local, sim.h_recvbuf, sizeof(val_t) * sim.n_ghost,
                                           cudaMemcpyHostToDevice, sim.s_comm));
            }
        }

        // (d) Boundary elements once ghost values have arrived (comm stream is in-order)
        if (n_boundary > 0) {
            updateElementsKernel<<<(n_boundary + BLOCK - 1) / BLOCK, BLOCK, 0, sim.s_comm>>>(
                0, n_boundary, n_local, sim.d_num_conn, sim.d_conn_idx, sim.d_conn_flux,
                sim.d_mat_idx, sim.d_materials, e_in, f_in, e_out, f_out);
        }

        // (e) Join streams: the next iteration reads everything written in this one
        CUDA_CHECK(cudaEventRecord(sim.ev_comp, sim.s_comp));
        CUDA_CHECK(cudaEventRecord(sim.ev_comm, sim.s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(sim.s_comm, sim.ev_comp, 0));
        CUDA_CHECK(cudaStreamWaitEvent(sim.s_comp, sim.ev_comm, 0));

        // Swap buffers
        cur ^= 1;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    sim.final_buf = cur;
}

// Copy final state back into world.elements_dynamic (original local ordering)
void downloadResults(DeviceSim& sim, World& world) {
    const int cur = sim.final_buf;
    const int n_local = sim.n_local;
    std::vector<val_t> h_energy(n_local), h_tflux(n_local);
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(h_energy.data(), sim.d_energy[cur], sizeof(val_t) * n_local,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tflux.data(), sim.d_tflux[cur], sizeof(val_t) * n_local,
                              cudaMemcpyDeviceToHost));
    }
#pragma omp parallel for schedule(static)
    for (int ni = 0; ni < n_local; ++ni) {
        ElementDynamic& e = world.elements_dynamic[sim.perm[ni]];
        e.current_energy = h_energy[ni];
        e.total_flux = h_tflux[ni];
    }
}

void destroyDeviceSim(DeviceSim& sim) {
    cudaFree(sim.d_num_conn);
    cudaFree(sim.d_conn_idx);
    cudaFree(sim.d_conn_flux);
    cudaFree(sim.d_mat_idx);
    cudaFree(sim.d_materials);
    for (int b = 0; b < 2; ++b) { cudaFree(sim.d_energy[b]); cudaFree(sim.d_tflux[b]); }
    cudaFree(sim.d_send_idx);
    cudaFree(sim.d_sendbuf);
    if (sim.h_sendbuf) cudaFreeHost(sim.h_sendbuf);
    if (sim.h_recvbuf) cudaFreeHost(sim.h_recvbuf);
    if (sim.s_comp) cudaStreamDestroy(sim.s_comp);
    if (sim.s_comm) cudaStreamDestroy(sim.s_comm);
    if (sim.ev_comp) cudaEventDestroy(sim.ev_comp);
    if (sim.ev_comm) cudaEventDestroy(sim.ev_comm);
}

// Gather the distributed dynamic state onto rank 0 (global ordering)
void gatherResults(World& world, int rank, int size) {
    const int n_local = static_cast<int>(world.end - world.start);
    std::vector<val_t> l_energy(n_local), l_tflux(n_local);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        l_energy[i] = world.elements_dynamic[i].current_energy;
        l_tflux[i] = world.elements_dynamic[i].total_flux;
    }

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(partitionStart(world.n_global, r + 1, size) -
                                     partitionStart(world.n_global, r, size));
        displs[r] = static_cast<int>(partitionStart(world.n_global, r, size));
    }

    std::vector<val_t> g_energy, g_tflux;
    if (rank == 0) {
        g_energy.resize(world.n_global);
        g_tflux.resize(world.n_global);
    }
    MPI_CHECK(MPI_Gatherv(l_energy.data(), n_local, MPI_DOUBLE, g_energy.data(), counts.data(),
                          displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gatherv(l_tflux.data(), n_local, MPI_DOUBLE, g_tflux.data(), counts.data(),
                          displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        world.elements_dynamic.resize(world.n_global);
        const idx_t n = world.n_global;
#pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            world.elements_dynamic[i].current_energy = g_energy[i];
            world.elements_dynamic[i].total_flux = g_tflux[i];
        }
    } else {
        world.elements_dynamic.clear();
        world.elements_dynamic.shrink_to_fit();
    }
    world.elements_dynamic_swap.clear();
    world.elements_dynamic_swap.shrink_to_fit();
}

// Validate simulation results (rank 0, sequential to keep the exact summation order)
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

// Compute a simple hash of the results for verification
// (XOR combination is associative and commutative, so the OpenMP reduction is exact)
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    const size_t n = elements.size();
#pragma omp parallel for schedule(static) reduction(^ : hash)
    for (size_t i = 0; i < n; ++i) {
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[i].current_energy, sizeof(uint64_t));
        std::memcpy(&f_bits, &elements[i].total_flux, sizeof(uint64_t));
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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    // Bind each rank to a GPU (round-robin over the GPUs visible on its node)
    {
        MPI_Comm node_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                      &node_comm));
        int node_rank = 0;
        MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
        MPI_CHECK(MPI_Comm_free(&node_comm));
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev < 1) {
            fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(node_rank % n_dev));
        CUDA_CHECK(cudaFree(nullptr));  // create context now, outside the timed region
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
        printf("\n");
    }

    // Build the unstructured mesh (local block on every rank)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (global totals, as in the original)
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

    // Build ghost layer, exchange pattern and device data
    DeviceSim sim;
    setupDeviceSim(world, sim, rank, size);

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Collect results on rank 0
    downloadResults(sim, world);
    destroyDeviceSim(sim);
    gatherResults(world, rank, size);

    int ret = 0;
    if (rank == 0) {
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

        // Validation
        if (validate) {
            bool valid = validateResults(world);
            if (!valid) ret = 1;
        }
        fflush(stdout);
    }

    MPI_CHECK(MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return ret;
}
