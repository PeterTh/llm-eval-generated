// Unstructured Mesh Energy Transfer Benchmark
//
// Hybrid parallelization:
//   * MPI    - the global element set is partitioned into contiguous index
//              ranges, one per rank.  Each rank builds only its own part of the
//              mesh, derives the set of remote ("halo") neighbors from the
//              connectivity and exchanges halo energies every iteration.
//   * CUDA   - all per-iteration work (flux computation, state update, halo
//              packing) runs on the GPU assigned to the rank.  Interior
//              elements are updated on one stream while the halo exchange
//              (pack -> D2H -> MPI -> H2D) is in flight; boundary elements are
//              updated once the halo has arrived.
//   * OpenMP - host-side setup (mesh construction, halo classification,
//              SoA conversion) and post-processing (hash) are multithreaded.
//
// This file is compiled as CUDA (see CMakeLists.txt).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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

// World state: the part of the mesh owned by this rank, i.e. global elements
// [first_global, first_global + n_local).
struct World {
    idx_t n_global = 0;
    idx_t first_global = 0;
    idx_t n_local = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
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
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
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

// Contiguous block partition of [0, n) over `size` ranks
static inline idx_t partitionStart(idx_t n, int size, int rank) {
    return (n * static_cast<idx_t>(rank)) / static_cast<idx_t>(size);
}

static inline int ownerOf(idx_t g, idx_t n, int size) {
    // Inverse of partitionStart: smallest rank r with partitionStart(r+1) > g
    int r = static_cast<int>((g * static_cast<idx_t>(size)) / n);
    if (r >= size) r = size - 1;
    while (r > 0 && partitionStart(n, size, r) > g) --r;
    while (r + 1 < size && partitionStart(n, size, r + 1) <= g) ++r;
    return r;
}

// Build a 2D square grid as an unstructured mesh (local portion only).
// This represents computation on arbitrarily-shaped geometries.
void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    world.n_global = n_elems;
    world.first_global = partitionStart(n_elems, size, rank);
    world.n_local = partitionStart(n_elems, size, rank + 1) - world.first_global;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements
    world.elements_static.resize(world.n_local);
    world.elements_dynamic.resize(world.n_local);

    const int last = n_elems_root - 1;
    const idx_t first = world.first_global;
    const idx_t n_local = world.n_local;

    // Initialize elements and build connectivity: each element connects to its
    // neighbors in the 2D grid (up, down, left, right)
#pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = first + li;
        const int x = static_cast<int>(idx / static_cast<idx_t>(n_elems_root));
        const int y = static_cast<int>(idx % static_cast<idx_t>(n_elems_root));

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
        if ((x == 0 && y == 0) || (x == last && y == last)) {
            elem.material_idx = INFLOW_MAT_ID;
        } else if ((x == 0 && y == last) || (x == last && y == 0)) {
            elem.material_idx = OUTFLOW_MAT_ID;
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Update a set of elements. If `list` is null the elements [begin, begin+count)
// are updated, otherwise the elements list[0..count).
// Connectivity is stored slot-major (SoA): conn_idx[j * n_local + i].
__global__ void updateKernel(const int begin, const int count, const int* __restrict__ list,
                             const int n_local, const int max_conn,
                             const val_t* __restrict__ energy_in,
                             const val_t* __restrict__ tflux_in,
                             val_t* __restrict__ energy_out,
                             val_t* __restrict__ tflux_out,
                             const val_t* __restrict__ coeff,
                             const val_t* __restrict__ ext_flow,
                             const int* __restrict__ n_conn,
                             const int* __restrict__ conn_idx,
                             const val_t* __restrict__ conn_flux) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = list ? list[t] : begin + t;

    const val_t e = energy_in[i];
    const val_t c = coeff[i];
    const int nc = n_conn[i];

    // Start with external flow
    val_t total_flux = ext_flow[i];

    // Add flux from all connected elements (same order as reference)
    for (int j = 0; j < nc; ++j) {
        const int nb = conn_idx[j * n_local + i];
        const val_t f = conn_flux[j * n_local + i];
        total_flux += (energy_in[nb] - e) * c * f * 0.25;
    }

    energy_out[i] = e + total_flux;
    tflux_out[i] = tflux_in[i] + fabs(total_flux);
}

// Gather energies of elements that remote ranks need into a send buffer
__global__ void packKernel(const int count, const int* __restrict__ send_list,
                           const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) out[t] = energy[send_list[t]];
}

// ---------------------------------------------------------------------------
// Distributed GPU simulation state
// ---------------------------------------------------------------------------
struct Simulation {
    int rank = 0, size = 1;
    int n_local = 0;      // owned elements (local ids [0, n_local))
    int n_halo = 0;       // remote elements (local ids [n_local, n_local+n_halo))
    int max_conn = 0;

    // Device arrays
    val_t* d_energy[2] = {nullptr, nullptr};  // size n_local + n_halo
    val_t* d_tflux[2] = {nullptr, nullptr};   // size n_local
    val_t* d_coeff = nullptr;
    val_t* d_ext = nullptr;
    int* d_nconn = nullptr;
    int* d_cidx = nullptr;
    val_t* d_cflux = nullptr;
    int* d_boundary = nullptr;    // local ids with at least one remote neighbor
    int* d_interior = nullptr;    // null if interior is a contiguous range
    int* d_send_list = nullptr;
    val_t* d_sendbuf = nullptr;

    int n_boundary = 0;
    int n_interior = 0;
    int interior_begin = 0;

    // Communication pattern
    std::vector<int> send_peers, send_counts, send_offsets;  // per peer
    std::vector<int> recv_peers, recv_counts, recv_offsets;  // per peer, offsets into halo
    int n_send = 0;
    val_t* h_sendbuf = nullptr;  // pinned
    val_t* h_recvbuf = nullptr;  // pinned, size n_halo
    std::vector<MPI_Request> requests;

    cudaStream_t s_comp = nullptr, s_comm = nullptr;
    cudaEvent_t ev_interior = nullptr, ev_boundary = nullptr;

    int cur = 0;  // index of the current energy/tflux buffer
};

static void setupSimulation(Simulation& sim, const World& world) {
    const int size = sim.size;
    const idx_t first = world.first_global;
    const idx_t n_local_u = world.n_local;
    const int n_local = static_cast<int>(n_local_u);
    sim.n_local = n_local;

    // -------- Determine the halo: remote global indices referenced locally ----
    // Per-thread collection, then merge
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<idx_t>> remote_per_thread(n_threads);
    std::vector<unsigned char> is_boundary(n_local, 0);
#pragma omp parallel
    {
        std::vector<idx_t>& rem = remote_per_thread[omp_get_thread_num()];
#pragma omp for schedule(static)
        for (int i = 0; i < n_local; ++i) {
            const ElementStatic& es = world.elements_static[i];
            for (idx_t j = 0; j < es.num_connections; ++j) {
                const idx_t g = es.connected_idx[j];
                if (g < first || g >= first + n_local_u) {
                    rem.push_back(g);
                    is_boundary[i] = 1;
                }
            }
        }
    }
    std::vector<idx_t> remote;
    for (auto& v : remote_per_thread) remote.insert(remote.end(), v.begin(), v.end());
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());
    // Sorted global indices -> grouped by owner rank (owners are monotone in g)
    sim.n_halo = static_cast<int>(remote.size());

    // Group halo by owner: recv_peers / counts / offsets
    {
        int pos = 0;
        while (pos < sim.n_halo) {
            const int owner = ownerOf(remote[pos], world.n_global, size);
            int end = pos;
            while (end < sim.n_halo && ownerOf(remote[end], world.n_global, size) == owner) ++end;
            sim.recv_peers.push_back(owner);
            sim.recv_counts.push_back(end - pos);
            sim.recv_offsets.push_back(pos);
            pos = end;
        }
    }

    // Tell owners which of their elements we need (all-to-all handshake)
    std::vector<int> want_counts(size, 0);
    for (size_t p = 0; p < sim.recv_peers.size(); ++p) want_counts[sim.recv_peers[p]] = sim.recv_counts[p];
    std::vector<int> give_counts(size, 0);
    MPI_CHECK(MPI_Alltoall(want_counts.data(), 1, MPI_INT, give_counts.data(), 1, MPI_INT, MPI_COMM_WORLD));

    std::vector<idx_t> send_globals;
    {
        std::vector<MPI_Request> reqs;
        for (int r = 0; r < size; ++r) {
            if (give_counts[r] > 0) {
                sim.send_peers.push_back(r);
                sim.send_counts.push_back(give_counts[r]);
                sim.send_offsets.push_back(sim.n_send);
                sim.n_send += give_counts[r];
            }
        }
        send_globals.resize(sim.n_send);
        for (size_t p = 0; p < sim.send_peers.size(); ++p) {
            reqs.emplace_back();
            MPI_CHECK(MPI_Irecv(send_globals.data() + sim.send_offsets[p], sim.send_counts[p],
                                MPI_UINT64_T, sim.send_peers[p], 100, MPI_COMM_WORLD, &reqs.back()));
        }
        for (size_t p = 0; p < sim.recv_peers.size(); ++p) {
            reqs.emplace_back();
            MPI_CHECK(MPI_Isend(remote.data() + sim.recv_offsets[p], sim.recv_counts[p],
                                MPI_UINT64_T, sim.recv_peers[p], 100, MPI_COMM_WORLD, &reqs.back()));
        }
        if (!reqs.empty()) MPI_CHECK(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));
    }
    std::vector<int> send_list(sim.n_send);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < sim.n_send; ++k) send_list[k] = static_cast<int>(send_globals[k] - first);

    // -------- Convert connectivity to local ids, SoA layout -----------------
    int max_conn = 0;
#pragma omp parallel for reduction(max : max_conn) schedule(static)
    for (int i = 0; i < n_local; ++i)
        max_conn = std::max(max_conn, static_cast<int>(world.elements_static[i].num_connections));
    sim.max_conn = max_conn;

    std::vector<val_t> coeff(n_local), ext(n_local), energy(n_local), tflux(n_local);
    std::vector<int> nconn(n_local);
    std::vector<int> cidx(static_cast<size_t>(max_conn) * n_local, 0);
    std::vector<val_t> cflux(static_cast<size_t>(max_conn) * n_local, 0.0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        const Material& mat = world.materials[es.material_idx];
        coeff[i] = mat.transfer_coeff;
        ext[i] = mat.external_flow;
        energy[i] = world.elements_dynamic[i].current_energy;
        tflux[i] = world.elements_dynamic[i].total_flux;
        nconn[i] = static_cast<int>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            int li;
            if (g >= first && g < first + n_local_u) {
                li = static_cast<int>(g - first);
            } else {
                li = n_local + static_cast<int>(std::lower_bound(remote.begin(), remote.end(), g) - remote.begin());
            }
            cidx[j * n_local + i] = li;
            cflux[j * n_local + i] = es.connected_flux[j];
        }
    }

    // -------- Boundary / interior split ------------------------------------
    std::vector<int> boundary;
    for (int i = 0; i < n_local; ++i) if (is_boundary[i]) boundary.push_back(i);
    sim.n_boundary = static_cast<int>(boundary.size());
    sim.n_interior = n_local - sim.n_boundary;

    // Interior as a contiguous range if possible (true for the block partition of
    // the grid), otherwise an explicit list.
    std::vector<int> interior;
    {
        int lo = 0;
        while (lo < n_local && is_boundary[lo]) ++lo;
        int hi = n_local;
        while (hi > lo && is_boundary[hi - 1]) --hi;
        if (hi - lo == sim.n_interior) {
            sim.interior_begin = lo;
        } else {
            interior.reserve(sim.n_interior);
            for (int i = 0; i < n_local; ++i) if (!is_boundary[i]) interior.push_back(i);
        }
    }

    // -------- Device allocation and upload ----------------------------------
    const size_t n_tot = static_cast<size_t>(n_local) + sim.n_halo;
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&sim.d_energy[b], std::max<size_t>(n_tot, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&sim.d_tflux[b], std::max<size_t>(n_local, 1) * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&sim.d_coeff, std::max<size_t>(n_local, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_ext, std::max<size_t>(n_local, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_nconn, std::max<size_t>(n_local, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_cidx, std::max<size_t>(cidx.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_cflux, std::max<size_t>(cflux.size(), 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&sim.d_boundary, std::max<size_t>(sim.n_boundary, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_send_list, std::max<size_t>(sim.n_send, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_sendbuf, std::max<size_t>(sim.n_send, 1) * sizeof(val_t)));
    if (!interior.empty()) {
        CUDA_CHECK(cudaMalloc(&sim.d_interior, interior.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(sim.d_interior, interior.data(), interior.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(sim.d_energy[0], energy.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_tflux[0], tflux.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_coeff, coeff.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_ext, ext.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_nconn, nconn.data(), n_local * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (!cidx.empty()) {
        CUDA_CHECK(cudaMemcpy(sim.d_cidx, cidx.data(), cidx.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sim.d_cflux, cflux.data(), cflux.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (sim.n_boundary > 0)
        CUDA_CHECK(cudaMemcpy(sim.d_boundary, boundary.data(), sim.n_boundary * sizeof(int), cudaMemcpyHostToDevice));
    if (sim.n_send > 0)
        CUDA_CHECK(cudaMemcpy(sim.d_send_list, send_list.data(), sim.n_send * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMallocHost(&sim.h_sendbuf, std::max<size_t>(sim.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recvbuf, std::max<size_t>(sim.n_halo, 1) * sizeof(val_t)));
    sim.requests.resize(sim.send_peers.size() + sim.recv_peers.size());

    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_interior, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_boundary, cudaEventDisableTiming));

    // Warm up: force module loading / kernel instantiation outside the timed region
    packKernel<<<1, 32, 0, sim.s_comm>>>(0, sim.d_send_list, sim.d_energy[0], sim.d_sendbuf);
    updateKernel<<<1, 32, 0, sim.s_comp>>>(0, 0, nullptr, sim.n_local, sim.max_conn,
                                          sim.d_energy[0], sim.d_tflux[0], sim.d_energy[1], sim.d_tflux[1],
                                          sim.d_coeff, sim.d_ext, sim.d_nconn, sim.d_cidx, sim.d_cflux);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void teardownSimulation(Simulation& sim) {
    for (int b = 0; b < 2; ++b) { cudaFree(sim.d_energy[b]); cudaFree(sim.d_tflux[b]); }
    cudaFree(sim.d_coeff); cudaFree(sim.d_ext); cudaFree(sim.d_nconn);
    cudaFree(sim.d_cidx); cudaFree(sim.d_cflux); cudaFree(sim.d_boundary);
    cudaFree(sim.d_interior); cudaFree(sim.d_send_list); cudaFree(sim.d_sendbuf);
    cudaFreeHost(sim.h_sendbuf); cudaFreeHost(sim.h_recvbuf);
    cudaStreamDestroy(sim.s_comp); cudaStreamDestroy(sim.s_comm);
    cudaEventDestroy(sim.ev_interior); cudaEventDestroy(sim.ev_boundary);
}

// Exchange halo energies for the current buffer: pack on GPU, D2H, MPI, H2D.
// Enqueued on s_comm; the MPI part blocks the host.
static void exchangeHalo(Simulation& sim) {
    val_t* d_cur = sim.d_energy[sim.cur];
    constexpr int BS = 256;
    if (sim.n_send > 0) {
        packKernel<<<(sim.n_send + BS - 1) / BS, BS, 0, sim.s_comm>>>(sim.n_send, sim.d_send_list, d_cur, sim.d_sendbuf);
        CUDA_CHECK(cudaMemcpyAsync(sim.h_sendbuf, sim.d_sendbuf, sim.n_send * sizeof(val_t),
                                   cudaMemcpyDeviceToHost, sim.s_comm));
    }
    int nreq = 0;
    for (size_t p = 0; p < sim.recv_peers.size(); ++p) {
        MPI_CHECK(MPI_Irecv(sim.h_recvbuf + sim.recv_offsets[p], sim.recv_counts[p], MPI_DOUBLE,
                            sim.recv_peers[p], 200, MPI_COMM_WORLD, &sim.requests[nreq++]));
    }
    if (sim.n_send > 0) CUDA_CHECK(cudaStreamSynchronize(sim.s_comm));
    for (size_t p = 0; p < sim.send_peers.size(); ++p) {
        MPI_CHECK(MPI_Isend(sim.h_sendbuf + sim.send_offsets[p], sim.send_counts[p], MPI_DOUBLE,
                            sim.send_peers[p], 200, MPI_COMM_WORLD, &sim.requests[nreq++]));
    }
    if (nreq > 0) MPI_CHECK(MPI_Waitall(nreq, sim.requests.data(), MPI_STATUSES_IGNORE));
    if (sim.n_halo > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_cur + sim.n_local, sim.h_recvbuf, sim.n_halo * sizeof(val_t),
                                   cudaMemcpyHostToDevice, sim.s_comm));
    }
}

// Run simulation for n_iters iterations
void runSimulation(Simulation& sim, const int n_iters) {
    constexpr int BS = 256;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int cur = sim.cur, nxt = 1 - cur;

        // Start the halo exchange of the current state on the comm stream
        exchangeHalo(sim);

        // Interior elements do not depend on the halo: overlap with communication
        if (sim.n_interior > 0) {
            updateKernel<<<(sim.n_interior + BS - 1) / BS, BS, 0, sim.s_comp>>>(
                sim.interior_begin, sim.n_interior, sim.d_interior, sim.n_local, sim.max_conn,
                sim.d_energy[cur], sim.d_tflux[cur], sim.d_energy[nxt], sim.d_tflux[nxt],
                sim.d_coeff, sim.d_ext, sim.d_nconn, sim.d_cidx, sim.d_cflux);
        }

        // Boundary elements once the halo has landed
        if (sim.n_boundary > 0) {
            updateKernel<<<(sim.n_boundary + BS - 1) / BS, BS, 0, sim.s_comm>>>(
                0, sim.n_boundary, sim.d_boundary, sim.n_local, sim.max_conn,
                sim.d_energy[cur], sim.d_tflux[cur], sim.d_energy[nxt], sim.d_tflux[nxt],
                sim.d_coeff, sim.d_ext, sim.d_nconn, sim.d_cidx, sim.d_cflux);
        }

        // Join the streams: the next iteration reads the buffer written by both
        CUDA_CHECK(cudaEventRecord(sim.ev_interior, sim.s_comp));
        CUDA_CHECK(cudaEventRecord(sim.ev_boundary, sim.s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(sim.s_comm, sim.ev_interior, 0));
        CUDA_CHECK(cudaStreamWaitEvent(sim.s_comp, sim.ev_boundary, 0));

        // Swap buffers
        sim.cur = nxt;
    }
    CUDA_CHECK(cudaStreamSynchronize(sim.s_comp));
    CUDA_CHECK(cudaStreamSynchronize(sim.s_comm));
    CUDA_CHECK(cudaGetLastError());
}

// Download the final local state from the device
static void downloadState(const Simulation& sim, std::vector<ElementDynamic>& elements) {
    std::vector<val_t> energy(sim.n_local), tflux(sim.n_local);
    if (sim.n_local > 0) {
        CUDA_CHECK(cudaMemcpy(energy.data(), sim.d_energy[sim.cur], sim.n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tflux.data(), sim.d_tflux[sim.cur], sim.n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    elements.resize(sim.n_local);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.n_local; ++i) {
        elements[i].current_energy = energy[i];
        elements[i].total_flux = tflux[i];
    }
}

// Validate simulation results (on the gathered global state, rank 0 only)
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
// The hash is an XOR combination, so partial hashes over disjoint index ranges
// can be combined with XOR (across threads and across ranks).
uint64_t computeHashLocal(const std::vector<ElementDynamic>& elements, idx_t first_global) {
    uint64_t hash = 0;
    const size_t n = elements.size();
#pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t li = 0; li < n; ++li) {
        const uint64_t i = first_global + li;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[li].current_energy, sizeof(uint64_t));
        std::memcpy(&f_bits, &elements[li].total_flux, sizeof(uint64_t));
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

// Gather the full dynamic state onto rank 0 (as a flat array of doubles)
static void gatherGlobal(const std::vector<ElementDynamic>& local, const World& world,
                         int rank, int size, std::vector<ElementDynamic>& global) {
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const idx_t s = partitionStart(world.n_global, size, r);
        const idx_t e = partitionStart(world.n_global, size, r + 1);
        counts[r] = static_cast<int>((e - s) * 2);
        displs[r] = static_cast<int>(s * 2);
    }
    if (rank == 0) global.resize(world.n_global);
    MPI_CHECK(MPI_Gatherv(local.data(), static_cast<int>(local.size() * 2), MPI_DOUBLE,
                          rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int parse_status = -1;  // -1: continue, otherwise exit code
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
            parse_status = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_status = 1;
            break;
        }
    }
    if (parse_status >= 0) {
        MPI_Finalize();
        return parse_status;
    }

    // Select the GPU: one per rank, round-robin over the devices visible on the node
    {
        MPI_Comm node_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm));
        int node_rank = 0;
        MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
        MPI_CHECK(MPI_Comm_free(&node_comm));
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(node_rank % n_dev));
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

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    // Calculate memory usage (global, as in the reference implementation)
    if (rank == 0) {
        const size_t static_mem = world.n_global * sizeof(ElementStatic);
        const size_t dynamic_mem = world.n_global * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    Simulation sim;
    sim.rank = rank;
    sim.size = size;
    setupSimulation(sim, world);

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Fetch results
    downloadState(sim, world.elements_dynamic);

    // Compute hash for verification (XOR-combined across ranks)
    const uint64_t local_hash = computeHashLocal(world.elements_dynamic, world.first_global);
    uint64_t hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD));

    std::vector<ElementDynamic> global_dynamic;
    if (printResults || validate) {
        gatherGlobal(world.elements_dynamic, world, rank, size, global_dynamic);
    }

    int exit_code = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

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
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dynamic.size());
            for (const auto& elem : global_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(global_dynamic);
            if (!valid) {
                exit_code = 1;
            }
        }
        fflush(stdout);
    }

    teardownSimulation(sim);
    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}
