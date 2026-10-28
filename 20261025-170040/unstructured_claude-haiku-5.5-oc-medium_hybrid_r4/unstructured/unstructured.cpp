// Unstructured mesh energy transfer benchmark.
//
// Parallelization (all three are always used):
//   MPI    - the element index space is block-partitioned across ranks. Each rank owns a
//            contiguous range of elements and keeps ghost copies of the remote neighbours
//            it reads. Halo exchange uses nonblocking point-to-point messages that only
//            involve ranks that actually share connectivity.
//   CUDA   - each rank drives one GPU (node-local rank modulo device count). The element
//            update is a CUDA kernel. Interior elements (no ghost neighbours) are updated
//            concurrently with the halo exchange; boundary elements are updated once the
//            halo has arrived.
//   OpenMP - host-side setup (mesh construction, ghost discovery, index remapping) and the
//            final state unpacking run as OpenMP parallel loops.
//
// The per-element arithmetic keeps the original evaluation order, and the device code uses
// non-contracting intrinsics, so results are bit-identical to the serial reference.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "ElementDynamic must be two doubles");

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr int kBlockSize = 256;
constexpr int kHaloTag = 17;

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define CHECK_MPI_STATUS(call)                                                    \
    do {                                                                          \
        int err_ = (call);                                                        \
        if (err_ != MPI_SUCCESS) {                                                \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "Error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

// ---------------------------------------------------------------------------
// Mesh description
// ---------------------------------------------------------------------------

std::vector<Material> makeMaterials() {
    return {Material{0.8, 0.0}, Material{0.8, 0.5}, Material{0.8, -0.5}};
}

// Material of global element g of an n x n grid. Corner assignments are applied in the
// same order as the original serial builder, so later assignments win.
idx_t materialOf(idx_t g, idx_t n) {
    const idx_t last = n - 1;
    idx_t mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;                  // (0, 0)
    if (g == last) mat = OUTFLOW_MAT_ID;              // (0, last)
    if (g == last * n) mat = OUTFLOW_MAT_ID;          // (last, 0)
    if (g == last * n + last) mat = INFLOW_MAT_ID;    // (last, last)
    return mat;
}

// Static description of global element g: 4-neighbour connectivity on a 2D grid,
// neighbours listed in the order (+x, -x, +y, -y), each with unit flux coefficient.
ElementStatic buildElementStatic(idx_t g, idx_t n_root) {
    ElementStatic e{};
    e.material_idx = materialOf(g, n_root);
    e.num_connections = 0;

    const int64_t n = static_cast<int64_t>(n_root);
    const int64_t x = static_cast<int64_t>(g / n_root);
    const int64_t y = static_cast<int64_t>(g % n_root);
    constexpr int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (const auto& off : offsets) {
        const int64_t nx = x + off[0];
        const int64_t ny = y + off[1];
        if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
            e.connected_idx[e.num_connections] = static_cast<idx_t>(nx * n + ny);
            e.connected_flux[e.num_connections] = 1.0;
            e.num_connections++;
        }
    }
    return e;
}

// Contiguous block partition of [0, N) over P ranks. Rank r owns [start[r], start[r+1]).
struct Partition {
    std::vector<idx_t> start;  // size P+1

    int owner(idx_t g) const {
        return static_cast<int>(std::upper_bound(start.begin(), start.end(), g) - start.begin()) - 1;
    }
};

Partition makePartition(idx_t N, int P) {
    Partition part;
    part.start.resize(static_cast<size_t>(P) + 1);
    const idx_t base = N / static_cast<idx_t>(P);
    const idx_t rem = N % static_cast<idx_t>(P);
    for (int r = 0; r <= P; ++r) {
        part.start[r] = static_cast<idx_t>(r) * base + std::min<idx_t>(static_cast<idx_t>(r), rem);
    }
    return part;
}

// Per-rank view of the mesh. Local element ids: [0, n_owned) owned, [n_owned, n_owned + n_ghost) ghost.
struct LocalMesh {
    idx_t begin = 0;
    idx_t end = 0;
    uint32_t n_owned = 0;
    uint32_t n_ghost = 0;
    uint32_t n_slots = 0;               // max connections over owned elements
    std::vector<uint8_t> nconn;         // [i]
    std::vector<uint8_t> mat;           // [i]
    std::vector<uint32_t> conn;         // [slot * n_owned + i] local index of neighbour
    std::vector<double> cflux;          // [slot * n_owned + i] connection flux coefficient
    std::vector<uint32_t> interior;     // owned elements with no ghost neighbour
    std::vector<uint32_t> boundary;     // owned elements with at least one ghost neighbour
    std::vector<double> mat_tc;         // per material transfer coefficient
    std::vector<double> mat_ext;        // per material external flow

    // Communication pattern (indexed by rank)
    std::vector<int> nbrs;              // ranks with traffic in either direction
    std::vector<int> send_count, recv_count;
    std::vector<int> send_off, recv_off;
    std::vector<uint32_t> send_local;   // owned local ids to send, grouped by destination rank
};

// Builds this rank's portion of the mesh: owned static data, ghost discovery,
// local index remapping, interior/boundary split, and the halo communication pattern.
LocalMesh buildLocalMesh(idx_t n_root, const Partition& part, int rank, int nranks,
                         const std::vector<Material>& mats, MPI_Comm comm) {
    LocalMesh m;
    m.begin = part.start[rank];
    m.end = part.start[rank + 1];
    const long long n_own = static_cast<long long>(m.end - m.begin);
    if (n_own > static_cast<long long>(UINT32_MAX)) fatal("too many elements per rank");
    m.n_owned = static_cast<uint32_t>(n_own);

    // 1. Static description of owned elements (global connectivity).
    std::vector<ElementStatic> own(static_cast<size_t>(n_own));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_own; ++i) {
        own[i] = buildElementStatic(m.begin + static_cast<idx_t>(i), n_root);
    }

    // 2. Ghost discovery: remote neighbours of owned elements, sorted and unique.
    //    Sorted order groups ghosts by owning rank, so each owner's ghosts are contiguous.
    std::vector<idx_t> ghosts;
    #pragma omp parallel
    {
        std::vector<idx_t> found;
        #pragma omp for schedule(static) nowait
        for (long long i = 0; i < n_own; ++i) {
            for (idx_t j = 0; j < own[i].num_connections; ++j) {
                const idx_t g = own[i].connected_idx[j];
                if (g < m.begin || g >= m.end) found.push_back(g);
            }
        }
        #pragma omp critical
        ghosts.insert(ghosts.end(), found.begin(), found.end());
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());

    if (static_cast<unsigned long long>(n_own) + ghosts.size() > UINT32_MAX) {
        fatal("too many local elements (including ghosts) per rank");
    }
    m.n_ghost = static_cast<uint32_t>(ghosts.size());

    // 3. Local connectivity in structure-of-arrays layout, remapped to local ids.
    uint32_t n_slots = 0;
    #pragma omp parallel for schedule(static) reduction(max : n_slots)
    for (long long i = 0; i < n_own; ++i) {
        n_slots = std::max(n_slots, static_cast<uint32_t>(own[i].num_connections));
    }
    m.n_slots = n_slots;
    const size_t nown = m.n_owned;
    m.nconn.resize(nown);
    m.mat.resize(nown);
    m.conn.assign(static_cast<size_t>(n_slots) * nown, 0);
    m.cflux.assign(static_cast<size_t>(n_slots) * nown, 0.0);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_own; ++i) {
        const ElementStatic& e = own[i];
        m.nconn[i] = static_cast<uint8_t>(e.num_connections);
        m.mat[i] = static_cast<uint8_t>(e.material_idx);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const idx_t g = e.connected_idx[j];
            uint32_t loc;
            if (g >= m.begin && g < m.end) {
                loc = static_cast<uint32_t>(g - m.begin);
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), g);
                loc = m.n_owned + static_cast<uint32_t>(it - ghosts.begin());
            }
            const size_t slot = static_cast<size_t>(j) * nown + static_cast<size_t>(i);
            m.conn[slot] = loc;
            m.cflux[slot] = e.connected_flux[j];
        }
    }

    // 4. Interior / boundary split (boundary elements read ghost values).
    for (uint32_t i = 0; i < m.n_owned; ++i) {
        bool touches_ghost = false;
        for (uint32_t j = 0; j < m.nconn[i]; ++j) {
            if (m.conn[static_cast<size_t>(j) * nown + i] >= m.n_owned) touches_ghost = true;
        }
        (touches_ghost ? m.boundary : m.interior).push_back(i);
    }

    m.mat_tc.resize(mats.size());
    m.mat_ext.resize(mats.size());
    for (size_t k = 0; k < mats.size(); ++k) {
        m.mat_tc[k] = mats[k].transfer_coeff;
        m.mat_ext[k] = mats[k].external_flow;
    }

    // 5. Halo communication pattern. Ghosts from rank p are requested from p; p learns
    //    which of its elements to send by receiving the requested global indices.
    const size_t P = static_cast<size_t>(nranks);
    m.recv_count.assign(P, 0);
    m.recv_off.assign(P, 0);
    for (size_t k = 0; k < ghosts.size(); ++k) {
        const int p = part.owner(ghosts[k]);
        if (m.recv_count[p] == 0) m.recv_off[p] = static_cast<int>(k);
        m.recv_count[p]++;
    }

    m.send_count.assign(P, 0);
    CHECK_MPI_STATUS(MPI_Alltoall(m.recv_count.data(), 1, MPI_INT,
                                  m.send_count.data(), 1, MPI_INT, comm));

    std::vector<int> req_disp(P, 0);
    int total_send = 0;
    for (size_t p = 0; p < P; ++p) {
        req_disp[p] = total_send;
        total_send += m.send_count[p];
    }
    std::vector<idx_t> requested(static_cast<size_t>(total_send));
    CHECK_MPI_STATUS(MPI_Alltoallv(ghosts.data(), m.recv_count.data(), m.recv_off.data(), MPI_UINT64_T,
                                   requested.data(), m.send_count.data(), req_disp.data(), MPI_UINT64_T,
                                   comm));

    m.send_off = req_disp;
    m.send_local.resize(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) {
        const idx_t g = requested[k];
        if (g < m.begin || g >= m.end) fatal("halo request for an element not owned by this rank");
        m.send_local[k] = static_cast<uint32_t>(g - m.begin);
    }

    for (int p = 0; p < nranks; ++p) {
        if (p != rank && (m.send_count[p] > 0 || m.recv_count[p] > 0)) m.nbrs.push_back(p);
    }
    return m;
}

// ---------------------------------------------------------------------------
// GPU kernels and device state
// ---------------------------------------------------------------------------

// Device view of the static mesh data.
struct MeshView {
    uint32_t n_owned;
    const uint8_t* nconn;
    const uint8_t* mat;
    const uint32_t* conn;
    const double* cflux;
    const double* mat_tc;
    const double* mat_ext;
};

// Gather the energies that other ranks need into a contiguous send buffer.
__global__ void packKernel(const double* __restrict__ E, const uint32_t* __restrict__ idx,
                           double* __restrict__ out, uint32_t n) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = E[idx[k]];
}

// One Jacobi-style update step for the listed owned elements.
// Evaluation order matches the original computeFlux():
//   total = ext; total += ((other - this) * tc * cf) * 0.25 over connections in order.
// The __*_rn intrinsics prevent FMA contraction so results match the serial code.
__global__ void updateKernel(MeshView mv, const uint32_t* __restrict__ list, uint32_t count,
                             const double* __restrict__ E_in, const double* __restrict__ F_in,
                             double* __restrict__ E_out, double* __restrict__ F_out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const uint32_t i = list[t];

    const double e_self = E_in[i];
    const uint32_t m = mv.mat[i];
    const double tc = mv.mat_tc[m];
    double total = mv.mat_ext[m];

    const uint32_t nc = mv.nconn[i];
    for (uint32_t j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * mv.n_owned + i;
        const double diff = __dsub_rn(E_in[mv.conn[slot]], e_self);
        const double flux = __dmul_rn(__dmul_rn(__dmul_rn(diff, tc), mv.cflux[slot]), 0.25);
        total = __dadd_rn(total, flux);
    }

    E_out[i] = __dadd_rn(e_self, total);
    F_out[i] = __dadd_rn(F_in[i], fabs(total));
}

struct GpuState {
    cudaStream_t s_int = nullptr;
    cudaStream_t s_bnd = nullptr;
    cudaStream_t s_comm = nullptr;
    cudaEvent_t ev_ghost = nullptr;

    uint32_t n_owned = 0;
    uint32_t n_ghost = 0;
    uint32_t n_send = 0;
    uint32_t n_int = 0;
    uint32_t n_bnd = 0;

    double* E[2] = {nullptr, nullptr};   // energy, double-buffered (owned + ghost slots)
    double* F[2] = {nullptr, nullptr};   // total flux, double-buffered (owned only)
    uint8_t* nconn = nullptr;
    uint8_t* mat = nullptr;
    uint32_t* conn = nullptr;
    double* cflux = nullptr;
    double* mat_tc = nullptr;
    double* mat_ext = nullptr;
    uint32_t* interior = nullptr;
    uint32_t* boundary = nullptr;
    uint32_t* send_idx = nullptr;
    double* send_buf = nullptr;          // device staging for outgoing halo values
    double* h_send = nullptr;            // pinned host staging, outgoing
    double* h_recv = nullptr;            // pinned host staging, incoming ghosts
    int cur = 0;                         // index of the current state buffer

    MeshView view() const {
        return MeshView{n_owned, nconn, mat, conn, cflux, mat_tc, mat_ext};
    }
};

template <typename T>
T* deviceAlloc(size_t n) {
    T* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

template <typename T>
T* pinnedAlloc(size_t n) {
    T* p = nullptr;
    CHECK_CUDA(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

template <typename T>
void toDevice(T* d, const std::vector<T>& h) {
    if (!h.empty()) CHECK_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

GpuState setupGpu(const LocalMesh& m) {
    GpuState g;
    CHECK_CUDA(cudaStreamCreateWithFlags(&g.s_int, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g.s_bnd, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&g.s_comm, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&g.ev_ghost, cudaEventDisableTiming));

    g.n_owned = m.n_owned;
    g.n_ghost = m.n_ghost;
    g.n_send = static_cast<uint32_t>(m.send_local.size());
    g.n_int = static_cast<uint32_t>(m.interior.size());
    g.n_bnd = static_cast<uint32_t>(m.boundary.size());

    const size_t n_total = static_cast<size_t>(m.n_owned) + m.n_ghost;
    const size_t n_own = m.n_owned;
    const size_t n_slot_entries = static_cast<size_t>(m.n_slots) * n_own;
    for (int k = 0; k < 2; ++k) {
        g.E[k] = deviceAlloc<double>(n_total);
        g.F[k] = deviceAlloc<double>(n_own);
        CHECK_CUDA(cudaMemset(g.E[k], 0, std::max<size_t>(n_total, 1) * sizeof(double)));
        CHECK_CUDA(cudaMemset(g.F[k], 0, std::max<size_t>(n_own, 1) * sizeof(double)));
    }

    g.nconn = deviceAlloc<uint8_t>(n_own);
    g.mat = deviceAlloc<uint8_t>(n_own);
    g.conn = deviceAlloc<uint32_t>(n_slot_entries);
    g.cflux = deviceAlloc<double>(n_slot_entries);
    g.mat_tc = deviceAlloc<double>(m.mat_tc.size());
    g.mat_ext = deviceAlloc<double>(m.mat_ext.size());
    g.interior = deviceAlloc<uint32_t>(m.interior.size());
    g.boundary = deviceAlloc<uint32_t>(m.boundary.size());
    g.send_idx = deviceAlloc<uint32_t>(m.send_local.size());
    g.send_buf = deviceAlloc<double>(m.send_local.size());
    g.h_send = pinnedAlloc<double>(m.send_local.size());
    g.h_recv = pinnedAlloc<double>(m.n_ghost);

    toDevice(g.nconn, m.nconn);
    toDevice(g.mat, m.mat);
    toDevice(g.conn, m.conn);
    toDevice(g.cflux, m.cflux);
    toDevice(g.mat_tc, m.mat_tc);
    toDevice(g.mat_ext, m.mat_ext);
    toDevice(g.interior, m.interior);
    toDevice(g.boundary, m.boundary);
    toDevice(g.send_idx, m.send_local);
    return g;
}

void freeGpu(GpuState& g) {
    for (int k = 0; k < 2; ++k) {
        CHECK_CUDA(cudaFree(g.E[k]));
        CHECK_CUDA(cudaFree(g.F[k]));
    }
    CHECK_CUDA(cudaFree(g.nconn));
    CHECK_CUDA(cudaFree(g.mat));
    CHECK_CUDA(cudaFree(g.conn));
    CHECK_CUDA(cudaFree(g.cflux));
    CHECK_CUDA(cudaFree(g.mat_tc));
    CHECK_CUDA(cudaFree(g.mat_ext));
    CHECK_CUDA(cudaFree(g.interior));
    CHECK_CUDA(cudaFree(g.boundary));
    CHECK_CUDA(cudaFree(g.send_idx));
    CHECK_CUDA(cudaFree(g.send_buf));
    CHECK_CUDA(cudaFreeHost(g.h_send));
    CHECK_CUDA(cudaFreeHost(g.h_recv));
    CHECK_CUDA(cudaEventDestroy(g.ev_ghost));
    CHECK_CUDA(cudaStreamDestroy(g.s_int));
    CHECK_CUDA(cudaStreamDestroy(g.s_bnd));
    CHECK_CUDA(cudaStreamDestroy(g.s_comm));
}

inline unsigned blocksFor(uint32_t n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

// Run n_iters update steps. Per step:
//   1. post halo receives; pack outgoing energies on the GPU and copy them to the host
//   2. interior update runs on the GPU while the host waits for the packed halo
//   3. send/receive halo values with neighbouring ranks
//   4. upload ghost values; boundary update runs once they are on the device
//   5. join all streams and swap buffers
void runSimulation(const LocalMesh& m, GpuState& g, MPI_Comm comm, int n_iters) {
    const MeshView view = g.view();
    std::vector<MPI_Request> reqs(2 * m.nbrs.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = 1 - g.cur;
        double* Ec = g.E[g.cur];
        double* Fc = g.F[g.cur];
        double* En = g.E[nxt];
        double* Fn = g.F[nxt];

        int nreq = 0;
        for (int p : m.nbrs) {
            if (m.recv_count[p] > 0) {
                MPI_Irecv(g.h_recv + m.recv_off[p], m.recv_count[p], MPI_DOUBLE, p, kHaloTag, comm,
                          &reqs[nreq++]);
            }
        }

        if (g.n_send > 0) {
            packKernel<<<blocksFor(g.n_send), kBlockSize, 0, g.s_comm>>>(Ec, g.send_idx, g.send_buf, g.n_send);
            CHECK_CUDA(cudaMemcpyAsync(g.h_send, g.send_buf, g.n_send * sizeof(double),
                                       cudaMemcpyDeviceToHost, g.s_comm));
        }

        if (g.n_int > 0) {
            updateKernel<<<blocksFor(g.n_int), kBlockSize, 0, g.s_int>>>(
                view, g.interior, g.n_int, Ec, Fc, En, Fn);
        }

        CHECK_CUDA(cudaStreamSynchronize(g.s_comm));
        for (int p : m.nbrs) {
            if (m.send_count[p] > 0) {
                MPI_Isend(g.h_send + m.send_off[p], m.send_count[p], MPI_DOUBLE, p, kHaloTag, comm,
                          &reqs[nreq++]);
            }
        }
        MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);

        if (g.n_ghost > 0) {
            CHECK_CUDA(cudaMemcpyAsync(Ec + g.n_owned, g.h_recv, g.n_ghost * sizeof(double),
                                       cudaMemcpyHostToDevice, g.s_comm));
        }
        CHECK_CUDA(cudaEventRecord(g.ev_ghost, g.s_comm));
        CHECK_CUDA(cudaStreamWaitEvent(g.s_bnd, g.ev_ghost, 0));

        if (g.n_bnd > 0) {
            updateKernel<<<blocksFor(g.n_bnd), kBlockSize, 0, g.s_bnd>>>(
                view, g.boundary, g.n_bnd, Ec, Fc, En, Fn);
        }

        CHECK_CUDA(cudaStreamSynchronize(g.s_int));
        CHECK_CUDA(cudaStreamSynchronize(g.s_bnd));
        CHECK_CUDA(cudaStreamSynchronize(g.s_comm));
        g.cur = nxt;
    }
}

// Copy this rank's owned final state to the host (OpenMP-parallel unpacking).
std::vector<ElementDynamic> downloadOwned(const GpuState& g) {
    std::vector<double> E(g.n_owned), F(g.n_owned);
    if (g.n_owned > 0) {
        CHECK_CUDA(cudaMemcpy(E.data(), g.E[g.cur], g.n_owned * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(F.data(), g.F[g.cur], g.n_owned * sizeof(double), cudaMemcpyDeviceToHost));
    }
    std::vector<ElementDynamic> out(g.n_owned);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(g.n_owned); ++i) {
        out[i] = ElementDynamic{E[i], F[i]};
    }
    return out;
}

// ---------------------------------------------------------------------------
// Reporting (rank 0, on the gathered global state)
// ---------------------------------------------------------------------------

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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root < 1) {
        if (root) printf("Error: grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;
    const idx_t N = static_cast<idx_t>(n_elems);

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Bind each rank to a GPU and split host threads between co-located ranks.
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    int node_size = 1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count < 1) fatal("no CUDA devices available");
    CHECK_CUDA(cudaSetDevice(node_rank % dev_count));
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / node_size));
    }

    // Build this rank's portion of the mesh
    const auto mats = makeMaterials();
    const Partition part = makePartition(N, nranks);
    const LocalMesh mesh = buildLocalMesh(static_cast<idx_t>(n_elems_root), part, rank, nranks, mats,
                                          MPI_COMM_WORLD);

    // Calculate memory usage (of the full logical mesh, as reported by the original benchmark)
    if (root) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    GpuState gpu = setupGpu(mesh);

    // Run simulation (timed region: all ranks, bracketed by barriers)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(mesh, gpu, MPI_COMM_WORLD, n_iters);
    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Gather the final state of all elements to rank 0
    std::vector<ElementDynamic> own_state = downloadOwned(gpu);
    std::vector<ElementDynamic> world_dyn;
    if (root) world_dyn.resize(static_cast<size_t>(N));

    if (2 * N > static_cast<idx_t>(INT_MAX)) fatal("grid too large for MPI gather counts");
    std::vector<int> counts(nranks), displs(nranks);
    for (int p = 0; p < nranks; ++p) {
        counts[p] = static_cast<int>(2 * (part.start[p + 1] - part.start[p]));
        displs[p] = static_cast<int>(2 * part.start[p]);
    }
    MPI_Gatherv(own_state.data(), static_cast<int>(2 * mesh.n_owned), MPI_DOUBLE,
                root ? world_dyn.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    freeGpu(gpu);

    int rc = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);

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
        const uint64_t hash = computeHash(world_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world_dyn.size());
            for (const auto& elem : world_dyn) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(world_dyn)) rc = 1;
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
