// Hybrid MPI + OpenMP + CUDA implementation of the unstructured mesh energy
// transfer benchmark.
//
// Decomposition:
//   * MPI: elements are split into contiguous global index ranges, one per rank.
//     Each iteration only the energies of owned elements that are neighbours of
//     another rank are exchanged (point-to-point with adjacent ranks).
//   * CUDA: each rank drives one GPU (selected by its node-local rank). Element
//     state lives on the device and is double buffered. Interior elements (no
//     remote neighbours) are updated concurrently with the halo exchange;
//     boundary elements are updated once the ghost values have arrived.
//   * OpenMP: host-side mesh construction, result reordering and the hash.
//
// Numerical semantics are identical to the sequential reference: every element
// performs the same floating-point operations in the same order (external flow
// first, then neighbours in connection order), and all reported statistics are
// computed over the same globally ordered arrays.

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

// Types to represent unstructured mesh elements
using idx_t = uint64_t;   // global element index
using val_t = double;
using lidx_t = uint32_t;  // local (owned) or ghost-extended element index

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Logical per-element record of the reference data layout. Only used to report
// the same memory figures as the reference implementation.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
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

static const Material MATERIALS[NUM_MATERIALS] = {
    {0.8, 0.0},    // Default material
    {0.8, 0.5},    // Inflow material
    {0.8, -0.5},   // Outflow material
};

constexpr int kThreads = 256;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Mesh construction (host, OpenMP)
// ---------------------------------------------------------------------------

// Connectivity of global element g of an n x n grid. Reproduces the reference
// rules: neighbours are visited in +x, -x, +y, -y order (bounds checked), each
// connection has flux coefficient 1.0, and the corner materials are assigned in
// the reference order (the last assignment wins).
static void buildElement(int n, idx_t g, idx_t& mat, int& ncon, idx_t* nbr, val_t* flux) {
    const idx_t last = static_cast<idx_t>(n - 1);
    const int64_t x = static_cast<int64_t>(g / static_cast<idx_t>(n));
    const int64_t y = static_cast<int64_t>(g % static_cast<idx_t>(n));

    mat = DEFAULT_MAT_ID;
    if (g == 0) mat = INFLOW_MAT_ID;                                   // (0, 0)
    if (g == last) mat = OUTFLOW_MAT_ID;                               // (0, last)
    if (g == last * static_cast<idx_t>(n)) mat = OUTFLOW_MAT_ID;       // (last, 0)
    if (g == last * static_cast<idx_t>(n) + last) mat = INFLOW_MAT_ID; // (last, last)

    static const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    ncon = 0;
    for (int k = 0; k < 4; ++k) {
        const int64_t nx = x + offsets[k][0];
        const int64_t ny = y + offsets[k][1];
        if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
            nbr[ncon] = static_cast<idx_t>(nx * n + ny);
            flux[ncon] = 1.0;
            ++ncon;
        }
    }
}

// Halo exchange entry: a peer rank, number of values, and offset into the
// corresponding host staging buffer (in values).
struct Halo {
    int peer;
    int count;
    int offset;
};

struct LocalMesh {
    idx_t begin = 0, end = 0;          // owned global range [begin, end)
    lidx_t n_local = 0;                // owned elements
    lidx_t n_ghost = 0;                // remote neighbours
    lidx_t n_int = 0;                  // interior elements are local ids [0, n_int)
    lidx_t K = 1;                      // ELL slot stride (max connections)
    std::vector<idx_t> l2g;            // local -> global
    std::vector<uint8_t> mat;          // material id per local element
    std::vector<uint8_t> ncon;         // connection count per local element
    std::vector<lidx_t> nb;            // [j * n_local + l]: extended index of neighbour j
    std::vector<val_t> nbflux;         // [j * n_local + l]: flux coefficient of neighbour j
    std::vector<idx_t> ghosts;         // sorted global ids of remote neighbours
    std::vector<Halo> recv_peers;      // ghost ranges received from each peer
    std::vector<Halo> send_peers;      // owned ranges sent to each peer
    std::vector<lidx_t> send_src;      // local ids of values to send (peer-major)
};

static LocalMesh buildLocalMesh(int n, const std::vector<idx_t>& starts, int rank, int nprocs, MPI_Comm comm) {
    LocalMesh M;
    M.begin = starts[rank];
    M.end = starts[rank + 1];
    const idx_t n_owned = M.end - M.begin;
    const idx_t b = M.begin, e = M.end;
    auto owned = [b, e](idx_t g) { return g >= b && g < e; };

    // Pass A: classify owned elements (boundary = has a remote neighbour) and
    // determine the maximum connection count.
    std::vector<uint8_t> is_bnd(n_owned, 0);
    int maxcon = 0;
#pragma omp parallel for schedule(static) reduction(max : maxcon)
    for (long long k = 0; k < static_cast<long long>(n_owned); ++k) {
        idx_t mat;
        int nc;
        idx_t nbr[MAX_CONNECTIONS];
        val_t fx[MAX_CONNECTIONS];
        buildElement(n, b + static_cast<idx_t>(k), mat, nc, nbr, fx);
        bool bnd = false;
        for (int j = 0; j < nc; ++j) {
            if (!owned(nbr[j])) bnd = true;
        }
        is_bnd[k] = bnd ? 1 : 0;
        maxcon = std::max(maxcon, nc);
    }

    // Pass B: local ordering (interior first, then boundary; ascending global
    // index within each group) and the set of ghost elements.
    std::vector<lidx_t> loc_of(n_owned);
    M.l2g.reserve(n_owned);
    for (idx_t k = 0; k < n_owned; ++k) {
        if (!is_bnd[k]) {
            loc_of[k] = static_cast<lidx_t>(M.l2g.size());
            M.l2g.push_back(b + k);
        }
    }
    M.n_int = static_cast<lidx_t>(M.l2g.size());
    for (idx_t k = 0; k < n_owned; ++k) {
        if (is_bnd[k]) {
            loc_of[k] = static_cast<lidx_t>(M.l2g.size());
            M.l2g.push_back(b + k);
        }
    }
    for (idx_t k = 0; k < n_owned; ++k) {
        if (!is_bnd[k]) continue;
        idx_t mat;
        int nc;
        idx_t nbr[MAX_CONNECTIONS];
        val_t fx[MAX_CONNECTIONS];
        buildElement(n, b + k, mat, nc, nbr, fx);
        for (int j = 0; j < nc; ++j) {
            if (!owned(nbr[j])) M.ghosts.push_back(nbr[j]);
        }
    }
    std::sort(M.ghosts.begin(), M.ghosts.end());
    M.ghosts.erase(std::unique(M.ghosts.begin(), M.ghosts.end()), M.ghosts.end());

    M.n_local = static_cast<lidx_t>(n_owned);
    if (static_cast<idx_t>(M.n_local) + M.ghosts.size() > static_cast<idx_t>(UINT32_MAX)) {
        fprintf(stderr, "Mesh too large for per-rank 32-bit local indexing\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    M.n_ghost = static_cast<lidx_t>(M.ghosts.size());
    M.K = static_cast<lidx_t>(std::max(maxcon, 1));

    // Halo layout. Ghosts are sorted, so they are grouped by owning rank in
    // ascending rank order; ghost i occupies extended slot n_local + i.
    auto owner_of = [&starts](idx_t g) {
        return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
    };
    std::vector<int> recv_cnt(nprocs, 0), recv_disp(nprocs, 0);
    for (idx_t g : M.ghosts) recv_cnt[owner_of(g)]++;
    for (int r = 1; r < nprocs; ++r) recv_disp[r] = recv_disp[r - 1] + recv_cnt[r - 1];

    // Tell each owner which of its elements we need.
    std::vector<int> send_cnt(nprocs, 0), send_disp(nprocs, 0);
    MPI_Alltoall(recv_cnt.data(), 1, MPI_INT, send_cnt.data(), 1, MPI_INT, comm);
    for (int r = 1; r < nprocs; ++r) send_disp[r] = send_disp[r - 1] + send_cnt[r - 1];
    const int total_send = send_disp[nprocs - 1] + send_cnt[nprocs - 1];
    std::vector<idx_t> requests(static_cast<size_t>(total_send));
    MPI_Alltoallv(M.ghosts.data(), recv_cnt.data(), recv_disp.data(), MPI_UINT64_T,
                  requests.data(), send_cnt.data(), send_disp.data(), MPI_UINT64_T, comm);

    M.send_src.resize(static_cast<size_t>(total_send));
    for (int k = 0; k < total_send; ++k) {
        M.send_src[k] = loc_of[requests[k] - b];
    }
    for (int r = 0; r < nprocs; ++r) {
        if (recv_cnt[r] > 0) M.recv_peers.push_back({r, recv_cnt[r], recv_disp[r]});
        if (send_cnt[r] > 0) M.send_peers.push_back({r, send_cnt[r], send_disp[r]});
    }

    // Pass C: ELL-style connectivity, slot-major for coalesced GPU access.
    const size_t nl = M.n_local;
    const size_t K = M.K;
    M.mat.assign(nl, 0);
    M.ncon.assign(nl, 0);
    M.nb.assign(K * nl, 0);
    M.nbflux.assign(K * nl, 0.0);
#pragma omp parallel for schedule(static)
    for (long long l = 0; l < static_cast<long long>(nl); ++l) {
        idx_t mat;
        int nc;
        idx_t nbr[MAX_CONNECTIONS];
        val_t fx[MAX_CONNECTIONS];
        buildElement(n, M.l2g[l], mat, nc, nbr, fx);
        M.mat[l] = static_cast<uint8_t>(mat);
        M.ncon[l] = static_cast<uint8_t>(nc);
        for (int j = 0; j < nc; ++j) {
            lidx_t e;
            if (owned(nbr[j])) {
                e = loc_of[nbr[j] - b];
            } else {
                const auto it = std::lower_bound(M.ghosts.begin(), M.ghosts.end(), nbr[j]);
                e = static_cast<lidx_t>(M.n_local + (it - M.ghosts.begin()));
            }
            M.nb[static_cast<size_t>(j) * nl + l] = e;
            M.nbflux[static_cast<size_t>(j) * nl + l] = fx[j];
        }
    }

    return M;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Energy flux between an element and one neighbour. Operation order matches the
// reference implementation exactly.
__device__ __forceinline__ val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

struct StepArgs {
    const val_t* E;      // current energies (owned + ghost region)
    const val_t* F;      // current accumulated flux (owned)
    val_t* En;           // next energies (owned)
    val_t* Fn;           // next accumulated flux (owned)
    const lidx_t* nb;    // ELL neighbour indices
    const val_t* nbflux; // ELL flux coefficients
    const uint8_t* ncon;
    const uint8_t* mat_id;
    const Material* mats;
    lidx_t n_local;
};

// Updates local elements [begin, end). Reads only E, writes only En/Fn, so
// elements are independent and any partition / ordering gives identical results.
__global__ void stepKernel(StepArgs a, lidx_t begin, lidx_t end) {
    const lidx_t stride = gridDim.x * blockDim.x;
    for (lidx_t l = begin + blockIdx.x * blockDim.x + threadIdx.x; l < end; l += stride) {
        const Material mat = a.mats[a.mat_id[l]];
        const val_t self = a.E[l];

        // Start with external flow, then add flux from each connection in order
        val_t total_flux = mat.external_flow;
        const unsigned nc = a.ncon[l];
        for (unsigned j = 0; j < nc; ++j) {
            const size_t off = static_cast<size_t>(j) * a.n_local + l;
            total_flux += computeFlux(mat, self, a.nbflux[off], a.E[a.nb[off]]);
        }

        a.En[l] = self + total_flux;
        a.Fn[l] = a.F[l] + fabs(total_flux);
    }
}

// Gathers the values that neighbouring ranks need into a contiguous buffer.
__global__ void packKernel(const val_t* __restrict__ E, const lidx_t* __restrict__ src,
                           val_t* __restrict__ buf, lidx_t n) {
    const lidx_t stride = gridDim.x * blockDim.x;
    for (lidx_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        buf[k] = E[src[k]];
    }
}

static unsigned blocksFor(lidx_t count) {
    return static_cast<unsigned>((static_cast<uint64_t>(count) + kThreads - 1) / kThreads);
}

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------

struct Device {
    cudaStream_t s0 = nullptr, s1 = nullptr;
    val_t* E[2] = {nullptr, nullptr};
    val_t* F[2] = {nullptr, nullptr};
    lidx_t* nb = nullptr;
    val_t* nbflux = nullptr;
    uint8_t* ncon = nullptr;
    uint8_t* mat = nullptr;
    Material* mats = nullptr;
    lidx_t* send_src = nullptr;
    val_t* send_buf = nullptr;
    val_t* h_send = nullptr;  // pinned host staging
    val_t* h_recv = nullptr;  // pinned host staging
    val_t* cur_E = nullptr;   // state after the last run
    val_t* cur_F = nullptr;
};

template <typename T>
static T* devAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

template <typename T>
static T* pinnedAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

static void copyToDevice(void* dst, const void* src, size_t bytes) {
    if (bytes) CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

static void setupDevice(Device& D, const LocalMesh& M) {
    const size_t nl = M.n_local;
    const size_t ext = static_cast<size_t>(M.n_local) + M.n_ghost;

    CUDA_CHECK(cudaStreamCreateWithFlags(&D.s0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&D.s1, cudaStreamNonBlocking));

    for (int b = 0; b < 2; ++b) {
        D.E[b] = devAlloc<val_t>(ext);
        D.F[b] = devAlloc<val_t>(nl);
        CUDA_CHECK(cudaMemset(D.E[b], 0, std::max<size_t>(ext, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(D.F[b], 0, std::max<size_t>(nl, 1) * sizeof(val_t)));
    }

    D.nb = devAlloc<lidx_t>(M.nb.size());
    D.nbflux = devAlloc<val_t>(M.nbflux.size());
    D.ncon = devAlloc<uint8_t>(nl);
    D.mat = devAlloc<uint8_t>(nl);
    D.mats = devAlloc<Material>(NUM_MATERIALS);
    D.send_src = devAlloc<lidx_t>(M.send_src.size());
    D.send_buf = devAlloc<val_t>(M.send_src.size());
    D.h_send = pinnedAlloc<val_t>(M.send_src.size());
    D.h_recv = pinnedAlloc<val_t>(M.n_ghost);

    copyToDevice(D.nb, M.nb.data(), M.nb.size() * sizeof(lidx_t));
    copyToDevice(D.nbflux, M.nbflux.data(), M.nbflux.size() * sizeof(val_t));
    copyToDevice(D.ncon, M.ncon.data(), M.ncon.size());
    copyToDevice(D.mat, M.mat.data(), M.mat.size());
    copyToDevice(D.mats, MATERIALS, sizeof(MATERIALS));
    copyToDevice(D.send_src, M.send_src.data(), M.send_src.size() * sizeof(lidx_t));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void freeDevice(Device& D) {
    for (int b = 0; b < 2; ++b) {
        cudaFree(D.E[b]);
        cudaFree(D.F[b]);
    }
    cudaFree(D.nb);
    cudaFree(D.nbflux);
    cudaFree(D.ncon);
    cudaFree(D.mat);
    cudaFree(D.mats);
    cudaFree(D.send_src);
    cudaFree(D.send_buf);
    cudaFreeHost(D.h_send);
    cudaFreeHost(D.h_recv);
    cudaStreamDestroy(D.s0);
    cudaStreamDestroy(D.s1);
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

// Exchanges halo energies with adjacent ranks through the pinned staging buffers.
static void exchangeHalos(const LocalMesh& M, const Device& D, MPI_Comm comm, std::vector<MPI_Request>& reqs) {
    reqs.clear();
    for (const Halo& h : M.recv_peers) {
        MPI_Request r;
        MPI_Irecv(D.h_recv + h.offset, h.count, MPI_DOUBLE, h.peer, 0, comm, &r);
        reqs.push_back(r);
    }
    for (const Halo& h : M.send_peers) {
        MPI_Request r;
        MPI_Isend(D.h_send + h.offset, h.count, MPI_DOUBLE, h.peer, 0, comm, &r);
        reqs.push_back(r);
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
}

static void runSimulation(Device& D, const LocalMesh& M, int n_iters, MPI_Comm comm) {
    std::vector<MPI_Request> reqs;
    reqs.reserve(M.recv_peers.size() + M.send_peers.size());

    val_t* Ec = D.E[0];
    val_t* Fc = D.F[0];
    val_t* En = D.E[1];
    val_t* Fn = D.F[1];

    StepArgs args{};
    args.nb = D.nb;
    args.nbflux = D.nbflux;
    args.ncon = D.ncon;
    args.mat_id = D.mat;
    args.mats = D.mats;
    args.n_local = M.n_local;

    const lidx_t n_send = static_cast<lidx_t>(M.send_src.size());
    const size_t ghost_bytes = static_cast<size_t>(M.n_ghost) * sizeof(val_t);

    for (int iter = 0; iter < n_iters; ++iter) {
        args.E = Ec;
        args.F = Fc;
        args.En = En;
        args.Fn = Fn;

        // 1. Gather outgoing halo values on the device and stage them on the host.
        if (n_send > 0) {
            packKernel<<<blocksFor(n_send), kThreads, 0, D.s0>>>(Ec, D.send_src, D.send_buf, n_send);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(D.h_send, D.send_buf, n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, D.s0));
            CUDA_CHECK(cudaStreamSynchronize(D.s0));
        }

        // 2. Interior elements need no remote data: compute them while the
        //    halo exchange is in flight.
        if (M.n_int > 0) {
            stepKernel<<<blocksFor(M.n_int), kThreads, 0, D.s1>>>(args, 0, M.n_int);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Halo exchange with neighbouring ranks (host-side MPI).
        exchangeHalos(M, D, comm, reqs);

        // 4. Upload ghost values into the extended part of the current buffer.
        if (M.n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(Ec + M.n_local, D.h_recv, ghost_bytes,
                                       cudaMemcpyHostToDevice, D.s0));
        }

        // 5. Boundary elements (those with remote neighbours).
        if (M.n_local > M.n_int) {
            stepKernel<<<blocksFor(M.n_local - M.n_int), kThreads, 0, D.s0>>>(args, M.n_int, M.n_local);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(D.s0));
        CUDA_CHECK(cudaStreamSynchronize(D.s1));

        // Swap buffers
        std::swap(Ec, En);
        std::swap(Fc, Fn);
    }

    D.cur_E = Ec;
    D.cur_F = Fc;
}

// Copies the owned state back and reorders it into global index order
// (relative to the rank's begin index).
static void fetchOwned(const Device& D, const LocalMesh& M, std::vector<val_t>& E, std::vector<val_t>& F) {
    const size_t nl = M.n_local;
    std::vector<val_t> he(nl), hf(nl);
    if (nl) {
        CUDA_CHECK(cudaMemcpy(he.data(), D.cur_E, nl * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hf.data(), D.cur_F, nl * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    E.resize(nl);
    F.resize(nl);
#pragma omp parallel for schedule(static)
    for (long long l = 0; l < static_cast<long long>(nl); ++l) {
        const size_t k = static_cast<size_t>(M.l2g[l] - M.begin);
        E[k] = he[l];
        F[k] = hf[l];
    }
}

// Hash contribution of the owned elements. The reference hash is an XOR over
// all elements, so the per-rank partial XORs combine exactly.
static uint64_t partialHash(const std::vector<val_t>& E, const std::vector<val_t>& F, idx_t begin) {
    uint64_t h = 0;
#pragma omp parallel for schedule(static) reduction(^ : h)
    for (long long k = 0; k < static_cast<long long>(E.size()); ++k) {
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &E[k], sizeof(e_bits));
        std::memcpy(&f_bits, &F[k], sizeof(f_bits));
        const uint64_t i = begin + static_cast<uint64_t>(k);
        h ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        h ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

// Validate simulation results (rank 0, global order)
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
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
        if (root) printf("Grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    const int n = n_elems_root;
    const idx_t N = static_cast<idx_t>(n) * static_cast<idx_t>(n);

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n, n, static_cast<unsigned long long>(N));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Bind each rank to a GPU on its node
    MPI_Comm shm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm);
    int local_rank = 0;
    MPI_Comm_rank(shm, &local_rank);
    MPI_Comm_free(&shm);

    int n_devices = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&n_devices);
    if (dev_err != cudaSuccess || n_devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
    CUDA_CHECK(cudaFree(0));  // create the context outside the timed region

    // Contiguous block partition of the element index space
    std::vector<idx_t> starts(nprocs + 1);
    {
        const idx_t q = N / static_cast<idx_t>(nprocs);
        const idx_t rem = N % static_cast<idx_t>(nprocs);
        for (int r = 0; r <= nprocs; ++r) {
            const idx_t rr = static_cast<idx_t>(r);
            starts[r] = q * rr + (rem * rr) / static_cast<idx_t>(nprocs);
        }
    }

    // Build the unstructured mesh
    if (root) printf("Building unstructured mesh...\n");
    LocalMesh M = buildLocalMesh(n, starts, rank, nprocs, comm);

    // Calculate memory usage (reported with the reference layout, global totals)
    const size_t static_mem = static_cast<size_t>(N) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(N) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    Device D;
    setupDevice(D, M);

    // Run simulation
    if (root) printf("Running simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(D, M, n_iters, comm);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_ns, 1, MPI_LONG_LONG, MPI_MAX, comm);
    const long duration_ms = static_cast<long>(elapsed_ns / 1000000LL);

    if (root) printf("Computation time: %ld ms\n", duration_ms);

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec =
        (static_cast<double>(n_measured_iters) * static_cast<double>(N)) / (duration_ms / 1000.0) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    // Final state in global order and distributed verification hash
    std::vector<val_t> E_owned, F_owned;
    fetchOwned(D, M, E_owned, F_owned);
    uint64_t hash = partialHash(E_owned, F_owned, M.begin);
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, comm);

    if (root) {
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather full results on the root rank when they are needed for output
    int status = 0;
    if (printResults || validate) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const idx_t cnt = starts[r + 1] - starts[r];
            if (cnt > static_cast<idx_t>(INT_MAX)) {
                fprintf(stderr, "Too many elements per rank for result gather\n");
                MPI_Abort(comm, 1);
            }
            counts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(starts[r]);
        }
        std::vector<val_t> allE, allF;
        if (root) {
            allE.resize(N);
            if (validate) allF.resize(N);
        }
        MPI_Gatherv(E_owned.data(), static_cast<int>(E_owned.size()), MPI_DOUBLE,
                    allE.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (validate) {
            MPI_Gatherv(F_owned.data(), static_cast<int>(F_owned.size()), MPI_DOUBLE,
                        allF.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        }

        if (root) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData(allE.begin(), allE.end());
                print_results(energyData, "ElementEnergy");
            }

            // Validation
            if (validate) {
                std::vector<ElementDynamic> elements(N);
                for (idx_t i = 0; i < N; ++i) {
                    elements[i].current_energy = allE[i];
                    elements[i].total_flux = allF[i];
                }
                if (!validateResults(elements)) status = 1;
            }
        }
        if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }

    freeDevice(D);
    MPI_Finalize();
    return status;
}
