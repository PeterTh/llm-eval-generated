// Unstructured mesh energy transfer benchmark.
//
// Hybrid parallelization:
//   * MPI    - the element index space is split into contiguous blocks, one per
//              rank. Each step, only the halo (ghost) values crossing rank
//              boundaries are exchanged, with non-blocking point-to-point
//              messages to the ranks that actually share a boundary.
//   * CUDA   - every rank drives one GPU (local rank mod device count). The
//              element update is a gather-based kernel over an unstructured
//              connectivity list. Interior elements (no remote dependency) are
//              updated concurrently with the halo exchange of boundary values.
//   * OpenMP - host-side mesh construction and the host-side gather/scatter
//              of results are parallelized across threads.
//
// The per-element arithmetic and the connection summation order are identical
// to the original serial code, so results are bitwise reproducible.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;   // global element index
using val_t = double;
using lidx_t = int32_t;   // rank-local index into [owned | ghost] storage

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;
constexpr int NUM_MATERIALS = 3;
constexpr int CUDA_BLOCK = 256;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

// Flux coefficient of every connection produced by the 2D square builder.
constexpr val_t CONNECTION_FLUX = 1.0;

// Logical size of the per-element static record (used only for the memory report).
constexpr size_t STATIC_RECORD_BYTES =
    2 * sizeof(idx_t) + MAX_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t));

static const Material kMaterials[NUM_MATERIALS] = {
    {0.8, 0.0},    // Default material
    {0.8, 0.5},    // Inflow material
    {0.8, -0.5},   // Outflow material
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Mesh construction
// ---------------------------------------------------------------------------

// Contiguous block decomposition of the global element index space.
struct BlockPartition {
    std::vector<int64_t> begins;  // size nranks + 1

    BlockPartition(int64_t n_elems, int nranks) : begins(nranks + 1) {
        for (int r = 0; r <= nranks; ++r) {
            begins[r] = (n_elems * r) / nranks;
        }
    }

    int owner(int64_t g) const {
        return static_cast<int>(std::upper_bound(begins.begin(), begins.end(), g) - begins.begin()) - 1;
    }
};

struct Neighbors {
    int count;
    int64_t idx[4];
};

// Connectivity of the 2D square grid, in the same order as the original builder
// (+x, -x, +y, -y), keeping only in-bounds neighbors.
inline Neighbors neighborsOf(int64_t g, int64_t n_root) {
    const int64_t x = g / n_root;
    const int64_t y = g % n_root;
    const int64_t dx[4] = {1, -1, 0, 0};
    const int64_t dy[4] = {0, 0, 1, -1};
    Neighbors out{0, {0, 0, 0, 0}};
    for (int n = 0; n < 4; ++n) {
        const int64_t nx = x + dx[n];
        const int64_t ny = y + dy[n];
        if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
            out.idx[out.count++] = nx * n_root + ny;
        }
    }
    return out;
}

// Material assignment: corners are overwritten in the same order as the original
// builder, so the last assignment wins when several corners coincide (n_root == 1).
inline uint8_t materialOf(int64_t g, int64_t n_root) {
    const int64_t last = n_root - 1;
    const int64_t x = g / n_root;
    const int64_t y = g % n_root;
    uint8_t m = DEFAULT_MAT_ID;
    if (x == 0 && y == 0) m = INFLOW_MAT_ID;
    if (x == 0 && y == last) m = OUTFLOW_MAT_ID;
    if (x == last && y == 0) m = OUTFLOW_MAT_ID;
    if (x == last && y == last) m = INFLOW_MAT_ID;
    return m;
}

struct HaloLink {
    int rank;
    int offset;  // element offset into send buffer (send links) or ghost slots (recv links)
    int count;
};

// Mesh data local to one MPI rank.
// Local owned elements are stored with interior elements first ([0, n_int)),
// then boundary elements ([n_int, n_own)). Ghost elements occupy
// [n_own, n_own + n_ghost), ordered by global index (hence grouped by owner).
struct LocalMesh {
    int64_t begin = 0;                 // first owned global index
    int32_t n_own = 0;
    int32_t n_int = 0;
    int32_t n_ghost = 0;
    int32_t max_conn = 0;
    std::vector<int64_t> owned_off;    // local owned index -> (global index - begin)
    std::vector<uint8_t> nconn;        // connections per owned element
    std::vector<uint8_t> mat;          // material id per owned element
    std::vector<int32_t> conn_idx;     // [slot * n_own + local] -> local index in [owned | ghost]
    std::vector<int32_t> send_local;   // owned local indices to send, grouped by link
    std::vector<HaloLink> send_links;
    std::vector<HaloLink> recv_links;
};

inline int exclusiveScan(const std::vector<int>& cnt, std::vector<int>& displ) {
    int acc = 0;
    for (size_t i = 0; i < cnt.size(); ++i) {
        displ[i] = acc;
        acc += cnt[i];
    }
    return acc;
}

LocalMesh buildLocalMesh(int64_t n_root, const BlockPartition& part, int rank, MPI_Comm comm) {
    const int nranks = static_cast<int>(part.begins.size()) - 1;
    LocalMesh m;
    m.begin = part.begins[rank];
    const int64_t end = part.begins[rank + 1];
    const int64_t n_own = end - m.begin;
    m.n_own = static_cast<int32_t>(n_own);

    // 1. Global connectivity and material of owned elements (independent per element).
    std::vector<Neighbors> nbr(n_own);
    std::vector<uint8_t> mat(n_own);
#pragma omp parallel for schedule(static)
    for (int64_t off = 0; off < n_own; ++off) {
        nbr[off] = neighborsOf(m.begin + off, n_root);
        mat[off] = materialOf(m.begin + off, n_root);
    }

    // 2. Ghost set: remote neighbors, sorted (therefore grouped by owner rank).
    std::vector<int64_t> ghosts;
    for (int64_t off = 0; off < n_own; ++off) {
        for (int j = 0; j < nbr[off].count; ++j) {
            const int64_t g = nbr[off].idx[j];
            if (g < m.begin || g >= end) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    m.n_ghost = static_cast<int32_t>(ghosts.size());

    // 3. Tell each owner which of its elements this rank needs.
    std::vector<int> req_out_cnt(nranks, 0), req_out_displ(nranks, 0);
    std::vector<int> req_in_cnt(nranks, 0), req_in_displ(nranks, 0);
    for (int64_t g : ghosts) req_out_cnt[part.owner(g)]++;
    MPI_Alltoall(req_out_cnt.data(), 1, MPI_INT, req_in_cnt.data(), 1, MPI_INT, comm);
    exclusiveScan(req_out_cnt, req_out_displ);
    const int n_requested = exclusiveScan(req_in_cnt, req_in_displ);
    std::vector<int64_t> requested(n_requested);
    MPI_Alltoallv(ghosts.data(), req_out_cnt.data(), req_out_displ.data(), MPI_INT64_T,
                  requested.data(), req_in_cnt.data(), req_in_displ.data(), MPI_INT64_T, comm);

    // 4. Interior/boundary classification. An element is a boundary element if it
    //    references a ghost or if another rank references it.
    std::vector<uint8_t> is_bnd(n_own, 0);
    for (int64_t off = 0; off < n_own; ++off) {
        for (int j = 0; j < nbr[off].count; ++j) {
            const int64_t g = nbr[off].idx[j];
            if (g < m.begin || g >= end) is_bnd[off] = 1;
        }
    }
    for (int64_t g : requested) is_bnd[g - m.begin] = 1;

    m.owned_off.reserve(n_own);
    for (int64_t off = 0; off < n_own; ++off) {
        if (!is_bnd[off]) m.owned_off.push_back(off);
    }
    m.n_int = static_cast<int32_t>(m.owned_off.size());
    for (int64_t off = 0; off < n_own; ++off) {
        if (is_bnd[off]) m.owned_off.push_back(off);
    }

    std::vector<int32_t> off_to_local(n_own);
    for (int64_t k = 0; k < n_own; ++k) {
        off_to_local[m.owned_off[k]] = static_cast<int32_t>(k);
    }

    // 5. Local connectivity: connection order matches the original builder.
    int max_conn = 0;
    for (int64_t off = 0; off < n_own; ++off) max_conn = std::max(max_conn, nbr[off].count);
    m.max_conn = max_conn;
    m.nconn.resize(n_own);
    m.mat.resize(n_own);
    m.conn_idx.assign(static_cast<size_t>(max_conn) * n_own, 0);
#pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n_own; ++k) {
        const int64_t off = m.owned_off[k];
        const Neighbors& nb = nbr[off];
        m.nconn[k] = static_cast<uint8_t>(nb.count);
        m.mat[k] = mat[off];
        for (int j = 0; j < nb.count; ++j) {
            const int64_t g = nb.idx[j];
            int32_t li;
            if (g >= m.begin && g < end) {
                li = off_to_local[g - m.begin];
            } else {
                li = static_cast<int32_t>(n_own + (std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin()));
            }
            m.conn_idx[static_cast<size_t>(j) * n_own + k] = li;
        }
    }

    // 6. Halo links. Send lists follow the requester's ascending ghost order, so
    //    values land in the correct ghost slots without any reordering.
    for (int r = 0; r < nranks; ++r) {
        if (req_in_cnt[r] > 0) {
            m.send_links.push_back({r, static_cast<int>(m.send_local.size()), req_in_cnt[r]});
            for (int i = 0; i < req_in_cnt[r]; ++i) {
                const int64_t g = requested[req_in_displ[r] + i];
                m.send_local.push_back(off_to_local[g - m.begin]);
            }
        }
        if (req_out_cnt[r] > 0) {
            m.recv_links.push_back({r, req_out_displ[r], req_out_cnt[r]});
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__constant__ Material c_materials[NUM_MATERIALS];

__device__ __forceinline__ double computeFlux(const Material& mat, double this_energy,
                                              double connection_flux, double other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update elements [first, first + count) of the local range: read the current
// energies (owned and ghost), write the next energies, accumulate |flux|.
__global__ void updateKernel(const double* __restrict__ e_cur, double* __restrict__ e_nxt,
                             double* __restrict__ f_acc, const int32_t* __restrict__ conn_idx,
                             const uint8_t* __restrict__ nconn, const uint8_t* __restrict__ mat,
                             int n_own, int first, int count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = first + t;

    const Material m = c_materials[mat[i]];
    const double self = e_cur[i];
    double total_flux = m.external_flow;
    const int nc = nconn[i];
    for (int j = 0; j < nc; ++j) {
        const size_t s = static_cast<size_t>(j) * n_own + i;
        total_flux += computeFlux(m, self, CONNECTION_FLUX, e_cur[conn_idx[s]]);
    }
    e_nxt[i] = self + total_flux;
    f_acc[i] = f_acc[i] + fabs(total_flux);
}

__global__ void packKernel(const double* __restrict__ e, const int32_t* __restrict__ idx,
                           double* __restrict__ buf, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) buf[t] = e[idx[t]];
}

// ---------------------------------------------------------------------------
// Per-rank device state
// ---------------------------------------------------------------------------

struct DeviceSim {
    double* e[2] = {nullptr, nullptr};  // energies [owned | ghost], double-buffered
    int cur = 0;                        // index of the current energy buffer
    double* f = nullptr;                // accumulated |flux| (owned)
    int32_t* conn_idx = nullptr;
    uint8_t* nconn = nullptr;
    uint8_t* mat = nullptr;
    int32_t* send_idx = nullptr;
    double* send_dev = nullptr;
    double* send_host = nullptr;        // pinned
    double* recv_host = nullptr;        // pinned, ghost-ordered
    cudaStream_t s_int{};               // interior updates (low priority)
    cudaStream_t s_bnd{};               // boundary updates and packing (high priority)
    cudaStream_t s_copy{};              // host/device transfers
    cudaEvent_t ev_pack{};
};

template <typename T>
T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

DeviceSim createDeviceSim(const LocalMesh& m) {
    DeviceSim s;
    const size_t n_own = m.n_own;
    const size_t n_all = n_own + m.n_ghost;
    const size_t n_send = m.send_local.size();

    s.e[0] = deviceAlloc<double>(n_all);
    s.e[1] = deviceAlloc<double>(n_all);
    s.f = deviceAlloc<double>(n_own);
    CUDA_CHECK(cudaMemset(s.e[0], 0, n_all * sizeof(double)));  // initial state is zero
    CUDA_CHECK(cudaMemset(s.e[1], 0, n_all * sizeof(double)));
    CUDA_CHECK(cudaMemset(s.f, 0, n_own * sizeof(double)));

    s.conn_idx = deviceAlloc<int32_t>(static_cast<size_t>(m.max_conn) * n_own);
    s.nconn = deviceAlloc<uint8_t>(n_own);
    s.mat = deviceAlloc<uint8_t>(n_own);
    CUDA_CHECK(cudaMemcpy(s.conn_idx, m.conn_idx.data(), m.conn_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.nconn, m.nconn.data(), n_own, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.mat, m.mat.data(), n_own, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, kMaterials, sizeof(kMaterials)));

    s.send_idx = deviceAlloc<int32_t>(n_send);
    s.send_dev = deviceAlloc<double>(n_send);
    if (n_send > 0) {
        CUDA_CHECK(cudaMemcpy(s.send_idx, m.send_local.data(), n_send * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMallocHost(&s.send_host, std::max<size_t>(n_send, 1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&s.recv_host, std::max<size_t>(m.n_ghost, 1) * sizeof(double)));

    int prio_least = 0, prio_greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&s.s_int, cudaStreamNonBlocking, prio_least));
    CUDA_CHECK(cudaStreamCreateWithPriority(&s.s_bnd, cudaStreamNonBlocking, prio_greatest));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.s_copy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_pack, cudaEventDisableTiming));
    return s;
}

void destroyDeviceSim(DeviceSim& s) {
    cudaFree(s.e[0]);
    cudaFree(s.e[1]);
    cudaFree(s.f);
    cudaFree(s.conn_idx);
    cudaFree(s.nconn);
    cudaFree(s.mat);
    cudaFree(s.send_idx);
    cudaFree(s.send_dev);
    cudaFreeHost(s.send_host);
    cudaFreeHost(s.recv_host);
    cudaStreamDestroy(s.s_int);
    cudaStreamDestroy(s.s_bnd);
    cudaStreamDestroy(s.s_copy);
    cudaEventDestroy(s.ev_pack);
}

// Advance the simulation n_iters steps. Per step:
//   1. interior update (s_int) runs concurrently with
//   2. boundary update (s_bnd) -> pack halo values -> D2H (s_copy),
//   3. MPI halo exchange of boundary values (host),
//   4. H2D of incoming ghost values into the next buffer.
void runSimulation(DeviceSim& s, const LocalMesh& m, MPI_Comm comm, int n_iters) {
    const int n_own = m.n_own;
    const int n_int = m.n_int;
    const int n_bnd = m.n_own - m.n_int;
    const int n_send = static_cast<int>(m.send_local.size());
    const int n_ghost = m.n_ghost;
    const int n_recv_links = static_cast<int>(m.recv_links.size());
    const int n_req = n_recv_links + static_cast<int>(m.send_links.size());

    auto grid = [](int count) { return (count + CUDA_BLOCK - 1) / CUDA_BLOCK; };
    std::vector<MPI_Request> reqs(std::max(n_req, 1));

    for (int iter = 0; iter < n_iters; ++iter) {
        double* e_cur = s.e[s.cur];
        double* e_nxt = s.e[1 - s.cur];

        // Post receives for incoming halo values first (requests [0, n_recv_links)).
        for (int r = 0; r < n_recv_links; ++r) {
            const HaloLink& link = m.recv_links[r];
            MPI_Irecv(s.recv_host + link.offset, link.count, MPI_DOUBLE, link.rank, 0, comm, &reqs[r]);
        }

        // Interior elements depend only on owned values: start them immediately.
        if (n_int > 0) {
            updateKernel<<<grid(n_int), CUDA_BLOCK, 0, s.s_int>>>(
                e_cur, e_nxt, s.f, s.conn_idx, s.nconn, s.mat, n_own, 0, n_int);
            CUDA_CHECK(cudaGetLastError());
        }

        // Boundary elements, then pack the values neighbors need.
        if (n_bnd > 0) {
            updateKernel<<<grid(n_bnd), CUDA_BLOCK, 0, s.s_bnd>>>(
                e_cur, e_nxt, s.f, s.conn_idx, s.nconn, s.mat, n_own, n_int, n_bnd);
            CUDA_CHECK(cudaGetLastError());
        }
        if (n_send > 0) {
            packKernel<<<grid(n_send), CUDA_BLOCK, 0, s.s_bnd>>>(e_nxt, s.send_idx, s.send_dev, n_send);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(s.ev_pack, s.s_bnd));
            CUDA_CHECK(cudaStreamWaitEvent(s.s_copy, s.ev_pack, 0));
            CUDA_CHECK(cudaMemcpyAsync(s.send_host, s.send_dev, n_send * sizeof(double),
                                       cudaMemcpyDeviceToHost, s.s_copy));
        }
        CUDA_CHECK(cudaStreamSynchronize(s.s_copy));

        // Halo exchange (overlaps with the interior kernel). Send requests follow the receive requests.
        for (int k = 0; k < static_cast<int>(m.send_links.size()); ++k) {
            const HaloLink& link = m.send_links[k];
            MPI_Isend(s.send_host + link.offset, link.count, MPI_DOUBLE, link.rank, 0, comm,
                      &reqs[n_recv_links + k]);
        }
        MPI_Waitall(n_req, reqs.data(), MPI_STATUSES_IGNORE);

        // Ghost values go directly into the ghost slots of the next buffer.
        if (n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(e_nxt + n_own, s.recv_host, n_ghost * sizeof(double),
                                       cudaMemcpyHostToDevice, s.s_copy));
        }
        CUDA_CHECK(cudaStreamSynchronize(s.s_copy));
        CUDA_CHECK(cudaStreamSynchronize(s.s_int));
        CUDA_CHECK(cudaStreamSynchronize(s.s_bnd));

        s.cur = 1 - s.cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Host-side validation and reporting (rank 0)
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

// Returns the process exit code. Every rank runs the same control flow; only
// rank 0 performs console output.
int runBenchmark(int argc, char** argv, int rank, int nranks, MPI_Comm comm) {
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (n_elems_root < 1) {
        if (root) printf("Grid size must be at least 1\n");
        return 1;
    }

    const int64_t n_elems = static_cast<int64_t>(n_elems_root) * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n_elems_root, n_elems_root, static_cast<long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Bind this rank to a GPU (ranks on a node are spread round-robin over its devices).
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

    // Build the local part of the mesh.
    const BlockPartition part(n_elems, nranks);
    LocalMesh mesh = buildLocalMesh(n_elems_root, part, rank, comm);

    if (root) {
        // Memory usage (logical footprint of the whole mesh, as in the reference layout)
        const size_t static_mem = static_cast<size_t>(n_elems) * STATIC_RECORD_BYTES;
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    DeviceSim sim = createDeviceSim(mesh);

    // Run simulation (timed across all ranks)
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, mesh, comm, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    const double local_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const long long duration_ms = static_cast<long long>(max_ms);

    // Gather the final owned state into global order on rank 0.
    const size_t n_own = mesh.n_own;
    std::vector<double> e_host(n_own), f_host(n_own);
    CUDA_CHECK(cudaMemcpy(e_host.data(), sim.e[sim.cur], n_own * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f_host.data(), sim.f, n_own * sizeof(double), cudaMemcpyDeviceToHost));
    destroyDeviceSim(sim);

    std::vector<ElementDynamic> packed(n_own);
#pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < static_cast<int64_t>(n_own); ++k) {
        packed[mesh.owned_off[k]] = {e_host[k], f_host[k]};
    }

    std::vector<ElementDynamic> global;
    std::vector<int> counts, displs;
    if (root) {
        global.resize(n_elems);
        counts.resize(nranks);
        displs.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            counts[r] = static_cast<int>(2 * (part.begins[r + 1] - part.begins[r]));
            displs[r] = static_cast<int>(2 * part.begins[r]);
        }
    }
    MPI_Gatherv(packed.data(), static_cast<int>(2 * n_own), MPI_DOUBLE,
                root ? reinterpret_cast<double*>(global.data()) : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (!root) return 0;

    printf("Computation time: %lld ms\n", duration_ms);

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

    // Compute hash for verification
    const uint64_t hash = computeHash(global);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(global.size());
        for (const auto& elem : global) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }

    // Validation
    if (validate) {
        bool valid = validateResults(global);
        if (!valid) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int ret = runBenchmark(argc, argv, rank, nranks, MPI_COMM_WORLD);

    MPI_Finalize();
    return ret;
}
