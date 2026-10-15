#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA implementation.
//  - MPI: contiguous block partition of the elements across ranks; ghost
//    energies are exchanged every iteration (only the values actually needed).
//  - CUDA: one GPU per rank (round robin per node); SoA layout, the interior
//    update overlaps with the halo exchange, the boundary update follows.
//  - OpenMP: host-side mesh construction, ghost analysis, hashing.

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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


constexpr int BOUNDARY_FLAG = 0x80;  // set in the per-element meta byte if it reads ghost data

__constant__ val_t d_coeff[3];
__constant__ val_t d_ext[3];

// Distributed world: this rank owns the global element range [lo, hi).
// Connectivity indices are local: [0, n_local) owned, [n_local, n_local+n_ghost) ghosts.
struct DistMesh {
    int rank = 0, size = 1;
    idx_t n_total = 0, lo = 0, hi = 0;
    size_t n_local = 0, n_ghost = 0;
    int max_conn = 0;
    std::vector<uint8_t> mat;          // material index
    std::vector<uint8_t> meta;         // num_connections | BOUNDARY_FLAG
    std::vector<uint32_t> conn_idx;    // [slot * n_local + i]
    std::vector<val_t> conn_flux;      // [slot * n_local + i]
    std::vector<uint32_t> boundary;    // local elements needing ghosts
    std::vector<idx_t> ghost_gid;      // sorted global ids of ghosts
    // halo exchange schedule
    std::vector<int> recv_peer, recv_cnt, recv_off;  // ghosts received from each peer
    std::vector<int> send_peer, send_cnt, send_off;
    std::vector<uint32_t> send_idx;    // local indices to send, grouped by peer
};

static void partitionRange(idx_t n, int p, int r, idx_t& lo, idx_t& hi) {
    const idx_t base = n / p, rem = n % p;
    lo = r * base + std::min<idx_t>(r, rem);
    hi = lo + base + (static_cast<idx_t>(r) < rem ? 1 : 0);
}

static int ownerOf(idx_t n, int p, idx_t g) {
    const idx_t base = n / p, rem = n % p;
    const idx_t cut = rem * (base + 1);
    return g < cut ? static_cast<int>(g / (base + 1)) : static_cast<int>(rem + (g - cut) / base);
}

// Build the local part of a 2D square grid as an unstructured mesh
void buildSquare2D(DistMesh& m, const int n_elems_root) {
    const idx_t n_total = static_cast<idx_t>(n_elems_root) * n_elems_root;
    m.n_total = n_total;
    partitionRange(n_total, m.size, m.rank, m.lo, m.hi);
    const size_t nl = m.n_local = m.hi - m.lo;
    const idx_t lo = m.lo, hi = m.hi;

    m.mat.assign(nl, DEFAULT_MAT_ID);
    m.meta.assign(nl, 0);
    const int MC = 4;  // 2D grid: at most 4 neighbours
    m.max_conn = nl ? MC : 0;
    std::vector<idx_t> gidx(static_cast<size_t>(MC) * nl);
    m.conn_flux.assign(static_cast<size_t>(MC) * nl, 0.0);

    const int last = n_elems_root - 1;
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nl; ++i) {
        const idx_t g = lo + i;
        const int x = static_cast<int>(g / n_elems_root);
        const int y = static_cast<int>(g % n_elems_root);
        int nc = 0;
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                gidx[nc * nl + i] = static_cast<idx_t>(nx) * n_elems_root + ny;
                m.conn_flux[nc * nl + i] = 1.0;
                nc++;
            }
        }
        m.meta[i] = static_cast<uint8_t>(nc);
        // Corner elements as inflow/outflow
        if (x == 0 && y == 0) m.mat[i] = INFLOW_MAT_ID;
        else if (x == 0 && y == last) m.mat[i] = OUTFLOW_MAT_ID;
        else if (x == last && y == 0) m.mat[i] = OUTFLOW_MAT_ID;
        else if (x == last && y == last) m.mat[i] = INFLOW_MAT_ID;
    }

    // Ghost analysis: collect remote neighbours
    std::vector<std::vector<idx_t>> tl(omp_get_max_threads());
    #pragma omp parallel
    {
        auto& v = tl[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for (size_t i = 0; i < nl; ++i) {
            const int nc = m.meta[i] & 0x7f;
            for (int j = 0; j < nc; ++j) {
                const idx_t g = gidx[j * nl + i];
                if (g < lo || g >= hi) v.push_back(g);
            }
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    for (auto& v : tl) m.ghost_gid.insert(m.ghost_gid.end(), v.begin(), v.end());
    std::sort(m.ghost_gid.begin(), m.ghost_gid.end());
    m.ghost_gid.erase(std::unique(m.ghost_gid.begin(), m.ghost_gid.end()), m.ghost_gid.end());
    m.n_ghost = m.ghost_gid.size();

    // Remap to local indices; flag boundary elements
    m.conn_idx.assign(static_cast<size_t>(MC) * nl, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nl; ++i) {
        const int nc = m.meta[i] & 0x7f;
        bool bnd = false;
        for (int j = 0; j < nc; ++j) {
            const idx_t g = gidx[j * nl + i];
            uint32_t l;
            if (g >= lo && g < hi) {
                l = static_cast<uint32_t>(g - lo);
            } else {
                l = static_cast<uint32_t>(nl + (std::lower_bound(m.ghost_gid.begin(), m.ghost_gid.end(), g) -
                                                m.ghost_gid.begin()));
                bnd = true;
            }
            m.conn_idx[j * nl + i] = l;
        }
        if (bnd) m.meta[i] |= BOUNDARY_FLAG;
    }
    for (size_t i = 0; i < nl; ++i)
        if (m.meta[i] & BOUNDARY_FLAG) m.boundary.push_back(static_cast<uint32_t>(i));

    // Communication schedule
    const int P = m.size;
    std::vector<int> rcnt(P, 0), scnt(P, 0);
    for (idx_t g : m.ghost_gid) rcnt[ownerOf(n_total, P, g)]++;
    MPI_Alltoall(rcnt.data(), 1, MPI_INT, scnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rdis(P, 0), sdis(P, 0);
    for (int p = 1; p < P; ++p) {
        rdis[p] = rdis[p - 1] + rcnt[p - 1];
        sdis[p] = sdis[p - 1] + scnt[p - 1];
    }
    const size_t n_send = static_cast<size_t>(sdis[P - 1]) + scnt[P - 1];
    std::vector<idx_t> req(n_send);
    MPI_Alltoallv(m.ghost_gid.data(), rcnt.data(), rdis.data(), MPI_UINT64_T, req.data(), scnt.data(),
                  sdis.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    m.send_idx.resize(n_send);
    for (size_t k = 0; k < n_send; ++k) m.send_idx[k] = static_cast<uint32_t>(req[k] - lo);
    for (int p = 0; p < P; ++p) {
        if (rcnt[p]) { m.recv_peer.push_back(p); m.recv_cnt.push_back(rcnt[p]); m.recv_off.push_back(rdis[p]); }
        if (scnt[p]) { m.send_peer.push_back(p); m.send_cnt.push_back(scnt[p]); m.send_off.push_back(sdis[p]); }
    }
}

// Update kernel. LIST=false: all non-boundary elements; LIST=true: the given list.
template <bool LIST>
__global__ void stepKernel(const uint32_t* __restrict__ list, size_t count, size_t nl, int max_conn,
                           const uint8_t* __restrict__ mat, const uint8_t* __restrict__ meta,
                           const uint32_t* __restrict__ cidx, const val_t* __restrict__ cflux,
                           const val_t* __restrict__ cur_e, const val_t* __restrict__ cur_f,
                           val_t* __restrict__ next_e, val_t* __restrict__ next_f) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < count;
         t += (size_t)gridDim.x * blockDim.x) {
        const size_t i = LIST ? list[t] : t;
        const uint8_t mt = meta[i];
        if (!LIST && (mt & BOUNDARY_FLAG)) continue;
        const int nc = mt & 0x7f;
        const uint8_t mi = mat[i];
        const val_t coeff = d_coeff[mi];
        const val_t e = cur_e[i];
        val_t total = d_ext[mi];
        for (int j = 0; j < nc; ++j) {
            const uint32_t nb = cidx[j * nl + i];
            const val_t cf = cflux[j * nl + i];
            total += (cur_e[nb] - e) * coeff * cf * 0.25;
        }
        next_e[i] = e + total;
        next_f[i] = cur_f[i] + fabs(total);
    }
}

__global__ void packKernel(const uint32_t* __restrict__ idx, size_t n, const val_t* __restrict__ e,
                           val_t* __restrict__ out) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t < n) out[t] = e[idx[t]];
}

template <typename T>
static T* devAlloc(size_t n, const T* host = nullptr) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    if (host && n) CUDA_CHECK(cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// Device-resident state and simulation driver. Leaves final state in h_energy/h_flux.
void runSimulation(const DistMesh& m, const int n_iters, std::vector<val_t>& h_energy,
                   std::vector<val_t>& h_flux, double& elapsed_ms) {
    const size_t nl = m.n_local, ng = m.n_ghost;
    const val_t coeff[3] = {0.8, 0.8, 0.8};
    const val_t ext[3] = {0.0, 0.5, -0.5};
    CUDA_CHECK(cudaMemcpyToSymbol(d_coeff, coeff, sizeof(coeff)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ext, ext, sizeof(ext)));

    const size_t nconn = static_cast<size_t>(m.max_conn) * nl;
    uint8_t* d_mat = devAlloc(nl, m.mat.data());
    uint8_t* d_meta = devAlloc(nl, m.meta.data());
    uint32_t* d_cidx = devAlloc(nconn, m.conn_idx.data());
    val_t* d_cflux = devAlloc(nconn, m.conn_flux.data());
    uint32_t* d_bnd = devAlloc(m.boundary.size(), m.boundary.data());
    uint32_t* d_send_idx = devAlloc(m.send_idx.size(), m.send_idx.data());

    val_t* d_e[2] = {devAlloc<val_t>(nl + ng), devAlloc<val_t>(nl + ng)};
    val_t* d_f[2] = {devAlloc<val_t>(nl), devAlloc<val_t>(nl)};
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMemset(d_e[b], 0, (nl + ng) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_f[b], 0, nl * sizeof(val_t)));
    }

    const size_t n_send = m.send_idx.size();
    val_t* d_sendbuf = devAlloc<val_t>(n_send);
    val_t *h_send = nullptr, *h_recv = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send, std::max<size_t>(n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_recv, std::max<size_t>(ng, 1) * sizeof(val_t)));

    cudaStream_t s_comp, s_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comm, cudaStreamNonBlocking));
    cudaEvent_t ev_bnd, ev_h2d;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_bnd, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d, cudaEventDisableTiming));

    const int TB = 256;
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const size_t maxBlocks = static_cast<size_t>(sms) * 16;
    auto blocksFor = [&](size_t n) { return (int)std::max<size_t>(1, std::min((n + TB - 1) / TB, maxBlocks)); };

    const bool comm = !m.recv_peer.empty() || !m.send_peer.empty();
    std::vector<MPI_Request> reqs(m.recv_peer.size() + m.send_peer.size());

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        if (comm) {
            // Pack and download the values neighbours need
            if (n_send) {
                packKernel<<<(int)((n_send + TB - 1) / TB), TB, 0, s_comm>>>(d_send_idx, n_send, d_e[cur],
                                                                             d_sendbuf);
                CUDA_CHECK(cudaMemcpyAsync(h_send, d_sendbuf, n_send * sizeof(val_t), cudaMemcpyDeviceToHost,
                                           s_comm));
            }
        }
        // Interior update overlaps with communication
        if (nl)
            stepKernel<false><<<blocksFor(nl), TB, 0, s_comp>>>(nullptr, nl, nl, m.max_conn, d_mat, d_meta,
                                                                  d_cidx, d_cflux, d_e[cur], d_f[cur], d_e[nxt],
                                                                  d_f[nxt]);
        if (comm) {
            CUDA_CHECK(cudaStreamSynchronize(s_comm));
            size_t r = 0;
            for (size_t k = 0; k < m.recv_peer.size(); ++k)
                MPI_Irecv(h_recv + m.recv_off[k], m.recv_cnt[k], MPI_DOUBLE, m.recv_peer[k], iter,
                          MPI_COMM_WORLD, &reqs[r++]);
            for (size_t k = 0; k < m.send_peer.size(); ++k)
                MPI_Isend(h_send + m.send_off[k], m.send_cnt[k], MPI_DOUBLE, m.send_peer[k], iter,
                          MPI_COMM_WORLD, &reqs[r++]);
            MPI_Waitall((int)r, reqs.data(), MPI_STATUSES_IGNORE);
            if (ng) {
                CUDA_CHECK(cudaMemcpyAsync(d_e[cur] + nl, h_recv, ng * sizeof(val_t), cudaMemcpyHostToDevice,
                                           s_comm));
            }
            CUDA_CHECK(cudaEventRecord(ev_h2d, s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(s_comp, ev_h2d, 0));
            if (!m.boundary.empty())
                stepKernel<true><<<blocksFor(m.boundary.size()), TB, 0, s_comp>>>(
                    d_bnd, m.boundary.size(), nl, m.max_conn, d_mat, d_meta, d_cidx, d_cflux, d_e[cur],
                    d_f[cur], d_e[nxt], d_f[nxt]);
            // Next iteration's pack must see the finished update
            CUDA_CHECK(cudaEventRecord(ev_bnd, s_comp));
            CUDA_CHECK(cudaStreamWaitEvent(s_comm, ev_bnd, 0));
        }
        cur = nxt;
    }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    h_energy.resize(nl);
    h_flux.resize(nl);
    if (nl) {
        CUDA_CHECK(cudaMemcpy(h_energy.data(), d_e[cur], nl * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), d_f[cur], nl * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventDestroy(ev_bnd));
    CUDA_CHECK(cudaEventDestroy(ev_h2d));
    CUDA_CHECK(cudaStreamDestroy(s_comp));
    CUDA_CHECK(cudaStreamDestroy(s_comm));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    cudaFree(d_mat); cudaFree(d_meta); cudaFree(d_cidx); cudaFree(d_cflux); cudaFree(d_bnd);
    cudaFree(d_send_idx); cudaFree(d_sendbuf);
    for (int b = 0; b < 2; ++b) { cudaFree(d_e[b]); cudaFree(d_f[b]); }
}

// Validate simulation results (full, ordered data on rank 0)
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

// Hash of the local part of the results (XOR-combinable across ranks)
uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux, idx_t first) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t k = 0; k < energy.size(); ++k) {
        const uint64_t i = first + k;
        uint64_t e, f;
        memcpy(&e, &energy[k], sizeof(e));
        memcpy(&f, &flux[k], sizeof(f));
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    DistMesh mesh;
    MPI_Comm_rank(MPI_COMM_WORLD, &mesh.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mesh.size);
    const bool root = mesh.rank == 0;

    // One GPU per rank (round robin among ranks on the same node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mesh.rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

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

    const int n_elems = n_elems_root * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mesh.size, omp_get_max_threads());
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(mesh, n_elems_root);

    // Calculate memory usage (same accounting as the reference)
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

    std::vector<val_t> energy, flux;
    double elapsed = 0.0;
    runSimulation(mesh, n_iters, energy, flux, elapsed);
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(elapsed);

    uint64_t local_hash = computeHash(energy, flux, mesh.lo);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);

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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    bool valid = true;
    if (printResults || validate) {
        // Gather full ordered results on rank 0
        std::vector<int> counts(mesh.size), displs(mesh.size);
        const int nl = static_cast<int>(mesh.n_local);
        MPI_Gather(&nl, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        int acc = 0;
        for (int p = 0; p < mesh.size; ++p) { displs[p] = acc; acc += counts[p]; }
        std::vector<val_t> all_e, all_f;
        if (root) { all_e.resize(n_elems); all_f.resize(n_elems); }
        MPI_Gatherv(energy.data(), nl, MPI_DOUBLE, all_e.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Gatherv(flux.data(), nl, MPI_DOUBLE, all_f.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (root) {
            if (printResults) print_results(all_e, "ElementEnergy");
            if (validate) valid = validateResults(all_e, all_f);
        }
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
