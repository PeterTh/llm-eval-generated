// Unstructured mesh energy transfer benchmark.
//
// Parallelization (hybrid, all three always active):
//   * MPI     - the global element index space is block-partitioned across ranks.
//               Each rank owns a contiguous slice of elements and exchanges only the
//               energy values of "ghost" neighbours with the ranks that own them.
//   * CUDA    - each rank drives one GPU. The per-element update is a data-parallel
//               kernel. Elements whose neighbours are all local ("interior") are
//               updated concurrently with the halo exchange; elements that depend on
//               ghost values ("boundary") are updated once the halo has arrived.
//   * OpenMP  - host-side construction of the local mesh, connectivity and the
//               ghost/communication plan is multi-threaded.
//
// Numerical semantics are identical to the sequential reference: every time step is
// a Jacobi-style update (all reads come from the previous state), and every element
// accumulates its fluxes in the same order with the same floating-point operations.

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
// Maximum number of distinct materials supported by the device kernel
constexpr int MAX_MATERIALS = 8;
// CUDA thread block size
constexpr unsigned CUDA_BLOCK = 256;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (global ids)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Host-side mesh of the locally owned elements
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;  // local partition only
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 2);                                             \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Partitioning helpers
// ---------------------------------------------------------------------------

// First global element owned by rank r under a block partition of N elements.
static inline idx_t partBegin(int r, int P, idx_t N) {
    return (N * static_cast<idx_t>(r)) / static_cast<idx_t>(P);
}

// Rank owning global element g.
static int ownerOf(idx_t g, int P, idx_t N) {
    int r = static_cast<int>((g * static_cast<idx_t>(P)) / N);
    if (r >= P) r = P - 1;
    while (partBegin(r, P, N) > g) --r;
    while (partBegin(r + 1, P, N) <= g) ++r;
    return r;
}

// ---------------------------------------------------------------------------
// Mesh construction (host, OpenMP over local elements)
// ---------------------------------------------------------------------------

// Build the static data of the locally owned elements [begin, end) of a 2D
// n_root x n_root square grid. Connectivity uses global element ids.
static void buildLocalMesh(World& world, const int n_root, const idx_t begin, const idx_t end) {
    const idx_t n = static_cast<idx_t>(n_root);
    const idx_t last = n - 1;
    const idx_t n_local = end - begin;

    // Initialize materials
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.assign(n_local, ElementStatic{});

    const long long offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t g = begin + li;
        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        const long long x = static_cast<long long>(g / n);
        const long long y = static_cast<long long>(g % n);

        // Connect to neighbors (up, down, left, right) that lie inside the grid
        for (int k = 0; k < 4; ++k) {
            const long long nx = x + offsets[k][0];
            const long long ny = y + offsets[k][1];
            if (nx >= 0 && nx < static_cast<long long>(n) && ny >= 0 && ny < static_cast<long long>(n)) {
                elem.connected_idx[elem.num_connections] =
                    static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Corner elements are inflow/outflow sources (applied in the original order;
        // later assignments win when the grid is degenerate)
        if (g == 0) elem.material_idx = INFLOW_MAT_ID;                 // [0][0]
        if (g == last) elem.material_idx = OUTFLOW_MAT_ID;             // [0][last]
        if (g == last * n) elem.material_idx = OUTFLOW_MAT_ID;         // [last][0]
        if (g == last * n + last) elem.material_idx = INFLOW_MAT_ID;   // [last][last]
    }
}

// Structure-of-arrays device-ready connectivity of the local partition.
struct HostLayout {
    idx_t begin = 0;
    idx_t n_local = 0;
    std::vector<uint8_t> material;     // [i]
    std::vector<uint8_t> num_conn;     // [i]
    std::vector<uint32_t> conn_idx;    // [j * n_local + i], extended local index
    std::vector<val_t> conn_flux;      // [j * n_local + i]
    std::vector<uint32_t> interior;    // local elements with only local neighbours
    std::vector<uint32_t> boundary;    // local elements with at least one ghost neighbour
    std::vector<idx_t> ghosts;         // sorted global ids of remote neighbours
};

// Translate the global connectivity into local/extended indices. Ghost elements
// occupy extended slots [n_local, n_local + n_ghost) in ascending global-id order.
static void buildConnectivity(const World& world, const idx_t begin, const idx_t end, HostLayout& L) {
    const idx_t n_local = end - begin;
    L.begin = begin;
    L.n_local = n_local;

    // Collect remote neighbours (per-thread buffers, merged afterwards)
    std::vector<std::vector<idx_t>> remote(static_cast<size_t>(omp_get_max_threads()));
    #pragma omp parallel
    {
        std::vector<idx_t>& mine = remote[static_cast<size_t>(omp_get_thread_num())];
        #pragma omp for schedule(static)
        for (idx_t li = 0; li < n_local; ++li) {
            const ElementStatic& e = world.elements_static[li];
            for (idx_t j = 0; j < e.num_connections; ++j) {
                const idx_t nb = e.connected_idx[j];
                if (nb < begin || nb >= end) mine.push_back(nb);
            }
        }
    }
    L.ghosts.clear();
    for (const auto& v : remote) L.ghosts.insert(L.ghosts.end(), v.begin(), v.end());
    std::sort(L.ghosts.begin(), L.ghosts.end());
    L.ghosts.erase(std::unique(L.ghosts.begin(), L.ghosts.end()), L.ghosts.end());

    const idx_t n_ghost = static_cast<idx_t>(L.ghosts.size());
    if (n_local + n_ghost >= std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr, "Partition too large for 32-bit device indices\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const size_t nl = static_cast<size_t>(n_local);
    L.material.assign(nl, 0);
    L.num_conn.assign(nl, 0);
    L.conn_idx.assign(nl * MAX_CONNECTIONS, 0);
    L.conn_flux.assign(nl * MAX_CONNECTIONS, 0.0);
    std::vector<uint8_t> is_boundary(nl, 0);

    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const ElementStatic& e = world.elements_static[li];
        L.material[li] = static_cast<uint8_t>(e.material_idx);
        L.num_conn[li] = static_cast<uint8_t>(e.num_connections);
        bool bnd = false;
        for (int j = 0; j < static_cast<int>(e.num_connections); ++j) {
            const idx_t nb = e.connected_idx[j];
            uint32_t ext;
            if (nb >= begin && nb < end) {
                ext = static_cast<uint32_t>(nb - begin);
            } else {
                const auto it = std::lower_bound(L.ghosts.begin(), L.ghosts.end(), nb);
                ext = static_cast<uint32_t>(n_local + static_cast<idx_t>(it - L.ghosts.begin()));
                bnd = true;
            }
            const size_t k = static_cast<size_t>(j) * nl + static_cast<size_t>(li);
            L.conn_idx[k] = ext;
            L.conn_flux[k] = e.connected_flux[j];
        }
        is_boundary[li] = bnd ? 1 : 0;
    }

    L.interior.clear();
    L.boundary.clear();
    for (idx_t li = 0; li < n_local; ++li) {
        (is_boundary[li] ? L.boundary : L.interior).push_back(static_cast<uint32_t>(li));
    }
}

// ---------------------------------------------------------------------------
// Halo exchange plan (MPI)
// ---------------------------------------------------------------------------

struct Plan {
    std::vector<int> send_count, send_displ;  // per rank: values sent to that rank
    std::vector<int> recv_count, recv_displ;  // per rank: ghost values received from it
    std::vector<int> send_peers, recv_peers;  // ranks with nonzero counts
    std::vector<uint32_t> send_idx;           // local ids of values to pack, peer-ordered
};

static void buildExchangePlan(const HostLayout& L, const idx_t N, const int rank, const int nranks, Plan& p) {
    (void)rank;
    p.recv_count.assign(nranks, 0);
    p.recv_displ.assign(nranks, 0);
    // Ghosts are sorted by global id, so they are grouped by owner in rank order.
    for (idx_t g : L.ghosts) p.recv_count[ownerOf(g, nranks, N)]++;
    int acc = 0;
    for (int r = 0; r < nranks; ++r) { p.recv_displ[r] = acc; acc += p.recv_count[r]; }

    // Owners learn which of their elements each rank needs
    p.send_count.assign(nranks, 0);
    p.send_displ.assign(nranks, 0);
    MPI_Alltoall(p.recv_count.data(), 1, MPI_INT, p.send_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    acc = 0;
    for (int r = 0; r < nranks; ++r) { p.send_displ[r] = acc; acc += p.send_count[r]; }

    std::vector<idx_t> requested(static_cast<size_t>(acc));
    MPI_Alltoallv(L.ghosts.data(), p.recv_count.data(), p.recv_displ.data(), MPI_UINT64_T,
                  requested.data(), p.send_count.data(), p.send_displ.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);

    p.send_idx.resize(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) {
        p.send_idx[k] = static_cast<uint32_t>(requested[k] - L.begin);
    }

    p.send_peers.clear();
    p.recv_peers.clear();
    for (int r = 0; r < nranks; ++r) {
        if (p.send_count[r] > 0) p.send_peers.push_back(r);
        if (p.recv_count[r] > 0) p.recv_peers.push_back(r);
    }
}

// ---------------------------------------------------------------------------
// Device data and kernels (CUDA)
// ---------------------------------------------------------------------------

struct DeviceMaterials {
    val_t transfer_coeff[MAX_MATERIALS];
    val_t external_flow[MAX_MATERIALS];
};

// Gather the energies that neighbouring ranks need into a contiguous send buffer.
__global__ void gatherSendKernel(const val_t* __restrict__ energy, const uint32_t* __restrict__ idx,
                                 val_t* __restrict__ out, uint32_t n) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) out[t] = energy[idx[t]];
}

// Update the listed elements: total flux accumulated in connection order, starting
// from the external flow, exactly as the reference computeFlux/runSimulation.
__global__ void updateElementsKernel(const uint32_t* __restrict__ list, uint32_t count, uint32_t n_local,
                                     const uint8_t* __restrict__ material,
                                     const uint8_t* __restrict__ num_conn,
                                     const uint32_t* __restrict__ conn_idx,
                                     const val_t* __restrict__ conn_flux,
                                     DeviceMaterials mats,
                                     const val_t* __restrict__ energy_in,
                                     const val_t* __restrict__ flux_in,
                                     val_t* __restrict__ energy_out,
                                     val_t* __restrict__ flux_out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const uint32_t i = list[t];

    const uint32_t m = material[i];
    const val_t coeff = mats.transfer_coeff[m];
    const val_t external = mats.external_flow[m];
    const val_t self = energy_in[i];
    const uint32_t nc = num_conn[i];

    val_t total_flux = external;
    for (uint32_t j = 0; j < nc; ++j) {
        const size_t k = static_cast<size_t>(j) * n_local + i;
        const val_t other = energy_in[conn_idx[k]];
        total_flux += (other - self) * coeff * conn_flux[k] * 0.25;
    }

    energy_out[i] = self + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

struct DeviceSim {
    uint32_t n_local = 0, n_ghost = 0, n_int = 0, n_bnd = 0, n_send = 0;
    uint8_t* material = nullptr;
    uint8_t* num_conn = nullptr;
    uint32_t* conn_idx = nullptr;
    val_t* conn_flux = nullptr;
    uint32_t* interior = nullptr;
    uint32_t* boundary = nullptr;
    uint32_t* send_idx = nullptr;
    val_t* energy[2] = {nullptr, nullptr};   // double-buffered; ghost slots at the tail
    val_t* flux[2] = {nullptr, nullptr};     // double-buffered
    val_t* send_buf = nullptr;               // device-packed halo values
    val_t* h_send = nullptr;                 // pinned host staging (send)
    val_t* h_recv = nullptr;                 // pinned host staging (receive)
    cudaStream_t s_int = nullptr;            // interior update stream
    cudaStream_t s_bnd = nullptr;            // halo transfer + boundary update stream
};

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

template <typename T>
static void uploadVector(T* d, const std::vector<T>& h) {
    if (!h.empty()) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

static val_t* pinnedAlloc(size_t count) {
    void* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(count, 1) * sizeof(val_t)));
    return static_cast<val_t*>(p);
}

static void setupDevice(DeviceSim& s, const HostLayout& L, const Plan& plan) {
    s.n_local = static_cast<uint32_t>(L.n_local);
    s.n_ghost = static_cast<uint32_t>(L.ghosts.size());
    s.n_int = static_cast<uint32_t>(L.interior.size());
    s.n_bnd = static_cast<uint32_t>(L.boundary.size());
    s.n_send = static_cast<uint32_t>(plan.send_idx.size());

    const size_t nl = s.n_local;
    const size_t next = nl + s.n_ghost;

    s.material = deviceAlloc<uint8_t>(nl);
    uploadVector(s.material, L.material);
    s.num_conn = deviceAlloc<uint8_t>(nl);
    uploadVector(s.num_conn, L.num_conn);
    s.conn_idx = deviceAlloc<uint32_t>(nl * MAX_CONNECTIONS);
    uploadVector(s.conn_idx, L.conn_idx);
    s.conn_flux = deviceAlloc<val_t>(nl * MAX_CONNECTIONS);
    uploadVector(s.conn_flux, L.conn_flux);
    s.interior = deviceAlloc<uint32_t>(s.n_int);
    uploadVector(s.interior, L.interior);
    s.boundary = deviceAlloc<uint32_t>(s.n_bnd);
    uploadVector(s.boundary, L.boundary);
    s.send_idx = deviceAlloc<uint32_t>(s.n_send);
    uploadVector(s.send_idx, plan.send_idx);

    for (int k = 0; k < 2; ++k) {
        s.energy[k] = deviceAlloc<val_t>(next);
        CUDA_CHECK(cudaMemset(s.energy[k], 0, std::max<size_t>(next, 1) * sizeof(val_t)));
        s.flux[k] = deviceAlloc<val_t>(nl);
        CUDA_CHECK(cudaMemset(s.flux[k], 0, std::max<size_t>(nl, 1) * sizeof(val_t)));
    }
    s.send_buf = deviceAlloc<val_t>(s.n_send);
    s.h_send = pinnedAlloc(s.n_send);
    s.h_recv = pinnedAlloc(s.n_ghost);

    CUDA_CHECK(cudaStreamCreate(&s.s_int));
    CUDA_CHECK(cudaStreamCreate(&s.s_bnd));
}

static void freeDevice(DeviceSim& s) {
    cudaFree(s.material);
    cudaFree(s.num_conn);
    cudaFree(s.conn_idx);
    cudaFree(s.conn_flux);
    cudaFree(s.interior);
    cudaFree(s.boundary);
    cudaFree(s.send_idx);
    for (int k = 0; k < 2; ++k) {
        cudaFree(s.energy[k]);
        cudaFree(s.flux[k]);
    }
    cudaFree(s.send_buf);
    cudaFreeHost(s.h_send);
    cudaFreeHost(s.h_recv);
    cudaStreamDestroy(s.s_int);
    cudaStreamDestroy(s.s_bnd);
}

static inline unsigned gridFor(uint32_t n) { return (n + CUDA_BLOCK - 1) / CUDA_BLOCK; }

// Run n_iters time steps. Returns the index (0/1) of the buffer holding the final state.
static int runSimulation(DeviceSim& s, const Plan& plan, const DeviceMaterials& mats, const int n_iters) {
    const uint32_t n_local = s.n_local;
    int cur = 0;

    std::vector<MPI_Request> requests;
    requests.reserve(plan.send_peers.size() + plan.recv_peers.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* e_in = s.energy[cur];
        val_t* f_in = s.flux[cur];
        val_t* e_out = s.energy[1 - cur];
        val_t* f_out = s.flux[1 - cur];

        // 1. Pack the halo values that other ranks need (device)
        if (s.n_send > 0) {
            gatherSendKernel<<<gridFor(s.n_send), CUDA_BLOCK, 0, s.s_bnd>>>(e_in, s.send_idx, s.send_buf, s.n_send);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(s.h_send, s.send_buf, s.n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s.s_bnd));
            CUDA_CHECK(cudaStreamSynchronize(s.s_bnd));
        }

        // 2. Interior elements need no ghost data: update them while the halo travels
        if (s.n_int > 0) {
            updateElementsKernel<<<gridFor(s.n_int), CUDA_BLOCK, 0, s.s_int>>>(
                s.interior, s.n_int, n_local, s.material, s.num_conn, s.conn_idx, s.conn_flux,
                mats, e_in, f_in, e_out, f_out);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Halo exchange with neighbouring ranks (host staging, pinned buffers)
        requests.clear();
        for (int r : plan.recv_peers) {
            MPI_Request req;
            MPI_Irecv(s.h_recv + plan.recv_displ[r], plan.recv_count[r], MPI_DOUBLE, r, 0,
                      MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        for (int q : plan.send_peers) {
            MPI_Request req;
            MPI_Isend(s.h_send + plan.send_displ[q], plan.send_count[q], MPI_DOUBLE, q, 0,
                      MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // Ghost values land in the tail of the current energy buffer
        if (s.n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(e_in + n_local, s.h_recv, s.n_ghost * sizeof(val_t),
                                       cudaMemcpyHostToDevice, s.s_bnd));
        }

        // 4. Boundary elements (depend on ghosts), after the halo has been installed
        if (s.n_bnd > 0) {
            updateElementsKernel<<<gridFor(s.n_bnd), CUDA_BLOCK, 0, s.s_bnd>>>(
                s.boundary, s.n_bnd, n_local, s.material, s.num_conn, s.conn_idx, s.conn_flux,
                mats, e_in, f_in, e_out, f_out);
            CUDA_CHECK(cudaGetLastError());
        }

        // 5. Both streams must finish before the buffers are swapped
        CUDA_CHECK(cudaStreamSynchronize(s.s_bnd));
        CUDA_CHECK(cudaStreamSynchronize(s.s_int));

        cur = 1 - cur;
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Validation / reporting (rank 0 only, operates on the global state)
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    auto finish = [](int code) {
        MPI_Finalize();
        return code;
    };

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            return finish(0);
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    if (n_elems_root < 1) {
        if (rank == 0) fprintf(stderr, "Grid size must be at least 1 (got %d)\n", n_elems_root);
        return finish(1);
    }

    const int n_elems = n_elems_root * n_elems_root;
    const idx_t N = static_cast<idx_t>(n_elems);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // One GPU per rank: ranks on a node are spread round-robin over its devices
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % n_devices));

    // Build this rank's partition of the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    const idx_t begin = partBegin(rank, nranks, N);
    const idx_t end = partBegin(rank + 1, nranks, N);

    World world;
    buildLocalMesh(world, n_elems_root, begin, end);
    HostLayout layout;
    buildConnectivity(world, begin, end, layout);
    Plan plan;
    buildExchangePlan(layout, N, rank, nranks, plan);

    DeviceMaterials mats{};
    if (world.materials.size() > static_cast<size_t>(MAX_MATERIALS)) {
        fprintf(stderr, "Too many materials\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    for (size_t m = 0; m < world.materials.size(); ++m) {
        mats.transfer_coeff[m] = world.materials[m].transfer_coeff;
        mats.external_flow[m] = world.materials[m].external_flow;
    }

    DeviceSim sim;
    setupDevice(sim, layout, plan);

    // Calculate memory usage (global, matching the reference accounting)
    const size_t static_mem = static_cast<size_t>(N) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(N) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int final_buf = runSimulation(sim, plan, mats, n_iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start).count();

    // Gather the final state (outside the timed region) in global element order
    std::vector<val_t> local_energy(static_cast<size_t>(layout.n_local));
    std::vector<val_t> local_flux(static_cast<size_t>(layout.n_local));
    if (layout.n_local > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), sim.energy[final_buf], local_energy.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), sim.flux[final_buf], local_flux.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<ElementDynamic> local_dyn(static_cast<size_t>(layout.n_local));
    for (size_t i = 0; i < local_dyn.size(); ++i) {
        local_dyn[i].current_energy = local_energy[i];
        local_dyn[i].total_flux = local_flux[i];
    }

    MPI_Datatype pair_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pair_type);
    MPI_Type_commit(&pair_type);
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = static_cast<int>(partBegin(r + 1, nranks, N) - partBegin(r, nranks, N));
        displs[r] = static_cast<int>(partBegin(r, nranks, N));
    }
    std::vector<ElementDynamic> global_dyn;
    if (rank == 0) global_dyn.resize(static_cast<size_t>(N));
    MPI_Gatherv(local_dyn.data(), static_cast<int>(local_dyn.size()), pair_type,
                global_dyn.data(), counts.data(), displs.data(), pair_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pair_type);

    int exit_code = 0;
    if (rank == 0) {
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
        const uint64_t hash = computeHash(global_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dyn.size());
            for (const auto& elem : global_dyn) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(global_dyn)) exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    freeDevice(sim);
    return finish(exit_code);
}
