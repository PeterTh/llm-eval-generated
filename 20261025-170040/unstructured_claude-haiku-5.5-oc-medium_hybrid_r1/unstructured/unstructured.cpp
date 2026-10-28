// Unstructured mesh energy transfer benchmark -- hybrid MPI + OpenMP + CUDA.
//
// Decomposition:
//   MPI    - the N = n*n global elements are block-partitioned over ranks. Each
//            rank owns a contiguous range of global indices and exchanges only
//            the energies of remote neighbors ("ghosts") with the ranks that
//            actually need them (point-to-point, neighbor-only).
//   CUDA   - each rank drives one GPU (local rank modulo device count). The
//            per-element update runs as a kernel. Interior elements (no remote
//            neighbors) are updated while the halo exchange is in flight.
//   OpenMP - host-side mesh construction, packing of the final state, and the
//            distributed result hash are computed with OpenMP worksharing.
//
// The arithmetic per element and the neighbor summation order are identical to
// the original serial implementation, so results (including the printed hash)
// are bit-for-bit identical for any number of ranks.

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;   // global element index
using val_t = double;
using lidx_t = uint32_t;  // rank-local index: owned elements first, then ghosts

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;
// Capacity of the material table in device constant memory
constexpr int MAX_MATERIALS = 256;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int HALO_TAG = 17;
constexpr int GATHER_TAG = 18;

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

[[noreturn]] void fatal(const char* msg) {
    fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            fatal("CUDA call failed");                                          \
        }                                                                       \
    } while (0)

std::vector<Material> buildMaterials() {
    return {Material{0.8, 0.0},    // Default material
            Material{0.8, 0.5},    // Inflow material
            Material{0.8, -0.5}};  // Outflow material
}

// Connectivity and material of global element g in an n x n grid.
// Neighbor order (+x, -x, +y, -y) and corner assignment match the original.
ElementStatic buildElementStatic(idx_t g, idx_t n) {
    ElementStatic e{};
    e.material_idx = DEFAULT_MAT_ID;
    e.num_connections = 0;

    const idx_t x = g / n;
    const idx_t y = g % n;
    const long long offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int k = 0; k < 4; ++k) {
        const long long nx = static_cast<long long>(x) + offsets[k][0];
        const long long ny = static_cast<long long>(y) + offsets[k][1];
        if (nx >= 0 && nx < static_cast<long long>(n) && ny >= 0 && ny < static_cast<long long>(n)) {
            e.connected_idx[e.num_connections] = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
            e.connected_flux[e.num_connections] = 1.0;
            e.num_connections++;
        }
    }

    // Corner elements as inflow/outflow (applied in the original order; last wins)
    const idx_t last = n - 1;
    if (g == 0) e.material_idx = INFLOW_MAT_ID;                    // (0, 0)
    if (g == last) e.material_idx = OUTFLOW_MAT_ID;                // (0, last)
    if (g == last * n) e.material_idx = OUTFLOW_MAT_ID;            // (last, 0)
    if (g == last * n + last) e.material_idx = INFLOW_MAT_ID;      // (last, last)
    return e;
}

// Block partition of N elements: rank r owns [starts[r], starts[r+1])
std::vector<idx_t> partitionStarts(idx_t N, int nprocs) {
    std::vector<idx_t> starts(nprocs + 1);
    const idx_t q = N / static_cast<idx_t>(nprocs);
    const idx_t rem = N % static_cast<idx_t>(nprocs);
    for (int r = 0; r <= nprocs; ++r) {
        const idx_t ir = static_cast<idx_t>(r);
        starts[r] = ir * q + std::min(ir, rem);
    }
    return starts;
}

int ownerOf(const std::vector<idx_t>& starts, idx_t g) {
    const auto it = std::upper_bound(starts.begin(), starts.end(), g);
    return static_cast<int>(it - starts.begin()) - 1;
}

// Per-rank mesh in local ordering. Owned elements are ordered interior-first
// (no remote neighbors) then boundary; ghosts are appended after the owned ones.
struct RankMesh {
    idx_t lo = 0;                 // first owned global index
    idx_t n_owned = 0;
    idx_t n_interior = 0;         // local [0, n_interior) need no ghost values
    idx_t n_ghost = 0;
    uint32_t max_conn = 0;
    std::vector<idx_t> global_of_local;  // owned local -> global index
    std::vector<lidx_t> local_of_owned;  // (global - lo) -> owned local index
    std::vector<idx_t> ghost_global;     // sorted; grouped by owning rank
    std::vector<uint8_t> material;       // per owned local element
    std::vector<uint8_t> num_conn;       // per owned local element
    std::vector<lidx_t> nbr;             // slot-major: nbr[j * n_owned + i]
    std::vector<val_t> flux;             // slot-major: flux[j * n_owned + i]
};

// Halo communication plan. Ghost slots for each owner are contiguous.
struct HaloPlan {
    std::vector<int> recv_count, recv_disp;  // values received from each rank
    std::vector<int> send_count, send_disp;  // values sent to each rank
    std::vector<lidx_t> send_local;          // owned local indices to pack, by destination
};

RankMesh buildRankMesh(idx_t n, const std::vector<idx_t>& starts, int rank) {
    RankMesh m;
    m.lo = starts[rank];
    m.n_owned = starts[rank + 1] - starts[rank];
    const idx_t lo = m.lo;
    const idx_t hi = lo + m.n_owned;
    const idx_t no = m.n_owned;

    // Classify owned elements: boundary if any neighbor is remote
    std::vector<uint8_t> is_boundary(no, 0);
#pragma omp parallel for schedule(static)
    for (std::int64_t k = 0; k < static_cast<std::int64_t>(no); ++k) {
        const ElementStatic s = buildElementStatic(lo + static_cast<idx_t>(k), n);
        for (idx_t j = 0; j < s.num_connections; ++j) {
            const idx_t nb = s.connected_idx[j];
            if (nb < lo || nb >= hi) {
                is_boundary[k] = 1;
                break;
            }
        }
    }

    // Remote neighbors of boundary elements form the ghost set
    for (idx_t k = 0; k < no; ++k) {
        if (!is_boundary[k]) continue;
        const ElementStatic s = buildElementStatic(lo + k, n);
        for (idx_t j = 0; j < s.num_connections; ++j) {
            const idx_t nb = s.connected_idx[j];
            if (nb < lo || nb >= hi) m.ghost_global.push_back(nb);
        }
    }
    std::sort(m.ghost_global.begin(), m.ghost_global.end());
    m.ghost_global.erase(std::unique(m.ghost_global.begin(), m.ghost_global.end()),
                         m.ghost_global.end());
    m.n_ghost = static_cast<idx_t>(m.ghost_global.size());

    // Local ordering: interior first, then boundary; ascending global index in each group
    m.global_of_local.reserve(no);
    for (idx_t k = 0; k < no; ++k) {
        if (!is_boundary[k]) m.global_of_local.push_back(lo + k);
    }
    m.n_interior = static_cast<idx_t>(m.global_of_local.size());
    for (idx_t k = 0; k < no; ++k) {
        if (is_boundary[k]) m.global_of_local.push_back(lo + k);
    }

    if (no + m.n_ghost > static_cast<idx_t>(std::numeric_limits<lidx_t>::max())) {
        fatal("per-rank element count exceeds 32-bit local index range; use more ranks");
    }

    m.local_of_owned.assign(no, 0);
    for (idx_t l = 0; l < no; ++l) {
        m.local_of_owned[m.global_of_local[l] - lo] = static_cast<lidx_t>(l);
    }

    // Max connections per element (sets the slot-major stride)
    uint32_t max_conn = 0;
#pragma omp parallel for reduction(max : max_conn) schedule(static)
    for (std::int64_t l = 0; l < static_cast<std::int64_t>(no); ++l) {
        const ElementStatic s = buildElementStatic(m.global_of_local[l], n);
        max_conn = std::max(max_conn, static_cast<uint32_t>(s.num_connections));
    }
    m.max_conn = max_conn;

    // Build local (ELL-style, slot-major) connectivity in local element order
    m.material.assign(no, 0);
    m.num_conn.assign(no, 0);
    m.nbr.assign(static_cast<size_t>(max_conn) * no, 0);
    m.flux.assign(static_cast<size_t>(max_conn) * no, 0.0);
#pragma omp parallel for schedule(static)
    for (std::int64_t ll = 0; ll < static_cast<std::int64_t>(no); ++ll) {
        const idx_t l = static_cast<idx_t>(ll);
        const ElementStatic s = buildElementStatic(m.global_of_local[l], n);
        m.material[l] = static_cast<uint8_t>(s.material_idx);
        m.num_conn[l] = static_cast<uint8_t>(s.num_connections);
        for (idx_t j = 0; j < s.num_connections; ++j) {
            const idx_t nb = s.connected_idx[j];
            lidx_t target;
            if (nb >= lo && nb < hi) {
                target = m.local_of_owned[nb - lo];
            } else {
                const auto pos = std::lower_bound(m.ghost_global.begin(), m.ghost_global.end(), nb);
                target = static_cast<lidx_t>(no + static_cast<idx_t>(pos - m.ghost_global.begin()));
            }
            const size_t slot = static_cast<size_t>(j) * no + l;
            m.nbr[slot] = target;
            m.flux[slot] = s.connected_flux[j];
        }
    }
    return m;
}

// Exchange ghost requests with the owning ranks to learn what each rank must send.
HaloPlan buildHaloPlan(RankMesh& m, const std::vector<idx_t>& starts, int nprocs, MPI_Comm comm) {
    HaloPlan h;
    h.recv_count.assign(nprocs, 0);
    for (idx_t g : m.ghost_global) h.recv_count[ownerOf(starts, g)]++;
    h.recv_disp.assign(nprocs, 0);
    for (int r = 1; r < nprocs; ++r) h.recv_disp[r] = h.recv_disp[r - 1] + h.recv_count[r - 1];

    h.send_count.assign(nprocs, 0);
    MPI_Alltoall(h.recv_count.data(), 1, MPI_INT, h.send_count.data(), 1, MPI_INT, comm);
    h.send_disp.assign(nprocs, 0);
    for (int r = 1; r < nprocs; ++r) h.send_disp[r] = h.send_disp[r - 1] + h.send_count[r - 1];
    const int total_send = h.send_disp[nprocs - 1] + h.send_count[nprocs - 1];

    std::vector<idx_t> wanted(total_send);
    MPI_Alltoallv(m.ghost_global.data(), h.recv_count.data(), h.recv_disp.data(), MPI_UINT64_T,
                  wanted.data(), h.send_count.data(), h.send_disp.data(), MPI_UINT64_T, comm);

    h.send_local.resize(total_send);
    for (int i = 0; i < total_send; ++i) {
        const idx_t g = wanted[i];
        if (g < m.lo || g >= m.lo + m.n_owned) fatal("halo request for an element this rank does not own");
        h.send_local[i] = m.local_of_owned[g - m.lo];
    }
    return h;
}

// ---------------------------------------------------------------------------
// Device-side storage
// ---------------------------------------------------------------------------

template <typename T>
struct DeviceDeleter {
    void operator()(T* p) const noexcept {
        if (p) cudaFree(p);
    }
};
template <typename T>
struct PinnedDeleter {
    void operator()(T* p) const noexcept {
        if (p) cudaFreeHost(p);
    }
};
template <typename T>
using DevPtr = std::unique_ptr<T, DeviceDeleter<T>>;
template <typename T>
using PinnedPtr = std::unique_ptr<T, PinnedDeleter<T>>;

template <typename T>
DevPtr<T> deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)));
    return DevPtr<T>(p);
}

template <typename T>
PinnedPtr<T> pinnedAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)));
    return PinnedPtr<T>(p);
}

template <typename T>
void uploadTo(DevPtr<T>& dst, const std::vector<T>& src) {
    if (!src.empty()) {
        CUDA_CHECK(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

__constant__ Material c_materials[MAX_MATERIALS];

// Gather the energies that neighbors need into a contiguous send buffer
__global__ void packKernel(const val_t* __restrict__ energy, const lidx_t* __restrict__ send_local,
                           val_t* __restrict__ send_buf, size_t count) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < count) send_buf[t] = energy[send_local[t]];
}

// One explicit update step for owned elements [begin, begin + count).
// Same arithmetic and neighbor order as the original computeFlux/runSimulation loop.
__global__ void updateKernel(const val_t* __restrict__ energy_in, val_t* __restrict__ energy_out,
                             val_t* __restrict__ total_flux, const uint8_t* __restrict__ material,
                             const uint8_t* __restrict__ num_conn, const lidx_t* __restrict__ nbr,
                             const val_t* __restrict__ flux, size_t n_owned, size_t begin, size_t count) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const size_t i = begin + t;

    const Material mat = c_materials[material[i]];
    const val_t self = energy_in[i];

    // Start with external flow, then add flux from all connected elements in order
    val_t total = mat.external_flow;
    const unsigned nc = num_conn[i];
    for (unsigned j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_owned + i;
        const val_t other = energy_in[nbr[slot]];
        total += (other - self) * mat.transfer_coeff * flux[slot] * 0.25;
    }

    energy_out[i] = self + total;
    total_flux[i] = total_flux[i] + fabs(total);
}

struct DeviceMesh {
    DevPtr<val_t> energy[2];
    DevPtr<val_t> total_flux;
    DevPtr<uint8_t> material;
    DevPtr<uint8_t> num_conn;
    DevPtr<lidx_t> nbr;
    DevPtr<val_t> flux;
    DevPtr<lidx_t> send_local;
    DevPtr<val_t> send_buf;
};

void launchUpdate(cudaStream_t stream, const DeviceMesh& d, const val_t* in, val_t* out,
                  size_t n_owned, size_t begin, size_t count) {
    if (count == 0) return;
    const unsigned blocks = static_cast<unsigned>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    updateKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(in, out, d.total_flux.get(), d.material.get(),
                                                           d.num_conn.get(), d.nbr.get(), d.flux.get(),
                                                           n_owned, begin, count);
    CUDA_CHECK(cudaGetLastError());
}

void selectDevice(MPI_Comm comm) {
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev == 0) fatal("no CUDA devices available");
    CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
}

// ---------------------------------------------------------------------------
// Validation and reporting (run on the root rank with the gathered state)
// ---------------------------------------------------------------------------

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
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    const bool root = (rank == 0);

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
        if (root) fprintf(stderr, "error: grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t N = n * n;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(N));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    const std::vector<Material> materials = buildMaterials();
    if (materials.size() > static_cast<size_t>(MAX_MATERIALS)) fatal("too many materials");

    selectDevice(comm);

    // Partition and build this rank's share of the mesh
    const std::vector<idx_t> starts = partitionStarts(N, nprocs);
    RankMesh mesh = buildRankMesh(n, starts, rank);
    HaloPlan halo = buildHaloPlan(mesh, starts, nprocs, comm);

    const size_t no = static_cast<size_t>(mesh.n_owned);
    const size_t n_ghost = static_cast<size_t>(mesh.n_ghost);
    const size_t n_send = halo.send_local.size();
    const size_t n_interior = static_cast<size_t>(mesh.n_interior);

    // Memory usage (global mesh accounting, as reported by the original benchmark)
    if (root) {
        const size_t static_mem = static_cast<size_t>(N) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(N) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Upload the rank-local mesh and allocate state on the device
    DeviceMesh dev;
    dev.energy[0] = deviceAlloc<val_t>(no + n_ghost);
    dev.energy[1] = deviceAlloc<val_t>(no + n_ghost);
    dev.total_flux = deviceAlloc<val_t>(no);
    dev.material = deviceAlloc<uint8_t>(no);
    dev.num_conn = deviceAlloc<uint8_t>(no);
    dev.nbr = deviceAlloc<lidx_t>(static_cast<size_t>(mesh.max_conn) * no);
    dev.flux = deviceAlloc<val_t>(static_cast<size_t>(mesh.max_conn) * no);
    dev.send_local = deviceAlloc<lidx_t>(n_send);
    dev.send_buf = deviceAlloc<val_t>(n_send);

    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, materials.data(), materials.size() * sizeof(Material)));
    uploadTo(dev.material, mesh.material);
    uploadTo(dev.num_conn, mesh.num_conn);
    uploadTo(dev.nbr, mesh.nbr);
    uploadTo(dev.flux, mesh.flux);
    uploadTo(dev.send_local, halo.send_local);
    // Initial state: zero energy and zero accumulated flux (including ghost slots)
    const size_t energy_slots = std::max<size_t>(no + n_ghost, 1);
    CUDA_CHECK(cudaMemset(dev.energy[0].get(), 0, energy_slots * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(dev.energy[1].get(), 0, energy_slots * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(dev.total_flux.get(), 0, std::max<size_t>(no, 1) * sizeof(val_t)));

    PinnedPtr<val_t> h_send = pinnedAlloc<val_t>(n_send);
    PinnedPtr<val_t> h_recv = pinnedAlloc<val_t>(n_ghost);

    cudaStream_t s_comm, s_interior;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_interior, cudaStreamNonBlocking));

    if (root) printf("Running simulation...\n");

    // -----------------------------------------------------------------------
    // Simulation loop
    // -----------------------------------------------------------------------
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    val_t* cur = dev.energy[0].get();
    val_t* nxt = dev.energy[1].get();
    const size_t n_owned = no;
    const size_t n_boundary_begin = n_interior;
    const size_t n_boundary = no - n_interior;

    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * static_cast<size_t>(nprocs));
    for (int iter = 0; iter < n_iters; ++iter) {
        reqs.clear();
        // Post receives for ghost energies first
        for (int r = 0; r < nprocs; ++r) {
            if (halo.recv_count[r] == 0) continue;
            MPI_Request q;
            MPI_Irecv(h_recv.get() + halo.recv_disp[r], halo.recv_count[r], MPI_DOUBLE, r,
                      HALO_TAG, comm, &q);
            reqs.push_back(q);
        }

        // Stage boundary energies to the host on the communication stream
        if (n_send > 0) {
            const unsigned blocks = static_cast<unsigned>((n_send + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            packKernel<<<blocks, THREADS_PER_BLOCK, 0, s_comm>>>(cur, dev.send_local.get(),
                                                                 dev.send_buf.get(), n_send);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_send.get(), dev.send_buf.get(), n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_comm));
        }

        // Interior elements need no ghost data: overlap their update with the exchange
        launchUpdate(s_interior, dev, cur, nxt, n_owned, 0, n_interior);

        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        for (int r = 0; r < nprocs; ++r) {
            if (halo.send_count[r] == 0) continue;
            MPI_Request q;
            MPI_Isend(h_send.get() + halo.send_disp[r], halo.send_count[r], MPI_DOUBLE, r,
                      HALO_TAG, comm, &q);
            reqs.push_back(q);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

        // Ghost slots follow the owned elements
        if (n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(cur + n_owned, h_recv.get(), n_ghost * sizeof(val_t),
                                       cudaMemcpyHostToDevice, s_comm));
        }
        // Boundary elements depend on ghost values
        launchUpdate(s_comm, dev, cur, nxt, n_owned, n_boundary_begin, n_boundary);

        CUDA_CHECK(cudaStreamSynchronize(s_interior));
        CUDA_CHECK(cudaStreamSynchronize(s_comm));

        std::swap(cur, nxt);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    const long long local_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long max_ns = 0;
    MPI_Reduce(&local_ns, &max_ns, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    const long long duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(max_ns)).count();

    // -----------------------------------------------------------------------
    // Pull the final state of owned elements back to the host (local order)
    // -----------------------------------------------------------------------
    std::vector<val_t> h_energy(no), h_flux(no);
    if (no > 0) {
        CUDA_CHECK(cudaMemcpy(h_energy.data(), cur, no * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), dev.total_flux.get(), no * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    // Result hash: per-element contributions are XOR-combined, so the
    // distributed reduction equals the serial fold over elements.
    uint64_t local_hash = 0;
#pragma omp parallel for reduction(^ : local_hash) schedule(static)
    for (std::int64_t l = 0; l < static_cast<std::int64_t>(no); ++l) {
        const uint64_t g = mesh.global_of_local[l];
        uint64_t e_bits = 0;
        uint64_t f_bits = 0;
        std::memcpy(&e_bits, &h_energy[l], sizeof(e_bits));
        std::memcpy(&f_bits, &h_flux[l], sizeof(f_bits));
        local_hash ^= (e_bits + g) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (f_bits + g) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    if (root) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(N)) /
            (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather the full state on the root for result printing / validation
    int valid = 1;
    if (printResults || validate) {
        std::vector<ElementDynamic> mine(no);
#pragma omp parallel for schedule(static)
        for (std::int64_t k = 0; k < static_cast<std::int64_t>(no); ++k) {
            const lidx_t l = mesh.local_of_owned[k];
            mine[k] = ElementDynamic{h_energy[l], h_flux[l]};
        }

        if (static_cast<idx_t>(no) * 2 > static_cast<idx_t>(std::numeric_limits<int>::max()) ||
            N * 2 > static_cast<idx_t>(std::numeric_limits<int>::max())) {
            fatal("state too large for -r/-v gathering; run without -r/-v at this size");
        }
        if (!root) {
            if (no > 0) {
                MPI_Send(mine.data(), static_cast<int>(no * 2), MPI_DOUBLE, 0, GATHER_TAG, comm);
            }
        } else {
            std::vector<ElementDynamic> all(static_cast<size_t>(N));
            std::copy(mine.begin(), mine.end(), all.begin() + static_cast<std::ptrdiff_t>(starts[0]));
            for (int r = 1; r < nprocs; ++r) {
                const idx_t cnt = starts[r + 1] - starts[r];
                if (cnt == 0) continue;
                MPI_Recv(reinterpret_cast<val_t*>(all.data() + starts[r]), static_cast<int>(cnt * 2),
                         MPI_DOUBLE, r, GATHER_TAG, comm, MPI_STATUS_IGNORE);
            }

            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all.size());
                for (const auto& elem : all) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) {
                valid = validateResults(all) ? 1 : 0;
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);

    CUDA_CHECK(cudaStreamDestroy(s_comm));
    CUDA_CHECK(cudaStreamDestroy(s_interior));
    MPI_Finalize();
    return valid ? 0 : 1;
}
