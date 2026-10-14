#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
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
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (logical mesh record size)
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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must be two packed doubles");

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr int CUDA_BLOCK = 256;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Contiguous block partition of the global element range [0, n) across ranks
idx_t blockStart(idx_t n, int nranks, int rank) {
    return n * static_cast<idx_t>(rank) / static_cast<idx_t>(nranks);
}

// Rank owning global element gid; starts holds nranks+1 block boundaries
int ownerOf(const std::vector<idx_t>& starts, idx_t gid) {
    return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), gid) - starts.begin()) - 1;
}

// Neighbors of element gid in the 2D square mesh (idx = x * n + y), in the
// original generation order (+x, -x, +y, -y). Returns the number of neighbors.
int buildNeighbors(idx_t gid, int n_root, idx_t* nbr) {
    const long long n = n_root;
    const long long x = static_cast<long long>(gid) / n;
    const long long y = static_cast<long long>(gid) % n;
    const long long offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    int count = 0;
    for (int k = 0; k < 4; ++k) {
        const long long nx = x + offsets[k][0];
        const long long ny = y + offsets[k][1];
        if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
            nbr[count++] = static_cast<idx_t>(nx * n + ny);
        }
    }
    return count;
}

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// One explicit update of the listed elements. Connection data is stored
// slot-major with the owned-element count as pitch, so slot j of element i is at
// j * pitch + i. Neighbors may be ghost elements (index >= pitch).
__global__ void stepKernel(const int* __restrict__ list, int count, int pitch,
                           const int* __restrict__ conn_idx, const val_t* __restrict__ conn_flux,
                           const int* __restrict__ nconn, const int* __restrict__ mat_idx,
                           const Material* __restrict__ materials,
                           const val_t* __restrict__ cur_e, const val_t* __restrict__ cur_f,
                           val_t* __restrict__ nxt_e, val_t* __restrict__ nxt_f) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    const int i = list[k];
    const Material mat = materials[mat_idx[i]];
    const val_t self_energy = cur_e[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const int n_conn = nconn[i];
    for (int j = 0; j < n_conn; ++j) {
        const int neighbor = conn_idx[j * pitch + i];
        total_flux += computeFlux(mat, self_energy, conn_flux[j * pitch + i], cur_e[neighbor]);
    }

    nxt_e[i] = self_energy + total_flux;
    nxt_f[i] = cur_f[i] + fabs(total_flux);
}

// Gather the energies other ranks need from this rank into a contiguous buffer
__global__ void packKernel(const int* __restrict__ idx, int count,
                           const val_t* __restrict__ src, val_t* __restrict__ dst) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) dst[k] = src[idx[k]];
}

template <typename T>
T* deviceAlloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

template <typename T>
T* pinnedAlloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

template <typename T>
void upload(T* dst, const std::vector<T>& src) {
    if (!src.empty()) CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// Each MPI rank drives one GPU; ranks on the same node are spread across devices
void selectDevice() {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
}

// One rank's share of the mesh and simulation state, resident on its GPU.
// Owned elements are the contiguous global range [lo, lo + n_owned). Ghost
// elements are remote neighbors whose energies are received every iteration.
class Simulation {
  public:
    Simulation(int n_root, MPI_Comm comm);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void run(int n_iters);
    std::vector<ElementDynamic> downloadOwned() const;
    idx_t lo() const { return lo_; }

  private:
    // Message exchanged with one neighboring rank; offset is into the send or receive buffer
    struct Link {
        int peer;
        int offset;
        int count;
    };

    void launchStep(const int* list, int count, const val_t* cur_e, const val_t* cur_f,
                    val_t* nxt_e, val_t* nxt_f);

    MPI_Comm comm_;
    int rank_ = 0;
    idx_t lo_ = 0;
    int n_owned_ = 0;
    int n_ghost_ = 0;
    int n_interior_ = 0;
    int n_boundary_ = 0;
    int n_send_ = 0;
    int cur_ = 0;

    std::vector<Link> sends_;
    std::vector<Link> recvs_;

    // Device state; energies carry ghost entries after the owned ones
    val_t* d_e_[2] = {nullptr, nullptr};
    val_t* d_f_[2] = {nullptr, nullptr};
    int* d_mat_ = nullptr;
    int* d_nconn_ = nullptr;
    int* d_conn_idx_ = nullptr;
    val_t* d_conn_flux_ = nullptr;
    Material* d_materials_ = nullptr;
    int* d_interior_ = nullptr;
    int* d_boundary_ = nullptr;
    int* d_send_idx_ = nullptr;
    val_t* d_send_ = nullptr;

    // Pinned host staging buffers; receive buffer is double-buffered by iteration parity
    val_t* h_send_ = nullptr;
    val_t* h_recv_[2] = {nullptr, nullptr};

    cudaStream_t s_comp_ = nullptr;
    cudaStream_t s_copy_ = nullptr;
    cudaEvent_t ev_comp_ = nullptr;
    cudaEvent_t ev_h2d_ = nullptr;
};

Simulation::Simulation(int n_root, MPI_Comm comm) : comm_(comm) {
    int nranks = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nranks);

    const idx_t n_elems = static_cast<idx_t>(n_root) * static_cast<idx_t>(n_root);
    std::vector<idx_t> starts(nranks + 1);
    for (int r = 0; r <= nranks; ++r) starts[r] = blockStart(n_elems, nranks, r);
    lo_ = starts[rank_];
    const idx_t hi = starts[rank_ + 1];
    const idx_t n_owned = hi - lo_;

    // Connectivity of the owned elements, with global neighbor ids
    std::vector<idx_t> nbr_gid(n_owned * MAX_CONNECTIONS);
    std::vector<val_t> nbr_flux(n_owned * MAX_CONNECTIONS);
    std::vector<int> nbr_cnt(n_owned);
    std::vector<int> mat(n_owned);

    const idx_t last = static_cast<idx_t>(n_root) - 1;
    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_owned; ++i) {
        const idx_t gid = lo_ + i;

        // Corner elements as inflow/outflow, assigned in the original order
        idx_t m = DEFAULT_MAT_ID;
        if (gid == 0) m = INFLOW_MAT_ID;
        if (gid == last) m = OUTFLOW_MAT_ID;
        if (gid == last * n_root) m = OUTFLOW_MAT_ID;
        if (gid == last * n_root + last) m = INFLOW_MAT_ID;
        mat[i] = static_cast<int>(m);

        nbr_cnt[i] = buildNeighbors(gid, n_root, &nbr_gid[i * MAX_CONNECTIONS]);
        for (int j = 0; j < nbr_cnt[i]; ++j) nbr_flux[i * MAX_CONNECTIONS + j] = 1.0;
    }

    // Ghost elements: remote neighbors, sorted so they are grouped by owner rank
    std::vector<idx_t> ghosts;
    for (idx_t i = 0; i < n_owned; ++i) {
        for (int j = 0; j < nbr_cnt[i]; ++j) {
            const idx_t nb = nbr_gid[i * MAX_CONNECTIONS + j];
            if (nb < lo_ || nb >= hi) ghosts.push_back(nb);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());

    int stride = 0;
    for (idx_t i = 0; i < n_owned; ++i) stride = std::max(stride, nbr_cnt[i]);

    // Exchange requests: tell each owner which of its elements we need
    std::vector<int> need(nranks, 0);
    std::vector<int> give(nranks, 0);
    for (idx_t g : ghosts) need[ownerOf(starts, g)]++;
    MPI_Alltoall(need.data(), 1, MPI_INT, give.data(), 1, MPI_INT, comm_);

    std::vector<int> need_off(nranks + 1, 0);
    std::vector<int> give_off(nranks + 1, 0);
    for (int r = 0; r < nranks; ++r) {
        need_off[r + 1] = need_off[r] + need[r];
        give_off[r + 1] = give_off[r] + give[r];
    }

    std::vector<idx_t> requested(give_off[nranks]);
    MPI_Alltoallv(ghosts.data(), need.data(), need_off.data(), MPI_UINT64_T,
                  requested.data(), give.data(), give_off.data(), MPI_UINT64_T, comm_);

    // Send list: local indices of the requested elements, grouped by requesting rank
    std::vector<int> send_idx(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) send_idx[k] = static_cast<int>(requested[k] - lo_);

    for (int r = 0; r < nranks; ++r) {
        if (give[r] > 0) sends_.push_back({r, give_off[r], give[r]});
        if (need[r] > 0) recvs_.push_back({r, need_off[r], need[r]});
    }

    n_owned_ = static_cast<int>(n_owned);
    n_ghost_ = static_cast<int>(ghosts.size());
    n_send_ = give_off[nranks];
    if (static_cast<idx_t>(n_owned_) + n_ghost_ > INT_MAX) {
        fprintf(stderr, "Too many elements per rank for 32-bit local indexing\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Split owned elements into interior (no ghost neighbors) and boundary.
    // Interior elements can be updated while the halo is in flight.
    std::vector<int> h_nconn(n_owned);
    std::vector<int> h_conn_idx(n_owned * stride);
    std::vector<val_t> h_conn_flux(n_owned * stride);
    std::vector<unsigned char> is_boundary(n_owned, 0);

    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_owned; ++i) {
        h_nconn[i] = nbr_cnt[i];
        unsigned char bnd = 0;
        for (int j = 0; j < nbr_cnt[i]; ++j) {
            const idx_t nb = nbr_gid[i * MAX_CONNECTIONS + j];
            idx_t local;
            if (nb >= lo_ && nb < hi) {
                local = nb - lo_;
            } else {
                bnd = 1;
                local = n_owned + static_cast<idx_t>(std::lower_bound(ghosts.begin(), ghosts.end(), nb) - ghosts.begin());
            }
            h_conn_idx[j * n_owned + i] = static_cast<int>(local);
            h_conn_flux[j * n_owned + i] = nbr_flux[i * MAX_CONNECTIONS + j];
        }
        is_boundary[i] = bnd;
    }

    std::vector<int> interior;
    std::vector<int> boundary;
    for (idx_t i = 0; i < n_owned; ++i) {
        (is_boundary[i] ? boundary : interior).push_back(static_cast<int>(i));
    }
    n_interior_ = static_cast<int>(interior.size());
    n_boundary_ = static_cast<int>(boundary.size());

    // Upload to the GPU
    const std::vector<Material> materials = {
        {0.8, 0.0},   // Default material
        {0.8, 0.5},   // Inflow material
        {0.8, -0.5},  // Outflow material
    };

    d_mat_ = deviceAlloc<int>(n_owned);
    upload(d_mat_, mat);
    d_nconn_ = deviceAlloc<int>(n_owned);
    upload(d_nconn_, h_nconn);
    d_conn_idx_ = deviceAlloc<int>(n_owned * stride);
    upload(d_conn_idx_, h_conn_idx);
    d_conn_flux_ = deviceAlloc<val_t>(n_owned * stride);
    upload(d_conn_flux_, h_conn_flux);
    d_materials_ = deviceAlloc<Material>(materials.size());
    upload(d_materials_, materials);
    d_interior_ = deviceAlloc<int>(interior.size());
    upload(d_interior_, interior);
    d_boundary_ = deviceAlloc<int>(boundary.size());
    upload(d_boundary_, boundary);

    d_send_idx_ = deviceAlloc<int>(n_send_);
    upload(d_send_idx_, send_idx);
    d_send_ = deviceAlloc<val_t>(n_send_);
    h_send_ = pinnedAlloc<val_t>(n_send_);
    h_recv_[0] = pinnedAlloc<val_t>(n_ghost_);
    h_recv_[1] = pinnedAlloc<val_t>(n_ghost_);

    for (int b = 0; b < 2; ++b) {
        d_e_[b] = deviceAlloc<val_t>(n_owned_ + n_ghost_);
        d_f_[b] = deviceAlloc<val_t>(n_owned_);
        CUDA_CHECK(cudaMemset(d_e_[b], 0, std::max<size_t>(n_owned_ + n_ghost_, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d_f_[b], 0, std::max<size_t>(n_owned_, 1) * sizeof(val_t)));
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comp_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_copy_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_comp_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d_, cudaEventDisableTiming));
}

Simulation::~Simulation() {
    for (int b = 0; b < 2; ++b) {
        cudaFree(d_e_[b]);
        cudaFree(d_f_[b]);
        cudaFreeHost(h_recv_[b]);
    }
    cudaFree(d_mat_);
    cudaFree(d_nconn_);
    cudaFree(d_conn_idx_);
    cudaFree(d_conn_flux_);
    cudaFree(d_materials_);
    cudaFree(d_interior_);
    cudaFree(d_boundary_);
    cudaFree(d_send_idx_);
    cudaFree(d_send_);
    cudaFreeHost(h_send_);
    cudaEventDestroy(ev_comp_);
    cudaEventDestroy(ev_h2d_);
    cudaStreamDestroy(s_comp_);
    cudaStreamDestroy(s_copy_);
}

void Simulation::launchStep(const int* list, int count, const val_t* cur_e, const val_t* cur_f,
                            val_t* nxt_e, val_t* nxt_f) {
    if (count == 0) return;
    const unsigned blocks = (count + CUDA_BLOCK - 1) / CUDA_BLOCK;
    stepKernel<<<blocks, CUDA_BLOCK, 0, s_comp_>>>(list, count, n_owned_, d_conn_idx_, d_conn_flux_,
                                                   d_nconn_, d_mat_, d_materials_,
                                                   cur_e, cur_f, nxt_e, nxt_f);
    CUDA_CHECK(cudaGetLastError());
}

void Simulation::run(int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        const val_t* cur_e = d_e_[cur_];
        const val_t* cur_f = d_f_[cur_];
        val_t* nxt_e = d_e_[1 - cur_];
        val_t* nxt_f = d_f_[1 - cur_];
        val_t* ghost_dst = d_e_[cur_] + n_owned_;
        val_t* recv_host = h_recv_[iter & 1];

        // Post receives first so incoming halo data never waits for a matching call
        std::vector<MPI_Request> reqs;
        reqs.reserve(sends_.size() + recvs_.size());
        for (const Link& l : recvs_) {
            MPI_Request req;
            MPI_Irecv(recv_host + l.offset, l.count, MPI_DOUBLE, l.peer, 0, comm_, &req);
            reqs.push_back(req);
        }

        // Pack boundary energies of the current state and copy them to the host
        CUDA_CHECK(cudaStreamWaitEvent(s_copy_, ev_comp_, 0));
        if (n_send_ > 0) {
            const unsigned blocks = (n_send_ + CUDA_BLOCK - 1) / CUDA_BLOCK;
            packKernel<<<blocks, CUDA_BLOCK, 0, s_copy_>>>(d_send_idx_, n_send_, cur_e, d_send_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_send_, d_send_, n_send_ * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_copy_));
        }

        // Interior elements need no ghost data: update them while the halo travels
        launchStep(d_interior_, n_interior_, cur_e, cur_f, nxt_e, nxt_f);

        CUDA_CHECK(cudaStreamSynchronize(s_copy_));
        for (const Link& l : sends_) {
            MPI_Request req;
            MPI_Isend(h_send_ + l.offset, l.count, MPI_DOUBLE, l.peer, 0, comm_, &req);
            reqs.push_back(req);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

        // Ghost layout is grouped by source rank, matching the receive buffer
        if (n_ghost_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(ghost_dst, recv_host, n_ghost_ * sizeof(val_t),
                                       cudaMemcpyHostToDevice, s_copy_));
        }
        CUDA_CHECK(cudaEventRecord(ev_h2d_, s_copy_));
        CUDA_CHECK(cudaStreamWaitEvent(s_comp_, ev_h2d_, 0));

        // Boundary elements depend on ghost energies of the current iteration
        launchStep(d_boundary_, n_boundary_, cur_e, cur_f, nxt_e, nxt_f);
        CUDA_CHECK(cudaEventRecord(ev_comp_, s_comp_));

        cur_ = 1 - cur_;
    }
    CUDA_CHECK(cudaStreamSynchronize(s_comp_));
    CUDA_CHECK(cudaStreamSynchronize(s_copy_));
}

std::vector<ElementDynamic> Simulation::downloadOwned() const {
    std::vector<val_t> energy(n_owned_);
    std::vector<val_t> flux(n_owned_);
    if (n_owned_ > 0) {
        CUDA_CHECK(cudaMemcpy(energy.data(), d_e_[cur_], n_owned_ * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flux.data(), d_f_[cur_], n_owned_ * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    std::vector<ElementDynamic> out(n_owned_);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_owned_; ++i) {
        out[i] = {energy[i], flux[i]};
    }
    return out;
}

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

// XOR-combinable hash of a contiguous block of elements starting at global index offset.
// XOR is associative and commutative, so per-rank partial hashes combine to the global hash.
uint64_t computeHash(const ElementDynamic* elements, size_t n, idx_t offset) {
    uint64_t digest = 0;
    #pragma omp parallel for schedule(static) reduction(^:digest)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        const idx_t gi = offset + static_cast<idx_t>(i);
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        digest ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
        digest ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    return digest;
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

int runBenchmark(int n_elems_root, int n_iters, bool validate, bool printResults) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = rank == 0;

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    selectDevice();
    Simulation sim(n_elems_root, MPI_COMM_WORLD);

    // Calculate memory usage
    if (root) {
        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation; the reported time is the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    sim.run(n_iters);
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(max_elapsed * 1000.0);

    // Compute hash for verification (combined across ranks)
    const std::vector<ElementDynamic> owned = sim.downloadOwned();
    const uint64_t local_digest = computeHash(owned.data(), owned.size(), sim.lo());
    uint64_t digest = 0;
    MPI_Allreduce(&local_digest, &digest, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);

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
        printf("  Result hash: %016lX\n", digest);
        printf("\n");
    }

    // Gather the full state on rank 0 for external output and validation
    if (printResults || validate) {
        std::vector<int> counts(nranks);
        std::vector<int> displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const idx_t lo = blockStart(n_elems, nranks, r);
            const idx_t hi = blockStart(n_elems, nranks, r + 1);
            counts[r] = static_cast<int>(2 * (hi - lo));
            displs[r] = static_cast<int>(2 * lo);
        }

        std::vector<ElementDynamic> all(root ? n_elems : 0);
        MPI_Gatherv(owned.data(), static_cast<int>(2 * owned.size()), MPI_DOUBLE,
                    all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root && printResults) {
            std::vector<double> energyData;
            energyData.reserve(all.size());
            for (const auto& elem : all) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            int valid = root ? validateResults(all) : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!valid) return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int status = 0;

    // Parse command line arguments
    bool exit_early = false;
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
            exit_early = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            exit_early = true;
            break;
        }
    }

    if (!exit_early && n_elems_root < 1) {
        if (rank == 0) {
            printf("Grid size must be at least 1\n");
            printUsage(argv[0]);
        }
        status = 1;
        exit_early = true;
    }

    if (!exit_early) {
        status = runBenchmark(n_elems_root, n_iters, validate, printResults);
    }

    MPI_Finalize();
    return status;
}
