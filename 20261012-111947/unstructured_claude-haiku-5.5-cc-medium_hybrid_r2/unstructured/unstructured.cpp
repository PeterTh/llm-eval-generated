#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
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
using idx_t = uint64_t;    // global element index
using local_t = uint32_t;  // rank-local element index (owned, then ghost)
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

// Material table indexed by material ID (uploaded to the device)
constexpr Material MATERIALS[3] = {
    {0.8, 0.0},   // Default material
    {0.8, 0.5},   // Inflow material
    {0.8, -0.5},  // Outflow material
};

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Build the static description of global element g of an n x n square 2D grid
// This represents computation on arbitrarily-shaped geometries
ElementStatic buildElementStatic(idx_t g, int n_elems_root) {
    ElementStatic elem{};
    const long long root = n_elems_root;
    const long long last = root - 1;
    const long long x = static_cast<long long>(g) / root;
    const long long y = static_cast<long long>(g) % root;

    // Set corner elements as inflow/outflow to create interesting dynamics
    elem.material_idx = DEFAULT_MAT_ID;
    if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
    if (x == 0 && y == last) elem.material_idx = OUTFLOW_MAT_ID;
    if (x == last && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
    if (x == last && y == last) elem.material_idx = INFLOW_MAT_ID;

    // Connect to neighbors (up, down, left, right) that are within bounds
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int n = 0; n < 4; ++n) {
        const long long nx = x + offsets[n][0];
        const long long ny = y + offsets[n][1];

        if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
            elem.connected_idx[elem.num_connections] = static_cast<idx_t>(nx * root + ny);
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
    }
    return elem;
}

// Compute energy flux between two elements (device version used by fluxKernel)
__device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                    val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Update elements [first, first + count) for one iteration. Reads energy_in for owned
// and ghost entries, writes energy_out for owned entries, and accumulates total_flux
// in place. Each element is handled by one thread, in the same summation order as the
// sequential reference: external flow first, then neighbors in connection order.
__global__ void fluxKernel(local_t first, local_t count, local_t n_owned,
                           const Material* __restrict__ materials,
                           const local_t* __restrict__ mat_idx,
                           const local_t* __restrict__ num_conn,
                           const local_t* __restrict__ conn,
                           const val_t* __restrict__ conn_flux,
                           const val_t* __restrict__ energy_in,
                           val_t* __restrict__ energy_out,
                           val_t* __restrict__ total_flux) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;

    const local_t i = first + static_cast<local_t>(t);
    const Material mat = materials[mat_idx[i]];
    const val_t self = energy_in[i];

    // Start with external flow
    val_t flux = mat.external_flow;

    // Add flux from all connected elements (slot-major layout: coalesced reads)
    const local_t n_conn = num_conn[i];
    for (local_t j = 0; j < n_conn; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_owned + i;
        flux += computeFlux(mat, self, conn_flux[slot], energy_in[conn[slot]]);
    }

    // Update element state
    energy_out[i] = self + flux;
    total_flux[i] += fabs(flux);
}

// Gather the values other ranks need into a contiguous send buffer
__global__ void packKernel(size_t count, const local_t* __restrict__ send_idx,
                           const val_t* __restrict__ energy, val_t* __restrict__ send_buf) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) return;
    send_buf[t] = energy[send_idx[t]];
}

// Contiguous block partition of the global element range over the MPI ranks.
// The first (n % nranks) ranks receive one extra element.
std::vector<idx_t> partitionBegins(idx_t n_total, int nranks) {
    std::vector<idx_t> begins(nranks + 1);
    const idx_t base = n_total / nranks;
    const idx_t extra = n_total % nranks;

    for (int r = 0; r <= nranks; ++r) {
        begins[r] = static_cast<idx_t>(r) * base + std::min<idx_t>(r, extra);
    }
    return begins;
}

// Rank that owns global element g
int ownerOf(const std::vector<idx_t>& begins, idx_t g) {
    return static_cast<int>(std::upper_bound(begins.begin(), begins.end(), g) - begins.begin()) - 1;
}

// Exclusive prefix sum of per-rank counts
std::vector<int> exclusiveScan(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size());
    int acc = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        displs[i] = acc;
        acc += counts[i];
    }
    return displs;
}

// The part of the mesh owned by one rank, in the layout used on the device
struct LocalMesh {
    idx_t gbegin = 0;                        // first owned global element
    local_t n_owned = 0;                     // owned elements: local ids [0, n_owned)
    local_t n_ghost = 0;                     // ghost copies of remote neighbors: [n_owned, n_owned + n_ghost)
    local_t n_sent = 0;                      // owned elements read by other ranks: local ids [0, n_sent)
    local_t slots = 1;                       // connection slots per element
    std::vector<local_t> natural_of_local;   // global-order position of each local element
    std::vector<local_t> mat_idx;
    std::vector<local_t> num_conn;
    std::vector<local_t> conn;               // slot-major: conn[slot * n_owned + i], local ids
    std::vector<val_t> conn_flux;            // same layout as conn
    std::vector<idx_t> ghost_gid;            // sorted global ids of ghost elements
    std::vector<int> recv_count, recv_displ; // ghost values received from each owner rank
    std::vector<int> send_count, send_displ; // owned values sent to each requesting rank
    std::vector<local_t> send_idx;           // local ids to send, grouped by destination rank
};

// Build this rank's share of the mesh. Each rank only materializes its own block
// plus the ghost list, so memory per rank scales with the block, not the full grid.
LocalMesh buildLocalMesh(int n_elems_root, const std::vector<idx_t>& begins,
                         int rank, int nranks, MPI_Comm comm) {
    LocalMesh mesh;
    mesh.gbegin = begins[rank];
    const idx_t gend = begins[rank + 1];
    const idx_t n_owned = gend - mesh.gbegin;
    mesh.n_owned = static_cast<local_t>(n_owned);

    // Static description of the owned elements, in global order
    std::vector<ElementStatic> natural(n_owned);
    #pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < n_owned; ++k) {
        natural[k] = buildElementStatic(mesh.gbegin + k, n_elems_root);
    }

    // Remote neighbors referenced by owned elements become ghost elements
    for (idx_t k = 0; k < n_owned; ++k) {
        for (idx_t j = 0; j < natural[k].num_connections; ++j) {
            const idx_t g = natural[k].connected_idx[j];
            if (g < mesh.gbegin || g >= gend) mesh.ghost_gid.push_back(g);
        }
    }
    std::sort(mesh.ghost_gid.begin(), mesh.ghost_gid.end());
    mesh.ghost_gid.erase(std::unique(mesh.ghost_gid.begin(), mesh.ghost_gid.end()),
                         mesh.ghost_gid.end());
    mesh.n_ghost = static_cast<local_t>(mesh.ghost_gid.size());

    // Sorted ghosts are grouped by owner, so each owner's block is contiguous
    mesh.recv_count.assign(nranks, 0);
    for (const idx_t g : mesh.ghost_gid) {
        mesh.recv_count[ownerOf(begins, g)]++;
    }
    mesh.recv_displ = exclusiveScan(mesh.recv_count);

    // Tell each owner which of its elements this rank needs
    std::vector<int> req_count(nranks);
    MPI_Alltoall(mesh.recv_count.data(), 1, MPI_INT, req_count.data(), 1, MPI_INT, comm);
    const std::vector<int> req_displ = exclusiveScan(req_count);
    const int n_req = req_displ.back() + req_count.back();

    std::vector<idx_t> req_gid(n_req);
    MPI_Alltoallv(mesh.ghost_gid.data(), mesh.recv_count.data(), mesh.recv_displ.data(), MPI_UINT64_T,
                  req_gid.data(), req_count.data(), req_displ.data(), MPI_UINT64_T, comm);

    // Owned elements that other ranks read go first in local order, so they can be
    // computed and sent while the remaining elements are still being updated
    std::vector<uint8_t> needed(n_owned, 0);
    for (const idx_t g : req_gid) {
        needed[g - mesh.gbegin] = 1;
    }

    std::vector<local_t> local_of_natural(n_owned);
    mesh.natural_of_local.resize(n_owned);
    local_t next = 0;
    for (idx_t k = 0; k < n_owned; ++k) {
        if (needed[k]) {
            local_of_natural[k] = next;
            mesh.natural_of_local[next++] = static_cast<local_t>(k);
        }
    }
    mesh.n_sent = next;
    for (idx_t k = 0; k < n_owned; ++k) {
        if (!needed[k]) {
            local_of_natural[k] = next;
            mesh.natural_of_local[next++] = static_cast<local_t>(k);
        }
    }

    // Send lists keep the requesters' ascending global order, which matches how the
    // requester places received ghosts
    mesh.send_count = req_count;
    mesh.send_displ = req_displ;
    mesh.send_idx.resize(n_req);
    for (int t = 0; t < n_req; ++t) {
        mesh.send_idx[t] = local_of_natural[req_gid[t] - mesh.gbegin];
    }

    // Local connectivity, with ghost references mapped past the owned block
    local_t slots = 1;
    for (idx_t k = 0; k < n_owned; ++k) {
        slots = std::max<local_t>(slots, static_cast<local_t>(natural[k].num_connections));
    }
    mesh.slots = slots;
    mesh.mat_idx.resize(n_owned);
    mesh.num_conn.resize(n_owned);
    mesh.conn.resize(static_cast<size_t>(slots) * n_owned);
    mesh.conn_flux.resize(static_cast<size_t>(slots) * n_owned);

    #pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n_owned; ++i) {
        const ElementStatic& elem = natural[mesh.natural_of_local[i]];
        mesh.mat_idx[i] = static_cast<local_t>(elem.material_idx);
        mesh.num_conn[i] = static_cast<local_t>(elem.num_connections);

        for (local_t j = 0; j < slots; ++j) {
            const size_t slot = static_cast<size_t>(j) * n_owned + i;
            mesh.conn[slot] = static_cast<local_t>(i);  // padding, never read
            mesh.conn_flux[slot] = 0.0;

            if (j < elem.num_connections) {
                const idx_t g = elem.connected_idx[j];
                if (g >= mesh.gbegin && g < gend) {
                    mesh.conn[slot] = local_of_natural[g - mesh.gbegin];
                } else {
                    const auto pos = std::lower_bound(mesh.ghost_gid.begin(), mesh.ghost_gid.end(), g);
                    mesh.conn[slot] = static_cast<local_t>(n_owned + (pos - mesh.ghost_gid.begin()));
                }
                mesh.conn_flux[slot] = elem.connected_flux[j];
            }
        }
    }
    return mesh;
}

// Device-resident state of this rank's block and its halo exchange
class Simulation {
  public:
    Simulation(const LocalMesh& mesh, MPI_Comm comm);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void run(int n_iters);

    // Current state of the owned elements, in global order
    void download(std::vector<ElementDynamic>& out) const;

  private:
    void step();
    void launchFlux(cudaStream_t stream, local_t first, local_t count,
                    const val_t* energy_in, val_t* energy_out);

    const LocalMesh& mesh_;
    MPI_Comm comm_;
    bool exchange_;                  // true if this rank sends or receives halo data
    std::vector<int> recv_ranks_;    // owners we receive ghosts from
    std::vector<int> send_ranks_;    // ranks we send owned elements to
    int cur_ = 0;                    // energy_[cur_] holds the current state

    Material* materials_ = nullptr;
    local_t* mat_idx_ = nullptr;
    local_t* num_conn_ = nullptr;
    local_t* conn_ = nullptr;
    val_t* conn_flux_ = nullptr;
    val_t* energy_[2] = {nullptr, nullptr};  // owned + ghost entries, double-buffered
    val_t* total_flux_ = nullptr;            // owned entries, updated in place
    local_t* send_idx_ = nullptr;
    val_t* send_buf_ = nullptr;              // device staging for halo sends
    val_t* host_send_ = nullptr;             // pinned host staging for MPI
    val_t* host_recv_ = nullptr;             // pinned host staging for MPI

    cudaStream_t send_stream_;   // computes elements others read, then packs and sends them
    cudaStream_t inner_stream_;  // computes the rest, overlapping communication
};

constexpr unsigned THREADS_PER_BLOCK = 256;

template <typename T>
T* deviceAlloc(size_t count) {
    const size_t bytes = std::max<size_t>(count, 1) * sizeof(T);
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    CUDA_CHECK(cudaMemset(ptr, 0, bytes));  // initial state is all zeros
    return ptr;
}

template <typename T>
void upload(T* dst, const std::vector<T>& src) {
    if (!src.empty()) {
        CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

Simulation::Simulation(const LocalMesh& mesh, MPI_Comm comm)
    : mesh_(mesh), comm_(comm),
      exchange_(mesh.n_ghost > 0 || !mesh.send_idx.empty()) {
    const size_t n_owned = mesh.n_owned;
    const size_t n_total = n_owned + mesh.n_ghost;
    const size_t n_conn = n_owned * mesh.slots;

    for (int r = 0; r < static_cast<int>(mesh.recv_count.size()); ++r) {
        if (mesh.recv_count[r] > 0) recv_ranks_.push_back(r);
        if (mesh.send_count[r] > 0) send_ranks_.push_back(r);
    }

    materials_ = deviceAlloc<Material>(3);
    CUDA_CHECK(cudaMemcpy(materials_, MATERIALS, sizeof(MATERIALS), cudaMemcpyHostToDevice));

    mat_idx_ = deviceAlloc<local_t>(n_owned);
    upload(mat_idx_, mesh.mat_idx);
    num_conn_ = deviceAlloc<local_t>(n_owned);
    upload(num_conn_, mesh.num_conn);
    conn_ = deviceAlloc<local_t>(n_conn);
    upload(conn_, mesh.conn);
    conn_flux_ = deviceAlloc<val_t>(n_conn);
    upload(conn_flux_, mesh.conn_flux);

    energy_[0] = deviceAlloc<val_t>(n_total);
    energy_[1] = deviceAlloc<val_t>(n_total);
    total_flux_ = deviceAlloc<val_t>(n_owned);

    send_idx_ = deviceAlloc<local_t>(mesh.send_idx.size());
    upload(send_idx_, mesh.send_idx);
    send_buf_ = deviceAlloc<val_t>(mesh.send_idx.size());

    const size_t n_send = std::max<size_t>(mesh.send_idx.size(), 1);
    const size_t n_recv = std::max<size_t>(mesh.n_ghost, 1);
    CUDA_CHECK(cudaMallocHost(&host_send_, n_send * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&host_recv_, n_recv * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreate(&send_stream_));
    CUDA_CHECK(cudaStreamCreate(&inner_stream_));
}

Simulation::~Simulation() {
    cudaStreamDestroy(send_stream_);
    cudaStreamDestroy(inner_stream_);
    cudaFreeHost(host_recv_);
    cudaFreeHost(host_send_);
    cudaFree(send_buf_);
    cudaFree(send_idx_);
    cudaFree(total_flux_);
    cudaFree(energy_[1]);
    cudaFree(energy_[0]);
    cudaFree(conn_flux_);
    cudaFree(conn_);
    cudaFree(num_conn_);
    cudaFree(mat_idx_);
    cudaFree(materials_);
}

void Simulation::launchFlux(cudaStream_t stream, local_t first, local_t count,
                            const val_t* energy_in, val_t* energy_out) {
    if (count == 0) return;
    const unsigned blocks = static_cast<unsigned>((static_cast<size_t>(count) + THREADS_PER_BLOCK - 1)
                                                  / THREADS_PER_BLOCK);
    fluxKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(first, count, mesh_.n_owned,
                                                         materials_, mat_idx_, num_conn_,
                                                         conn_, conn_flux_,
                                                         energy_in, energy_out, total_flux_);
    CUDA_CHECK(cudaGetLastError());
}

// One iteration: read the current state, write the next one, then swap buffers.
// Ghost entries of the next state are refreshed from the owning ranks before the swap.
void Simulation::step() {
    const val_t* energy_in = energy_[cur_];
    val_t* energy_out = energy_[1 - cur_];
    const local_t n_owned = mesh_.n_owned;

    if (!exchange_) {
        launchFlux(send_stream_, 0, n_owned, energy_in, energy_out);
    } else {
        // Post receives first so incoming halo values can land during the update
        std::vector<MPI_Request> requests(recv_ranks_.size() + send_ranks_.size());
        size_t n_req = 0;
        for (const int r : recv_ranks_) {
            MPI_Irecv(host_recv_ + mesh_.recv_displ[r], mesh_.recv_count[r], MPI_DOUBLE,
                      r, 0, comm_, &requests[n_req++]);
        }

        // Elements other ranks read are computed and packed first
        launchFlux(send_stream_, 0, mesh_.n_sent, energy_in, energy_out);
        const size_t n_send = mesh_.send_idx.size();
        if (n_send > 0) {
            const unsigned blocks = static_cast<unsigned>((n_send + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            packKernel<<<blocks, THREADS_PER_BLOCK, 0, send_stream_>>>(n_send, send_idx_, energy_out, send_buf_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(host_send_, send_buf_, n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, send_stream_));
        }

        // The remaining elements overlap with the halo communication
        launchFlux(inner_stream_, mesh_.n_sent, n_owned - mesh_.n_sent, energy_in, energy_out);

        CUDA_CHECK(cudaStreamSynchronize(send_stream_));
        for (const int r : send_ranks_) {
            MPI_Isend(host_send_ + mesh_.send_displ[r], mesh_.send_count[r], MPI_DOUBLE,
                      r, 0, comm_, &requests[n_req++]);
        }
        MPI_Waitall(static_cast<int>(n_req), requests.data(), MPI_STATUSES_IGNORE);

        // Ghosts are contiguous in owner order, matching the receive layout
        if (mesh_.n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(energy_out + n_owned, host_recv_,
                                       static_cast<size_t>(mesh_.n_ghost) * sizeof(val_t),
                                       cudaMemcpyHostToDevice, send_stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(send_stream_));
        CUDA_CHECK(cudaStreamSynchronize(inner_stream_));
    }

    cur_ = 1 - cur_;
}

void Simulation::run(int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Simulation::download(std::vector<ElementDynamic>& out) const {
    const size_t n_owned = mesh_.n_owned;
    std::vector<val_t> energy(n_owned);
    std::vector<val_t> flux(n_owned);
    CUDA_CHECK(cudaMemcpy(energy.data(), energy_[cur_], n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), total_flux_, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));

    out.resize(n_owned);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_owned; ++i) {
        out[mesh_.natural_of_local[i]] = ElementDynamic{energy[i], flux[i]};
    }
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

int runBenchmark(int argc, char** argv, int rank, int nranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (n_elems_root <= 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive\n");
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    if (n_elems > static_cast<idx_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Grid size too large: element count must not exceed %d\n", INT_MAX);
        return 1;
    }

    // Bind each rank to one GPU on its node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

    // Share the node's cores among its ranks for the host-side OpenMP phases
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / local_size));
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
               static_cast<unsigned long long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    const std::vector<idx_t> begins = partitionBegins(n_elems, nranks);
    const LocalMesh mesh = buildLocalMesh(n_elems_root, begins, rank, nranks, MPI_COMM_WORLD);
    Simulation sim(mesh, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate memory usage (global, matching the original layout)
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

    // Time the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    sim.run(n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    const int64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    int64_t max_elapsed_ns = 0;
    MPI_Reduce(&elapsed_ns, &max_elapsed_ns, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::nanoseconds(max_elapsed_ns)).count();

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems))
                                          / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Collect the final state in global order on rank 0
    std::vector<ElementDynamic> local_state;
    sim.download(local_state);

    std::vector<ElementDynamic> elements_dynamic(rank == 0 ? n_elems : 0);
    MPI_Datatype pair_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pair_type);
    MPI_Type_commit(&pair_type);

    std::vector<int> counts(nranks);
    std::vector<int> displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = static_cast<int>(begins[r + 1] - begins[r]);
        displs[r] = static_cast<int>(begins[r]);
    }
    MPI_Gatherv(local_state.data(), static_cast<int>(mesh.n_owned), pair_type,
                elements_dynamic.data(), counts.data(), displs.data(), pair_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pair_type);

    int valid = 1;
    if (rank == 0) {
        // Compute hash for verification
        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(elements_dynamic.size());
            for (const auto& elem : elements_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            valid = validateResults(elements_dynamic) ? 1 : 0;
        }
    }

    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return (validate && !valid) ? 1 : 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int status = runBenchmark(argc, argv, rank, nranks);

    MPI_Finalize();
    return status;
}
