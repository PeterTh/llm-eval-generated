#include <mpi.h>
#include <cuda_runtime.h>

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

constexpr Material MATERIALS[3] = {
    {0.8, 0.0},    // Default material
    {0.8, 0.5},    // Inflow material
    {0.8, -0.5},   // Outflow material
};

// CUDA threads per block for the element kernels
constexpr unsigned THREADS_PER_BLOCK = 256;

// MPI tag for halo exchange messages
constexpr int HALO_TAG = 17;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Contiguous block [begin, end) of global element ids owned by one rank
struct Range {
    idx_t begin;
    idx_t end;
};

// Balanced block partition of n elements over nranks ranks
Range partitionRange(int rank, int nranks, idx_t n) {
    const idx_t base = n / nranks;
    const idx_t rem = n % nranks;
    const idx_t r = static_cast<idx_t>(rank);
    const idx_t begin = r * base + std::min(r, rem);
    return {begin, begin + base + (r < rem ? 1 : 0)};
}

// Neighbouring rank with halo traffic, and the offsets of its slices in the packed buffers
struct Neighbor {
    int rank;
    int send_off;
    int send_cnt;
    int recv_off;
    int recv_cnt;
};

// Host-side description of this rank's part of the mesh
struct LocalMesh {
    Range owned{0, 0};
    int n_own = 0;
    int n_ghost = 0;
    std::vector<val_t> transfer_coeff;   // Per owned element, from its material
    std::vector<val_t> external_flow;    // Per owned element, from its material
    std::vector<int> conn_off;           // CSR offsets into conn_idx / conn_flux, size n_own + 1
    std::vector<int> conn_idx;           // Local index: owned [0, n_own), ghost [n_own, n_own + n_ghost)
    std::vector<val_t> conn_flux;
    std::vector<int> interior;           // Owned elements with only owned neighbours
    std::vector<int> boundary;           // Owned elements with at least one ghost neighbour
    std::vector<idx_t> ghost_global;     // Sorted global ids of ghost elements
    std::vector<int> send_idx;           // Owned local indices sent to neighbours, grouped by neighbour
    std::vector<Neighbor> neighbors;
};

// Build the static records of the elements in `owned` for an n_root x n_root 2D grid
std::vector<ElementStatic> buildSquare2DRange(Range owned, const int n_root) {
    const idx_t n = static_cast<idx_t>(n_root);
    const idx_t last = n - 1;
    std::vector<ElementStatic> elems(owned.end - owned.begin);

    #pragma omp parallel for schedule(static)
    for (idx_t g = owned.begin; g < owned.end; ++g) {
        ElementStatic& elem = elems[g - owned.begin];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right) that lie inside the grid
        const int64_t x = static_cast<int64_t>(g / n);
        const int64_t y = static_cast<int64_t>(g % n);
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int k = 0; k < 4; ++k) {
            const int64_t nx = x + offsets[k][0];
            const int64_t ny = y + offsets[k][1];
            if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
                elem.connected_idx[elem.num_connections] = static_cast<idx_t>(nx) * n + static_cast<idx_t>(ny);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Corner elements as inflow/outflow to create interesting dynamics.
        // Later assignments override earlier ones, as in the serial generator.
        if (g == 0) elem.material_idx = INFLOW_MAT_ID;
        if (g == last) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * n) elem.material_idx = OUTFLOW_MAT_ID;
        if (g == last * n + last) elem.material_idx = INFLOW_MAT_ID;
    }
    return elems;
}

// Exclusive prefix sum of counts, giving displacements for the matching counts
std::vector<int> exclusiveScan(const std::vector<int>& counts) {
    std::vector<int> offsets(counts.size());
    int acc = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = acc;
        acc += counts[i];
    }
    return offsets;
}

// Build this rank's mesh partition, ghost layer and halo exchange plan
LocalMesh buildLocalMesh(int rank, int nranks, const int n_root) {
    LocalMesh m;
    const idx_t n_total = static_cast<idx_t>(n_root) * static_cast<idx_t>(n_root);
    m.owned = partitionRange(rank, nranks, n_total);
    m.n_own = static_cast<int>(m.owned.end - m.owned.begin);

    const std::vector<ElementStatic> elems = buildSquare2DRange(m.owned, n_root);

    // CSR connectivity offsets
    m.conn_off.resize(m.n_own + 1);
    m.conn_off[0] = 0;
    for (int k = 0; k < m.n_own; ++k) {
        m.conn_off[k + 1] = m.conn_off[k] + static_cast<int>(elems[k].num_connections);
    }
    const int n_conn = m.conn_off[m.n_own];
    m.conn_idx.resize(n_conn);
    m.conn_flux.resize(n_conn);

    // Ghost elements: non-owned neighbours, sorted so they group by owning rank
    for (const ElementStatic& elem : elems) {
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = elem.connected_idx[j];
            if (g < m.owned.begin || g >= m.owned.end) m.ghost_global.push_back(g);
        }
    }
    std::sort(m.ghost_global.begin(), m.ghost_global.end());
    m.ghost_global.erase(std::unique(m.ghost_global.begin(), m.ghost_global.end()), m.ghost_global.end());
    m.n_ghost = static_cast<int>(m.ghost_global.size());

    m.transfer_coeff.resize(m.n_own);
    m.external_flow.resize(m.n_own);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < m.n_own; ++k) {
        const Material& mat = MATERIALS[elems[k].material_idx];
        m.transfer_coeff[k] = mat.transfer_coeff;
        m.external_flow[k] = mat.external_flow;

        // Connection order is preserved: it fixes the floating-point summation order
        for (idx_t j = 0; j < elems[k].num_connections; ++j) {
            const idx_t g = elems[k].connected_idx[j];
            int local;
            if (g >= m.owned.begin && g < m.owned.end) {
                local = static_cast<int>(g - m.owned.begin);
            } else {
                const auto it = std::lower_bound(m.ghost_global.begin(), m.ghost_global.end(), g);
                local = m.n_own + static_cast<int>(it - m.ghost_global.begin());
            }
            m.conn_idx[m.conn_off[k] + j] = local;
            m.conn_flux[m.conn_off[k] + j] = elems[k].connected_flux[j];
        }
    }

    // Split owned elements by whether their update needs ghost data
    for (int k = 0; k < m.n_own; ++k) {
        bool touches_ghost = false;
        for (int j = m.conn_off[k]; j < m.conn_off[k + 1]; ++j) {
            touches_ghost |= m.conn_idx[j] >= m.n_own;
        }
        (touches_ghost ? m.boundary : m.interior).push_back(k);
    }

    // Halo plan: each rank tells the owners which of their elements it needs
    std::vector<idx_t> starts(nranks + 1);
    for (int r = 0; r < nranks; ++r) starts[r] = partitionRange(r, nranks, n_total).begin;
    starts[nranks] = n_total;
    auto ownerOf = [&](idx_t g) {
        return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
    };

    std::vector<int> req_cnt(nranks, 0);
    for (idx_t g : m.ghost_global) req_cnt[ownerOf(g)]++;
    const std::vector<int> req_off = exclusiveScan(req_cnt);

    std::vector<int> give_cnt(nranks);
    MPI_Alltoall(req_cnt.data(), 1, MPI_INT, give_cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    const std::vector<int> give_off = exclusiveScan(give_cnt);
    const int n_give = give_off.back() + give_cnt.back();

    std::vector<idx_t> give_ids(n_give);
    MPI_Alltoallv(m.ghost_global.data(), req_cnt.data(), req_off.data(), MPI_UINT64_T,
                  give_ids.data(), give_cnt.data(), give_off.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    m.send_idx.resize(n_give);
    for (int i = 0; i < n_give; ++i) {
        m.send_idx[i] = static_cast<int>(give_ids[i] - m.owned.begin);
    }
    for (int r = 0; r < nranks; ++r) {
        if (r == rank || (req_cnt[r] == 0 && give_cnt[r] == 0)) continue;
        m.neighbors.push_back({r, give_off[r], give_cnt[r], req_off[r], req_cnt[r]});
    }
    return m;
}

// RAII owner of a device allocation (zero-initialized)
template <typename T>
class DeviceVector {
public:
    explicit DeviceVector(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, std::max<size_t>(count, 1) * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr_, 0, std::max<size_t>(count, 1) * sizeof(T)));
    }
    DeviceVector(const std::vector<T>& host) : DeviceVector(host.size()) {
        if (!host.empty()) {
            CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    ~DeviceVector() { cudaFree(ptr_); }
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    T* data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// RAII owner of page-locked host memory, so host<->device copies run at full PCIe speed
class PinnedVector {
public:
    explicit PinnedVector(size_t count) {
        CUDA_CHECK(cudaMallocHost(&ptr_, std::max<size_t>(count, 1) * sizeof(val_t)));
    }
    ~PinnedVector() { cudaFreeHost(ptr_); }
    PinnedVector(const PinnedVector&) = delete;
    PinnedVector& operator=(const PinnedVector&) = delete;

    val_t* data() const { return ptr_; }

private:
    val_t* ptr_ = nullptr;
};

// Pack the energies that neighbouring ranks need into a contiguous buffer
__global__ void packHalo(const int* __restrict__ send_idx, int count,
                         const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    out[k] = energy[send_idx[k]];
}

// Explicit energy transfer update for the elements listed in `list`. Each thread owns one element and
// only reads the previous energies (owned and ghost), so the update is race-free and matches the serial
// double-buffered reference. The arithmetic and summation order mirror computeFlux() exactly.
__global__ void updateElements(const int* __restrict__ list, int count,
                               const int* __restrict__ conn_off, const int* __restrict__ conn_idx,
                               const val_t* __restrict__ conn_flux,
                               const val_t* __restrict__ transfer_coeff,
                               const val_t* __restrict__ external_flow,
                               const val_t* __restrict__ energy_cur, val_t* __restrict__ energy_next,
                               val_t* __restrict__ total_flux) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int i = list[k];
    const val_t self = energy_cur[i];
    const val_t coeff = transfer_coeff[i];

    // Start with external flow, then add flux from all connected elements in order
    val_t flux = external_flow[i];
    for (int j = conn_off[i]; j < conn_off[i + 1]; ++j) {
        flux += (energy_cur[conn_idx[j]] - self) * coeff * conn_flux[j] * 0.25;
    }

    energy_next[i] = self + flux;
    total_flux[i] += fabs(flux);
}

// Per-rank GPU state for the owned elements and their ghost copies
class DeviceSolver {
public:
    DeviceSolver(const LocalMesh& m)
        : n_own_(m.n_own),
          n_ghost_(m.n_ghost),
          n_interior_(static_cast<int>(m.interior.size())),
          n_boundary_(static_cast<int>(m.boundary.size())),
          n_send_(static_cast<int>(m.send_idx.size())),
          conn_off_(m.conn_off),
          conn_idx_(m.conn_idx),
          conn_flux_(m.conn_flux),
          transfer_coeff_(m.transfer_coeff),
          external_flow_(m.external_flow),
          interior_(m.interior),
          boundary_(m.boundary),
          send_idx_(m.send_idx),
          send_buf_(m.send_idx.size()),
          flux_(m.n_own),
          energy_{DeviceVector<val_t>(m.n_own + m.n_ghost), DeviceVector<val_t>(m.n_own + m.n_ghost)},
          h_send_(m.send_idx.size()),
          h_recv_(m.n_ghost) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    ~DeviceSolver() { cudaStreamDestroy(stream_); }
    DeviceSolver(const DeviceSolver&) = delete;
    DeviceSolver& operator=(const DeviceSolver&) = delete;

    // Advance n_iters iterations; ghost energies are refreshed from neighbour ranks every iteration
    void run(const LocalMesh& mesh, int n_iters) {
        std::vector<MPI_Request> reqs(2 * mesh.neighbors.size());
        for (int iter = 0; iter < n_iters; ++iter) {
            // Stage the boundary energies that neighbours need into pinned host memory
            if (n_send_ > 0) {
                packHalo<<<blocksFor(n_send_), THREADS_PER_BLOCK, 0, stream_>>>(
                    send_idx_.data(), n_send_, energy_[cur_].data(), send_buf_.data());
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(h_send_.data(), send_buf_.data(), n_send_ * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, stream_));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            // Post the halo exchange
            int n_req = 0;
            for (const Neighbor& nb : mesh.neighbors) {
                if (nb.recv_cnt > 0) {
                    MPI_Irecv(h_recv_.data() + nb.recv_off, nb.recv_cnt, MPI_DOUBLE, nb.rank, HALO_TAG,
                              MPI_COMM_WORLD, &reqs[n_req++]);
                }
                if (nb.send_cnt > 0) {
                    MPI_Isend(h_send_.data() + nb.send_off, nb.send_cnt, MPI_DOUBLE, nb.rank, HALO_TAG,
                              MPI_COMM_WORLD, &reqs[n_req++]);
                }
            }

            // Interior elements need no ghost data: update them while the halo is in flight
            launchUpdate(interior_, n_interior_);
            MPI_Waitall(n_req, reqs.data(), MPI_STATUSES_IGNORE);

            // Upload the received ghost energies, then update the boundary elements
            if (n_ghost_ > 0) {
                CUDA_CHECK(cudaMemcpyAsync(energy_[cur_].data() + n_own_, h_recv_.data(), n_ghost_ * sizeof(val_t),
                                           cudaMemcpyHostToDevice, stream_));
            }
            launchUpdate(boundary_, n_boundary_);

            cur_ = 1 - cur_;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Copy the owned state back and assemble the full result on rank 0, in global element order
    std::vector<ElementDynamic> gather(int rank, int nranks, idx_t n_total) {
        std::vector<val_t> energy(n_own_), flux(n_own_);
        if (n_own_ > 0) {
            CUDA_CHECK(cudaMemcpy(energy.data(), energy_[cur_].data(), n_own_ * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(flux.data(), flux_.data(), n_own_ * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        std::vector<ElementDynamic> own(n_own_);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n_own_; ++k) {
            own[k] = {energy[k], flux[k]};
        }

        // Each ElementDynamic is two contiguous doubles, so it can be gathered as one MPI record
        MPI_Datatype record;
        MPI_Type_contiguous(2, MPI_DOUBLE, &record);
        MPI_Type_commit(&record);

        std::vector<ElementDynamic> all;
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const Range rr = partitionRange(r, nranks, n_total);
            counts[r] = static_cast<int>(rr.end - rr.begin);
            displs[r] = static_cast<int>(rr.begin);
        }
        if (rank == 0) all.resize(n_total);
        MPI_Gatherv(own.data(), n_own_, record, all.data(), counts.data(), displs.data(), record, 0, MPI_COMM_WORLD);
        MPI_Type_free(&record);
        return all;
    }

private:
    static unsigned blocksFor(size_t count) {
        return static_cast<unsigned>((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    }

    // Update the elements in `list` from energy_[cur_] into energy_[1 - cur_]
    void launchUpdate(const DeviceVector<int>& list, int count) {
        if (count == 0) return;
        updateElements<<<blocksFor(count), THREADS_PER_BLOCK, 0, stream_>>>(
            list.data(), count, conn_off_.data(), conn_idx_.data(), conn_flux_.data(),
            transfer_coeff_.data(), external_flow_.data(),
            energy_[cur_].data(), energy_[1 - cur_].data(), flux_.data());
        CUDA_CHECK(cudaGetLastError());
    }

    int n_own_;
    int n_ghost_;
    int n_interior_;
    int n_boundary_;
    int n_send_;
    DeviceVector<int> conn_off_;
    DeviceVector<int> conn_idx_;
    DeviceVector<val_t> conn_flux_;
    DeviceVector<val_t> transfer_coeff_;
    DeviceVector<val_t> external_flow_;
    DeviceVector<int> interior_;
    DeviceVector<int> boundary_;
    DeviceVector<int> send_idx_;
    DeviceVector<val_t> send_buf_;
    DeviceVector<val_t> flux_;
    DeviceVector<val_t> energy_[2];
    PinnedVector h_send_;
    PinnedVector h_recv_;
    cudaStream_t stream_ = nullptr;
    int cur_ = 0;
};

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

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit together)
    int exit_code = -1;
    for (int i = 1; i < argc && exit_code < 0; ++i) {
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
            exit_code = 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exit_code = 1;
        }
    }
    if (exit_code < 0 && (n_elems_root < 1 || static_cast<int64_t>(n_elems_root) * n_elems_root > INT_MAX)) {
        if (rank == 0) printf("Grid size must be between 1 and %d\n", static_cast<int>(std::sqrt(INT_MAX)));
        exit_code = 1;
    }
    if (exit_code >= 0) {
        MPI_Finalize();
        return exit_code;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Bind each rank to a GPU on its node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

    // Build the unstructured mesh (each rank builds only its partition)
    if (rank == 0) printf("Building unstructured mesh...\n");
    const idx_t n_total = static_cast<idx_t>(n_elems);
    LocalMesh mesh = buildLocalMesh(rank, nranks, n_elems_root);
    DeviceSolver solver(mesh);

    // Calculate memory usage
    if (rank == 0) {
        const size_t static_mem = n_total * sizeof(ElementStatic);
        const size_t dynamic_mem = n_total * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
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

    solver.run(mesh, n_iters);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather the final state on rank 0 (outside the timed region)
    std::vector<ElementDynamic> elements = solver.gather(rank, nranks, n_total);

    int status = 0;
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
        const uint64_t hash = computeHash(elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(elements.size());
            for (const auto& elem : elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(elements);
            if (!valid) {
                status = 1;
            }
        }
    }

    // Every rank returns the same status as rank 0
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
