#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Neighbors per element in the 2D grid (stride of the connectivity arrays)
constexpr int GRID_NEIGHBORS = 4;
static_assert(GRID_NEIGHBORS <= MAX_CONNECTIONS, "grid neighbors exceed MAX_CONNECTIONS");

// Threads per CUDA block for the element kernels
constexpr uint32_t CUDA_BLOCK_SIZE = 256;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (logical layout, used for memory reporting)
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
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t), "ElementDynamic must hold two packed values");

// World state (materials everywhere; elements_dynamic holds the full result on rank 0)
struct World {
    std::vector<Material> materials;
    std::vector<ElementDynamic> elements_dynamic;
};

// Material type IDs
constexpr uint32_t DEFAULT_MAT_ID = 0;
constexpr uint32_t INFLOW_MAT_ID = 1;
constexpr uint32_t OUTFLOW_MAT_ID = 2;

// Abort all ranks if a CUDA call fails
#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

inline void checkCuda(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line, call, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Owning pointers for device and pinned host memory
struct DeviceFree {
    void operator()(void* p) const { cudaFree(p); }
};
struct PinnedFree {
    void operator()(void* p) const { cudaFreeHost(p); }
};
template <typename T>
using DevicePtr = std::unique_ptr<T, DeviceFree>;
template <typename T>
using PinnedPtr = std::unique_ptr<T, PinnedFree>;

template <typename T>
DevicePtr<T> allocDevice(size_t n) {
    T* p = nullptr;
    if (n > 0) CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return DevicePtr<T>(p);
}

template <typename T>
PinnedPtr<T> allocPinned(size_t n) {
    T* p = nullptr;
    if (n > 0) CUDA_CHECK(cudaMallocHost(&p, n * sizeof(T)));
    return PinnedPtr<T>(p);
}

// Copy a host vector to device memory (no-op when empty)
template <typename T>
void uploadVec(T* dst, const std::vector<T>& src) {
    if (!src.empty()) CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// Number of CUDA blocks needed to cover count threads
inline uint32_t blocksFor(uint32_t count) {
    return (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
}

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Material table, identical on every rank
void buildMaterials(World& world) {
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
}

// Rank r owns global elements [rangeStart(r), rangeStart(r + 1)); the blocks are contiguous and balanced
idx_t rangeStart(int r, idx_t n_elems, int n_ranks) {
    return static_cast<idx_t>(r) * n_elems / static_cast<idx_t>(n_ranks);
}

// This rank's slice of the mesh. Neighbor slots index owned elements [0, n_local) and then
// ghost copies of remote neighbors [n_local, n_local + ghosts.size())
struct LocalMesh {
    idx_t lo = 0;                    // First owned global element
    uint32_t n_local = 0;            // Number of owned elements
    std::vector<uint32_t> conn;      // [GRID_NEIGHBORS][n_local] neighbor slots
    std::vector<double> cflux;       // [GRID_NEIGHBORS][n_local] connection flux coefficients
    std::vector<uint8_t> nconn;      // [n_local] number of connections
    std::vector<uint32_t> mat;       // [n_local] material index
    std::vector<uint32_t> interior;  // Owned elements whose neighbors are all owned
    std::vector<uint32_t> boundary;  // Owned elements with at least one ghost neighbor
    std::vector<idx_t> ghosts;       // Sorted global ids of remote neighbors
};

// Build this rank's slice of the 2D square grid (same connectivity and materials as the full mesh)
LocalMesh buildLocalMesh(int n_elems_root, int rank, int n_ranks) {
    const int64_t n = n_elems_root;
    const idx_t N = static_cast<idx_t>(n_elems_root);
    const idx_t n_elems = N * N;
    const idx_t last = N - 1;
    const idx_t lo = rangeStart(rank, n_elems, n_ranks);
    const idx_t hi = rangeStart(rank + 1, n_elems, n_ranks);
    const size_t n_local = hi - lo;

    LocalMesh mesh;
    mesh.lo = lo;
    mesh.n_local = static_cast<uint32_t>(n_local);
    mesh.conn.resize(GRID_NEIGHBORS * n_local);
    mesh.cflux.resize(GRID_NEIGHBORS * n_local);
    mesh.nconn.resize(n_local);
    mesh.mat.resize(n_local);

    // Global neighbor ids, slot-major, converted to local slots once ghosts are known
    std::vector<idx_t> conn_global(GRID_NEIGHBORS * n_local);
    constexpr int offsets[GRID_NEIGHBORS][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    // Corner elements: (0,0) inflow, (0,last) outflow, (last,0) outflow, (last,last) inflow
    const idx_t corners[4] = {0, last, last * N, last * N + last};
    const uint32_t corner_mat[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_local; ++i) {
        const idx_t g = lo + i;
        const int64_t x = static_cast<int64_t>(g / N);
        const int64_t y = static_cast<int64_t>(g % N);

        uint32_t m = DEFAULT_MAT_ID;
        for (int c = 0; c < 4; ++c) {
            if (g == corners[c]) m = corner_mat[c];
        }
        mesh.mat[i] = m;

        // Connect to neighbors (right, left, up, down) that are inside the grid
        uint8_t nc = 0;
        for (int k = 0; k < GRID_NEIGHBORS; ++k) {
            const int64_t nx = x + offsets[k][0];
            const int64_t ny = y + offsets[k][1];
            if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                const size_t slot = static_cast<size_t>(nc) * n_local + i;
                conn_global[slot] = static_cast<idx_t>(nx * n + ny);
                mesh.cflux[slot] = 1.0;
                ++nc;
            }
        }
        mesh.nconn[i] = nc;
    }

    // Remote neighbors become ghost slots, numbered in ascending global order
    for (size_t i = 0; i < n_local; ++i) {
        for (uint8_t j = 0; j < mesh.nconn[i]; ++j) {
            const idx_t g = conn_global[j * n_local + i];
            if (g < lo || g >= hi) mesh.ghosts.push_back(g);
        }
    }
    std::sort(mesh.ghosts.begin(), mesh.ghosts.end());
    mesh.ghosts.erase(std::unique(mesh.ghosts.begin(), mesh.ghosts.end()), mesh.ghosts.end());
    if (n_local + mesh.ghosts.size() > std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr, "Mesh slice too large for 32-bit local indices\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<uint8_t> has_ghost(n_local, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_local; ++i) {
        for (uint8_t j = 0; j < mesh.nconn[i]; ++j) {
            const size_t slot = j * n_local + i;
            const idx_t g = conn_global[slot];
            if (g >= lo && g < hi) {
                mesh.conn[slot] = static_cast<uint32_t>(g - lo);
            } else {
                const auto it = std::lower_bound(mesh.ghosts.begin(), mesh.ghosts.end(), g);
                mesh.conn[slot] = static_cast<uint32_t>(n_local + static_cast<size_t>(it - mesh.ghosts.begin()));
                has_ghost[i] = 1;
            }
        }
    }

    for (size_t i = 0; i < n_local; ++i) {
        (has_ghost[i] ? mesh.boundary : mesh.interior).push_back(static_cast<uint32_t>(i));
    }
    return mesh;
}

// Ghost-exchange pattern: which owned values to send to whom, and where received values land
struct HaloPlan {
    MPI_Comm graph = MPI_COMM_NULL;            // Distributed-graph communicator over exchange partners
    std::vector<int> src_counts, src_displs;   // Ghost values received from each source rank
    std::vector<int> dst_counts, dst_displs;   // Owned values sent to each destination rank
    std::vector<uint32_t> send_idx;            // Owned local elements to send, grouped by destination
};

// Exchange ghost requests with their owners once, so each owner knows which elements to send
HaloPlan buildHaloPlan(const LocalMesh& mesh, idx_t n_elems, int n_ranks) {
    std::vector<idx_t> starts(n_ranks + 1);
    for (int r = 0; r <= n_ranks; ++r) starts[r] = rangeStart(r, n_elems, n_ranks);

    // Ghosts are sorted by global id, so each owner's ghosts form one contiguous block
    std::vector<int> need(n_ranks, 0);
    for (idx_t g : mesh.ghosts) {
        const int owner = static_cast<int>(std::upper_bound(starts.begin(), starts.end(), g) - starts.begin()) - 1;
        ++need[owner];
    }
    std::vector<int> need_displs(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) need_displs[r] = need_displs[r - 1] + need[r - 1];

    std::vector<int> supply(n_ranks, 0);
    MPI_Alltoall(need.data(), 1, MPI_INT, supply.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> supply_displs(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) supply_displs[r] = supply_displs[r - 1] + supply[r - 1];

    // Each owner receives the global ids that other ranks need from it
    std::vector<idx_t> requested(supply_displs.empty() ? 0 : supply_displs.back() + supply.back());
    MPI_Alltoallv(mesh.ghosts.data(), need.data(), need_displs.data(), MPI_UINT64_T,
                  requested.data(), supply.data(), supply_displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);

    HaloPlan plan;
    plan.send_idx.resize(requested.size());
    for (size_t k = 0; k < requested.size(); ++k) {
        plan.send_idx[k] = static_cast<uint32_t>(requested[k] - mesh.lo);
    }

    // Communicate only with ranks that actually exchange data
    std::vector<int> sources, destinations;
    for (int r = 0; r < n_ranks; ++r) {
        if (need[r] > 0) {
            sources.push_back(r);
            plan.src_counts.push_back(need[r]);
            plan.src_displs.push_back(need_displs[r]);
        }
        if (supply[r] > 0) {
            destinations.push_back(r);
            plan.dst_counts.push_back(supply[r]);
            plan.dst_displs.push_back(supply_displs[r]);
        }
    }
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, static_cast<int>(sources.size()), sources.data(), MPI_UNWEIGHTED,
                                   static_cast<int>(destinations.size()), destinations.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, 0, &plan.graph);
    return plan;
}

// Device-side view of the local mesh (struct-of-arrays, slot-major)
struct DeviceMesh {
    uint32_t n_local;
    const uint32_t* conn;      // [GRID_NEIGHBORS][n_local] neighbor slots
    const double* cflux;      // [GRID_NEIGHBORS][n_local] connection flux coefficients
    const uint8_t* nconn;     // [n_local]
    const uint32_t* mat;      // [n_local]
    const Material* mats;     // [n_materials]
};

// Gather the owned values that neighbor ranks need into a contiguous buffer
__global__ void packKernel(const uint32_t* __restrict__ send_idx, uint32_t count,
                           const double* __restrict__ energy, double* __restrict__ out) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) out[t] = energy[send_idx[t]];
}

// One update of the elements in list: reads cur (owned and ghost slots), writes nxt, accumulates flux
__global__ void stepKernel(DeviceMesh mesh, const uint32_t* __restrict__ list, uint32_t count,
                           const double* __restrict__ cur, double* __restrict__ nxt,
                           double* __restrict__ flux) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const uint32_t i = list[t];
    const Material& mat = mesh.mats[mesh.mat[i]];
    const val_t self = cur[i];

    // Same accumulation order as the reference: external flow, then neighbors in slot order
    val_t total_flux = mat.external_flow;
    const uint32_t n_conn = mesh.nconn[i];
    for (uint32_t j = 0; j < n_conn; ++j) {
        const size_t slot = static_cast<size_t>(j) * mesh.n_local + i;
        total_flux += computeFlux(mat, self, mesh.cflux[slot], cur[mesh.conn[slot]]);
    }

    nxt[i] = self + total_flux;
    flux[i] += fabs(total_flux);
}

// GPU solver for this rank's share of the mesh
class Solver {
  public:
    Solver(const LocalMesh& mesh, HaloPlan plan, const std::vector<Material>& materials);
    ~Solver();
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    // Advance the state n_iters steps, exchanging ghost energies with neighbor ranks each step
    void run(int n_iters);
    // Final state of the owned elements, in global order
    std::vector<ElementDynamic> download() const;

  private:
    DeviceMesh view() const;
    void launchStep(cudaStream_t stream, const uint32_t* list, uint32_t count);

    uint32_t n_local_;
    uint32_t n_interior_;
    uint32_t n_boundary_;
    uint32_t n_recv_;  // Ghost values received per step (equals the ghost slot count)
    HaloPlan plan_;
    cudaStream_t outer_ = nullptr;  // Packing, halo copies and boundary elements
    cudaStream_t inner_ = nullptr;  // Interior elements, overlapped with the halo exchange

    DevicePtr<uint32_t> conn_, mat_, interior_, boundary_, send_idx_;
    DevicePtr<double> cflux_;
    DevicePtr<uint8_t> nconn_;
    DevicePtr<Material> mats_;
    DevicePtr<double> energy_a_, energy_b_;  // Double-buffered energies (owned + ghost slots)
    DevicePtr<double> flux_;                 // Accumulated flux, updated in place
    DevicePtr<double> send_buf_;
    PinnedPtr<double> h_send_, h_recv_;      // Pinned host staging for the halo exchange
    double* cur_ = nullptr;
    double* nxt_ = nullptr;
};

Solver::Solver(const LocalMesh& mesh, HaloPlan plan, const std::vector<Material>& materials)
    : n_local_(mesh.n_local),
      n_interior_(static_cast<uint32_t>(mesh.interior.size())),
      n_boundary_(static_cast<uint32_t>(mesh.boundary.size())),
      n_recv_(static_cast<uint32_t>(mesh.ghosts.size())),
      plan_(std::move(plan)) {
    const size_t n_ext = mesh.n_local + mesh.ghosts.size();
    const size_t n_send = plan_.send_idx.size();
    const size_t n_recv = n_recv_;

    CUDA_CHECK(cudaStreamCreateWithFlags(&outer_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&inner_, cudaStreamNonBlocking));

    conn_ = allocDevice<uint32_t>(mesh.conn.size());
    uploadVec(conn_.get(), mesh.conn);
    cflux_ = allocDevice<double>(mesh.cflux.size());
    uploadVec(cflux_.get(), mesh.cflux);
    nconn_ = allocDevice<uint8_t>(mesh.nconn.size());
    uploadVec(nconn_.get(), mesh.nconn);
    mat_ = allocDevice<uint32_t>(mesh.mat.size());
    uploadVec(mat_.get(), mesh.mat);
    mats_ = allocDevice<Material>(materials.size());
    uploadVec(mats_.get(), materials);
    interior_ = allocDevice<uint32_t>(mesh.interior.size());
    uploadVec(interior_.get(), mesh.interior);
    boundary_ = allocDevice<uint32_t>(mesh.boundary.size());
    uploadVec(boundary_.get(), mesh.boundary);
    send_idx_ = allocDevice<uint32_t>(n_send);
    uploadVec(send_idx_.get(), plan_.send_idx);

    energy_a_ = allocDevice<double>(n_ext);
    energy_b_ = allocDevice<double>(n_ext);
    flux_ = allocDevice<double>(n_local_);
    send_buf_ = allocDevice<double>(n_send);
    if (n_ext > 0) {
        CUDA_CHECK(cudaMemset(energy_a_.get(), 0, n_ext * sizeof(double)));
        CUDA_CHECK(cudaMemset(energy_b_.get(), 0, n_ext * sizeof(double)));
    }
    if (n_local_ > 0) CUDA_CHECK(cudaMemset(flux_.get(), 0, n_local_ * sizeof(double)));

    h_send_ = allocPinned<double>(n_send);
    h_recv_ = allocPinned<double>(n_recv);

    cur_ = energy_a_.get();
    nxt_ = energy_b_.get();
}

Solver::~Solver() {
    cudaStreamDestroy(outer_);
    cudaStreamDestroy(inner_);
    MPI_Comm_free(&plan_.graph);
}

DeviceMesh Solver::view() const {
    return DeviceMesh{n_local_, conn_.get(), cflux_.get(), nconn_.get(), mat_.get(), mats_.get()};
}

void Solver::launchStep(cudaStream_t stream, const uint32_t* list, uint32_t count) {
    if (count == 0) return;
    stepKernel<<<blocksFor(count), CUDA_BLOCK_SIZE, 0, stream>>>(view(), list, count, cur_, nxt_, flux_.get());
    CUDA_CHECK(cudaGetLastError());
}

void Solver::run(int n_iters) {
    const uint32_t n_send = static_cast<uint32_t>(plan_.send_idx.size());

    for (int it = 0; it < n_iters; ++it) {
        // Stage the owned values that neighbors need in pinned host memory
        if (n_send > 0) {
            packKernel<<<blocksFor(n_send), CUDA_BLOCK_SIZE, 0, outer_>>>(send_idx_.get(), n_send, cur_, send_buf_.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_send_.get(), send_buf_.get(), n_send * sizeof(double),
                                       cudaMemcpyDeviceToHost, outer_));
            CUDA_CHECK(cudaStreamSynchronize(outer_));
        }

        // Interior elements read only owned values, so they compute while the halo is exchanged
        launchStep(inner_, interior_.get(), n_interior_);
        MPI_Neighbor_alltoallv(h_send_.get(), plan_.dst_counts.data(), plan_.dst_displs.data(), MPI_DOUBLE,
                               h_recv_.get(), plan_.src_counts.data(), plan_.src_displs.data(), MPI_DOUBLE,
                               plan_.graph);

        // Ghost slots follow the owned slots; received values already arrive in ghost order
        if (n_recv_ > 0) {
            CUDA_CHECK(cudaMemcpyAsync(cur_ + n_local_, h_recv_.get(), n_recv_ * sizeof(double),
                                       cudaMemcpyHostToDevice, outer_));
        }

        // Boundary elements need the received ghost values
        launchStep(outer_, boundary_.get(), n_boundary_);
        CUDA_CHECK(cudaStreamSynchronize(outer_));
        CUDA_CHECK(cudaStreamSynchronize(inner_));
        std::swap(cur_, nxt_);
    }
}

std::vector<ElementDynamic> Solver::download() const {
    std::vector<double> energy(n_local_), flux(n_local_);
    if (n_local_ > 0) {
        CUDA_CHECK(cudaMemcpy(energy.data(), cur_, n_local_ * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flux.data(), flux_.get(), n_local_ * sizeof(double), cudaMemcpyDeviceToHost));
    }
    std::vector<ElementDynamic> out(n_local_);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_local_; ++i) {
        out[i] = ElementDynamic{energy[i], flux[i]};
    }
    return out;
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : world.elements_dynamic) {
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

// Compute a simple hash of the results for verification (XOR-combined, so order-independent)
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    #pragma omp parallel for schedule(static) reduction(^ : hash)
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

// Run the benchmark on this rank's share of the mesh; returns the process exit code
int runBenchmark(int argc, char** argv, int rank, int n_ranks) {
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

    if (n_elems_root < 1) {
        if (rank == 0) printf("Grid size must be at least 1\n");
        return 1;
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

    // Bind each rank to one GPU on its node
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaSetDevice(node_rank % n_devices));

    // Build the unstructured mesh (each rank builds only its slice)
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildMaterials(world);
    const LocalMesh mesh = buildLocalMesh(n_elems_root, rank, n_ranks);
    HaloPlan plan = buildHaloPlan(mesh, static_cast<idx_t>(n_elems), n_ranks);
    Solver solver(mesh, std::move(plan), world.materials);

    // Calculate memory usage (logical per-element footprint of the full mesh)
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation (all ranks start together; the reported time is the slowest rank)
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    solver.run(n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    const long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full final state on rank 0 (global element order)
    const std::vector<ElementDynamic> local_state = solver.download();
    std::vector<int> counts(n_ranks), displs(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        const idx_t first = rangeStart(r, static_cast<idx_t>(n_elems), n_ranks);
        const idx_t after = rangeStart(r + 1, static_cast<idx_t>(n_elems), n_ranks);
        counts[r] = static_cast<int>(2 * (after - first));
        displs[r] = static_cast<int>(2 * first);
    }
    if (rank == 0) world.elements_dynamic.resize(n_elems);
    MPI_Gatherv(local_state.data(), static_cast<int>(2 * local_state.size()), MPI_DOUBLE,
                world.elements_dynamic.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

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
        const uint64_t hash = computeHash(world.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(world.elements_dynamic.size());
            for (const auto& elem : world.elements_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
    }

    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) valid = validateResults(world) ? 1 : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const int rc = runBenchmark(argc, argv, rank, n_ranks);

    MPI_Finalize();
    return rc;
}
