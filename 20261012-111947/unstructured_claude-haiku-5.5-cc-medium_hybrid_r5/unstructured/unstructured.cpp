#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
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

// Host-side view of the elements owned by one rank
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Contiguous block partition of global element indices across MPI ranks
struct Partition {
    std::vector<idx_t> bounds;  // size = nranks + 1; rank r owns [bounds[r], bounds[r+1])

    idx_t lo(int r) const { return bounds[r]; }
    idx_t hi(int r) const { return bounds[r + 1]; }
    int owner(idx_t g) const {
        return static_cast<int>(std::upper_bound(bounds.begin(), bounds.end(), g) - bounds.begin()) - 1;
    }
};

// Build the elements owned by [lo, hi) of a 2D square grid as an unstructured mesh.
// Connectivity refers to global element indices.
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const idx_t lo, const idx_t hi) {
    const idx_t n = static_cast<idx_t>(n_elems_root);
    const idx_t last = n - 1;
    const idx_t n_owned = hi - lo;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.resize(n_owned);

    const long long offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    #pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < n_owned; ++k) {
        const idx_t idx = lo + k;
        const long long x = static_cast<long long>(idx / n);
        const long long y = static_cast<long long>(idx % n);
        ElementStatic& elem = world.elements_static[k];

        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Set corner elements as inflow/outflow to create interesting dynamics
        if (idx == 0) elem.material_idx = INFLOW_MAT_ID;                          // (0, 0)
        if (idx == last) elem.material_idx = OUTFLOW_MAT_ID;                      // (0, last)
        if (idx == last * n) elem.material_idx = OUTFLOW_MAT_ID;                  // (last, 0)
        if (idx == last * n + last) elem.material_idx = INFLOW_MAT_ID;            // (last, last)

        // Connect to neighbors (up, down, left, right) that lie within bounds
        for (int nb = 0; nb < 4; ++nb) {
            const long long nx = x + offsets[nb][0];
            const long long ny = y + offsets[nb][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                elem.connected_idx[elem.num_connections] = static_cast<idx_t>(nx * n_elems_root + ny);
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }
}

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

// Device-side read-only view of the static mesh data (structure-of-arrays, stride = n_owned)
struct MeshView {
    const int* mat;
    const int* n_conn;
    const int* conn;
    const double* cflux;
    const Material* mats;
    int stride;
};

// One thread per element: update elements in [begin, end) from cur_* into nxt_*.
// Reads cur_e for neighbours (owned and ghost), writes only nxt_e/nxt_f at index i.
__global__ void updateKernel(MeshView m, int begin, int end,
                             const double* __restrict__ cur_e, const double* __restrict__ cur_f,
                             double* __restrict__ nxt_e, double* __restrict__ nxt_f) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const Material mat = m.mats[m.mat[i]];
    const val_t this_energy = cur_e[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements, in the same order as the host code
    const int n_conn = m.n_conn[i];
    for (int j = 0; j < n_conn; ++j) {
        const size_t slot = static_cast<size_t>(j) * m.stride + i;
        total_flux += computeFlux(mat, this_energy, m.cflux[slot], cur_e[m.conn[slot]]);
    }

    nxt_e[i] = this_energy + total_flux;
    nxt_f[i] = cur_f[i] + fabs(total_flux);
}

// Pack the owned energies that neighbouring ranks need into a contiguous send buffer
__global__ void gatherKernel(const int* __restrict__ idx, const double* __restrict__ src,
                             double* __restrict__ dst, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) dst[k] = src[idx[k]];
}

// Communication plan for ghost-element exchange between ranks
struct HaloPlan {
    MPI_Comm graph = MPI_COMM_NULL;           // Distributed graph: sources = owners, destinations = requesters
    std::vector<int> send_counts, send_displs;  // Per destination (graph order)
    std::vector<int> recv_counts, recv_displs;  // Per source (graph order)
    std::vector<int> send_local;                // Owned local indices to pack, grouped by destination
};

// Exclusive prefix sum
std::vector<int> exclusiveScan(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
    return displs;
}

// Build the halo plan. local_of_owned maps (global - lo) to local index.
HaloPlan buildHaloPlan(MPI_Comm comm, int size, const Partition& part, idx_t lo,
                       const std::vector<idx_t>& ghosts, const std::vector<idx_t>& local_of_owned) {
    HaloPlan plan;

    // Number of ghosts needed from each owner. Ghosts are sorted, so each owner's
    // ghosts are contiguous and owners appear in ascending rank order.
    std::vector<int> req_from(size, 0);
    for (idx_t g : ghosts) req_from[part.owner(g)]++;

    // Number of requests each rank will send to me
    std::vector<int> req_to(size, 0);
    MPI_Alltoall(req_from.data(), 1, MPI_INT, req_to.data(), 1, MPI_INT, comm);

    // Send my ghost requests to their owners
    const std::vector<int> req_from_displs = exclusiveScan(req_from);
    const std::vector<int> req_to_displs = exclusiveScan(req_to);
    const int n_requested = req_to_displs.empty() ? 0 : req_to_displs.back() + req_to.back();
    std::vector<idx_t> requested(n_requested);
    MPI_Alltoallv(ghosts.data(), req_from.data(), req_from_displs.data(), MPI_UINT64_T,
                  requested.data(), req_to.data(), req_to_displs.data(), MPI_UINT64_T, comm);

    // Owners translate requested global ids to local indices, grouped by requester
    plan.send_local.resize(n_requested);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_requested; ++i) {
        plan.send_local[i] = static_cast<int>(local_of_owned[requested[i] - lo]);
    }

    // Graph neighbourhood: ranks I receive from (owners) and send to (requesters), ascending
    std::vector<int> sources, dests;
    for (int r = 0; r < size; ++r) {
        if (req_from[r] > 0) {
            sources.push_back(r);
            plan.recv_counts.push_back(req_from[r]);
        }
        if (req_to[r] > 0) {
            dests.push_back(r);
            plan.send_counts.push_back(req_to[r]);
        }
    }
    plan.send_displs = exclusiveScan(plan.send_counts);
    plan.recv_displs = exclusiveScan(plan.recv_counts);

    MPI_Dist_graph_create_adjacent(comm, static_cast<int>(sources.size()), sources.data(), MPI_UNWEIGHTED,
                                   static_cast<int>(dests.size()), dests.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, 0, &plan.graph);
    return plan;
}

// Device-side mesh and simulation state for one rank
struct DeviceState {
    int n_owned = 0;     // Locally owned elements, local indices [0, n_owned)
    int n_interior = 0;  // Owned elements with only owned neighbours: [0, n_interior)
    int n_ghost = 0;     // Ghost elements, local indices [n_owned, n_owned + n_ghost)
    int n_send = 0;
    int n_recv = 0;

    MeshView view{};
    Material* mats = nullptr;
    int* mat = nullptr;
    int* n_conn = nullptr;
    int* conn = nullptr;
    double* cflux = nullptr;

    double* energy[2] = {nullptr, nullptr};  // Size n_owned + n_ghost
    double* flux[2] = {nullptr, nullptr};    // Size n_owned

    int* send_local = nullptr;
    double* send_buf = nullptr;
    double* h_send = nullptr;  // Pinned host staging for halo send
    double* h_recv = nullptr;  // Pinned host staging for halo receive

    cudaStream_t s_comm{};
    cudaStream_t s_interior{};
};

template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

template <typename T>
T* pinnedAlloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

// Upload the rank-local mesh (already remapped to local indices) and allocate state
void uploadMesh(DeviceState& dev, const World& world, const std::vector<ElementStatic>& local_static,
                const std::vector<idx_t>& ghosts) {
    const int n_owned = static_cast<int>(local_static.size());
    dev.n_owned = n_owned;
    dev.n_ghost = static_cast<int>(ghosts.size());
    const int n_total = n_owned + dev.n_ghost;

    // Structure-of-arrays layout so that consecutive threads read consecutive memory
    std::vector<int> h_mat(n_owned), h_n_conn(n_owned);
    std::vector<int> h_conn(static_cast<size_t>(MAX_CONNECTIONS) * n_owned, 0);
    std::vector<double> h_cflux(static_cast<size_t>(MAX_CONNECTIONS) * n_owned, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_owned; ++i) {
        const ElementStatic& e = local_static[i];
        h_mat[i] = static_cast<int>(e.material_idx);
        h_n_conn[i] = static_cast<int>(e.num_connections);
        for (int j = 0; j < static_cast<int>(e.num_connections); ++j) {
            const size_t slot = static_cast<size_t>(j) * n_owned + i;
            h_conn[slot] = static_cast<int>(e.connected_idx[j]);
            h_cflux[slot] = e.connected_flux[j];
        }
    }

    dev.mats = deviceAlloc<Material>(world.materials.size());
    dev.mat = deviceAlloc<int>(n_owned);
    dev.n_conn = deviceAlloc<int>(n_owned);
    dev.conn = deviceAlloc<int>(static_cast<size_t>(MAX_CONNECTIONS) * n_owned);
    dev.cflux = deviceAlloc<double>(static_cast<size_t>(MAX_CONNECTIONS) * n_owned);
    CUDA_CHECK(cudaMemcpy(dev.mats, world.materials.data(), world.materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.mat, h_mat.data(), h_mat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.n_conn, h_n_conn.data(), h_n_conn.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.conn, h_conn.data(), h_conn.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.cflux, h_cflux.data(), h_cflux.size() * sizeof(double), cudaMemcpyHostToDevice));
    dev.view = MeshView{dev.mat, dev.n_conn, dev.conn, dev.cflux, dev.mats, n_owned};

    // Dynamic state, both buffers zero-initialized as in the original
    for (int b = 0; b < 2; ++b) {
        dev.energy[b] = deviceAlloc<double>(n_total);
        dev.flux[b] = deviceAlloc<double>(n_owned);
        CUDA_CHECK(cudaMemset(dev.energy[b], 0, std::max(n_total, 1) * sizeof(double)));
        CUDA_CHECK(cudaMemset(dev.flux[b], 0, std::max(n_owned, 1) * sizeof(double)));
    }

    CUDA_CHECK(cudaStreamCreate(&dev.s_comm));
    CUDA_CHECK(cudaStreamCreate(&dev.s_interior));
}

// Advance the simulation for n_iters iterations entirely on the device, exchanging
// ghost energies with neighbouring ranks each iteration. Returns the buffer index
// holding the final state.
int runSimulation(DeviceState& dev, const HaloPlan& plan, const int n_iters) {
    const int n_boundary_begin = dev.n_interior;
    const int n_boundary = dev.n_owned - dev.n_interior;
    constexpr int threads = 256;
    const int interior_blocks = (dev.n_interior + threads - 1) / threads;
    const int boundary_blocks = (n_boundary + threads - 1) / threads;
    const int send_blocks = (dev.n_send + threads - 1) / threads;

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = 1 - cur;
        double* cur_e = dev.energy[cur];
        double* cur_f = dev.flux[cur];
        double* nxt_e = dev.energy[nxt];
        double* nxt_f = dev.flux[nxt];

        // 1. Pack and copy the energies that neighbours need to host
        if (dev.n_send > 0) {
            gatherKernel<<<send_blocks, threads, 0, dev.s_comm>>>(dev.send_local, cur_e, dev.send_buf, dev.n_send);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(dev.h_send, dev.send_buf, dev.n_send * sizeof(double),
                                       cudaMemcpyDeviceToHost, dev.s_comm));
        }

        // 2. Interior elements need no remote data: update them while the halo is exchanged
        if (dev.n_interior > 0) {
            updateKernel<<<interior_blocks, threads, 0, dev.s_interior>>>(
                dev.view, 0, dev.n_interior, cur_e, cur_f, nxt_e, nxt_f);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Exchange ghost energies with neighbouring ranks
        CUDA_CHECK(cudaStreamSynchronize(dev.s_comm));
        MPI_Neighbor_alltoallv(dev.h_send, plan.send_counts.data(), plan.send_displs.data(), MPI_DOUBLE,
                               dev.h_recv, plan.recv_counts.data(), plan.recv_displs.data(), MPI_DOUBLE,
                               plan.graph);
        if (dev.n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(cur_e + dev.n_owned, dev.h_recv, dev.n_ghost * sizeof(double),
                                       cudaMemcpyHostToDevice, dev.s_comm));
        }

        // 4. Boundary elements, which read ghost energies
        if (n_boundary > 0) {
            updateKernel<<<boundary_blocks, threads, 0, dev.s_comm>>>(
                dev.view, n_boundary_begin, dev.n_owned, cur_e, cur_f, nxt_e, nxt_f);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(dev.s_interior));
        CUDA_CHECK(cudaStreamSynchronize(dev.s_comm));
        cur = nxt;
    }
    return cur;
}

// Hash of the elements owned by this rank, using global element indices.
// Combining with XOR makes the global hash independent of the rank decomposition.
uint64_t computeLocalHash(const std::vector<double>& energy, const std::vector<double>& flux, idx_t offset) {
    uint64_t hash = 0;
    const size_t n = energy.size();
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const size_t gi = static_cast<size_t>(offset) + i;
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &energy[i], sizeof(e_bits));
        std::memcpy(&f_bits, &flux[i], sizeof(f_bits));
        // Simple hash combining energy and flux values
        hash ^= (e_bits + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

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
        if (root) fprintf(stderr, "Grid size must be at least 1\n");
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank: pick the device by node-local rank
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    int node_size = 1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));

    // Split host threads between the ranks sharing a node
    omp_set_num_threads(std::max(1, omp_get_num_procs() / node_size));

    const int n_elems = n_elems_root * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Contiguous block partition of elements across ranks
    Partition part;
    part.bounds.resize(size + 1);
    for (int r = 0; r <= size; ++r) {
        part.bounds[r] = static_cast<idx_t>(n_elems) * static_cast<idx_t>(r) / static_cast<idx_t>(size);
    }
    const idx_t lo = part.lo(rank);
    const idx_t hi = part.hi(rank);
    const idx_t n_owned = hi - lo;

    // Build the local part of the unstructured mesh
    if (root) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, lo, hi);

    // Ghosts: remote elements referenced by owned elements, sorted by global id
    std::vector<idx_t> ghosts;
    for (const ElementStatic& elem : world.elements_static) {
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t nb = elem.connected_idx[j];
            if (nb < lo || nb >= hi) ghosts.push_back(nb);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());

    // Reorder owned elements so that those with only owned neighbours come first
    std::vector<idx_t> local_of_owned(n_owned);
    idx_t n_interior = 0;
    std::vector<char> is_interior(n_owned, 1);  // char rather than bool: written concurrently
    #pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < n_owned; ++k) {
        const ElementStatic& elem = world.elements_static[k];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t nb = elem.connected_idx[j];
            if (nb < lo || nb >= hi) is_interior[k] = 0;
        }
    }
    for (idx_t k = 0; k < n_owned; ++k) {
        if (is_interior[k]) local_of_owned[k] = n_interior++;
    }
    idx_t next = n_interior;
    for (idx_t k = 0; k < n_owned; ++k) {
        if (!is_interior[k]) local_of_owned[k] = next++;
    }

    // Local element order, with connectivity remapped to local indices
    std::vector<ElementStatic> local_static(n_owned);
    std::vector<idx_t> global_of_local(n_owned);
    #pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < n_owned; ++k) {
        local_static[local_of_owned[k]] = world.elements_static[k];
        global_of_local[local_of_owned[k]] = lo + k;
    }
    #pragma omp parallel for schedule(static)
    for (idx_t l = 0; l < n_owned; ++l) {
        ElementStatic& elem = local_static[l];
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t nb = elem.connected_idx[j];
            if (nb >= lo && nb < hi) {
                elem.connected_idx[j] = local_of_owned[nb - lo];
            } else {
                const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), nb);
                elem.connected_idx[j] = n_owned + static_cast<idx_t>(it - ghosts.begin());
            }
        }
    }

    // Communication plan and device upload
    HaloPlan plan = buildHaloPlan(MPI_COMM_WORLD, size, part, lo, ghosts, local_of_owned);
    DeviceState dev;
    dev.n_interior = static_cast<int>(n_interior);
    dev.n_send = static_cast<int>(plan.send_local.size());
    dev.n_recv = static_cast<int>(ghosts.size());
    uploadMesh(dev, world, local_static, ghosts);
    dev.send_local = deviceAlloc<int>(dev.n_send);
    dev.send_buf = deviceAlloc<double>(dev.n_send);
    dev.h_send = pinnedAlloc<double>(dev.n_send);
    dev.h_recv = pinnedAlloc<double>(dev.n_recv);
    if (dev.n_send > 0) {
        CUDA_CHECK(cudaMemcpy(dev.send_local, plan.send_local.data(), dev.n_send * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // Calculate memory usage (global, as in the original)
    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (root) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    const int final_buf = runSimulation(dev, plan, n_iters);

    const double t_elapsed = MPI_Wtime() - t_start;
    double t_max = 0.0;
    MPI_Reduce(&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long long duration_ms = static_cast<long long>(t_max * 1000.0);

    if (root) {
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
    }

    // Copy final state back and reorder to global element order for this rank
    std::vector<double> local_energy(n_owned), local_flux(n_owned);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), dev.energy[final_buf], n_owned * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), dev.flux[final_buf], n_owned * sizeof(double),
                          cudaMemcpyDeviceToHost));
    std::vector<double> owned_energy(n_owned), owned_flux(n_owned);
    #pragma omp parallel for schedule(static)
    for (idx_t k = 0; k < n_owned; ++k) {
        owned_energy[k] = local_energy[local_of_owned[k]];
        owned_flux[k] = local_flux[local_of_owned[k]];
    }

    // Compute hash for verification (XOR-combined across ranks)
    uint64_t local_hash = computeLocalHash(owned_energy, owned_flux, lo);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather the full state on the root for printing and validation
    int exit_code = 0;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(part.hi(r) - part.lo(r));
            displs[r] = static_cast<int>(part.lo(r));
        }
        std::vector<double> all_energy(root ? n_elems : 0), all_flux(root ? n_elems : 0);
        MPI_Gatherv(owned_energy.data(), static_cast<int>(n_owned), MPI_DOUBLE,
                    all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(owned_flux.data(), static_cast<int>(n_owned), MPI_DOUBLE,
                    all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(all_energy, "ElementEnergy");
            }

            // Validation
            if (validate) {
                World full;
                full.elements_dynamic.resize(n_elems);
                for (int i = 0; i < n_elems; ++i) {
                    full.elements_dynamic[i] = ElementDynamic{all_energy[i], all_flux[i]};
                }
                if (!validateResults(full)) {
                    exit_code = 1;
                }
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(dev.s_comm));
    CUDA_CHECK(cudaStreamDestroy(dev.s_interior));
    MPI_Comm_free(&plan.graph);
    MPI_Finalize();
    return exit_code;
}
