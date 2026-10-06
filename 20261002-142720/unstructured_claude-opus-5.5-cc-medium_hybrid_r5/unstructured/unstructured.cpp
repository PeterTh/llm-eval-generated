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

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block partition of the global element index range over MPI ranks
struct Partition {
    idx_t n_global;
    int n_ranks;
    idx_t base, rem;
    Partition(idx_t n, int p) : n_global(n), n_ranks(p), base(n / p), rem(n % p) {}
    idx_t begin(int r) const { return r * base + std::min<idx_t>(r, rem); }
    idx_t end(int r) const { return begin(r + 1); }
    int owner(idx_t g) const {
        const idx_t split = rem * (base + 1);
        if (g < split) return static_cast<int>(g / (base + 1));
        return static_cast<int>(rem + (g - split) / base);
    }
};

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
// Only elements in the global range [lo, hi) are built (owned by this rank).
void buildSquare2D(World& world, const int n_elems_root, const idx_t lo, const idx_t hi) {
    const idx_t n_local = hi - lo;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    const idx_t root = static_cast<idx_t>(n_elems_root);

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = lo + li;
        const int64_t x = static_cast<int64_t>(idx / root);
        const int64_t y = static_cast<int64_t>(idx % root);
        ElementStatic& elem = world.elements_static[li];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        world.elements_dynamic[li].current_energy = 0.0;
        world.elements_dynamic[li].total_flux = 0.0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * root + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const idx_t last = root - 1;
    auto setMat = [&](idx_t g, idx_t mat) {
        if (g >= lo && g < hi) world.elements_static[g - lo].material_idx = mat;
    };
    setMat(0 * root + 0, INFLOW_MAT_ID);
    setMat(0 * root + last, OUTFLOW_MAT_ID);
    setMat(last * root + 0, OUTFLOW_MAT_ID);
    setMat(last * root + last, INFLOW_MAT_ID);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

constexpr int MAX_DEVICE_MATERIALS = 256;
__constant__ Material d_materials[MAX_DEVICE_MATERIALS];

// Compute energy flux between two elements
__host__ __device__ inline val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

// Update elements [begin, end) of the local (reordered) element range.
// Energy arrays hold n_local owned values followed by ghost values.
// If UNIFORM, every connection uses the coefficient uniform_flux (conn_flux unused).
template <bool UNIFORM>
__global__ void __launch_bounds__(256)
updateKernel(const int begin, const int end, const int n_local,
             const uint8_t* __restrict__ mat_idx, const uint8_t* __restrict__ num_conn,
             const int* __restrict__ conn_idx, const val_t* __restrict__ conn_flux,
             const val_t uniform_flux,
             const val_t* __restrict__ energy_in, const val_t* __restrict__ flux_in,
             val_t* __restrict__ energy_out, val_t* __restrict__ flux_out) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const Material mat = d_materials[mat_idx[i]];
    const val_t e = energy_in[i];
    const int nc = num_conn[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (same order as the serial code)
    for (int j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_local + i;
        const val_t cf = UNIFORM ? uniform_flux : conn_flux[slot];
        total_flux += computeFlux(mat.transfer_coeff, e, cf, __ldg(&energy_in[conn_idx[slot]]));
    }

    // Update element state
    energy_out[i] = e + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Gather owned energies that neighbor ranks need into a contiguous buffer
__global__ void packKernel(const int n, const int* __restrict__ idx,
                           const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = energy[idx[i]];
}

// Distributed simulation state for one rank / one GPU
struct DeviceSim {
    int n_local = 0;        // owned elements
    int n_ghost = 0;        // halo elements
    int n_interior = 0;     // owned elements [0, n_interior) have no ghost neighbors
    int max_conn = 0;
    bool uniform_flux = true;
    val_t uniform_flux_value = 1.0;

    std::vector<idx_t> local_to_global;   // reordered local index -> global index

    // Halo exchange plan
    std::vector<int> nbr_ranks;
    std::vector<int> send_counts, send_offsets;   // per neighbor
    std::vector<int> recv_counts, recv_offsets;   // per neighbor (offsets into ghost range)
    int n_send = 0;

    // Device buffers
    uint8_t* d_mat = nullptr;
    uint8_t* d_nconn = nullptr;
    int* d_conn = nullptr;
    val_t* d_conn_flux = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    int* d_send_idx = nullptr;
    val_t* d_send_buf = nullptr;

    // Pinned host staging buffers
    val_t* h_send = nullptr;
    val_t* h_recv = nullptr;

    cudaStream_t s_interior, s_boundary;
    cudaEvent_t ev_packed;
    int cur = 0;
};

// Build the local reordering, halo plan and device data structures
void setupDeviceSim(DeviceSim& sim, const World& world, const Partition& part, const int rank,
                    MPI_Comm comm) {
    const idx_t lo = part.begin(rank);
    const idx_t hi = part.end(rank);
    const idx_t n_local_g = hi - lo;
    if (n_local_g > static_cast<idx_t>(std::numeric_limits<int>::max() / MAX_CONNECTIONS)) {
        fprintf(stderr, "Too many elements per rank; use more MPI ranks\n");
        MPI_Abort(comm, 1);
    }
    if (world.materials.size() > MAX_DEVICE_MATERIALS) {
        fprintf(stderr, "Too many materials\n");
        MPI_Abort(comm, 1);
    }
    const int n_local = static_cast<int>(n_local_g);
    sim.n_local = n_local;

    // Classify owned elements: interior (all neighbors owned) vs boundary
    std::vector<uint8_t> is_boundary(n_local);
    int max_conn = 0;
    bool uniform = true;
    const val_t first_flux = (n_local > 0 && world.elements_static[0].num_connections > 0)
                                 ? world.elements_static[0].connected_flux[0] : 1.0;
    #pragma omp parallel for schedule(static) reduction(max : max_conn) reduction(&& : uniform)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        uint8_t b = 0;
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            if (g < lo || g >= hi) b = 1;
            uniform = uniform && (es.connected_flux[j] == first_flux);
        }
        is_boundary[i] = b;
        max_conn = std::max(max_conn, static_cast<int>(es.num_connections));
    }
    MPI_Allreduce(MPI_IN_PLACE, &max_conn, 1, MPI_INT, MPI_MAX, comm);
    {
        // Uniform coefficient must agree across all ranks
        int u = uniform ? 1 : 0;
        double fmin = uniform ? first_flux : 0.0, fmax = fmin;
        if (!uniform) { fmin = std::numeric_limits<double>::max(); fmax = std::numeric_limits<double>::lowest(); }
        MPI_Allreduce(MPI_IN_PLACE, &u, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &fmin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &fmax, 1, MPI_DOUBLE, MPI_MAX, comm);
        // Bitwise equality of all coefficients (no NaNs expected)
        sim.uniform_flux = (u == 1) && (fmin == fmax);
        sim.uniform_flux_value = fmin;
    }
    sim.max_conn = max_conn;

    // Reorder: interior elements first, then boundary (stable, preserves locality)
    std::vector<int> new_of_old(n_local);
    sim.local_to_global.resize(n_local);
    int n_int = 0;
    for (int i = 0; i < n_local; ++i) if (!is_boundary[i]) new_of_old[i] = n_int++;
    sim.n_interior = n_int;
    int nb = n_int;
    for (int i = 0; i < n_local; ++i) if (is_boundary[i]) new_of_old[i] = nb++;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) sim.local_to_global[new_of_old[i]] = lo + i;

    // Collect unique ghost global indices (sorted => grouped by owner rank)
    std::vector<idx_t> ghosts;
    for (int i = 0; i < n_local; ++i) {
        if (!is_boundary[i]) continue;
        const ElementStatic& es = world.elements_static[i];
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            if (g < lo || g >= hi) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    sim.n_ghost = static_cast<int>(ghosts.size());

    auto ghostLocal = [&](idx_t g) -> int {
        const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), g);
        return n_local + static_cast<int>(it - ghosts.begin());
    };

    // Build device-layout (SoA, slot-major) connectivity
    std::vector<uint8_t> h_mat(n_local), h_nconn(n_local);
    std::vector<int> h_conn(static_cast<size_t>(max_conn) * n_local, 0);
    std::vector<val_t> h_cflux(sim.uniform_flux ? 0 : static_cast<size_t>(max_conn) * n_local, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& es = world.elements_static[i];
        const int ni = new_of_old[i];
        h_mat[ni] = static_cast<uint8_t>(es.material_idx);
        h_nconn[ni] = static_cast<uint8_t>(es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            const int l = (g >= lo && g < hi) ? new_of_old[g - lo] : ghostLocal(g);
            const size_t slot = j * static_cast<size_t>(n_local) + ni;
            h_conn[slot] = l;
            if (!sim.uniform_flux) h_cflux[slot] = es.connected_flux[j];
        }
    }

    // Halo plan: tell each owner which of its elements we need
    int n_ranks;
    MPI_Comm_size(comm, &n_ranks);
    std::vector<int> need_count(n_ranks, 0), give_count(n_ranks, 0);
    for (const idx_t g : ghosts) need_count[part.owner(g)]++;
    MPI_Alltoall(need_count.data(), 1, MPI_INT, give_count.data(), 1, MPI_INT, comm);
    std::vector<int> need_displ(n_ranks, 0), give_displ(n_ranks, 0);
    for (int r = 1; r < n_ranks; ++r) {
        need_displ[r] = need_displ[r - 1] + need_count[r - 1];
        give_displ[r] = give_displ[r - 1] + give_count[r - 1];
    }
    const int total_give = give_displ[n_ranks - 1] + give_count[n_ranks - 1];
    std::vector<idx_t> give_globals(total_give);
    MPI_Alltoallv(ghosts.data(), need_count.data(), need_displ.data(), MPI_UINT64_T,
                  give_globals.data(), give_count.data(), give_displ.data(), MPI_UINT64_T, comm);
    std::vector<int> h_send_idx(total_give);
    for (int k = 0; k < total_give; ++k) h_send_idx[k] = new_of_old[give_globals[k] - lo];
    sim.n_send = total_give;
    for (int r = 0; r < n_ranks; ++r) {
        if (need_count[r] == 0 && give_count[r] == 0) continue;
        sim.nbr_ranks.push_back(r);
        sim.send_counts.push_back(give_count[r]);
        sim.send_offsets.push_back(give_displ[r]);
        sim.recv_counts.push_back(need_count[r]);
        sim.recv_offsets.push_back(need_displ[r]);
    }

    // Device allocations and uploads
    const size_t n_tot = static_cast<size_t>(n_local) + sim.n_ghost;
    CUDA_CHECK(cudaMalloc(&sim.d_mat, std::max<size_t>(n_local, 1)));
    CUDA_CHECK(cudaMalloc(&sim.d_nconn, std::max<size_t>(n_local, 1)));
    CUDA_CHECK(cudaMalloc(&sim.d_conn, std::max<size_t>(h_conn.size(), 1) * sizeof(int)));
    if (!sim.uniform_flux)
        CUDA_CHECK(cudaMalloc(&sim.d_conn_flux, std::max<size_t>(h_cflux.size(), 1) * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&sim.d_energy[b], std::max<size_t>(n_tot, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&sim.d_flux[b], std::max<size_t>(n_local, 1) * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&sim.d_send_idx, std::max(total_give, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sim.d_send_buf, std::max(total_give, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_send, std::max(total_give, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recv, std::max(sim.n_ghost, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(sim.d_mat, h_mat.data(), n_local, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_nconn, h_nconn.data(), n_local, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_conn, h_conn.data(), h_conn.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!sim.uniform_flux)
        CUDA_CHECK(cudaMemcpy(sim.d_conn_flux, h_cflux.data(), h_cflux.size() * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_send_idx, h_send_idx.data(), total_give * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_materials, world.materials.data(),
                                  world.materials.size() * sizeof(Material)));

    // Initial dynamic state
    std::vector<val_t> h_e(n_local), h_f(n_local);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        h_e[new_of_old[i]] = world.elements_dynamic[i].current_energy;
        h_f[new_of_old[i]] = world.elements_dynamic[i].total_flux;
    }
    CUDA_CHECK(cudaMemcpy(sim.d_energy[0], h_e.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_flux[0], h_f.data(), n_local * sizeof(val_t), cudaMemcpyHostToDevice));
    sim.cur = 0;

    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_interior, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_boundary, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_packed, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static inline void launchUpdate(DeviceSim& sim, int begin, int end, cudaStream_t s) {
    if (end <= begin) return;
    constexpr int threads = 256;
    const int blocks = (end - begin + threads - 1) / threads;
    const int c = sim.cur, n = 1 - sim.cur;
    if (sim.uniform_flux)
        updateKernel<true><<<blocks, threads, 0, s>>>(begin, end, sim.n_local, sim.d_mat, sim.d_nconn,
                                                      sim.d_conn, nullptr, sim.uniform_flux_value,
                                                      sim.d_energy[c], sim.d_flux[c],
                                                      sim.d_energy[n], sim.d_flux[n]);
    else
        updateKernel<false><<<blocks, threads, 0, s>>>(begin, end, sim.n_local, sim.d_mat, sim.d_nconn,
                                                       sim.d_conn, sim.d_conn_flux, 0.0,
                                                       sim.d_energy[c], sim.d_flux[c],
                                                       sim.d_energy[n], sim.d_flux[n]);
}

// Run simulation for n_iters iterations
void runSimulation(DeviceSim& sim, const int n_iters, MPI_Comm comm) {
    const int n_nbr = static_cast<int>(sim.nbr_ranks.size());
    std::vector<MPI_Request> reqs(2 * n_nbr);

    for (int iter = 0; iter < n_iters; ++iter) {
        const int c = sim.cur;

        // Pack outgoing halo values (boundary stream) and stage to host
        if (sim.n_send > 0) {
            packKernel<<<(sim.n_send + 255) / 256, 256, 0, sim.s_boundary>>>(
                sim.n_send, sim.d_send_idx, sim.d_energy[c], sim.d_send_buf);
            CUDA_CHECK(cudaMemcpyAsync(sim.h_send, sim.d_send_buf, sim.n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, sim.s_boundary));
            CUDA_CHECK(cudaEventRecord(sim.ev_packed, sim.s_boundary));
        }

        // Interior update overlaps with the halo exchange
        launchUpdate(sim, 0, sim.n_interior, sim.s_interior);

        if (n_nbr > 0) {
            for (int k = 0; k < n_nbr; ++k)
                MPI_Irecv(sim.h_recv + sim.recv_offsets[k], sim.recv_counts[k], MPI_DOUBLE,
                          sim.nbr_ranks[k], 0, comm, &reqs[k]);
            if (sim.n_send > 0) CUDA_CHECK(cudaEventSynchronize(sim.ev_packed));
            for (int k = 0; k < n_nbr; ++k)
                MPI_Isend(sim.h_send + sim.send_offsets[k], sim.send_counts[k], MPI_DOUBLE,
                          sim.nbr_ranks[k], 0, comm, &reqs[n_nbr + k]);
            MPI_Waitall(2 * n_nbr, reqs.data(), MPI_STATUSES_IGNORE);
        }
        if (sim.n_ghost > 0)
            CUDA_CHECK(cudaMemcpyAsync(sim.d_energy[c] + sim.n_local, sim.h_recv,
                                       sim.n_ghost * sizeof(val_t), cudaMemcpyHostToDevice,
                                       sim.s_boundary));

        // Boundary update once ghosts are available
        launchUpdate(sim, sim.n_interior, sim.n_local, sim.s_boundary);

        CUDA_CHECK(cudaStreamSynchronize(sim.s_interior));
        CUDA_CHECK(cudaStreamSynchronize(sim.s_boundary));

        // Swap buffers
        sim.cur = 1 - sim.cur;
    }
    CUDA_CHECK(cudaGetLastError());
}

// Download the local state into world.elements_dynamic (original local order)
void downloadState(const DeviceSim& sim, World& world, const idx_t lo) {
    const int n_local = sim.n_local;
    std::vector<val_t> h_e(n_local), h_f(n_local);
    CUDA_CHECK(cudaMemcpy(h_e.data(), sim.d_energy[sim.cur], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f.data(), sim.d_flux[sim.cur], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        ElementDynamic& d = world.elements_dynamic[sim.local_to_global[i] - lo];
        d.current_energy = h_e[i];
        d.total_flux = h_f[i];
    }
}

// Gather all elements' dynamic state to rank 0 (in global order)
void gatherToRoot(const std::vector<ElementDynamic>& local, std::vector<ElementDynamic>& global,
                  const Partition& part, const int rank, MPI_Comm comm) {
    constexpr size_t CHUNK = 1 << 26;  // elements per message (keeps counts within int range)
    MPI_Datatype dyn_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &dyn_type);
    MPI_Type_commit(&dyn_type);
    if (rank == 0) {
        global.resize(part.n_global);
        std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < part.n_ranks; ++r) {
            const idx_t b = part.begin(r), n = part.end(r) - b;
            for (idx_t off = 0; off < n; off += CHUNK) {
                const int cnt = static_cast<int>(std::min<idx_t>(CHUNK, n - off));
                MPI_Recv(global.data() + b + off, cnt, dyn_type, r, 1, comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        const idx_t n = local.size();
        for (idx_t off = 0; off < n; off += CHUNK) {
            const int cnt = static_cast<int>(std::min<idx_t>(CHUNK, n - off));
            MPI_Send(local.data() + off, cnt, dyn_type, 0, 1, comm);
        }
    }
    MPI_Type_free(&dyn_type);
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
// (partial hash over elements whose global indices start at offset; XOR-combinable)
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const idx_t offset = 0) {
    uint64_t hash = 0;
    const size_t n = elements.size();
    #pragma omp parallel for schedule(static) reduction(^ : hash)
    for (size_t li = 0; li < n; ++li) {
        const size_t i = li + offset;
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[li].current_energy, sizeof(e_bits));
        std::memcpy(&f_bits, &elements[li].total_flux, sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, n_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Bind each rank to a GPU on its node
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(comm, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
        CUDA_CHECK(cudaFree(nullptr));  // create context outside timed region
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    const idx_t n_elems_g = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    const Partition part(n_elems_g, n_ranks);
    const idx_t lo = part.begin(rank), hi = part.end(rank);
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    
        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, lo, hi);
    
    // Calculate memory usage (global)
    if (rank == 0) {
        const size_t static_mem = n_elems_g * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems_g * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    DeviceSim sim;
    setupDeviceSim(sim, world, part, rank, comm);
    // Static connectivity now lives on the device
    std::vector<ElementStatic>().swap(world.elements_static);
    std::vector<ElementDynamic>().swap(world.elements_dynamic_swap);
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    fflush(stdout);
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(sim, n_iters, comm);
    MPI_Barrier(comm);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    downloadState(sim, world, lo);
    uint64_t local_hash = computeHash(world.elements_dynamic, lo);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    std::vector<ElementDynamic> all_dynamic;
    if (printResults || validate) gatherToRoot(world.elements_dynamic, all_dynamic, part, rank, comm);

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
    
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems_g)) / (duration_ms / 1000.0) / 1e9;
    
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
    
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    
        // Hash for verification
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    
        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(all_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < all_dynamic.size(); ++i) {
                energyData[i] = all_dynamic[i].current_energy;
            }
            print_results(energyData, "ElementEnergy");
        }
    
        // Validation
        if (validate) {
            bool valid = validateResults(all_dynamic);
            if (!valid) {
                ret = 1;
            }
        }
    }
    fflush(stdout);
    MPI_Bcast(&ret, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return ret;
}
