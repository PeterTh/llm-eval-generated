// Hybrid MPI + OpenMP + CUDA version of the unstructured mesh benchmark.
//
// Parallelization strategy:
//  * MPI:    the global element range is block-partitioned over ranks. Each rank
//            builds only its own elements, derives the halo (ghost) elements it
//            needs from the connectivity and exchanges ghost energies every
//            iteration with exactly the ranks it shares connections with.
//  * CUDA:   each rank drives one GPU (round-robin over the GPUs of a node). The
//            element update runs as a CUDA kernel on a compact SoA / ELL layout.
//            Interior elements (no ghost dependency) are updated concurrently with
//            the halo exchange; boundary elements are updated afterwards.
//  * OpenMP: all host-side work (mesh construction, halo analysis, layout
//            conversion, result reductions) is multithreaded.
//
// The arithmetic is performed in exactly the same order as the serial code with
// explicitly rounded operations, so results are bitwise identical.

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements (global)
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Rank-local part of the world: elements [elem_begin, elem_end) of the global mesh
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    int64_t elem_begin = 0;
    int64_t elem_end = 0;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block partition of [0, n) over n_parts parts
static inline int64_t blockStart(int64_t n, int n_parts, int part) {
    const int64_t base = n / n_parts;
    const int64_t rem = n % n_parts;
    return part * base + std::min<int64_t>(part, rem);
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries.
// Only the elements in [elem_begin, elem_end) are built on this rank.
void buildSquare2D(World& world, const int n_elems_root, int64_t elem_begin, int64_t elem_end) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elem_begin = elem_begin;
    world.elem_end = elem_end;
    const int64_t n_local = elem_end - elem_begin;

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (int64_t li = 0; li < n_local; ++li) {
        const int64_t idx = elem_begin + li;
        const int64_t x = idx / n_elems_root;
        const int64_t y = idx % n_elems_root;
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
                const int64_t neighbor_idx = nx * n_elems_root + ny;
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int64_t last = n_elems_root - 1;
    auto setMat = [&](int64_t gidx, idx_t mat) {
        if (gidx >= elem_begin && gidx < elem_end) world.elements_static[gidx - elem_begin].material_idx = mat;
    };
    setMat(0 * n_elems_root + 0, INFLOW_MAT_ID);
    setMat(0 * n_elems_root + last, OUTFLOW_MAT_ID);
    setMat(last * n_elems_root + 0, OUTFLOW_MAT_ID);
    setMat(last * n_elems_root + last, INFLOW_MAT_ID);
}

// ---------------------------------------------------------------------------
// GPU part
// ---------------------------------------------------------------------------

constexpr int MAX_MATERIALS = 256;
__constant__ val_t c_transfer_coeff[MAX_MATERIALS];
__constant__ val_t c_external_flow[MAX_MATERIALS];

// Connection flux coefficient storage modes (all exact w.r.t. the original data)
constexpr int WEIGHTS_ONE = 0;     // all coefficients are exactly 1.0 -> no storage
constexpr int WEIGHTS_FLOAT = 1;   // all coefficients exactly representable as float
constexpr int WEIGHTS_DOUBLE = 2;  // general case

template <int WMODE> struct WeightType { using type = double; };
template <> struct WeightType<WEIGHTS_FLOAT> { using type = float; };

// Compute energy flux between two elements (same operation order as the serial code)
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    val_t f = __dmul_rn(__dsub_rn(other_energy, this_energy), transfer_coeff);
    f = __dmul_rn(f, connection_flux);
    return __dmul_rn(f, 0.25);
}

// Element update kernel over local elements [begin, end).
// Connectivity is stored in ELL format, connection-major (nbr[j * stride + i]).
// meta[i] = (material << 4) | num_connections
template <int WMODE, int MAXDEG>
__global__ void __launch_bounds__(256)
updateKernel(int begin, int end, int stride, int maxdeg,
             const val_t* __restrict__ energy_in, const val_t* __restrict__ flux_in,
             val_t* __restrict__ energy_out, val_t* __restrict__ flux_out,
             const int* __restrict__ nbr,
             const typename WeightType<WMODE>::type* __restrict__ weights,
             const uint16_t* __restrict__ meta) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const unsigned m = __ldg(&meta[i]);
    const int nconn = m & 0xF;
    const int mat = m >> 4;
    const val_t coeff = c_transfer_coeff[mat];
    const val_t my_energy = __ldg(&energy_in[i]);

    // Start with external flow
    val_t total_flux = c_external_flow[mat];

    const int deg = (MAXDEG > 0) ? MAXDEG : maxdeg;
    #pragma unroll
    for (int j = 0; j < deg; ++j) {
        if (j < nconn) {
            const int n = __ldg(&nbr[(size_t)j * stride + i]);
            const val_t other = __ldg(&energy_in[n]);
            val_t w;
            if constexpr (WMODE == WEIGHTS_ONE) {
                w = 1.0;
            } else {
                w = (val_t)__ldg(&weights[(size_t)j * stride + i]);
            }
            total_flux = __dadd_rn(total_flux, computeFlux(coeff, my_energy, w, other));
        }
    }

    energy_out[i] = __dadd_rn(my_energy, total_flux);
    flux_out[i] = __dadd_rn(__ldg(&flux_in[i]), fabs(total_flux));
}

__global__ void packKernel(int n, const int* __restrict__ idx, const val_t* __restrict__ energy,
                           val_t* __restrict__ buf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = energy[idx[i]];
}

template <int WMODE, int MAXDEG>
static void launchUpdate(int begin, int end, int stride, int maxdeg, const val_t* e_in,
                         const val_t* f_in, val_t* e_out, val_t* f_out, const int* nbr,
                         const void* w, const uint16_t* meta, cudaStream_t s) {
    constexpr int BS = 256;
    const int n = end - begin;
    const int blocks = n > 0 ? (n + BS - 1) / BS : 1;  // 1 block for warm-up launches
    updateKernel<WMODE, MAXDEG><<<blocks, BS, 0, s>>>(
        begin, end, stride, maxdeg, e_in, f_in, e_out, f_out, nbr,
        static_cast<const typename WeightType<WMODE>::type*>(w), meta);
}

using LaunchFn = void (*)(int, int, int, int, const val_t*, const val_t*, val_t*, val_t*,
                          const int*, const void*, const uint16_t*, cudaStream_t);

template <int WMODE>
static LaunchFn selectDeg(int maxdeg) {
    switch (maxdeg) {
        case 4: return launchUpdate<WMODE, 4>;
        case 6: return launchUpdate<WMODE, 6>;
        case 8: return launchUpdate<WMODE, 8>;
        default: return launchUpdate<WMODE, 0>;
    }
}

static LaunchFn selectLaunch(int wmode, int maxdeg) {
    if (wmode == WEIGHTS_ONE) return selectDeg<WEIGHTS_ONE>(maxdeg);
    if (wmode == WEIGHTS_FLOAT) return selectDeg<WEIGHTS_FLOAT>(maxdeg);
    return selectDeg<WEIGHTS_DOUBLE>(maxdeg);
}

// Distributed simulation state of one rank
struct Simulation {
    int rank = 0, n_ranks = 1;
    int n_own = 0;        // owned elements
    int n_interior = 0;   // owned elements [0, n_interior) need no ghost data
    int n_ghost = 0;      // ghost elements stored at local indices [n_own, n_own + n_ghost)
    int maxdeg = 0;
    int wmode = WEIGHTS_ONE;
    std::vector<int> local_to_global;  // owned local index -> global index - elem_begin

    // Halo exchange pattern
    std::vector<int> nb_ranks;                 // neighbor ranks
    std::vector<int> send_off, send_cnt;       // per neighbor, into send buffer
    std::vector<int> recv_off, recv_cnt;       // per neighbor, into recv buffer
    int n_send = 0;

    // Device data
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    int* d_nbr = nullptr;
    void* d_weights = nullptr;
    uint16_t* d_meta = nullptr;
    int* d_send_idx = nullptr;
    val_t* d_send_buf = nullptr;
    val_t* h_send_buf = nullptr;
    val_t* h_recv_buf = nullptr;
    cudaStream_t s_comp = nullptr, s_comm = nullptr;
    cudaEvent_t ev_int = nullptr, ev_bnd = nullptr;
    LaunchFn launch = nullptr;
    int cur = 0;

    std::vector<MPI_Request> reqs;
};

static void setupSimulation(Simulation& sim, const World& world, MPI_Comm comm,
                            const std::vector<int64_t>& rank_start) {
    const int64_t lo = world.elem_begin;
    const int64_t hi = world.elem_end;
    const int n_own = static_cast<int>(hi - lo);
    sim.n_own = n_own;

    if ((int)world.materials.size() > MAX_MATERIALS) {
        fprintf(stderr, "Too many materials\n");
        MPI_Abort(comm, 1);
    }

    // --- Classify owned elements: interior vs boundary; determine max degree and weight mode
    std::vector<char> is_boundary(n_own, 0);
    int maxdeg = 0;
    bool all_one = true, all_float = true;
    #pragma omp parallel for schedule(static) reduction(max : maxdeg) reduction(&& : all_one, all_float)
    for (int i = 0; i < n_own; ++i) {
        const ElementStatic& es = world.elements_static[i];
        maxdeg = std::max(maxdeg, (int)es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const int64_t g = (int64_t)es.connected_idx[j];
            if (g < lo || g >= hi) is_boundary[i] = 1;
            const val_t w = es.connected_flux[j];
            all_one = all_one && (w == 1.0 && !std::signbit(w));
            const float wf = (float)w;
            all_float = all_float && ((val_t)wf == w || std::isnan(w)) && (std::signbit(wf) == std::signbit(w));
        }
    }
    int flags[3] = {maxdeg, all_one ? 1 : 0, all_float ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, &flags[0], 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &flags[1], 2, MPI_INT, MPI_MIN, comm);
    sim.maxdeg = maxdeg = flags[0];
    sim.wmode = flags[1] ? WEIGHTS_ONE : (flags[2] ? WEIGHTS_FLOAT : WEIGHTS_DOUBLE);
    sim.launch = selectLaunch(sim.wmode, maxdeg);

    // --- Local ordering: interior elements first, then boundary (stable)
    sim.local_to_global.resize(n_own);
    std::vector<int> global_to_local(n_own);
    {
        int k = 0;
        for (int i = 0; i < n_own; ++i) if (!is_boundary[i]) sim.local_to_global[k++] = i;
        sim.n_interior = k;
        for (int i = 0; i < n_own; ++i) if (is_boundary[i]) sim.local_to_global[k++] = i;
    }
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < n_own; ++l) global_to_local[sim.local_to_global[l]] = l;

    // --- Ghost elements: sorted unique global indices outside the owned range
    std::vector<int64_t> ghosts;
    for (int i = 0; i < n_own; ++i) {
        if (!is_boundary[i]) continue;
        const ElementStatic& es = world.elements_static[i];
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const int64_t g = (int64_t)es.connected_idx[j];
            if (g < lo || g >= hi) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    sim.n_ghost = (int)ghosts.size();

    // Owner of each ghost (ghosts sorted -> grouped by owner)
    std::vector<int> recv_counts(sim.n_ranks, 0);
    for (int64_t g : ghosts) {
        const int owner = (int)(std::upper_bound(rank_start.begin(), rank_start.end(), g) - rank_start.begin()) - 1;
        recv_counts[owner]++;
    }
    std::vector<int> send_counts(sim.n_ranks, 0);
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm);

    std::vector<int> rdispl(sim.n_ranks, 0), sdispl(sim.n_ranks, 0);
    for (int r = 1; r < sim.n_ranks; ++r) {
        rdispl[r] = rdispl[r - 1] + recv_counts[r - 1];
        sdispl[r] = sdispl[r - 1] + send_counts[r - 1];
    }
    sim.n_send = sdispl[sim.n_ranks - 1] + send_counts[sim.n_ranks - 1];
    std::vector<int64_t> requested(sim.n_send);
    MPI_Alltoallv(ghosts.data(), recv_counts.data(), rdispl.data(), MPI_INT64_T,
                  requested.data(), send_counts.data(), sdispl.data(), MPI_INT64_T, comm);

    for (int r = 0; r < sim.n_ranks; ++r) {
        if (recv_counts[r] == 0 && send_counts[r] == 0) continue;
        sim.nb_ranks.push_back(r);
        sim.send_off.push_back(sdispl[r]);
        sim.send_cnt.push_back(send_counts[r]);
        sim.recv_off.push_back(rdispl[r]);
        sim.recv_cnt.push_back(recv_counts[r]);
    }
    sim.reqs.resize(2 * sim.nb_ranks.size());

    std::vector<int> send_idx(sim.n_send);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < sim.n_send; ++k) send_idx[k] = global_to_local[requested[k] - lo];

    // --- Convert connectivity to local ELL layout
    auto toLocal = [&](int64_t g) -> int {
        if (g >= lo && g < hi) return global_to_local[g - lo];
        return n_own + (int)(std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin());
    };
    const size_t ell_size = (size_t)maxdeg * n_own;
    std::vector<int> nbr(ell_size, 0);
    std::vector<double> w_d(sim.wmode == WEIGHTS_DOUBLE ? ell_size : 0, 0.0);
    std::vector<float> w_f(sim.wmode == WEIGHTS_FLOAT ? ell_size : 0, 0.0f);
    std::vector<uint16_t> meta(n_own);
    std::vector<val_t> energy(n_own + sim.n_ghost, 0.0), flux(n_own, 0.0);
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < n_own; ++l) {
        const int i = sim.local_to_global[l];
        const ElementStatic& es = world.elements_static[i];
        meta[l] = (uint16_t)((es.material_idx << 4) | es.num_connections);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const size_t p = j * (size_t)n_own + l;
            nbr[p] = toLocal((int64_t)es.connected_idx[j]);
            if (sim.wmode == WEIGHTS_DOUBLE) w_d[p] = es.connected_flux[j];
            if (sim.wmode == WEIGHTS_FLOAT) w_f[p] = (float)es.connected_flux[j];
        }
        energy[l] = world.elements_dynamic[i].current_energy;
        flux[l] = world.elements_dynamic[i].total_flux;
    }

    // --- Upload
    std::vector<Material> mats = world.materials;
    std::vector<val_t> coeff(mats.size()), ext(mats.size());
    for (size_t m = 0; m < mats.size(); ++m) { coeff[m] = mats[m].transfer_coeff; ext[m] = mats[m].external_flow; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, coeff.data(), coeff.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, ext.data(), ext.size() * sizeof(val_t)));

    const size_t n_tot = (size_t)n_own + sim.n_ghost;
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&sim.d_energy[b], std::max<size_t>(n_tot, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&sim.d_flux[b], std::max<size_t>(n_own, 1) * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMemcpy(sim.d_energy[0], energy.data(), n_tot * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_energy[1], energy.data(), n_tot * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_flux[0], flux.data(), n_own * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&sim.d_nbr, std::max<size_t>(ell_size, 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(sim.d_nbr, nbr.data(), ell_size * sizeof(int), cudaMemcpyHostToDevice));
    if (sim.wmode == WEIGHTS_DOUBLE) {
        CUDA_CHECK(cudaMalloc(&sim.d_weights, std::max<size_t>(ell_size, 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(sim.d_weights, w_d.data(), ell_size * sizeof(double), cudaMemcpyHostToDevice));
    } else if (sim.wmode == WEIGHTS_FLOAT) {
        CUDA_CHECK(cudaMalloc(&sim.d_weights, std::max<size_t>(ell_size, 1) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(sim.d_weights, w_f.data(), ell_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&sim.d_meta, std::max<size_t>(n_own, 1) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(sim.d_meta, meta.data(), n_own * sizeof(uint16_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&sim.d_send_idx, std::max(sim.n_send, 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(sim.d_send_idx, send_idx.data(), sim.n_send * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&sim.d_send_buf, std::max(sim.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_send_buf, std::max(sim.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&sim.h_recv_buf, std::max(sim.n_ghost, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sim.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_int, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sim.ev_bnd, cudaEventDisableTiming));
    sim.cur = 0;
}

// Start halo exchange of the current energy buffer: pack on GPU, copy to host
static void haloPack(Simulation& sim) {
    if (sim.n_send > 0) {
        constexpr int BS = 256;
        packKernel<<<(sim.n_send + BS - 1) / BS, BS, 0, sim.s_comm>>>(
            sim.n_send, sim.d_send_idx, sim.d_energy[sim.cur], sim.d_send_buf);
        CUDA_CHECK(cudaMemcpyAsync(sim.h_send_buf, sim.d_send_buf, sim.n_send * sizeof(val_t),
                                   cudaMemcpyDeviceToHost, sim.s_comm));
    }
}

// Exchange host buffers via MPI and upload received ghosts into the current buffer
static void haloExchange(Simulation& sim) {
    CUDA_CHECK(cudaStreamSynchronize(sim.s_comm));
    const int nn = (int)sim.nb_ranks.size();
    for (int k = 0; k < nn; ++k)
        MPI_Irecv(sim.h_recv_buf + sim.recv_off[k], sim.recv_cnt[k], MPI_DOUBLE, sim.nb_ranks[k], 0,
                  MPI_COMM_WORLD, &sim.reqs[k]);
    for (int k = 0; k < nn; ++k)
        MPI_Isend(sim.h_send_buf + sim.send_off[k], sim.send_cnt[k], MPI_DOUBLE, sim.nb_ranks[k], 0,
                  MPI_COMM_WORLD, &sim.reqs[nn + k]);
    MPI_Waitall(2 * nn, sim.reqs.data(), MPI_STATUSES_IGNORE);
    if (sim.n_ghost > 0)
        CUDA_CHECK(cudaMemcpyAsync(sim.d_energy[sim.cur] + sim.n_own, sim.h_recv_buf,
                                   sim.n_ghost * sizeof(val_t), cudaMemcpyHostToDevice, sim.s_comm));
}

static void launchRange(Simulation& sim, int begin, int end, cudaStream_t s) {
    const int c = sim.cur, n = 1 - sim.cur;
    sim.launch(begin, end, sim.n_own, sim.maxdeg, sim.d_energy[c], sim.d_flux[c], sim.d_energy[n],
               sim.d_flux[n], sim.d_nbr, sim.d_weights, sim.d_meta, s);
}

// Warm up kernels, MPI connections and copies (does not modify the simulation state
// other than refreshing ghost values, which are overwritten before every use)
static void warmup(Simulation& sim) {
    launchRange(sim, 0, 0, sim.s_comp);
    haloPack(sim);
    haloExchange(sim);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

// Run simulation for n_iters iterations
void runSimulation(Simulation& sim, const int n_iters) {
    const bool has_halo = !sim.nb_ranks.empty();
    for (int iter = 0; iter < n_iters; ++iter) {
        if (!has_halo) {
            launchRange(sim, 0, sim.n_own, sim.s_comp);
        } else {
            // Previous iteration must be complete before current buffer is read / next written
            CUDA_CHECK(cudaStreamWaitEvent(sim.s_comm, sim.ev_int, 0));
            haloPack(sim);
            CUDA_CHECK(cudaStreamWaitEvent(sim.s_comp, sim.ev_bnd, 0));
            if (sim.n_interior > 0) launchRange(sim, 0, sim.n_interior, sim.s_comp);
            CUDA_CHECK(cudaEventRecord(sim.ev_int, sim.s_comp));
            haloExchange(sim);
            if (sim.n_own > sim.n_interior) launchRange(sim, sim.n_interior, sim.n_own, sim.s_comm);
            CUDA_CHECK(cudaEventRecord(sim.ev_bnd, sim.s_comm));
        }
        // Swap buffers
        sim.cur = 1 - sim.cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

// Download local results in global (owned block) order
static void downloadResults(const Simulation& sim, std::vector<ElementDynamic>& out) {
    std::vector<val_t> e(sim.n_own), f(sim.n_own);
    CUDA_CHECK(cudaMemcpy(e.data(), sim.d_energy[sim.cur], sim.n_own * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f.data(), sim.d_flux[sim.cur], sim.n_own * sizeof(val_t), cudaMemcpyDeviceToHost));
    out.resize(sim.n_own);
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < sim.n_own; ++l) {
        out[sim.local_to_global[l]] = ElementDynamic{e[l], f[l]};
    }
}

static void freeSimulation(Simulation& sim) {
    for (int b = 0; b < 2; ++b) { cudaFree(sim.d_energy[b]); cudaFree(sim.d_flux[b]); }
    cudaFree(sim.d_nbr); cudaFree(sim.d_weights); cudaFree(sim.d_meta);
    cudaFree(sim.d_send_idx); cudaFree(sim.d_send_buf);
    cudaFreeHost(sim.h_send_buf); cudaFreeHost(sim.h_recv_buf);
    cudaStreamDestroy(sim.s_comp); cudaStreamDestroy(sim.s_comm);
    cudaEventDestroy(sim.ev_int); cudaEventDestroy(sim.ev_bnd);
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    // Sequential summation order kept for bitwise-identical output
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

// Compute a simple hash of the (local block of) results for verification.
// The hash is an XOR over elements, so partial hashes combine with XOR.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, uint64_t global_offset) {
    uint64_t hash = 0;
    const int64_t n = (int64_t)elements.size();
    #pragma omp parallel for schedule(static) reduction(^ : hash)
    for (int64_t k = 0; k < n; ++k) {
        const uint64_t i = global_offset + (uint64_t)k;
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &elements[k].current_energy, sizeof(uint64_t));
        memcpy(&f_bits, &elements[k].total_flux, sizeof(uint64_t));
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

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

    // Select GPU: round-robin over the GPUs of this node
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev <= 0) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
        CUDA_CHECK(cudaFree(nullptr));  // create context
    }

    const int n_elems = n_elems_root * n_elems_root;
    const int64_t n_elems_global = (int64_t)n_elems_root * n_elems_root;

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

    // Block partition of the elements over ranks
    std::vector<int64_t> rank_start(n_ranks + 1);
    for (int r = 0; r <= n_ranks; ++r) rank_start[r] = blockStart(n_elems_global, n_ranks, r);
    if (n_elems_global / std::max(n_ranks, 1) > std::numeric_limits<int>::max() / 2) {
        if (rank == 0) fprintf(stderr, "Too many elements per rank\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    World world;
    buildSquare2D(world, n_elems_root, rank_start[rank], rank_start[rank + 1]);

    // Calculate memory usage (global, as in the serial version)
    const size_t static_mem = (size_t)n_elems_global * sizeof(ElementStatic);
    const size_t dynamic_mem = (size_t)n_elems_global * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    Simulation sim;
    sim.rank = rank;
    sim.n_ranks = n_ranks;
    setupSimulation(sim, world, MPI_COMM_WORLD, rank_start);
    // Static mesh data is now on the GPU; release host copy
    std::vector<ElementStatic>().swap(world.elements_static);
    warmup(sim);

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters);

    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_ns, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::nanoseconds(elapsed_ns)).count();

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems_global) / (duration_ms / 1000.0) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Compute hash for verification
    std::vector<ElementDynamic> local_results;
    downloadResults(sim, local_results);
    freeSimulation(sim);
    uint64_t local_hash = computeHash(local_results, (uint64_t)rank_start[rank]);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Gather full results to rank 0 when needed
    int exit_code = 0;
    if (printResults || validate) {
        std::vector<int> counts(n_ranks), displs(n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            counts[r] = (int)((rank_start[r + 1] - rank_start[r]) * 2);
            displs[r] = (int)(rank_start[r] * 2);
        }
        std::vector<ElementDynamic> all;
        if (rank == 0) all.resize(n_elems_global);
        MPI_Gatherv(local_results.data(), counts[rank], MPI_DOUBLE, all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                std::vector<double> energyData(all.size());
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < all.size(); ++k) energyData[k] = all[k].current_energy;
                print_results(energyData, "ElementEnergy");
            }

            // Validation
            if (validate) {
                bool valid = validateResults(all);
                if (!valid) exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}
