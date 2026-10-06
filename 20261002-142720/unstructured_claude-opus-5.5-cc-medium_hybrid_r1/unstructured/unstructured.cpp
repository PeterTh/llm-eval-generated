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

// Hybrid MPI + OpenMP + CUDA version:
//  - MPI:    the element range is block-partitioned across ranks; ghost values of
//            neighbouring elements are exchanged every iteration (halo exchange
//            discovered generically from the unstructured connectivity).
//  - CUDA:   each rank drives one GPU which performs the element updates. Interior
//            elements are updated while the halo exchange is in flight.
//  - OpenMP: host-side mesh construction, renumbering, result post-processing.

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

// World state (elements_static holds only this rank's part of the mesh,
// elements_dynamic holds the gathered global state on rank 0 when needed)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Block partition of the global element range among MPI ranks
inline int64_t partStart(int64_t n, int nranks, int r) {
    return (n * r) / nranks;
}
inline int partOwner(int64_t n, int nranks, int64_t g) {
    int r = static_cast<int>(((g + 1) * nranks - 1) / n);
    while (r > 0 && partStart(n, nranks, r) > g) --r;
    while (r + 1 < nranks && partStart(n, nranks, r + 1) <= g) ++r;
    return r;
}

// Build elements [lo, hi) of a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root, const int64_t lo, const int64_t hi) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.elements_static.resize(hi - lo);

    const int64_t R = n_elems_root;
    const int64_t last = R - 1;

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (int64_t g = lo; g < hi; ++g) {
        const int64_t x = g / R;
        const int64_t y = g % R;
        ElementStatic& elem = world.elements_static[g - lo];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < R && ny >= 0 && ny < R) {
                elem.connected_idx[elem.num_connections] = nx * R + ny;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        if ((x == 0 && y == last) || (x == last && y == 0))
            elem.material_idx = OUTFLOW_MAT_ID;
        if ((x == 0 && y == 0) || (x == last && y == last))
            elem.material_idx = INFLOW_MAT_ID;
    }
}

// ---------------------------------------------------------------------------
// Distributed mesh + device state
// ---------------------------------------------------------------------------
struct DistMesh {
    int rank = 0, nranks = 1;
    int64_t n_global = 0, lo = 0, hi = 0;
    int n_local = 0;      // owned elements, local ids [0, n_local)
    int n_interior = 0;   // owned elements without ghost neighbours: local ids [0, n_interior)
    int n_ghost = 0;      // ghost elements: local ids [n_local, n_local + n_ghost)

    std::vector<int> global_to_local;   // (g - lo) -> local id

    // Halo exchange description
    std::vector<int> recv_ranks, recv_counts, recv_offsets;   // offsets into ghost region
    std::vector<int> send_ranks, send_counts, send_offsets;   // offsets into send buffer
    int n_send = 0;

    // Device arrays (SoA, connection slot j of element i at [j * n_local + i])
    uint8_t* d_nconn = nullptr;
    uint32_t* d_mat = nullptr;
    int* d_cidx = nullptr;
    val_t* d_cflux = nullptr;
    Material* d_mats = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};   // n_local + n_ghost
    val_t* d_flux[2] = {nullptr, nullptr};     // n_local
    int cur = 0;

    int* d_send_idx = nullptr;
    val_t* d_send_buf = nullptr;
    val_t* h_send_buf = nullptr;   // pinned
    val_t* h_recv_buf = nullptr;   // pinned

    cudaStream_t s_compute{}, s_comm{};
    cudaEvent_t ev_packed{}, ev_ghosts{};
};

template <typename T>
T* deviceUpload(const std::vector<T>& h) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty())
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Renumber owned elements (interior first, boundary last), discover ghosts and the
// halo-exchange pattern from the connectivity, and upload everything to the GPU.
void setupDistributed(DistMesh& dm, const World& world) {
    const int64_t lo = dm.lo, hi = dm.hi;
    const auto& es = world.elements_static;
    const int64_t n_local64 = hi - lo;
    if (n_local64 > std::numeric_limits<int>::max() / 2) {
        fprintf(stderr, "Local partition too large; use more MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int n_local = static_cast<int>(n_local64);
    dm.n_local = n_local;

    // Classify owned elements
    std::vector<uint8_t> is_boundary(n_local, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        for (idx_t j = 0; j < es[i].num_connections; ++j) {
            const int64_t g = static_cast<int64_t>(es[i].connected_idx[j]);
            if (g < lo || g >= hi) { is_boundary[i] = 1; break; }
        }
    }

    // Renumber, preserving original order within each class
    std::vector<int64_t> local_to_global(n_local);
    dm.global_to_local.resize(n_local);
    int c = 0;
    for (int i = 0; i < n_local; ++i)
        if (!is_boundary[i]) { dm.global_to_local[i] = c; local_to_global[c++] = lo + i; }
    dm.n_interior = c;
    for (int i = 0; i < n_local; ++i)
        if (is_boundary[i]) { dm.global_to_local[i] = c; local_to_global[c++] = lo + i; }

    // Ghosts: sorted unique global ids (hence grouped by owner rank)
    std::vector<int64_t> ghosts;
    for (int i = dm.n_interior; i < n_local; ++i) {
        const ElementStatic& e = es[local_to_global[i] - lo];
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const int64_t g = static_cast<int64_t>(e.connected_idx[j]);
            if (g < lo || g >= hi) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    dm.n_ghost = static_cast<int>(ghosts.size());

    // What we request from each owner
    std::vector<int> req_counts(dm.nranks, 0), req_displs(dm.nranks, 0);
    for (int64_t g : ghosts) req_counts[partOwner(dm.n_global, dm.nranks, g)]++;
    for (int r = 1; r < dm.nranks; ++r) req_displs[r] = req_displs[r - 1] + req_counts[r - 1];
    for (int r = 0; r < dm.nranks; ++r) {
        if (req_counts[r] > 0) {
            dm.recv_ranks.push_back(r);
            dm.recv_counts.push_back(req_counts[r]);
            dm.recv_offsets.push_back(req_displs[r]);
        }
    }

    // What others request from us
    std::vector<int> srv_counts(dm.nranks, 0), srv_displs(dm.nranks, 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, srv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < dm.nranks; ++r) srv_displs[r] = srv_displs[r - 1] + srv_counts[r - 1];
    dm.n_send = srv_displs[dm.nranks - 1] + srv_counts[dm.nranks - 1];
    std::vector<int64_t> srv_ids(std::max(dm.n_send, 1));
    ghosts.reserve(1);
    MPI_Alltoallv(ghosts.data(), req_counts.data(), req_displs.data(), MPI_INT64_T,
                  srv_ids.data(), srv_counts.data(), srv_displs.data(), MPI_INT64_T,
                  MPI_COMM_WORLD);
    std::vector<int> send_idx(dm.n_send);
    for (int k = 0; k < dm.n_send; ++k) send_idx[k] = dm.global_to_local[srv_ids[k] - lo];
    for (int r = 0; r < dm.nranks; ++r) {
        if (srv_counts[r] > 0) {
            dm.send_ranks.push_back(r);
            dm.send_counts.push_back(srv_counts[r]);
            dm.send_offsets.push_back(srv_displs[r]);
        }
    }

    // Connectivity in local numbering (SoA for coalesced GPU access)
    int max_conn = 0;
    for (int i = 0; i < n_local; ++i)
        max_conn = std::max(max_conn, static_cast<int>(es[i].num_connections));
    const size_t stride = static_cast<size_t>(n_local);
    std::vector<uint8_t> h_nconn(n_local);
    std::vector<uint32_t> h_mat(n_local);
    std::vector<int> h_cidx(max_conn * stride, 0);
    std::vector<val_t> h_cflux(max_conn * stride, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        const ElementStatic& e = es[local_to_global[i] - lo];
        h_nconn[i] = static_cast<uint8_t>(e.num_connections);
        h_mat[i] = static_cast<uint32_t>(e.material_idx);
        for (idx_t j = 0; j < e.num_connections; ++j) {
            const int64_t g = static_cast<int64_t>(e.connected_idx[j]);
            int l;
            if (g >= lo && g < hi)
                l = dm.global_to_local[g - lo];
            else
                l = n_local + static_cast<int>(
                        std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin());
            h_cidx[j * stride + i] = l;
            h_cflux[j * stride + i] = e.connected_flux[j];
        }
    }

    // Upload
    dm.d_nconn = deviceUpload(h_nconn);
    dm.d_mat = deviceUpload(h_mat);
    dm.d_cidx = deviceUpload(h_cidx);
    dm.d_cflux = deviceUpload(h_cflux);
    dm.d_mats = deviceUpload(world.materials);
    dm.d_send_idx = deviceUpload(send_idx);

    // Initialize all elements with zero energy and zero accumulated flux
    const size_t n_ext = std::max<size_t>(stride + dm.n_ghost, 1);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dm.d_energy[b], n_ext * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dm.d_flux[b], std::max<size_t>(stride, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dm.d_energy[b], 0, n_ext * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dm.d_flux[b], 0, std::max<size_t>(stride, 1) * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&dm.d_send_buf, std::max(dm.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&dm.h_send_buf, std::max(dm.n_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&dm.h_recv_buf, std::max(dm.n_ghost, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaStreamCreateWithFlags(&dm.s_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&dm.s_comm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&dm.ev_packed, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&dm.ev_ghosts, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeDistributed(DistMesh& dm) {
    cudaFree(dm.d_nconn); cudaFree(dm.d_mat); cudaFree(dm.d_cidx); cudaFree(dm.d_cflux);
    cudaFree(dm.d_mats); cudaFree(dm.d_send_idx); cudaFree(dm.d_send_buf);
    for (int b = 0; b < 2; ++b) { cudaFree(dm.d_energy[b]); cudaFree(dm.d_flux[b]); }
    cudaFreeHost(dm.h_send_buf); cudaFreeHost(dm.h_recv_buf);
    cudaStreamDestroy(dm.s_compute); cudaStreamDestroy(dm.s_comm);
    cudaEventDestroy(dm.ev_packed); cudaEventDestroy(dm.ev_ghosts);
}

// Compute energy flux between two elements
// (IEEE round-to-nearest ops in the original evaluation order, no contraction)
__device__ __forceinline__ val_t computeFlux(const Material& mat, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return __dmul_rn(__dmul_rn(__dmul_rn(__dsub_rn(other_energy, this_energy),
                                         mat.transfer_coeff), connection_flux), 0.25);
}

constexpr int BLOCK_SIZE = 256;

__global__ void __launch_bounds__(BLOCK_SIZE)
updateKernel(const int begin, const int end, const size_t stride,
             const uint8_t* __restrict__ nconn, const uint32_t* __restrict__ mat_idx,
             const int* __restrict__ cidx, const val_t* __restrict__ cflux,
             const Material* __restrict__ mats,
             const val_t* __restrict__ e_in, const val_t* __restrict__ f_in,
             val_t* __restrict__ e_out, val_t* __restrict__ f_out) {
    const int i = begin + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= end) return;

    const val_t e = e_in[i];
    const Material mat = mats[mat_idx[i]];
    const int nc = nconn[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    for (int j = 0; j < nc; ++j) {
        const size_t s = j * stride + i;
        const val_t other = __ldg(&e_in[cidx[s]]);
        total_flux = __dadd_rn(total_flux, computeFlux(mat, e, cflux[s], other));
    }

    // Update element state
    e_out[i] = __dadd_rn(e, total_flux);
    f_out[i] = __dadd_rn(f_in[i], fabs(total_flux));
}

__global__ void packKernel(const int n, const int* __restrict__ idx,
                           const val_t* __restrict__ e, val_t* __restrict__ buf) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) buf[k] = e[idx[k]];
}

inline void launchUpdate(const DistMesh& dm, const int begin, const int end, cudaStream_t s) {
    if (end <= begin) return;
    const unsigned nb = static_cast<unsigned>((end - begin + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int in = dm.cur, out = 1 - dm.cur;
    updateKernel<<<nb, BLOCK_SIZE, 0, s>>>(begin, end, static_cast<size_t>(dm.n_local),
                                           dm.d_nconn, dm.d_mat, dm.d_cidx, dm.d_cflux, dm.d_mats,
                                           dm.d_energy[in], dm.d_flux[in],
                                           dm.d_energy[out], dm.d_flux[out]);
}

// Run simulation for n_iters iterations
void runSimulation(DistMesh& dm, const int n_iters) {
    const bool has_halo = dm.n_send > 0 || dm.n_ghost > 0;
    std::vector<MPI_Request> reqs(dm.recv_ranks.size() + dm.send_ranks.size());

    for (int iter = 0; iter < n_iters; ++iter) {
        if (!has_halo) {
            launchUpdate(dm, 0, dm.n_local, dm.s_compute);
            dm.cur = 1 - dm.cur;   // Swap buffers
            continue;
        }

        val_t* e_cur = dm.d_energy[dm.cur];

        // Post receives early
        int nr = 0;
        for (size_t k = 0; k < dm.recv_ranks.size(); ++k)
            MPI_Irecv(dm.h_recv_buf + dm.recv_offsets[k], dm.recv_counts[k], MPI_DOUBLE,
                      dm.recv_ranks[k], 0, MPI_COMM_WORLD, &reqs[nr++]);

        // Pack values requested by neighbours, then update interior elements
        // while the halo exchange is in flight
        if (dm.n_send > 0) {
            packKernel<<<(dm.n_send + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, dm.s_compute>>>(
                dm.n_send, dm.d_send_idx, e_cur, dm.d_send_buf);
            CUDA_CHECK(cudaEventRecord(dm.ev_packed, dm.s_compute));
        }
        launchUpdate(dm, 0, dm.n_interior, dm.s_compute);

        if (dm.n_send > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(dm.s_comm, dm.ev_packed, 0));
            CUDA_CHECK(cudaMemcpyAsync(dm.h_send_buf, dm.d_send_buf, dm.n_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, dm.s_comm));
            CUDA_CHECK(cudaStreamSynchronize(dm.s_comm));
        }
        for (size_t k = 0; k < dm.send_ranks.size(); ++k)
            MPI_Isend(dm.h_send_buf + dm.send_offsets[k], dm.send_counts[k], MPI_DOUBLE,
                      dm.send_ranks[k], 0, MPI_COMM_WORLD, &reqs[nr++]);
        MPI_Waitall(nr, reqs.data(), MPI_STATUSES_IGNORE);

        // Upload ghost values, then update boundary elements
        if (dm.n_ghost > 0) {
            CUDA_CHECK(cudaMemcpyAsync(e_cur + dm.n_local, dm.h_recv_buf, dm.n_ghost * sizeof(val_t),
                                       cudaMemcpyHostToDevice, dm.s_comm));
            CUDA_CHECK(cudaEventRecord(dm.ev_ghosts, dm.s_comm));
            CUDA_CHECK(cudaStreamWaitEvent(dm.s_compute, dm.ev_ghosts, 0));
        }
        launchUpdate(dm, dm.n_interior, dm.n_local, dm.s_compute);

        // Swap buffers
        dm.cur = 1 - dm.cur;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(dm.s_compute));
}

// Copy owned results back to the host in global element order
std::vector<ElementDynamic> downloadResults(const DistMesh& dm) {
    const int n = dm.n_local;
    std::vector<val_t> e(n), f(n);
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(e.data(), dm.d_energy[dm.cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(f.data(), dm.d_flux[dm.cur], n * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    std::vector<ElementDynamic> out(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int l = dm.global_to_local[i];
        out[i].current_energy = e[l];
        out[i].total_flux = f[l];
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

// Compute a simple hash of the results for verification.
// elements holds global indices [first, first + size); XOR-combining is order
// independent, so partial hashes from all ranks can be XOR-reduced.
uint64_t computeHash(const std::vector<ElementDynamic>& elements, const uint64_t first) {
    uint64_t hash = 0;
    const int64_t n = static_cast<int64_t>(elements.size());
    #pragma omp parallel for schedule(static) reduction(^ : hash)
    for (int64_t k = 0; k < n; ++k) {
        const uint64_t i = first + static_cast<uint64_t>(k);
        // Simple hash combining energy and flux values
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

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

    // One GPU per rank (round-robin over the GPUs of each node)
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev <= 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
        CUDA_CHECK(cudaFree(nullptr));   // create context outside the timed region
    }

    const int n_elems = n_elems_root * n_elems_root;
    const int64_t n_global = static_cast<int64_t>(n_elems_root) * n_elems_root;

    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build this rank's part of the unstructured mesh
    DistMesh dm;
    dm.rank = rank;
    dm.nranks = nranks;
    dm.n_global = n_global;
    dm.lo = partStart(n_global, nranks, rank);
    dm.hi = partStart(n_global, nranks, rank + 1);
    World world;
    buildSquare2D(world, n_elems_root, dm.lo, dm.hi);

    // Calculate memory usage (global mesh, original data layout)
    const size_t static_mem = static_cast<size_t>(n_global) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_global) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (root) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    setupDistributed(dm, world);
    world.elements_static.clear();
    world.elements_static.shrink_to_fit();

    // Run simulation
    if (root) {
        printf("Running simulation...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(dm, n_iters);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Results back to host, hash via XOR-reduction over ranks
    std::vector<ElementDynamic> local_dyn = downloadResults(dm);
    const uint64_t local_hash = computeHash(local_dyn, static_cast<uint64_t>(dm.lo));
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    // Gather the full state on rank 0 if it is needed for output/validation
    if (printResults || validate) {
        MPI_Datatype dyn_type;
        MPI_Type_contiguous(2, MPI_DOUBLE, &dyn_type);
        MPI_Type_commit(&dyn_type);
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            displs[r] = static_cast<int>(partStart(n_global, nranks, r));
            counts[r] = static_cast<int>(partStart(n_global, nranks, r + 1) - displs[r]);
        }
        if (root) world.elements_dynamic.resize(n_global);
        MPI_Gatherv(local_dyn.data(), static_cast<int>(local_dyn.size()), dyn_type,
                    world.elements_dynamic.data(), counts.data(), displs.data(), dyn_type,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&dyn_type);
    }
    freeDistributed(dm);

    int ret = 0;
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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(world.elements_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < energyData.size(); ++i)
                energyData[i] = world.elements_dynamic[i].current_energy;
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            if (!validateResults(world)) ret = 1;
        }
        fflush(stdout);
    }
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return ret;
}
