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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

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

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);

    // Initialize all elements with default material and zero energy
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Pack energies of boundary-owned elements into a contiguous send buffer
__global__ void packKernel(const val_t* __restrict__ energy,
                           const int* __restrict__ send_idx,
                           val_t* __restrict__ sendbuf, const int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {
        sendbuf[t] = energy[send_idx[t]];
    }
}

// Update a list of owned elements. Per-element flux accumulation follows the
// exact connection order of the original serial code so results match.
__global__ void updateKernel(const int* __restrict__ list, const int count,
                             const int n_owned,
                             const val_t* __restrict__ transfer,
                             const val_t* __restrict__ external,
                             const int* __restrict__ ncon,
                             const int* __restrict__ conn_idx,
                             const val_t* __restrict__ conn_flux,
                             const val_t* __restrict__ energy_in,
                             const val_t* __restrict__ flux_in,
                             val_t* __restrict__ energy_out,
                             val_t* __restrict__ flux_out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = list[t];

    const val_t this_energy = energy_in[i];
    const val_t coeff = transfer[i];
    const int nc = ncon[i];

    // Start with external flow
    val_t total_flux = external[i];

    // Add flux from all connected elements
    for (int j = 0; j < nc; ++j) {
        const size_t off = static_cast<size_t>(j) * n_owned + i;
        const val_t other_energy = energy_in[conn_idx[off]];
        total_flux += (other_energy - this_energy) * coeff * conn_flux[off] * 0.25;
    }

    // Update element state
    energy_out[i] = this_energy + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// Block partition of [0, n_elems) over ranks: first (n % p) ranks get one extra
static inline int64_t partitionBegin(const int64_t n, const int p, const int r) {
    const int64_t base = n / p;
    const int64_t rem = n % p;
    return r * base + std::min<int64_t>(r, rem);
}

static inline int ownerOf(const int64_t g, const int64_t n, const int p) {
    const int64_t base = n / p;
    const int64_t rem = n % p;
    const int64_t cutoff = (base + 1) * rem;
    if (base == 0) return static_cast<int>(g);
    if (g < cutoff) return static_cast<int>(g / (base + 1));
    return static_cast<int>(rem + (g - cutoff) / base);
}

// Hybrid MPI+OpenMP+CUDA simulation context. Each rank owns a contiguous block
// of elements, keeps its share on the local GPU in SoA layout, and exchanges
// halo energies with neighboring owners every iteration. Interior elements are
// updated concurrently with the halo pack/exchange/unpack.
struct HybridSim {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, nprocs = 1;
    int64_t n_elems = 0, begin = 0, n_owned = 0;
    int n_halo = 0, n_send = 0;
    int n_interior = 0, n_boundary = 0;

    // Per-rank halo exchange bookkeeping (host)
    std::vector<int> recv_counts, recv_offsets;  // halo entries per owner rank
    std::vector<int> send_counts, send_offsets;  // entries requested from us

    // Device data
    val_t *d_transfer = nullptr, *d_external = nullptr;
    int *d_ncon = nullptr, *d_conn_idx = nullptr;
    val_t *d_conn_flux = nullptr;
    val_t *d_energy_a = nullptr, *d_energy_b = nullptr;   // n_owned + n_halo
    val_t *d_flux_a = nullptr, *d_flux_b = nullptr;       // n_owned
    val_t *d_energy_in = nullptr, *d_energy_out = nullptr;
    val_t *d_flux_in = nullptr, *d_flux_out = nullptr;
    int *d_interior = nullptr, *d_boundary = nullptr;
    int *d_send_idx = nullptr;
    val_t *d_sendbuf = nullptr;
    val_t *h_sendbuf = nullptr, *h_recvbuf = nullptr;     // pinned

    cudaStream_t s_comp = nullptr, s_comm = nullptr;
    cudaEvent_t ev_halo = nullptr, ev_upd = nullptr;

    void setup(const World& world) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nprocs);

        // Bind each rank to a GPU based on its node-local rank
        MPI_Comm node_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_devices));
        CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

        n_elems = static_cast<int64_t>(world.elements_static.size());
        begin = partitionBegin(n_elems, nprocs, rank);
        n_owned = partitionBegin(n_elems, nprocs, rank + 1) - begin;

        // Collect remote neighbor indices referenced by owned elements
        std::vector<int64_t> halo_globals;
        for (int64_t i = 0; i < n_owned; ++i) {
            const ElementStatic& es = world.elements_static[begin + i];
            for (idx_t j = 0; j < es.num_connections; ++j) {
                const int64_t g = static_cast<int64_t>(es.connected_idx[j]);
                if (g < begin || g >= begin + n_owned) halo_globals.push_back(g);
            }
        }
        std::sort(halo_globals.begin(), halo_globals.end());
        halo_globals.erase(std::unique(halo_globals.begin(), halo_globals.end()),
                           halo_globals.end());
        n_halo = static_cast<int>(halo_globals.size());

        // Sorted halo indices are grouped by owner rank (contiguous ownership)
        recv_counts.assign(nprocs, 0);
        for (const int64_t g : halo_globals) recv_counts[ownerOf(g, n_elems, nprocs)]++;
        recv_offsets.assign(nprocs + 1, 0);
        for (int r = 0; r < nprocs; ++r) recv_offsets[r + 1] = recv_offsets[r] + recv_counts[r];

        // Tell each owner which of its elements we need; the indices we receive
        // in turn form our send list, in the order the requester expects.
        send_counts.assign(nprocs, 0);
        MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm);
        send_offsets.assign(nprocs + 1, 0);
        for (int r = 0; r < nprocs; ++r) send_offsets[r + 1] = send_offsets[r] + send_counts[r];
        n_send = send_offsets[nprocs];

        std::vector<int64_t> send_globals(n_send);
        MPI_Alltoallv(halo_globals.data(), recv_counts.data(), recv_offsets.data(),
                      MPI_INT64_T, send_globals.data(), send_counts.data(),
                      send_offsets.data(), MPI_INT64_T, comm);

        std::vector<int> send_local(n_send);
        for (int i = 0; i < n_send; ++i) {
            send_local[i] = static_cast<int>(send_globals[i] - begin);
        }

        // Global index -> local index (owned block first, then sorted halo)
        auto toLocal = [&](int64_t g) -> int {
            if (g >= begin && g < begin + n_owned) return static_cast<int>(g - begin);
            const auto it = std::lower_bound(halo_globals.begin(), halo_globals.end(), g);
            return static_cast<int>(n_owned + (it - halo_globals.begin()));
        };

        // Build SoA host mirrors (connectivity transposed for coalesced loads)
        std::vector<val_t> h_transfer(n_owned), h_external(n_owned);
        std::vector<int> h_ncon(n_owned);
        std::vector<int> h_conn_idx(static_cast<size_t>(MAX_CONNECTIONS) * n_owned);
        std::vector<val_t> h_conn_flux(static_cast<size_t>(MAX_CONNECTIONS) * n_owned);
        std::vector<val_t> h_energy(n_owned + n_halo, 0.0), h_flux(n_owned);
        std::vector<uint8_t> is_boundary(n_owned);

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_owned; ++i) {
            const ElementStatic& es = world.elements_static[begin + i];
            const Material& mat = world.materials[es.material_idx];
            h_transfer[i] = mat.transfer_coeff;
            h_external[i] = mat.external_flow;
            h_ncon[i] = static_cast<int>(es.num_connections);
            uint8_t bnd = 0;
            for (idx_t j = 0; j < es.num_connections; ++j) {
                const int64_t g = static_cast<int64_t>(es.connected_idx[j]);
                const int lidx = toLocal(g);
                h_conn_idx[j * n_owned + i] = lidx;
                h_conn_flux[j * n_owned + i] = es.connected_flux[j];
                if (lidx >= n_owned) bnd = 1;
            }
            is_boundary[i] = bnd;
            h_energy[i] = world.elements_dynamic[begin + i].current_energy;
            h_flux[i] = world.elements_dynamic[begin + i].total_flux;
        }

        std::vector<int> interior_list, boundary_list;
        interior_list.reserve(n_owned);
        for (int64_t i = 0; i < n_owned; ++i) {
            (is_boundary[i] ? boundary_list : interior_list).push_back(static_cast<int>(i));
        }
        n_interior = static_cast<int>(interior_list.size());
        n_boundary = static_cast<int>(boundary_list.size());

        // Device allocations and uploads
        const size_t conn_sz = static_cast<size_t>(MAX_CONNECTIONS) * n_owned;
        CUDA_CHECK(cudaMalloc(&d_transfer, n_owned * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_external, n_owned * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_ncon, n_owned * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_conn_idx, conn_sz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_conn_flux, conn_sz * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_energy_a, (n_owned + n_halo) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_energy_b, (n_owned + n_halo) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_flux_a, n_owned * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_flux_b, n_owned * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d_interior, std::max(n_interior, 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_boundary, std::max(n_boundary, 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_send_idx, std::max(n_send, 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sendbuf, std::max(n_send, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&h_sendbuf, std::max(n_send, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&h_recvbuf, std::max(n_halo, 1) * sizeof(val_t)));

        CUDA_CHECK(cudaMemcpy(d_transfer, h_transfer.data(), n_owned * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_external, h_external.data(), n_owned * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ncon, h_ncon.data(), n_owned * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conn_idx, h_conn_idx.data(), conn_sz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux.data(), conn_sz * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_energy_a, h_energy.data(), (n_owned + n_halo) * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_energy_b, h_energy.data(), (n_owned + n_halo) * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_flux_a, h_flux.data(), n_owned * sizeof(val_t), cudaMemcpyHostToDevice));
        if (n_interior > 0)
            CUDA_CHECK(cudaMemcpy(d_interior, interior_list.data(), n_interior * sizeof(int), cudaMemcpyHostToDevice));
        if (n_boundary > 0)
            CUDA_CHECK(cudaMemcpy(d_boundary, boundary_list.data(), n_boundary * sizeof(int), cudaMemcpyHostToDevice));
        if (n_send > 0)
            CUDA_CHECK(cudaMemcpy(d_send_idx, send_local.data(), n_send * sizeof(int), cudaMemcpyHostToDevice));

        d_energy_in = d_energy_a;
        d_energy_out = d_energy_b;
        d_flux_in = d_flux_a;
        d_flux_out = d_flux_b;

        CUDA_CHECK(cudaStreamCreate(&s_comp));
        CUDA_CHECK(cudaStreamCreate(&s_comm));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_halo, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_upd, cudaEventDisableTiming));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void run(const int n_iters) {
        constexpr int BLOCK = 256;
        const bool exchange = (nprocs > 1);
        std::vector<MPI_Request> reqs;

        for (int iter = 0; iter < n_iters; ++iter) {
            if (exchange) {
                // Pack current boundary energies once the previous update is done
                CUDA_CHECK(cudaStreamWaitEvent(s_comm, ev_upd, 0));
                if (n_send > 0) {
                    packKernel<<<(n_send + BLOCK - 1) / BLOCK, BLOCK, 0, s_comm>>>(
                        d_energy_in, d_send_idx, d_sendbuf, n_send);
                    CUDA_CHECK(cudaMemcpyAsync(h_sendbuf, d_sendbuf,
                                               n_send * sizeof(val_t),
                                               cudaMemcpyDeviceToHost, s_comm));
                }
            }

            // Interior elements never touch the halo region: overlap with comm
            if (n_interior > 0) {
                updateKernel<<<(n_interior + BLOCK - 1) / BLOCK, BLOCK, 0, s_comp>>>(
                    d_interior, n_interior, static_cast<int>(n_owned),
                    d_transfer, d_external, d_ncon, d_conn_idx, d_conn_flux,
                    d_energy_in, d_flux_in, d_energy_out, d_flux_out);
            }

            if (exchange) {
                CUDA_CHECK(cudaStreamSynchronize(s_comm));
                reqs.clear();
                for (int r = 0; r < nprocs; ++r) {
                    if (recv_counts[r] > 0) {
                        reqs.emplace_back();
                        MPI_Irecv(h_recvbuf + recv_offsets[r], recv_counts[r],
                                  MPI_DOUBLE, r, 0, comm, &reqs.back());
                    }
                }
                for (int r = 0; r < nprocs; ++r) {
                    if (send_counts[r] > 0) {
                        reqs.emplace_back();
                        MPI_Isend(h_sendbuf + send_offsets[r], send_counts[r],
                                  MPI_DOUBLE, r, 0, comm, &reqs.back());
                    }
                }
                MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

                if (n_halo > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(d_energy_in + n_owned, h_recvbuf,
                                               n_halo * sizeof(val_t),
                                               cudaMemcpyHostToDevice, s_comm));
                    CUDA_CHECK(cudaEventRecord(ev_halo, s_comm));
                    CUDA_CHECK(cudaStreamWaitEvent(s_comp, ev_halo, 0));
                }
            }

            if (n_boundary > 0) {
                updateKernel<<<(n_boundary + BLOCK - 1) / BLOCK, BLOCK, 0, s_comp>>>(
                    d_boundary, n_boundary, static_cast<int>(n_owned),
                    d_transfer, d_external, d_ncon, d_conn_idx, d_conn_flux,
                    d_energy_in, d_flux_in, d_energy_out, d_flux_out);
            }
            CUDA_CHECK(cudaEventRecord(ev_upd, s_comp));

            // Swap buffers
            std::swap(d_energy_in, d_energy_out);
            std::swap(d_flux_in, d_flux_out);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    // Gather final element states from all GPUs into world.elements_dynamic on
    // rank 0 (other ranks receive only their own block back into place).
    void gather(World& world) {
        std::vector<val_t> h_energy(n_owned), h_flux(n_owned);
        if (n_owned > 0) {
            CUDA_CHECK(cudaMemcpy(h_energy.data(), d_energy_in, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_flux.data(), d_flux_in, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        std::vector<ElementDynamic> local(n_owned);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_owned; ++i) {
            local[i].current_energy = h_energy[i];
            local[i].total_flux = h_flux[i];
        }

        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = static_cast<int>(2 * (partitionBegin(n_elems, nprocs, r + 1) -
                                              partitionBegin(n_elems, nprocs, r)));
            displs[r] = static_cast<int>(2 * partitionBegin(n_elems, nprocs, r));
        }
        MPI_Gatherv(local.data(), static_cast<int>(2 * n_owned), MPI_DOUBLE,
                    world.elements_dynamic.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        if (rank != 0) {
            std::copy(local.begin(), local.end(), world.elements_dynamic.begin() + begin);
        }
    }

    void cleanup() {
        cudaEventDestroy(ev_halo);
        cudaEventDestroy(ev_upd);
        cudaStreamDestroy(s_comp);
        cudaStreamDestroy(s_comm);
        cudaFreeHost(h_sendbuf);
        cudaFreeHost(h_recvbuf);
        cudaFree(d_sendbuf);
        cudaFree(d_send_idx);
        cudaFree(d_boundary);
        cudaFree(d_interior);
        cudaFree(d_flux_b);
        cudaFree(d_flux_a);
        cudaFree(d_energy_b);
        cudaFree(d_energy_a);
        cudaFree(d_conn_flux);
        cudaFree(d_conn_idx);
        cudaFree(d_ncon);
        cudaFree(d_external);
        cudaFree(d_transfer);
    }
};

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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    // XOR combining is commutative, so the parallel reduction is bit-identical
    #pragma omp parallel for schedule(static) reduction(^:hash)
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid parallelism: %d MPI ranks x %d OpenMP threads + CUDA\n",
               nprocs, omp_get_max_threads());
        printf("\n");

        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root);

    if (rank == 0) {
        // Calculate memory usage
        const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        // Run simulation
        printf("Running simulation...\n");
    }

    HybridSim sim;
    sim.setup(world);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    sim.run(n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    sim.gather(world);
    sim.cleanup();

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

        // Validation
        if (validate) {
            bool valid = validateResults(world);
            if (!valid) {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
