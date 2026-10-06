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
    std::vector<ElementDynamic> elements_dynamic;   // locally owned elements only
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries.
// Each MPI rank only builds the elements it owns: global indices [elem_begin, elem_end).
// Connectivity is stored with global element indices.
void buildSquare2D(World& world, const int n_elems_root, const idx_t elem_begin, const idx_t elem_end) {
    const idx_t n_local = elem_end - elem_begin;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate elements
    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);

    const idx_t nr = static_cast<idx_t>(n_elems_root);
    const idx_t last = nr - 1;

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (idx_t l = 0; l < n_local; ++l) {
        const idx_t idx = elem_begin + l;
        const long long x = static_cast<long long>(idx / nr);
        const long long y = static_cast<long long>(idx % nr);
        ElementStatic& elem = world.elements_static[l];
        elem.material_idx = DEFAULT_MAT_ID;
        elem.num_connections = 0;
        world.elements_dynamic[l].current_energy = 0.0;
        world.elements_dynamic[l].total_flux = 0.0;

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int n = 0; n < 4; ++n) {
            const long long nx = x + offsets[n][0];
            const long long ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * nr + static_cast<idx_t>(ny);
                elem.connected_idx[elem.num_connections] = neighbor_idx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }

        // Set corner elements as inflow/outflow to create interesting dynamics
        // (applied in the same order as the reference so coinciding corners match)
        if (idx == 0 * nr + 0) elem.material_idx = INFLOW_MAT_ID;
        if (idx == 0 * nr + last) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * nr + 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (idx == last * nr + last) elem.material_idx = INFLOW_MAT_ID;
    }
}

// ---------------------------------------------------------------------------
// Distributed / device data structures
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

struct NeighborComm {
    int rank;
    int send_offset, send_count;
    int recv_offset, recv_count;
};

// Local (per-rank) device representation of the mesh.
// Local element numbering: [0, n_interior) interior elements (all neighbors local),
// [n_interior, n_local) boundary elements (some neighbor owned by another rank),
// [n_local, n_local + n_halo) halo copies of remote elements (energy only).
struct DeviceMesh {
    int n_local = 0, n_interior = 0, n_halo = 0, max_conn = 0;
    std::vector<idx_t> perm;            // local index -> global index - elem_begin
    std::vector<NeighborComm> neighbors;
    int total_send = 0, total_recv = 0;

    uint8_t* d_num_conn = nullptr;
    uint8_t* d_mat = nullptr;
    int* d_conn_idx = nullptr;          // [j * n_local + i]
    void* d_conn_flux = nullptr;        // [j * n_local + i], double or (lossless) float
    bool flux_as_float = false;
    val_t* d_mat_coeff = nullptr;
    val_t* d_mat_ext = nullptr;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    int* d_send_idx = nullptr;
    val_t* d_send_buf = nullptr;
    val_t* h_send_buf = nullptr;        // pinned
    val_t* h_recv_buf = nullptr;        // pinned
};

// Energy update kernel for local elements [first, first + count)
// FluxT is float only when all connection flux coefficients are exactly representable.
template <typename FluxT>
__global__ void __launch_bounds__(256)
updateKernel(const int first, const int count, const int stride,
             const uint8_t* __restrict__ num_conn, const uint8_t* __restrict__ mat_idx,
             const int* __restrict__ conn_idx, const FluxT* __restrict__ conn_flux,
             const val_t* __restrict__ mat_coeff, const val_t* __restrict__ mat_ext,
             const val_t* __restrict__ energy_in, const val_t* __restrict__ flux_in,
             val_t* __restrict__ energy_out, val_t* __restrict__ flux_out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int i = first + t;

    const int m = mat_idx[i];
    const val_t coeff = mat_coeff[m];
    const val_t e = energy_in[i];

    // Start with external flow
    val_t total_flux = mat_ext[m];

    // Add flux from all connected elements (same order as the reference)
    const int nc = num_conn[i];
    for (int j = 0; j < nc; ++j) {
        const int nb = __ldg(&conn_idx[j * stride + i]);
        const val_t cf = static_cast<val_t>(__ldg(&conn_flux[j * stride + i]));
        total_flux += (__ldg(&energy_in[nb]) - e) * coeff * cf * 0.25;
    }

    energy_out[i] = e + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

__global__ void packKernel(const int count, const int* __restrict__ idx,
                           const val_t* __restrict__ energy, val_t* __restrict__ buf) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) buf[t] = energy[idx[t]];
}

static inline int ownerOf(const std::vector<idx_t>& offsets, idx_t g) {
    return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), g) - offsets.begin()) - 1;
}

// Build local numbering, halo communication pattern, and upload to the device.
void setupDeviceMesh(const World& world, DeviceMesh& dm, const std::vector<idx_t>& offsets,
                     const int rank, const int nranks) {
    const idx_t elem_begin = offsets[rank];
    const idx_t elem_end = offsets[rank + 1];
    const idx_t n_local = elem_end - elem_begin;
    if (n_local > static_cast<idx_t>(std::numeric_limits<int>::max() / 2)) {
        fprintf(stderr, "Too many elements per rank; use more MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (world.materials.size() > 256) {
        fprintf(stderr, "Too many materials\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    dm.n_local = static_cast<int>(n_local);

    // Classify interior vs boundary elements
    std::vector<uint8_t> is_boundary(n_local, 0);
    int max_conn = 0;
    #pragma omp parallel for schedule(static) reduction(max : max_conn)
    for (idx_t l = 0; l < n_local; ++l) {
        const ElementStatic& es = world.elements_static[l];
        max_conn = std::max(max_conn, static_cast<int>(es.num_connections));
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            if (g < elem_begin || g >= elem_end) { is_boundary[l] = 1; break; }
        }
    }
    dm.max_conn = max_conn;

    dm.perm.resize(n_local);
    std::vector<int> old2new(n_local);
    std::vector<idx_t> halo_globals;
    {
        int k = 0;
        for (idx_t l = 0; l < n_local; ++l) if (!is_boundary[l]) { dm.perm[k] = l; old2new[l] = k++; }
        dm.n_interior = k;
        for (idx_t l = 0; l < n_local; ++l) if (is_boundary[l]) {
            dm.perm[k] = l; old2new[l] = k++;
            const ElementStatic& es = world.elements_static[l];
            for (idx_t j = 0; j < es.num_connections; ++j) {
                const idx_t g = es.connected_idx[j];
                if (g < elem_begin || g >= elem_end) halo_globals.push_back(g);
            }
        }
    }
    // Sorted unique halo list -> grouped by owner rank (offsets are monotonic)
    std::sort(halo_globals.begin(), halo_globals.end());
    halo_globals.erase(std::unique(halo_globals.begin(), halo_globals.end()), halo_globals.end());
    dm.n_halo = static_cast<int>(halo_globals.size());

    // Exchange request lists
    std::vector<int> recv_counts(nranks, 0), send_counts(nranks, 0);
    for (idx_t g : halo_globals) recv_counts[ownerOf(offsets, g)]++;
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rdispl(nranks, 0), sdispl(nranks, 0);
    for (int r = 1; r < nranks; ++r) {
        rdispl[r] = rdispl[r - 1] + recv_counts[r - 1];
        sdispl[r] = sdispl[r - 1] + send_counts[r - 1];
    }
    dm.total_recv = dm.n_halo;
    dm.total_send = sdispl[nranks - 1] + send_counts[nranks - 1];
    std::vector<unsigned long long> requested(dm.total_send);
    std::vector<unsigned long long> halo_ull(halo_globals.begin(), halo_globals.end());
    MPI_Alltoallv(halo_ull.data(), recv_counts.data(), rdispl.data(), MPI_UNSIGNED_LONG_LONG,
                  requested.data(), send_counts.data(), sdispl.data(), MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);
    for (int r = 0; r < nranks; ++r) {
        if (recv_counts[r] > 0 || send_counts[r] > 0) {
            dm.neighbors.push_back(NeighborComm{r, sdispl[r], send_counts[r], rdispl[r], recv_counts[r]});
        }
    }
    std::vector<int> send_idx(dm.total_send);
    for (int k = 0; k < dm.total_send; ++k) send_idx[k] = old2new[requested[k] - elem_begin];

    // Device-friendly connectivity arrays in local numbering
    const int nl = dm.n_local;
    const int mc = std::max(max_conn, 1);
    std::vector<uint8_t> h_num_conn(nl), h_mat(nl);
    std::vector<int> h_conn_idx(static_cast<size_t>(mc) * nl, 0);
    std::vector<val_t> h_conn_flux(static_cast<size_t>(mc) * nl, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nl; ++i) {
        const ElementStatic& es = world.elements_static[dm.perm[i]];
        h_num_conn[i] = static_cast<uint8_t>(es.num_connections);
        h_mat[i] = static_cast<uint8_t>(es.material_idx);
        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t g = es.connected_idx[j];
            int li;
            if (g >= elem_begin && g < elem_end) {
                li = old2new[g - elem_begin];
            } else {
                li = nl + static_cast<int>(std::lower_bound(halo_globals.begin(), halo_globals.end(), g) -
                                           halo_globals.begin());
            }
            h_conn_idx[j * static_cast<size_t>(nl) + i] = li;
            h_conn_flux[j * static_cast<size_t>(nl) + i] = es.connected_flux[j];
        }
    }
    std::vector<val_t> h_energy(nl + dm.n_halo, 0.0), h_flux(nl);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nl; ++i) {
        h_energy[i] = world.elements_dynamic[dm.perm[i]].current_energy;
        h_flux[i] = world.elements_dynamic[dm.perm[i]].total_flux;
    }
    std::vector<val_t> h_coeff, h_ext;
    for (const Material& m : world.materials) { h_coeff.push_back(m.transfer_coeff); h_ext.push_back(m.external_flow); }

    const size_t ne = static_cast<size_t>(nl) + dm.n_halo;
    CUDA_CHECK(cudaMalloc(&dm.d_num_conn, std::max(nl, 1)));
    CUDA_CHECK(cudaMalloc(&dm.d_mat, std::max(nl, 1)));
    CUDA_CHECK(cudaMalloc(&dm.d_conn_idx, std::max<size_t>(h_conn_idx.size(), 1) * sizeof(int)));
    // Store flux coefficients in single precision if that is lossless (saves bandwidth)
    bool all_float = true;
    #pragma omp parallel for schedule(static) reduction(&& : all_float)
    for (size_t k = 0; k < h_conn_flux.size(); ++k) {
        all_float = all_float && (static_cast<val_t>(static_cast<float>(h_conn_flux[k])) == h_conn_flux[k]);
    }
    dm.flux_as_float = all_float;
    const size_t flux_elem_size = all_float ? sizeof(float) : sizeof(val_t);
    CUDA_CHECK(cudaMalloc(&dm.d_conn_flux, std::max<size_t>(h_conn_flux.size(), 1) * flux_elem_size));
    CUDA_CHECK(cudaMalloc(&dm.d_mat_coeff, h_coeff.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dm.d_mat_ext, h_ext.size() * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dm.d_energy[b], std::max<size_t>(ne, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dm.d_flux[b], std::max(nl, 1) * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMalloc(&dm.d_send_idx, std::max(dm.total_send, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dm.d_send_buf, std::max(dm.total_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&dm.h_send_buf, std::max(dm.total_send, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&dm.h_recv_buf, std::max(dm.total_recv, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(dm.d_num_conn, h_num_conn.data(), nl, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.d_mat, h_mat.data(), nl, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.d_conn_idx, h_conn_idx.data(), h_conn_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (all_float) {
        std::vector<float> h_conn_flux_f(h_conn_flux.size());
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < h_conn_flux.size(); ++k) h_conn_flux_f[k] = static_cast<float>(h_conn_flux[k]);
        CUDA_CHECK(cudaMemcpy(dm.d_conn_flux, h_conn_flux_f.data(), h_conn_flux_f.size() * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(dm.d_conn_flux, h_conn_flux.data(), h_conn_flux.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dm.d_mat_coeff, h_coeff.data(), h_coeff.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm.d_mat_ext, h_ext.data(), h_ext.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMemcpy(dm.d_energy[b], h_energy.data(), ne * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm.d_flux[b], h_flux.data(), nl * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (dm.total_send > 0) {
        CUDA_CHECK(cudaMemcpy(dm.d_send_idx, send_idx.data(), dm.total_send * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeDeviceMesh(DeviceMesh& dm) {
    cudaFree(dm.d_num_conn); cudaFree(dm.d_mat); cudaFree(dm.d_conn_idx); cudaFree(dm.d_conn_flux);
    cudaFree(dm.d_mat_coeff); cudaFree(dm.d_mat_ext);
    for (int b = 0; b < 2; ++b) { cudaFree(dm.d_energy[b]); cudaFree(dm.d_flux[b]); }
    cudaFree(dm.d_send_idx); cudaFree(dm.d_send_buf);
    cudaFreeHost(dm.h_send_buf); cudaFreeHost(dm.h_recv_buf);
}

static inline void launchUpdate(const DeviceMesh& dm, int first, int count, int cur, cudaStream_t s) {
    if (count <= 0) return;
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    if (dm.flux_as_float) {
        updateKernel<float><<<blocks, threads, 0, s>>>(
            first, count, dm.n_local, dm.d_num_conn, dm.d_mat, dm.d_conn_idx,
            static_cast<const float*>(dm.d_conn_flux), dm.d_mat_coeff, dm.d_mat_ext,
            dm.d_energy[cur], dm.d_flux[cur], dm.d_energy[1 - cur], dm.d_flux[1 - cur]);
    } else {
        updateKernel<val_t><<<blocks, threads, 0, s>>>(
            first, count, dm.n_local, dm.d_num_conn, dm.d_mat, dm.d_conn_idx,
            static_cast<const val_t*>(dm.d_conn_flux), dm.d_mat_coeff, dm.d_mat_ext,
            dm.d_energy[cur], dm.d_flux[cur], dm.d_energy[1 - cur], dm.d_flux[1 - cur]);
    }
}

// Run simulation for n_iters iterations; returns index of buffer holding the final state
int runSimulation(DeviceMesh& dm, const int n_iters) {
    int cur = 0;
    const bool has_comm = !dm.neighbors.empty();

    cudaStream_t s_comp, s_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_comm, cudaStreamNonBlocking));
    cudaEvent_t ev_comp, ev_comm;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_comp, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_comm, cudaEventDisableTiming));

    std::vector<MPI_Request> reqs(2 * dm.neighbors.size());
    const int n_boundary = dm.n_local - dm.n_interior;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (!has_comm) {
            launchUpdate(dm, 0, dm.n_local, cur, s_comp);
            cur = 1 - cur;
            continue;
        }

        // Post receives for halo values
        int nreq = 0;
        for (const NeighborComm& nb : dm.neighbors) {
            if (nb.recv_count > 0) {
                MPI_Irecv(dm.h_recv_buf + nb.recv_offset, nb.recv_count, MPI_DOUBLE, nb.rank, 0,
                          MPI_COMM_WORLD, &reqs[nreq++]);
            }
        }

        // Pack and download boundary values to send
        if (dm.total_send > 0) {
            packKernel<<<(dm.total_send + 255) / 256, 256, 0, s_comm>>>(
                dm.total_send, dm.d_send_idx, dm.d_energy[cur], dm.d_send_buf);
            CUDA_CHECK(cudaMemcpyAsync(dm.h_send_buf, dm.d_send_buf, dm.total_send * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, s_comm));
        }

        // Interior update overlaps with halo exchange
        launchUpdate(dm, 0, dm.n_interior, cur, s_comp);

        CUDA_CHECK(cudaStreamSynchronize(s_comm));
        for (const NeighborComm& nb : dm.neighbors) {
            if (nb.send_count > 0) {
                MPI_Isend(dm.h_send_buf + nb.send_offset, nb.send_count, MPI_DOUBLE, nb.rank, 0,
                          MPI_COMM_WORLD, &reqs[nreq++]);
            }
        }
        MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);

        // Upload halo and update boundary elements
        if (dm.total_recv > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dm.d_energy[cur] + dm.n_local, dm.h_recv_buf,
                                       dm.total_recv * sizeof(val_t), cudaMemcpyHostToDevice, s_comm));
        }
        launchUpdate(dm, dm.n_interior, n_boundary, cur, s_comm);

        // Both halves must complete before the next iteration uses the new state
        CUDA_CHECK(cudaEventRecord(ev_comp, s_comp));
        CUDA_CHECK(cudaEventRecord(ev_comm, s_comm));
        CUDA_CHECK(cudaStreamWaitEvent(s_comm, ev_comp, 0));
        CUDA_CHECK(cudaStreamWaitEvent(s_comp, ev_comm, 0));

        // Swap buffers
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaStreamSynchronize(s_comp));
    CUDA_CHECK(cudaStreamSynchronize(s_comm));
    CUDA_CHECK(cudaGetLastError());

    cudaEventDestroy(ev_comp);
    cudaEventDestroy(ev_comm);
    cudaStreamDestroy(s_comp);
    cudaStreamDestroy(s_comm);
    return cur;
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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
// XOR-combination is order independent, so each rank hashes its own elements
// (using global indices) and the partial hashes are XOR-reduced.
uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux, const idx_t elem_begin) {
    uint64_t hash = 0;
    const long long n = static_cast<long long>(energy.size());
    #pragma omp parallel for schedule(static) reduction(^ : hash)
    for (long long l = 0; l < n; ++l) {
        const uint64_t i = elem_begin + static_cast<uint64_t>(l);
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &energy[l], sizeof(e_bits));
        memcpy(&f_bits, &flux[l], sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather a distributed array (in global element order) to rank 0
std::vector<val_t> gatherToRoot(const std::vector<val_t>& local, const std::vector<idx_t>& offsets,
                                const int rank, const int nranks) {
    std::vector<val_t> global;
    if (rank == 0) {
        global.resize(offsets[nranks]);
        std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < nranks; ++r) {
            const idx_t cnt = offsets[r + 1] - offsets[r];
            if (cnt > 0) {
                MPI_Recv(global.data() + offsets[r], static_cast<int>(cnt), MPI_DOUBLE, r, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (!local.empty()) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    return global;
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
        CUDA_CHECK(cudaFree(nullptr));  // initialize context
    }

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
    const idx_t n_elems_global = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);
    
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

    // Block partition of elements across ranks
    std::vector<idx_t> offsets(nranks + 1);
    for (int r = 0; r <= nranks; ++r) {
        offsets[r] = static_cast<idx_t>((static_cast<unsigned __int128>(n_elems_global) * r) / nranks);
    }
    const idx_t elem_begin = offsets[rank];
    const idx_t elem_end = offsets[rank + 1];

    World world;
    buildSquare2D(world, n_elems_root, elem_begin, elem_end);
    
    // Calculate memory usage
    const size_t static_mem = n_elems_global * sizeof(ElementStatic);
    const size_t dynamic_mem = n_elems_global * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Set up local numbering, halo exchange and device data
    DeviceMesh dm;
    setupDeviceMesh(world, dm, offsets, rank, nranks);
    world.elements_static.clear();
    world.elements_static.shrink_to_fit();
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    const int final_buf = runSimulation(dm, n_iters);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Download final local state and restore global element order
    const int nl = dm.n_local;
    std::vector<val_t> tmp_e(nl), tmp_f(nl);
    if (nl > 0) {
        CUDA_CHECK(cudaMemcpy(tmp_e.data(), dm.d_energy[final_buf], nl * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tmp_f.data(), dm.d_flux[final_buf], nl * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    std::vector<val_t> local_energy(nl), local_flux(nl);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nl; ++i) {
        local_energy[dm.perm[i]] = tmp_e[i];
        local_flux[dm.perm[i]] = tmp_f[i];
    }
    freeDeviceMesh(dm);

    // Compute hash for verification
    const uint64_t local_hash = computeHash(local_energy, local_flux, elem_begin);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

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
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Gather final state to rank 0 for output / validation
    int status = 0;
    if (printResults || validate) {
        std::vector<val_t> energy = gatherToRoot(local_energy, offsets, rank, nranks);
        std::vector<val_t> flux;
        if (validate) flux = gatherToRoot(local_flux, offsets, rank, nranks);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(energy, "ElementEnergy");
            }
            // Validation
            if (validate) {
                bool valid = validateResults(energy, flux);
                if (!valid) status = 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Finalize();
    return status;
}
