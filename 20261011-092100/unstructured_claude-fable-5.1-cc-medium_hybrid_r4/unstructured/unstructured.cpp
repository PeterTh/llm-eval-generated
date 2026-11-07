// Unstructured Mesh Energy Transfer Benchmark
//
// Hybrid parallel implementation:
//   * MPI    : 1D domain decomposition of the element index space over ranks
//              (one rank per GPU), generic halo exchange derived from the
//              connectivity, final gather of results to rank 0.
//   * CUDA   : the per-iteration element update runs on the GPU using a
//              structure-of-arrays layout with a transposed connectivity table
//              for coalesced access. Interior elements are updated on one
//              stream while the halo is exchanged and the boundary elements
//              are updated on a second stream (communication/computation
//              overlap).
//   * OpenMP : multithreaded host-side mesh construction, halo/permutation
//              set-up, result un-permutation and hashing.
//
// Numerical semantics are identical to the original serial code: every element
// accumulates its flux contributions in the same order with the same
// expressions, so results are bitwise reproducible across rank counts.

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

// Static connectivity information for each element (reference layout; used for
// the memory report so the output matches the original program).
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

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,     \
                    __LINE__, cudaGetErrorString(err__));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define MPI_CHECK(call)                                                           \
    do {                                                                          \
        int err__ = (call);                                                       \
        if (err__ != MPI_SUCCESS) {                                               \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, \
                    __LINE__, err__);                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Local (per-rank) mesh partition in structure-of-arrays layout
// ---------------------------------------------------------------------------
struct LocalMesh {
    // Global partition
    idx_t n_global = 0;
    idx_t g_begin = 0;   // first global element index owned by this rank
    idx_t g_end = 0;     // one past last global element index owned by this rank
    std::vector<idx_t> offsets;  // size+1 partition offsets

    // Local elements (in *permuted* local order: boundary first, then interior)
    idx_t n_local = 0;
    idx_t n_boundary = 0;  // elements that reference at least one remote element
    idx_t n_halo = 0;      // number of remote elements referenced locally

    // perm[old_local] = new_local  (old_local = global - g_begin)
    std::vector<int32_t> perm;

    // SoA static data, indexed by new local index; connectivity tables are
    // transposed: conn_idx[j * n_local + i]
    std::vector<uint8_t> material_idx;
    std::vector<uint8_t> num_connections;
    std::vector<int32_t> conn_idx;   // local/halo slot indices
    std::vector<val_t> conn_flux;

    // Halo exchange description
    std::vector<int> recv_counts, recv_displs;  // per rank, halo slots are n_local + recv_displs[r] ...
    std::vector<int> send_counts, send_displs;
    std::vector<int32_t> send_local;  // new-local indices to pack, grouped by destination rank

    std::vector<Material> materials;
};

// Owner rank of a global element index
static inline int ownerOf(const std::vector<idx_t>& offsets, idx_t g) {
    return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), g) - offsets.begin()) - 1;
}

// Build a 2D square grid as an unstructured mesh, restricted to this rank's
// partition of the global element index space. This represents computation on
// arbitrarily-shaped geometries; everything after the connectivity generation
// is independent of the grid structure.
static void buildSquare2D(LocalMesh& mesh, const int n_elems_root, const int rank, const int size) {
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * static_cast<idx_t>(n_elems_root);

    // Initialize materials
    mesh.materials.emplace_back(Material{0.8, 0.0});    // Default material
    mesh.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    mesh.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Contiguous block partition of the global index space
    mesh.n_global = n_elems;
    mesh.offsets.resize(size + 1);
    {
        const idx_t base = n_elems / size;
        const idx_t rem = n_elems % size;
        mesh.offsets[0] = 0;
        for (int r = 0; r < size; ++r) {
            mesh.offsets[r + 1] = mesh.offsets[r] + base + (static_cast<idx_t>(r) < rem ? 1 : 0);
        }
    }
    mesh.g_begin = mesh.offsets[rank];
    mesh.g_end = mesh.offsets[rank + 1];
    mesh.n_local = mesh.g_end - mesh.g_begin;
    const idx_t n_local = mesh.n_local;

    // Temporary connectivity with global neighbour indices, old local order
    std::vector<uint8_t> mat(n_local, static_cast<uint8_t>(DEFAULT_MAT_ID));
    std::vector<uint8_t> nconn(n_local, 0);
    std::vector<idx_t> gconn(n_local * MAX_CONNECTIONS);
    std::vector<val_t> gflux(n_local * MAX_CONNECTIONS);

    // Build connectivity: each element connects to its neighbors in 2D grid
    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t idx = mesh.g_begin + li;
        const int x = static_cast<int>(idx / static_cast<idx_t>(n_elems_root));
        const int y = static_cast<int>(idx % static_cast<idx_t>(n_elems_root));

        // Connect to neighbors (up, down, left, right)
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        uint8_t nc = 0;
        for (int n = 0; n < 4; ++n) {
            const int nx = x + offsets[n][0];
            const int ny = y + offsets[n][1];

            // Check if neighbor is within bounds
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const idx_t neighbor_idx = static_cast<idx_t>(nx) * n_elems_root + ny;
                gconn[li * MAX_CONNECTIONS + nc] = neighbor_idx;
                gflux[li * MAX_CONNECTIONS + nc] = 1.0;
                nc++;
            }
        }
        nconn[li] = nc;
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    {
        const idx_t last = static_cast<idx_t>(n_elems_root) - 1;
        const idx_t nr = static_cast<idx_t>(n_elems_root);
        const idx_t corners[4] = {0 * nr + 0, 0 * nr + last,
                                  last * nr + 0, last * nr + last};
        const idx_t corner_mat[4] = {INFLOW_MAT_ID, OUTFLOW_MAT_ID, OUTFLOW_MAT_ID, INFLOW_MAT_ID};
        for (int c = 0; c < 4; ++c) {
            if (corners[c] >= mesh.g_begin && corners[c] < mesh.g_end) {
                mat[corners[c] - mesh.g_begin] = static_cast<uint8_t>(corner_mat[c]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Halo analysis (generic: derived purely from the connectivity)
    // ------------------------------------------------------------------
    std::vector<idx_t> remote;  // sorted unique remote global indices
    std::vector<uint8_t> is_boundary(n_local, 0);
    #pragma omp parallel
    {
        std::vector<idx_t> mine;
        #pragma omp for schedule(static) nowait
        for (idx_t li = 0; li < n_local; ++li) {
            for (int j = 0; j < nconn[li]; ++j) {
                const idx_t g = gconn[li * MAX_CONNECTIONS + j];
                if (g < mesh.g_begin || g >= mesh.g_end) {
                    mine.push_back(g);
                    is_boundary[li] = 1;
                }
            }
        }
        #pragma omp critical
        remote.insert(remote.end(), mine.begin(), mine.end());
    }
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());
    mesh.n_halo = remote.size();

    // Halo slots: n_local + k for remote[k]; remote[] is sorted, so entries
    // are naturally grouped by owning rank.
    mesh.recv_counts.assign(size, 0);
    mesh.recv_displs.assign(size, 0);
    for (idx_t k = 0; k < mesh.n_halo; ++k) {
        mesh.recv_counts[ownerOf(mesh.offsets, remote[k])]++;
    }
    for (int r = 1; r < size; ++r) {
        mesh.recv_displs[r] = mesh.recv_displs[r - 1] + mesh.recv_counts[r - 1];
    }

    // Tell every rank which of its elements we need; learn what they need from us
    mesh.send_counts.assign(size, 0);
    mesh.send_displs.assign(size, 0);
    MPI_CHECK(MPI_Alltoall(mesh.recv_counts.data(), 1, MPI_INT, mesh.send_counts.data(), 1, MPI_INT,
                           MPI_COMM_WORLD));
    for (int r = 1; r < size; ++r) {
        mesh.send_displs[r] = mesh.send_displs[r - 1] + mesh.send_counts[r - 1];
    }
    const int n_send = mesh.send_displs[size - 1] + mesh.send_counts[size - 1];
    std::vector<idx_t> send_global(n_send);
    MPI_CHECK(MPI_Alltoallv(remote.data(), mesh.recv_counts.data(), mesh.recv_displs.data(), MPI_UINT64_T,
                            send_global.data(), mesh.send_counts.data(), mesh.send_displs.data(),
                            MPI_UINT64_T, MPI_COMM_WORLD));

    // ------------------------------------------------------------------
    // Local permutation: boundary elements first, interior afterwards, so
    // that both groups are contiguous ranges for the kernels.
    // ------------------------------------------------------------------
    mesh.perm.resize(n_local);
    {
        idx_t nb = 0;
        for (idx_t li = 0; li < n_local; ++li) nb += is_boundary[li];
        mesh.n_boundary = nb;
        idx_t next_b = 0, next_i = nb;
        for (idx_t li = 0; li < n_local; ++li) {
            mesh.perm[li] = static_cast<int32_t>(is_boundary[li] ? next_b++ : next_i++);
        }
    }

    mesh.send_local.resize(n_send);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n_send; ++k) {
        mesh.send_local[k] = mesh.perm[send_global[k] - mesh.g_begin];
    }

    // ------------------------------------------------------------------
    // Final SoA tables in permuted order, with local/halo slot indices
    // ------------------------------------------------------------------
    mesh.material_idx.resize(n_local);
    mesh.num_connections.resize(n_local);
    mesh.conn_idx.assign(n_local * MAX_CONNECTIONS, 0);
    mesh.conn_flux.assign(n_local * MAX_CONNECTIONS, 0.0);

    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const idx_t ni = mesh.perm[li];
        mesh.material_idx[ni] = mat[li];
        mesh.num_connections[ni] = nconn[li];
        for (int j = 0; j < nconn[li]; ++j) {
            const idx_t g = gconn[li * MAX_CONNECTIONS + j];
            int32_t slot;
            if (g >= mesh.g_begin && g < mesh.g_end) {
                slot = mesh.perm[g - mesh.g_begin];
            } else {
                const idx_t k = std::lower_bound(remote.begin(), remote.end(), g) - remote.begin();
                slot = static_cast<int32_t>(n_local + k);
            }
            mesh.conn_idx[static_cast<idx_t>(j) * n_local + ni] = slot;
            mesh.conn_flux[static_cast<idx_t>(j) * n_local + ni] = gflux[li * MAX_CONNECTIONS + j];
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Compute energy flux between two elements (same expression as the original)
__device__ __forceinline__ val_t computeFlux(val_t transfer_coeff, val_t this_energy,
                                             val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

__global__ void updateKernel(const idx_t begin, const idx_t end, const idx_t stride,
                             const val_t* __restrict__ energy_cur,
                             const val_t* __restrict__ tflux_cur,
                             val_t* __restrict__ energy_next,
                             val_t* __restrict__ tflux_next,
                             const uint8_t* __restrict__ material_idx,
                             const uint8_t* __restrict__ num_connections,
                             const int32_t* __restrict__ conn_idx,
                             const val_t* __restrict__ conn_flux,
                             const Material* __restrict__ materials) {
    const idx_t i = begin + static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const val_t e = energy_cur[i];
    const Material mat = materials[material_idx[i]];
    const int nc = num_connections[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements (same order as the original)
    for (int j = 0; j < nc; ++j) {
        const int32_t nb = conn_idx[static_cast<idx_t>(j) * stride + i];
        const val_t f = conn_flux[static_cast<idx_t>(j) * stride + i];
        total_flux += computeFlux(mat.transfer_coeff, e, f, __ldg(&energy_cur[nb]));
    }

    // Update element state
    energy_next[i] = e + total_flux;
    tflux_next[i] = tflux_cur[i] + fabs(total_flux);
}

__global__ void packKernel(const int n, const int32_t* __restrict__ send_local,
                           const val_t* __restrict__ energy, val_t* __restrict__ out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) out[k] = energy[send_local[k]];
}

// ---------------------------------------------------------------------------
// Device-side simulation state
// ---------------------------------------------------------------------------
struct DeviceState {
    val_t* energy[2] = {nullptr, nullptr};  // n_local + n_halo entries each
    val_t* tflux[2] = {nullptr, nullptr};   // n_local entries each
    uint8_t* material_idx = nullptr;
    uint8_t* num_connections = nullptr;
    int32_t* conn_idx = nullptr;
    val_t* conn_flux = nullptr;
    Material* materials = nullptr;
    int32_t* send_local = nullptr;
    val_t* send_buf = nullptr;     // device pack buffer
    val_t* send_host = nullptr;    // pinned host staging
    val_t* recv_host = nullptr;    // pinned host staging
    int n_send = 0;
    cudaStream_t s_interior = nullptr;
    cudaStream_t s_halo = nullptr;
};

static void deviceUpload(DeviceState& d, const LocalMesh& mesh) {
    const size_t n_tot = mesh.n_local + mesh.n_halo;
    const size_t nz = std::max<size_t>(n_tot, 1);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d.energy[b], nz * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.energy[b], 0, nz * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&d.tflux[b], std::max<size_t>(mesh.n_local, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(d.tflux[b], 0, std::max<size_t>(mesh.n_local, 1) * sizeof(val_t)));
    }
    const size_t n_conn = std::max<size_t>(mesh.n_local * MAX_CONNECTIONS, 1);
    CUDA_CHECK(cudaMalloc(&d.material_idx, std::max<size_t>(mesh.n_local, 1)));
    CUDA_CHECK(cudaMalloc(&d.num_connections, std::max<size_t>(mesh.n_local, 1)));
    CUDA_CHECK(cudaMalloc(&d.conn_idx, n_conn * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.conn_flux, n_conn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.materials, mesh.materials.size() * sizeof(Material)));
    if (mesh.n_local > 0) {
        CUDA_CHECK(cudaMemcpy(d.material_idx, mesh.material_idx.data(), mesh.n_local, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.num_connections, mesh.num_connections.data(), mesh.n_local, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.conn_idx, mesh.conn_idx.data(), mesh.conn_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.conn_flux, mesh.conn_flux.data(), mesh.conn_flux.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d.materials, mesh.materials.data(), mesh.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    d.n_send = static_cast<int>(mesh.send_local.size());
    const size_t ns = std::max<size_t>(d.n_send, 1);
    const size_t nh = std::max<size_t>(mesh.n_halo, 1);
    CUDA_CHECK(cudaMalloc(&d.send_local, ns * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.send_buf, ns * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&d.send_host, ns * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&d.recv_host, nh * sizeof(val_t)));
    if (d.n_send > 0) {
        CUDA_CHECK(cudaMemcpy(d.send_local, mesh.send_local.data(), d.n_send * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&d.s_interior, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&d.s_halo, cudaStreamNonBlocking));
}

static void deviceFree(DeviceState& d) {
    for (int b = 0; b < 2; ++b) {
        cudaFree(d.energy[b]);
        cudaFree(d.tflux[b]);
    }
    cudaFree(d.material_idx);
    cudaFree(d.num_connections);
    cudaFree(d.conn_idx);
    cudaFree(d.conn_flux);
    cudaFree(d.materials);
    cudaFree(d.send_local);
    cudaFree(d.send_buf);
    cudaFreeHost(d.send_host);
    cudaFreeHost(d.recv_host);
    cudaStreamDestroy(d.s_interior);
    cudaStreamDestroy(d.s_halo);
}

// Run simulation for n_iters iterations. On return, cur indicates which buffer
// holds the final state.
static void runSimulation(DeviceState& d, const LocalMesh& mesh, const int n_iters, int& cur,
                          const int size) {
    constexpr int BLOCK = 256;
    const idx_t n_local = mesh.n_local;
    const idx_t n_bnd = mesh.n_boundary;
    const idx_t n_int = n_local - n_bnd;
    const int n_halo = static_cast<int>(mesh.n_halo);

    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * size);

    cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        val_t* e_cur = d.energy[cur];
        val_t* e_nxt = d.energy[nxt];
        val_t* t_cur = d.tflux[cur];
        val_t* t_nxt = d.tflux[nxt];

        // 1) Interior elements: no remote dependencies, start immediately
        if (n_int > 0) {
            const unsigned grid = static_cast<unsigned>((n_int + BLOCK - 1) / BLOCK);
            updateKernel<<<grid, BLOCK, 0, d.s_interior>>>(
                n_bnd, n_local, n_local, e_cur, t_cur, e_nxt, t_nxt, d.material_idx,
                d.num_connections, d.conn_idx, d.conn_flux, d.materials);
        }

        // 2) Halo exchange of current energies on the second stream
        if (size > 1) {
            if (d.n_send > 0) {
                const unsigned grid = static_cast<unsigned>((d.n_send + BLOCK - 1) / BLOCK);
                packKernel<<<grid, BLOCK, 0, d.s_halo>>>(d.n_send, d.send_local, e_cur, d.send_buf);
                CUDA_CHECK(cudaMemcpyAsync(d.send_host, d.send_buf, d.n_send * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, d.s_halo));
            }
            CUDA_CHECK(cudaStreamSynchronize(d.s_halo));

            reqs.clear();
            for (int r = 0; r < size; ++r) {
                if (mesh.recv_counts[r] > 0) {
                    reqs.emplace_back();
                    MPI_CHECK(MPI_Irecv(d.recv_host + mesh.recv_displs[r], mesh.recv_counts[r], MPI_DOUBLE,
                                        r, 0, MPI_COMM_WORLD, &reqs.back()));
                }
            }
            for (int r = 0; r < size; ++r) {
                if (mesh.send_counts[r] > 0) {
                    reqs.emplace_back();
                    MPI_CHECK(MPI_Isend(d.send_host + mesh.send_displs[r], mesh.send_counts[r], MPI_DOUBLE,
                                        r, 0, MPI_COMM_WORLD, &reqs.back()));
                }
            }
            if (!reqs.empty()) {
                MPI_CHECK(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));
            }
            if (n_halo > 0) {
                CUDA_CHECK(cudaMemcpyAsync(e_cur + n_local, d.recv_host, n_halo * sizeof(val_t),
                                           cudaMemcpyHostToDevice, d.s_halo));
            }
        }

        // 3) Boundary elements once the halo has arrived
        if (n_bnd > 0) {
            const unsigned grid = static_cast<unsigned>((n_bnd + BLOCK - 1) / BLOCK);
            updateKernel<<<grid, BLOCK, 0, d.s_halo>>>(
                0, n_bnd, n_local, e_cur, t_cur, e_nxt, t_nxt, d.material_idx,
                d.num_connections, d.conn_idx, d.conn_flux, d.materials);
        }

        // Both streams must finish before the buffers are swapped
        CUDA_CHECK(cudaStreamSynchronize(d.s_interior));
        CUDA_CHECK(cudaStreamSynchronize(d.s_halo));

        // Swap buffers
        cur = nxt;
    }
    CUDA_CHECK(cudaGetLastError());
}

// Gather the final dynamic state in global element order on rank 0
static void gatherResults(const DeviceState& d, const LocalMesh& mesh, const int cur,
                          std::vector<ElementDynamic>& global, const int rank, const int size) {
    const idx_t n_local = mesh.n_local;
    std::vector<val_t> e_perm(std::max<idx_t>(n_local, 1)), t_perm(std::max<idx_t>(n_local, 1));
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(e_perm.data(), d.energy[cur], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(t_perm.data(), d.tflux[cur], n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    // Undo the local permutation
    std::vector<ElementDynamic> local(std::max<idx_t>(n_local, 1));
    #pragma omp parallel for schedule(static)
    for (idx_t li = 0; li < n_local; ++li) {
        const int32_t ni = mesh.perm[li];
        local[li].current_energy = e_perm[ni];
        local[li].total_flux = t_perm[ni];
    }

    MPI_Datatype elem_type;
    MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &elem_type));
    MPI_CHECK(MPI_Type_commit(&elem_type));

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(mesh.offsets[r + 1] - mesh.offsets[r]);
        displs[r] = static_cast<int>(mesh.offsets[r]);
    }
    if (rank == 0) global.resize(mesh.n_global);
    MPI_CHECK(MPI_Gatherv(local.data(), static_cast<int>(n_local), elem_type,
                          rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), elem_type,
                          0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&elem_type));
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements_dynamic) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements_dynamic) {
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

// Compute a simple hash of the results for verification (XOR is associative
// and commutative, so the OpenMP reduction is exact).
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    const size_t n = elements.size();
    #pragma omp parallel for schedule(static) reduction(^ : hash)
    for (size_t i = 0; i < n; ++i) {
        // Simple hash combining energy and flux values
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &elements[i].current_energy, sizeof(uint64_t));
        std::memcpy(&f_bits, &elements[i].total_flux, sizeof(uint64_t));
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    // Select the GPU for this rank: one rank per device on each node
    {
        MPI_Comm node_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm));
        int node_rank = 0;
        MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
        MPI_CHECK(MPI_Comm_free(&node_comm));
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        if (n_dev < 1) {
            fprintf(stderr, "No CUDA devices available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(node_rank % n_dev));
        CUDA_CHECK(cudaFree(nullptr));  // initialise the context up front
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
        printf("\n");
    }

    // Build the unstructured mesh (distributed)
    if (rank == 0) printf("Building unstructured mesh...\n");
    LocalMesh mesh;
    buildSquare2D(mesh, n_elems_root, rank, size);

    DeviceState dev;
    deviceUpload(dev, mesh);

    // Calculate memory usage (reference layout, global problem)
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    runSimulation(dev, mesh, n_iters, cur, size);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));

    // Collect results on rank 0
    std::vector<ElementDynamic> elements_dynamic;
    gatherResults(dev, mesh, cur, elements_dynamic, rank, size);
    deviceFree(dev);

    int exit_code = 0;
    if (rank == 0) {
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

        // Compute hash for verification
        const uint64_t hash = computeHash(elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData(elements_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < elements_dynamic.size(); ++i) {
                energyData[i] = elements_dynamic[i].current_energy;
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(elements_dynamic);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}
