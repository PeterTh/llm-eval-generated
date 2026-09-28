// Unstructured mesh energy transfer benchmark.
//
// Hybrid parallelization:
//   * MPI   - the mesh is partitioned into contiguous element blocks, one block
//             per rank; the connectivity that crosses a partition boundary is
//             exchanged as a halo every iteration (point-to-point, overlapped
//             with the interior computation).
//   * CUDA  - the per-iteration element update runs on a GPU (one GPU per rank,
//             assigned round-robin over the node-local ranks).
//   * OpenMP- mesh construction, halo index setup and the host-side reductions
//             (hash / validation packing) are threaded.
//
// The floating point operations per element are performed in exactly the same
// order as in the serial reference, so results are bit-identical.

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
// Utilities
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(err_));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

static int mpi_rank = 0;
static int mpi_size = 1;

// printf that only produces output on rank 0
#define rprintf(...)                                                                           \
    do {                                                                                       \
        if (mpi_rank == 0) printf(__VA_ARGS__);                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Distributed mesh
// ---------------------------------------------------------------------------

// The distributed mesh keeps the local elements in a structure-of-arrays layout
// that is friendly to the GPU. Connections are stored connection-major
// (conn[j * n_local + i]) so that consecutive threads access consecutive
// addresses.
struct DistributedMesh {
    // Global partitioning: rank r owns elements [offsets[r], offsets[r + 1])
    std::vector<int64_t> offsets;
    int64_t n_global = 0;
    int64_t gstart = 0;
    int64_t n_local = 0;
    int64_t n_halo = 0;
    int max_conn = 0;  // max number of connections of any local element

    // Host side mesh (only needed during setup)
    std::vector<Material> materials;

    // Device side static data
    uint8_t* d_mat_idx = nullptr;   // [n_local]
    uint8_t* d_num_conn = nullptr;  // [n_local]
    int32_t* d_conn_idx = nullptr;  // [max_conn * n_local], local indices
    val_t* d_conn_flux = nullptr;   // [max_conn * n_local]
    Material* d_materials = nullptr;

    // Device side dynamic data (energy buffers carry the halo at the tail)
    val_t* d_energy[2] = {nullptr, nullptr};  // [n_local + n_halo]
    val_t* d_flux = nullptr;                  // [n_local]
    int cur = 0;

    // Halo exchange
    std::vector<int> nbr_send_ranks, nbr_send_counts, nbr_send_displs;
    std::vector<int> nbr_recv_ranks, nbr_recv_counts, nbr_recv_displs;
    int64_t n_send = 0;
    int32_t* d_send_idx = nullptr;  // [n_send] local indices to be packed
    val_t* d_send_buf = nullptr;
    val_t* h_send_buf = nullptr;  // pinned
    val_t* h_recv_buf = nullptr;  // pinned
    std::vector<MPI_Request> requests;

    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_comm = nullptr;
    cudaEvent_t evt_compute = nullptr;
    cudaEvent_t evt_halo = nullptr;

    // Range of local elements that only depend on locally owned elements
    int64_t interior_begin = 0;
    int64_t interior_end = 0;
};

// Neighbor connectivity of a global element of the 2D grid, identical in order
// to the reference implementation.
static inline int gridConnections(int64_t g, int n_elems_root, int64_t* nbr, val_t* flux) {
    const int64_t x = g / n_elems_root;
    const int64_t y = g % n_elems_root;
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    int n = 0;
    for (int k = 0; k < 4; ++k) {
        const int64_t nx = x + offsets[k][0];
        const int64_t ny = y + offsets[k][1];
        if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
            nbr[n] = nx * n_elems_root + ny;
            flux[n] = 1.0;
            ++n;
        }
    }
    return n;
}

static inline uint8_t gridMaterial(int64_t g, int n_elems_root) {
    const int64_t x = g / n_elems_root;
    const int64_t y = g % n_elems_root;
    const int64_t last = n_elems_root - 1;

    // Corner elements are inflow/outflow to create interesting dynamics
    if (x == 0 && y == 0) return static_cast<uint8_t>(INFLOW_MAT_ID);
    if (x == last && y == last) return static_cast<uint8_t>(INFLOW_MAT_ID);
    if (x == 0 && y == last) return static_cast<uint8_t>(OUTFLOW_MAT_ID);
    if (x == last && y == 0) return static_cast<uint8_t>(OUTFLOW_MAT_ID);
    return static_cast<uint8_t>(DEFAULT_MAT_ID);
}

// Owner rank of a global element index
static inline int ownerOf(const std::vector<int64_t>& offsets, int64_t g) {
    const auto it = std::upper_bound(offsets.begin(), offsets.end(), g);
    return static_cast<int>(it - offsets.begin()) - 1;
}

// Build the distributed 2D square grid (partitioned by grid rows so that the
// halo of a contiguous element block stays minimal).
void buildSquare2D(DistributedMesh& mesh, const int n_elems_root) {
    const int64_t n_elems = static_cast<int64_t>(n_elems_root) * n_elems_root;

    // Initialize materials
    mesh.materials.push_back(Material{0.8, 0.0});    // Default material
    mesh.materials.push_back(Material{0.8, 0.5});    // Inflow material
    mesh.materials.push_back(Material{0.8, -0.5});   // Outflow material

    // Row-block partitioning
    mesh.n_global = n_elems;
    mesh.offsets.resize(mpi_size + 1);
    for (int r = 0; r <= mpi_size; ++r) {
        const int64_t row = (static_cast<int64_t>(n_elems_root) * r) / mpi_size;
        mesh.offsets[r] = row * n_elems_root;
    }
    mesh.gstart = mesh.offsets[mpi_rank];
    mesh.n_local = mesh.offsets[mpi_rank + 1] - mesh.gstart;

    const int64_t n_local = mesh.n_local;
    const int64_t gstart = mesh.gstart;
    const int64_t gend = gstart + n_local;

    // Pass 1: connection counts (also gives the actual max connectivity)
    std::vector<uint8_t> num_conn(n_local);
    int max_conn = 0;
#pragma omp parallel for schedule(static) reduction(max : max_conn)
    for (int64_t i = 0; i < n_local; ++i) {
        int64_t nbr[MAX_CONNECTIONS];
        val_t flux[MAX_CONNECTIONS];
        const int n = gridConnections(gstart + i, n_elems_root, nbr, flux);
        num_conn[i] = static_cast<uint8_t>(n);
        max_conn = std::max(max_conn, n);
    }
    // The connection stride must be consistent across ranks (it is only used
    // locally, but keeping it uniform makes the layout deterministic).
    MPI_Allreduce(MPI_IN_PLACE, &max_conn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    mesh.max_conn = max_conn;

    // Pass 2: fill connectivity with global neighbor indices
    std::vector<int64_t> conn_gidx(static_cast<size_t>(max_conn) * n_local, -1);
    std::vector<val_t> conn_flux(static_cast<size_t>(max_conn) * n_local, 0.0);
    std::vector<uint8_t> mat_idx(n_local);

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_local; ++i) {
        int64_t nbr[MAX_CONNECTIONS];
        val_t flux[MAX_CONNECTIONS];
        const int n = gridConnections(gstart + i, n_elems_root, nbr, flux);
        for (int j = 0; j < n; ++j) {
            conn_gidx[static_cast<size_t>(j) * n_local + i] = nbr[j];
            conn_flux[static_cast<size_t>(j) * n_local + i] = flux[j];
        }
        mat_idx[i] = gridMaterial(gstart + i, n_elems_root);
    }

    // Collect the external (remote) elements this rank needs
    std::vector<int64_t> halo_gidx;
    {
        const int nthreads = omp_get_max_threads();
        std::vector<std::vector<int64_t>> per_thread(nthreads);
#pragma omp parallel
        {
            auto& mine = per_thread[omp_get_thread_num()];
#pragma omp for schedule(static) nowait
            for (int64_t i = 0; i < n_local; ++i) {
                for (int j = 0; j < num_conn[i]; ++j) {
                    const int64_t g = conn_gidx[static_cast<size_t>(j) * n_local + i];
                    if (g < gstart || g >= gend) mine.push_back(g);
                }
            }
        }
        for (auto& v : per_thread) halo_gidx.insert(halo_gidx.end(), v.begin(), v.end());
        std::sort(halo_gidx.begin(), halo_gidx.end());
        halo_gidx.erase(std::unique(halo_gidx.begin(), halo_gidx.end()), halo_gidx.end());
    }
    mesh.n_halo = static_cast<int64_t>(halo_gidx.size());

    // Translate connectivity to local indices: owned elements keep their local
    // index, remote elements are placed behind them in the energy buffer.
    // Sorted halo indices are automatically grouped by owner rank.
    std::vector<int32_t> conn_lidx(static_cast<size_t>(max_conn) * n_local, 0);
    int64_t bnd_lo_end = 0;              // external deps live in [0, bnd_lo_end)
    int64_t bnd_hi_start = n_local;      // ... and in [bnd_hi_start, n_local)
    {
        int64_t lo = 0, hi = n_local;
#pragma omp parallel for schedule(static) reduction(max : lo) reduction(min : hi)
        for (int64_t i = 0; i < n_local; ++i) {
            bool external = false;
            for (int j = 0; j < max_conn; ++j) {
                const size_t k = static_cast<size_t>(j) * n_local + i;
                const int64_t g = conn_gidx[k];
                if (j >= num_conn[i]) {
                    conn_lidx[k] = 0;  // padding, never read
                } else if (g >= gstart && g < gend) {
                    conn_lidx[k] = static_cast<int32_t>(g - gstart);
                } else {
                    const auto it = std::lower_bound(halo_gidx.begin(), halo_gidx.end(), g);
                    conn_lidx[k] = static_cast<int32_t>(n_local + (it - halo_gidx.begin()));
                    external = true;
                }
            }
            if (external) {
                if (i < n_local / 2) {
                    lo = std::max(lo, i + 1);
                } else {
                    hi = std::min(hi, i);
                }
            }
        }
        bnd_lo_end = lo;
        bnd_hi_start = std::max(hi, lo);
    }
    mesh.interior_begin = bnd_lo_end;
    mesh.interior_end = bnd_hi_start;

    // ---- Halo exchange setup -------------------------------------------------
    std::vector<int> recv_counts(mpi_size, 0), recv_displs(mpi_size, 0);
    for (int64_t k = 0; k < mesh.n_halo; ++k) recv_counts[ownerOf(mesh.offsets, halo_gidx[k])]++;
    for (int r = 1; r < mpi_size; ++r) recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];

    std::vector<int> send_counts(mpi_size, 0), send_displs(mpi_size, 0);
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < mpi_size; ++r) send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    mesh.n_send = send_displs[mpi_size - 1] + send_counts[mpi_size - 1];

    // Tell every owner which of its elements we need
    std::vector<int64_t> send_gidx(mesh.n_send);
    MPI_Alltoallv(halo_gidx.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                  send_gidx.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  MPI_COMM_WORLD);

    std::vector<int32_t> send_lidx(mesh.n_send);
#pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < mesh.n_send; ++k) {
        send_lidx[k] = static_cast<int32_t>(send_gidx[k] - gstart);
    }

    for (int r = 0; r < mpi_size; ++r) {
        if (r == mpi_rank) continue;
        if (send_counts[r] > 0) {
            mesh.nbr_send_ranks.push_back(r);
            mesh.nbr_send_counts.push_back(send_counts[r]);
            mesh.nbr_send_displs.push_back(send_displs[r]);
        }
        if (recv_counts[r] > 0) {
            mesh.nbr_recv_ranks.push_back(r);
            mesh.nbr_recv_counts.push_back(recv_counts[r]);
            mesh.nbr_recv_displs.push_back(recv_displs[r]);
        }
    }
    mesh.requests.resize(mesh.nbr_send_ranks.size() + mesh.nbr_recv_ranks.size());

    // ---- Device allocation and upload ---------------------------------------
    CUDA_CHECK(cudaStreamCreate(&mesh.stream_compute));
    CUDA_CHECK(cudaStreamCreate(&mesh.stream_comm));
    CUDA_CHECK(cudaEventCreateWithFlags(&mesh.evt_compute, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&mesh.evt_halo, cudaEventDisableTiming));

    const size_t n_dyn = static_cast<size_t>(n_local + mesh.n_halo);
    const size_t conn_elems = static_cast<size_t>(max_conn) * n_local;

    CUDA_CHECK(cudaMalloc(&mesh.d_mat_idx, std::max<size_t>(1, n_local)));
    CUDA_CHECK(cudaMalloc(&mesh.d_num_conn, std::max<size_t>(1, n_local)));
    CUDA_CHECK(cudaMalloc(&mesh.d_conn_idx, std::max<size_t>(1, conn_elems) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_conn_flux, std::max<size_t>(1, conn_elems) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_materials, mesh.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&mesh.d_energy[0], std::max<size_t>(1, n_dyn) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_energy[1], std::max<size_t>(1, n_dyn) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&mesh.d_flux, std::max<size_t>(1, n_local) * sizeof(val_t)));

    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(mesh.d_mat_idx, mat_idx.data(), n_local, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mesh.d_num_conn, num_conn.data(), n_local, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mesh.d_conn_idx, conn_lidx.data(), conn_elems * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mesh.d_conn_flux, conn_flux.data(), conn_elems * sizeof(val_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(mesh.d_materials, mesh.materials.data(),
                          mesh.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

    // All elements start with zero energy and zero accumulated flux
    CUDA_CHECK(cudaMemset(mesh.d_energy[0], 0, std::max<size_t>(1, n_dyn) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(mesh.d_energy[1], 0, std::max<size_t>(1, n_dyn) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(mesh.d_flux, 0, std::max<size_t>(1, n_local) * sizeof(val_t)));

    if (mesh.n_send > 0) {
        CUDA_CHECK(cudaMalloc(&mesh.d_send_idx, mesh.n_send * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&mesh.d_send_buf, mesh.n_send * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(mesh.d_send_idx, send_lidx.data(), mesh.n_send * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaHostAlloc(&mesh.h_send_buf, mesh.n_send * sizeof(val_t),
                                 cudaHostAllocDefault));
    }
    if (mesh.n_halo > 0) {
        CUDA_CHECK(cudaHostAlloc(&mesh.h_recv_buf, mesh.n_halo * sizeof(val_t),
                                 cudaHostAllocDefault));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeMesh(DistributedMesh& mesh) {
    cudaFree(mesh.d_mat_idx);
    cudaFree(mesh.d_num_conn);
    cudaFree(mesh.d_conn_idx);
    cudaFree(mesh.d_conn_flux);
    cudaFree(mesh.d_materials);
    cudaFree(mesh.d_energy[0]);
    cudaFree(mesh.d_energy[1]);
    cudaFree(mesh.d_flux);
    cudaFree(mesh.d_send_idx);
    cudaFree(mesh.d_send_buf);
    cudaFreeHost(mesh.h_send_buf);
    cudaFreeHost(mesh.h_recv_buf);
    cudaStreamDestroy(mesh.stream_compute);
    cudaStreamDestroy(mesh.stream_comm);
    cudaEventDestroy(mesh.evt_compute);
    cudaEventDestroy(mesh.evt_halo);
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Update elements [begin, end) of the local partition. The flux contributions
// are accumulated in exactly the reference order.
__global__ void updateElementsKernel(const int64_t begin, const int64_t end,
                                     const int64_t n_local,
                                     const uint8_t* __restrict__ mat_idx,
                                     const uint8_t* __restrict__ num_conn,
                                     const int32_t* __restrict__ conn_idx,
                                     const val_t* __restrict__ conn_flux,
                                     const Material* __restrict__ materials,
                                     const val_t* __restrict__ energy_in,
                                     val_t* __restrict__ energy_out,
                                     val_t* __restrict__ flux_acc) {
    const int64_t i = begin + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= end) return;

    const Material mat = materials[mat_idx[i]];
    const val_t energy = energy_in[i];

    // Start with external flow
    val_t total_flux = mat.external_flow;

    // Add flux from all connected elements
    const int nc = num_conn[i];
    for (int j = 0; j < nc; ++j) {
        const size_t k = static_cast<size_t>(j) * n_local + i;
        const val_t neighbor_energy = energy_in[conn_idx[k]];
        total_flux += (neighbor_energy - energy) * mat.transfer_coeff * conn_flux[k] * 0.25;
    }

    // Update element state
    energy_out[i] = energy + total_flux;
    flux_acc[i] += fabs(total_flux);
}

__global__ void packHaloKernel(const int64_t n, const int32_t* __restrict__ idx,
                               const val_t* __restrict__ energy, val_t* __restrict__ buf) {
    const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) buf[k] = energy[idx[k]];
}

static constexpr int BLOCK_SIZE = 256;

static inline void launchUpdate(DistributedMesh& mesh, int64_t begin, int64_t end,
                                const val_t* in, val_t* out) {
    if (end <= begin) return;
    const int64_t n = end - begin;
    const int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateElementsKernel<<<static_cast<unsigned>(blocks), BLOCK_SIZE, 0, mesh.stream_compute>>>(
        begin, end, mesh.n_local, mesh.d_mat_idx, mesh.d_num_conn, mesh.d_conn_idx,
        mesh.d_conn_flux, mesh.d_materials, in, out, mesh.d_flux);
}

// Run simulation for n_iters iterations
void runSimulation(DistributedMesh& mesh, const int n_iters) {
    const bool distributed = (mesh.n_halo > 0 || mesh.n_send > 0);

    for (int iter = 0; iter < n_iters; ++iter) {
        const val_t* in = mesh.d_energy[mesh.cur];
        val_t* out = mesh.d_energy[mesh.cur ^ 1];

        if (distributed) {
            // Make the communication stream wait for the previous update
            CUDA_CHECK(cudaEventRecord(mesh.evt_compute, mesh.stream_compute));
            CUDA_CHECK(cudaStreamWaitEvent(mesh.stream_comm, mesh.evt_compute, 0));

            // Pack the boundary energies this rank has to provide
            if (mesh.n_send > 0) {
                const int64_t blocks = (mesh.n_send + BLOCK_SIZE - 1) / BLOCK_SIZE;
                packHaloKernel<<<static_cast<unsigned>(blocks), BLOCK_SIZE, 0, mesh.stream_comm>>>(
                    mesh.n_send, mesh.d_send_idx, in, mesh.d_send_buf);
                CUDA_CHECK(cudaMemcpyAsync(mesh.h_send_buf, mesh.d_send_buf,
                                           mesh.n_send * sizeof(val_t), cudaMemcpyDeviceToHost,
                                           mesh.stream_comm));
            }
            CUDA_CHECK(cudaStreamSynchronize(mesh.stream_comm));

            int nreq = 0;
            for (size_t k = 0; k < mesh.nbr_recv_ranks.size(); ++k) {
                MPI_Irecv(mesh.h_recv_buf + mesh.nbr_recv_displs[k], mesh.nbr_recv_counts[k],
                          MPI_DOUBLE, mesh.nbr_recv_ranks[k], 0, MPI_COMM_WORLD,
                          &mesh.requests[nreq++]);
            }
            for (size_t k = 0; k < mesh.nbr_send_ranks.size(); ++k) {
                MPI_Isend(mesh.h_send_buf + mesh.nbr_send_displs[k], mesh.nbr_send_counts[k],
                          MPI_DOUBLE, mesh.nbr_send_ranks[k], 0, MPI_COMM_WORLD,
                          &mesh.requests[nreq++]);
            }

            // Overlap: elements that do not depend on remote data
            launchUpdate(mesh, mesh.interior_begin, mesh.interior_end, in, out);

            MPI_Waitall(nreq, mesh.requests.data(), MPI_STATUSES_IGNORE);

            if (mesh.n_halo > 0) {
                CUDA_CHECK(cudaMemcpyAsync(const_cast<val_t*>(in) + mesh.n_local, mesh.h_recv_buf,
                                           mesh.n_halo * sizeof(val_t), cudaMemcpyHostToDevice,
                                           mesh.stream_comm));
                CUDA_CHECK(cudaEventRecord(mesh.evt_halo, mesh.stream_comm));
                CUDA_CHECK(cudaStreamWaitEvent(mesh.stream_compute, mesh.evt_halo, 0));
            }

            // Remaining (boundary) elements
            launchUpdate(mesh, 0, mesh.interior_begin, in, out);
            launchUpdate(mesh, mesh.interior_end, mesh.n_local, in, out);
        } else {
            launchUpdate(mesh, 0, mesh.n_local, in, out);
        }

        // Swap buffers
        mesh.cur ^= 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(mesh.stream_compute));
}

// Copy the local results back to the host
void downloadResults(const DistributedMesh& mesh, std::vector<val_t>& energy,
                     std::vector<val_t>& flux) {
    energy.resize(mesh.n_local);
    flux.resize(mesh.n_local);
    if (mesh.n_local == 0) return;
    CUDA_CHECK(cudaMemcpy(energy.data(), mesh.d_energy[mesh.cur], mesh.n_local * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), mesh.d_flux, mesh.n_local * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// Gather a distributed array in global element order on rank 0
void gatherGlobal(const DistributedMesh& mesh, const std::vector<val_t>& local,
                  std::vector<val_t>& global) {
    if (mpi_rank == 0) global.resize(mesh.n_global);
    std::vector<int> counts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        counts[r] = static_cast<int>(mesh.offsets[r + 1] - mesh.offsets[r]);
        displs[r] = static_cast<int>(mesh.offsets[r]);
    }
    MPI_Gatherv(local.data(), static_cast<int>(mesh.n_local), MPI_DOUBLE,
                mpi_rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

// Validate simulation results (rank 0, global data)
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

// Compute a simple hash of the results for verification. The hash combines the
// per-element contributions with XOR, so it can be reduced across threads and
// ranks without changing the result.
uint64_t computeHash(const DistributedMesh& mesh, const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    uint64_t hash = 0;
#pragma omp parallel for schedule(static) reduction(^ : hash)
    for (int64_t i = 0; i < mesh.n_local; ++i) {
        // Simple hash combining energy and flux values
        const uint64_t g = static_cast<uint64_t>(mesh.gstart + i);
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &energy[i], sizeof(uint64_t));
        memcpy(&f_bits, &flux[i], sizeof(uint64_t));
        hash ^= (e_bits + g) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + g) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
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

// Assign one GPU per rank, round-robin over the node-local ranks
static void selectDevice() {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation outside of the timed region
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            rprintf("Unknown option: %s\n", argv[i]);
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0) {
        rprintf("Grid size must be positive\n");
        MPI_Finalize();
        return 1;
    }

    selectDevice();

    const int64_t n_elems = static_cast<int64_t>(n_elems_root) * n_elems_root;

    rprintf("Unstructured Mesh Energy Transfer Benchmark\n");
    rprintf("============================================\n");
    rprintf("Grid size: %d x %d = %lld elements\n", n_elems_root, n_elems_root,
            static_cast<long long>(n_elems));
    rprintf("Iterations: %d\n", n_iters);
    rprintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rprintf("Parallelization: %d MPI rank(s) x %d OpenMP thread(s) + CUDA\n", mpi_size,
            omp_get_max_threads());
    rprintf("\n");

    // Build the distributed unstructured mesh
    rprintf("Building unstructured mesh...\n");
    DistributedMesh mesh;
    buildSquare2D(mesh, n_elems_root);

    // Calculate memory usage (global mesh, as in the reference)
    const size_t element_static_bytes = sizeof(idx_t) * 2 + sizeof(idx_t) * MAX_CONNECTIONS +
                                        sizeof(val_t) * MAX_CONNECTIONS;
    const size_t static_mem = static_cast<size_t>(n_elems) * element_static_bytes;
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    rprintf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
            total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
            dynamic_mem / (1024.0 * 1024.0));
    rprintf("\n");

    // Run simulation
    rprintf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(mesh, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

    rprintf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));

    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec =
        (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;

    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;

    rprintf("Performance:\n");
    rprintf("  Time per iteration: %.4f ms\n", time_per_iter);
    rprintf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    rprintf("  Performance: %.4f GFLOPS\n", gflops);

    // Fetch the results
    std::vector<val_t> energy_local, flux_local;
    downloadResults(mesh, energy_local, flux_local);

    // Compute hash for verification
    const uint64_t hash = computeHash(mesh, energy_local, flux_local);
    rprintf("  Result hash: %016lX\n", hash);
    rprintf("\n");

    int exit_code = 0;

    // Print results for external validation
    if (printResults || validate) {
        std::vector<val_t> energy_global;
        gatherGlobal(mesh, energy_local, energy_global);

        if (printResults && mpi_rank == 0) {
            print_results(energy_global, "ElementEnergy");
        }

        if (validate) {
            std::vector<val_t> flux_global;
            gatherGlobal(mesh, flux_local, flux_global);
            if (mpi_rank == 0) {
                if (!validateResults(energy_global, flux_global)) exit_code = 1;
            }
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    freeMesh(mesh);
    MPI_Finalize();
    return exit_code;
}
