#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

static_assert(std::is_same_v<val_t, double>, "MPI/CUDA paths assume double");

// ---------------------------------------------------------------------------
// CUDA error handling
// ---------------------------------------------------------------------------
static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(x) cudaCheck((x), #x)

// ---------------------------------------------------------------------------
// Domain decomposition
// ---------------------------------------------------------------------------
struct Decomp {
    int n = 0;
    int px = 1;
    int py = 1;
    int rank = 0;
    int cx = 0;
    int cy = 0;
    bool active = true;
    int x0 = 0, x1 = 0;
    int y0 = 0, y1 = 0;
};

static inline void tileBounds(int n, int px, int py, int cx, int cy,
                              int& x0, int& x1, int& y0, int& y1) {
    x0 = static_cast<int>((static_cast<long long>(cx) * n) / px);
    x1 = static_cast<int>((static_cast<long long>(cx + 1) * n) / px);
    y0 = static_cast<int>((static_cast<long long>(cy) * n) / py);
    y1 = static_cast<int>((static_cast<long long>(cy + 1) * n) / py);
}

// Local, padded mesh representation.  Rows/columns of ghost cells are kept
// around the owned tile so that every connection can be evaluated by reading
// a single padded index.
struct LocalMesh {
    int nx = 0;
    int ny = 0;
    int pitch = 2;
    int local_count = 0;

    std::vector<int> conn_idx;      // [connection][element]
    std::vector<val_t> conn_flux;   // [connection][element]
    std::vector<int> num_conn;      // [element]
    std::vector<val_t> coeff;       // [element]
    std::vector<val_t> ext_flow;    // [element]
};

// Build the local tile of the same logical square grid used by the reference
// code.  Connection ordering is identical to the reference implementation.
static void buildLocalMesh(LocalMesh& mesh, const Decomp& d) {
    mesh.nx = d.x1 - d.x0;
    mesh.ny = d.y1 - d.y0;
    mesh.pitch = mesh.ny + 2;
    mesh.local_count = mesh.nx * mesh.ny;

    const size_t conn_entries =
        static_cast<size_t>(MAX_CONNECTIONS) * static_cast<size_t>(mesh.local_count);

    mesh.conn_idx.assign(conn_entries, 0);
    mesh.conn_flux.assign(conn_entries, 0.0);
    mesh.num_conn.assign(static_cast<size_t>(mesh.local_count), 0);
    mesh.coeff.assign(static_cast<size_t>(mesh.local_count), 0.8);
    mesh.ext_flow.assign(static_cast<size_t>(mesh.local_count), 0.0);

    const int n = d.n;
    const int last = n - 1;
    const int lcount = mesh.local_count;
    const int ny = mesh.ny;
    const int pitch = mesh.pitch;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lcount; ++i) {
        const int lx = i / ny;
        const int ly = i - lx * ny;
        const int gx = d.x0 + lx;
        const int gy = d.y0 + ly;
        const int p = lx + 1;
        const int q = ly + 1;

        // Reproduce the reference sequence of corner material assignments.
        idx_t mat = DEFAULT_MAT_ID;
        if (gx == 0 && gy == 0) mat = INFLOW_MAT_ID;
        if (gx == 0 && gy == last) mat = OUTFLOW_MAT_ID;
        if (gx == last && gy == 0) mat = OUTFLOW_MAT_ID;
        if (gx == last && gy == last) mat = INFLOW_MAT_ID;

        mesh.coeff[i] = 0.8;
        mesh.ext_flow[i] = (mat == INFLOW_MAT_ID) ? 0.5
                          : (mat == OUTFLOW_MAT_ID) ? -0.5
                                                     : 0.0;

        int c = 0;
        // +x
        if (gx + 1 < n) {
            mesh.conn_idx[static_cast<size_t>(c) * lcount + i] = (p + 1) * pitch + q;
            mesh.conn_flux[static_cast<size_t>(c) * lcount + i] = 1.0;
            ++c;
        }
        // -x
        if (gx - 1 >= 0) {
            mesh.conn_idx[static_cast<size_t>(c) * lcount + i] = (p - 1) * pitch + q;
            mesh.conn_flux[static_cast<size_t>(c) * lcount + i] = 1.0;
            ++c;
        }
        // +y
        if (gy + 1 < n) {
            mesh.conn_idx[static_cast<size_t>(c) * lcount + i] = p * pitch + (q + 1);
            mesh.conn_flux[static_cast<size_t>(c) * lcount + i] = 1.0;
            ++c;
        }
        // -y
        if (gy - 1 >= 0) {
            mesh.conn_idx[static_cast<size_t>(c) * lcount + i] = p * pitch + (q - 1);
            mesh.conn_flux[static_cast<size_t>(c) * lcount + i] = 1.0;
            ++c;
        }
        mesh.num_conn[i] = c;
    }
}

// ---------------------------------------------------------------------------
// CUDA simulation kernel
// ---------------------------------------------------------------------------
__global__ void energyTransferKernel(
    const val_t* __restrict__ energy_in,
    val_t* __restrict__ energy_out,
    const val_t* __restrict__ flux_in,
    val_t* __restrict__ flux_out,
    const int* __restrict__ conn_idx,
    const val_t* __restrict__ conn_flux,
    const int* __restrict__ num_conn,
    const val_t* __restrict__ coeff,
    const val_t* __restrict__ ext_flow,
    int nx, int ny, int pitch, int local_count) {
    const int ly = blockIdx.x * blockDim.x + threadIdx.x;
    const int lx = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= nx || ly >= ny) return;

    const int i = lx * ny + ly;
    const int p = (lx + 1) * pitch + (ly + 1);

    const val_t this_energy = energy_in[p];
    const val_t c = coeff[i];
    val_t total_flux = ext_flow[i];
    const int nc = num_conn[i];

    #pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const val_t other = energy_in[conn_idx[static_cast<size_t>(j) * local_count + i]];
            total_flux += (other - this_energy) * c *
                          conn_flux[static_cast<size_t>(j) * local_count + i] * 0.25;
        }
    }

    energy_out[p] = this_energy + total_flux;
    flux_out[p] = flux_in[p] + fabs(total_flux);
}

// Gathering/scattering a strided ghost column through a contiguous buffer is
// considerably faster than a narrow cudaMemcpy2D for large tiles.
__global__ void packColumnKernel(const val_t* __restrict__ energy,
                                 val_t* __restrict__ out,
                                 int q, int nx, int pitch) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nx) out[p] = energy[(p + 1) * pitch + q];
}

__global__ void unpackColumnKernel(val_t* __restrict__ energy,
                                   const val_t* __restrict__ in,
                                   int q, int nx, int pitch) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nx) energy[(p + 1) * pitch + q] = in[p];
}

struct DeviceData {
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux[2] = {nullptr, nullptr};
    int* conn_idx = nullptr;
    val_t* conn_flux = nullptr;
    int* num_conn = nullptr;
    val_t* coeff = nullptr;
    val_t* ext_flow = nullptr;
    size_t padded_size = 0;
    int cur = 0;
};

static void initDeviceData(DeviceData& dev, const LocalMesh& mesh) {
    dev.padded_size = static_cast<size_t>(mesh.nx + 2) * static_cast<size_t>(mesh.ny + 2);
    if (dev.padded_size == 0) dev.padded_size = 1;

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.energy[b]),
                              dev.padded_size * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.flux[b]),
                              dev.padded_size * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.energy[b], 0, dev.padded_size * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.flux[b], 0, dev.padded_size * sizeof(val_t)));
    }

    const size_t lc = static_cast<size_t>(mesh.local_count);
    const size_t conn_entries = static_cast<size_t>(MAX_CONNECTIONS) * lc;
    const size_t conn_bytes = (conn_entries ? conn_entries : 1) * sizeof(int);
    const size_t conn_flux_bytes = (conn_entries ? conn_entries : 1) * sizeof(val_t);
    const size_t elem_bytes = (lc ? lc : 1) * sizeof(val_t);
    const size_t elem_int_bytes = (lc ? lc : 1) * sizeof(int);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.conn_idx), conn_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.conn_flux), conn_flux_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.num_conn), elem_int_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.coeff), elem_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.ext_flow), elem_bytes));

    if (lc > 0) {
        CUDA_CHECK(cudaMemcpy(dev.conn_idx, mesh.conn_idx.data(), conn_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.conn_flux, mesh.conn_flux.data(), conn_flux_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.num_conn, mesh.num_conn.data(), elem_int_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.coeff, mesh.coeff.data(), elem_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.ext_flow, mesh.ext_flow.data(), elem_bytes,
                              cudaMemcpyHostToDevice));
    }
}

static void freeDeviceData(DeviceData& dev) {
    for (int b = 0; b < 2; ++b) {
        if (dev.energy[b]) cudaFree(dev.energy[b]);
        if (dev.flux[b]) cudaFree(dev.flux[b]);
    }
    if (dev.conn_idx) cudaFree(dev.conn_idx);
    if (dev.conn_flux) cudaFree(dev.conn_flux);
    if (dev.num_conn) cudaFree(dev.num_conn);
    if (dev.coeff) cudaFree(dev.coeff);
    if (dev.ext_flow) cudaFree(dev.ext_flow);
}

// ---------------------------------------------------------------------------
// Halo exchange
// ---------------------------------------------------------------------------
enum { DIR_N = 0, DIR_S = 1, DIR_W = 2, DIR_E = 3 };

enum {
    TAG_XP = 101,
    TAG_XM = 102,
    TAG_YP = 103,
    TAG_YM = 104
};

struct HaloBuffers {
    val_t* send[4] = {nullptr, nullptr, nullptr, nullptr};
    val_t* recv[4] = {nullptr, nullptr, nullptr, nullptr};
    val_t* dev_send[4] = {nullptr, nullptr, nullptr, nullptr};
    val_t* dev_recv[4] = {nullptr, nullptr, nullptr, nullptr};
};

static void allocHaloBuffers(HaloBuffers& h, const LocalMesh& mesh) {
    const size_t max_len =
        static_cast<size_t>(std::max(std::max(mesh.nx, mesh.ny), 1));
    const size_t bytes = max_len * sizeof(val_t);
    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.send[i]), bytes,
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h.recv[i]), bytes,
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h.dev_send[i]), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h.dev_recv[i]), bytes));
    }
}

static void freeHaloBuffers(HaloBuffers& h) {
    for (int i = 0; i < 4; ++i) {
        if (h.send[i]) cudaFreeHost(h.send[i]);
        if (h.recv[i]) cudaFreeHost(h.recv[i]);
        if (h.dev_send[i]) cudaFree(h.dev_send[i]);
        if (h.dev_recv[i]) cudaFree(h.dev_recv[i]);
    }
}

static inline void copyRowToHost(val_t* dst, const val_t* src, int len) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(len) * sizeof(val_t),
                               cudaMemcpyDeviceToHost, 0));
}

static inline void copyRowToDevice(val_t* dst, const val_t* src, int len) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(len) * sizeof(val_t),
                               cudaMemcpyHostToDevice, 0));
}

static inline void copyColToHost(val_t* dst, const val_t* src, val_t* dev_tmp,
                                 int q, int len, int pitch) {
    const int threads = 128;
    const int blocks = (len + threads - 1) / threads;
    packColumnKernel<<<blocks, threads>>>(src, dev_tmp, q, len, pitch);
    CUDA_CHECK(cudaGetLastError());
    copyRowToHost(dst, dev_tmp, len);
}

static inline void copyColToDevice(val_t* dst, const val_t* src, val_t* dev_tmp,
                                   int q, int len, int pitch) {
    copyRowToDevice(dev_tmp, src, len);
    const int threads = 128;
    const int blocks = (len + threads - 1) / threads;
    unpackColumnKernel<<<blocks, threads>>>(dst, dev_tmp, q, len, pitch);
    CUDA_CHECK(cudaGetLastError());
}

static void exchangeHalos(const Decomp& d, const LocalMesh& mesh,
                          val_t* d_energy, HaloBuffers& h, MPI_Comm comm) {
    if (!d.active || mesh.local_count == 0) return;
    if (d.px == 1 && d.py == 1) return;

    const int nx = mesh.nx;
    const int ny = mesh.ny;
    const int pitch = mesh.pitch;

    const bool north = d.cx > 0;
    const bool south = d.cx + 1 < d.px;
    const bool west = d.cy > 0;
    const bool east = d.cy + 1 < d.py;

    // Pack owned boundary data into pinned host buffers.
    if (north) copyRowToHost(h.send[DIR_N], d_energy + 1 * pitch + 1, ny);
    if (south) copyRowToHost(h.send[DIR_S], d_energy + nx * pitch + 1, ny);
    if (west) copyColToHost(h.send[DIR_W], d_energy, h.dev_send[DIR_W], 1, nx, pitch);
    if (east) copyColToHost(h.send[DIR_E], d_energy, h.dev_send[DIR_E], ny, nx, pitch);

    // The sends below use pinned host memory, so the copy must complete first.
    CUDA_CHECK(cudaStreamSynchronize(0));

    MPI_Request reqs[8];
    int nreq = 0;

    if (north) {
        MPI_Irecv(h.recv[DIR_N], ny, MPI_DOUBLE, d.rank - 1, TAG_XP, comm, &reqs[nreq++]);
        MPI_Isend(h.send[DIR_N], ny, MPI_DOUBLE, d.rank - 1, TAG_XM, comm, &reqs[nreq++]);
    }
    if (south) {
        MPI_Irecv(h.recv[DIR_S], ny, MPI_DOUBLE, d.rank + 1, TAG_XM, comm, &reqs[nreq++]);
        MPI_Isend(h.send[DIR_S], ny, MPI_DOUBLE, d.rank + 1, TAG_XP, comm, &reqs[nreq++]);
    }
    if (west) {
        MPI_Irecv(h.recv[DIR_W], nx, MPI_DOUBLE, d.rank - d.px, TAG_YP, comm, &reqs[nreq++]);
        MPI_Isend(h.send[DIR_W], nx, MPI_DOUBLE, d.rank - d.px, TAG_YM, comm, &reqs[nreq++]);
    }
    if (east) {
        MPI_Irecv(h.recv[DIR_E], nx, MPI_DOUBLE, d.rank + d.px, TAG_YM, comm, &reqs[nreq++]);
        MPI_Isend(h.send[DIR_E], nx, MPI_DOUBLE, d.rank + d.px, TAG_YP, comm, &reqs[nreq++]);
    }

    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Scatter received data into the ghost rows/columns.
    if (north) copyRowToDevice(d_energy + 0 * pitch + 1, h.recv[DIR_N], ny);
    if (south) copyRowToDevice(d_energy + (nx + 1) * pitch + 1, h.recv[DIR_S], ny);
    if (west) copyColToDevice(d_energy, h.recv[DIR_W], h.dev_recv[DIR_W], 0, nx, pitch);
    if (east) copyColToDevice(d_energy, h.recv[DIR_E], h.dev_recv[DIR_E], ny + 1, nx, pitch);
}

// ---------------------------------------------------------------------------
// Simulation driver
// ---------------------------------------------------------------------------
static void runSimulation(const Decomp& d, const LocalMesh& mesh,
                          DeviceData& dev, HaloBuffers& halo,
                          int n_iters, MPI_Comm comm) {
    if (mesh.local_count == 0) return;

    dim3 block(32, 8);
    dim3 grid((mesh.ny + block.x - 1) / block.x,
              (mesh.nx + block.y - 1) / block.y);

    for (int iter = 0; iter < n_iters; ++iter) {
        const int cur = dev.cur;
        const int nxt = 1 - cur;

        energyTransferKernel<<<grid, block>>>(
            dev.energy[cur], dev.energy[nxt],
            dev.flux[cur], dev.flux[nxt],
            dev.conn_idx, dev.conn_flux, dev.num_conn,
            dev.coeff, dev.ext_flow,
            mesh.nx, mesh.ny, mesh.pitch, mesh.local_count);
        CUDA_CHECK(cudaGetLastError());

        dev.cur = nxt;

        // The final halo exchange would only update data that is never read.
        if (iter + 1 < n_iters) {
            exchangeHalos(d, mesh, dev.energy[dev.cur], halo, comm);
        }
    }

    // Make sure the final (possibly asynchronous) kernel has completed before
    // the caller stops the timer.
    CUDA_CHECK(cudaStreamSynchronize(0));
}

// ---------------------------------------------------------------------------
// Validation / hashing
// ---------------------------------------------------------------------------
static bool validateResults(const std::vector<val_t>& energies,
                            const std::vector<val_t>& fluxes) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energies.size(); ++i) {
        energy_sum += energies[i];
        flux_sum += fluxes[i];
        energy_max = std::max(energies[i], energy_max);
        energy_min = std::min(energies[i], energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
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

static uint64_t computeHash(const std::vector<val_t>& energies,
                            const std::vector<val_t>& fluxes) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energies.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&energies[i]);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&fluxes[i]);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;
    const char* badArg = nullptr;

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
            showHelp = true;
        } else {
            badOption = true;
            badArg = argv[i];
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (badOption) {
        if (rank == 0) {
            printf("Unknown option: %s\n", badArg);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const long long n_elems =
        static_cast<long long>(n_elems_root) * static_cast<long long>(n_elems_root);
    const int n = n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %lld elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Choose a 2D process grid, clamped so that no active rank owns an empty
    // tile.  Any surplus ranks remain idle but still participate in the final
    // gather.
    int dims[2] = {1, 1};
    if (rank == 0) {
        dims[0] = 0;
        dims[1] = 0;
        MPI_Dims_create(nprocs, 2, dims);
    }
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int px = dims[0];
    int py = dims[1];
    if (n > 0) {
        px = std::min(px, n);
        py = std::min(py, n);
    } else {
        px = 1;
        py = 1;
    }
    if (px < 1) px = 1;
    if (py < 1) py = 1;
    const int active_ranks = px * py;

    Decomp d;
    d.n = n;
    d.px = px;
    d.py = py;
    d.rank = rank;
    d.active = rank < active_ranks;
    if (d.active) {
        d.cx = rank % px;
        d.cy = rank / px;
        tileBounds(n, px, py, d.cx, d.cy, d.x0, d.x1, d.y0, d.y1);
    } else {
        d.cx = 0;
        d.cy = 0;
        d.x0 = d.x1 = d.y0 = d.y1 = 0;
    }

    // Bind one GPU per rank on each node (round-robin over local ranks).
    int local_rank = rank;
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &node_comm);
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    if (rank == 0) {
        printf("Building unstructured mesh...\n");
    }

    LocalMesh mesh;
    buildLocalMesh(mesh, d);

    DeviceData dev;
    initDeviceData(dev, mesh);

    HaloBuffers halo;
    allocHaloBuffers(halo, mesh);

    if (rank == 0) {
        const size_t static_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(d, mesh, dev, halo, n_iters, MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Copy the owned interior to the host and gather it on rank 0 in global
    // element order.
    std::vector<val_t> local_energy(static_cast<size_t>(mesh.local_count));
    std::vector<val_t> local_flux(static_cast<size_t>(mesh.local_count));
    if (mesh.local_count > 0) {
        const size_t row_bytes = static_cast<size_t>(mesh.ny) * sizeof(val_t);
        CUDA_CHECK(cudaMemcpy2D(local_energy.data(), row_bytes,
                                dev.energy[dev.cur] + mesh.pitch + 1,
                                static_cast<size_t>(mesh.pitch) * sizeof(val_t),
                                row_bytes, mesh.nx, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(local_flux.data(), row_bytes,
                                dev.flux[dev.cur] + mesh.pitch + 1,
                                static_cast<size_t>(mesh.pitch) * sizeof(val_t),
                                row_bytes, mesh.nx, cudaMemcpyDeviceToHost));
    }

    std::vector<val_t> sendbuf(static_cast<size_t>(mesh.local_count) * 2);
    const int lcount = mesh.local_count;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < lcount; ++i) {
        sendbuf[static_cast<size_t>(2) * i] = local_energy[i];
        sendbuf[static_cast<size_t>(2) * i + 1] = local_flux[i];
    }

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<val_t> recvbuf;
    long long recv_total = 0;

    if (rank == 0) {
        recvcounts.resize(static_cast<size_t>(nprocs));
        displs.resize(static_cast<size_t>(nprocs));
        for (int r = 0; r < nprocs; ++r) {
            int cnt = 0;
            if (r < active_ranks) {
                const int rcx = r % px;
                const int rcy = r / px;
                int rx0, rx1, ry0, ry1;
                tileBounds(n, px, py, rcx, rcy, rx0, rx1, ry0, ry1);
                cnt = (rx1 - rx0) * (ry1 - ry0);
            }
            recvcounts[r] = cnt * 2;
            displs[r] = static_cast<int>(recv_total);
            recv_total += static_cast<long long>(cnt) * 2;
        }
        recvbuf.resize(static_cast<size_t>(recv_total));
    }

    MPI_Gatherv(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<val_t> all_energy;
    std::vector<val_t> all_flux;

    if (rank == 0) {
        all_energy.assign(static_cast<size_t>(n_elems), 0.0);
        all_flux.assign(static_cast<size_t>(n_elems), 0.0);

        for (int r = 0; r < active_ranks; ++r) {
            const int rcx = r % px;
            const int rcy = r / px;
            int rx0, rx1, ry0, ry1;
            tileBounds(n, px, py, rcx, rcy, rx0, rx1, ry0, ry1);
            const int rnx = rx1 - rx0;
            const int rny = ry1 - ry0;
            const int off = displs[r];

            #pragma omp parallel for schedule(static)
            for (int lx = 0; lx < rnx; ++lx) {
                const int base = off + lx * rny * 2;
                const size_t gbase =
                    static_cast<size_t>(rx0 + lx) * static_cast<size_t>(n) +
                    static_cast<size_t>(ry0);
                for (int ly = 0; ly < rny; ++ly) {
                    all_energy[gbase + ly] = recvbuf[base + 2 * ly];
                    all_flux[gbase + ly] = recvbuf[base + 2 * ly + 1];
                }
            }
        }

        const long long duration_ms =
            static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const double duration_s = elapsed > 0.0 ? elapsed : 1e-30;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) *
             static_cast<double>(n_elems)) / duration_s / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(all_energy, all_flux);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }

        if (validate) {
            if (!validateResults(all_energy, all_flux)) {
                freeHaloBuffers(halo);
                freeDeviceData(dev);
                MPI_Finalize();
                return 1;
            }
        }
    }

    freeHaloBuffers(halo);
    freeDeviceData(dev);

    MPI_Finalize();
    return 0;
}
