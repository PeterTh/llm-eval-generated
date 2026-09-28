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

// Hybrid parallelization:
//  - MPI: contiguous 1D domain decomposition of the element array across ranks,
//    with a generic (connectivity-driven) halo exchange each iteration.
//  - CUDA: the per-iteration element update runs on one GPU per rank, using a
//    structure-of-arrays, transposed-connectivity layout for coalesced access.
//  - OpenMP: multithreaded host-side mesh construction, connectivity remapping
//    and result assembly.

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

// Dynamic state for each element (kept for host-side validation/output)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Domain decomposition helpers: rank r owns a contiguous range of elements.
// ---------------------------------------------------------------------------
struct Decomposition {
    int64_t n_elems;
    int size;
    int64_t base;  // n_elems / size
    int64_t rem;   // n_elems % size

    int64_t start(int r) const { return r * base + std::min<int64_t>(r, rem); }
    int64_t count(int r) const { return base + (r < rem ? 1 : 0); }
    int owner(int64_t g) const {
        const int64_t big = rem * (base + 1);
        if (g < big) return static_cast<int>(g / (base + 1));
        return static_cast<int>(rem + (g - big) / base);
    }
};

// ---------------------------------------------------------------------------
// Local portion of the mesh in structure-of-arrays layout.
// Connectivity is stored transposed (connection-major) for coalesced GPU loads;
// neighbor indices are remapped to local indices, with off-rank neighbors
// referring to halo slots appended after the owned elements.
// ---------------------------------------------------------------------------
struct LocalMesh {
    int64_t n_local = 0;
    int64_t start = 0;
    std::vector<int> num_connections;      // [n_local]
    std::vector<int64_t> conn_global;      // [MAX_CONNECTIONS][n_local], global neighbor idx
    std::vector<unsigned int> conn_local;  // [MAX_CONNECTIONS][n_local], remapped
    std::vector<val_t> conn_flux;          // [MAX_CONNECTIONS][n_local]
    std::vector<val_t> transfer_coeff;     // per-element material coefficient
    std::vector<val_t> external_flow;      // per-element material external flow

    // Halo exchange plan
    std::vector<int64_t> halo_globals;       // sorted global indices of halo elements
    std::vector<int> recv_counts, recv_displs;  // per rank, in halo value units
    std::vector<int> send_counts, send_displs;
    std::vector<unsigned int> send_local;    // local indices of owned values to send
};

// Build the local slice of the 2D square grid mesh (same connectivity and
// material assignment semantics as the original serial buildSquare2D).
void buildSquare2DLocal(LocalMesh& mesh, const Decomposition& dec, int rank,
                        const int n_elems_root) {
    const std::vector<Material> materials = {
        {0.8, 0.0},   // Default material
        {0.8, 0.5},   // Inflow material
        {0.8, -0.5},  // Outflow material
    };

    const int64_t n_local = dec.count(rank);
    const int64_t start = dec.start(rank);
    mesh.n_local = n_local;
    mesh.start = start;

    mesh.num_connections.assign(n_local, 0);
    mesh.conn_global.assign(static_cast<size_t>(MAX_CONNECTIONS) * n_local, 0);
    mesh.conn_flux.assign(static_cast<size_t>(MAX_CONNECTIONS) * n_local, 0.0);
    mesh.transfer_coeff.resize(n_local);
    mesh.external_flow.resize(n_local);

    const int64_t last = n_elems_root - 1;
    const int64_t inflow_a = 0;                              // (0, 0)
    const int64_t outflow_a = last;                          // (0, last)
    const int64_t outflow_b = last * n_elems_root;           // (last, 0)
    const int64_t inflow_b = last * n_elems_root + last;     // (last, last)

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_local; ++i) {
        const int64_t g = start + i;
        const int64_t x = g / n_elems_root;
        const int64_t y = g % n_elems_root;

        // Connect to neighbors (up, down, left, right) — same order as serial
        const int64_t offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int nc = 0;
        for (int n = 0; n < 4; ++n) {
            const int64_t nx = x + offsets[n][0];
            const int64_t ny = y + offsets[n][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                mesh.conn_global[static_cast<size_t>(nc) * n_local + i] =
                    nx * n_elems_root + ny;
                mesh.conn_flux[static_cast<size_t>(nc) * n_local + i] = 1.0;
                nc++;
            }
        }
        mesh.num_connections[i] = nc;

        idx_t mat = DEFAULT_MAT_ID;
        if (g == inflow_a || g == inflow_b) mat = INFLOW_MAT_ID;
        else if (g == outflow_a || g == outflow_b) mat = OUTFLOW_MAT_ID;
        mesh.transfer_coeff[i] = materials[mat].transfer_coeff;
        mesh.external_flow[i] = materials[mat].external_flow;
    }
}

// Build the halo exchange plan from the (arbitrary) connectivity and remap the
// global neighbor indices to local/halo indices.
void buildHaloPlan(LocalMesh& mesh, const Decomposition& dec, int size,
                   MPI_Comm comm) {
    const int64_t n_local = mesh.n_local;
    const int64_t start = mesh.start;
    const int64_t end = start + n_local;

    // Collect all off-rank neighbor indices (unique, sorted).
    std::vector<int64_t> remotes;
    {
        const int nthreads = omp_get_max_threads();
        std::vector<std::vector<int64_t>> tl(nthreads);
#pragma omp parallel
        {
            auto& mine = tl[omp_get_thread_num()];
#pragma omp for schedule(static)
            for (int64_t i = 0; i < n_local; ++i) {
                const int nc = mesh.num_connections[i];
                for (int j = 0; j < nc; ++j) {
                    const int64_t g =
                        mesh.conn_global[static_cast<size_t>(j) * n_local + i];
                    if (g < start || g >= end) mine.push_back(g);
                }
            }
        }
        for (auto& v : tl) remotes.insert(remotes.end(), v.begin(), v.end());
        std::sort(remotes.begin(), remotes.end());
        remotes.erase(std::unique(remotes.begin(), remotes.end()), remotes.end());
    }
    mesh.halo_globals = remotes;

    // Per-rank receive counts (remotes are sorted, ownership is monotonic).
    mesh.recv_counts.assign(size, 0);
    for (const int64_t g : remotes) mesh.recv_counts[dec.owner(g)]++;
    mesh.recv_displs.assign(size, 0);
    for (int r = 1; r < size; ++r)
        mesh.recv_displs[r] = mesh.recv_displs[r - 1] + mesh.recv_counts[r - 1];

    // Exchange requested-index lists to learn what we must send.
    mesh.send_counts.assign(size, 0);
    MPI_Alltoall(mesh.recv_counts.data(), 1, MPI_INT, mesh.send_counts.data(), 1,
                 MPI_INT, comm);
    mesh.send_displs.assign(size, 0);
    for (int r = 1; r < size; ++r)
        mesh.send_displs[r] = mesh.send_displs[r - 1] + mesh.send_counts[r - 1];
    const int n_send = mesh.send_displs[size - 1] + mesh.send_counts[size - 1];

    std::vector<int64_t> send_globals(n_send);
    MPI_Alltoallv(remotes.data(), mesh.recv_counts.data(), mesh.recv_displs.data(),
                  MPI_INT64_T, send_globals.data(), mesh.send_counts.data(),
                  mesh.send_displs.data(), MPI_INT64_T, comm);

    mesh.send_local.resize(n_send);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_send; ++k)
        mesh.send_local[k] = static_cast<unsigned int>(send_globals[k] - start);

    // Remap connectivity: owned neighbors -> [0, n_local), halo neighbors ->
    // n_local + position in the sorted halo list.
    mesh.conn_local.assign(static_cast<size_t>(MAX_CONNECTIONS) * n_local, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_local; ++i) {
        const int nc = mesh.num_connections[i];
        for (int j = 0; j < nc; ++j) {
            const size_t slot = static_cast<size_t>(j) * n_local + i;
            const int64_t g = mesh.conn_global[slot];
            if (g >= start && g < end) {
                mesh.conn_local[slot] = static_cast<unsigned int>(g - start);
            } else {
                const auto it = std::lower_bound(mesh.halo_globals.begin(),
                                                 mesh.halo_globals.end(), g);
                mesh.conn_local[slot] = static_cast<unsigned int>(
                    n_local + (it - mesh.halo_globals.begin()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
constexpr int BLOCK_SIZE = 256;

// Gather owned boundary values into the halo send buffer.
__global__ void packKernel(const val_t* __restrict__ energy,
                           const unsigned int* __restrict__ send_local,
                           val_t* __restrict__ sendbuf, int n_send) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_send) sendbuf[k] = energy[send_local[k]];
}

// One simulation step: same flux computation and accumulation order as the
// serial reference (external flow first, then connections in order).
__global__ void stepKernel(int64_t n_local,
                           const val_t* __restrict__ energy_in,
                           const val_t* __restrict__ flux_in,
                           val_t* __restrict__ energy_out,
                           val_t* __restrict__ flux_out,
                           const val_t* __restrict__ transfer_coeff,
                           const val_t* __restrict__ external_flow,
                           const int* __restrict__ num_connections,
                           const unsigned int* __restrict__ conn_local,
                           const val_t* __restrict__ conn_flux) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_local) return;

    const val_t e = energy_in[i];
    const val_t coeff = transfer_coeff[i];
    const int nc = num_connections[i];

    val_t total_flux = external_flow[i];
    for (int j = 0; j < nc; ++j) {
        const size_t slot = static_cast<size_t>(j) * n_local + i;
        const val_t other = energy_in[conn_local[slot]];
        total_flux += (other - e) * coeff * conn_flux[slot] * 0.25;
    }

    energy_out[i] = e + total_flux;
    flux_out[i] = flux_in[i] + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Simulation driver: CUDA update + MPI halo exchange per iteration.
// ---------------------------------------------------------------------------
struct DeviceState {
    val_t* energy[2] = {nullptr, nullptr};  // n_local + n_halo each
    val_t* flux[2] = {nullptr, nullptr};    // n_local each
    val_t* transfer_coeff = nullptr;
    val_t* external_flow = nullptr;
    int* num_connections = nullptr;
    unsigned int* conn_local = nullptr;
    val_t* conn_flux = nullptr;
    unsigned int* send_local = nullptr;
    val_t* sendbuf = nullptr;
    val_t* h_sendbuf = nullptr;  // pinned host staging
    val_t* h_recvbuf = nullptr;
    int cur = 0;
};

void uploadMesh(DeviceState& dev, const LocalMesh& mesh) {
    const int64_t n_local = mesh.n_local;
    const int64_t n_halo = static_cast<int64_t>(mesh.halo_globals.size());
    const int64_t n_send = static_cast<int64_t>(mesh.send_local.size());
    const size_t conn_len = static_cast<size_t>(MAX_CONNECTIONS) * n_local;

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dev.energy[b],
                              std::max<int64_t>(n_local + n_halo, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.energy[b], 0,
                              std::max<int64_t>(n_local + n_halo, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dev.flux[b],
                              std::max<int64_t>(n_local, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(dev.flux[b], 0,
                              std::max<int64_t>(n_local, 1) * sizeof(val_t)));
    }
    if (n_local > 0) {
        CUDA_CHECK(cudaMalloc(&dev.transfer_coeff, n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dev.external_flow, n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dev.num_connections, n_local * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dev.conn_local, conn_len * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dev.conn_flux, conn_len * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(dev.transfer_coeff, mesh.transfer_coeff.data(),
                              n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.external_flow, mesh.external_flow.data(),
                              n_local * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.num_connections, mesh.num_connections.data(),
                              n_local * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.conn_local, mesh.conn_local.data(),
                              conn_len * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.conn_flux, mesh.conn_flux.data(),
                              conn_len * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (n_send > 0) {
        CUDA_CHECK(cudaMalloc(&dev.send_local, n_send * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dev.sendbuf, n_send * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(dev.send_local, mesh.send_local.data(),
                              n_send * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMallocHost(&dev.h_sendbuf, n_send * sizeof(val_t)));
    }
    if (n_halo > 0) {
        CUDA_CHECK(cudaMallocHost(&dev.h_recvbuf, n_halo * sizeof(val_t)));
    }
}

void runSimulation(DeviceState& dev, const LocalMesh& mesh, const int n_iters,
                   int size, MPI_Comm comm) {
    const int64_t n_local = mesh.n_local;
    const int64_t n_halo = static_cast<int64_t>(mesh.halo_globals.size());
    const int n_send = static_cast<int>(mesh.send_local.size());

    // Ranks we actually exchange with (sparse neighborhood).
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * size);

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* e_cur = dev.energy[dev.cur];
        val_t* e_nxt = dev.energy[1 - dev.cur];

        // Pack and stage boundary values, then exchange halos.
        if (n_send > 0) {
            const int blocks = (n_send + BLOCK_SIZE - 1) / BLOCK_SIZE;
            packKernel<<<blocks, BLOCK_SIZE>>>(e_cur, dev.send_local, dev.sendbuf,
                                               n_send);
            CUDA_CHECK(cudaMemcpy(dev.h_sendbuf, dev.sendbuf,
                                  n_send * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        reqs.clear();
        for (int r = 0; r < size; ++r) {
            if (mesh.recv_counts[r] > 0) {
                reqs.emplace_back();
                MPI_Irecv(dev.h_recvbuf + mesh.recv_displs[r], mesh.recv_counts[r],
                          MPI_DOUBLE, r, 0, comm, &reqs.back());
            }
            if (mesh.send_counts[r] > 0) {
                reqs.emplace_back();
                MPI_Isend(dev.h_sendbuf + mesh.send_displs[r], mesh.send_counts[r],
                          MPI_DOUBLE, r, 0, comm, &reqs.back());
            }
        }
        if (!reqs.empty())
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                        MPI_STATUSES_IGNORE);
        if (n_halo > 0) {
            CUDA_CHECK(cudaMemcpy(e_cur + n_local, dev.h_recvbuf,
                                  n_halo * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        // Update all owned elements on the GPU.
        if (n_local > 0) {
            const int blocks =
                static_cast<int>((n_local + BLOCK_SIZE - 1) / BLOCK_SIZE);
            stepKernel<<<blocks, BLOCK_SIZE>>>(
                n_local, e_cur, dev.flux[dev.cur], e_nxt, dev.flux[1 - dev.cur],
                dev.transfer_coeff, dev.external_flow, dev.num_connections,
                dev.conn_local, dev.conn_flux);
        }

        // Swap buffers
        dev.cur = 1 - dev.cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Validation / hashing (identical to the serial reference, run on rank 0)
// ---------------------------------------------------------------------------
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
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind each rank to a GPU (round-robin over the devices on its node).
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &local_comm);
        int local_rank = 0;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
        int n_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_devices));
        CUDA_CHECK(cudaSetDevice(local_rank % n_devices));
    }

    const int64_t n_elems = static_cast<int64_t>(n_elems_root) * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %ld elements\n", n_elems_root, n_elems_root,
               static_cast<long>(n_elems));
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size,
               omp_get_max_threads());
        printf("\n");
    }

    // Build the local slice of the unstructured mesh and the halo plan.
    if (rank == 0) printf("Building unstructured mesh...\n");
    Decomposition dec{n_elems, size, n_elems / size, n_elems % size};
    LocalMesh mesh;
    buildSquare2DLocal(mesh, dec, rank, n_elems_root);
    buildHaloPlan(mesh, dec, size, MPI_COMM_WORLD);

    // Calculate memory usage (global, matching the reference accounting:
    // static = connectivity + material info, dynamic = two state buffers).
    if (rank == 0) {
        const size_t elem_static_bytes =
            2 * sizeof(idx_t) + MAX_CONNECTIONS * (sizeof(idx_t) + sizeof(val_t));
        const size_t static_mem = n_elems * elem_static_bytes;
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // Upload the mesh to the GPU.
    DeviceState dev;
    uploadMesh(dev, mesh);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(dev, mesh, n_iters, size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Fetch local results from the GPU.
    std::vector<val_t> local_energy(std::max<int64_t>(mesh.n_local, 1));
    std::vector<val_t> local_flux(std::max<int64_t>(mesh.n_local, 1));
    if (mesh.n_local > 0) {
        CUDA_CHECK(cudaMemcpy(local_energy.data(), dev.energy[dev.cur],
                              mesh.n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(), dev.flux[dev.cur],
                              mesh.n_local * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    // Gather the full result on rank 0 (contiguous decomposition -> Gatherv
    // reassembles the original element ordering).
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(dec.count(r));
        displs[r] = static_cast<int>(dec.start(r));
    }
    std::vector<val_t> global_energy, global_flux;
    if (rank == 0) {
        global_energy.resize(n_elems);
        global_flux.resize(n_elems);
    }
    MPI_Gatherv(local_energy.data(), static_cast<int>(mesh.n_local), MPI_DOUBLE,
                global_energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(local_flux.data(), static_cast<int>(mesh.n_local), MPI_DOUBLE,
                global_flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

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

        std::vector<ElementDynamic> elements(n_elems);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_elems; ++i) {
            elements[i].current_energy = global_energy[i];
            elements[i].total_flux = global_flux[i];
        }

        // Compute hash for verification
        const uint64_t hash = computeHash(elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(global_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            bool valid = validateResults(elements);
            if (!valid) {
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
