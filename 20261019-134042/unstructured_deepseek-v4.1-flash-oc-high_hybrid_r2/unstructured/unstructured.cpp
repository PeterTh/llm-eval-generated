#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

__constant__ val_t c_transfer_coeff[MAX_CONNECTIONS];
__constant__ val_t c_external_flow[MAX_CONNECTIONS];

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err = (call);                                                       \
        if (err != MPI_SUCCESS) {                                               \
            char msg[MPI_MAX_ERROR_STRING];                                     \
            int len = 0;                                                        \
            MPI_Error_string(err, msg, &len);                                   \
            std::fprintf(stderr, "MPI error: %.*s (%s:%d)\n", len, msg,         \
                         __FILE__, __LINE__);                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

struct Decomp {
    int rank;
    int size;
    int n;
    int x0;
    int x1;
    int lx;
};

struct DeviceMesh {
    uint32_t* material = nullptr;
    uint32_t* num_connections = nullptr;
    idx_t* connected_idx = nullptr;
    val_t* connected_flux = nullptr;
    val_t* energy = nullptr;
    val_t* energy_swap = nullptr;
    val_t* flux = nullptr;
    val_t* flux_swap = nullptr;
    int local_elems = 0;
    int ext_elems = 0;
};

struct HaloBuffers {
    val_t* send_first = nullptr;
    val_t* send_last = nullptr;
    val_t* recv_first = nullptr;
    val_t* recv_last = nullptr;
};

static Decomp makeDecomp(const int n, const int rank, const int size) {
    const int base = n / size;
    const int rem = n % size;
    const int x0 = rank * base + std::min(rank, rem);
    const int lx = base + (rank < rem ? 1 : 0);
    return Decomp{rank, size, n, x0, x0 + lx, lx};
}

__global__ __launch_bounds__(256)
void simulateKernel(const uint32_t* __restrict__ material,
                    const uint32_t* __restrict__ num_connections,
                    const idx_t* __restrict__ connected_idx,
                    const val_t* __restrict__ connected_flux,
                    const val_t* __restrict__ energy_in,
                    val_t* __restrict__ energy_out,
                    const val_t* __restrict__ flux_in,
                    val_t* __restrict__ flux_out,
                    const int local_elems,
                    const int row_offset) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_elems) {
        return;
    }

    const int o = i + row_offset;
    const uint32_t mat = material[i];
    const uint32_t nc = num_connections[i];
    const val_t coeff = c_transfer_coeff[mat];
    const val_t e = energy_in[o];

    val_t total_flux = c_external_flow[mat];
    for (uint32_t j = 0; j < nc; ++j) {
        const size_t k = static_cast<size_t>(j) *
                         static_cast<size_t>(local_elems) +
                         static_cast<size_t>(i);
        const idx_t neighbor = connected_idx[k];
        const val_t conn_flux = connected_flux[k];
        total_flux += (energy_in[neighbor] - e) * coeff * conn_flux * 0.25;
    }

    energy_out[o] = e + total_flux;
    flux_out[o] = flux_in[o] + std::fabs(total_flux);
}

static void buildLocalMesh(const Decomp& d,
                           const std::vector<Material>& materials,
                           std::vector<uint32_t>& material,
                           std::vector<uint32_t>& num_connections,
                           std::vector<idx_t>& connected_idx,
                           std::vector<val_t>& connected_flux,
                           std::vector<val_t>& transfer_coeff,
                           std::vector<val_t>& external_flow) {
    const int n = d.n;
    const int local_elems = d.lx * n;
    const size_t stride = static_cast<size_t>(local_elems);

    material.resize(static_cast<size_t>(local_elems));
    num_connections.resize(static_cast<size_t>(local_elems));
    connected_idx.resize(static_cast<size_t>(MAX_CONNECTIONS) * stride);
    connected_flux.resize(static_cast<size_t>(MAX_CONNECTIONS) * stride);

    transfer_coeff.resize(materials.size());
    external_flow.resize(materials.size());
    for (size_t m = 0; m < materials.size(); ++m) {
        transfer_coeff[m] = materials[m].transfer_coeff;
        external_flow[m] = materials[m].external_flow;
    }

    const int last = n - 1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_elems; ++i) {
        const int r = i / n;
        const int y = i - r * n;
        const int x = d.x0 + r;

        uint32_t mat = static_cast<uint32_t>(DEFAULT_MAT_ID);
        if (x == 0 && y == 0) {
            mat = static_cast<uint32_t>(INFLOW_MAT_ID);
        } else if (x == 0 && y == last) {
            mat = static_cast<uint32_t>(OUTFLOW_MAT_ID);
        } else if (x == last && y == 0) {
            mat = static_cast<uint32_t>(OUTFLOW_MAT_ID);
        } else if (x == last && y == last) {
            mat = static_cast<uint32_t>(INFLOW_MAT_ID);
        }
        material[i] = mat;

        uint32_t nc = 0;
        const size_t row = static_cast<size_t>(r + 1);
        const size_t base = static_cast<size_t>(i);

        if (x + 1 < n) {
            connected_idx[static_cast<size_t>(nc) * stride + base] =
                static_cast<idx_t>((row + 1) * n + y);
            connected_flux[static_cast<size_t>(nc) * stride + base] = 1.0;
            ++nc;
        }
        if (x - 1 >= 0) {
            connected_idx[static_cast<size_t>(nc) * stride + base] =
                static_cast<idx_t>((row - 1) * n + y);
            connected_flux[static_cast<size_t>(nc) * stride + base] = 1.0;
            ++nc;
        }
        if (y + 1 < n) {
            connected_idx[static_cast<size_t>(nc) * stride + base] =
                static_cast<idx_t>(row * n + (y + 1));
            connected_flux[static_cast<size_t>(nc) * stride + base] = 1.0;
            ++nc;
        }
        if (y - 1 >= 0) {
            connected_idx[static_cast<size_t>(nc) * stride + base] =
                static_cast<idx_t>(row * n + (y - 1));
            connected_flux[static_cast<size_t>(nc) * stride + base] = 1.0;
            ++nc;
        }
        num_connections[i] = nc;
    }
}

static void uploadMesh(DeviceMesh& dm,
                       const std::vector<uint32_t>& material,
                       const std::vector<uint32_t>& num_connections,
                       const std::vector<idx_t>& connected_idx,
                       const std::vector<val_t>& connected_flux,
                       const std::vector<val_t>& transfer_coeff,
                       const std::vector<val_t>& external_flow,
                       const int local_elems,
                       const int ext_elems) {
    dm.local_elems = local_elems;
    dm.ext_elems = ext_elems;

    const size_t mat_bytes = static_cast<size_t>(local_elems) * sizeof(uint32_t);
    const size_t conn_bytes = static_cast<size_t>(MAX_CONNECTIONS) *
                              static_cast<size_t>(local_elems) * sizeof(idx_t);
    const size_t flux_bytes = static_cast<size_t>(MAX_CONNECTIONS) *
                              static_cast<size_t>(local_elems) * sizeof(val_t);
    const size_t dyn_bytes = static_cast<size_t>(ext_elems) * sizeof(val_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.material),
                          mat_bytes ? mat_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.num_connections),
                          mat_bytes ? mat_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.connected_idx),
                          conn_bytes ? conn_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.connected_flux),
                          flux_bytes ? flux_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.energy),
                          dyn_bytes ? dyn_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.energy_swap),
                          dyn_bytes ? dyn_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.flux),
                          dyn_bytes ? dyn_bytes : 1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dm.flux_swap),
                          dyn_bytes ? dyn_bytes : 1));

    if (local_elems > 0) {
        CUDA_CHECK(cudaMemcpy(dm.material, material.data(), mat_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm.num_connections, num_connections.data(),
                              mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm.connected_idx, connected_idx.data(),
                              conn_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dm.connected_flux, connected_flux.data(),
                              flux_bytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(dm.energy, 0, dyn_bytes));
    CUDA_CHECK(cudaMemset(dm.energy_swap, 0, dyn_bytes));
    CUDA_CHECK(cudaMemset(dm.flux, 0, dyn_bytes));
    CUDA_CHECK(cudaMemset(dm.flux_swap, 0, dyn_bytes));

    const size_t num_mats = transfer_coeff.size();
    CUDA_CHECK(cudaMemcpyToSymbol(c_transfer_coeff, transfer_coeff.data(),
                                  num_mats * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_external_flow, external_flow.data(),
                                  num_mats * sizeof(val_t)));
}

static void freeMesh(DeviceMesh& dm) {
    if (dm.material) cudaFree(dm.material);
    if (dm.num_connections) cudaFree(dm.num_connections);
    if (dm.connected_idx) cudaFree(dm.connected_idx);
    if (dm.connected_flux) cudaFree(dm.connected_flux);
    if (dm.energy) cudaFree(dm.energy);
    if (dm.energy_swap) cudaFree(dm.energy_swap);
    if (dm.flux) cudaFree(dm.flux);
    if (dm.flux_swap) cudaFree(dm.flux_swap);
    dm = DeviceMesh{};
}

static void allocHaloBuffers(HaloBuffers& hb, const Decomp& d) {
    if (d.size <= 1 || d.n <= 0) {
        return;
    }
    const size_t bytes = static_cast<size_t>(d.n) * sizeof(val_t);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hb.send_first), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hb.send_last), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hb.recv_first), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hb.recv_last), bytes));
}

static void freeHaloBuffers(HaloBuffers& hb) {
    if (hb.send_first) cudaFreeHost(hb.send_first);
    if (hb.send_last) cudaFreeHost(hb.send_last);
    if (hb.recv_first) cudaFreeHost(hb.recv_first);
    if (hb.recv_last) cudaFreeHost(hb.recv_last);
    hb = HaloBuffers{};
}

static void exchangeHalos(const Decomp& d, DeviceMesh& dm,
                          HaloBuffers& hb, MPI_Comm comm) {
    if (d.size <= 1 || d.lx == 0 || d.n <= 0) {
        return;
    }

    const int prev = d.rank - 1;
    const int next = d.rank + 1;
    const bool has_prev = d.x0 > 0;
    const bool has_next = d.x1 < d.n;
    const int count = d.n;
    const size_t bytes = static_cast<size_t>(count) * sizeof(val_t);

    constexpr int TAG_FIRST = 10;
    constexpr int TAG_LAST = 11;

    if (has_prev) {
        CUDA_CHECK(cudaMemcpy(hb.send_first,
                              dm.energy + static_cast<size_t>(d.n),
                              bytes, cudaMemcpyDeviceToHost));
    }
    if (has_next) {
        CUDA_CHECK(cudaMemcpy(hb.send_last,
                              dm.energy + static_cast<size_t>(d.lx) * d.n,
                              bytes, cudaMemcpyDeviceToHost));
    }

    MPI_Request reqs[4];
    int nreq = 0;
    if (has_prev) {
        MPI_CHECK(MPI_Irecv(hb.recv_first, count, MPI_DOUBLE, prev,
                            TAG_LAST, comm, &reqs[nreq++]));
    }
    if (has_next) {
        MPI_CHECK(MPI_Irecv(hb.recv_last, count, MPI_DOUBLE, next,
                            TAG_FIRST, comm, &reqs[nreq++]));
    }
    if (has_prev) {
        MPI_CHECK(MPI_Isend(hb.send_first, count, MPI_DOUBLE, prev,
                            TAG_FIRST, comm, &reqs[nreq++]));
    }
    if (has_next) {
        MPI_CHECK(MPI_Isend(hb.send_last, count, MPI_DOUBLE, next,
                            TAG_LAST, comm, &reqs[nreq++]));
    }

    MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

    if (has_prev) {
        CUDA_CHECK(cudaMemcpy(dm.energy, hb.recv_first, bytes,
                              cudaMemcpyHostToDevice));
    }
    if (has_next) {
        CUDA_CHECK(cudaMemcpy(dm.energy +
                              static_cast<size_t>(d.lx + 1) * d.n,
                              hb.recv_last, bytes, cudaMemcpyHostToDevice));
    }
}

static void runSimulation(const Decomp& d, DeviceMesh& dm,
                          HaloBuffers& hb, const int n_iters,
                          MPI_Comm comm) {
    constexpr int threads = 256;
    const int blocks = (dm.local_elems + threads - 1) / threads;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (dm.local_elems > 0) {
            simulateKernel<<<blocks, threads>>>(
                dm.material, dm.num_connections, dm.connected_idx,
                dm.connected_flux, dm.energy, dm.energy_swap,
                dm.flux, dm.flux_swap, dm.local_elems, d.n);
            CUDA_CHECK(cudaGetLastError());
        }

        std::swap(dm.energy, dm.energy_swap);
        std::swap(dm.flux, dm.flux_swap);

        exchangeHalos(d, dm, hb, comm);
    }
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0;
    int size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    const int n_elems = n_elems_root * n_elems_root;
    const Decomp decomp = makeDecomp(n_elems_root, rank, size);
    const int local_elems = decomp.lx * n_elems_root;
    const int ext_elems = (decomp.lx + 2) * n_elems_root;

    int local_rank = 0;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");
    }

    std::vector<Material> materials;
    materials.emplace_back(Material{0.8, 0.0});
    materials.emplace_back(Material{0.8, 0.5});
    materials.emplace_back(Material{0.8, -0.5});

    std::vector<uint32_t> material;
    std::vector<uint32_t> num_connections;
    std::vector<idx_t> connected_idx;
    std::vector<val_t> connected_flux;
    std::vector<val_t> transfer_coeff;
    std::vector<val_t> external_flow;

    buildLocalMesh(decomp, materials, material, num_connections,
                   connected_idx, connected_flux, transfer_coeff,
                   external_flow);

    DeviceMesh dm;
    uploadMesh(dm, material, num_connections, connected_idx,
               connected_flux, transfer_coeff, external_flow,
               local_elems, ext_elems);

    HaloBuffers hb;
    allocHaloBuffers(hb, decomp);

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

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(decomp, dm, hb, n_iters, MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();

    double local_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start).count();
    double duration_ms = local_duration_ms;
    MPI_CHECK(MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &local_duration_ms,
                         &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    std::vector<val_t> local_energy;
    std::vector<val_t> local_flux;
    if (local_elems > 0) {
        local_energy.resize(static_cast<size_t>(local_elems));
        local_flux.resize(static_cast<size_t>(local_elems));
        CUDA_CHECK(cudaMemcpy(local_energy.data(),
                              dm.energy + static_cast<size_t>(n_elems_root),
                              static_cast<size_t>(local_elems) * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_flux.data(),
                              dm.flux + static_cast<size_t>(n_elems_root),
                              static_cast<size_t>(local_elems) * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> recvcounts(static_cast<size_t>(size));
    std::vector<int> displs(static_cast<size_t>(size));
    for (int r = 0; r < size; ++r) {
        const Decomp rd = makeDecomp(n_elems_root, r, size);
        recvcounts[r] = rd.lx * n_elems_root;
        displs[r] = rd.x0 * n_elems_root;
    }

    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (rank == 0) {
        global_energy.resize(static_cast<size_t>(n_elems));
        global_flux.resize(static_cast<size_t>(n_elems));
    }

    val_t dummy = 0.0;
    val_t* send_energy = local_elems > 0 ? local_energy.data() : &dummy;
    val_t* send_flux = local_elems > 0 ? local_flux.data() : &dummy;

    MPI_CHECK(MPI_Gatherv(send_energy, local_elems, MPI_DOUBLE,
                          rank == 0 ? global_energy.data() : nullptr,
                          recvcounts.data(), displs.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gatherv(send_flux, local_elems, MPI_DOUBLE,
                          rank == 0 ? global_flux.data() : nullptr,
                          recvcounts.data(), displs.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));

    bool valid = true;
    if (rank == 0) {
        std::vector<ElementDynamic> global_dynamic(static_cast<size_t>(n_elems));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_elems; ++i) {
            global_dynamic[i].current_energy = global_energy[i];
            global_dynamic[i].total_flux = global_flux[i];
        }

        printf("Computation time: %lld ms\n",
               static_cast<long long>(duration_ms));

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double measured_ms = duration_ms > 0.0 ? duration_ms : 1e-9;
        const double time_per_iter =
            measured_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            (measured_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t result_hash = computeHash(global_dynamic);
        printf("  Result hash: %016lX\n", result_hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dynamic.size());
            for (const auto& elem : global_dynamic) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            World world;
            world.elements_dynamic = std::move(global_dynamic);
            valid = validateResults(world);
        }
    }

    int valid_int = valid ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD));

    freeHaloBuffers(hb);
    freeMesh(dm);

    MPI_CHECK(MPI_Finalize());

    return valid_int ? 0 : 1;
}
