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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

// Dynamic state for each element (used for result hashing / output)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;
constexpr int NUM_MATERIALS = 3;

__constant__ Material d_materials[NUM_MATERIALS];

// Distributed world: this rank owns a contiguous block of grid rows. Elements
// are stored in global order, with one halo row (of n_root elements) on each
// side. Local element index = global index - (row_begin - 1) * n_root.
// Static connectivity is stored structure-of-arrays (slot-major) on the GPU.
struct World {
    int n_root = 0;
    int row_begin = 0, row_end = 0;   // owned global rows [row_begin, row_end)
    size_t n_owned = 0;               // owned elements
    size_t n_local = 0;               // owned + halo elements

    // Device data
    uint8_t* d_mat = nullptr;
    uint8_t* d_ncon = nullptr;
    uint32_t* d_cidx = nullptr;       // [MAX_CONNECTIONS][n_owned]
    val_t* d_cflux = nullptr;         // [MAX_CONNECTIONS][n_owned]
    val_t* d_energy[2] = {nullptr, nullptr};  // [n_local]
    val_t* d_flux[2] = {nullptr, nullptr};    // [n_owned]
};

static idx_t materialFor(int x, int y, int n_root) {
    // Same assignment order as the original (later writes win when n_root == 1)
    const int last = n_root - 1;
    idx_t m = DEFAULT_MAT_ID;
    if (x == 0 && y == 0) m = INFLOW_MAT_ID;
    if (x == 0 && y == last) m = OUTFLOW_MAT_ID;
    if (x == last && y == 0) m = OUTFLOW_MAT_ID;
    if (x == last && y == last) m = INFLOW_MAT_ID;
    return m;
}

// Build this rank's part of a 2D square grid as an unstructured mesh
void buildSquare2D(World& world, const int n_elems_root, const int row_begin, const int row_end) {
    const Material mats[NUM_MATERIALS] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    CUDA_CHECK(cudaMemcpyToSymbol(d_materials, mats, sizeof(mats)));

    world.n_root = n_elems_root;
    world.row_begin = row_begin;
    world.row_end = row_end;
    const size_t n = n_elems_root;
    const size_t rows = row_end - row_begin;
    world.n_owned = rows * n;
    world.n_local = (rows + 2) * n;
    const size_t no = world.n_owned;
    const long long base = (long long)(row_begin - 1) * (long long)n;

    std::vector<uint8_t> h_mat(no), h_ncon(no);
    std::vector<uint32_t> h_cidx(no * MAX_CONNECTIONS, 0);
    std::vector<val_t> h_cflux(no * MAX_CONNECTIONS, 0.0);

    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < (long long)no; ++li) {
        const int x = row_begin + (int)(li / (long long)n);
        const int y = (int)(li % (long long)n);
        h_mat[li] = (uint8_t)materialFor(x, y, n_elems_root);
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int nc = 0;
        for (int k = 0; k < 4; ++k) {
            const int nx = x + offsets[k][0];
            const int ny = y + offsets[k][1];
            if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                const long long g = (long long)nx * n_elems_root + ny;
                h_cidx[(size_t)nc * no + li] = (uint32_t)(g - base);
                h_cflux[(size_t)nc * no + li] = 1.0;
                nc++;
            }
        }
        h_ncon[li] = (uint8_t)nc;
    }

    CUDA_CHECK(cudaMalloc(&world.d_mat, no));
    CUDA_CHECK(cudaMalloc(&world.d_ncon, no));
    CUDA_CHECK(cudaMalloc(&world.d_cidx, no * MAX_CONNECTIONS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&world.d_cflux, no * MAX_CONNECTIONS * sizeof(val_t)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&world.d_energy[b], world.n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&world.d_flux[b], no * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(world.d_energy[b], 0, world.n_local * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(world.d_flux[b], 0, no * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMemcpy(world.d_mat, h_mat.data(), no, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_ncon, h_ncon.data(), no, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_cidx, h_cidx.data(), no * MAX_CONNECTIONS * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(world.d_cflux, h_cflux.data(), no * MAX_CONNECTIONS * sizeof(val_t), cudaMemcpyHostToDevice));
}

void destroyWorld(World& w) {
    cudaFree(w.d_mat); cudaFree(w.d_ncon); cudaFree(w.d_cidx); cudaFree(w.d_cflux);
    for (int b = 0; b < 2; ++b) { cudaFree(w.d_energy[b]); cudaFree(w.d_flux[b]); }
}

// Update owned elements [first, first + count) (owned-relative indices).
// Floating-point contraction is disabled (--fmad=false) to match the reference
// operation order bit for bit.
__global__ void updateKernel(const uint8_t* __restrict__ mat_idx,
                             const uint8_t* __restrict__ ncon,
                             const uint32_t* __restrict__ cidx,
                             const val_t* __restrict__ cflux,
                             const val_t* __restrict__ e_in,
                             const val_t* __restrict__ f_in,
                             val_t* __restrict__ e_out,
                             val_t* __restrict__ f_out,
                             size_t n_owned, size_t halo_off,
                             size_t first, size_t count) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const size_t i = first + t;
    const size_t li = i + halo_off;
    const Material mat = d_materials[mat_idx[i]];
    const val_t e = e_in[li];
    val_t total_flux = mat.external_flow;
    const int nc = ncon[i];
    #pragma unroll
    for (int j = 0; j < MAX_CONNECTIONS; ++j) {
        if (j < nc) {
            const val_t other = e_in[cidx[(size_t)j * n_owned + i]];
            total_flux += (other - e) * mat.transfer_coeff * cflux[(size_t)j * n_owned + i] * 0.25;
        }
    }
    e_out[li] = e + total_flux;
    f_out[i] = f_in[i] + fabs(total_flux);
}

static void launchUpdate(World& w, int buf, size_t first, size_t count, cudaStream_t s) {
    if (count == 0) return;
    const int threads = 256;
    const unsigned blocks = (unsigned)((count + threads - 1) / threads);
    updateKernel<<<blocks, threads, 0, s>>>(w.d_mat, w.d_ncon, w.d_cidx, w.d_cflux,
                                            w.d_energy[buf], w.d_flux[buf],
                                            w.d_energy[buf ^ 1], w.d_flux[buf ^ 1],
                                            w.n_owned, (size_t)w.n_root, first, count);
    CUDA_CHECK(cudaGetLastError());
}

// Run simulation for n_iters iterations. Returns the buffer index holding the result.
// Boundary rows are computed first so their exchange overlaps interior compute.
int runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int down = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    const size_t n = world.n_root;
    const size_t rows = world.row_end - world.row_begin;

    cudaStream_t sb, si;
    CUDA_CHECK(cudaStreamCreate(&sb));
    CUDA_CHECK(cudaStreamCreate(&si));

    val_t *h_send_up, *h_send_dn, *h_recv_up, *h_recv_dn;
    CUDA_CHECK(cudaMallocHost(&h_send_up, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_send_dn, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_recv_up, n * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&h_recv_dn, n * sizeof(val_t)));

    int cur = 0;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int nxt = cur ^ 1;
        const bool exchange = (iter + 1 < n_iters) && size > 1;

        // Boundary rows (first and last owned row)
        launchUpdate(world, cur, 0, n, sb);
        if (rows > 1) launchUpdate(world, cur, (rows - 1) * n, n, sb);
        // Interior rows
        if (rows > 2) launchUpdate(world, cur, n, (rows - 2) * n, si);

        if (exchange) {
            val_t* en = world.d_energy[nxt];
            // local row r (1..rows) holds owned row r-1
            CUDA_CHECK(cudaMemcpyAsync(h_send_up, en + n, n * sizeof(val_t), cudaMemcpyDeviceToHost, sb));
            CUDA_CHECK(cudaMemcpyAsync(h_send_dn, en + rows * n, n * sizeof(val_t), cudaMemcpyDeviceToHost, sb));
            CUDA_CHECK(cudaStreamSynchronize(sb));
            MPI_Request req[4];
            MPI_Irecv(h_recv_up, (int)n, MPI_DOUBLE, up, 0, comm, &req[0]);
            MPI_Irecv(h_recv_dn, (int)n, MPI_DOUBLE, down, 1, comm, &req[1]);
            MPI_Isend(h_send_up, (int)n, MPI_DOUBLE, up, 1, comm, &req[2]);
            MPI_Isend(h_send_dn, (int)n, MPI_DOUBLE, down, 0, comm, &req[3]);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(en, h_recv_up, n * sizeof(val_t), cudaMemcpyHostToDevice, sb));
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(en + (rows + 1) * n, h_recv_dn, n * sizeof(val_t), cudaMemcpyHostToDevice, sb));
        }
        CUDA_CHECK(cudaStreamSynchronize(sb));
        CUDA_CHECK(cudaStreamSynchronize(si));
        cur = nxt;
    }

    cudaFreeHost(h_send_up); cudaFreeHost(h_send_dn);
    cudaFreeHost(h_recv_up); cudaFreeHost(h_recv_dn);
    cudaStreamDestroy(sb);
    cudaStreamDestroy(si);
    return cur;
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

// Compute a simple hash of this rank's results; global index i = first_idx + local.
// XOR-combination makes the per-rank partial hashes composable.
uint64_t computeHash(const val_t* energy, const val_t* flux, size_t count, size_t first_idx) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (long long k = 0; k < (long long)count; ++k) {
        const uint64_t i = first_idx + (size_t)k;
        uint64_t e_bits, f_bits;
        memcpy(&e_bits, &energy[k], sizeof(e_bits));
        memcpy(&f_bits, &flux[k], sizeof(f_bits));
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
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const bool root = (world_rank == 0);

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
    if (n_elems_root < 1) {
        if (root) printf("Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank, round-robin over the GPUs of each node
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
        int local_rank;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        int n_dev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&n_dev));
        CUDA_CHECK(cudaSetDevice(local_rank % n_dev));
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", world_size, omp_get_max_threads());
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Row-block decomposition over the ranks that receive at least one row
    const int active = std::min(world_size, n_elems_root);
    const int rbase = n_elems_root / active, rrem = n_elems_root % active;
    int row_begin = 0, row_end = 0;
    if (world_rank < active) {
        row_begin = world_rank * rbase + std::min(world_rank, rrem);
        row_end = row_begin + rbase + (world_rank < rrem ? 1 : 0);
    }
    MPI_Comm sim_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &sim_comm);

    World world;
    if (world_rank < active) buildSquare2D(world, n_elems_root, row_begin, row_end);
    
    if (root) {
        // Calculate memory usage (total over all ranks, original accounting)
        const size_t static_mem = (size_t)n_elems * (sizeof(idx_t) * 2 + sizeof(idx_t) * MAX_CONNECTIONS + sizeof(val_t) * MAX_CONNECTIONS);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    int res_buf = 0;
    if (world_rank < active) res_buf = runSimulation(world, n_iters, sim_comm);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Fetch owned results to host
    const size_t n_owned = world_rank < active ? world.n_owned : 0;
    std::vector<val_t> h_energy(n_owned), h_flux(n_owned);
    if (n_owned) {
        CUDA_CHECK(cudaMemcpy(h_energy.data(), world.d_energy[res_buf] + world.n_root, n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flux.data(), world.d_flux[res_buf], n_owned * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    const uint64_t local_hash = computeHash(h_energy.data(), h_flux.data(), n_owned, (size_t)row_begin * n_elems_root);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    // Gather full results on rank 0 only if needed
    std::vector<ElementDynamic> elements;
    if (printResults || validate) {
        std::vector<int> counts(world_size, 0), displs(world_size, 0);
        for (int r = 0; r < active; ++r) {
            const int rb = r * rbase + std::min(r, rrem);
            const int re = rb + rbase + (r < rrem ? 1 : 0);
            counts[r] = (re - rb) * n_elems_root;
            displs[r] = rb * n_elems_root;
        }
        std::vector<val_t> all_e, all_f;
        if (root) { all_e.resize(n_elems); all_f.resize(n_elems); }
        MPI_Gatherv(h_energy.data(), (int)n_owned, MPI_DOUBLE, all_e.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(h_flux.data(), (int)n_owned, MPI_DOUBLE, all_f.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            elements.resize(n_elems);
            for (int i = 0; i < n_elems; ++i) elements[i] = {all_e[i], all_f[i]};
        }
    }

    if (world_rank < active) destroyWorld(world);
    int exit_code = 0;

    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = ((double)n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        
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
            std::vector<double> energyData;
            energyData.reserve(elements.size());
            for (const auto& elem : elements) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }
        
        // Validation
        if (validate && !validateResults(elements)) exit_code = 1;
    }

    if (sim_comm != MPI_COMM_NULL) MPI_Comm_free(&sim_comm);
    MPI_Finalize();
    return exit_code;
}
