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

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 8;
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

static void cudaCheck(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// The local state includes one ghost row on either side of the owned band.
struct LocalWorld {
    int root, rows, first_row;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements;
    std::vector<val_t> energy, energy_swap, total_flux, total_flux_swap;
};

static void buildLocal(LocalWorld& w, int root, int rank, int ranks) {
    w.root = root;
    w.first_row = (root * rank) / ranks;
    const int last_row = (root * (rank + 1)) / ranks;
    w.rows = last_row - w.first_row;
    const size_t local_slots = static_cast<size_t>(w.rows + 2) * root;
    w.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    w.elements.resize(static_cast<size_t>(w.rows) * root);
    w.energy.assign(local_slots, 0.0);
    w.energy_swap.assign(local_slots, 0.0);
    w.total_flux.assign(local_slots, 0.0);
    w.total_flux_swap.assign(local_slots, 0.0);

    #pragma omp parallel for schedule(static)
    for (int local_row = 0; local_row < w.rows; ++local_row) {
        const int global_row = w.first_row + local_row;
        for (int y = 0; y < root; ++y) {
            const int out = local_row * root + y;
            ElementStatic& e = w.elements[out];
            e.material_idx = DEFAULT_MAT_ID;
            e.num_connections = 0;
            const int global_index = global_row * root + y;
            if (global_index == 0 || global_index == root - 1 ||
                global_index == (root - 1) * root || global_index == root * root - 1) {
                e.material_idx = (global_index == 0 || global_index == root * root - 1)
                    ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            }
            const int dx[4] = {1, -1, 0, 0};
            const int dy[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int nx = global_row + dx[k], ny = y + dy[k];
                if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
                    // Local row zero and rows+1 are the MPI halo rows.
                    e.connected_idx[e.num_connections] =
                        static_cast<idx_t>((nx - w.first_row + 1) * root + ny);
                    e.connected_flux[e.num_connections++] = 1.0;
                }
            }
        }
    }
}

__global__ void updateKernel(const ElementStatic* elements, const Material* materials,
                             const val_t* current, val_t* next, val_t* accumulated,
                             int root, int rows) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(root) * rows;
    if (i >= count) return;
    const ElementStatic& e = elements[i];
    const val_t self = current[(i / root + 1) * root + (i % root)];
    const Material& mat = materials[e.material_idx];
    val_t flux = mat.external_flow;
    for (idx_t k = 0; k < e.num_connections; ++k) {
        const val_t other = current[e.connected_idx[k]];
        flux += (other - self) * mat.transfer_coeff * e.connected_flux[k] * 0.25;
    }
    const size_t slot = (i / root + 1) * root + (i % root);
    next[slot] = self + flux;
    accumulated[slot] += fabs(flux);
}

// Exchange only the rows referenced by cross-rank connectivity.
struct HaloBuffers {
    std::vector<val_t> send_top, send_bottom, recv_top, recv_bottom;
    explicit HaloBuffers(int root) : send_top(root), send_bottom(root), recv_top(root), recv_bottom(root) {}
};

static void exchangeHalos(LocalWorld& w, val_t* d_energy, int rank, int ranks, HaloBuffers& halo) {
    const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int down = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    cudaCheck(cudaMemcpy(halo.send_top.data(), d_energy + w.root, w.root * sizeof(val_t), cudaMemcpyDeviceToHost), "copy top halo");
    cudaCheck(cudaMemcpy(halo.send_bottom.data(), d_energy + static_cast<size_t>(w.rows) * w.root, w.root * sizeof(val_t), cudaMemcpyDeviceToHost), "copy bottom halo");
    MPI_Sendrecv(halo.send_top.data(), w.root, MPI_DOUBLE, up, 101, halo.recv_bottom.data(), w.root, MPI_DOUBLE, down, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(halo.send_bottom.data(), w.root, MPI_DOUBLE, down, 102, halo.recv_top.data(), w.root, MPI_DOUBLE, up, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (up != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_energy, halo.recv_top.data(), w.root * sizeof(val_t), cudaMemcpyHostToDevice), "copy top ghost");
    if (down != MPI_PROC_NULL) cudaCheck(cudaMemcpy(d_energy + static_cast<size_t>(w.rows + 1) * w.root, halo.recv_bottom.data(), w.root * sizeof(val_t), cudaMemcpyHostToDevice), "copy bottom ghost");
}

static void runSimulation(LocalWorld& w, int iterations, int rank, int ranks) {
    const size_t n = w.elements.size(), slots = w.energy.size();
    ElementStatic* d_elements = nullptr; Material* d_materials = nullptr;
    val_t *d_energy = nullptr, *d_swap = nullptr, *d_flux = nullptr;
    cudaCheck(cudaMalloc(&d_elements, n * sizeof(ElementStatic)), "allocate elements");
    cudaCheck(cudaMalloc(&d_materials, w.materials.size() * sizeof(Material)), "allocate materials");
    cudaCheck(cudaMalloc(&d_energy, slots * sizeof(val_t)), "allocate energy");
    cudaCheck(cudaMalloc(&d_swap, slots * sizeof(val_t)), "allocate energy swap");
    cudaCheck(cudaMalloc(&d_flux, slots * sizeof(val_t)), "allocate flux");
    cudaCheck(cudaMemcpy(d_elements, w.elements.data(), n * sizeof(ElementStatic), cudaMemcpyHostToDevice), "upload elements");
    cudaCheck(cudaMemcpy(d_materials, w.materials.data(), w.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "upload materials");
    cudaCheck(cudaMemcpy(d_energy, w.energy.data(), slots * sizeof(val_t), cudaMemcpyHostToDevice), "upload energy");
    cudaCheck(cudaMemset(d_flux, 0, slots * sizeof(val_t)), "clear flux");
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    HaloBuffers halo(w.root);
    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(w, d_energy, rank, ranks, halo);
        updateKernel<<<blocks, threads>>>(d_elements, d_materials, d_energy, d_swap, d_flux, w.root, w.rows);
        cudaCheck(cudaGetLastError(), "launch update kernel");
        cudaCheck(cudaDeviceSynchronize(), "complete update kernel");
        std::swap(d_energy, d_swap);
    }
    cudaCheck(cudaMemcpy(w.energy.data() + w.root, d_energy + w.root, n * sizeof(val_t), cudaMemcpyDeviceToHost), "download energy");
    cudaCheck(cudaMemcpy(w.total_flux.data() + w.root, d_flux + w.root, n * sizeof(val_t), cudaMemcpyDeviceToHost), "download flux");
    cudaFree(d_elements); cudaFree(d_materials); cudaFree(d_energy); cudaFree(d_swap); cudaFree(d_flux);
}

static uint64_t computeHash(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energy.size(); ++i) {
        uint64_t a, b; std::memcpy(&a, &energy[i], sizeof(a)); std::memcpy(&b, &flux[i], sizeof(b));
        hash ^= (a + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (b + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t es = 0.0, fs = 0.0, emin = std::numeric_limits<val_t>::max(), emax = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:es,fs) reduction(min:emin) reduction(max:emax)
    for (size_t i = 0; i < energy.size(); ++i) { es += energy[i]; fs += flux[i]; emin = std::min(emin, energy[i]); emax = std::max(emax, energy[i]); }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", es, fs, emin, emax);
    const bool valid = std::isfinite(es) && std::isfinite(fs) && std::isfinite(emin) && std::isfinite(emax);
    if (std::abs(es) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

static void usage(const char* name) {
    printf("Usage: %s [-n grid] [-i iterations] [-v] [-r] [-h]\n", name);
}

static int localRank(int global_rank) {
    const char* names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"};
    for (const char* name : names) {
        if (const char* value = std::getenv(name)) return atoi(value);
    }
    return global_rank;
}

int main(int argc, char** argv) {
    int provided = 0; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int root = 512, iterations = 10; bool validate = false, emit_results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) root = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) emit_results = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (root < 1 || iterations < 0 || root < ranks) { if (rank == 0) fprintf(stderr, "Grid must be positive and at least the MPI rank count.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int devices = 0; cudaCheck(cudaGetDeviceCount(&devices), "find CUDA devices");
    if (!devices) { if (rank == 0) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank(rank) % devices), "select CUDA device");
    LocalWorld local; buildLocal(local, root, rank, ranks);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n\n", root, root, root * root, iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads(), devices);
        printf("Running hybrid MPI/OpenMP/CUDA simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(local, iterations, rank, ranks);
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_ms = std::chrono::duration<double, std::milli>(end - start).count(), time_ms = [&] { double x = 0; MPI_Reduce(&local_ms, &x, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return x; }();
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { const int a = (root * r) / ranks, b = (root * (r + 1)) / ranks; counts[r] = (b - a) * root; displs[r] = a * root; }
    std::vector<val_t> energy, flux; if (rank == 0) { energy.resize(static_cast<size_t>(root) * root); flux.resize(energy.size()); }
    MPI_Gatherv(local.energy.data() + root, counts[rank], MPI_DOUBLE, rank == 0 ? energy.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local.total_flux.data() + root, counts[rank], MPI_DOUBLE, rank == 0 ? flux.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const int measured = std::max(iterations - 1, 1); const double sec = time_ms / 1000.0;
        printf("Computation time: %.0f ms\nTime per iteration: %.4f ms\nElements/sec: %.4f GigaElements/s\nPerformance: %.4f GFLOPS\nResult hash: %016lX\n", time_ms, time_ms / measured, measured * static_cast<double>(root) * root / sec / 1e9, measured * static_cast<double>(root) * root / sec / 1e9 * 22.0, computeHash(energy, flux));
        if (emit_results) print_results(energy, "ElementEnergy");
        if (validate && !validateResults(energy, flux)) { MPI_Finalize(); return 1; }
    }
    MPI_Finalize(); return 0;
}
