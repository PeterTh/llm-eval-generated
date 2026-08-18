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
#include <cuda_runtime.h>
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
struct ElementDynamic { val_t current_energy, total_flux; };

struct World {
    int global_root = 0, first_row = 0, local_rows = 0, rank = 0, ranks = 1;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
};

static void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void buildSquare2D(World& world, int root, int rank, int ranks) {
    world.global_root = root; world.rank = rank; world.ranks = ranks;
    const int base = root / ranks, remainder = root % ranks;
    world.local_rows = base + (rank < remainder ? 1 : 0);
    world.first_row = rank * base + std::min(rank, remainder);
    const size_t count = static_cast<size_t>(world.local_rows) * root;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(count);
    world.elements_dynamic.assign(count, {0.0, 0.0});
    world.elements_dynamic_swap.assign(count, {0.0, 0.0});

    #pragma omp parallel for schedule(static)
    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int x = world.first_row + local_x;
        for (int y = 0; y < root; ++y) {
            const size_t i = static_cast<size_t>(local_x) * root + y;
            auto& e = world.elements_static[i];
            e.material_idx = DEFAULT_MAT_ID; e.num_connections = 0;
            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0], ny = y + offset[1];
                if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
                    // Local indices include one ghost row below the owned rows.
                    const int owner = (nx < (base + 1) * remainder)
                        ? nx / (base + 1)
                        : remainder + (nx - (base + 1) * remainder) / base;
                    if (owner == rank) {
                        const int lx = nx - world.first_row;
                        e.connected_idx[e.num_connections] = static_cast<idx_t>(lx * root + ny + root);
                    } else {
                        // Filled with the appropriate ghost-row index below.
                        e.connected_idx[e.num_connections] = static_cast<idx_t>((nx < world.first_row ? 0 : world.local_rows + 1) * root + ny);
                    }
                    e.connected_flux[e.num_connections++] = 1.0;
                }
            }
            if ((x == 0 && y == 0) || (x == root - 1 && y == root - 1)) e.material_idx = INFLOW_MAT_ID;
            if ((x == 0 && y == root - 1) || (x == root - 1 && y == 0)) e.material_idx = OUTFLOW_MAT_ID;
        }
    }
}

__global__ void updateKernel(const ElementStatic* statics, const Material* materials,
                             const ElementDynamic* current, ElementDynamic* next,
                             int count, int root) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const ElementStatic& s = statics[i];
    const ElementDynamic old = current[i + root];
    const Material& mat = materials[s.material_idx];
    val_t flux = mat.external_flow;
    #pragma unroll
    for (idx_t j = 0; j < s.num_connections; ++j) {
        const val_t neighbour = current[s.connected_idx[j]].current_energy;
        flux += (neighbour - old.current_energy) * mat.transfer_coeff * s.connected_flux[j] * 0.25;
    }
    next[i + root].current_energy = old.current_energy + flux;
    next[i + root].total_flux = old.total_flux + fabs(flux);
}

static void exchangeHalos(World& world, std::vector<ElementDynamic>& state) {
    const int root = world.global_root;
    const int up = world.rank == 0 ? MPI_PROC_NULL : world.rank - 1;
    const int down = world.rank + 1 == world.ranks ? MPI_PROC_NULL : world.rank + 1;
    MPI_Status status;
    mpiCheck(MPI_Sendrecv(state.data() + root, root * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                          up, 41, state.data() + static_cast<size_t>(world.local_rows + 1) * root,
                          root * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, down, 41,
                          MPI_COMM_WORLD, &status), "top halo exchange");
    mpiCheck(MPI_Sendrecv(state.data() + static_cast<size_t>(world.local_rows) * root,
                          root * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE, down, 42,
                          state.data(), root * static_cast<int>(sizeof(ElementDynamic)), MPI_BYTE,
                          up, 42, MPI_COMM_WORLD, &status), "bottom halo exchange");
}

void runSimulation(World& world, int iterations) {
    const int root = world.global_root;
    const int count = world.local_rows * root;
    ElementStatic* d_static = nullptr; Material* d_materials = nullptr;
    ElementDynamic *d_current = nullptr, *d_swap = nullptr;
    cudaCheck(cudaMalloc(&d_static, count * sizeof(ElementStatic)), "allocate static mesh");
    cudaCheck(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)), "allocate materials");
    cudaCheck(cudaMalloc(&d_current, (world.local_rows + 2) * root * sizeof(ElementDynamic)), "allocate state");
    cudaCheck(cudaMalloc(&d_swap, (world.local_rows + 2) * root * sizeof(ElementDynamic)), "allocate state swap");
    cudaCheck(cudaMemcpy(d_static, world.elements_static.data(), count * sizeof(ElementStatic), cudaMemcpyHostToDevice), "copy mesh");
    cudaCheck(cudaMemcpy(d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "copy materials");

    std::vector<ElementDynamic> host((world.local_rows + 2) * root), next(host.size());
    std::copy(world.elements_dynamic.begin(), world.elements_dynamic.end(), host.begin() + root);
    cudaCheck(cudaMemcpy(d_current, host.data(), host.size() * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "copy initial state");
    const int threads = 256, blocks = (count + threads - 1) / threads;
    for (int iter = 0; iter < iterations; ++iter) {
        cudaCheck(cudaMemcpy(host.data(), d_current, host.size() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "download halo state");
        exchangeHalos(world, host);
        cudaCheck(cudaMemcpy(d_current, host.data(), host.size() * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "upload halo state");
        updateKernel<<<blocks, threads>>>(d_static, d_materials, d_current, d_swap, count, root);
        cudaCheck(cudaGetLastError(), "launch update kernel");
        std::swap(d_current, d_swap);
    }
    cudaCheck(cudaDeviceSynchronize(), "finish update kernel");
    cudaCheck(cudaMemcpy(host.data(), d_current, host.size() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "download final state");
    std::copy(host.begin() + root, host.begin() + root + count, world.elements_dynamic.begin());
    cudaFree(d_static); cudaFree(d_materials); cudaFree(d_current); cudaFree(d_swap);
}

bool validateResults(const World& world) {
    double local_energy = 0.0, local_flux = 0.0, local_min = std::numeric_limits<double>::max();
    double local_max = std::numeric_limits<double>::lowest();
    #pragma omp parallel for reduction(+:local_energy,local_flux) reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& e = world.elements_dynamic[i]; local_energy += e.current_energy; local_flux += e.total_flux;
        local_min = std::min(local_min, e.current_energy); local_max = std::max(local_max, e.current_energy);
    }
    double energy = 0, flux = 0, maximum = 0, minimum = 0;
    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (world.rank != 0) return true;
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy, flux, minimum, maximum);
    if (!std::isfinite(energy) || !std::isfinite(flux) || !std::isfinite(minimum) || !std::isfinite(maximum)) {
        printf("  ERROR: non-finite result\n"); return false;
    }
    if (std::abs(energy) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: PASSED\n"); return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t result = 0;
    #pragma omp parallel for reduction(^:result)
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto* e = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const auto* f = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        result ^= (*e + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (*f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

void printUsage(const char* name) { printf("Usage: %s [-n num] [-i num] [-v] [-r] [-h]\n", name); }

int main(int argc, char** argv) {
    int initialized = 0; mpiCheck(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized) mpiCheck(MPI_Init(&argc, &argv), "MPI_Init");
    int rank = 0, ranks = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm = MPI_COMM_NULL; int local_rank = 0;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm), "split shared communicator");
    MPI_Comm_rank(local_comm, &local_rank);
    int root = 512, iterations = 10; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) root = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Comm_free(&local_comm); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Comm_free(&local_comm); MPI_Finalize(); return 1; }
    }
    if (root <= 0 || iterations < 0 || ranks > root) { if (rank == 0) fprintf(stderr, "Invalid grid, iterations, or MPI decomposition\n"); MPI_Comm_free(&local_comm); MPI_Finalize(); return 1; }
    int device_count = 0; cudaCheck(cudaGetDeviceCount(&device_count), "discover CUDA devices");
    cudaCheck(cudaSetDevice(local_rank % device_count), "select rank-local CUDA device");
    World world; buildSquare2D(world, root, rank, ranks);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", root, root, root * root, iterations, validate ? "enabled" : "disabled");
        const size_t static_mem = static_cast<size_t>(root) * root * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(root) * root * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\nRunning simulation with %d MPI rank(s), %d OpenMP thread(s), and CUDA\n", (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0, ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD); auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, iterations);
    MPI_Barrier(MPI_COMM_WORLD); auto end = std::chrono::high_resolution_clock::now();
    const long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (rank == 0) {
        const int measured = std::max(iterations - 1, 1); const double seconds = duration_ms / 1000.0;
        const double geps = measured * static_cast<double>(root) * root / seconds / 1e9;
        printf("Computation time: %ld ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", duration_ms, duration_ms / static_cast<double>(measured), geps, geps * 22.0);
    }
    // Gather in global row order so the original hash and -r output remain unchanged.
    std::vector<ElementDynamic> all;
    if (rank == 0) all.resize(static_cast<size_t>(root) * root);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) { const int b = root / ranks, rem = root % ranks; counts[r] = (b + (r < rem)) * root * static_cast<int>(sizeof(ElementDynamic)); displacements[r] = (r * b + std::min(r, rem)) * root * static_cast<int>(sizeof(ElementDynamic)); }
    mpiCheck(MPI_Gatherv(world.elements_dynamic.data(), counts[rank], MPI_BYTE, all.data(), counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD), "gather results");
    if (rank == 0) {
        printf("  Result hash: %016lX\n\n", computeHash(all));
        if (print_results_flag) { std::vector<double> energies; energies.reserve(all.size()); for (const auto& e : all) energies.push_back(e.current_energy); print_results(energies, "ElementEnergy"); }
    }
    const bool valid = validateResults(world); MPI_Comm_free(&local_comm); MPI_Finalize(); return valid ? 0 : 1;
}
