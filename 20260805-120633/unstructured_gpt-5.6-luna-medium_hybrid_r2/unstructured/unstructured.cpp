#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 8;

struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy, total_flux; };
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

constexpr idx_t DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void buildSquare2D(World& world, int root) {
    const size_t n = static_cast<size_t>(root) * root;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(n);
    world.elements_dynamic.assign(n, {0.0, 0.0});
    world.elements_dynamic_swap.assign(n, {0.0, 0.0});

#pragma omp parallel for schedule(static)
    for (int x = 0; x < root; ++x) {
        for (int y = 0; y < root; ++y) {
            const size_t i = static_cast<size_t>(x) * root + y;
            auto& e = world.elements_static[i];
            e.material_idx = DEFAULT_MAT_ID;
            e.num_connections = 0;
            constexpr int dx[4] = {1, -1, 0, 0};
            constexpr int dy[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dx[k], ny = y + dy[k];
                if (nx >= 0 && nx < root && ny >= 0 && ny < root) {
                    const int c = static_cast<int>(e.num_connections++);
                    e.connected_idx[c] = static_cast<idx_t>(nx * root + ny);
                    e.connected_flux[c] = 1.0;
                }
            }
        }
    }
    const int last = root - 1;
    world.elements_static[0].material_idx = INFLOW_MAT_ID;
    world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[static_cast<size_t>(last) * root].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[n - 1].material_idx = INFLOW_MAT_ID;
}

__global__ void updateKernel(const ElementStatic* __restrict__ stat,
                             const Material* __restrict__ mats,
                             const ElementDynamic* __restrict__ current,
                             ElementDynamic* __restrict__ next,
                             size_t begin, size_t end) {
    const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= end) return;
    const ElementStatic e = stat[i];
    const ElementDynamic old = current[i];
    const Material mat = mats[e.material_idx];
    val_t flux = mat.external_flow;
#pragma unroll
    for (idx_t j = 0; j < e.num_connections; ++j) {
        const ElementDynamic neighbor = current[e.connected_idx[j]];
        flux += (neighbor.current_energy - old.current_energy) *
                mat.transfer_coeff * e.connected_flux[j] * 0.25;
    }
    next[i].current_energy = old.current_energy + flux;
    next[i].total_flux = old.total_flux + fabs(flux);
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num> grid size (default: 512)\n"
                "  -i <num> iterations (default: 10)\n  -v validate\n"
                "  -r print results\n  -h help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int root = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) root = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (root <= 0 || iterations < 0 || ranks > root) {
        if (rank == 0) std::fprintf(stderr, "Require grid size > 0, iterations >= 0, and MPI ranks <= grid rows.\n");
        MPI_Finalize(); return 1;
    }
    const size_t n = static_cast<size_t>(root) * root;
    const size_t first_row = static_cast<size_t>(rank) * root / ranks;
    const size_t last_row = static_cast<size_t>(rank + 1) * root / ranks;
    const size_t begin = first_row * root, end = last_row * root;
    const size_t local_count = end - begin;
    const int device_count = [] { int c = 0; cudaGetDeviceCount(&c); return c; }();
    if (device_count <= 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    checkCuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    World world;
    buildSquare2D(world, root);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n",
                     root, root, n, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n\n",
                     ranks, omp_get_max_threads(), device_count);
    }

    Material* d_materials = nullptr;
    ElementStatic* d_static = nullptr;
    ElementDynamic *d_current = nullptr, *d_next = nullptr;
    checkCuda(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)), "cudaMalloc materials");
    checkCuda(cudaMalloc(&d_static, n * sizeof(ElementStatic)), "cudaMalloc connectivity");
    checkCuda(cudaMalloc(&d_current, n * sizeof(ElementDynamic)), "cudaMalloc current");
    checkCuda(cudaMalloc(&d_next, n * sizeof(ElementDynamic)), "cudaMalloc next");
    checkCuda(cudaMemcpy(d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "copy materials");
    checkCuda(cudaMemcpy(d_static, world.elements_static.data(), n * sizeof(ElementStatic), cudaMemcpyHostToDevice), "copy connectivity");
    checkCuda(cudaMemset(d_current, 0, n * sizeof(ElementDynamic)), "clear current");
    checkCuda(cudaMemset(d_next, 0, n * sizeof(ElementDynamic)), "clear next");

    std::vector<val_t> recv_energy(root);
    const int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int following = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    constexpr int block_size = 256;
    for (int iter = 0; iter < iterations; ++iter) {
        if (local_count) {
            const int blocks = static_cast<int>((local_count + block_size - 1) / block_size);
            updateKernel<<<blocks, block_size>>>(d_static, d_materials, d_current, d_next, begin, end);
            checkCuda(cudaGetLastError(), "updateKernel launch");
            checkCuda(cudaDeviceSynchronize(), "updateKernel execution");
        }
        std::swap(d_current, d_next);

        // Exchange only the two rows which can be read by a neighboring rank.
        // Use separate sends for the two boundaries; this also handles one-row partitions.
        std::vector<val_t> lower(root), upper(root);
        if (rank > 0) checkCuda(cudaMemcpy(lower.data(), d_current + begin, root * sizeof(val_t), cudaMemcpyDeviceToHost), "copy lower boundary");
        if (rank + 1 < ranks) checkCuda(cudaMemcpy(upper.data(), d_current + end - root, root * sizeof(val_t), cudaMemcpyDeviceToHost), "copy upper boundary");
        MPI_Sendrecv(lower.data(), root, MPI_DOUBLE, previous, 41, recv_energy.data(), root, MPI_DOUBLE, following, 41, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank + 1 < ranks) checkCuda(cudaMemcpy(d_current + end, recv_energy.data(), root * sizeof(val_t), cudaMemcpyHostToDevice), "install upper halo");
        MPI_Sendrecv(upper.data(), root, MPI_DOUBLE, following, 42, recv_energy.data(), root, MPI_DOUBLE, previous, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > 0) checkCuda(cudaMemcpy(d_current + begin - root, recv_energy.data(), root * sizeof(val_t), cudaMemcpyHostToDevice), "install lower halo");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, worst_elapsed = 0.0;
    MPI_Reduce(&elapsed, &worst_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed result in global order for validation, hashing, and -r.
    std::vector<ElementDynamic> local(local_count), result;
    if (local_count) checkCuda(cudaMemcpy(local.data(), d_current + begin, local_count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "copy result");
    std::vector<int> counts(ranks), displacements(ranks);
    const int local_bytes = static_cast<int>(local_count * sizeof(ElementDynamic));
    MPI_Gather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) { int offset = 0; for (int r = 0; r < ranks; ++r) { displacements[r] = offset; offset += counts[r]; } result.resize(n); }
    MPI_Gatherv(local.data(), local_bytes, MPI_BYTE, result.data(), counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int measured = std::max(iterations - 1, 1);
        const double ms = worst_elapsed * 1000.0;
        const double geps = measured * static_cast<double>(n) / (worst_elapsed * 1e9);
        std::printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
                     ms, ms / measured, geps, geps * 22.0);
        uint64_t hash = 0;
#pragma omp parallel for reduction(^:hash) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const uint64_t* e = reinterpret_cast<const uint64_t*>(&result[i].current_energy);
            const uint64_t* f = reinterpret_cast<const uint64_t*>(&result[i].total_flux);
            hash ^= (*e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f + i) * 0xbf58476d1ce4e5b9ULL;
        }
        std::printf("  Result hash: %016lX\n\n", hash);
        if (printResults) {
            std::vector<double> energy(n);
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i) energy[i] = result[i].current_energy;
            print_results(energy, "ElementEnergy");
        }
        if (validate) {
            double energy_sum = 0.0, flux_sum = 0.0;
            double energy_min = std::numeric_limits<double>::max();
            double energy_max = std::numeric_limits<double>::lowest();
#pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(min:energy_min) reduction(max:energy_max)
            for (size_t i = 0; i < n; ++i) { energy_sum += result[i].current_energy; flux_sum += result[i].total_flux; energy_min = std::min(energy_min, result[i].current_energy); energy_max = std::max(energy_max, result[i].current_energy); }
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
            if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) || !std::isfinite(energy_min) || !std::isfinite(energy_max)) { std::printf("  Validation: FAILED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            std::printf("  Validation: PASSED\n");
        }
    }
    cudaFree(d_materials); cudaFree(d_static); cudaFree(d_current); cudaFree(d_next);
    MPI_Finalize();
    return 0;
}
