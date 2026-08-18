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

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy, total_flux; };
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
};

constexpr idx_t DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

static void mpiCheck(int code, const char* operation) {
    if (code != MPI_SUCCESS) {
        char text[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(code, text, &length);
        std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, text);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}

static void cudaCheck(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(code));
    }
}

void buildSquare2D(World& world, int n) {
    const int count = n * n;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(count);
    world.elements_dynamic.resize(count);
    world.elements_dynamic_swap.resize(count);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i] = {0.0, 0.0};
        world.elements_dynamic_swap[i] = {0.0, 0.0};
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < n; ++x) for (int y = 0; y < n; ++y) {
        const int index = x * n + y;
        auto& element = world.elements_static[index];
        constexpr int dx[4] = {1, -1, 0, 0};
        constexpr int dy[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            const int nx = x + dx[k], ny = y + dy[k];
            if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                const int slot = static_cast<int>(element.num_connections++);
                element.connected_idx[slot] = nx * n + ny;
                element.connected_flux[slot] = 1.0;
            }
        }
    }
    const int last = n - 1;
    world.elements_static[0].material_idx = INFLOW_MAT_ID;
    world.elements_static[last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n + last].material_idx = INFLOW_MAT_ID;
}

__device__ __forceinline__ val_t deviceFlux(const Material& material,
                                             const ElementDynamic& self,
                                             val_t coefficient,
                                             const ElementDynamic& other) {
    return (other.current_energy - self.current_energy) *
           material.transfer_coeff * coefficient * 0.25;
}

__global__ void updateKernel(const ElementStatic* statics, const Material* materials,
                             const ElementDynamic* current, ElementDynamic* next,
                             int begin, int end) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    const ElementStatic& element = statics[i];
    const ElementDynamic self = current[i];
    const Material material = materials[element.material_idx];
    val_t total_flux = material.external_flow;
    // Keep this loop and its order identical to the reference implementation.
    for (idx_t j = 0; j < element.num_connections; ++j)
        total_flux += deviceFlux(material, self, element.connected_flux[j],
                                 current[element.connected_idx[j]]);
    next[i].current_energy = self.current_energy + total_flux;
    next[i].total_flux = self.total_flux + fabs(total_flux);
}

void runSimulation(World& world, int iterations, int rank, int ranks) {
    const int count = static_cast<int>(world.elements_static.size());
    const int begin = (count * rank) / ranks;
    const int end = (count * (rank + 1)) / ranks;
    const int local_count = end - begin;
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA accelerator is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(rank % device_count), "cudaSetDevice");

    ElementStatic* d_static = nullptr; Material* d_materials = nullptr;
    ElementDynamic *d_current = nullptr, *d_next = nullptr;
    cudaCheck(cudaMalloc(&d_static, world.elements_static.size() * sizeof(ElementStatic)), "cudaMalloc static");
    cudaCheck(cudaMalloc(&d_materials, world.materials.size() * sizeof(Material)), "cudaMalloc materials");
    cudaCheck(cudaMalloc(&d_current, count * sizeof(ElementDynamic)), "cudaMalloc current");
    cudaCheck(cudaMalloc(&d_next, count * sizeof(ElementDynamic)), "cudaMalloc next");
    cudaCheck(cudaMemcpy(d_static, world.elements_static.data(), count * sizeof(ElementStatic), cudaMemcpyHostToDevice), "copy static");
    cudaCheck(cudaMemcpy(d_materials, world.materials.data(), world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice), "copy materials");

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int b = (count * r) / ranks, e = (count * (r + 1)) / ranks;
        counts[r] = 2 * (e - b); displacements[r] = 2 * b;
    }
    cudaCheck(cudaMemcpy(d_current, world.elements_dynamic.data(), count * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "copy initial state");
    constexpr int threads = 256;
    const int blocks = (local_count + threads - 1) / threads;
    for (int iter = 0; iter < iterations; ++iter) {
        if (local_count > 0) {
            updateKernel<<<blocks, threads>>>(d_static, d_materials, d_current, d_next, begin, end);
            cudaCheck(cudaGetLastError(), "updateKernel launch");
            cudaCheck(cudaMemcpy(world.elements_dynamic_swap.data() + begin, d_next + begin,
                                 local_count * sizeof(ElementDynamic), cudaMemcpyDeviceToHost), "copy local state");
        }
        mpiCheck(MPI_Allgatherv(world.elements_dynamic_swap.data() + begin, 2 * local_count, MPI_DOUBLE,
                                world.elements_dynamic_swap.data(), counts.data(), displacements.data(),
                                MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Allgatherv");
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        cudaCheck(cudaMemcpy(d_current, world.elements_dynamic.data(), count * sizeof(ElementDynamic), cudaMemcpyHostToDevice), "copy exchanged state");
    }
    cudaCheck(cudaFree(d_static), "cudaFree static");
    cudaCheck(cudaFree(d_materials), "cudaFree materials");
    cudaCheck(cudaFree(d_current), "cudaFree current");
    cudaCheck(cudaFree(d_next), "cudaFree next");
}

bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min) schedule(static)
    for (std::size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const auto& e = world.elements_dynamic[i];
        energy_sum += e.current_energy; flux_sum += e.total_flux;
        energy_max = std::max(energy_max, e.current_energy);
        energy_min = std::min(energy_min, e.current_energy);
    }
    std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum) || !std::isfinite(flux_sum) || !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: non-finite result\n"); return false;
    }
    if (std::abs(energy_sum) > 1e-8) std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: PASSED\n"); return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t result = 0;
    #pragma omp parallel for reduction(^:result) schedule(static)
    for (std::size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        result ^= (*e + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (*f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num> Grid size (default: 512)\n  -i <num> Iterations (default: 10)\n  -v Validation\n  -r Print results\n  -h Help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0; mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    int rank = 0, ranks = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iterations < 0) { if (rank == 0) std::fprintf(stderr, "Invalid dimensions or iteration count\n"); MPI_Finalize(); return 1; }
    World world; buildSquare2D(world, n);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nMPI ranks: %d\nOpenMP threads/rank: %d\nCUDA devices: one per rank (round-robin)\n\n", n, n, n*n, iterations, ranks, omp_get_max_threads());
    }
    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, iterations, rank, ranks);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long elapsed_ms = 0; mpiCheck(MPI_Reduce(&local_ms, &elapsed_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce timing");
    if (rank == 0) {
        const int measured = std::max(iterations - 1, 1);
        const double per_iter = static_cast<double>(elapsed_ms) / measured;
        const double geps = measured * static_cast<double>(n*n) / (std::max(1LL, elapsed_ms) / 1000.0) / 1e9;
        std::printf("Computation time: %lld ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", elapsed_ms, per_iter, geps, geps * 22.0, computeHash(world.elements_dynamic));
        if (printResults) { std::vector<double> energy(world.elements_dynamic.size());
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < energy.size(); ++i) energy[i] = world.elements_dynamic[i].current_energy;
            print_results(energy, "ElementEnergy"); }
        if (validate && !validateResults(world)) { MPI_Finalize(); return 1; }
    }
    MPI_Finalize(); return 0;
}
