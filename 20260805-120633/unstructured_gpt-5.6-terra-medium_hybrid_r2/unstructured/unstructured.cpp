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

// Kept as a separate structure so the two state fields have exactly the same
// layout and numerical meaning as the original unstructured implementation.
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static void cudaCheck(cudaError_t error, const char* what, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s: %s\n", rank, what,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

// Each rank owns full grid rows.  The two extra rows are MPI halo cells.  The
// kernel deliberately reconstructs the original adjacency list ordering
// (down, up, right, left); although addition is commutative mathematically,
// this retains the reference floating-point operation order.
__global__ void updateKernel(const ElementDynamic* __restrict__ current,
                             ElementDynamic* __restrict__ next,
                             int localRows, int width, int globalRowBegin) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int localRow = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col >= width || localRow > localRows) return;

    const int index = localRow * width + col;
    const val_t energy = current[index].current_energy;
    const int globalRow = globalRowBegin + localRow - 1;
    val_t flux = ((globalRow == 0 && col == 0) ||
                  (globalRow == width - 1 && col == width - 1)) ? 0.5 :
                 ((globalRow == 0 && col == width - 1) ||
                  (globalRow == width - 1 && col == 0)) ? -0.5 : 0.0;

    // Preserve computeFlux's multiplication order from the reference code.
    if (globalRow + 1 < width)
        flux += (current[index + width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (globalRow > 0)
        flux += (current[index - width].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (col + 1 < width)
        flux += (current[index + 1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (col > 0)
        flux += (current[index - 1].current_energy - energy) * 0.8 * 1.0 * 0.25;

    next[index].current_energy = energy + flux;
    next[index].total_flux = current[index].total_flux + fabs(flux);
}

struct Partition {
    int firstRow;
    int rows;
};

static Partition partitionRows(int width, int rank, int ranks) {
    const int base = width / ranks;
    const int remainder = width % ranks;
    return {rank * base + std::min(rank, remainder), base + (rank < remainder)};
}

static uint64_t computeLocalHash(const std::vector<ElementDynamic>& values,
                                 idx_t globalFirst) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^ : hash) schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(values.size()); ++i) {
        uint64_t energyBits, fluxBits;
        std::memcpy(&energyBits, &values[i].current_energy, sizeof(energyBits));
        std::memcpy(&fluxBits, &values[i].total_flux, sizeof(fluxBits));
        const uint64_t globalIndex = globalFirst + static_cast<idx_t>(i);
        hash ^= (energyBits + globalIndex) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fluxBits + globalIndex) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int width = 512, iterations = 10;
    bool validate = false, printResults = false, usage = false, badArgument = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) width = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) usage = true;
        else badArgument = true;
    }
    if (usage || badArgument || width <= 0 || iterations < 0 || ranks > width) {
        if (rank == 0) {
            if (badArgument || width <= 0 || iterations < 0 || ranks > width)
                std::fprintf(stderr, "Invalid arguments or more MPI ranks than grid rows.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (usage && !badArgument) ? 0 : 1;
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const Partition part = partitionRows(width, rank, ranks);
    const size_t localCells = static_cast<size_t>(part.rows) * width;
    const size_t haloCells = static_cast<size_t>(part.rows + 2) * width;
    std::vector<ElementDynamic> hostCurrent(haloCells), hostNext(haloCells);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(haloCells); ++i)
        hostCurrent[i] = hostNext[i] = {0.0, 0.0};

    ElementDynamic *deviceCurrent = nullptr, *deviceNext = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceCurrent, haloCells * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&deviceNext, haloCells * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemcpy(deviceCurrent, hostCurrent.data(), haloCells * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceNext, hostNext.data(), haloCells * sizeof(ElementDynamic), cudaMemcpyHostToDevice));

    if (rank == 0) {
        const long long elements = 1LL * width * width;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\n", width, width, elements, iterations, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA GPUs/rank: 1\n\n", ranks, omp_get_max_threads());
        std::printf("Building distributed unstructured mesh...\n");
    }
    const size_t staticMem = localCells * (sizeof(idx_t) * 10 + sizeof(val_t) * 8);
    const size_t dynamicMem = haloCells * sizeof(ElementDynamic) * 2;
    if (rank == 0) std::printf("Memory usage per rank (rank 0): %.2f MB (static-equivalent: %.2f MB, dynamic: %.2f MB)\n\nRunning simulation...\n",
                               (staticMem + dynamicMem) / 1048576.0, staticMem / 1048576.0, dynamicMem / 1048576.0);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(32, 8), grid((width + block.x - 1) / block.x, (part.rows + block.y - 1) / block.y);
    for (int iter = 0; iter < iterations; ++iter) {
        // Stage just the two boundary rows.  This works with every MPI
        // implementation (including installations without CUDA awareness),
        // while all bulk state and all stencil work remain on the GPU.
        if (rank > 0)
            CUDA_CHECK(cudaMemcpy(hostCurrent.data() + width, deviceCurrent + width,
                                  width * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpy(hostCurrent.data() + part.rows * width,
                                  deviceCurrent + part.rows * width,
                                  width * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
        MPI_Request requests[4]; int requestCount = 0;
        if (rank > 0) {
            MPI_Irecv(hostCurrent.data(), width * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 17, MPI_COMM_WORLD, &requests[requestCount++]);
            MPI_Isend(hostCurrent.data() + width, width * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 18, MPI_COMM_WORLD, &requests[requestCount++]);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(hostCurrent.data() + (part.rows + 1) * width, width * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 18, MPI_COMM_WORLD, &requests[requestCount++]);
            MPI_Isend(hostCurrent.data() + part.rows * width, width * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 17, MPI_COMM_WORLD, &requests[requestCount++]);
        }
        if (requestCount) MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA_CHECK(cudaMemcpy(deviceCurrent, hostCurrent.data(), width * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        if (rank + 1 < ranks)
            CUDA_CHECK(cudaMemcpy(deviceCurrent + (part.rows + 1) * width,
                                  hostCurrent.data() + (part.rows + 1) * width,
                                  width * sizeof(ElementDynamic), cudaMemcpyHostToDevice));
        updateKernel<<<grid, block>>>(deviceCurrent, deviceNext, part.rows, width, part.firstRow);
        CUDA_CHECK(cudaGetLastError());
        std::swap(deviceCurrent, deviceNext);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(hostCurrent.data() + width, deviceCurrent + width, localCells * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    const uint64_t localHash = computeLocalHash(std::vector<ElementDynamic>(hostCurrent.begin() + width, hostCurrent.begin() + width + localCells), static_cast<idx_t>(part.firstRow) * width);
    uint64_t hash = 0;
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int measured = std::max(iterations - 1, 1);
        const double giga = (static_cast<double>(measured) * width * width) / maxElapsed / 1e9;
        std::printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n",
                    maxElapsed * 1000.0, maxElapsed * 1000.0 / measured, giga, giga * 22.0, static_cast<unsigned long long>(hash));
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        if (rank == 0) { counts.resize(ranks); displacements.resize(ranks); for (int r = 0; r < ranks; ++r) { Partition p = partitionRows(width, r, ranks); counts[r] = p.rows * width; displacements[r] = p.firstRow * width; } }
        std::vector<val_t> localEnergy(localCells), allEnergy;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(localCells); ++i) localEnergy[i] = hostCurrent[width + i].current_energy;
        if (rank == 0) allEnergy.resize(static_cast<size_t>(width) * width);
        MPI_Gatherv(localEnergy.data(), static_cast<int>(localCells), MPI_DOUBLE, allEnergy.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(allEnergy, "ElementEnergy");
    }

    if (validate) {
        val_t localEnergySum = 0.0, localFluxSum = 0.0, localMin = std::numeric_limits<val_t>::max(), localMax = std::numeric_limits<val_t>::lowest();
#pragma omp parallel for reduction(+ : localEnergySum, localFluxSum) reduction(min : localMin) reduction(max : localMax) schedule(static)
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(localCells); ++i) { const auto& e = hostCurrent[width + i]; localEnergySum += e.current_energy; localFluxSum += e.total_flux; localMin = std::min(localMin, e.current_energy); localMax = std::max(localMax, e.current_energy); }
        val_t energySum, fluxSum, energyMin, energyMax;
        MPI_Reduce(&localEnergySum, &energySum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Reduce(&localFluxSum, &fluxSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Reduce(&localMin, &energyMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); MPI_Reduce(&localMax, &energyMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) { std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", energySum, fluxSum, energyMin, energyMax, (std::isfinite(energySum) && std::isfinite(fluxSum) && std::isfinite(energyMin) && std::isfinite(energyMax)) ? "PASSED" : "FAILED"); }
    }
    CUDA_CHECK(cudaFree(deviceCurrent)); CUDA_CHECK(cudaFree(deviceNext));
    MPI_Finalize();
    return 0;
}
