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

using val_t = double;

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// One rank owns a contiguous band of rows.  The two extra rows are MPI halos.
// Keeping the dynamic fields in structure-of-arrays form is substantially more
// bandwidth efficient than the original host-side array-of-structures layout.
__global__ void updateRows(const val_t* __restrict__ energy,
                           const val_t* __restrict__ flux,
                           val_t* __restrict__ nextEnergy,
                           val_t* __restrict__ nextFlux,
                           int firstRow, int localRows, int n, int begin, int end) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int localRow = begin + blockIdx.y;
    if (col >= n || localRow >= end || localRow >= localRows) return;

    const int p = (localRow + 1) * n + col;
    const int globalRow = firstRow + localRow;
    const val_t e = energy[p];
    val_t total = ((globalRow == 0 && col == 0) || (globalRow == n - 1 && col == n - 1)) ? 0.5 :
                  (((globalRow == 0 && col == n - 1) || (globalRow == n - 1 && col == 0)) ? -0.5 : 0.0);
    // Match the original connectivity order: down, up, right, left.
    // Keep the original operation sequence (including its rounding behavior).
    if (globalRow + 1 < n) total += ((energy[p + n] - e) * 0.8) * 1.0 * 0.25;
    if (globalRow > 0)     total += ((energy[p - n] - e) * 0.8) * 1.0 * 0.25;
    if (col + 1 < n)       total += ((energy[p + 1] - e) * 0.8) * 1.0 * 0.25;
    if (col > 0)           total += ((energy[p - 1] - e) * 0.8) * 1.0 * 0.25;
    nextEnergy[p] = e + total;
    nextFlux[p] = flux[p] + fabs(total);
}

static uint64_t computeHash(const std::vector<val_t>& energy,
                            const std::vector<val_t>& flux, uint64_t globalOffset) {
    uint64_t hash = 0;
    for (size_t i = 0; i < energy.size(); ++i) {
        uint64_t e, f;
        memcpy(&e, &energy[i], sizeof(e));
        memcpy(&f, &flux[i], sizeof(f));
        const uint64_t g = globalOffset + i;
        hash ^= (e + g) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + g) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n"
           "  -i <num>     Number of simulation iterations (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int n = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iterations < 0 || ranks > n) {
        if (!rank) fprintf(stderr, "Grid size must be positive, iterations non-negative, and MPI ranks no greater than grid rows.\n");
        MPI_Finalize(); return 1;
    }

    const int base = n / ranks, remainder = n % ranks;
    const int rows = base + (rank < remainder);
    const int firstRow = rank * base + std::min(rank, remainder);
    const int previous = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    const size_t localCount = static_cast<size_t>(rows) * n;
    const size_t paddedCount = static_cast<size_t>(rows + 2) * n;

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    val_t *dEnergy, *dFlux, *dNextEnergy, *dNextFlux;
    CUDA_CHECK(cudaMalloc(&dEnergy, paddedCount * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dFlux, paddedCount * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dNextEnergy, paddedCount * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dNextFlux, paddedCount * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(dEnergy, 0, paddedCount * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(dFlux, 0, paddedCount * sizeof(val_t)));

    std::vector<val_t> sendTop(n), sendBottom(n), recvTop(n), recvBottom(n);
    if (!rank) {
        const long long elems = static_cast<long long>(n) * n;
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n\n",
               n, n, elems, iterations, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        const double bytes = (sizeof(val_t) * 4.0 * (n + 2) * n * ranks) / (1024.0 * 1024.0);
        printf("Distributed device memory usage: %.2f MB\n\nRunning simulation...\n", bytes);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto started = std::chrono::steady_clock::now();
    const dim3 block(256), grid((n + block.x - 1) / block.x, rows);
    for (int it = 0; it < iterations; ++it) {
        MPI_Request requests[4]; int requestCount = 0;
        CUDA_CHECK(cudaMemcpy(sendTop.data(), dEnergy + n, n * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sendBottom.data(), dEnergy + static_cast<size_t>(rows) * n, n * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Irecv(recvTop.data(), n, MPI_DOUBLE, previous, 41, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Irecv(recvBottom.data(), n, MPI_DOUBLE, next, 40, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendTop.data(), n, MPI_DOUBLE, previous, 40, MPI_COMM_WORLD, &requests[requestCount++]);
        MPI_Isend(sendBottom.data(), n, MPI_DOUBLE, next, 41, MPI_COMM_WORLD, &requests[requestCount++]);
        if (rows > 2) updateRows<<<dim3(grid.x, rows - 2), block>>>(dEnergy, dFlux, dNextEnergy, dNextFlux, firstRow, rows, n, 1, rows - 1);
        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        if (previous != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(dEnergy, recvTop.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        if (next != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(dEnergy + static_cast<size_t>(rows + 1) * n, recvBottom.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        updateRows<<<dim3(grid.x, 1), block>>>(dEnergy, dFlux, dNextEnergy, dNextFlux, firstRow, rows, n, 0, 1);
        if (rows > 1) updateRows<<<dim3(grid.x, 1), block>>>(dEnergy, dFlux, dNextEnergy, dNextFlux, firstRow, rows, n, rows - 1, rows);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(dEnergy, dNextEnergy); std::swap(dFlux, dNextFlux);
    }
    const double localMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    double durationMs; MPI_Reduce(&localMs, &durationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<val_t> energy(localCount), flux(localCount);
    CUDA_CHECK(cudaMemcpy(energy.data(), dEnergy + n, localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flux.data(), dFlux + n, localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    // OpenMP accelerates independent host reductions / hash preparation on each rank.
    val_t localEnergy = 0, localFlux = 0, localMin = std::numeric_limits<val_t>::max(), localMax = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:localEnergy,localFlux) reduction(min:localMin) reduction(max:localMax)
    for (long long i = 0; i < static_cast<long long>(localCount); ++i) { localEnergy += energy[i]; localFlux += flux[i]; localMin = std::min(localMin, energy[i]); localMax = std::max(localMax, energy[i]); }
    const uint64_t localHash = computeHash(energy, flux, static_cast<uint64_t>(firstRow) * n);
    uint64_t globalHash; MPI_Reduce(&localHash, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    val_t sumEnergy, sumFlux, minEnergy, maxEnergy;
    MPI_Reduce(&localEnergy, &sumEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localFlux, &sumFlux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMin, &minEnergy, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &maxEnergy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        const double measured = std::max(1, iterations - 1); const double safeMs = std::max(durationMs, 1e-9);
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", durationMs, durationMs / measured, (measured * n * static_cast<double>(n)) / (safeMs * 1e6), (measured * n * static_cast<double>(n)) / (safeMs * 1e6) * 22.0, static_cast<unsigned long long>(globalHash));
        if (validate) { printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", sumEnergy, sumFlux, minEnergy, maxEnergy, (std::isfinite(sumEnergy) && std::isfinite(sumFlux) && std::isfinite(minEnergy) && std::isfinite(maxEnergy)) ? "PASSED" : "FAILED"); }
    }
    if (printResults) {
        std::vector<val_t> allEnergy; if (!rank) allEnergy.resize(static_cast<size_t>(n) * n);
        std::vector<int> counts(ranks), offsets(ranks); for (int r = 0; r < ranks; ++r) { counts[r] = (base + (r < remainder)) * n; offsets[r] = (r * base + std::min(r, remainder)) * n; }
        MPI_Gatherv(energy.data(), static_cast<int>(localCount), MPI_DOUBLE, rank ? nullptr : allEnergy.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(allEnergy, "ElementEnergy");
    }
    CUDA_CHECK(cudaFree(dEnergy)); CUDA_CHECK(cudaFree(dFlux)); CUDA_CHECK(cudaFree(dNextEnergy)); CUDA_CHECK(cudaFree(dNextFlux));
    MPI_Finalize(); return 0;
}
