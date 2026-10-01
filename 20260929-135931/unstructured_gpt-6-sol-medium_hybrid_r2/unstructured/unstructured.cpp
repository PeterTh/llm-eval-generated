#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

static void check(cudaError_t code, const char* where) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// The benchmark builds a square grid. Preserve its neighbor order (down, up,
// right, left) and its arithmetic for each element.
__global__ void step(const double* energy, const double* flux,
                     double* nextEnergy, double* nextFlux,
                     int n, int firstRow, int rows, int boundary) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int count = (boundary ? min(rows, 2) : max(rows - 2, 0)) * n;
    if (k >= count) return;
    int row = boundary ? (k / n == 0 ? 0 : rows - 1) : 1 + k / n;
    int col = k % n, x = firstRow + row;
    int at = (row + 1) * n + col;
    double old = energy[at];
    double total = ((x == 0 && col == 0) || (x == n - 1 && col == n - 1)) ? 0.5 :
                   ((x == 0 && col == n - 1) || (x == n - 1 && col == 0)) ? -0.5 : 0.0;
    if (x + 1 < n) total += (energy[at + n] - old) * 0.8 * 1.0 * 0.25;
    if (x > 0)     total += (energy[at - n] - old) * 0.8 * 1.0 * 0.25;
    if (col + 1 < n) total += (energy[at + 1] - old) * 0.8 * 1.0 * 0.25;
    if (col > 0)   total += (energy[at - 1] - old) * 0.8 * 1.0 * 0.25;
    nextEnergy[at] = old + total;
    nextFlux[row * n + col] = flux[row * n + col] + fabs(total);
}

static void usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, iters = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { bad = true; if (!rank) printf("Unknown option: %s\n", argv[i]); break; }
    }
    if (n <= 0 || int64_t(n) * n > INT32_MAX || iters < 0) bad = true;
    if (help || bad) {
        if (!rank) usage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    int active = std::min(n, size);
    MPI_Comm work;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
    if (rank >= active) { MPI_Finalize(); return 0; }
    int rows = n / active + (rank < n % active);
    int first = rank * (n / active) + std::min(rank, n % active);
    int count = rows * n;
    MPI_Comm shared;
    MPI_Comm_split_type(work, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, devices = 0;
    MPI_Comm_rank(shared, &localRank);
    check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    check(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&shared);

    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n",
               n, n, n*n, iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        double staticMem = 0.0; // Grid connectivity is computed from coordinates.
        double dynamicMem = (double(n) * n * 4 + double(active) * 4 * n) * sizeof(double);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (staticMem + dynamicMem) / 1048576.0, staticMem / 1048576.0, dynamicMem / 1048576.0);
        printf("Running simulation...\n");
    }
    double *energy[2], *flux[2];
    size_t energyBytes = size_t(rows + 2) * n * sizeof(double);
    size_t dataBytes = size_t(count) * sizeof(double);
    for (int b = 0; b < 2; ++b) {
        check(cudaMalloc(&energy[b], energyBytes), "cudaMalloc energy");
        check(cudaMalloc(&flux[b], dataBytes), "cudaMalloc flux");
        check(cudaMemset(energy[b], 0, energyBytes), "cudaMemset energy");
        check(cudaMemset(flux[b], 0, dataBytes), "cudaMemset flux");
    }
    double *sendTop, *sendBottom, *recvTop, *recvBottom;
    check(cudaMallocHost(&sendTop, n * sizeof(double)), "cudaMallocHost");
    check(cudaMallocHost(&sendBottom, n * sizeof(double)), "cudaMallocHost");
    check(cudaMallocHost(&recvTop, n * sizeof(double)), "cudaMallocHost");
    check(cudaMallocHost(&recvBottom, n * sizeof(double)), "cudaMallocHost");
    cudaStream_t compute, transfer;
    check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "cudaStreamCreate");
    check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "cudaStreamCreate");

    MPI_Barrier(work);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iters; ++iter) {
        int old = iter & 1, next = old ^ 1;
        int inner = std::max(rows - 2, 0) * n;
        if (inner) step<<<(inner + 255) / 256, 256, 0, compute>>>(energy[old], flux[old],
                          energy[next], flux[next], n, first, rows, 0);
        check(cudaGetLastError(), "interior kernel");
        int prev = rank ? rank - 1 : MPI_PROC_NULL;
        int following = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
        if (prev != MPI_PROC_NULL)
            check(cudaMemcpyAsync(sendTop, energy[old] + n, n * sizeof(double),
                                  cudaMemcpyDeviceToHost, transfer), "download top");
        if (following != MPI_PROC_NULL)
            check(cudaMemcpyAsync(sendBottom, energy[old] + rows * n, n * sizeof(double),
                                  cudaMemcpyDeviceToHost, transfer), "download bottom");
        check(cudaStreamSynchronize(transfer), "boundary download");
        MPI_Request requests[4];
        MPI_Irecv(recvTop, n, MPI_DOUBLE, prev, 2, work, &requests[0]);
        MPI_Irecv(recvBottom, n, MPI_DOUBLE, following, 1, work, &requests[1]);
        MPI_Isend(sendTop, n, MPI_DOUBLE, prev, 1, work, &requests[2]);
        MPI_Isend(sendBottom, n, MPI_DOUBLE, following, 2, work, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (prev != MPI_PROC_NULL)
            check(cudaMemcpyAsync(energy[old], recvTop, n * sizeof(double),
                                  cudaMemcpyHostToDevice, transfer), "upload top");
        if (following != MPI_PROC_NULL)
            check(cudaMemcpyAsync(energy[old] + (rows + 1) * n, recvBottom, n * sizeof(double),
                                  cudaMemcpyHostToDevice, transfer), "upload bottom");
        check(cudaStreamSynchronize(transfer), "boundary upload");
        int edge = std::min(rows, 2) * n;
        step<<<(edge + 255) / 256, 256, 0, compute>>>(energy[old], flux[old],
                      energy[next], flux[next], n, first, rows, 1);
        check(cudaGetLastError(), "boundary kernel");
        check(cudaStreamSynchronize(compute), "iteration kernel");
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, work);
    std::vector<double> localEnergy(count), localFlux(count);
    int final = iters & 1;
    check(cudaMemcpy(localEnergy.data(), energy[final] + n, dataBytes,
                     cudaMemcpyDeviceToHost), "download energy");
    check(cudaMemcpy(localFlux.data(), flux[final], dataBytes,
                     cudaMemcpyDeviceToHost), "download flux");
    std::vector<int> counts, offsets;
    std::vector<double> allEnergy, allFlux;
    if (!rank) {
        counts.resize(active); offsets.resize(active);
        allEnergy.resize(size_t(n) * n); allFlux.resize(size_t(n) * n);
        for (int r = 0; r < active; ++r) {
            counts[r] = (n / active + (r < n % active)) * n;
            offsets[r] = (r * (n / active) + std::min(r, n % active)) * n;
        }
    }
    MPI_Gatherv(localEnergy.data(), count, MPI_DOUBLE, allEnergy.data(), counts.data(),
                offsets.data(), MPI_DOUBLE, 0, work);
    MPI_Gatherv(localFlux.data(), count, MPI_DOUBLE, allFlux.data(), counts.data(),
                offsets.data(), MPI_DOUBLE, 0, work);
    int result = 0;
    if (!rank) {
        double ms = duration * 1000.0;
        int measured = std::max(iters - 1, 1);
        double geps = (double(measured) * n * n) / duration / 1e9;
        printf("Computation time: %ld ms\n", long(ms));
        printf("Performance:\n  Time per iteration: %.4f ms\n", ms / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
               geps, geps * 22.0);
        uint64_t hash = 0;
#pragma omp parallel for reduction(^:hash) schedule(static) if(allEnergy.size() >= 16384) num_threads(std::min(8, omp_get_max_threads()))
        for (int64_t i = 0; i < int64_t(allEnergy.size()); ++i) {
            uint64_t e, f;
            memcpy(&e, &allEnergy[i], sizeof(e));
            memcpy(&f, &allFlux[i], sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
        if (printResults) print_results(allEnergy, "ElementEnergy");
        if (validate) {
            double energySum = 0, fluxSum = 0;
            double energyMin = std::numeric_limits<double>::max();
            double energyMax = std::numeric_limits<double>::lowest();
            for (size_t i = 0; i < allEnergy.size(); ++i) {
                energySum += allEnergy[i]; fluxSum += allFlux[i];
                energyMin = std::min(energyMin, allEnergy[i]);
                energyMax = std::max(energyMax, allEnergy[i]);
            }
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", energySum, fluxSum);
            printf("  Energy range: [%.6f, %.6f]\n", energyMin, energyMax);
            if (!std::isfinite(energySum) || !std::isfinite(fluxSum) ||
                !std::isfinite(energyMin) || !std::isfinite(energyMax)) {
                printf("  ERROR: nonfinite result\n"); result = 1;
            } else {
                if (std::abs(energySum) > 1e-8)
                    printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
                printf("  Validation: PASSED\n");
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, work);
    cudaStreamDestroy(compute); cudaStreamDestroy(transfer);
    for (int b = 0; b < 2; ++b) { cudaFree(energy[b]); cudaFree(flux[b]); }
    cudaFreeHost(sendTop); cudaFreeHost(sendBottom);
    cudaFreeHost(recvTop); cudaFreeHost(recvBottom);
    MPI_Comm_free(&work);
    MPI_Finalize();
    return result;
}
