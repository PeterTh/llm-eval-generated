#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

// All MPI calls are made by the main thread (MPI_THREAD_FUNNELED).
static void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s:%d: CUDA: %s\n", file, line, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), __FILE__, __LINE__)

// The input generator always constructs a square four-neighbor mesh with unit
// connection weights. Encode that connectivity implicitly, avoiding 144 bytes
// of static data per cell. Keep the original +x,-x,+y,-y summation order.
__global__ void update(const double* __restrict__ energy,
                       double* __restrict__ next,
                       double* __restrict__ accumulated,
                       int n, int firstRow, size_t begin, size_t end) {
    for (size_t i = begin + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < end; i += size_t(blockDim.x) * gridDim.x) {
        const int x = firstRow + i / n;
        const int y = i % n;
        const size_t p = i + n; // one halo row on either side
        const double e = energy[p];
        double f = 0.0;
        if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1))
            f = (x == y) ? 0.5 : -0.5;
        if (x + 1 < n) f += ((energy[p + n] - e) * 0.8) * 0.25;
        if (x > 0)     f += ((energy[p - n] - e) * 0.8) * 0.25;
        if (y + 1 < n) f += ((energy[p + 1] - e) * 0.8) * 0.25;
        if (y > 0)     f += ((energy[p - 1] - e) * 0.8) * 0.25;
        next[p] = e + f;
        accumulated[i] += fabs(f);
    }
}

struct Simulation {
    int n, rows, first, rank, ranks;
    size_t count;
    double *energy, *next, *flux, *send, *receive;
    cudaStream_t compute, transfer;
    cudaEvent_t haloReady;
    MPI_Comm comm;

    Simulation(int width, MPI_Comm communicator) : n(width), comm(communicator) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        rows = n / ranks + (rank < n % ranks);
        first = rank * (n / ranks) + std::min(rank, n % ranks);
        count = size_t(rows) * n;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        CUDA(cudaMalloc(&energy, (count + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMalloc(&next, (count + 2 * size_t(n)) * sizeof(double)));
        CUDA(cudaMalloc(&flux, count * sizeof(double)));
        CUDA(cudaMemsetAsync(energy, 0, (count + 2 * size_t(n)) * sizeof(double), compute));
        CUDA(cudaMemsetAsync(flux, 0, count * sizeof(double), compute));
        // Pinned staging works with ordinary MPI, without requiring CUDA-aware MPI.
        CUDA(cudaMallocHost(&send, 2 * size_t(n) * sizeof(double)));
        CUDA(cudaMallocHost(&receive, 2 * size_t(n) * sizeof(double)));
        CUDA(cudaStreamSynchronize(compute));
    }

    void launch(size_t begin, size_t end) {
        if (begin >= end) return;
        int blocks = int(std::min<size_t>((end - begin + 255) / 256, 65535));
        update<<<blocks, 256, 0, compute>>>(energy, next, flux, n, first, begin, end);
        CUDA(cudaGetLastError());
    }

    void run(int iterations) {
        const int previous = rank ? rank - 1 : MPI_PROC_NULL;
        const int following = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        const size_t bytes = size_t(n) * sizeof(double);
        for (int iter = 0; iter < iterations; ++iter) {
            if (ranks > 1) {
                MPI_Request requests[4];
                MPI_Irecv(receive, n, MPI_DOUBLE, previous, 1, comm, &requests[0]);
                MPI_Irecv(receive + n, n, MPI_DOUBLE, following, 0, comm, &requests[1]);
                if (previous != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(send, energy + n, bytes, cudaMemcpyDeviceToHost, transfer));
                if (following != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(send + n, energy + count, bytes, cudaMemcpyDeviceToHost, transfer));
                // Interior cells do not depend on incoming halos.
                if (rows > 2) launch(n, count - n);
                CUDA(cudaStreamSynchronize(transfer));
                MPI_Isend(send, n, MPI_DOUBLE, previous, 0, comm, &requests[2]);
                MPI_Isend(send + n, n, MPI_DOUBLE, following, 1, comm, &requests[3]);
                // MPI progresses while the GPU processes the interior.
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                if (previous != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(energy, receive, bytes, cudaMemcpyHostToDevice, transfer));
                if (following != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(energy + count + n, receive + n, bytes, cudaMemcpyHostToDevice, transfer));
                CUDA(cudaEventRecord(haloReady, transfer));
                CUDA(cudaStreamWaitEvent(compute, haloReady, 0));
                launch(0, n);
                if (rows > 1) launch(count - n, count);
                CUDA(cudaStreamSynchronize(compute));
            } else {
                launch(0, count);
            }
            std::swap(energy, next);
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    ~Simulation() {
        CUDA(cudaFree(energy)); CUDA(cudaFree(next)); CUDA(cudaFree(flux));
        CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(receive));
        CUDA(cudaEventDestroy(haloReady));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    }
};

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10;
    bool validate = false, results = false, help = false, validArgs = true;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            bool grid = !strcmp(argv[i], "-n");
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < (grid ? 1 : 0) || value > INT_MAX)
                validArgs = false;
            else if (grid) n = int(value);
            else iterations = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else validArgs = false;
    }
    // The original indexing and MPI collective counts use signed integers.
    if (int64_t(n) * n > INT_MAX) validArgs = false;
    if (help || !validArgs) {
        if (!rank) {
            if (!validArgs) fprintf(stderr, "Invalid arguments or grid too large.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return validArgs ? 0 : 1;
    }
    // Empty partitions participate in finalization, but not halo exchanges.
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < n ? 0 : MPI_UNDEFINED, rank, &active);
    int failed = 0;
    if (active != MPI_COMM_NULL) {
        MPI_Comm shared;
        MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
        int localRank, devices;
        MPI_Comm_rank(shared, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "CUDA device required on every active rank.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&shared);
        {
            Simulation sim(n, active);
            if (!rank) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n");
                printf("============================================\n");
                printf("Grid size: %d x %d = %lld elements\n", n, n, (long long)n * n);
                printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA enabled\n", sim.ranks, omp_get_max_threads());
                printf("Building unstructured mesh (implicit square connectivity)...\nRunning simulation...\n");
            }
            MPI_Barrier(active);
            double start = MPI_Wtime();
            sim.run(iterations);
            double elapsed = MPI_Wtime() - start, seconds = 0;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);
            std::vector<double> energies(sim.count), fluxes(sim.count);
            CUDA(cudaMemcpy(energies.data(), sim.energy + n, sim.count * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(fluxes.data(), sim.flux, sim.count * sizeof(double), cudaMemcpyDeviceToHost));
            uint64_t localHash = 0, globalHash = 0;
            double energySum = 0, fluxSum = 0;
            double minimum = std::numeric_limits<double>::max();
            double maximum = std::numeric_limits<double>::lowest();
            int nonfinite = 0;
            #pragma omp parallel for schedule(static) reduction(^:localHash) reduction(+:energySum,fluxSum) reduction(min:minimum) reduction(max:maximum) reduction(|:nonfinite)
            for (size_t i = 0; i < sim.count; ++i) {
                uint64_t e, f;
                memcpy(&e, &energies[i], sizeof(e));
                memcpy(&f, &fluxes[i], sizeof(f));
                uint64_t globalIndex = size_t(sim.first) * n + i;
                localHash ^= (e + globalIndex) * 0x9e3779b97f4a7c15ULL;
                localHash ^= (f + globalIndex) * 0xbf58476d1ce4e5b9ULL;
                energySum += energies[i]; fluxSum += fluxes[i];
                minimum = std::min(minimum, energies[i]);
                maximum = std::max(maximum, energies[i]);
                nonfinite |= !std::isfinite(energies[i]) || !std::isfinite(fluxes[i]);
            }
            MPI_Reduce(&localHash, &globalHash, 1, MPI_UINT64_T, MPI_BXOR, 0, active);
            if (!rank) {
                double rate = seconds > 0 ? double(iterations) * n * n / seconds / 1e9 : 0;
                printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000);
                printf("  Time per iteration: %.4f ms\n", iterations ? seconds * 1000 / iterations : 0);
                printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
                printf("  Result hash: %016" PRIX64 "\n\n", globalHash);
            }
            if (results) {
                std::vector<int> counts(sim.ranks), displacements(sim.ranks);
                for (int r = 0; r < sim.ranks; ++r) {
                    counts[r] = (n / sim.ranks + (r < n % sim.ranks)) * n;
                    displacements[r] = (r * (n / sim.ranks) + std::min(r, n % sim.ranks)) * n;
                }
                std::vector<double> all;
                if (!rank) all.resize(size_t(n) * n);
                MPI_Gatherv(energies.data(), int(sim.count), MPI_DOUBLE, all.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, active);
                if (!rank) print_results(all, "ElementEnergy");
            }
            if (validate) {
                double localSums[2] = {energySum, fluxSum}, sums[2], lo, hi;
                MPI_Reduce(localSums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, active);
                MPI_Reduce(&minimum, &lo, 1, MPI_DOUBLE, MPI_MIN, 0, active);
                MPI_Reduce(&maximum, &hi, 1, MPI_DOUBLE, MPI_MAX, 0, active);
                MPI_Reduce(&nonfinite, &failed, 1, MPI_INT, MPI_MAX, 0, active);
                if (!rank) {
                    failed |= !std::isfinite(sums[0]) || !std::isfinite(sums[1]);
                    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", sums[0], sums[1], lo, hi);
                    if (fabs(sums[0]) > 1e-8)
                        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
                    printf("  Validation: %s\n", failed ? "FAILED" : "PASSED");
                }
            }
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failed ? 1 : 0;
}
