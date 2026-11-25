#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) checkCuda((call), #call)

// Energy arrays include a ghost row at either end. Accumulated flux is
// private to each element, so it can be updated in place.
__global__ void update(const double* __restrict__ energy, double* __restrict__ next,
                       double* __restrict__ flux, int n, int globalStart,
                       int firstRow, int rowStride, int count) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || j >= count) return;
    const int row = firstRow + j * rowStride;
    const int globalRow = globalStart + row;
    const size_t i = size_t(row + 1) * n + col;
    const double e = energy[i];
    double change = 0.0;
    if ((globalRow == 0 || globalRow == n - 1) && (col == 0 || col == n - 1))
        change = (globalRow == col) ? 0.5 : -0.5;

    // Preserve the connection traversal and rounding order of the mesh builder.
    if (globalRow + 1 < n) change += ((energy[i + n] - e) * 0.8) * 0.25;
    if (globalRow > 0)     change += ((energy[i - n] - e) * 0.8) * 0.25;
    if (col + 1 < n)       change += ((energy[i + 1] - e) * 0.8) * 0.25;
    if (col > 0)           change += ((energy[i - 1] - e) * 0.8) * 0.25;
    next[i] = e + change;
    flux[i - n] += fabs(change);
}

struct Slab {
    int n, start, rows, above, below;
    double *energy = nullptr, *next = nullptr, *flux = nullptr;
    double* staging = nullptr;
    cudaStream_t compute{}, transfer{};
    cudaEvent_t ready{};

    size_t cells() const { return size_t(rows) * n; }
    size_t energyBytes() const { return (cells() + 2 * n) * sizeof(double); }

    void allocate() {
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaMalloc(&energy, energyBytes()));
        CUDA(cudaMalloc(&next, energyBytes()));
        CUDA(cudaMalloc(&flux, cells() * sizeof(double)));
        CUDA(cudaMemsetAsync(energy, 0, energyBytes(), compute));
        CUDA(cudaMemsetAsync(flux, 0, cells() * sizeof(double), compute));
        if (above != MPI_PROC_NULL || below != MPI_PROC_NULL) {
            bool direct = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
            direct = MPIX_Query_cuda_support() != 0;
#endif
            if (!direct) {
                CUDA(cudaMallocHost(&staging, 4 * n * sizeof(double)));
                CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
            }
            CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    void launch(int first, int stride, int count) {
        if (count == 0) return;
        const dim3 block(32, 4);
        const dim3 grid((n + 31) / 32, (count + 3) / 4);
        update<<<grid, block, 0, compute>>>(energy, next, flux, n, start, first, stride, count);
        CUDA(cudaGetLastError());
    }

    void simulate(int iterations) {
        const size_t bytes = n * sizeof(double);
        const int top = above != MPI_PROC_NULL;
        const int bottom = below != MPI_PROC_NULL;
        for (int iter = 0; iter < iterations; ++iter) {
            if ((!top && !bottom) || iter == 0) {
                // The initial remote energies are all zero.
                launch(0, 1, rows);
            } else {
                double* sendTop = energy + n;
                double* sendBottom = energy + cells();
                double* recvTop = energy;
                double* recvBottom = energy + cells() + n;
                CUDA(cudaEventRecord(ready, compute));
                if (staging) {
                    sendTop = staging;
                    sendBottom = staging + n;
                    recvTop = staging + 2*n;
                    recvBottom = staging + 3*n;
                    CUDA(cudaStreamWaitEvent(transfer, ready, 0));
                    if (top) CUDA(cudaMemcpyAsync(sendTop, energy + n, bytes,
                                                  cudaMemcpyDeviceToHost, transfer));
                    if (bottom) CUDA(cudaMemcpyAsync(sendBottom, energy + cells(), bytes,
                                                     cudaMemcpyDeviceToHost, transfer));
                }

                const int edges = std::min(rows, top + bottom);
                launch(top, 1, rows - edges);
                if (staging) CUDA(cudaStreamSynchronize(transfer));
                else CUDA(cudaEventSynchronize(ready));

                MPI_Request req[4];
                MPI_Irecv(recvTop, n, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(recvBottom, n, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(sendTop, n, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(sendBottom, n, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &req[3]);
                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
                if (staging) {
                    if (top) CUDA(cudaMemcpyAsync(energy, recvTop, bytes,
                                                  cudaMemcpyHostToDevice, transfer));
                    if (bottom) CUDA(cudaMemcpyAsync(energy + cells() + n, recvBottom, bytes,
                                                     cudaMemcpyHostToDevice, transfer));
                    CUDA(cudaEventRecord(ready, transfer));
                    CUDA(cudaStreamWaitEvent(compute, ready, 0));
                }
                launch(top ? 0 : rows - 1, rows - 1, edges);
            }
            std::swap(energy, next);
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    void release() {
        if (ready) CUDA(cudaEventDestroy(ready));
        if (staging) {
            CUDA(cudaStreamDestroy(transfer));
            CUDA(cudaFreeHost(staging));
        }
        CUDA(cudaFree(flux));
        CUDA(cudaFree(next));
        CUDA(cudaFree(energy));
        CUDA(cudaStreamDestroy(compute));
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
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n = n_elems_root;
    if (n < 1 || int64_t(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and fit in an MPI element count.\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    const int active = std::min(n, ranks);
    const auto rowStart = [=](int r) { return r * (n / ranks) + std::min(r, n % ranks); };
    Slab slab{n, rowStart(rank), rowStart(rank + 1) - rowStart(rank),
              rank > 0 ? rank - 1 : MPI_PROC_NULL, rank + 1 < active ? rank + 1 : MPI_PROC_NULL};
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
        printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        double mib = (3.0 * n * n + 4.0 * n * active) * sizeof(double) / (1024 * 1024);
        printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n", mib, mib);
        printf("Running simulation...\n");
    }
    if (slab.rows) {
        int devices;
        CUDA(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        slab.allocate();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (slab.rows) slab.simulate(n_iters);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double steps = std::max(0, n_iters);
        const double rate = steps * n * n / seconds / 1e9;
        printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000);
        printf("  Time per iteration: %.4f ms\n", steps ? seconds * 1000 / steps : 0.);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
    }

    std::vector<double> energy(slab.cells()), flux(slab.cells());
    if (slab.rows) {
        CUDA(cudaMemcpy(energy.data(), slab.energy + n, slab.cells() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(flux.data(), slab.flux, slab.cells() * sizeof(double), cudaMemcpyDeviceToHost));
        slab.release();
    }
    uint64_t localHash = 0, resultHash;
    double energySum = 0, fluxSum = 0;
    double low = std::numeric_limits<double>::max(), high = -low;
    #pragma omp parallel for schedule(static) reduction(^:localHash) reduction(+:energySum,fluxSum) reduction(min:low) reduction(max:high)
    for (size_t i = 0; i < energy.size(); ++i) {
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        const uint64_t index = size_t(slab.start) * n + i;
        localHash ^= (e + index) * 0x9e3779b97f4a7c15ULL;
        localHash ^= (f + index) * 0xbf58476d1ce4e5b9ULL;
        if (validate) {
            energySum += energy[i];
            fluxSum += flux[i];
            low = std::min(low, energy[i]);
            high = std::max(high, energy[i]);
        }
    }
    MPI_Reduce(&localHash, &resultHash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("  Result hash: %016" PRIX64 "\n\n", resultHash);

    if (printResults) {
        std::vector<int> counts, offsets;
        std::vector<double> global;
        if (rank == 0) {
            global.resize(size_t(n) * n);
            counts.resize(ranks);
            offsets.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                offsets[r] = rowStart(r) * n;
                counts[r] = rowStart(r + 1) * n - offsets[r];
            }
        }
        MPI_Gatherv(energy.data(), static_cast<int>(energy.size()), MPI_DOUBLE, global.data(),
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "ElementEnergy");
    }
    int status = 0;
    if (validate) {
        double sums[] = {energySum, fluxSum}, totals[2], minimum, maximum;
        MPI_Reduce(sums, totals, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&low, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&high, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", totals[0], totals[1]);
            printf("  Energy range: [%.6f, %.6f]\n", minimum, maximum);
            if (std::abs(totals[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            status = !(std::isfinite(totals[0]) && std::isfinite(totals[1]) &&
                       std::isfinite(minimum) && std::isfinite(maximum));
            printf("  Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
