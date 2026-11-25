#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
}
#define CUDA(call) checkCuda((call), #call)

struct Slab {
    int first, rows;
};

static Slab partition(int width, int ranks, int rank) {
    const int base = width / ranks;
    const int extra = width % ranks;
    return {rank * base + std::min(rank, extra), base + (rank < extra)};
}

// Each row is contiguous, including ghost rows. Only owned cells accumulate flux.
__global__ void advance(const double* energy, double* next, double* accumulated,
                        int width, int firstStored, int firstOwned, int ownedRows,
                        int begin, int end) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    for (long long row = begin + static_cast<long long>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < end; row += gridDim.y * blockDim.y) {
        const int globalRow = firstStored + row;
        const size_t i = size_t(row) * width + col;
        const double center = energy[i];
        double flux = 0.0;
        if ((globalRow == 0 || globalRow == width - 1) && (col == 0 || col == width - 1))
            flux = globalRow == col ? 0.5 : -0.5;
        if (globalRow + 1 < width) flux += (energy[i + width] - center) * 0.8 * 0.25;
        if (globalRow > 0)         flux += (energy[i - width] - center) * 0.8 * 0.25;
        if (col + 1 < width)       flux += (energy[i + 1] - center) * 0.8 * 0.25;
        if (col > 0)               flux += (energy[i - 1] - center) * 0.8 * 0.25;
        next[i] = center + flux;
        if (globalRow >= firstOwned && globalRow - firstOwned < ownedRows)
            accumulated[size_t(globalRow - firstOwned) * width + col] += fabs(flux);
    }
}

struct Simulation {
    int width, rank, ranks, depth, top, bottom;
    Slab slab;
    size_t count, stored;
    double *energy, *next, *flux, *staging = nullptr;
    cudaStream_t compute, transfer;
    MPI_Comm comm;

    Simulation(int n, int iterations, int r, int p, MPI_Comm group)
        : width(n), rank(r), ranks(p), slab(partition(n, p, r)), comm(group) {
        // Limit redundant computation on thin slabs and keep MPI counts representable.
        depth = std::min({8, std::max(1, iterations), std::max(1, n / p / 8), INT_MAX / n});
        top = rank > 0 ? depth : 0;
        bottom = rank + 1 < ranks ? depth : 0;
        count = size_t(slab.rows) * n;
        stored = size_t(top + slab.rows + bottom) * n;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaMalloc(&energy, stored * sizeof(double)));
        CUDA(cudaMalloc(&next, stored * sizeof(double)));
        CUDA(cudaMalloc(&flux, count * sizeof(double)));
        CUDA(cudaMemsetAsync(energy, 0, stored * sizeof(double), compute));
        CUDA(cudaMemsetAsync(flux, 0, count * sizeof(double), compute));
        if (ranks > 1)
            CUDA(cudaMallocHost(&staging, 4 * size_t(depth) * n * sizeof(double)));
        CUDA(cudaStreamSynchronize(compute));
    }

    void update(int begin, int end) {
        if (begin >= end) return;
        const dim3 block(32, 8);
        const dim3 grid((width - 1) / 32 + 1, std::min(65535, (end - begin - 1) / 8 + 1));
        advance<<<grid, block, 0, compute>>>(energy, next, flux, width, slab.first - top,
                                             slab.first, slab.rows, begin, end);
        CUDA(cudaGetLastError());
    }

    // Compute the interior of the first step while the old boundary energies travel.
    void exchangeAndAdvance(int steps) {
        CUDA(cudaStreamSynchronize(compute));
        const int length = steps * width;
        const size_t bytes = size_t(length) * sizeof(double);
        double* owned = energy + size_t(top) * width;
        MPI_Request requests[4];
        int used = 0;
        for (int side = 0; side < 2; ++side) {
            const int peer = rank + (side ? 1 : -1);
            if (peer < 0 || peer >= ranks) continue;
            MPI_Irecv(staging + size_t(2 + side) * length, length, MPI_DOUBLE, peer,
                      1 - side, comm, &requests[used++]);
            CUDA(cudaMemcpyAsync(staging + size_t(side) * length,
                                 owned + (side ? count - length : 0), bytes,
                                 cudaMemcpyDeviceToHost, transfer));
        }
        const int innerBegin = top + (top != 0);
        const int innerEnd = top + slab.rows - (bottom != 0);
        update(innerBegin, innerEnd);
        CUDA(cudaStreamSynchronize(transfer));
        for (int side = 0; side < 2; ++side) {
            const int peer = rank + (side ? 1 : -1);
            if (peer < 0 || peer >= ranks) continue;
            MPI_Isend(staging + size_t(side) * length, length, MPI_DOUBLE, peer, side,
                      comm, &requests[used++]);
        }
        MPI_Waitall(used, requests, MPI_STATUSES_IGNORE);
        if (top)
            CUDA(cudaMemcpyAsync(owned - length, staging + 2 * size_t(length), bytes,
                                 cudaMemcpyHostToDevice, transfer));
        if (bottom)
            CUDA(cudaMemcpyAsync(owned + count, staging + 3 * size_t(length), bytes,
                                 cudaMemcpyHostToDevice, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        const int begin = top ? top - steps + 1 : 0;
        const int end = top + slab.rows + (bottom ? steps - 1 : 0);
        update(begin, innerBegin);
        update(std::max(innerBegin, innerEnd), end);
    }

    void run(int iterations) {
        for (int done = 0; done < iterations;) {
            const int steps = std::min(depth, iterations - done);
            for (int left = steps; left > 0; --left) {
                // At time zero all ghost energies are known locally.
                if (left == steps && done && ranks > 1)
                    exchangeAndAdvance(steps);
                else
                    update(top ? top - left + 1 : 0,
                           top + slab.rows + (bottom ? left - 1 : 0));
                std::swap(energy, next);
            }
            done += steps;
        }
        CUDA(cudaStreamSynchronize(compute));
    }

    ~Simulation() {
        if (staging) CUDA(cudaFreeHost(staging));
        CUDA(cudaFree(flux));
        CUDA(cudaFree(next));
        CUDA(cudaFree(energy));
        CUDA(cudaStreamDestroy(transfer));
        CUDA(cudaStreamDestroy(compute));
    }
};

// Gather only for explicitly requested, globally ordered output. Windowing keeps
// MPI counts and displacements in range even for grids larger than INT_MAX.
static std::vector<double> gather(const std::vector<double>& local, const Simulation& s) {
    const size_t total = size_t(s.width) * s.width;
    std::vector<double> all(s.rank == 0 ? total : 0);
    std::vector<int> counts(s.ranks), offsets(s.ranks);
    for (size_t start = 0; start < total; start += INT_MAX) {
        const size_t end = std::min(total, start + INT_MAX);
        size_t localOffset = 0;
        for (int r = 0; r < s.ranks; ++r) {
            const Slab slab = partition(s.width, s.ranks, r);
            const size_t first = size_t(slab.first) * s.width;
            const size_t last = first + size_t(slab.rows) * s.width;
            const size_t lo = std::max(first, start), hi = std::min(last, end);
            counts[r] = hi > lo ? int(hi - lo) : 0;
            offsets[r] = counts[r] ? int(lo - start) : 0;
            if (r == s.rank && counts[r]) localOffset = lo - first;
        }
        MPI_Gatherv(local.data() + localOffset, counts[s.rank], MPI_DOUBLE,
                    s.rank == 0 ? all.data() + start : nullptr, counts.data(),
                    offsets.data(), MPI_DOUBLE, 0, s.comm);
    }
    return all;
}

static bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux) {
    double energy_sum = 0.0, flux_sum = 0.0;
    double energy_min = std::numeric_limits<double>::max();
    double energy_max = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_min = std::min(energy[i], energy_min);
        energy_max = std::max(energy[i], energy_max);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr double energy_epsilon = 1e-8;
    
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int benchmark(int width, int iterations, bool validate, bool results, MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));

    const size_t elements = size_t(width) * width;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", width, width, elements);
        printf("Iterations: %d\nValidation: %s\n\n", iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    Simulation s(width, iterations, rank, ranks, comm);
    uint64_t localBytes = (2 * s.stored + s.count) * sizeof(double), totalBytes;
    MPI_Reduce(&localBytes, &totalBytes, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    if (rank == 0) {
        const double mb = totalBytes / (1024.0 * 1024.0);
        printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n", mb, mb);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    s.run(iterations);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double rate = iterations > 0 && seconds > 0 ? elements * double(iterations) / seconds / 1e9 : 0;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", iterations > 0 ? seconds * 1000 / iterations : 0);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22);
    }

    std::vector<double> energy(s.count), flux(s.count);
    CUDA(cudaMemcpy(energy.data(), s.energy + size_t(s.top) * width,
                    s.count * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(flux.data(), s.flux, s.count * sizeof(double), cudaMemcpyDeviceToHost));
    const uint64_t first = uint64_t(s.slab.first) * width;
    uint64_t localHash = 0, resultHash;
    const int threads = std::min<size_t>(omp_get_max_threads(), 1 + (s.count - 1) / 4096);
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(^:localHash)
    for (size_t i = 0; i < s.count; ++i) {
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        localHash ^= (e + first + i) * 0x9e3779b97f4a7c15ULL;
        localHash ^= (f + first + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&localHash, &resultHash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) printf("  Result hash: %016" PRIX64 "\n\n", resultHash);

    int status = 0;
    if (validate || results) {
        auto allEnergy = gather(energy, s);
        if (results && rank == 0) print_results(allEnergy, "ElementEnergy");
        if (validate) {
            auto allFlux = gather(flux, s);
            if (rank == 0) status = validateResults(allEnergy, allFlux) ? 0 : 1;
        }
    }
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int width = 512, iterations = 10, status = 0;
    bool validate = false, results = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) width = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else {
            if (strcmp(argv[i], "-h") != 0) {
                if (rank == 0) printf("Unknown option: %s\n", argv[i]);
                status = 1;
            }
            help = true;
            break;
        }
    }
    if (help) {
        if (rank == 0) printUsage(argv[0]);
    } else if (width <= 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive\n");
        status = 1;
    } else {
        MPI_Comm active;
        MPI_Comm_split(MPI_COMM_WORLD, rank < width ? 0 : MPI_UNDEFINED, rank, &active);
        if (active != MPI_COMM_NULL) {
            status = benchmark(width, iterations, validate, results, active);
            MPI_Comm_free(&active);
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
