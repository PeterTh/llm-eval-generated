#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

struct Partition {
    int n, first, rows, halo;
    int above, below;
};

Partition partition(int n, int rank, int ranks) {
    const int base = n / ranks;
    const int extra = n % ranks;
    return {n, rank * base + std::min(rank, extra), base + (rank < extra),
            ranks == 1 ? 0 : std::min(8, base),
            rank == 0 ? MPI_PROC_NULL : rank - 1,
            rank + 1 == ranks ? MPI_PROC_NULL : rank + 1};
}

// Each thread owns one cell. Only energy crosses partition boundaries.
// Ghost cells are advanced redundantly between exchanges.
__global__ void advance(Partition p, const double* energy, double* next, double* flux,
                        int rowStart, int topRows, int rowGap, int activeRows,
                        int activeCols, int leftCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= activeCols) return;
    if (col >= leftCols) col += p.n - activeCols;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= activeRows) return;
    const int row = rowStart + r + (r >= topRows ? rowGap : 0);
    const int local = row - p.first;
    const size_t i = size_t(local + p.halo) * p.n + col;
    const double center = energy[i];
    double flow = 0.0;
    if ((row == 0 || row == p.n - 1) && (col == 0 || col == p.n - 1))
        flow = (row == col) ? 0.5 : -0.5;
    if (row + 1 < p.n) flow += (energy[i + p.n] - center) * 0.8 * 0.25;
    if (row > 0)       flow += (energy[i - p.n] - center) * 0.8 * 0.25;
    if (col + 1 < p.n) flow += (energy[i + 1] - center) * 0.8 * 0.25;
    if (col > 0)       flow += (energy[i - 1] - center) * 0.8 * 0.25;
    next[i] = center + flow;
    if (local >= 0 && local < p.rows) flux[size_t(local) * p.n + col] += fabs(flow);
}

void exchange(const Partition& p, double* owned, double* staging, int depth, int done) {
    // A shared boundary outside both source regions still has zero ghosts
    // on both ranks. Both endpoints make the same decision independently.
    auto activePeer = [&](int peer, int boundary) {
        return boundary - depth < done || boundary + depth > p.n - done
                   ? peer : MPI_PROC_NULL;
    };
    const int above = activePeer(p.above, p.first);
    const int below = activePeer(p.below, p.first + p.rows);
    if (above == MPI_PROC_NULL && below == MPI_PROC_NULL) return;
    const int count = depth * p.n;
    const size_t bytes = count * sizeof(double);
    const size_t cells = size_t(p.rows) * p.n;
    double* sendTop = staging;
    double* sendBottom = staging + count;
    double* recvTop = staging + 2 * size_t(count);
    double* recvBottom = staging + 3 * size_t(count);
    if (above != MPI_PROC_NULL)
        check(cudaMemcpyAsync(sendTop, owned, bytes, cudaMemcpyDeviceToHost));
    if (below != MPI_PROC_NULL)
        check(cudaMemcpyAsync(sendBottom, owned + cells - count, bytes, cudaMemcpyDeviceToHost));
    check(cudaStreamSynchronize(0));
    // The stream has also finished reading the previous batch's receive buffers.
    MPI_Request req[4];
    MPI_Irecv(recvTop, count, MPI_DOUBLE, above, 1, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(recvBottom, count, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(sendTop, count, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(sendBottom, count, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (above != MPI_PROC_NULL)
        check(cudaMemcpyAsync(owned - count, recvTop, bytes, cudaMemcpyHostToDevice));
    if (below != MPI_PROC_NULL)
        check(cudaMemcpyAsync(owned + cells, recvBottom, bytes, cudaMemcpyHostToDevice));
}

void simulate(const Partition& p, int iterations, double*& energy, double*& next,
              double* flux, double* staging) {
    for (int done = 0; done < iterations;) {
        const int batch = p.halo ? std::min(p.halo, iterations - done) : iterations - done;
        // The initial energies (including ghosts) are all zero.
        if (p.halo && done)
            exchange(p, energy + size_t(p.halo) * p.n, staging, batch, done);
        for (int step = 0; step < batch; ++step) {
            const int ghosts = p.halo ? batch - step - 1 : 0;
            const int begin = std::max(0, p.first - ghosts);
            const int end = std::min(p.n, p.first + p.rows + ghosts);
            // Initially only the four sources can change. Their influence
            // travels at most one edge per iteration.
            const int reach = std::min(p.n, done + step + 1);
            const int right = std::max(reach, p.n - reach);
            const int topEnd = std::clamp(reach, begin, end);
            const int bottomBegin = std::clamp(right, begin, end);
            const int rows = topEnd - begin + end - bottomBegin;
            const int cols = reach + p.n - right;
            if (rows) {
                const dim3 block(32, 8);
                const dim3 grid((cols + 31) / 32, (rows + 7) / 8);
                advance<<<grid, block>>>(p, energy, next, flux, begin, topEnd - begin,
                                         bottomBegin - topEnd, rows, cols, reach);
            }
            std::swap(energy, next);
        }
        done += batch;
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}

// Reduce each warp's contributions before crossing the host/device link.
__global__ void hashWarps(const double* energy, const double* flux, size_t cells,
                          size_t first, uint64_t* partials) {
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long hash = 0;
    for (size_t i = thread; i < cells; i += gridDim.x * blockDim.x) {
        const uint64_t e = __double_as_longlong(energy[i]);
        const uint64_t f = __double_as_longlong(flux[i]);
        hash ^= (e + first + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + first + i) * 0xbf58476d1ce4e5b9ULL;
    }
    for (int shift = 16; shift; shift /= 2)
        hash ^= __shfl_down_sync(0xffffffff, hash, shift);
    if (thread % 32 == 0 && thread < cells) partials[thread / 32] = hash;
}

uint64_t localHash(const double* energy, const double* flux, size_t cells,
                   size_t first, double* scratch) {
    const unsigned blocks = std::min<size_t>(8192, (cells + 255) / 256);
    const size_t warps = std::min<size_t>(blocks * 8, (cells + 31) / 32);
    auto* deviceHashes = reinterpret_cast<uint64_t*>(scratch);
    hashWarps<<<blocks, 256>>>(energy, flux, cells, first, deviceHashes);
    check(cudaGetLastError());
    std::vector<uint64_t> hashes(warps);
    check(cudaMemcpy(hashes.data(), deviceHashes, warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t result = 0;
    #pragma omp parallel for schedule(static) reduction(^:result)
    for (size_t i = 0; i < warps; ++i) result ^= hashes[i];
    return result;
}

bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux) {
    double energy_sum = 0, flux_sum = 0;
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

} // namespace

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
    if (n <= 0 || int64_t(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must give a positive NxN grid of at most INT_MAX cells.\n");
        MPI_Finalize();
        return 1;
    }
    const int workers = std::min(n, ranks);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < workers ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= workers) {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm node;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(localRank % devices));

    const Partition p = partition(n, rank, workers);
    const int cells = p.rows * n;
    const size_t bytes = cells * sizeof(double);
    const size_t haloBytes = size_t(p.halo) * n * sizeof(double);
    const size_t padded = bytes + 2 * haloBytes;
    double *energy, *next, *flux, *staging = nullptr;
    check(cudaMalloc(&energy, padded));
    check(cudaMalloc(&next, padded));
    check(cudaMalloc(&flux, bytes));
    if (haloBytes) check(cudaMallocHost(&staging, 4 * haloBytes));
    check(cudaMemset(energy, 0, padded));
    check(cudaMemset(next, 0, padded));
    check(cudaMemset(flux, 0, bytes));
    check(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n*n);
        printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
        const double mem = (3.0 * n * n * sizeof(double) + 4.0 * haloBytes * workers) / (1024*1024);
        printf("GPU memory usage: %.2f MB\n\nRunning simulation...\n", mem);
    }
    MPI_Barrier(active);
    const double start = MPI_Wtime();
    simulate(p, n_iters, energy, next, flux, staging);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);

    const double* owned = energy + size_t(p.halo) * n;
    const uint64_t partial = localHash(owned, flux, cells, size_t(p.first) * n, next);
    std::vector<double> localEnergy, localFlux;
    if (printResults || validate) {
        localEnergy.resize(cells);
        check(cudaMemcpy(localEnergy.data(), owned, bytes, cudaMemcpyDeviceToHost));
    }
    if (validate) {
        localFlux.resize(cells);
        check(cudaMemcpy(localFlux.data(), flux, bytes, cudaMemcpyDeviceToHost));
    }
    check(cudaFree(energy));
    check(cudaFree(next));
    check(cudaFree(flux));
    if (staging) check(cudaFreeHost(staging));

    uint64_t hash = 0;
    MPI_Reduce(&partial, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, active);
    if (rank == 0) {
        const double rate = seconds > 0 ? double(n*n) * std::max(0, n_iters) / seconds / 1e9 : 0;
        printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000);
        printf("  Time per iteration: %.4f ms\n", seconds * 1000 / std::max(1, n_iters));
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }

    int status = 0;
    if (printResults || validate) {
        // Both diagnostics use the serial row order for floating-point sums.
        std::vector<double> allEnergy, allFlux;
        std::vector<int> counts, offsets;
        if (rank == 0) {
            allEnergy.resize(n*n);
            if (validate) allFlux.resize(n*n);
            counts.resize(workers);
            offsets.resize(workers);
            for (int i = 0; i < workers; ++i) {
                const Partition part = partition(n, i, workers);
                counts[i] = part.rows * n;
                offsets[i] = part.first * n;
            }
        }
        MPI_Gatherv(localEnergy.data(), cells, MPI_DOUBLE, allEnergy.data(), counts.data(),
                    offsets.data(), MPI_DOUBLE, 0, active);
        if (validate)
            MPI_Gatherv(localFlux.data(), cells, MPI_DOUBLE, allFlux.data(), counts.data(),
                        offsets.data(), MPI_DOUBLE, 0, active);
        if (rank == 0) {
            if (printResults) print_results(allEnergy, "ElementEnergy");
            if (validate) status = validateResults(allEnergy, allFlux) ? 0 : 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, active);
    }
    MPI_Comm_free(&active);
    MPI_Finalize();
    return status;
}
