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

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) check((call), #call)

struct Slab {
    int first, rows;
};

static Slab partition(int n, int ranks, int r) {
    const int base = n / ranks, extra = n % ranks;
    return {r * base + std::min(r, extra), base + (r < extra)};
}

// At time t only rows and columns within t cells of a corner can carry energy.
// Each axis is the union of two intervals, merged when the fronts meet.
__global__ void update(const double* input, double* output, double* flux, int n, int origin,
                       int owned, int rows, int left, int width, int first, int top, int bottom,
                       int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height)
        return;
    if (col >= left)
        col += n - width;
    row = row < top ? first + row : bottom + (row - top);
    const size_t i = size_t(row - origin) * n + col;
    const double e = input[i];
    double change = 0.0;
    if ((row == 0 || row == n - 1) && (col == 0 || col == n - 1))
        change = row == col ? 0.5 : -0.5;
    if (row < n - 1)
        change += ((input[i + n] - e) * 0.8) * 0.25;
    if (row > 0)
        change += ((input[i - n] - e) * 0.8) * 0.25;
    if (col < n - 1)
        change += ((input[i + 1] - e) * 0.8) * 0.25;
    if (col > 0)
        change += ((input[i - 1] - e) * 0.8) * 0.25;
    output[i] = e + change;
    if (row >= owned && row < owned + rows)
        flux[size_t(row - owned) * n + col] += fabs(change);
}

// Copy contiguous energy strips through pinned memory for portable MPI transport.
// An interface outside the wavefront has an all-zero halo on both ranks.
static void exchange(double* energy, double* staging, Slab slab, int n, int depth, int pad,
                     int elapsed, int rank) {
    const int count = depth * n;
    const size_t bytes = size_t(count) * sizeof(double);
    const int edges[] = {slab.first, slab.first + slab.rows};
    MPI_Request requests[4];
    bool active[2];
    int pending = 0;
    for (int side = 0; side < 2; ++side) {
        const int edge = edges[side];
        active[side] = edge > 0 && edge < n && std::min(edge, n - edge) - depth < elapsed;
        if (!active[side])
            continue;
        double* send = staging + size_t(2 * side) * count;
        const int start = pad + (side ? slab.rows - depth : 0);
        CUDA(cudaMemcpyAsync(send, energy + size_t(start) * n, bytes, cudaMemcpyDeviceToHost));
    }
    if (!active[0] && !active[1])
        return;
    CUDA(cudaStreamSynchronize(0));
    for (int side = 0; side < 2; ++side) {
        if (!active[side])
            continue;
        MPI_Irecv(staging + size_t(2 * side + 1) * count, count, MPI_DOUBLE, rank + (side ? 1 : -1),
                  0, MPI_COMM_WORLD, &requests[pending++]);
        MPI_Isend(staging + size_t(2 * side) * count, count, MPI_DOUBLE, rank + (side ? 1 : -1), 0,
                  MPI_COMM_WORLD, &requests[pending++]);
    }
    MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
    for (int side = 0; side < 2; ++side) {
        if (!active[side])
            continue;
        const int start = pad + (side ? slab.rows : -depth);
        CUDA(cudaMemcpyAsync(energy + size_t(start) * n, staging + size_t(2 * side + 1) * count,
                             bytes, cudaMemcpyHostToDevice));
    }
}

static uint64_t bits(double value) {
    uint64_t word;
    memcpy(&word, &value, sizeof(word));
    return word;
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

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
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
            if (rank == 0)
                printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0)
                printf("Unknown option: %s\n", argv[i]);
            if (rank == 0)
                printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    const int n = n_elems_root;
    if (n <= 0 || int64_t(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0)
            fprintf(stderr, "Grid dimensions exceed the supported element range\n");
        MPI_Finalize();
        return 1;
    }
    const int cells = n * n;
    const Slab slab = partition(n, ranks, rank);
    const int local = slab.rows * n;
    const int depth = std::min(8, std::max(1, n / ranks));
    const int pad = ranks > 1 ? depth : 0;
    const size_t plane = size_t(slab.rows + 2 * pad) * n;
    double *current = nullptr, *next = nullptr, *flux = nullptr, *staging = nullptr;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int node_rank;
    MPI_Comm_rank(node, &node_rank);
    MPI_Comm_free(&node);
    if (local) {
        int devices;
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(node_rank % devices));
        // Load the kernel before timing, including any driver-side PTX compilation.
        cudaFuncAttributes attributes;
        CUDA(cudaFuncGetAttributes(&attributes, update));
        CUDA(cudaMalloc(&current, plane * sizeof(double)));
        CUDA(cudaMalloc(&next, plane * sizeof(double)));
        CUDA(cudaMalloc(&flux, size_t(local) * sizeof(double)));
        if (pad)
            CUDA(cudaMallocHost(&staging, size_t(4) * depth * n * sizeof(double)));
        CUDA(cudaMemset(current, 0, plane * sizeof(double)));
        CUDA(cudaMemset(next, 0, plane * sizeof(double)));
        CUDA(cudaMemset(flux, 0, size_t(local) * sizeof(double)));
        CUDA(cudaDeviceSynchronize());
    }
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, cells);
        printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
        const size_t ghosts = size_t(2) * pad * n * std::min(ranks, n);
        const double mb = (3.0 * cells + 2.0 * ghosts) * sizeof(double) / (1024 * 1024);
        printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n", mb, mb);
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; local && t < n_iters;) {
        const int batch = std::min(depth, n_iters - t);
        if (pad && t)
            exchange(current, staging, slab, n, batch, pad, t, rank);
        for (int step = 1; step <= batch; ++step) {
            // Recompute successively narrower ghost strips until only owned rows remain.
            const int halo = pad ? batch - step : 0;
            const int first = std::max(0, slab.first - halo);
            const int end = std::min(n, slab.first + slab.rows + halo);
            const int left = std::min(n, t + step);
            const int right = std::max(left, n - left);
            const int width = left + (n - right);
            const int top = std::max(0, std::min(end, left) - first);
            const int bottom = std::max(first, right);
            const int height = top + std::max(0, end - bottom);
            if (height) {
                const dim3 block(32, 8);
                const dim3 grid((width + 31) / 32, (height + 7) / 8);
                update<<<grid, block>>>(current, next, flux, n, slab.first - pad, slab.first,
                                        slab.rows, left, width, first, top, bottom, height);
                CUDA(cudaGetLastError());
            }
            std::swap(current, next);
        }
        t += batch;
    }
    if (local)
        CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double rate = double(std::max(0, n_iters)) * cells / (seconds * 1e9);
        printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000);
        printf("  Time per iteration: %.4f ms\n", seconds * 1000 / std::max(n_iters, 1));
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate,
               rate * 22);
    }

    std::vector<double> energy(local), total_flux(local);
    if (local) {
        CUDA(cudaMemcpy(energy.data(), current + size_t(pad) * n, size_t(local) * sizeof(double),
                        cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(total_flux.data(), flux, size_t(local) * sizeof(double),
                        cudaMemcpyDeviceToHost));
        CUDA(cudaFree(current));
        CUDA(cudaFree(next));
        CUDA(cudaFree(flux));
        if (pad)
            CUDA(cudaFreeHost(staging));
    }
    uint64_t hash = 0, global_hash;
    double sum = 0, flow = 0, low = INFINITY, high = -INFINITY;
    const int threads = std::min(omp_get_max_threads(), 1 + local / 32768);
#pragma omp parallel for num_threads(threads) schedule(static) reduction(^ : hash)                 \
    reduction(+ : sum, flow) reduction(min : low) reduction(max : high)
    for (int i = 0; i < local; ++i) {
        const uint64_t index = size_t(slab.first) * n + i;
        const double e = energy[i], f = total_flux[i];
        hash ^= (bits(e) + index) * 0x9e3779b97f4a7c15ULL;
        hash ^= (bits(f) + index) * 0xbf58476d1ce4e5b9ULL;
        if (validate) {
            sum += e;
            flow += f;
            low = std::min(low, e);
            high = std::max(high, e);
        }
    }
    MPI_Reduce(&hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
    if (printResults) {
        std::vector<int> counts, offsets;
        std::vector<double> all;
        if (rank == 0) {
            counts.resize(ranks);
            offsets.resize(ranks);
            all.resize(cells);
            for (int r = 0; r < ranks; ++r) {
                const Slab part = partition(n, ranks, r);
                counts[r] = part.rows * n;
                offsets[r] = part.first * n;
            }
        }
        MPI_Gatherv(energy.data(), local, MPI_DOUBLE, all.data(), counts.data(), offsets.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            print_results(all, "ElementEnergy");
    }
    int valid = 1;
    if (validate) {
        double sums[] = {sum, flow}, global[2], minimum, maximum;
        MPI_Reduce(sums, global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&low, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&high, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", global[0],
                   global[1]);
            printf("  Energy range: [%.6f, %.6f]\n", minimum, maximum);
            if (std::abs(global[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            valid = std::isfinite(global[0]) && std::isfinite(global[1]) &&
                    std::isfinite(minimum) && std::isfinite(maximum);
            printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
