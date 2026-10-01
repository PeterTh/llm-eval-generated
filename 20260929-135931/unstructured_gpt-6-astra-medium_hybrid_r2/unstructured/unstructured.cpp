#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error at line %d (%s): %s\n",
                rank, line, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call, __LINE__)

struct Tile {
    int n, rows, cols, x0, y0;
    int pitch;
};

static void partition(int n, int parts, int coordinate, int& start, int& length) {
    length = n / parts + (coordinate < n % parts);
    start = coordinate * (n / parts) + std::min(coordinate, n % parts);
}

// buildSquare2D always generates unit-weight axial connections and coefficient
// 0.8. Encode that connectivity implicitly, eliminating the large, redundant
// adjacency arrays. Only energy needs a halo; cumulative flux is strictly local.
__device__ __forceinline__ void updateElement(
        const double* __restrict__ energy, double* __restrict__ next,
        double* __restrict__ flux, Tile t, int x, int y) {
    const size_t p = size_t(x + 1) * t.pitch + y + 1;
    const int gx = t.x0 + x, gy = t.y0 + y;
    const double e = energy[p];
    double f = 0.0;
    // Preserve the original corner assignment order, including the 1x1 case.
    if (gx == 0 && gy == 0) f = 0.5;
    if (gx == 0 && gy == t.n - 1) f = -0.5;
    if (gx == t.n - 1 && gy == 0) f = -0.5;
    if (gx == t.n - 1 && gy == t.n - 1) f = 0.5;
    // Same neighbor order and rounding as computeFlux in the serial benchmark.
    // Do not replace the two multiplications with a precomputed coefficient.
    if (gx + 1 < t.n) f += (energy[p + t.pitch] - e) * 0.8 * 0.25;
    if (gx > 0)       f += (energy[p - t.pitch] - e) * 0.8 * 0.25;
    if (gy + 1 < t.n) f += (energy[p + 1] - e) * 0.8 * 0.25;
    if (gy > 0)       f += (energy[p - 1] - e) * 0.8 * 0.25;
    next[p] = e + f;
    flux[size_t(x) * t.cols + y] += fabs(f);
}

template<bool Interior>
__global__ void updateGrid(const double* energy, double* next, double* flux, Tile t) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x + (Interior ? 1 : 0);
    const int x = blockIdx.y * blockDim.y + threadIdx.y + (Interior ? 1 : 0);
    if (x < t.rows - (Interior ? 1 : 0) && y < t.cols - (Interior ? 1 : 0))
        updateElement(energy, next, flux, t, x, y);
}

__global__ void updateBoundary(const double* energy, double* next, double* flux,
                               Tile t, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int x, y;
    if (i < t.cols) { x = 0; y = i; }
    else if (i < 2 * t.cols) { x = t.rows - 1; y = i - t.cols; }
    else {
        const int sides = t.cols > 1 ? 2 : 1;
        x = 1 + (i - 2 * t.cols) / sides;
        y = ((i - 2 * t.cols) % sides) ? t.cols - 1 : 0;
    }
    updateElement(energy, next, flux, t, x, y);
}

// Packed order: x-minus, x-plus, y-minus, y-plus. The same layout is used for
// sends and receives. Pinned staging works with ordinary and CUDA-aware MPI.
__global__ void packHalo(const double* energy, double* halo, Tile t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t.cols) {
        halo[i] = energy[t.pitch + i + 1];
        halo[t.cols + i] = energy[size_t(t.rows) * t.pitch + i + 1];
    }
    if (i < t.rows) {
        halo[2 * t.cols + i] = energy[size_t(i + 1) * t.pitch + 1];
        halo[2 * t.cols + t.rows + i] = energy[size_t(i + 1) * t.pitch + t.cols];
    }
}

__global__ void unpackHalo(const double* halo, double* energy, Tile t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t.cols) {
        energy[i + 1] = halo[i];
        energy[size_t(t.rows + 1) * t.pitch + i + 1] = halo[t.cols + i];
    }
    if (i < t.rows) {
        energy[size_t(i + 1) * t.pitch] = halo[2 * t.cols + i];
        energy[size_t(i + 1) * t.pitch + t.cols + 1] = halo[2 * t.cols + t.rows + i];
    }
}

struct World {
    Tile tile;
    double *energy = nullptr, *next = nullptr, *flux = nullptr;
    double *sendDevice = nullptr, *recvDevice = nullptr;
    double *sendHost = nullptr, *recvHost = nullptr;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, haloReady;
    size_t energyBytes, fluxBytes, haloBytes;

    explicit World(Tile t) : tile(t) {
        energyBytes = size_t(t.rows + 2) * t.pitch * sizeof(double);
        fluxBytes = size_t(t.rows) * t.cols * sizeof(double);
        haloBytes = size_t(2 * (t.rows + t.cols)) * sizeof(double);
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
        CUDA(cudaMalloc(&energy, energyBytes));
        CUDA(cudaMalloc(&next, energyBytes));
        CUDA(cudaMalloc(&flux, fluxBytes));
        CUDA(cudaMalloc(&sendDevice, haloBytes));
        CUDA(cudaMalloc(&recvDevice, haloBytes));
        CUDA(cudaMallocHost(&sendHost, haloBytes));
        CUDA(cudaMallocHost(&recvHost, haloBytes));
        CUDA(cudaMemsetAsync(energy, 0, energyBytes, compute));
        CUDA(cudaMemsetAsync(next, 0, energyBytes, compute));
        CUDA(cudaMemsetAsync(flux, 0, fluxBytes, compute));
        CUDA(cudaMemsetAsync(recvDevice, 0, haloBytes, compute));
        // Physical-boundary ghost cells are unused, but initialized nonetheless.
        std::memset(recvHost, 0, haloBytes);
        CUDA(cudaStreamSynchronize(compute));
    }
    ~World() {
        CUDA(cudaFree(energy)); CUDA(cudaFree(next)); CUDA(cudaFree(flux));
        CUDA(cudaFree(sendDevice)); CUDA(cudaFree(recvDevice));
        CUDA(cudaFreeHost(sendHost)); CUDA(cudaFreeHost(recvHost));
        CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(haloReady));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    }
};

static void runSimulation(World& w, int iterations, MPI_Comm cart, const int peers[4]) {
    const Tile t = w.tile;
    const dim3 block(32, 8);
    const dim3 grid((t.cols + 31) / 32, (t.rows + 7) / 8);
    const int haloBlocks = (std::max(t.rows, t.cols) + 255) / 256;
    const int edges = t.rows == 1 ? t.cols :
                      2 * t.cols + (t.rows - 2) * std::min(t.cols, 2);
    const int counts[4] = {t.cols, t.cols, t.rows, t.rows};
    const int offsets[4] = {0, t.cols, 2 * t.cols, 2 * t.cols + t.rows};
    bool distributed = false;
    for (int d = 0; d < 4; ++d) distributed |= peers[d] != MPI_PROC_NULL;
    bool cudaAware = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    for (int iter = 0; iter < iterations; ++iter) {
        if (!distributed) {
            updateGrid<false><<<grid, block, 0, w.compute>>>(w.energy, w.next, w.flux, t);
        } else {
            MPI_Request requests[8];
            int requestsCount = 0;
            CUDA(cudaEventRecord(w.ready, w.compute));
            CUDA(cudaStreamWaitEvent(w.transfer, w.ready, 0));
            packHalo<<<haloBlocks, 256, 0, w.transfer>>>(w.energy, w.sendDevice, t);
            if (!cudaAware)
                CUDA(cudaMemcpyAsync(w.sendHost, w.sendDevice, w.haloBytes,
                                     cudaMemcpyDeviceToHost, w.transfer));
            // Interior work proceeds while the host drives MPI progress.
            if (t.rows > 2 && t.cols > 2) {
                const dim3 inner((t.cols - 2 + 31) / 32, (t.rows - 2 + 7) / 8);
                updateGrid<true><<<inner, block, 0, w.compute>>>(w.energy, w.next, w.flux, t);
            }
            CUDA(cudaStreamSynchronize(w.transfer));
            // This also finishes the previous iteration's unpack/H2D before
            // MPI can overwrite its receive buffer. MPI has no stream ordering.
            double* receive = cudaAware ? w.recvDevice : w.recvHost;
            double* send = cudaAware ? w.sendDevice : w.sendHost;
            for (int d = 0; d < 4; ++d)
                if (peers[d] != MPI_PROC_NULL)
                    MPI_Irecv(receive + offsets[d], counts[d], MPI_DOUBLE,
                              peers[d], d ^ 1, cart, &requests[requestsCount++]);
            for (int d = 0; d < 4; ++d)
                if (peers[d] != MPI_PROC_NULL)
                    MPI_Isend(send + offsets[d], counts[d], MPI_DOUBLE,
                              peers[d], d, cart, &requests[requestsCount++]);
            MPI_Waitall(requestsCount, requests, MPI_STATUSES_IGNORE);
            if (!cudaAware)
                CUDA(cudaMemcpyAsync(w.recvDevice, w.recvHost, w.haloBytes,
                                     cudaMemcpyHostToDevice, w.transfer));
            unpackHalo<<<haloBlocks, 256, 0, w.transfer>>>(w.recvDevice, w.energy, t);
            CUDA(cudaEventRecord(w.haloReady, w.transfer));
            CUDA(cudaStreamWaitEvent(w.compute, w.haloReady, 0));
            updateBoundary<<<(edges + 255) / 256, 256, 0, w.compute>>>(
                w.energy, w.next, w.flux, t, edges);
        }
        CUDA(cudaGetLastError());
        std::swap(w.energy, w.next);
    }
    CUDA(cudaStreamSynchronize(w.compute));
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, int& value) {
    char* end;
    errno = 0;
    const long number = std::strtol(text, &end, 10);
    if (errno || end == text || *end || number < 0 || number > INT_MAX) return false;
    value = static_cast<int>(number);
    return true;
}

static void printDistributedResults(const std::vector<double>& energy, Tile t,
                                    MPI_Comm cart, int rank, int size, const int dims[2]) {
    if (rank != 0) {
        MPI_Send(energy.data(), static_cast<int>(energy.size()), MPI_DOUBLE, 0, 9, cart);
        return;
    }
    std::vector<double> global(size_t(t.n) * t.n);
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < t.rows; ++x)
        std::copy_n(energy.data() + size_t(x) * t.cols, t.cols,
                    global.data() + size_t(t.x0 + x) * t.n + t.y0);
    // Receive each tile directly into its global row-major position. Gathering
    // is only needed for -r, whose FNV hash is order-dependent.
    for (int source = 1; source < size; ++source) {
        int coords[2], x0, y0, rows, cols;
        MPI_Cart_coords(cart, source, 2, coords);
        partition(t.n, dims[0], coords[0], x0, rows);
        partition(t.n, dims[1], coords[1], y0, cols);
        MPI_Datatype tileType;
        MPI_Type_vector(rows, cols, t.n, MPI_DOUBLE, &tileType);
        MPI_Type_commit(&tileType);
        MPI_Recv(global.data() + size_t(x0) * t.n + y0, 1, tileType,
                 source, 9, cart, MPI_STATUS_IGNORE);
        MPI_Type_free(&tileType);
    }
    print_results(global, "ElementEnergy");
}

static int benchmark(int n, int iterations, bool validate, bool printResults,
                     MPI_Comm cart, int rank, int size, const int dims[2]) {
    int coords[2], peers[4];
    MPI_Cart_coords(cart, rank, 2, coords);
    MPI_Cart_shift(cart, 0, 1, &peers[0], &peers[1]);
    MPI_Cart_shift(cart, 1, 1, &peers[2], &peers[3]);
    Tile t{};
    t.n = n;
    partition(n, dims[0], coords[0], t.x0, t.rows);
    partition(n, dims[1], coords[1], t.y0, t.cols);
    // Pad each row to a warp boundary for coalesced device accesses.
    t.pitch = ((t.cols + 2 + 31) / 32) * 32;
    World world(t);
    const uint64_t memory = 2 * world.energyBytes + world.fluxBytes + 2 * world.haloBytes;
    uint64_t globalMemory = 0;
    MPI_Reduce(&memory, &globalMemory, 1, MPI_UINT64_T, MPI_SUM, 0, cart);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", n, n,
               static_cast<unsigned long long>(uint64_t(n) * n));
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI partition: %d x %d (%d ranks), OpenMP threads/rank: %d\n",
               dims[0], dims[1], size, omp_get_max_threads());
        printf("Device memory usage (all ranks): %.2f MB\n\n", globalMemory / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    runSimulation(world, iterations, cart, peers);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const double rate = seconds > 0 ? double(n) * n * iterations / seconds / 1e9 : 0;
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Performance:\n  Time per iteration: %.4f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
    }

    const size_t count = size_t(t.rows) * t.cols;
    std::vector<double> energy(count), flux(count);
    CUDA(cudaMemcpy2D(energy.data(), size_t(t.cols) * sizeof(double),
                      world.energy + t.pitch + 1, size_t(t.pitch) * sizeof(double),
                      size_t(t.cols) * sizeof(double), t.rows, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(flux.data(), world.flux, world.fluxBytes, cudaMemcpyDeviceToHost));
    uint64_t localHash = 0, resultHash = 0;
    double energySum = 0, fluxSum = 0;
    double energyMin = std::numeric_limits<double>::max();
    double energyMax = std::numeric_limits<double>::lowest();
    int nonfinite = 0;
    // OpenMP handles host processing without MPI calls from worker threads.
    #pragma omp parallel for schedule(static) reduction(^:localHash) \
        reduction(+:energySum,fluxSum) reduction(min:energyMin) \
        reduction(max:energyMax) reduction(|:nonfinite)
    for (size_t i = 0; i < count; ++i) {
        const uint64_t globalIndex = uint64_t(t.x0 + i / t.cols) * n + t.y0 + i % t.cols;
        uint64_t e, f;
        std::memcpy(&e, &energy[i], sizeof(e));
        std::memcpy(&f, &flux[i], sizeof(f));
        localHash ^= (e + globalIndex) * 0x9e3779b97f4a7c15ULL;
        localHash ^= (f + globalIndex) * 0xbf58476d1ce4e5b9ULL;
        if (validate) {
            energySum += energy[i];
            fluxSum += flux[i];
            energyMin = std::min(energyMin, energy[i]);
            energyMax = std::max(energyMax, energy[i]);
            nonfinite |= !std::isfinite(energy[i]) || !std::isfinite(flux[i]);
        }
    }
    MPI_Reduce(&localHash, &resultHash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart);
    if (rank == 0) printf("  Result hash: %016" PRIX64 "\n\n", resultHash);
    if (printResults) printDistributedResults(energy, t, cart, rank, size, dims);
    int failed = 0;
    if (validate) {
        double sums[2] = {energySum, fluxSum}, totals[2], minimum, maximum;
        MPI_Reduce(sums, totals, 2, MPI_DOUBLE, MPI_SUM, 0, cart);
        MPI_Reduce(&energyMin, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&energyMax, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        MPI_Reduce(&nonfinite, &failed, 1, MPI_INT, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n",
                   totals[0], totals[1]);
            printf("  Energy range: [%.6f, %.6f]\n", minimum, maximum);
            failed |= !std::isfinite(totals[0]) || !std::isfinite(totals[1]) ||
                      !std::isfinite(minimum) || !std::isfinite(maximum);
            if (std::abs(totals[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, cart);
    return failed;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) bad |= !parseNumber(argv[++i], n);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) bad |= !parseNumber(argv[++i], iterations);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]); bad = true; }
    }
    // The original benchmark uses signed-int element counts. Check its valid
    // range explicitly instead of overflowing allocations or MPI counts.
    bad |= n < 1 || uint64_t(n) * n > INT_MAX;
    if (help || bad) {
        if (rank == 0) {
            if (bad) fprintf(stderr, "Require 1 <= N <= 46340 and nonnegative iterations.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return bad ? 1 : 0;
    }

    // Avoid empty tiles when the launch has more ranks than the mesh can use.
    int active = static_cast<int>(std::min(uint64_t(size), uint64_t(n) * n));
    int dims[2];
    do {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(active, 2, dims);
        if (dims[0] <= n && dims[1] <= n) break;
        --active;
    } while (active > 0);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    int result = 0;
    if (rank < active) {
        MPI_Comm node;
        MPI_Comm_split_type(workers, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank, localSize, devices;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_size(node, &localSize);
        CUDA(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: this benchmark requires a CUDA GPU\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Also supports schedulers exposing one GPU per rank via CUDA_VISIBLE_DEVICES.
        CUDA(cudaSetDevice(localRank % devices));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
        MPI_Comm_free(&node);
        int periods[2] = {0, 0};
        MPI_Comm cart;
        MPI_Cart_create(workers, 2, dims, periods, 0, &cart);
        try {
            result = benchmark(n, iterations, validate, printResults, cart, rank, active, dims);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
