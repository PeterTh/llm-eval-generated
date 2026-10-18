#include <algorithm>
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
using idx_t = uint64_t;

static void cudaCheck(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d, line %d: %s: %s\n", rank, line,
                expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call, __LINE__)

// A Cartesian block of the mesh. Connectivity is implicit because buildSquare2D
// always generated this topology, with unit connection flux and coefficient .8.
// Only energy needs a halo; accumulated flux is private to its owning element.
struct World {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 1, n = 0;
    int dims[2] = {1, 1}, coords[2] = {0, 0};
    int neighbors[4];                    // +x, -x, +y, -y (reference order)
    int rows = 0, cols = 0, x0 = 0, y0 = 0;
    size_t pitch = 0;
    val_t *energy = nullptr, *next = nullptr, *flux = nullptr;
    val_t *sendDevice = nullptr, *recvDevice = nullptr;
    val_t *sendHost = nullptr, *recvHost = nullptr;
    size_t offsets[4];
    int counts[4];
    cudaStream_t compute, exchange;
    cudaEvent_t done, haloReady;
    std::vector<val_t> hostEnergy, hostFlux;
};

static int blockSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}

static int blockStart(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts);
}

void buildSquare2D(World& w, int n) {
    w.n = n;
    MPI_Comm_rank(w.comm, &w.rank);
    MPI_Comm_size(w.comm, &w.size);
    int periods[2];
    MPI_Cart_get(w.comm, 2, w.dims, periods, w.coords);
    MPI_Cart_shift(w.comm, 0, 1, &w.neighbors[1], &w.neighbors[0]);
    MPI_Cart_shift(w.comm, 1, 1, &w.neighbors[3], &w.neighbors[2]);
    w.rows = blockSize(n, w.dims[0], w.coords[0]);
    w.cols = blockSize(n, w.dims[1], w.coords[1]);
    w.x0 = blockStart(n, w.dims[0], w.coords[0]);
    w.y0 = blockStart(n, w.dims[1], w.coords[1]);
    // Pad the row stride to a multiple of 128 bytes for coalesced accesses.
    w.pitch = (static_cast<size_t>(w.cols) + 2 + 15) / 16 * 16;
    size_t bytes = (static_cast<size_t>(w.rows) + 2) * w.pitch * sizeof(val_t);
    CUDA(cudaMalloc(&w.energy, bytes));
    CUDA(cudaMalloc(&w.next, bytes));
    CUDA(cudaMalloc(&w.flux, bytes));
    CUDA(cudaMemset(w.energy, 0, bytes));
    CUDA(cudaMemset(w.next, 0, bytes));
    CUDA(cudaMemset(w.flux, 0, bytes));
    CUDA(cudaStreamCreateWithFlags(&w.compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&w.exchange, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&w.done, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&w.haloReady, cudaEventDisableTiming));
    w.counts[0] = w.counts[1] = w.cols;
    w.counts[2] = w.counts[3] = w.rows;
    size_t total = 0;
    for (int d = 0; d < 4; ++d) {
        w.offsets[d] = total;
        total += w.counts[d];
    }
    if (w.size > 1) {
        CUDA(cudaMalloc(&w.sendDevice, total * sizeof(val_t)));
        CUDA(cudaMalloc(&w.recvDevice, total * sizeof(val_t)));
        // Pinned staging works with ordinary MPI, without requiring CUDA-aware MPI.
        CUDA(cudaMallocHost(&w.sendHost, total * sizeof(val_t)));
        CUDA(cudaMallocHost(&w.recvHost, total * sizeof(val_t)));
    }
    w.hostEnergy.resize(static_cast<size_t>(w.rows) * w.cols);
    w.hostFlux.resize(w.hostEnergy.size());
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaEventRecord(w.done, w.compute));
}

__global__ void packHalo(const val_t* energy, val_t* packed,
                         size_t pitch, int rows, int cols) {
    size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t count = 2 * (static_cast<size_t>(rows) + cols);
    for (; k < count; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (k < static_cast<size_t>(cols))
            packed[k] = energy[rows * pitch + k + 1];
        else if (k < 2 * static_cast<size_t>(cols))
            packed[k] = energy[pitch + k - cols + 1];
        else if (k < 2 * static_cast<size_t>(cols) + rows)
            packed[k] = energy[(k - 2 * cols + 1) * pitch + cols];
        else
            packed[k] = energy[(k - 2 * cols - rows + 1) * pitch + 1];
    }
}

__global__ void unpackHalo(val_t* energy, const val_t* packed,
                           size_t pitch, int rows, int cols,
                           bool xp, bool xm, bool yp, bool ym) {
    size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t count = 2 * (static_cast<size_t>(rows) + cols);
    for (; k < count; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (k < static_cast<size_t>(cols)) {
            if (xp) energy[(rows + 1) * pitch + k + 1] = packed[k];
        } else if (k < 2 * static_cast<size_t>(cols)) {
            if (xm) energy[k - cols + 1] = packed[k];
        } else if (k < 2 * static_cast<size_t>(cols) + rows) {
            if (yp) energy[(k - 2 * cols + 1) * pitch + cols + 1] = packed[k];
        } else {
            if (ym) energy[(k - 2 * cols - rows + 1) * pitch] = packed[k];
        }
    }
}

// region: 0 = full block, 1 = strict interior, 2 = perimeter. Every owned
// element is written exactly once, even for one-row/one-column partitions.
__global__ void update(const val_t* energy, val_t* next, val_t* accumulated,
                       size_t pitch, int rows, int cols, int x0, int y0,
                       int n, int region, size_t count) {
    size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    for (; k < count; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int x, y;
        if (region == 1) {
            x = static_cast<int>(k / (cols - 2)) + 1;
            y = static_cast<int>(k % (cols - 2)) + 1;
        } else if (region == 2 && rows > 1 && cols > 1) {
            if (k < static_cast<size_t>(cols)) { x = 0; y = k; }
            else if (k < 2 * static_cast<size_t>(cols)) { x = rows - 1; y = k - cols; }
            else {
                size_t side = k - 2 * cols;
                x = static_cast<int>(side / 2) + 1;
                y = (side % 2 == 0) ? 0 : cols - 1;
            }
        } else {
            x = k / cols;
            y = k % cols;
        }
        size_t i = (x + 1) * pitch + y + 1;
        int gx = x0 + x, gy = y0 + y;
        val_t e = energy[i];
        val_t total = 0.0;
        if ((gx == 0 || gx == n - 1) && (gy == 0 || gy == n - 1))
            total = ((gx == 0 && gy == 0) || (gx == n - 1 && gy == n - 1)) ? 0.5 : -0.5;
        // Retain the original connection order and floating-point operations.
        if (gx + 1 < n) total += (energy[i + pitch] - e) * 0.8 * 1.0 * 0.25;
        if (gx > 0)     total += (energy[i - pitch] - e) * 0.8 * 1.0 * 0.25;
        if (gy + 1 < n) total += (energy[i + 1] - e) * 0.8 * 1.0 * 0.25;
        if (gy > 0)     total += (energy[i - 1] - e) * 0.8 * 1.0 * 0.25;
        next[i] = e + total;
        accumulated[i] += fabs(total);
    }
}

static unsigned blocks(size_t count) {
    return static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
}

void runSimulation(World& w, int iterations) {
    size_t owned = static_cast<size_t>(w.rows) * w.cols;
    size_t interior = static_cast<size_t>(std::max(w.rows - 2, 0)) * std::max(w.cols - 2, 0);
    size_t perimeter = owned - interior;
    size_t haloCount = 2 * (static_cast<size_t>(w.rows) + w.cols);
    for (int iter = 0; iter < iterations; ++iter) {
        if (w.size == 1) {
            update<<<blocks(owned), 256, 0, w.compute>>>(w.energy, w.next, w.flux,
                w.pitch, w.rows, w.cols, w.x0, w.y0, w.n, 0, owned);
        } else {
            // MPI calls stay on the main thread (MPI_THREAD_FUNNELED).
            MPI_Request requests[8];
            int nr = 0;
            for (int d = 0; d < 4; ++d)
                if (w.neighbors[d] != MPI_PROC_NULL)
                    MPI_Irecv(w.recvHost + w.offsets[d], w.counts[d], MPI_DOUBLE,
                              w.neighbors[d], d ^ 1, w.comm, &requests[nr++]);
            CUDA(cudaStreamWaitEvent(w.exchange, w.done, 0));
            packHalo<<<blocks(haloCount), 256, 0, w.exchange>>>(w.energy,
                w.sendDevice, w.pitch, w.rows, w.cols);
            CUDA(cudaMemcpyAsync(w.sendHost, w.sendDevice, haloCount * sizeof(val_t),
                                 cudaMemcpyDeviceToHost, w.exchange));
            if (interior)
                update<<<blocks(interior), 256, 0, w.compute>>>(w.energy, w.next,
                    w.flux, w.pitch, w.rows, w.cols, w.x0, w.y0, w.n, 1, interior);
            CUDA(cudaStreamSynchronize(w.exchange));
            for (int d = 0; d < 4; ++d)
                if (w.neighbors[d] != MPI_PROC_NULL)
                    MPI_Isend(w.sendHost + w.offsets[d], w.counts[d], MPI_DOUBLE,
                              w.neighbors[d], d, w.comm, &requests[nr++]);
            MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
            CUDA(cudaMemcpyAsync(w.recvDevice, w.recvHost, haloCount * sizeof(val_t),
                                 cudaMemcpyHostToDevice, w.exchange));
            unpackHalo<<<blocks(haloCount), 256, 0, w.exchange>>>(w.energy,
                w.recvDevice, w.pitch, w.rows, w.cols,
                w.neighbors[0] != MPI_PROC_NULL, w.neighbors[1] != MPI_PROC_NULL,
                w.neighbors[2] != MPI_PROC_NULL, w.neighbors[3] != MPI_PROC_NULL);
            CUDA(cudaEventRecord(w.haloReady, w.exchange));
            CUDA(cudaStreamWaitEvent(w.compute, w.haloReady, 0));
            update<<<blocks(perimeter), 256, 0, w.compute>>>(w.energy, w.next,
                w.flux, w.pitch, w.rows, w.cols, w.x0, w.y0, w.n, 2, perimeter);
            CUDA(cudaEventRecord(w.done, w.compute));
        }
        std::swap(w.energy, w.next);
    }
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(w.compute));
}

static void download(World& w) {
    CUDA(cudaMemcpy2D(w.hostEnergy.data(), w.cols * sizeof(val_t),
        w.energy + w.pitch + 1, w.pitch * sizeof(val_t),
        w.cols * sizeof(val_t), w.rows, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy2D(w.hostFlux.data(), w.cols * sizeof(val_t),
        w.flux + w.pitch + 1, w.pitch * sizeof(val_t),
        w.cols * sizeof(val_t), w.rows, cudaMemcpyDeviceToHost));
}

// Hash uses original global element indices, so XOR reduction is independent
// of decomposition. memcpy avoids the reference's strict-aliasing violation.
static uint64_t computeHash(const World& w) {
    uint64_t local = 0, global = 0;
    #pragma omp parallel for reduction(^:local) schedule(static)
    for (size_t k = 0; k < w.hostEnergy.size(); ++k) {
        uint64_t e, f;
        memcpy(&e, &w.hostEnergy[k], sizeof(e));
        memcpy(&f, &w.hostFlux[k], sizeof(f));
        uint64_t i = (w.x0 + k / w.cols) * static_cast<uint64_t>(w.n) + w.y0 + k % w.cols;
        local ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        local ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&local, &global, 1, MPI_UINT64_T, MPI_BXOR, 0, w.comm);
    return global;
}

static bool validateResults(const World& w) {
    val_t energy = 0, flux = 0;
    val_t lo = std::numeric_limits<val_t>::max();
    val_t hi = std::numeric_limits<val_t>::lowest();
    int finite = 1;
    #pragma omp parallel for reduction(+:energy,flux) reduction(min:lo,finite) reduction(max:hi) schedule(static)
    for (size_t i = 0; i < w.hostEnergy.size(); ++i) {
        energy += w.hostEnergy[i];
        flux += w.hostFlux[i];
        lo = std::min(lo, w.hostEnergy[i]);
        hi = std::max(hi, w.hostEnergy[i]);
        finite = std::min(finite, int(std::isfinite(w.hostEnergy[i]) && std::isfinite(w.hostFlux[i])));
    }
    val_t sums[2] = {energy, flux}, global[2], minEnergy, maxEnergy;
    int allFinite;
    MPI_Allreduce(sums, global, 2, MPI_DOUBLE, MPI_SUM, w.comm);
    MPI_Allreduce(&lo, &minEnergy, 1, MPI_DOUBLE, MPI_MIN, w.comm);
    MPI_Allreduce(&hi, &maxEnergy, 1, MPI_DOUBLE, MPI_MAX, w.comm);
    MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, w.comm);
    bool valid = allFinite && std::isfinite(global[0]) && std::isfinite(global[1])
                 && std::isfinite(minEnergy) && std::isfinite(maxEnergy);
    if (w.rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
               "  Energy range: [%.6f, %.6f]\n", global[0], global[1], minEnergy, maxEnergy);
        if (std::abs(global[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

static void printResults(const World& w) {
    // Gather only on request; normal runs retain distributed storage throughout.
    std::vector<int> counts(w.size), displacements(w.size);
    int total = 0;
    for (int rank = 0; rank < w.size; ++rank) {
        int c[2];
        MPI_Cart_coords(w.comm, rank, 2, c);
        counts[rank] = blockSize(w.n, w.dims[0], c[0]) * blockSize(w.n, w.dims[1], c[1]);
        displacements[rank] = total;
        total += counts[rank];
    }
    std::vector<val_t> packed(w.rank == 0 ? total : 0);
    MPI_Gatherv(w.hostEnergy.data(), static_cast<int>(w.hostEnergy.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, w.comm);
    if (w.rank == 0) {
        std::vector<val_t> ordered(total);
        #pragma omp parallel for schedule(static)
        for (int rank = 0; rank < w.size; ++rank) {
            // No MPI calls inside an OpenMP region: Cartesian reorder is disabled.
            int cx = rank / w.dims[1], cy = rank % w.dims[1];
            int rows = blockSize(w.n, w.dims[0], cx);
            int cols = blockSize(w.n, w.dims[1], cy);
            int x0 = blockStart(w.n, w.dims[0], cx), y0 = blockStart(w.n, w.dims[1], cy);
            for (int x = 0; x < rows; ++x)
                std::copy_n(packed.data() + displacements[rank] + static_cast<size_t>(x) * cols,
                            cols, ordered.data() + static_cast<size_t>(x0 + x) * w.n + y0);
        }
        print_results(ordered, "ElementEnergy");
    }
}

static void destroy(World& w) {
    CUDA(cudaFree(w.energy));
    CUDA(cudaFree(w.next));
    CUDA(cudaFree(w.flux));
    if (w.size > 1) {
        CUDA(cudaFree(w.sendDevice));
        CUDA(cudaFree(w.recvDevice));
        CUDA(cudaFreeHost(w.sendHost));
        CUDA(cudaFreeHost(w.recvHost));
    }
    CUDA(cudaEventDestroy(w.done));
    CUDA(cudaEventDestroy(w.haloReady));
    CUDA(cudaStreamDestroy(w.compute));
    CUDA(cudaStreamDestroy(w.exchange));
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInt(const char* text, int& value) {
    char* end = nullptr;
    long v = strtol(text, &end, 10);
    if (!*text || *end || v < 0 || v > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(v);
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512, iterations = 10, status = 0;
    bool validate = false, results = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            bool grid = !strcmp(argv[i], "-n");
            if (!parseInt(argv[++i], grid ? n : iterations)) status = 1;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else {
            if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            status = 1;
        }
    }
    // The original uses int global element counts. Keep counts representable in
    // MPI's count type as well, while using size_t for padded/device indexing.
    if (n <= 0 || n > 46340) {
        if (rank == 0) fprintf(stderr, "Grid size must be in [1, 46340]\n");
        status = 1;
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }

    // Use the largest nonempty rectangular process grid, breaking ties in favor
    // of a square to minimize halo traffic. Excess ranks participate in shutdown.
    int dims[2] = {1, 1}, active = 1;
    for (int x = 1; x <= std::min(n, size); ++x) {
        int y = std::min(n, size / x);
        if (x * y > active || (x * y == active && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            dims[0] = x; dims[1] = y; active = x * y;
        }
    }
    MPI_Comm participating;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &participating);
    if (rank < active) {
        MPI_Comm node;
        MPI_Comm_split_type(participating, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank, devices;
        MPI_Comm_rank(node, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&node);
        World world;
        int periods[2] = {0, 0};
        MPI_Cart_create(participating, 2, dims, periods, 0, &world.comm);
        buildSquare2D(world, n);
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %d elements\n", n, n, n * n);
            printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (%d x %d), OpenMP threads/rank: %d\n", active, dims[0], dims[1], omp_get_max_threads());
            printf("Building unstructured mesh...\nRunning simulation...\n");
        }
        MPI_Barrier(world.comm);
        double start = MPI_Wtime();
        runSimulation(world, iterations);
        double elapsed = MPI_Wtime() - start, maximum;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
        download(world);
        uint64_t hash = computeHash(world);
        if (rank == 0) {
            double rate = maximum > 0 ? static_cast<double>(iterations) * n * n / maximum / 1e9 : 0;
            printf("Computation time: %.3f ms\nPerformance:\n", maximum * 1000);
            printf("  Time per iteration: %.4f ms\n", iterations ? maximum * 1000 / iterations : 0);
            printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
            printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));
        }
        if (results) printResults(world);
        if (validate && !validateResults(world)) status = 1;
        destroy(world);
        MPI_Comm_free(&world.comm);
        MPI_Comm_free(&participating);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
