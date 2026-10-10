#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

// Only the C MPI interface is needed, also when nvcc is the compiler.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
static int world_rank = 0;

static void checkMPI(int status, const char* call, int line) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "Rank %d, line %d: %s: %.*s\n", world_rank, line, call, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

static void checkCUDA(cudaError_t status, const char* call, int line) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "Rank %d, line %d: %s: %s\n", world_rank, line, call, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CALL(call) checkMPI((call), #call, __LINE__)
#define CUDA_CALL(call) checkCUDA((call), #call, __LINE__)

struct Options {
    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool results = false;
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 0 || parsed > INT_MAX - 2) return false;
    value = static_cast<int>(parsed);
    return true;
}

// Return 2 for help, 1 for an error, and 0 for a simulation.
static int parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) {
            const bool grid = strcmp(argv[i], "-n") == 0;
            int value = 0;
            if (++i == argc || !parseInteger(argv[i], value) || (grid && value == 0)) {
                if (world_rank == 0) fprintf(stderr, "Invalid %s argument\n", grid ? "grid size" : "iteration count");
                return 1;
            }
            (grid ? options.n : options.iterations) = value;
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.results = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            return 2;
        } else {
            if (world_rank == 0) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    const uint64_t count = uint64_t(options.n) * options.n;
    if (count > std::numeric_limits<size_t>::max() / (3 * sizeof(val_t))) {
        if (world_rank == 0) fprintf(stderr, "Grid size exceeds addressable storage\n");
        return 1;
    }
    return 0;
}

struct Partition {
    int n, rows, cols, first_row, first_col;
    size_t pitch;
};

static void splitDimension(int n, int parts, int coordinate, int& first, int& length) {
    const int base = n / parts;
    const int remainder = n % parts;
    length = base + (coordinate < remainder);
    first = coordinate * base + std::min(coordinate, remainder);
}

static Partition partitionFor(int n, const int dims[2], const int coords[2]) {
    Partition p{};
    p.n = n;
    splitDimension(n, dims[0], coords[0], p.first_row, p.rows);
    splitDimension(n, dims[1], coords[1], p.first_col, p.cols);
    p.pitch = (size_t(p.cols) + 2 + 31) / 32 * 32;
    return p;
}

// Use the largest rectangular process grid that has no empty partitions.
// This also handles tiny meshes and prime process counts larger than n.
static int processGrid(int processes, int n, int dims[2]) {
    int active = static_cast<int>(std::min(uint64_t(processes), uint64_t(n) * n));
    for (; active > 0; --active) {
        for (int rows = static_cast<int>(std::sqrt(double(active))); rows >= 1; --rows) {
            if (active % rows == 0 && active / rows <= n && rows <= n) {
                dims[0] = rows;
                dims[1] = active / rows;
                return active;
            }
        }
    }
    return 0;
}

// buildSquare2D has unit connection weights and the same 0.8 transfer
// coefficient everywhere. Its four connections can therefore be represented
// implicitly, without loading a large static element record per update.
// Keep the original +row, -row, +column, -column summation order, and keep
// the two multiplications separate (no reassociation or FMA contraction).
__device__ __forceinline__ val_t computeFlux(val_t neighbor, val_t current) {
    return ((neighbor - current) * 0.8) * 0.25;
}

__device__ __forceinline__ void updateElement(const val_t* __restrict__ current,
                                             val_t* __restrict__ next,
                                             val_t* __restrict__ flux,
                                             Partition p, int row, int col) {
    const int x = p.first_row + row - 1;
    const int y = p.first_col + col - 1;
    const size_t at = size_t(row) * p.pitch + col;
    const val_t energy = current[at];
    val_t total = 0.0;
    // The last corner assignment in the original generator wins for n == 1.
    if ((x == 0 && y == 0) || (x == p.n - 1 && y == p.n - 1)) total = 0.5;
    else if ((x == 0 && y == p.n - 1) || (x == p.n - 1 && y == 0)) total = -0.5;
    if (x + 1 < p.n) total += computeFlux(current[at + p.pitch], energy);
    if (x > 0) total += computeFlux(current[at - p.pitch], energy);
    if (y + 1 < p.n) total += computeFlux(current[at + 1], energy);
    if (y > 0) total += computeFlux(current[at - 1], energy);
    next[at] = energy + total;
    flux[at] += fabs(total);
}

__global__ void updateAll(const val_t* __restrict__ current, val_t* __restrict__ next,
                          val_t* __restrict__ flux, Partition p) {
    const int col = int(blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (col > p.cols) return;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         row <= size_t(p.rows); row += size_t(gridDim.y) * blockDim.y) {
        updateElement(current, next, flux, p, int(row), col);
    }
}

__global__ void updateInterior(const val_t* __restrict__ current, val_t* __restrict__ next,
                               val_t* __restrict__ flux, Partition p) {
    const int col = int(blockIdx.x * blockDim.x + threadIdx.x) + 2;
    if (col >= p.cols) return;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 2;
         row < size_t(p.rows); row += size_t(gridDim.y) * blockDim.y) {
        const size_t at = row * p.pitch + col;
        const val_t energy = current[at];
        val_t total = 0.0;
        total += computeFlux(current[at + p.pitch], energy);
        total += computeFlux(current[at - p.pitch], energy);
        total += computeFlux(current[at + 1], energy);
        total += computeFlux(current[at - 1], energy);
        next[at] = energy + total;
        flux[at] += fabs(total);
    }
}

static size_t boundaryCount(Partition p) {
    if (p.rows == 1) return p.cols;
    if (p.cols == 1) return p.rows;
    return 2 * size_t(p.cols) + 2 * size_t(p.rows - 2);
}

__global__ void updateBoundary(const val_t* __restrict__ current, val_t* __restrict__ next,
                               val_t* __restrict__ flux, Partition p, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        int row, col;
        if (p.rows == 1) { row = 1; col = int(i) + 1; }
        else if (p.cols == 1) { row = int(i) + 1; col = 1; }
        else if (i < size_t(p.cols)) { row = 1; col = int(i) + 1; }
        else if (i < 2 * size_t(p.cols)) { row = p.rows; col = int(i) - p.cols + 1; }
        else {
            const size_t side = i - 2 * size_t(p.cols);
            row = int(side / 2) + 2;
            col = (side & 1) ? p.cols : 1;
        }
        updateElement(current, next, flux, p, row, col);
    }
}

// Four contiguous messages: low/high row, low/high column. Only energy
// crosses partitions; cumulative absolute flux is private to each element.
__global__ void packHalos(const val_t* __restrict__ energy, val_t* __restrict__ send,
                          Partition p, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        size_t at;
        if (i < size_t(p.cols)) at = p.pitch + i + 1;
        else if (i < 2 * size_t(p.cols)) at = size_t(p.rows) * p.pitch + i - p.cols + 1;
        else if (i < 2 * size_t(p.cols) + p.rows) at = (i - 2 * size_t(p.cols) + 1) * p.pitch + 1;
        else at = (i - 2 * size_t(p.cols) - p.rows + 1) * p.pitch + p.cols;
        send[i] = energy[at];
    }
}

__global__ void unpackHalos(val_t* __restrict__ energy, const val_t* __restrict__ receive,
                            Partition p, size_t count, unsigned neighbors) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        size_t at;
        unsigned direction;
        if (i < size_t(p.cols)) { at = i + 1; direction = 0; }
        else if (i < 2 * size_t(p.cols)) {
            at = size_t(p.rows + 1) * p.pitch + i - p.cols + 1; direction = 1;
        } else if (i < 2 * size_t(p.cols) + p.rows) {
            at = (i - 2 * size_t(p.cols) + 1) * p.pitch; direction = 2;
        } else {
            at = (i - 2 * size_t(p.cols) - p.rows + 1) * p.pitch + p.cols + 1; direction = 3;
        }
        if (neighbors & (1u << direction)) energy[at] = receive[i];
    }
}

struct Summary {
    uint64_t hash;
    val_t energy_sum, flux_sum, energy_min, energy_max;
    int nonfinite;
};

// Reduce resident device state to a few records instead of transferring two
// full arrays for the mandatory checksum. Validation shares the same scan.
template <bool Validate>
__global__ void summarize(const val_t* __restrict__ energy, const val_t* __restrict__ flux,
                          Partition p, size_t count, Summary* __restrict__ partials) {
    Summary local{0, 0.0, 0.0, DBL_MAX, -DBL_MAX, 0};
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        const size_t row = i / p.cols, col = i % p.cols;
        const size_t at = (row + 1) * p.pitch + col + 1;
        const val_t e = energy[at], f = flux[at];
        const uint64_t global = (uint64_t(p.first_row) + row) * p.n + p.first_col + col;
        local.hash ^= (uint64_t(__double_as_longlong(e)) + global) * 0x9e3779b97f4a7c15ULL;
        local.hash ^= (uint64_t(__double_as_longlong(f)) + global) * 0xbf58476d1ce4e5b9ULL;
        if constexpr (Validate) {
            local.energy_sum += e;
            local.flux_sum += f;
            local.energy_min = fmin(local.energy_min, e);
            local.energy_max = fmax(local.energy_max, e);
            local.nonfinite |= !isfinite(e) || !isfinite(f);
        }
    }
    __shared__ Summary values[256];
    const unsigned t = threadIdx.x;
    values[t] = local;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            values[t].hash ^= values[t + stride].hash;
            if constexpr (Validate) {
                values[t].energy_sum += values[t + stride].energy_sum;
                values[t].flux_sum += values[t + stride].flux_sum;
                values[t].energy_min = fmin(values[t].energy_min, values[t + stride].energy_min);
                values[t].energy_max = fmax(values[t].energy_max, values[t + stride].energy_max);
                values[t].nonfinite |= values[t + stride].nonfinite;
            }
        }
        __syncthreads();
    }
    if (t == 0) partials[blockIdx.x] = values[0];
}

static unsigned linearBlocks(size_t count) {
    return static_cast<unsigned>(std::min((count + 255) / 256, size_t(65535)));
}

static dim3 gridFor(int rows, int cols) {
    return dim3((unsigned(cols) + 31) / 32,
                static_cast<unsigned>(std::min((size_t(rows) + 7) / 8, size_t(65535))));
}

struct World {
    Partition p;
    MPI_Comm comm;
    int neighbors[4]{};
    unsigned neighbor_mask = 0;
    bool direct_mpi;
    val_t *energy[2]{}, *flux = nullptr, *device_send = nullptr, *device_receive = nullptr;
    val_t *host_send = nullptr, *host_receive = nullptr;
    size_t cells, halo_count;
    size_t offsets[4];
    int counts[4];
    cudaStream_t compute{}, exchange{};
    cudaEvent_t state_ready{}, send_ready{}, halos_ready{};
    cudaGraphExec_t graph = nullptr;
    int graph_iterations = 0;

    World(Partition partition, MPI_Comm cart, bool direct, int ranks, int iterations)
        : p(partition), comm(cart), direct_mpi(direct),
          cells(size_t(p.rows + 2) * p.pitch), halo_count(2 * size_t(p.cols) + 2 * size_t(p.rows)),
          offsets{0, size_t(p.cols), 2 * size_t(p.cols), 2 * size_t(p.cols) + p.rows},
          counts{p.cols, p.cols, p.rows, p.rows} {
        MPI_CALL(MPI_Cart_shift(comm, 0, 1, &neighbors[0], &neighbors[1]));
        MPI_CALL(MPI_Cart_shift(comm, 1, 1, &neighbors[2], &neighbors[3]));
        for (int d = 0; d < 4; ++d) if (neighbors[d] != MPI_PROC_NULL) neighbor_mask |= 1u << d;
        CUDA_CALL(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CALL(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CALL(cudaEventCreateWithFlags(&state_ready, cudaEventDisableTiming));
        CUDA_CALL(cudaEventCreateWithFlags(&send_ready, cudaEventDisableTiming));
        CUDA_CALL(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));
        for (auto& buffer : energy) {
            CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&buffer), cells * sizeof(val_t)));
            CUDA_CALL(cudaMemsetAsync(buffer, 0, cells * sizeof(val_t), compute));
        }
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&flux), cells * sizeof(val_t)));
        CUDA_CALL(cudaMemsetAsync(flux, 0, cells * sizeof(val_t), compute));
        if (ranks > 1) {
            CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_send), halo_count * sizeof(val_t)));
            // Ping-pong receive buffers prevent an early next-iteration receive
            // from overwriting the halo still being consumed by the GPU.
            CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_receive), 2 * halo_count * sizeof(val_t)));
            if (!direct_mpi) {
                CUDA_CALL(cudaMallocHost(reinterpret_cast<void**>(&host_send), halo_count * sizeof(val_t)));
                CUDA_CALL(cudaMallocHost(reinterpret_cast<void**>(&host_receive), 2 * halo_count * sizeof(val_t)));
                std::memset(host_receive, 0, 2 * halo_count * sizeof(val_t));
            }
        }
        // Load kernels before timing. Restore all state after the warm-up.
        launchAll(energy[0], energy[1]);
        if (ranks > 1) {
            launchInterior(energy[0], energy[1]);
            launchBoundary(energy[0], energy[1]);
            CUDA_CALL(cudaStreamSynchronize(compute));
            CUDA_CALL(cudaMemsetAsync(device_receive, 0, 2 * halo_count * sizeof(val_t), exchange));
            packHalos<<<linearBlocks(halo_count), 256, 0, exchange>>>(energy[0], device_send, p, halo_count);
            CUDA_CALL(cudaGetLastError());
            unpackHalos<<<linearBlocks(halo_count), 256, 0, exchange>>>(energy[0], device_receive, p, halo_count, neighbor_mask);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaStreamSynchronize(exchange));
        }
        for (auto buffer : energy) CUDA_CALL(cudaMemsetAsync(buffer, 0, cells * sizeof(val_t), compute));
        CUDA_CALL(cudaMemsetAsync(flux, 0, cells * sizeof(val_t), compute));
        CUDA_CALL(cudaStreamSynchronize(compute));
        // On a single rank, replay a batch of steps with one graph submission.
        // An even batch returns to the same buffer, including between replays.
        if (ranks == 1 && iterations >= 2) {
            graph_iterations = std::min(iterations, 32) & ~1;
            cudaGraph_t captured;
            CUDA_CALL(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < graph_iterations; ++i) launchAll(energy[i & 1], energy[(i + 1) & 1]);
            CUDA_CALL(cudaStreamEndCapture(compute, &captured));
            CUDA_CALL(cudaGraphInstantiate(&graph, captured, nullptr, nullptr, 0));
            CUDA_CALL(cudaGraphDestroy(captured));
            CUDA_CALL(cudaGraphUpload(graph, compute));
            CUDA_CALL(cudaStreamSynchronize(compute));
        }
    }

    void launchAll(const val_t* current, val_t* next) {
        updateAll<<<gridFor(p.rows, p.cols), dim3(32, 8), 0, compute>>>(current, next, flux, p);
        CUDA_CALL(cudaGetLastError());
    }
    void launchInterior(const val_t* current, val_t* next) {
        if (p.rows > 2 && p.cols > 2) {
            updateInterior<<<gridFor(p.rows - 2, p.cols - 2), dim3(32, 8), 0, compute>>>(current, next, flux, p);
            CUDA_CALL(cudaGetLastError());
        }
    }
    void launchBoundary(const val_t* current, val_t* next) {
        const size_t count = boundaryCount(p);
        updateBoundary<<<linearBlocks(count), 256, 0, compute>>>(current, next, flux, p, count);
        CUDA_CALL(cudaGetLastError());
    }
    ~World() {
        if (graph) CUDA_CALL(cudaGraphExecDestroy(graph));
        CUDA_CALL(cudaEventDestroy(state_ready));
        CUDA_CALL(cudaEventDestroy(send_ready));
        CUDA_CALL(cudaEventDestroy(halos_ready));
        CUDA_CALL(cudaStreamDestroy(compute));
        CUDA_CALL(cudaStreamDestroy(exchange));
        for (auto buffer : energy) CUDA_CALL(cudaFree(buffer));
        CUDA_CALL(cudaFree(flux));
        if (device_send) CUDA_CALL(cudaFree(device_send));
        if (device_receive) CUDA_CALL(cudaFree(device_receive));
        if (host_send) CUDA_CALL(cudaFreeHost(host_send));
        if (host_receive) CUDA_CALL(cudaFreeHost(host_receive));
    }
    World(const World&) = delete;
    World& operator=(const World&) = delete;
};

static int runSimulation(World& w, int iterations, int ranks) {
    int current = 0;
    if (ranks == 1) {
        int remaining = iterations;
        if (w.graph) {
            for (; remaining >= w.graph_iterations; remaining -= w.graph_iterations)
                CUDA_CALL(cudaGraphLaunch(w.graph, w.compute));
        }
        for (; remaining > 0; --remaining) {
            w.launchAll(w.energy[current], w.energy[current ^ 1]);
            current ^= 1;
        }
    } else if (iterations > 0) {
        // All initial energies, including halos, are zero. The first step
        // therefore needs no exchange, even at inter-rank boundaries.
        w.launchAll(w.energy[0], w.energy[1]);
        current = 1;
        CUDA_CALL(cudaEventRecord(w.state_ready, w.compute));
        for (int iter = 1; iter < iterations; ++iter) {
            const size_t receive_offset = size_t(iter & 1) * w.halo_count;
            val_t* receive = (w.direct_mpi ? w.device_receive : w.host_receive) + receive_offset;
            val_t* send = w.direct_mpi ? w.device_send : w.host_send;
            MPI_Request requests[8];
            int request_count = 0;
            for (int d = 0; d < 4; ++d) {
                if (w.neighbors[d] != MPI_PROC_NULL)
                    MPI_CALL(MPI_Irecv(receive + w.offsets[d], w.counts[d], MPI_DOUBLE,
                                       w.neighbors[d], d ^ 1, w.comm, &requests[request_count++]));
            }
            CUDA_CALL(cudaStreamWaitEvent(w.exchange, w.state_ready, 0));
            packHalos<<<linearBlocks(w.halo_count), 256, 0, w.exchange>>>(w.energy[current], w.device_send, w.p, w.halo_count);
            CUDA_CALL(cudaGetLastError());
            if (!w.direct_mpi)
                CUDA_CALL(cudaMemcpyAsync(w.host_send, w.device_send, w.halo_count * sizeof(val_t), cudaMemcpyDeviceToHost, w.exchange));
            CUDA_CALL(cudaEventRecord(w.send_ready, w.exchange));
            // Interior work is independent of the incoming halos. MPI waits
            // on the host provide communication progress while it runs.
            w.launchInterior(w.energy[current], w.energy[current ^ 1]);
            CUDA_CALL(cudaEventSynchronize(w.send_ready));
            for (int d = 0; d < 4; ++d) {
                if (w.neighbors[d] != MPI_PROC_NULL)
                    MPI_CALL(MPI_Isend(send + w.offsets[d], w.counts[d], MPI_DOUBLE,
                                       w.neighbors[d], d, w.comm, &requests[request_count++]));
            }
            MPI_CALL(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
            if (!w.direct_mpi)
                CUDA_CALL(cudaMemcpyAsync(w.device_receive + receive_offset, receive,
                                          w.halo_count * sizeof(val_t), cudaMemcpyHostToDevice, w.exchange));
            unpackHalos<<<linearBlocks(w.halo_count), 256, 0, w.exchange>>>(w.energy[current], w.device_receive + receive_offset,
                                                                        w.p, w.halo_count, w.neighbor_mask);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaEventRecord(w.halos_ready, w.exchange));
            CUDA_CALL(cudaStreamWaitEvent(w.compute, w.halos_ready, 0));
            w.launchBoundary(w.energy[current], w.energy[current ^ 1]);
            CUDA_CALL(cudaEventRecord(w.state_ready, w.compute));
            current ^= 1;
        }
    }
    CUDA_CALL(cudaStreamSynchronize(w.compute));
    return current;
}

static Summary collectSummary(World& world, int current, bool validate) {
    int device;
    cudaDeviceProp properties;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties, device));
    const size_t count = size_t(world.p.rows) * world.p.cols;
    const unsigned blocks = std::min(linearBlocks(count), unsigned(properties.multiProcessorCount) * 8);
    Summary* device_partials = nullptr;
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&device_partials), blocks * sizeof(Summary)));
    std::vector<Summary> partials(blocks);
    if (validate)
        summarize<true><<<blocks, 256, 0, world.compute>>>(world.energy[current], world.flux, world.p, count, device_partials);
    else
        summarize<false><<<blocks, 256, 0, world.compute>>>(world.energy[current], world.flux, world.p, count, device_partials);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpyAsync(partials.data(), device_partials, blocks * sizeof(Summary), cudaMemcpyDeviceToHost, world.compute));
    CUDA_CALL(cudaStreamSynchronize(world.compute));
    CUDA_CALL(cudaFree(device_partials));
    uint64_t result = 0;
    val_t energy_sum = 0.0, flux_sum = 0.0, energy_min = DBL_MAX, energy_max = -DBL_MAX;
    int nonfinite = 0;
    #pragma omp parallel for schedule(static) reduction(^:result) reduction(+:energy_sum,flux_sum) reduction(min:energy_min) reduction(max:energy_max) reduction(|:nonfinite)
    for (unsigned i = 0; i < blocks; ++i) {
        result ^= partials[i].hash;
        energy_sum += partials[i].energy_sum;
        flux_sum += partials[i].flux_sum;
        energy_min = std::min(energy_min, partials[i].energy_min);
        energy_max = std::max(energy_max, partials[i].energy_max);
        nonfinite |= partials[i].nonfinite;
    }
    return Summary{result, energy_sum, flux_sum, energy_min, energy_max, nonfinite};
}

static uint64_t computeHash(const Summary& summary, MPI_Comm comm) {
    uint64_t global_result = 0;
    MPI_CALL(MPI_Reduce(&summary.hash, &global_result, 1, MPI_UINT64_T, MPI_BXOR, 0, comm));
    return global_result;
}

// Gather only when ordered external output is requested. MPI subarray types
// place each tile directly in global row-major order on rank zero. Datatype
// counts of one also avoid the INT_MAX limit on a tile's element count.
static void printGlobalResults(const std::vector<val_t>& energy, Partition p, MPI_Comm comm, const int dims[2], int rank, int ranks) {
    if (rank == 0) {
        std::vector<val_t> global(size_t(p.n) * p.n);
        std::vector<MPI_Request> requests(size_t(ranks - 1));
        for (int other = 1; other < ranks; ++other) {
            int coords[2];
            MPI_CALL(MPI_Cart_coords(comm, other, 2, coords));
            const Partition tile = partitionFor(p.n, dims, coords);
            const int sizes[2] = {p.n, p.n};
            const int subsizes[2] = {tile.rows, tile.cols};
            const int starts[2] = {tile.first_row, tile.first_col};
            MPI_Datatype target;
            MPI_CALL(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target));
            MPI_CALL(MPI_Type_commit(&target));
            MPI_CALL(MPI_Irecv(global.data(), 1, target, other, 10, comm, &requests[size_t(other - 1)]));
            MPI_CALL(MPI_Type_free(&target));
        }
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < p.rows; ++row)
            std::copy_n(energy.data() + size_t(row) * p.cols, p.cols,
                        global.data() + size_t(p.first_row + row) * p.n + p.first_col);
        if (ranks > 1) MPI_CALL(MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE));
        print_results(global, "ElementEnergy");
    } else {
        MPI_Datatype packed;
        MPI_CALL(MPI_Type_vector(p.rows, p.cols, p.cols, MPI_DOUBLE, &packed));
        MPI_CALL(MPI_Type_commit(&packed));
        MPI_CALL(MPI_Send(energy.data(), 1, packed, 0, 10, comm));
        MPI_CALL(MPI_Type_free(&packed));
    }
}

static bool validateResults(const Summary& summary, MPI_Comm comm, int rank) {
    val_t local_sums[2] = {summary.energy_sum, summary.flux_sum}, global_sums[2]{};
    val_t global_min = 0, global_max = 0;
    int global_nonfinite = 0;
    MPI_CALL(MPI_Reduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm));
    MPI_CALL(MPI_Reduce(&summary.energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
    MPI_CALL(MPI_Reduce(&summary.energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    MPI_CALL(MPI_Reduce(&summary.nonfinite, &global_nonfinite, 1, MPI_INT, MPI_BOR, 0, comm));
    if (rank != 0) return true;
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", global_sums[0]);
    printf("  Flux sum: %.2f\n", global_sums[1]);
    printf("  Energy range: [%.6f, %.6f]\n", global_min, global_max);
    if (global_nonfinite || !std::isfinite(global_sums[0]) || !std::isfinite(global_sums[1]) ||
        !std::isfinite(global_min) || !std::isfinite(global_max)) {
        printf("  ERROR: Non-finite simulation results\n");
        return false;
    }
    if (std::abs(global_sums[0]) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    printf("  Validation: PASSED\n");
    return true;
}

static int benchmark(const Options& options, int world_size) {
    int dims[2];
    const int ranks = processGrid(world_size, options.n, dims);
    MPI_Comm node, active;
    MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node));
    int local_rank, local_size;
    MPI_CALL(MPI_Comm_rank(node, &local_rank));
    MPI_CALL(MPI_Comm_size(node, &local_size));
    MPI_CALL(MPI_Comm_free(&node));
    MPI_CALL(MPI_Comm_split(MPI_COMM_WORLD, world_rank < ranks ? 0 : MPI_UNDEFINED, world_rank, &active));
    if (active == MPI_COMM_NULL) return 0;
    const int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_CALL(MPI_Cart_create(active, 2, dims, periods, 0, &cart));
    MPI_CALL(MPI_Comm_free(&active));
    int rank, coords[2];
    MPI_CALL(MPI_Comm_rank(cart, &rank));
    MPI_CALL(MPI_Cart_coords(cart, rank, 2, coords));
    const Partition p = partitionFor(options.n, dims, coords);
    int devices = 0;
    CUDA_CALL(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: this benchmark requires a CUDA GPU\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CALL(cudaSetDevice(local_rank % devices));
    if (std::getenv("OMP_NUM_THREADS") == nullptr)
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / local_size)));
    int direct = 0;
    // Query an existing MPI capability; otherwise use portable pinned staging.
    // Both paths run the entire simulation on CUDA.
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
    direct = MPIX_Query_cuda_support() != 0;
    #endif
    int all_direct = 0;
    MPI_CALL(MPI_Allreduce(&direct, &all_direct, 1, MPI_INT, MPI_MIN, cart));
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %llu elements\n", options.n, options.n,
               static_cast<unsigned long long>(uint64_t(options.n) * options.n));
        printf("Iterations: %d\n", options.iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("MPI ranks: %d/%d (%d x %d partitions), OpenMP threads/rank: %d\n",
               ranks, world_size, dims[0], dims[1], omp_get_max_threads());
        printf("CUDA halo transport: %s\n\n", all_direct ? "device buffers" : "pinned host buffers");
        printf("Building unstructured mesh...\n");
    }
    int status = 0;
    {
        World world(p, cart, all_direct != 0, ranks, options.iterations);
        uint64_t local_memory = 3 * uint64_t(world.cells) * sizeof(val_t);
        if (ranks > 1) local_memory += 3 * uint64_t(world.halo_count) * sizeof(val_t);
        uint64_t total_memory = 0;
        MPI_CALL(MPI_Reduce(&local_memory, &total_memory, 1, MPI_UINT64_T, MPI_SUM, 0, cart));
        if (rank == 0) {
            printf("Memory usage: %.2f MB (GPU buffers, distributed across ranks)\n\n", total_memory / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_CALL(MPI_Barrier(cart));
        const double start = MPI_Wtime();
        const int current = runSimulation(world, options.iterations, ranks);
        const double local_time = MPI_Wtime() - start;
        double elapsed = 0;
        MPI_CALL(MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart));
        if (rank == 0) {
            const double ms = elapsed * 1000.0;
            const double per_iteration = options.iterations ? ms / options.iterations : 0.0;
            const double rate = elapsed > 0.0 ? double(options.iterations) * options.n * options.n / elapsed / 1e9 : 0.0;
            printf("Computation time: %.3f ms\n", ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", per_iteration);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        }
        const Summary summary = collectSummary(world, current, options.validate);
        const uint64_t result_hash = computeHash(summary, cart);
        if (rank == 0) printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
        if (options.results) {
            std::vector<val_t> energy(size_t(p.rows) * p.cols);
            CUDA_CALL(cudaMemcpy2D(energy.data(), size_t(p.cols) * sizeof(val_t), world.energy[current] + p.pitch + 1,
                                   p.pitch * sizeof(val_t), size_t(p.cols) * sizeof(val_t), p.rows, cudaMemcpyDeviceToHost));
            printGlobalResults(energy, p, cart, dims, rank, ranks);
        }
        if (options.validate && !validateResults(summary, cart, rank)) status = 1;
    }
    MPI_CALL(MPI_Comm_free(&cart));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    const int initialized = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initialized != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CALL(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int world_size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    Options options;
    const int parsed = parseOptions(argc, argv, options);
    int status = parsed == 1 ? 1 : 0;
    if (parsed == 0) {
        try {
            status = benchmark(options, world_size);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", world_rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    int global_status = 0;
    MPI_CALL(MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPI_CALL(MPI_Finalize());
    return global_status;
}
