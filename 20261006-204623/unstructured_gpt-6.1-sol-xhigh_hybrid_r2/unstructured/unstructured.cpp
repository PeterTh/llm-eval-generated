#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;
using idx_t = uint64_t;

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: %s: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

enum Face { TOP, BOTTOM, LEFT, RIGHT };
constexpr int opposite[4] = {BOTTOM, TOP, RIGHT, LEFT};

struct Partition {
    int n, rows, cols, row_start, col_start;
    size_t pitch, elements;
};

static void splitAxis(int n, int parts, int coordinate, int& start, int& count) {
    const int base = n / parts, extra = n % parts;
    count = base + (coordinate < extra);
    start = coordinate * base + std::min(coordinate, extra);
}

static Partition partitionFor(int n, const int dims[2], const int coords[2]) {
    Partition p{};
    p.n = n;
    splitAxis(n, dims[0], coords[0], p.row_start, p.rows);
    splitAxis(n, dims[1], coords[1], p.col_start, p.cols);
    p.pitch = (static_cast<size_t>(p.cols) + 33) & ~size_t(31);
    p.elements = static_cast<size_t>(p.rows) * p.cols;
    return p;
}

// The input mesh always has four unit-weight edges, in the order +x,-x,+y,-y.
// Represent these edges implicitly rather than moving 144 bytes of static
// connectivity per cell. Missing edges are skipped, not replaced by zero values.
// Energy is double buffered; cumulative flux only depends on the same cell.
template<bool interior>
__device__ __forceinline__ val_t updateCell(
    const val_t* __restrict__ input, val_t* __restrict__ output,
    val_t* __restrict__ accumulated, const Partition p, int row, int col) {
    const size_t i = static_cast<size_t>(row) * p.pitch + col;
    const val_t energy = input[i];
    val_t flow = 0.0;
    const int x = p.row_start + row - 1, y = p.col_start + col - 1;
    if (!interior && (x == 0 || x == p.n - 1) &&
                     (y == 0 || y == p.n - 1)) {
        // Also preserves the final corner assignment for the one-cell mesh.
        flow = ((x == 0 && y == 0) || (x == p.n - 1 && y == p.n - 1))
                   ? 0.5 : -0.5;
    }
    // Preserve the optimized reference's ordered flux accumulation: its
    // four-edge interior and two-edge corners round each flux separately,
    // while a three-edge boundary has a fused final scalar tail. Explicitly
    // contract only that tail so subnormal rounding is preserved as well.
    const bool fused_tail = !interior &&
        ((x == 0 || x == p.n - 1) != (y == 0 || y == p.n - 1));
    if (interior || x + 1 < p.n)
        flow += (input[i + p.pitch] - energy) * 0.8 * 0.25;
    if (interior || x > 0)
        flow += (input[i - p.pitch] - energy) * 0.8 * 0.25;
    if (interior || y + 1 < p.n) {
        const val_t term = (input[i + 1] - energy) * 0.8;
        flow = fused_tail && y == 0 ? __fma_rn(term, 0.25, flow) : flow + term * 0.25;
    }
    if (interior || y > 0) {
        const val_t term = (input[i - 1] - energy) * 0.8;
        flow = fused_tail ? __fma_rn(term, 0.25, flow) : flow + term * 0.25;
    }

    const val_t next = energy + flow;
    output[i] = next;
    accumulated[static_cast<size_t>(row - 1) * p.cols + col - 1] += fabs(flow);
    return next;
}

template<bool interior>
__global__ void updateCells(const val_t* __restrict__ input,
                            val_t* __restrict__ output,
                            val_t* __restrict__ accumulated, Partition p) {
    const int first = interior ? 2 : 1;
    const int rows = p.rows - (interior ? 2 : 0);
    const int cols = p.cols - (interior ? 2 : 0);
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= static_cast<size_t>(cols)) return;
    for (size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < static_cast<size_t>(rows);
         row += static_cast<size_t>(blockDim.y) * gridDim.y)
        updateCell<interior>(input, output, accumulated, p,
                             static_cast<int>(row) + first,
                             static_cast<int>(col) + first);
}

static size_t perimeterSize(const Partition& p) {
    return static_cast<size_t>(p.cols) * (p.rows > 1 ? 2 : 1) +
           static_cast<size_t>(std::max(p.rows - 2, 0)) * (p.cols > 1 ? 2 : 1);
}

// Update each boundary cell once and simultaneously pack its outgoing halos.
// This avoids a separate pack kernel and races at corners or on thin tiles.
__global__ void updateBoundary(const val_t* __restrict__ input,
                               val_t* __restrict__ output,
                               val_t* __restrict__ accumulated,
                               val_t* __restrict__ send, Partition p,
                               size_t perimeter) {
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < perimeter; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t t = k;
        int row, col;
        if (t < static_cast<size_t>(p.cols)) {
            row = 1; col = static_cast<int>(t) + 1;
        } else if (p.rows > 1 && (t -= p.cols) < static_cast<size_t>(p.cols)) {
            row = p.rows; col = static_cast<int>(t) + 1;
        } else {
            if (p.rows > 1) t -= p.cols;
            const int sides = p.cols > 1 ? 2 : 1;
            row = static_cast<int>(t / sides) + 2;
            col = (sides == 2 && t % 2) ? p.cols : 1;
        }
        const val_t energy = updateCell<false>(input, output, accumulated, p, row, col);
        if (row == 1) send[col - 1] = energy;
        if (row == p.rows) send[static_cast<size_t>(p.cols) + col - 1] = energy;
        if (col == 1) send[2 * static_cast<size_t>(p.cols) + row - 1] = energy;
        if (col == p.cols)
            send[2 * static_cast<size_t>(p.cols) + p.rows + row - 1] = energy;
    }
}

__global__ void unpackHalos(const val_t* __restrict__ receive,
                            val_t* __restrict__ energy, Partition p,
                            int top, int bottom, int left, int right) {
    const size_t count = 2 * (static_cast<size_t>(p.rows) + p.cols);
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (k < static_cast<size_t>(p.cols)) {
            if (top != MPI_PROC_NULL) energy[k + 1] = receive[k];
        } else if (k < 2 * static_cast<size_t>(p.cols)) {
            if (bottom != MPI_PROC_NULL)
                energy[static_cast<size_t>(p.rows + 1) * p.pitch + k - p.cols + 1] =
                    receive[k];
        } else if (k < 2 * static_cast<size_t>(p.cols) + p.rows) {
            if (left != MPI_PROC_NULL)
                energy[(k - 2 * static_cast<size_t>(p.cols) + 1) * p.pitch] =
                    receive[k];
        } else if (right != MPI_PROC_NULL) {
            energy[(k - 2 * static_cast<size_t>(p.cols) - p.rows + 1) * p.pitch +
                   p.cols + 1] = receive[k];
        }
    }
}

struct Statistics {
    val_t energy_sum, flux_sum, energy_min, energy_max;
    idx_t hash;
    int invalid;
};

__device__ Statistics warpReduce(Statistics a) {
    for (int delta = 16; delta; delta /= 2) {
        a.energy_sum += __shfl_down_sync(0xffffffff, a.energy_sum, delta);
        a.flux_sum += __shfl_down_sync(0xffffffff, a.flux_sum, delta);
        a.energy_min = fmin(a.energy_min,
                            __shfl_down_sync(0xffffffff, a.energy_min, delta));
        a.energy_max = fmax(a.energy_max,
                            __shfl_down_sync(0xffffffff, a.energy_max, delta));
        a.hash ^= static_cast<idx_t>(__shfl_down_sync(
            0xffffffff, static_cast<unsigned long long>(a.hash), delta));
        a.invalid |= __shfl_down_sync(0xffffffff, a.invalid, delta);
    }
    return a;
}

// Reduce on the GPU before copying to the CPU. Normal runs transfer only a
// small set of block statistics, never the entire distributed mesh.
__global__ void reduceResults(const val_t* __restrict__ energy,
                              const val_t* __restrict__ flux,
                              Statistics* __restrict__ results, Partition p) {
    Statistics a{0.0, 0.0, DBL_MAX, -DBL_MAX, 0, 0};
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < p.elements; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t row = k / p.cols, col = k % p.cols;
        const val_t e = energy[(row + 1) * p.pitch + col + 1], f = flux[k];
        const idx_t global = (static_cast<idx_t>(p.row_start) + row) * p.n +
                             p.col_start + col;
        a.energy_sum += e;
        a.flux_sum += f;
        a.energy_min = fmin(a.energy_min, e);
        a.energy_max = fmax(a.energy_max, e);
        a.hash ^= (static_cast<idx_t>(__double_as_longlong(e)) + global) *
                      0x9e3779b97f4a7c15ULL;
        a.hash ^= (static_cast<idx_t>(__double_as_longlong(f)) + global) *
                      0xbf58476d1ce4e5b9ULL;
        a.invalid |= !isfinite(e) || !isfinite(f);
    }
    a = warpReduce(a);
    __shared__ Statistics warps[8];
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warps[warp] = a;
    __syncthreads();
    if (warp == 0) {
        a = lane < 8 ? warps[lane] :
                      Statistics{0.0, 0.0, DBL_MAX, -DBL_MAX, 0, 0};
        a = warpReduce(a);
        if (lane == 0) results[blockIdx.x] = a;
    }
}

static unsigned int linearBlocks(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
}

static dim3 cellBlocks(const Partition& p, bool interior) {
    const int rows = p.rows - (interior ? 2 : 0);
    const int cols = p.cols - (interior ? 2 : 0);
    return dim3(static_cast<unsigned int>((static_cast<size_t>(cols) + 31) / 32),
                static_cast<unsigned int>(std::min<size_t>(
                    (static_cast<size_t>(rows) + 7) / 8, 65535)));
}

static bool cudaAwareMPI(MPI_Comm comm) {
    int supported = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    supported = MPIX_Query_cuda_support() != 0;
#endif
    int all_supported = 0;
    MPI_CHECK(MPI_Allreduce(&supported, &all_supported, 1, MPI_INT, MPI_MIN, comm));
    return all_supported != 0;
}

struct World {
    Partition p;
    MPI_Comm comm;
    int neighbors[4]{};
    bool direct;
    size_t energy_count, halo_count;
    val_t* energy[2]{};
    val_t* accumulated = nullptr;
    val_t* device_send = nullptr;
    val_t* device_receive = nullptr;
    val_t* host_send = nullptr;
    val_t* host_receive = nullptr;
    Statistics* device_statistics = nullptr;
    Statistics* host_statistics = nullptr;
    int statistics_blocks, current = 0;
    MPI_Request requests[8]{};
    int receive_count = 0;
    cudaStream_t compute = nullptr, exchange = nullptr;
    cudaEvent_t send_ready = nullptr, interior_ready = nullptr, step_ready = nullptr;
    cudaGraph_t pair_graph = nullptr;
    cudaGraphExec_t pair_execution = nullptr;

    World(Partition partition, MPI_Comm cart, bool device_mpi, int iterations)
        : p(partition), comm(cart), direct(device_mpi) {
        MPI_CHECK(MPI_Cart_shift(comm, 0, 1, &neighbors[TOP], &neighbors[BOTTOM]));
        MPI_CHECK(MPI_Cart_shift(comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]));
        if (p.pitch > std::numeric_limits<size_t>::max() /
                          (static_cast<size_t>(p.rows) + 2) / sizeof(val_t) ||
            p.elements > std::numeric_limits<size_t>::max() / sizeof(val_t))
            throw std::runtime_error("Grid allocation size overflows size_t");
        energy_count = p.pitch * (static_cast<size_t>(p.rows) + 2);
        halo_count = 2 * (static_cast<size_t>(p.rows) + p.cols);
        cudaDeviceProp properties{};
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        statistics_blocks = static_cast<int>(std::min<size_t>(
            (p.elements + 255) / 256, static_cast<size_t>(properties.multiProcessorCount) * 8));

        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&send_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&interior_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&step_ready, cudaEventDisableTiming));
        for (auto& buffer : energy) {
            CUDA_CHECK(cudaMalloc(&buffer, energy_count * sizeof(val_t)));
            CUDA_CHECK(cudaMemsetAsync(buffer, 0, energy_count * sizeof(val_t), compute));
        }
        CUDA_CHECK(cudaMalloc(&accumulated, p.elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemsetAsync(accumulated, 0, p.elements * sizeof(val_t), compute));
        CUDA_CHECK(cudaMalloc(&device_send, halo_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&device_receive, halo_count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&device_statistics, statistics_blocks * sizeof(Statistics)));
        CUDA_CHECK(cudaMallocHost(&host_statistics, statistics_blocks * sizeof(Statistics)));
        if (!direct) {
            CUDA_CHECK(cudaMallocHost(&host_send, halo_count * sizeof(val_t)));
            CUDA_CHECK(cudaMallocHost(&host_receive, halo_count * sizeof(val_t)));
            // Parallel first touch of communication buffers on the host.
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < halo_count; ++k) {
                host_send[k] = 0.0;
                host_receive[k] = 0.0;
            }
        }

        const size_t offsets[4] = {0, static_cast<size_t>(p.cols),
                                   2 * static_cast<size_t>(p.cols),
                                   2 * static_cast<size_t>(p.cols) + p.rows};
        const int counts[4] = {p.cols, p.cols, p.rows, p.rows};
        val_t* send = direct ? device_send : host_send;
        val_t* receive = direct ? device_receive : host_receive;
        for (int face = 0; face < 4; ++face) {
            if (neighbors[face] == MPI_PROC_NULL) continue;
            MPI_CHECK(MPI_Recv_init(receive + offsets[face], counts[face], MPI_DOUBLE,
                                    neighbors[face], opposite[face], comm,
                                    &requests[receive_count++]));
        }
        int send_count = 0;
        for (int face = 0; face < 4; ++face) {
            if (neighbors[face] == MPI_PROC_NULL) continue;
            MPI_CHECK(MPI_Send_init(send + offsets[face], counts[face], MPI_DOUBLE,
                                    neighbors[face], face, comm,
                                    &requests[receive_count + send_count++]));
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        // Force module loading before the benchmark timer.
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateCells<false>));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateCells<true>));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateBoundary));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, unpackHalos));
        if (receive_count == 0 && iterations >= 2) {
            CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            updateCells<false><<<cellBlocks(p, false), dim3(32, 8), 0, compute>>>(
                energy[0], energy[1], accumulated, p);
            updateCells<false><<<cellBlocks(p, false), dim3(32, 8), 0, compute>>>(
                energy[1], energy[0], accumulated, p);
            CUDA_CHECK(cudaStreamEndCapture(compute, &pair_graph));
            CUDA_CHECK(cudaGraphInstantiate(&pair_execution, pair_graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(pair_execution, compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
        }
    }

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        for (int k = 0; k < 2 * receive_count; ++k) MPI_Request_free(&requests[k]);
        if (pair_execution) cudaGraphExecDestroy(pair_execution);
        if (pair_graph) cudaGraphDestroy(pair_graph);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(exchange);
        cudaEventDestroy(send_ready);
        cudaEventDestroy(interior_ready);
        cudaEventDestroy(step_ready);
        for (auto buffer : energy) cudaFree(buffer);
        cudaFree(accumulated);
        cudaFree(device_send);
        cudaFree(device_receive);
        cudaFree(device_statistics);
        cudaFreeHost(host_statistics);
        if (host_send) cudaFreeHost(host_send);
        if (host_receive) cudaFreeHost(host_receive);
    }

    void runSimulation(int iterations) {
        if (receive_count == 0) {
            int iter = 0;
            for (; iter + 1 < iterations; iter += 2)
                CUDA_CHECK(cudaGraphLaunch(pair_execution, compute));
            if (iter < iterations) {
                updateCells<false><<<cellBlocks(p, false), dim3(32, 8), 0, compute>>>(
                    energy[0], energy[1], accumulated, p);
                CUDA_CHECK(cudaGetLastError());
            }
            current = iterations % 2;
        } else {
            for (int iter = 0; iter < iterations; ++iter) {
                const int next = 1 - current;
                // Initially every halo is zero, so the first exchange can be
                // skipped. Exchange the newly computed boundaries for the next
                // iteration, and omit communication after the final update.
                const bool communicate = iter + 1 < iterations;
                updateBoundary<<<linearBlocks(perimeterSize(p)), 256, 0, exchange>>>(
                    energy[current], energy[next], accumulated, device_send, p,
                    perimeterSize(p));
                CUDA_CHECK(cudaGetLastError());
                if (p.rows > 2 && p.cols > 2) {
                    updateCells<true><<<cellBlocks(p, true), dim3(32, 8), 0, compute>>>(
                        energy[current], energy[next], accumulated, p);
                    CUDA_CHECK(cudaGetLastError());
                }
                CUDA_CHECK(cudaEventRecord(interior_ready, compute));
                if (communicate) {
                    if (!direct)
                        CUDA_CHECK(cudaMemcpyAsync(host_send, device_send,
                            halo_count * sizeof(val_t), cudaMemcpyDeviceToHost, exchange));
                    CUDA_CHECK(cudaEventRecord(send_ready, exchange));
                    // MPI is called only by the main thread. While it progresses
                    // communication, the GPU independently updates the interior.
                    CUDA_CHECK(cudaEventSynchronize(send_ready));
                    // This event also guarantees the previous unpack/staging
                    // has finished before MPI reuses the receive buffers.
                    MPI_CHECK(MPI_Startall(receive_count, requests));
                    MPI_CHECK(MPI_Startall(receive_count, requests + receive_count));
                    MPI_CHECK(MPI_Waitall(2 * receive_count, requests, MPI_STATUSES_IGNORE));
                    if (!direct)
                        CUDA_CHECK(cudaMemcpyAsync(device_receive, host_receive,
                            halo_count * sizeof(val_t), cudaMemcpyHostToDevice, exchange));
                    unpackHalos<<<linearBlocks(halo_count), 256, 0, exchange>>>(
                        device_receive, energy[next], p, neighbors[TOP], neighbors[BOTTOM],
                        neighbors[LEFT], neighbors[RIGHT]);
                    CUDA_CHECK(cudaGetLastError());
                }
                // Both streams must finish this generation before either stream
                // reads or overwrites the double-buffered state of the next one.
                CUDA_CHECK(cudaStreamWaitEvent(exchange, interior_ready, 0));
                CUDA_CHECK(cudaEventRecord(step_ready, exchange));
                CUDA_CHECK(cudaStreamWaitEvent(compute, step_ready, 0));
                current = next;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(exchange));
    }

    Statistics statistics() {
        reduceResults<<<statistics_blocks, 256, 0, compute>>>(
            energy[current], accumulated, device_statistics, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host_statistics, device_statistics,
            statistics_blocks * sizeof(Statistics), cudaMemcpyDeviceToHost, compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        val_t energy_sum = 0.0, flux_sum = 0.0, energy_min = DBL_MAX, energy_max = -DBL_MAX;
        idx_t result_hash = 0;
        int invalid = 0;
        #pragma omp parallel for schedule(static) \
            reduction(+:energy_sum,flux_sum) reduction(min:energy_min) \
            reduction(max:energy_max) reduction(^:result_hash) reduction(|:invalid)
        for (int k = 0; k < statistics_blocks; ++k) {
            const auto& s = host_statistics[k];
            energy_sum += s.energy_sum;
            flux_sum += s.flux_sum;
            energy_min = std::min(energy_min, s.energy_min);
            energy_max = std::max(energy_max, s.energy_max);
            result_hash ^= s.hash;
            invalid |= s.invalid;
        }
        Statistics global{};
        const val_t sums[2] = {energy_sum, flux_sum};
        val_t global_sums[2]{};
        MPI_CHECK(MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm));
        global.energy_sum = global_sums[0];
        global.flux_sum = global_sums[1];
        MPI_CHECK(MPI_Reduce(&energy_min, &global.energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm));
        MPI_CHECK(MPI_Reduce(&energy_max, &global.energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        MPI_CHECK(MPI_Reduce(&result_hash, &global.hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm));
        MPI_CHECK(MPI_Reduce(&invalid, &global.invalid, 1, MPI_INT, MPI_LOR, 0, comm));
        return global;
    }

    void printResults(int rank, int ranks, const int dims[2]) {
        val_t* local = nullptr;
        CUDA_CHECK(cudaMallocHost(&local, p.elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy2D(local, static_cast<size_t>(p.cols) * sizeof(val_t),
            energy[current] + p.pitch + 1, p.pitch * sizeof(val_t),
            static_cast<size_t>(p.cols) * sizeof(val_t), p.rows, cudaMemcpyDeviceToHost));
        if (rank == 0) {
            // Only -r gathers the full field. Subarray receive types put each
            // rectangular tile directly in the original global row-major order.
            std::vector<val_t> all(static_cast<size_t>(p.n) * p.n);
            std::vector<MPI_Request> receives(ranks - 1);
            for (int peer = 1; peer < ranks; ++peer) {
                int coords[2];
                MPI_CHECK(MPI_Cart_coords(comm, peer, 2, coords));
                const Partition tile = partitionFor(p.n, dims, coords);
                const int sizes[2] = {p.n, p.n};
                const int subsizes[2] = {tile.rows, tile.cols};
                const int starts[2] = {tile.row_start, tile.col_start};
                MPI_Datatype type;
                MPI_CHECK(MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                                    MPI_ORDER_C, MPI_DOUBLE, &type));
                MPI_CHECK(MPI_Type_commit(&type));
                MPI_CHECK(MPI_Irecv(all.data(), 1, type, peer, 4, comm, &receives[peer - 1]));
                MPI_CHECK(MPI_Type_free(&type));
            }
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < p.rows; ++row)
                std::memcpy(all.data() + static_cast<size_t>(p.row_start + row) * p.n +
                                p.col_start,
                            local + static_cast<size_t>(row) * p.cols,
                            static_cast<size_t>(p.cols) * sizeof(val_t));
            if (!receives.empty())
                MPI_CHECK(MPI_Waitall(ranks - 1, receives.data(), MPI_STATUSES_IGNORE));
            print_results(all, "ElementEnergy");
        } else {
            // A datatype keeps the MPI message count representable even when
            // a tile contains more than INT_MAX doubles.
            MPI_Datatype type;
            MPI_CHECK(MPI_Type_vector(p.rows, p.cols, p.cols, MPI_DOUBLE, &type));
            MPI_CHECK(MPI_Type_commit(&type));
            MPI_CHECK(MPI_Send(local, 1, type, 0, 4, comm));
            MPI_CHECK(MPI_Type_free(&type));
        }
        CUDA_CHECK(cudaFreeHost(local));
    }
};

static bool validateResults(const Statistics& s) {
    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", s.energy_sum);
    std::printf("  Flux sum: %.2f\n", s.flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", s.energy_min, s.energy_max);
    if (s.invalid || !std::isfinite(s.energy_sum) || !std::isfinite(s.flux_sum) ||
        !std::isfinite(s.energy_min) || !std::isfinite(s.energy_max)) {
        std::printf("  ERROR: Simulation results are not finite\n");
        return false;
    }
    if (std::abs(s.energy_sum) > 1e-8)
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static int parseNumber(const char* text, int minimum) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < minimum ||
        value > std::numeric_limits<int>::max() - 2)
        throw std::runtime_error("Invalid grid size or iteration count");
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    int provided = 0;
    const int initialization = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initialization != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Parsing on rank zero gives a single usage/error message and identical
    // parameters on every rank. Code 1 is help, code 2 is invalid arguments.
    int options[5] = {512, 10, 0, 0, 0};
    if (rank == 0) {
        try {
            for (int k = 1; k < argc; ++k) {
                if (std::strcmp(argv[k], "-n") == 0 && k + 1 < argc)
                    options[0] = parseNumber(argv[++k], 1);
                else if (std::strcmp(argv[k], "-i") == 0 && k + 1 < argc)
                    options[1] = parseNumber(argv[++k], 0);
                else if (std::strcmp(argv[k], "-v") == 0) options[2] = 1;
                else if (std::strcmp(argv[k], "-r") == 0) options[3] = 1;
                else if (std::strcmp(argv[k], "-h") == 0) { options[4] = 1; break; }
                else throw std::runtime_error("Unknown or incomplete command line option");
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            options[4] = 2;
        }
        if (options[4]) printUsage(argv[0]);
    }
    MPI_CHECK(MPI_Bcast(options, 5, MPI_INT, 0, MPI_COMM_WORLD));
    if (options[4]) {
        MPI_CHECK(MPI_Finalize());
        return options[4] == 1 ? 0 : 1;
    }

    int exit_code = 0;
    try {
        const int n = options[0], iterations = options[1];
        // Avoid empty partitions when the process count exceeds a grid axis.
        // Idle ranks still participate in MPI initialization/finalization.
        int active = static_cast<int>(std::min<idx_t>(ranks, static_cast<idx_t>(n) * n));
        int dims[2] = {0, 0};
        for (;;) {
            dims[0] = dims[1] = 0;
            MPI_CHECK(MPI_Dims_create(active, 2, dims));
            if (dims[0] <= n && dims[1] <= n) break;
            --active;
        }
        MPI_Comm work = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED,
                                 rank, &work));
        if (work != MPI_COMM_NULL) {
            MPI_Comm node = MPI_COMM_NULL;
            MPI_CHECK(MPI_Comm_split_type(work, MPI_COMM_TYPE_SHARED, rank,
                                          MPI_INFO_NULL, &node));
            int local_rank = 0, devices = 0;
            MPI_CHECK(MPI_Comm_rank(node, &local_rank));
            CUDA_CHECK(cudaGetDeviceCount(&devices));
            if (devices == 0) throw std::runtime_error("A CUDA device is required on every active rank");
            CUDA_CHECK(cudaSetDevice(local_rank % devices));
            MPI_CHECK(MPI_Comm_free(&node));
            const int periods[2] = {0, 0};
            MPI_Comm cart = MPI_COMM_NULL;
            MPI_CHECK(MPI_Cart_create(work, 2, dims, periods, 0, &cart));
            MPI_CHECK(MPI_Comm_free(&work));
            int coords[2];
            MPI_CHECK(MPI_Cart_coords(cart, rank, 2, coords));
            const Partition p = partitionFor(n, dims, coords);
            if (!std::getenv("OMP_NUM_THREADS")) {
                const int useful = static_cast<int>(std::min<size_t>(
                    8, std::max<size_t>(1, p.elements / 32768)));
                omp_set_num_threads(std::min(omp_get_max_threads(), useful));
            }
            const bool direct = cudaAwareMPI(cart);
            {
                World world(p, cart, direct, iterations);
                const idx_t local_bytes =
                    (2 * world.energy_count + p.elements + 2 * world.halo_count) *
                        sizeof(val_t) + world.statistics_blocks * sizeof(Statistics);
                idx_t total_bytes = 0;
                MPI_CHECK(MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_UINT64_T,
                                     MPI_SUM, 0, cart));
                if (rank == 0) {
                    std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
                    std::printf("============================================\n");
                    std::printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n,
                                static_cast<idx_t>(n) * n);
                    std::printf("Iterations: %d\n", iterations);
                    std::printf("Validation: %s\n", options[2] ? "enabled" : "disabled");
                    std::printf("MPI ranks: %d of %d (%d x %d), OpenMP threads/rank: %d\n",
                                active, ranks, dims[0], dims[1], omp_get_max_threads());
                    std::printf("CUDA halo exchange: %s\n\n",
                                direct ? "device buffers" : "pinned host staging");
                    std::printf("Building unstructured mesh (implicit connectivity)...\n");
                    std::printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n",
                                total_bytes / (1024.0 * 1024.0),
                                total_bytes / (1024.0 * 1024.0));
                    std::printf("Running simulation...\n");
                }
                MPI_CHECK(MPI_Barrier(cart));
                const double start = MPI_Wtime();
                world.runSimulation(iterations);
                const double elapsed = MPI_Wtime() - start;
                double seconds = 0.0;
                MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart));
                const Statistics results = world.statistics();
                if (rank == 0) {
                    const double throughput = iterations && seconds > 0.0
                        ? (static_cast<double>(n) * n * iterations) / seconds / 1e9 : 0.0;
                    std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
                    std::printf("Performance:\n");
                    std::printf("  Time per iteration: %.4f ms\n",
                                iterations ? seconds * 1000.0 / iterations : 0.0);
                    std::printf("  Elements/sec: %.4f GigaElements/s\n", throughput);
                    std::printf("  Performance: %.4f GFLOPS\n", throughput * 22.0);
                    std::printf("  Result hash: %016" PRIX64 "\n\n", results.hash);
                }
                if (options[3]) world.printResults(rank, active, dims);
                if (rank == 0 && options[2] && !validateResults(results)) exit_code = 1;
                MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, cart));
            }
            MPI_CHECK(MPI_Comm_free(&cart));
        }
        MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_CHECK(MPI_Finalize());
    return exit_code;
}
