#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};
static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t));

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = 0;
    MPI_Error_string(error, message, &length);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}
#define MPI_CHECK(call) mpiCheck((call), #call)

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// MPI owns a rectangular subset of the square mesh. Its topology is exactly
// the original mesh's connectivity, so no per-element neighbor records are
// needed. Only energies, not accumulated fluxes, require ghost cells.
struct World {
    int n = 0, rows = 0, cols = 0, row_begin = 0, col_begin = 0;
    size_t stride = 0, count = 0, padded_count = 0;
    int host_threads = 1;
    MPI_Comm comm = MPI_COMM_NULL;
    int neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t face_offset[4] = {};
    int face_count[4] = {};
    MPI_Request requests[8] = {};
    int request_count = 0;
    bool device_mpi = false;
    val_t *energy = nullptr, *energy_next = nullptr, *flux = nullptr;
    uint8_t* material = nullptr;
    val_t *send_device = nullptr, *recv_device = nullptr;
    val_t *send_host = nullptr, *recv_host = nullptr;
    uint64_t* hash_device = nullptr;
    cudaStream_t compute = nullptr, halo = nullptr;
    cudaEvent_t step_done = nullptr, halo_ready = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_steps = 0;

    World() = default;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        for (int i = 0; i < request_count; ++i) MPI_Request_free(&requests[i]);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (step_done) cudaEventDestroy(step_done);
        if (halo_ready) cudaEventDestroy(halo_ready);
        if (compute) cudaStreamDestroy(compute);
        if (halo) cudaStreamDestroy(halo);
        cudaFree(energy);
        cudaFree(energy_next);
        cudaFree(flux);
        cudaFree(material);
        cudaFree(send_device);
        cudaFree(recv_device);
        cudaFree(hash_device);
        if (send_host) cudaFreeHost(send_host);
        if (recv_host) cudaFreeHost(recv_host);
    }
};

static int partitionSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}
static int partitionBegin(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts);
}

// Choose the largest usable process grid, then minimize its halo perimeter.
// In particular, a prime process count must not create empty subdomains.
static int chooseProcessGrid(int processes, int n, int dims[2]) {
    const uint64_t cells = static_cast<uint64_t>(n) * n;
    int active = static_cast<int>(std::min<uint64_t>(processes, cells));
    for (; active > 0; --active) {
        uint64_t best = std::numeric_limits<uint64_t>::max();
        for (int d = 1; d <= active / d; ++d) {
            if (active % d) continue;
            const int other = active / d;
            const uint64_t perimeter = static_cast<uint64_t>(d) + other;
            if (d <= n && other <= n && perimeter < best) {
                dims[0] = d;
                dims[1] = other;
                best = perimeter;
            }
        }
        if (best != std::numeric_limits<uint64_t>::max()) return active;
    }
    return 0;
}

static void selectDevice(MPI_Comm participants) {
    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(participants, MPI_COMM_TYPE_SHARED, 0,
                                 MPI_INFO_NULL, &node));
    int local_rank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(node, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "The hybrid benchmark requires a CUDA device on every participating node.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Also works when the launcher exposes one GPU per rank via
    // CUDA_VISIBLE_DEVICES. Otherwise use one rank per visible GPU.
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&node));
}

static void buildSquare2D(World& w, int n, MPI_Comm comm) {
    w.n = n;
    w.comm = comm;
    int dims[2], periods[2], coords[2];
    MPI_CHECK(MPI_Cart_get(comm, 2, dims, periods, coords));
    w.rows = partitionSize(n, dims[0], coords[0]);
    w.cols = partitionSize(n, dims[1], coords[1]);
    w.row_begin = partitionBegin(n, dims[0], coords[0]);
    w.col_begin = partitionBegin(n, dims[1], coords[1]);
    w.stride = static_cast<size_t>(w.cols) + 2;
    w.count = static_cast<size_t>(w.rows) * w.cols;
    w.padded_count = (static_cast<size_t>(w.rows) + 2) * w.stride;
    if (w.padded_count > std::numeric_limits<size_t>::max() / sizeof(val_t)) {
        fprintf(stderr, "Subdomain allocation size overflows size_t.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    w.host_threads = std::min(omp_get_max_threads(),
        static_cast<int>(std::min<size_t>(32, std::max<size_t>(2, w.count / 32768))));
    MPI_CHECK(MPI_Cart_shift(comm, 0, 1, &w.neighbors[0], &w.neighbors[1]));
    MPI_CHECK(MPI_Cart_shift(comm, 1, 1, &w.neighbors[2], &w.neighbors[3]));
    w.face_count[0] = w.face_count[1] = w.cols;
    w.face_count[2] = w.face_count[3] = w.rows;
    for (int i = 1; i < 4; ++i)
        w.face_offset[i] = w.face_offset[i - 1] + w.face_count[i - 1];
    const size_t face_values = 2 * (static_cast<size_t>(w.rows) + w.cols);

    CUDA_CHECK(cudaStreamCreateWithFlags(&w.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&w.halo, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&w.step_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&w.halo_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&w.energy, w.padded_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&w.energy_next, w.padded_count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&w.flux, w.count * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&w.material, w.count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&w.hash_device, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemsetAsync(w.energy, 0, w.padded_count * sizeof(val_t), w.compute));
    CUDA_CHECK(cudaMemsetAsync(w.energy_next, 0, w.padded_count * sizeof(val_t), w.compute));
    CUDA_CHECK(cudaMemsetAsync(w.flux, 0, w.count * sizeof(val_t), w.compute));

    // OpenMP builds the compact material map. Successive assignments preserve
    // the original corner precedence even for a one-element mesh.
    std::vector<uint8_t> materials(w.count);
    #pragma omp parallel for schedule(static) num_threads(w.host_threads)
    for (int local_row = 0; local_row < w.rows; ++local_row) {
        const int row = w.row_begin + local_row;
        for (int local_col = 0; local_col < w.cols; ++local_col) {
            const int col = w.col_begin + local_col;
            uint8_t m = 0;
            if (row == 0 && col == 0) m = 1;
            if (row == 0 && col == n - 1) m = 2;
            if (row == n - 1 && col == 0) m = 2;
            if (row == n - 1 && col == n - 1) m = 1;
            materials[static_cast<size_t>(local_row) * w.cols + local_col] = m;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(w.material, materials.data(), w.count,
                               cudaMemcpyHostToDevice, w.compute));
    CUDA_CHECK(cudaEventRecord(w.step_done, w.compute));
    CUDA_CHECK(cudaStreamSynchronize(w.compute));

    #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    w.device_mpi = MPIX_Query_cuda_support() != 0;
    #endif
    // All communicating ranks must agree on the buffer kind.
    int aware = w.device_mpi ? 1 : 0, all_aware = 0;
    MPI_CHECK(MPI_Allreduce(&aware, &all_aware, 1, MPI_INT, MPI_MIN, comm));
    w.device_mpi = all_aware != 0;
    bool communicating = false;
    for (int neighbor : w.neighbors) communicating |= neighbor != MPI_PROC_NULL;
    if (!communicating) return;

    CUDA_CHECK(cudaMalloc(&w.send_device, face_values * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&w.recv_device, face_values * sizeof(val_t)));
    if (!w.device_mpi) {
        CUDA_CHECK(cudaMallocHost(&w.send_host, face_values * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&w.recv_host, face_values * sizeof(val_t)));
        std::memset(w.recv_host, 0, face_values * sizeof(val_t));
    }
    CUDA_CHECK(cudaMemset(w.recv_device, 0, face_values * sizeof(val_t)));
    val_t* send = w.device_mpi ? w.send_device : w.send_host;
    val_t* recv = w.device_mpi ? w.recv_device : w.recv_host;
    // Persistent requests amortize setup across iterations. Opposite faces
    // have equal lengths even with uneven rectangular partitions.
    for (int d = 0; d < 4; ++d) {
        if (w.neighbors[d] == MPI_PROC_NULL) continue;
        MPI_CHECK(MPI_Recv_init(recv + w.face_offset[d], w.face_count[d], MPI_DOUBLE,
            w.neighbors[d], 100 + (d ^ 1), comm, &w.requests[w.request_count++]));
        MPI_CHECK(MPI_Send_init(send + w.face_offset[d], w.face_count[d], MPI_DOUBLE,
            w.neighbors[d], 100 + d, comm, &w.requests[w.request_count++]));
    }
}

// The order is x+1, x-1, y+1, y-1, exactly as in the original connectivity
// loop. Do not reassociate additions or contract multiply/add into FMA.
// Accumulated flux has no neighbor dependencies and can be updated in place.
template<bool Interior>
__device__ __forceinline__ void updateCell(
    const val_t* __restrict__ energy, val_t* __restrict__ next,
    val_t* __restrict__ flux, const uint8_t* __restrict__ material,
    int row, int col, int rows, int cols, size_t stride,
    int row_begin, int col_begin, int n) {
    const size_t offset = static_cast<size_t>(row) * stride + col;
    const size_t local = static_cast<size_t>(row - 1) * cols + col - 1;
    const val_t center = energy[offset];
    val_t total = 0.0;
    if constexpr (!Interior) {
        const uint8_t m = material[local];
        total = m == 1 ? 0.5 : (m == 2 ? -0.5 : 0.0);
    }
    if (Interior || row_begin + row < n)
        total += (energy[offset + stride] - center) * 0.8 * 0.25;
    if (Interior || row_begin + row > 1)
        total += (energy[offset - stride] - center) * 0.8 * 0.25;
    if (Interior || col_begin + col < n)
        total += (energy[offset + 1] - center) * 0.8 * 0.25;
    if (Interior || col_begin + col > 1)
        total += (energy[offset - 1] - center) * 0.8 * 0.25;
    next[offset] = center + total;
    flux[local] += fabs(total);
    (void)rows;
}

template<bool Interior>
__global__ void updateKernel(const val_t* energy, val_t* next, val_t* flux,
    const uint8_t* material, int rows, int cols, size_t stride,
    int row_begin, int col_begin, int n) {
    constexpr int start = Interior ? 2 : 1;
    const int last_row = rows - (Interior ? 1 : 0);
    const int last_col = cols - (Interior ? 1 : 0);
    const int col = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (col > last_col) return;
    for (int row = blockIdx.y * blockDim.y + threadIdx.y + start;
         row <= last_row; row += gridDim.y * blockDim.y) {
        updateCell<Interior>(energy, next, flux, material, row, col, rows, cols,
                             stride, row_begin, col_begin, n);
    }
}

__global__ void updateBoundaryKernel(const val_t* energy, val_t* next, val_t* flux,
    const uint8_t* material, int rows, int cols, size_t stride,
    int row_begin, int col_begin, int n, size_t boundary_count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < boundary_count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int row, col;
        if (i < static_cast<size_t>(cols)) {
            row = 1; col = i + 1;
        } else if (rows > 1 && i < 2 * static_cast<size_t>(cols)) {
            row = rows; col = i - cols + 1;
        } else {
            const size_t edge = i - static_cast<size_t>(cols) * (rows > 1 ? 2 : 1);
            row = edge / (cols > 1 ? 2 : 1) + 2;
            col = cols > 1 && (edge & 1) ? cols : 1;
        }
        updateCell<false>(energy, next, flux, material, row, col, rows, cols,
                          stride, row_begin, col_begin, n);
    }
}

template<bool Pack>
__global__ void haloKernel(val_t* energy, val_t* buffer, int rows, int cols,
                           size_t stride, size_t face_values) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < face_values; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        size_t offset;
        if (i < static_cast<size_t>(cols)) {
            offset = (Pack ? stride : 0) + i + 1;
        } else if (i < 2 * static_cast<size_t>(cols)) {
            offset = static_cast<size_t>(Pack ? rows : rows + 1) * stride + i - cols + 1;
        } else if (i < 2 * static_cast<size_t>(cols) + rows) {
            offset = (i - 2 * static_cast<size_t>(cols) + 1) * stride + (Pack ? 1 : 0);
        } else {
            offset = (i - 2 * static_cast<size_t>(cols) - rows + 1) * stride +
                     (Pack ? cols : cols + 1);
        }
        if constexpr (Pack) buffer[i] = energy[offset];
        else energy[offset] = buffer[i];
    }
}

static unsigned linearBlocks(size_t count) {
    return static_cast<unsigned>(std::min<size_t>(65535, (count + 255) / 256));
}

template<bool Interior>
static void launchUpdate(World& w) {
    const int rows = w.rows - (Interior ? 2 : 0);
    const int cols = w.cols - (Interior ? 2 : 0);
    if (rows <= 0 || cols <= 0) return;
    const dim3 block(32, 8);
    const dim3 grid((static_cast<unsigned>(cols) + 31) / 32,
                   std::min(65535u, (static_cast<unsigned>(rows) + 7) / 8));
    updateKernel<Interior><<<grid, block, 0, w.compute>>>(w.energy, w.energy_next,
        w.flux, w.material, w.rows, w.cols, w.stride, w.row_begin, w.col_begin, w.n);
    CUDA_CHECK(cudaGetLastError());
}

static void prepareGraph(World& w, int n_iters) {
    if (w.request_count || n_iters == 0) return;
    // A bounded even batch allows repeated graph launches without changing
    // captured buffer addresses. Short runs use a single exact-sized graph.
    w.graph_steps = std::min(n_iters, 64);
    CUDA_CHECK(cudaStreamBeginCapture(w.compute, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < w.graph_steps; ++i) {
        launchUpdate<false>(w);
        std::swap(w.energy, w.energy_next);
    }
    if (w.graph_steps & 1) std::swap(w.energy, w.energy_next);
    CUDA_CHECK(cudaStreamEndCapture(w.compute, &w.graph));
    CUDA_CHECK(cudaGraphInstantiate(&w.graph_exec, w.graph, nullptr, nullptr, 0));
}

static void runSimulation(World& w, int n_iters) {
    if (w.graph_exec) {
        const int batches = n_iters / w.graph_steps;
        for (int i = 0; i < batches; ++i)
            CUDA_CHECK(cudaGraphLaunch(w.graph_exec, w.compute));
        if (w.graph_steps & 1) std::swap(w.energy, w.energy_next);
        for (int i = batches * w.graph_steps; i < n_iters; ++i) {
            launchUpdate<false>(w);
            std::swap(w.energy, w.energy_next);
        }
    } else {
        const size_t face_values = 2 * (static_cast<size_t>(w.rows) + w.cols);
        const size_t boundary_count = static_cast<size_t>(w.cols) * (w.rows > 1 ? 2 : 1) +
            static_cast<size_t>(std::max(w.rows - 2, 0)) * (w.cols > 1 ? 2 : 1);
        for (int i = 0; i < n_iters; ++i) {
            // All initial energies and ghosts are zero, so the first step
            // needs no exchange. Each later step communicates previous state.
            if (i == 0 || !w.request_count) {
                launchUpdate<false>(w);
            } else {
                CUDA_CHECK(cudaStreamWaitEvent(w.halo, w.step_done, 0));
                haloKernel<true><<<linearBlocks(face_values), 256, 0, w.halo>>>(
                    w.energy, w.send_device, w.rows, w.cols, w.stride, face_values);
                CUDA_CHECK(cudaGetLastError());
                if (!w.device_mpi)
                    CUDA_CHECK(cudaMemcpyAsync(w.send_host, w.send_device,
                        face_values * sizeof(val_t), cudaMemcpyDeviceToHost, w.halo));
                launchUpdate<true>(w);
                // Synchronizing the pack stream also ensures the previous
                // unpack has released the persistent MPI receive buffers.
                CUDA_CHECK(cudaStreamSynchronize(w.halo));
                MPI_CHECK(MPI_Startall(w.request_count, w.requests));
                MPI_CHECK(MPI_Waitall(w.request_count, w.requests, MPI_STATUSES_IGNORE));
                if (!w.device_mpi)
                    CUDA_CHECK(cudaMemcpyAsync(w.recv_device, w.recv_host,
                        face_values * sizeof(val_t), cudaMemcpyHostToDevice, w.halo));
                haloKernel<false><<<linearBlocks(face_values), 256, 0, w.halo>>>(
                    w.energy, w.recv_device, w.rows, w.cols, w.stride, face_values);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(w.halo_ready, w.halo));
                CUDA_CHECK(cudaStreamWaitEvent(w.compute, w.halo_ready, 0));
                updateBoundaryKernel<<<linearBlocks(boundary_count), 256, 0, w.compute>>>(
                    w.energy, w.energy_next, w.flux, w.material, w.rows, w.cols,
                    w.stride, w.row_begin, w.col_begin, w.n, boundary_count);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(w.step_done, w.compute));
            std::swap(w.energy, w.energy_next);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(w.compute));
}

// XOR is independent of reduction order. Global element indices reproduce the
// original checksum exactly without gathering the entire mesh onto rank zero.
__global__ void hashKernel(const val_t* energy, const val_t* flux, size_t count,
    int cols, size_t stride, int row_begin, int col_begin, int n, uint64_t* result) {
    uint64_t value = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = i / cols, col = i % cols;
        const uint64_t global = (row_begin + row) * static_cast<uint64_t>(n) + col_begin + col;
        const uint64_t e = static_cast<uint64_t>(__double_as_longlong(energy[(row + 1) * stride + col + 1]));
        const uint64_t f = static_cast<uint64_t>(__double_as_longlong(flux[i]));
        value ^= (e + global) * 0x9e3779b97f4a7c15ULL;
        value ^= (f + global) * 0xbf58476d1ce4e5b9ULL;
    }
    __shared__ uint64_t values[256];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(offset))
            values[threadIdx.x] ^= values[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicXor(reinterpret_cast<unsigned long long*>(result),
                  static_cast<unsigned long long>(values[0]));
}

static uint64_t computeHash(World& w) {
    CUDA_CHECK(cudaMemsetAsync(w.hash_device, 0, sizeof(uint64_t), w.compute));
    hashKernel<<<std::min(1024u, linearBlocks(w.count)), 256, 0, w.compute>>>(w.energy,
        w.flux, w.count, w.cols, w.stride, w.row_begin, w.col_begin, w.n, w.hash_device);
    CUDA_CHECK(cudaGetLastError());
    uint64_t local = 0, global = 0;
    CUDA_CHECK(cudaMemcpyAsync(&local, w.hash_device, sizeof(local),
                               cudaMemcpyDeviceToHost, w.compute));
    CUDA_CHECK(cudaStreamSynchronize(w.compute));
    MPI_CHECK(MPI_Reduce(&local, &global, 1, MPI_UINT64_T, MPI_BXOR, 0, w.comm));
    return global;
}

// Full, ordered results are gathered only for the explicitly requested -v/-r
// output. Derived receive types place 2D blocks directly into global row order;
// row batches also avoid MPI's int count limit for very large meshes.
static std::vector<ElementDynamic> gatherResults(World& w) {
    val_t* packed = nullptr;
    CUDA_CHECK(cudaMallocHost(&packed, 2 * w.count * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy2DAsync(packed, w.cols * sizeof(val_t), w.energy + w.stride + 1,
        w.stride * sizeof(val_t), w.cols * sizeof(val_t), w.rows,
        cudaMemcpyDeviceToHost, w.compute));
    CUDA_CHECK(cudaMemcpyAsync(packed + w.count, w.flux, w.count * sizeof(val_t),
                               cudaMemcpyDeviceToHost, w.compute));
    CUDA_CHECK(cudaStreamSynchronize(w.compute));
    std::vector<ElementDynamic> local(w.count);
    #pragma omp parallel for schedule(static) num_threads(w.host_threads)
    for (size_t i = 0; i < w.count; ++i)
        local[i] = {packed[i], packed[w.count + i]};
    CUDA_CHECK(cudaFreeHost(packed));

    int rank = 0, size = 0, dims[2], periods[2], coords[2];
    MPI_CHECK(MPI_Comm_rank(w.comm, &rank));
    MPI_CHECK(MPI_Comm_size(w.comm, &size));
    MPI_CHECK(MPI_Cart_get(w.comm, 2, dims, periods, coords));
    MPI_Datatype element_type;
    MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &element_type));
    MPI_CHECK(MPI_Type_commit(&element_type));
    std::vector<ElementDynamic> global;
    if (rank == 0) {
        global.resize(static_cast<size_t>(w.n) * w.n);
        #pragma omp parallel for schedule(static) num_threads(w.host_threads)
        for (int row = 0; row < w.rows; ++row)
            std::memcpy(global.data() + static_cast<size_t>(w.row_begin + row) * w.n + w.col_begin,
                        local.data() + static_cast<size_t>(row) * w.cols,
                        w.cols * sizeof(ElementDynamic));
        for (int peer = 1; peer < size; ++peer) {
            MPI_CHECK(MPI_Cart_coords(w.comm, peer, 2, coords));
            const int rows = partitionSize(w.n, dims[0], coords[0]);
            const int cols = partitionSize(w.n, dims[1], coords[1]);
            const int row_begin = partitionBegin(w.n, dims[0], coords[0]);
            const int col_begin = partitionBegin(w.n, dims[1], coords[1]);
            for (int first = 0; first < rows;) {
                const int batch = std::min(rows - first, INT_MAX / cols);
                MPI_Datatype rectangle;
                MPI_CHECK(MPI_Type_create_hvector(batch, cols,
                    static_cast<MPI_Aint>(w.n) * sizeof(ElementDynamic), element_type, &rectangle));
                MPI_CHECK(MPI_Type_commit(&rectangle));
                MPI_CHECK(MPI_Recv(global.data() + static_cast<size_t>(row_begin + first) * w.n + col_begin,
                                   1, rectangle, peer, 200, w.comm, MPI_STATUS_IGNORE));
                MPI_CHECK(MPI_Type_free(&rectangle));
                first += batch;
            }
        }
    } else {
        for (int first = 0; first < w.rows;) {
            const int batch = std::min(w.rows - first, INT_MAX / w.cols);
            MPI_CHECK(MPI_Send(local.data() + static_cast<size_t>(first) * w.cols,
                               batch * w.cols, element_type, 0, 200, w.comm));
            first += batch;
        }
    }
    MPI_CHECK(MPI_Type_free(&element_type));
    return global;
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    
    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;
    
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int minimum, int maximum, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < minimum || parsed > maximum) return false;
    value = static_cast<int>(parsed);
    return true;
}

static int benchmark(int n, int n_iters, bool validate, bool printResults,
                     MPI_Comm comm, int processes, int active, const int dims[2]) {
    int rank = 0;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    World world;
    buildSquare2D(world, n, comm);
    prepareGraph(world, n_iters);
    const uint64_t n_elems = static_cast<uint64_t>(n) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI (%d/%d ranks, %d x %d), OpenMP, CUDA\n",
               active, processes, dims[0], dims[1]);
        printf("Halo transport: %s\n\n", world.device_mpi ? "CUDA-aware MPI" : "pinned host buffers");
        printf("Building unstructured mesh...\n");
    }
    const double static_mem = world.count * sizeof(uint8_t);
    const double dynamic_mem = (2 * world.padded_count + world.count) * sizeof(val_t);
    double local_memory[2] = {static_mem, dynamic_mem}, total_memory[2] = {};
    MPI_CHECK(MPI_Reduce(local_memory, total_memory, 2, MPI_DOUBLE, MPI_SUM, 0, comm));
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
            (total_memory[0] + total_memory[1]) / (1024.0 * 1024.0),
            total_memory[0] / (1024.0 * 1024.0), total_memory[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    const uint64_t result_hash = computeHash(world);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        const double time_per_iter = n_iters ? duration_ms / n_iters : 0.0;
        const double rate = duration > 0.0 ? static_cast<double>(n_iters) * n_elems / duration / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }
    int result = 0;
    if (validate || printResults) {
        const auto elements = gatherResults(world);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData(elements.size());
                #pragma omp parallel for schedule(static) num_threads(world.host_threads)
                for (size_t i = 0; i < elements.size(); ++i)
                    energyData[i] = elements[i].current_energy;
                print_results(energyData, "ElementEnergy");
            }
            if (validate && !validateResults(elements)) result = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, comm));
    return result;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, processes = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &processes));
    // Parse once and broadcast, including help/errors, so all ranks follow
    // the same collectives and only rank zero prints benchmark output.
    int options[5] = {512, 10, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "-i") == 0) {
                const bool grid = std::strcmp(argv[i], "-n") == 0;
                if (i + 1 >= argc || !parseInteger(argv[++i], grid ? 1 : 0,
                                                  grid ? INT_MAX - 2 : INT_MAX,
                                                  options[grid ? 0 : 1])) {
                    fprintf(stderr, "Invalid %s value.\n", grid ? "grid size" : "iteration count");
                    options[4] = 1;
                    break;
                }
            } else if (std::strcmp(argv[i], "-v") == 0) options[2] = 1;
            else if (std::strcmp(argv[i], "-r") == 0) options[3] = 1;
            else if (std::strcmp(argv[i], "-h") == 0) { options[4] = 2; break; }
            else { printf("Unknown option: %s\n", argv[i]); options[4] = 1; break; }
        }
        if (options[4]) printUsage(argv[0]);
    }
    MPI_CHECK(MPI_Bcast(options, 5, MPI_INT, 0, MPI_COMM_WORLD));
    if (options[4]) {
        MPI_Finalize();
        return options[4] == 2 ? 0 : 1;
    }

    int result = 0;
    try {
        int dims[2] = {};
        const int active = chooseProcessGrid(processes, options[0], dims);
        MPI_Comm participants;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED,
                                 rank, &participants));
        if (participants != MPI_COMM_NULL) {
            selectDevice(participants);
            MPI_Comm cart;
            int periods[2] = {0, 0};
            MPI_CHECK(MPI_Cart_create(participants, 2, dims, periods, 0, &cart));
            MPI_CHECK(MPI_Comm_set_errhandler(cart, MPI_ERRORS_RETURN));
            result = benchmark(options[0], options[1], options[2] != 0, options[3] != 0,
                               cart, processes, active, dims);
            MPI_CHECK(MPI_Comm_free(&cart));
            MPI_CHECK(MPI_Comm_free(&participants));
        }
        MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
