#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;
constexpr int CUDA_BLOCK_SIZE = 256;

[[noreturn]] void failMpi(const char* expression, int error, const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char message[MPI_MAX_ERROR_STRING] = {};
    int message_length = 0;
    MPI_Error_string(error, message, &message_length);
    std::fprintf(stderr, "Rank %d: MPI error at %s:%d in %s: %.*s\n", rank, file,
                 line, expression, message_length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(expression)                                                   \
    do {                                                                        \
        const int mpi_check_error = (expression);                               \
        if (mpi_check_error != MPI_SUCCESS) {                                   \
            failMpi(#expression, mpi_check_error, __FILE__, __LINE__);          \
        }                                                                       \
    } while (false)

[[noreturn]] void failCuda(const char* expression, cudaError_t error,
                           const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d in %s: %s\n", rank, file,
                 line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression)                                                  \
    do {                                                                        \
        const cudaError_t cuda_check_error = (expression);                      \
        if (cuda_check_error != cudaSuccess) {                                  \
            failCuda(#expression, cuda_check_error, __FILE__, __LINE__);        \
        }                                                                       \
    } while (false)

struct Domain {
    int global_size = 0;
    int first_row = 0;
    int first_col = 0;
    int rows = 0;
    int cols = 0;
    int coords[2] = {0, 0};
    int dims[2] = {1, 1};
    int top = MPI_PROC_NULL;
    int bottom = MPI_PROC_NULL;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;

    [[nodiscard]] size_t elements() const {
        return static_cast<size_t>(rows) * static_cast<size_t>(cols);
    }
};

struct HostState {
    std::vector<val_t> energy;
    std::vector<val_t> total_flux;
};

int blockSize(int global_size, int partitions, int coordinate) {
    const int base = global_size / partitions;
    const int remainder = global_size % partitions;
    return base + (coordinate < remainder ? 1 : 0);
}

int blockStart(int global_size, int partitions, int coordinate) {
    const int base = global_size / partitions;
    const int remainder = global_size % partitions;
    return coordinate * base + std::min(coordinate, remainder);
}

Domain makeDomain(int n, MPI_Comm cart_comm) {
    Domain domain;
    domain.global_size = n;
    int periods[2] = {0, 0};
    MPI_CHECK(MPI_Cart_get(cart_comm, 2, domain.dims, periods, domain.coords));
    domain.rows = blockSize(n, domain.dims[0], domain.coords[0]);
    domain.cols = blockSize(n, domain.dims[1], domain.coords[1]);
    domain.first_row = blockStart(n, domain.dims[0], domain.coords[0]);
    domain.first_col = blockStart(n, domain.dims[1], domain.coords[1]);
    MPI_CHECK(MPI_Cart_shift(cart_comm, 0, 1, &domain.top, &domain.bottom));
    MPI_CHECK(MPI_Cart_shift(cart_comm, 1, 1, &domain.left, &domain.right));
    return domain;
}

class DeviceState {
public:
    explicit DeviceState(const Domain& domain)
        : rows_(domain.rows), cols_(domain.cols), pitch_(domain.cols + 2) {
        const size_t energy_elements =
            static_cast<size_t>(rows_ + 2) * static_cast<size_t>(pitch_);
        const size_t local_elements = domain.elements();

        CUDA_CHECK(cudaMalloc(&energy_a_, energy_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&energy_b_, energy_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&flux_a_, local_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&flux_b_, local_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(energy_a_, 0, energy_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(energy_b_, 0, energy_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(flux_a_, 0, local_elements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(flux_b_, 0, local_elements * sizeof(val_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&boundary_stream_, cudaStreamNonBlocking));
    }

    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;

    ~DeviceState() {
        if (interior_stream_ != nullptr) cudaStreamDestroy(interior_stream_);
        if (boundary_stream_ != nullptr) cudaStreamDestroy(boundary_stream_);
        if (energy_a_ != nullptr) cudaFree(energy_a_);
        if (energy_b_ != nullptr) cudaFree(energy_b_);
        if (flux_a_ != nullptr) cudaFree(flux_a_);
        if (flux_b_ != nullptr) cudaFree(flux_b_);
    }

    void swap() {
        std::swap(energy_a_, energy_b_);
        std::swap(flux_a_, flux_b_);
    }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(interior_stream_));
        CUDA_CHECK(cudaStreamSynchronize(boundary_stream_));
    }

    [[nodiscard]] val_t* energyRead() const { return energy_a_; }
    [[nodiscard]] val_t* energyWrite() const { return energy_b_; }
    [[nodiscard]] val_t* fluxRead() const { return flux_a_; }
    [[nodiscard]] val_t* fluxWrite() const { return flux_b_; }
    [[nodiscard]] int pitch() const { return pitch_; }
    [[nodiscard]] cudaStream_t interiorStream() const { return interior_stream_; }
    [[nodiscard]] cudaStream_t boundaryStream() const { return boundary_stream_; }

private:
    int rows_ = 0;
    int cols_ = 0;
    int pitch_ = 0;
    val_t* energy_a_ = nullptr;
    val_t* energy_b_ = nullptr;
    val_t* flux_a_ = nullptr;
    val_t* flux_b_ = nullptr;
    cudaStream_t interior_stream_ = nullptr;
    cudaStream_t boundary_stream_ = nullptr;
};

struct HaloPointers {
    val_t* top = nullptr;
    val_t* bottom = nullptr;
    val_t* left = nullptr;
    val_t* right = nullptr;
};

class HaloBuffers {
public:
    HaloBuffers(const Domain& domain, bool use_host_staging)
        : rows_(domain.rows), cols_(domain.cols),
          side_elements_(2 * static_cast<size_t>(domain.cols) +
                         2 * static_cast<size_t>(domain.rows)),
          use_host_staging_(use_host_staging) {
        CUDA_CHECK(cudaMalloc(&device_storage_, 2 * side_elements_ * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(device_storage_, 0,
                              2 * side_elements_ * sizeof(val_t)));
        assign(device_storage_, device_send_, device_recv_);
        if (use_host_staging_) {
            CUDA_CHECK(cudaHostAlloc(&host_storage_, 2 * side_elements_ * sizeof(val_t),
                                     cudaHostAllocPortable));
            std::memset(host_storage_, 0, 2 * side_elements_ * sizeof(val_t));
            assign(host_storage_, host_send_, host_recv_);
        }
    }

    HaloBuffers(const HaloBuffers&) = delete;
    HaloBuffers& operator=(const HaloBuffers&) = delete;

    ~HaloBuffers() {
        if (device_storage_ != nullptr) cudaFree(device_storage_);
        if (host_storage_ != nullptr) cudaFreeHost(host_storage_);
    }

    [[nodiscard]] HaloPointers& deviceSend() { return device_send_; }
    [[nodiscard]] HaloPointers& deviceRecv() { return device_recv_; }
    [[nodiscard]] HaloPointers& hostSend() { return host_send_; }
    [[nodiscard]] HaloPointers& hostRecv() { return host_recv_; }
    [[nodiscard]] uint64_t deviceBytes() const {
        return 2ULL * side_elements_ * sizeof(val_t);
    }

    void stageSendsToHost(cudaStream_t stream) {
        if (!use_host_staging_) return;
        CUDA_CHECK(cudaMemcpyAsync(host_storage_, device_storage_,
                                   side_elements_ * sizeof(val_t),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void stageReceivesToDevice(cudaStream_t stream) {
        if (!use_host_staging_) return;
        CUDA_CHECK(cudaMemcpyAsync(device_storage_ + side_elements_,
                                   host_storage_ + side_elements_,
                                   side_elements_ * sizeof(val_t),
                                   cudaMemcpyHostToDevice, stream));
    }

private:
    void assign(val_t* storage, HaloPointers& send, HaloPointers& recv) const {
        send.top = storage;
        send.bottom = send.top + cols_;
        send.left = send.bottom + cols_;
        send.right = send.left + rows_;
        recv.top = storage + side_elements_;
        recv.bottom = recv.top + cols_;
        recv.left = recv.bottom + cols_;
        recv.right = recv.left + rows_;
    }

    int rows_ = 0;
    int cols_ = 0;
    size_t side_elements_ = 0;
    bool use_host_staging_ = true;
    val_t* device_storage_ = nullptr;
    val_t* host_storage_ = nullptr;
    HaloPointers device_send_;
    HaloPointers device_recv_;
    HaloPointers host_send_;
    HaloPointers host_recv_;
};

__device__ __forceinline__ val_t externalFlow(int global_row, int global_col,
                                              int n) {
    const int last = n - 1;
    // This order exactly matches the four material assignments in buildSquare2D,
    // including their overlap for a 1x1 mesh.
    if (global_row == last && global_col == last) return INFLOW;
    if ((global_row == 0 && global_col == last) ||
        (global_row == last && global_col == 0)) return OUTFLOW;
    if (global_row == 0 && global_col == 0) return INFLOW;
    return 0.0;
}

__device__ __forceinline__ val_t connectionContribution(val_t current,
                                                         val_t neighbor) {
    return (neighbor - current) * TRANSFER_COEFF * CONNECTION_FLUX * 0.25;
}

__global__ void packBoundaries(const val_t* __restrict__ energy, int rows,
                               int cols, int pitch, HaloPointers send) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < max(rows, cols); i += blockDim.x * gridDim.x) {
        if (i < cols) {
            send.top[i] = energy[pitch + 1 + i];
            send.bottom[i] = energy[static_cast<size_t>(rows) * pitch + 1 + i];
        }
        if (i < rows) {
            send.left[i] = energy[static_cast<size_t>(i + 1) * pitch + 1];
            send.right[i] = energy[static_cast<size_t>(i + 1) * pitch + cols];
        }
    }
}

__device__ __forceinline__ void updateBoundaryPoint(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ accumulated_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_accumulated_flux,
    HaloPointers received, int local_row, int local_col, int rows, int cols,
    int pitch, int first_row, int first_col, int n);

__global__ void updateInterior(const val_t* __restrict__ energy,
                               const val_t* __restrict__ accumulated_flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_accumulated_flux,
                               int rows, int cols, int pitch) {
    const int interior_rows = rows - 2;
    const int interior_cols = cols - 2;
    const size_t count =
        static_cast<size_t>(interior_rows) * static_cast<size_t>(interior_cols);

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int local_row = static_cast<int>(linear / interior_cols) + 1;
        const int local_col = static_cast<int>(linear % interior_cols) + 1;
        const size_t energy_index =
            static_cast<size_t>(local_row + 1) * pitch + local_col + 1;
        const size_t local_index = static_cast<size_t>(local_row) * cols + local_col;
        const val_t current = energy[energy_index];
        val_t flux = 0.0;
        // Preserve the original +x, -x, +y, -y connection order.
        flux += connectionContribution(current, energy[energy_index + pitch]);
        flux += connectionContribution(current, energy[energy_index - pitch]);
        flux += connectionContribution(current, energy[energy_index + 1]);
        flux += connectionContribution(current, energy[energy_index - 1]);
        next_energy[energy_index] = current + flux;
        next_accumulated_flux[local_index] = accumulated_flux[local_index] + fabs(flux);
    }
}

__global__ void updateAll(const val_t* __restrict__ energy,
                          const val_t* __restrict__ accumulated_flux,
                          val_t* __restrict__ next_energy,
                          val_t* __restrict__ next_accumulated_flux,
                          HaloPointers received, int rows, int cols, int pitch,
                          int first_row, int first_col, int n) {
    const size_t count = static_cast<size_t>(rows) * cols;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int local_row = static_cast<int>(linear / cols);
        const int local_col = static_cast<int>(linear % cols);
        updateBoundaryPoint(energy, accumulated_flux, next_energy,
                            next_accumulated_flux, received, local_row, local_col,
                            rows, cols, pitch, first_row, first_col, n);
    }
}

__device__ __forceinline__ void updateBoundaryPoint(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ accumulated_flux,
    val_t* __restrict__ next_energy,
    val_t* __restrict__ next_accumulated_flux,
    HaloPointers received, int local_row, int local_col, int rows, int cols,
    int pitch, int first_row, int first_col, int n) {
    const int global_row = first_row + local_row;
    const int global_col = first_col + local_col;
    const size_t energy_index =
        static_cast<size_t>(local_row + 1) * pitch + local_col + 1;
    const size_t local_index = static_cast<size_t>(local_row) * cols + local_col;
    const val_t current = energy[energy_index];
    val_t flux = externalFlow(global_row, global_col, n);

    // +x (bottom), -x (top), +y (right), -y (left), matching the CPU baseline.
    if (global_row + 1 < n) {
        const val_t neighbor = local_row + 1 < rows
                                   ? energy[energy_index + pitch]
                                   : received.bottom[local_col];
        flux += connectionContribution(current, neighbor);
    }
    if (global_row > 0) {
        const val_t neighbor = local_row > 0
                                   ? energy[energy_index - pitch]
                                   : received.top[local_col];
        flux += connectionContribution(current, neighbor);
    }
    if (global_col + 1 < n) {
        const val_t neighbor = local_col + 1 < cols
                                   ? energy[energy_index + 1]
                                   : received.right[local_row];
        flux += connectionContribution(current, neighbor);
    }
    if (global_col > 0) {
        const val_t neighbor = local_col > 0
                                   ? energy[energy_index - 1]
                                   : received.left[local_row];
        flux += connectionContribution(current, neighbor);
    }

    next_energy[energy_index] = current + flux;
    next_accumulated_flux[local_index] = accumulated_flux[local_index] + fabs(flux);
}

__global__ void updateBoundary(const val_t* __restrict__ energy,
                               const val_t* __restrict__ accumulated_flux,
                               val_t* __restrict__ next_energy,
                               val_t* __restrict__ next_accumulated_flux,
                               HaloPointers received, int rows, int cols, int pitch,
                               int first_row, int first_col, int n) {
    size_t perimeter = 0;
    if (rows == 1) {
        perimeter = cols;
    } else if (cols == 1) {
        perimeter = rows;
    } else {
        perimeter = 2ULL * cols + 2ULL * (rows - 2);
    }

    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < perimeter; p += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int local_row = 0;
        int local_col = 0;
        if (rows == 1) {
            local_col = static_cast<int>(p);
        } else if (cols == 1) {
            local_row = static_cast<int>(p);
        } else if (p < static_cast<size_t>(cols)) {
            local_col = static_cast<int>(p);
        } else if (p < 2ULL * static_cast<size_t>(cols)) {
            local_row = rows - 1;
            local_col = static_cast<int>(p) - cols;
        } else {
            const int edge_index = static_cast<int>(p - 2ULL * cols);
            local_row = edge_index / 2 + 1;
            local_col = (edge_index & 1) == 0 ? 0 : cols - 1;
        }
        updateBoundaryPoint(energy, accumulated_flux, next_energy,
                            next_accumulated_flux, received, local_row, local_col,
                            rows, cols, pitch, first_row, first_col, n);
    }
}

int launchBlocks(size_t work_items, int multiprocessors) {
    if (work_items == 0) return 0;
    const size_t required = (work_items + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    const size_t occupancy_limit = static_cast<size_t>(multiprocessors) * 8;
    return static_cast<int>(std::min(required, occupancy_limit));
}

void warmupDeviceKernels(const Domain& domain, DeviceState& state,
                         HaloBuffers& halos) {
    // CUDA 12 loads individual kernels lazily. Zero-work launches make that
    // one-time cost setup rather than benchmark time without changing state.
    packBoundaries<<<1, 1, 0, state.boundaryStream()>>>(
        state.energyRead(), 0, 0, state.pitch(), halos.deviceSend());
    updateInterior<<<1, 1, 0, state.interiorStream()>>>(
        state.energyRead(), state.fluxRead(), state.energyWrite(),
        state.fluxWrite(), 2, 2, state.pitch());
    updateBoundary<<<1, 1, 0, state.boundaryStream()>>>(
        state.energyRead(), state.fluxRead(), state.energyWrite(),
        state.fluxWrite(), halos.deviceRecv(), 1, 0, state.pitch(),
        domain.first_row, domain.first_col, domain.global_size);
    updateAll<<<1, 1, 0, state.interiorStream()>>>(
        state.energyRead(), state.fluxRead(), state.energyWrite(),
        state.fluxWrite(), halos.deviceRecv(), 0, 0, state.pitch(),
        domain.first_row, domain.first_col, domain.global_size);
    CUDA_CHECK(cudaGetLastError());
    state.synchronize();
}

bool queryCudaAwareMpi() {
    bool supported = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    supported = MPIX_Query_cuda_support() != 0;
#endif
    // MPI implementations without a query extension can be enabled explicitly;
    // zero forces the portable pinned-host path for debugging or compatibility.
    if (const char* value = std::getenv("UNSTRUCTURED_CUDA_AWARE_MPI")) {
        supported = std::strcmp(value, "0") != 0;
    }
    return supported;
}

void exchangeHalos(const Domain& domain, MPI_Comm cart_comm,
                   HaloPointers send, HaloPointers recv) {
    enum : int { TO_TOP = 0, TO_BOTTOM = 1, TO_LEFT = 2, TO_RIGHT = 3 };
    MPI_Request requests[8];
    int count = 0;
    MPI_CHECK(MPI_Irecv(recv.top, domain.cols, MPI_DOUBLE, domain.top, TO_BOTTOM,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Irecv(recv.bottom, domain.cols, MPI_DOUBLE, domain.bottom, TO_TOP,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Irecv(recv.left, domain.rows, MPI_DOUBLE, domain.left, TO_RIGHT,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Irecv(recv.right, domain.rows, MPI_DOUBLE, domain.right, TO_LEFT,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Isend(send.top, domain.cols, MPI_DOUBLE, domain.top, TO_TOP,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Isend(send.bottom, domain.cols, MPI_DOUBLE, domain.bottom, TO_BOTTOM,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Isend(send.left, domain.rows, MPI_DOUBLE, domain.left, TO_LEFT,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Isend(send.right, domain.rows, MPI_DOUBLE, domain.right, TO_RIGHT,
                        cart_comm, &requests[count++]));
    MPI_CHECK(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
}

void runSimulation(const Domain& domain, MPI_Comm cart_comm, DeviceState& state,
                   HaloBuffers& halos, int n_iters, bool cuda_aware_mpi,
                   int multiprocessors) {
    const size_t interior_items =
        domain.rows > 2 && domain.cols > 2
            ? static_cast<size_t>(domain.rows - 2) * (domain.cols - 2)
            : 0;
    const size_t perimeter_items = domain.rows == 1
                                       ? static_cast<size_t>(domain.cols)
                                   : domain.cols == 1
                                       ? static_cast<size_t>(domain.rows)
                                       : 2ULL * domain.cols + 2ULL * (domain.rows - 2);
    const int pack_blocks = launchBlocks(std::max(domain.rows, domain.cols),
                                         multiprocessors);
    const int interior_blocks = launchBlocks(interior_items, multiprocessors);
    const int boundary_blocks = launchBlocks(perimeter_items, multiprocessors);
    const int all_blocks = launchBlocks(domain.elements(), multiprocessors);
    const bool has_remote_neighbors =
        domain.top != MPI_PROC_NULL || domain.bottom != MPI_PROC_NULL ||
        domain.left != MPI_PROC_NULL || domain.right != MPI_PROC_NULL;

    // On one MPI rank there is no communication dependency. Queue all Jacobi
    // iterations in one stream, avoiding a host synchronization per iteration.
    if (!has_remote_neighbors) {
        for (int iter = 0; iter < n_iters; ++iter) {
            updateAll<<<all_blocks, CUDA_BLOCK_SIZE, 0, state.interiorStream()>>>(
                state.energyRead(), state.fluxRead(), state.energyWrite(),
                state.fluxWrite(), halos.deviceRecv(), domain.rows, domain.cols,
                state.pitch(), domain.first_row, domain.first_col,
                domain.global_size);
            CUDA_CHECK(cudaGetLastError());
            state.swap();
        }
        state.synchronize();
        return;
    }

    for (int iter = 0; iter < n_iters; ++iter) {
        if (iter != 0) state.synchronize();

        packBoundaries<<<pack_blocks, CUDA_BLOCK_SIZE, 0, state.boundaryStream()>>>(
            state.energyRead(), domain.rows, domain.cols, state.pitch(),
            halos.deviceSend());
        CUDA_CHECK(cudaGetLastError());

        if (interior_blocks > 0) {
            updateInterior<<<interior_blocks, CUDA_BLOCK_SIZE, 0,
                             state.interiorStream()>>>(
                state.energyRead(), state.fluxRead(), state.energyWrite(),
                state.fluxWrite(), domain.rows, domain.cols, state.pitch());
            CUDA_CHECK(cudaGetLastError());
        }

        halos.stageSendsToHost(state.boundaryStream());
        CUDA_CHECK(cudaStreamSynchronize(state.boundaryStream()));
        HaloPointers send = cuda_aware_mpi ? halos.deviceSend() : halos.hostSend();
        HaloPointers recv = cuda_aware_mpi ? halos.deviceRecv() : halos.hostRecv();
        exchangeHalos(domain, cart_comm, send, recv);
        halos.stageReceivesToDevice(state.boundaryStream());

        updateBoundary<<<boundary_blocks, CUDA_BLOCK_SIZE, 0,
                         state.boundaryStream()>>>(
            state.energyRead(), state.fluxRead(), state.energyWrite(),
            state.fluxWrite(), halos.deviceRecv(), domain.rows, domain.cols,
            state.pitch(), domain.first_row, domain.first_col, domain.global_size);
        CUDA_CHECK(cudaGetLastError());
        state.swap();
    }
    state.synchronize();
}

HostState copyResultsToHost(const Domain& domain, const DeviceState& state) {
    HostState host;
    host.energy.resize(domain.elements());
    host.total_flux.resize(domain.elements());
    CUDA_CHECK(cudaMemcpy2D(host.energy.data(),
                            static_cast<size_t>(domain.cols) * sizeof(val_t),
                            state.energyRead() + state.pitch() + 1,
                            static_cast<size_t>(state.pitch()) * sizeof(val_t),
                            static_cast<size_t>(domain.cols) * sizeof(val_t),
                            domain.rows, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host.total_flux.data(), state.fluxRead(),
                          domain.elements() * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

struct ResultStats {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    uint64_t hash = 0;
    int finite = 1;
};

ResultStats analyzeLocalResults(const Domain& domain, const HostState& state) {
    ResultStats stats;
    const int64_t count = static_cast<int64_t>(domain.elements());
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    uint64_t result_hash = 0;
    int all_finite = 1;

#pragma omp parallel for schedule(static) reduction(+ : energy_sum, flux_sum)    \
    reduction(min : energy_min) reduction(max : energy_max)                     \
    reduction(^ : result_hash) reduction(& : all_finite)
    for (int64_t i = 0; i < count; ++i) {
        const val_t energy = state.energy[static_cast<size_t>(i)];
        const val_t flux = state.total_flux[static_cast<size_t>(i)];
        const int local_row = static_cast<int>(i / domain.cols);
        const int local_col = static_cast<int>(i % domain.cols);
        const uint64_t global_index =
            static_cast<uint64_t>(domain.first_row + local_row) * domain.global_size +
            domain.first_col + local_col;
        const uint64_t energy_bits = std::bit_cast<uint64_t>(energy);
        const uint64_t flux_bits = std::bit_cast<uint64_t>(flux);
        energy_sum += energy;
        flux_sum += flux;
        energy_min = std::min(energy_min, energy);
        energy_max = std::max(energy_max, energy);
        result_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        result_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        all_finite &= std::isfinite(energy) && std::isfinite(flux);
    }
    stats.energy_sum = energy_sum;
    stats.flux_sum = flux_sum;
    stats.energy_min = energy_min;
    stats.energy_max = energy_max;
    stats.hash = result_hash;
    stats.finite = all_finite;
    return stats;
}

ResultStats reduceResultStats(const ResultStats& local, MPI_Comm comm, int root) {
    ResultStats global;
    MPI_CHECK(MPI_Reduce(&local.energy_sum, &global.energy_sum, 1, MPI_DOUBLE,
                         MPI_SUM, root, comm));
    MPI_CHECK(MPI_Reduce(&local.flux_sum, &global.flux_sum, 1, MPI_DOUBLE,
                         MPI_SUM, root, comm));
    MPI_CHECK(MPI_Reduce(&local.energy_min, &global.energy_min, 1, MPI_DOUBLE,
                         MPI_MIN, root, comm));
    MPI_CHECK(MPI_Reduce(&local.energy_max, &global.energy_max, 1, MPI_DOUBLE,
                         MPI_MAX, root, comm));
    MPI_CHECK(MPI_Reduce(&local.hash, &global.hash, 1, MPI_UINT64_T, MPI_BXOR,
                         root, comm));
    MPI_CHECK(MPI_Reduce(&local.finite, &global.finite, 1, MPI_INT, MPI_LAND,
                         root, comm));
    return global;
}

std::vector<val_t> gatherEnergies(const Domain& domain,
                                  const std::vector<val_t>& local_energy,
                                  MPI_Comm comm, int rank, int ranks) {
    const int local_count = static_cast<int>(domain.elements());
    int metadata[4] = {domain.first_row, domain.first_col, domain.rows, domain.cols};
    std::vector<int> all_metadata(rank == 0 ? 4 * ranks : 0);
    std::vector<int> counts(rank == 0 ? ranks : 0);
    MPI_CHECK(MPI_Gather(metadata, 4, MPI_INT, all_metadata.data(), 4, MPI_INT, 0,
                         comm));
    MPI_CHECK(MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                         comm));

    std::vector<int> displacements(rank == 0 ? ranks : 0);
    std::vector<val_t> packed;
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < ranks; ++r) {
            displacements[r] = offset;
            offset += counts[r];
        }
        packed.resize(static_cast<size_t>(offset));
    }
    MPI_CHECK(MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, packed.data(),
                          counts.data(), displacements.data(), MPI_DOUBLE, 0, comm));

    if (rank != 0) return {};
    std::vector<val_t> global(static_cast<size_t>(domain.global_size) *
                              domain.global_size);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < ranks; ++r) {
        const int first_row = all_metadata[4 * r];
        const int first_col = all_metadata[4 * r + 1];
        const int rows = all_metadata[4 * r + 2];
        const int cols = all_metadata[4 * r + 3];
        for (int row = 0; row < rows; ++row) {
            std::memcpy(global.data() +
                            static_cast<size_t>(first_row + row) * domain.global_size +
                            first_col,
                        packed.data() + displacements[r] +
                            static_cast<size_t>(row) * cols,
                        static_cast<size_t>(cols) * sizeof(val_t));
        }
    }
    return global;
}

bool validateResults(const ResultStats& stats) {
    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", stats.energy_sum);
    std::printf("  Flux sum: %.2f\n", stats.flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", stats.energy_min,
                stats.energy_max);
    if (!stats.finite || !std::isfinite(stats.energy_sum) ||
        !std::isfinite(stats.flux_sum) || !std::isfinite(stats.energy_min) ||
        !std::isfinite(stats.energy_max)) {
        std::printf("  ERROR: Result contains a non-finite value\n");
        return false;
    }
    constexpr val_t energy_epsilon = 1e-8;
    if (std::abs(stats.energy_sum) > energy_epsilon) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    std::printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int selectDevice(int rank, int& devices_on_node, int& ranks_on_node) {
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &ranks_on_node));
    CUDA_CHECK(cudaGetDeviceCount(&devices_on_node));
    if (devices_on_node <= 0) {
        std::fprintf(stderr, "Rank %d: no CUDA accelerator is available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = local_rank % devices_on_node;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    return device;
}

int runProgram(int argc, char** argv) {
    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    int n = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    bool arguments_valid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            arguments_valid = false;
        }
    }
    if (show_help || !arguments_valid) {
        if (rank == 0) printUsage(argv[0]);
        return arguments_valid ? 0 : 1;
    }
    const int64_t global_elements = static_cast<int64_t>(n) * n;
    if (n <= 0 || n_iters < 0 || global_elements > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be positive with N*N <= INT_MAX, and "
                         "iterations must be non-negative.\n");
        }
        return 1;
    }

    int dims[2] = {0, 0};
    MPI_CHECK(MPI_Dims_create(ranks, 2, dims));
    if (dims[0] > n || dims[1] > n) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "The %d x %d MPI process grid exceeds the %d x %d mesh. "
                         "Use no more ranks per dimension than mesh elements.\n",
                         dims[0], dims[1], n, n);
        }
        return 1;
    }
    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm));
    Domain domain = makeDomain(n, cart_comm);

    int devices_on_node = 0;
    int ranks_on_node = 1;
    const int device = selectDevice(rank, devices_on_node, ranks_on_node);
    omp_set_dynamic(0);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / ranks_on_node));
    }
    const int openmp_threads = omp_get_max_threads();
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));
    const bool cuda_aware_mpi = queryCudaAwareMpi();

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\n", n, n,
                    static_cast<long long>(global_elements));
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks (%d x %d), up to %d OpenMP "
                    "threads/rank\n",
                    ranks, dims[0], dims[1], openmp_threads);
        std::printf("CUDA: %s; MPI halo path: %s\n\n", device_properties.name,
                    cuda_aware_mpi ? "CUDA-aware direct" : "pinned-host staged");
        std::printf("Building distributed unstructured mesh...\n");
    }

    DeviceState device_state(domain);
    HaloBuffers halo_buffers(domain, !cuda_aware_mpi);
    CUDA_CHECK(cudaDeviceSynchronize());
    warmupDeviceKernels(domain, device_state, halo_buffers);
    const bool has_remote_neighbors =
        domain.top != MPI_PROC_NULL || domain.bottom != MPI_PROC_NULL ||
        domain.left != MPI_PROC_NULL || domain.right != MPI_PROC_NULL;
    if (has_remote_neighbors) {
        HaloPointers warm_send =
            cuda_aware_mpi ? halo_buffers.deviceSend() : halo_buffers.hostSend();
        HaloPointers warm_recv =
            cuda_aware_mpi ? halo_buffers.deviceRecv() : halo_buffers.hostRecv();
        exchangeHalos(domain, cart_comm, warm_send, warm_recv);
    }
    const uint64_t local_dynamic_bytes =
        2ULL * static_cast<uint64_t>(domain.rows + 2) * (domain.cols + 2) *
            sizeof(val_t) +
        2ULL * domain.elements() * sizeof(val_t) + halo_buffers.deviceBytes();
    uint64_t aggregate_dynamic_bytes = 0;
    uint64_t max_dynamic_bytes = 0;
    MPI_CHECK(MPI_Reduce(&local_dynamic_bytes, &aggregate_dynamic_bytes, 1,
                         MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_dynamic_bytes, &max_dynamic_bytes, 1, MPI_UINT64_T,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        std::printf("Memory usage: %.2f MB aggregate (max %.2f MB/rank)\n\n",
                    aggregate_dynamic_bytes / (1024.0 * 1024.0),
                    max_dynamic_bytes / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    runSimulation(domain, cart_comm, device_state, halo_buffers, n_iters,
                  cuda_aware_mpi, device_properties.multiProcessorCount);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    HostState host_state = copyResultsToHost(domain, device_state);
    const ResultStats local_stats = analyzeLocalResults(domain, host_state);
    const ResultStats global_stats = reduceResultStats(local_stats, MPI_COMM_WORLD, 0);

    if (rank == 0) {
        const double elapsed_ms = elapsed_seconds * 1000.0;
        const double time_per_iter = n_iters > 0 ? elapsed_ms / n_iters : 0.0;
        const double giga_elements_per_second =
            elapsed_seconds > 0.0
                ? (static_cast<double>(n_iters) * global_elements) /
                      elapsed_seconds / 1.0e9
                : 0.0;
        const double gflops = giga_elements_per_second * 22.0;
        std::printf("Computation time: %.3f ms\n", elapsed_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(global_stats.hash));
    }

    if (print_results_requested) {
        std::vector<val_t> global_energy =
            gatherEnergies(domain, host_state.energy, MPI_COMM_WORLD, rank, ranks);
        if (rank == 0) print_results(global_energy, "ElementEnergy");
    }

    bool valid = true;
    if (validate && rank == 0) valid = validateResults(global_stats);
    int valid_int = valid ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&cart_comm));
    return valid_int != 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int init_error =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (init_error != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int result = runProgram(argc, argv);
    MPI_Finalize();
    return result;
}
