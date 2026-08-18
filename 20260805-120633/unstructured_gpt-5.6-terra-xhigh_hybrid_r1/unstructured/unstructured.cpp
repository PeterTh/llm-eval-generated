#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using val_t = double;

namespace {

constexpr int kHaloWidth = 1;
constexpr int kTagXToLower = 100;
constexpr int kTagXToHigher = 101;
constexpr int kTagYToLower = 200;
constexpr int kTagYToHigher = 201;

[[noreturn]] void cudaFail(cudaError_t status, const char* operation,
                           const char* file, int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d while executing %s: %s\n",
                 file, line, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(operation)                                                   \
    do {                                                                        \
        const cudaError_t cuda_status_ = (operation);                          \
        if (cuda_status_ != cudaSuccess) {                                     \
            cudaFail(cuda_status_, #operation, __FILE__, __LINE__);            \
        }                                                                       \
    } while (false)

struct Decomposition {
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dimensions[2] = {1, 1};
    int coordinates[2] = {0, 0};
    int x_lower = MPI_PROC_NULL;
    int x_higher = MPI_PROC_NULL;
    int y_lower = MPI_PROC_NULL;
    int y_higher = MPI_PROC_NULL;
    int global_n = 0;
    int offset_x = 0;
    int offset_y = 0;
    int local_nx = 0;
    int local_ny = 0;

    int pitch() const { return local_ny + 2 * kHaloWidth; }
    size_t elementCountWithHalo() const {
        return static_cast<size_t>(local_nx + 2 * kHaloWidth) * pitch();
    }
};

int blockExtent(int global_extent, int coordinate, int dimensions) {
    const int base = global_extent / dimensions;
    const int remainder = global_extent % dimensions;
    return base + (coordinate < remainder ? 1 : 0);
}

int blockOffset(int global_extent, int coordinate, int dimensions) {
    const int base = global_extent / dimensions;
    const int remainder = global_extent % dimensions;
    return coordinate * base + std::min(coordinate, remainder);
}

Decomposition makeDecomposition(MPI_Comm parent_communicator, int global_n) {
    Decomposition decomposition;
    decomposition.global_n = global_n;

    MPI_Comm_size(parent_communicator, &decomposition.size);
    int dimensions[2] = {0, 0};
    MPI_Dims_create(decomposition.size, 2, dimensions);
    const int periods[2] = {0, 0};
    MPI_Cart_create(parent_communicator, 2, dimensions, periods, 0,
                    &decomposition.communicator);
    if (decomposition.communicator == MPI_COMM_NULL) {
        std::fprintf(stderr, "Unable to create the MPI Cartesian communicator\n");
        MPI_Abort(parent_communicator, EXIT_FAILURE);
    }

    MPI_Comm_rank(decomposition.communicator, &decomposition.rank);
    decomposition.dimensions[0] = dimensions[0];
    decomposition.dimensions[1] = dimensions[1];
    MPI_Cart_coords(decomposition.communicator, decomposition.rank, 2,
                    decomposition.coordinates);
    MPI_Cart_shift(decomposition.communicator, 0, 1, &decomposition.x_lower,
                   &decomposition.x_higher);
    MPI_Cart_shift(decomposition.communicator, 1, 1, &decomposition.y_lower,
                   &decomposition.y_higher);

    decomposition.local_nx =
        blockExtent(global_n, decomposition.coordinates[0], dimensions[0]);
    decomposition.local_ny =
        blockExtent(global_n, decomposition.coordinates[1], dimensions[1]);
    decomposition.offset_x =
        blockOffset(global_n, decomposition.coordinates[0], dimensions[0]);
    decomposition.offset_y =
        blockOffset(global_n, decomposition.coordinates[1], dimensions[1]);
    return decomposition;
}

// A Cartesian decomposition cannot give useful work to every rank when the
// requested process count has an incompatible factorization for a tiny grid.
// Keep the largest rectangular subset active rather than creating zero-sized
// subdomains; the remaining ranks simply finalize without entering the solver.
int activeRankCount(int requested_ranks, int global_n) {
    for (int candidate = requested_ranks; candidate > 0; --candidate) {
        int dimensions[2] = {0, 0};
        MPI_Dims_create(candidate, 2, dimensions);
        if (dimensions[0] <= global_n && dimensions[1] <= global_n) {
            return candidate;
        }
    }
    return 1;
}

void selectDevice(const Decomposition& decomposition) {
    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(decomposition.communicator, MPI_COMM_TYPE_SHARED,
                        decomposition.rank, MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (decomposition.rank == 0) {
            std::fprintf(stderr, "No CUDA devices are visible to this MPI job\n");
        }
        MPI_Abort(decomposition.communicator, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_communicator);
}

struct DeviceState {
    val_t* energy = nullptr;
    val_t* total_flux = nullptr;
    val_t* next_energy = nullptr;
    val_t* next_total_flux = nullptr;

    void allocate(size_t count) {
        CUDA_CHECK(cudaMalloc(&energy, count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&total_flux, count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&next_energy, count * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&next_total_flux, count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(energy, 0, count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(total_flux, 0, count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(next_energy, 0, count * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(next_total_flux, 0, count * sizeof(val_t)));
    }

    void swap() {
        std::swap(energy, next_energy);
        std::swap(total_flux, next_total_flux);
    }

    void release() {
        if (energy != nullptr) CUDA_CHECK(cudaFree(energy));
        if (total_flux != nullptr) CUDA_CHECK(cudaFree(total_flux));
        if (next_energy != nullptr) CUDA_CHECK(cudaFree(next_energy));
        if (next_total_flux != nullptr) CUDA_CHECK(cudaFree(next_total_flux));
        energy = total_flux = next_energy = next_total_flux = nullptr;
    }
};

// Pinned staging buffers make CUDA copies asynchronous with the interior kernel and
// also work with both CUDA-aware and ordinary MPI implementations.
struct HaloBuffers {
    val_t* x_low_send = nullptr;
    val_t* x_high_send = nullptr;
    val_t* x_low_recv = nullptr;
    val_t* x_high_recv = nullptr;
    val_t* y_low_send = nullptr;
    val_t* y_high_send = nullptr;
    val_t* y_low_recv = nullptr;
    val_t* y_high_recv = nullptr;

    void allocate(const Decomposition& decomposition) {
        const size_t x_bytes = static_cast<size_t>(decomposition.local_ny) * sizeof(val_t);
        const size_t y_bytes = static_cast<size_t>(decomposition.local_nx) * sizeof(val_t);
        CUDA_CHECK(cudaMallocHost(&x_low_send, x_bytes));
        CUDA_CHECK(cudaMallocHost(&x_high_send, x_bytes));
        CUDA_CHECK(cudaMallocHost(&x_low_recv, x_bytes));
        CUDA_CHECK(cudaMallocHost(&x_high_recv, x_bytes));
        CUDA_CHECK(cudaMallocHost(&y_low_send, y_bytes));
        CUDA_CHECK(cudaMallocHost(&y_high_send, y_bytes));
        CUDA_CHECK(cudaMallocHost(&y_low_recv, y_bytes));
        CUDA_CHECK(cudaMallocHost(&y_high_recv, y_bytes));
    }

    void release() {
        if (x_low_send != nullptr) CUDA_CHECK(cudaFreeHost(x_low_send));
        if (x_high_send != nullptr) CUDA_CHECK(cudaFreeHost(x_high_send));
        if (x_low_recv != nullptr) CUDA_CHECK(cudaFreeHost(x_low_recv));
        if (x_high_recv != nullptr) CUDA_CHECK(cudaFreeHost(x_high_recv));
        if (y_low_send != nullptr) CUDA_CHECK(cudaFreeHost(y_low_send));
        if (y_high_send != nullptr) CUDA_CHECK(cudaFreeHost(y_high_send));
        if (y_low_recv != nullptr) CUDA_CHECK(cudaFreeHost(y_low_recv));
        if (y_high_recv != nullptr) CUDA_CHECK(cudaFreeHost(y_high_recv));
        x_low_send = x_high_send = x_low_recv = x_high_recv = nullptr;
        y_low_send = y_high_send = y_low_recv = y_high_recv = nullptr;
    }
};

__device__ __forceinline__ val_t externalFlow(int global_x, int global_y,
                                              int global_n) {
    // These predicates reproduce the order in which buildSquare2D assigns the
    // four corners, including its well-defined one-cell-grid behavior.
    if ((global_x == 0 && global_y == 0) ||
        (global_x == global_n - 1 && global_y == global_n - 1)) {
        return 0.5;
    }
    if ((global_x == 0 && global_y == global_n - 1) ||
        (global_x == global_n - 1 && global_y == 0)) {
        return -0.5;
    }
    return 0.0;
}

__device__ __forceinline__ val_t computeFlux(val_t this_energy,
                                              val_t neighbor_energy) {
    return (neighbor_energy - this_energy) * 0.8 * 1.0 * 0.25;
}

__device__ __forceinline__ void updateElement(
    const val_t* __restrict__ energy, const val_t* __restrict__ total_flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_total_flux,
    int local_x, int local_y, int local_nx, int local_ny, int pitch,
    int global_x_offset, int global_y_offset, int global_n) {
    const int global_x = global_x_offset + local_x;
    const int global_y = global_y_offset + local_y;
    const size_t index = static_cast<size_t>(local_x + kHaloWidth) * pitch +
                         local_y + kHaloWidth;
    const val_t this_energy = energy[index];
    val_t flux = externalFlow(global_x, global_y, global_n);

    // Keep precisely the original (x+1, x-1, y+1, y-1) connection order.
    if (global_x + 1 < global_n) flux += computeFlux(this_energy, energy[index + pitch]);
    if (global_x > 0) flux += computeFlux(this_energy, energy[index - pitch]);
    if (global_y + 1 < global_n) flux += computeFlux(this_energy, energy[index + 1]);
    if (global_y > 0) flux += computeFlux(this_energy, energy[index - 1]);

    next_energy[index] = this_energy + flux;
    next_total_flux[index] = total_flux[index] + fabs(flux);
}

__global__ void updateInteriorKernel(
    const val_t* __restrict__ energy, const val_t* __restrict__ total_flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_total_flux,
    int local_nx, int local_ny, int pitch, int global_x_offset,
    int global_y_offset, int global_n) {
    const int local_y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
    const int local_x = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    if (local_x < local_nx - 1 && local_y < local_ny - 1) {
        updateElement(energy, total_flux, next_energy, next_total_flux, local_x,
                      local_y, local_nx, local_ny, pitch, global_x_offset,
                      global_y_offset, global_n);
    }
}

__global__ void updateBoundaryKernel(
    const val_t* __restrict__ energy, const val_t* __restrict__ total_flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_total_flux,
    int local_nx, int local_ny, int pitch, int global_x_offset,
    int global_y_offset, int global_n) {
    const int local_y = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_x = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (local_x >= local_nx || local_y >= local_ny) return;

    const bool interior = local_x > 0 && local_x < local_nx - 1 &&
                          local_y > 0 && local_y < local_ny - 1;
    if (!interior) {
        updateElement(energy, total_flux, next_energy, next_total_flux, local_x,
                      local_y, local_nx, local_ny, pitch, global_x_offset,
                      global_y_offset, global_n);
    }
}

dim3 blockShape() { return dim3(32, 8, 1); }

dim3 gridShape(int nx, int ny, bool interior) {
    const int work_x = interior ? std::max(nx - 2, 0) : nx;
    const int work_y = interior ? std::max(ny - 2, 0) : ny;
    const dim3 block = blockShape();
    return dim3((work_y + block.x - 1) / block.x,
                (work_x + block.y - 1) / block.y, 1);
}

void launchInterior(const Decomposition& decomposition, const DeviceState& state,
                    cudaStream_t compute_stream) {
    if (decomposition.local_nx <= 2 || decomposition.local_ny <= 2) return;
    updateInteriorKernel<<<gridShape(decomposition.local_nx, decomposition.local_ny, true),
                            blockShape(), 0, compute_stream>>>(
        state.energy, state.total_flux, state.next_energy, state.next_total_flux,
        decomposition.local_nx, decomposition.local_ny, decomposition.pitch(),
        decomposition.offset_x, decomposition.offset_y, decomposition.global_n);
    CUDA_CHECK(cudaGetLastError());
}

void launchBoundary(const Decomposition& decomposition, const DeviceState& state,
                    cudaStream_t compute_stream) {
    updateBoundaryKernel<<<gridShape(decomposition.local_nx, decomposition.local_ny, false),
                            blockShape(), 0, compute_stream>>>(
        state.energy, state.total_flux, state.next_energy, state.next_total_flux,
        decomposition.local_nx, decomposition.local_ny, decomposition.pitch(),
        decomposition.offset_x, decomposition.offset_y, decomposition.global_n);
    CUDA_CHECK(cudaGetLastError());
}

void exchangeHalos(const Decomposition& decomposition, const DeviceState& state,
                   HaloBuffers& buffers, cudaStream_t communication_stream,
                   cudaEvent_t halos_ready) {
    const int pitch = decomposition.pitch();
    const size_t row_bytes = static_cast<size_t>(decomposition.local_ny) * sizeof(val_t);
    const size_t low_row = static_cast<size_t>(kHaloWidth) * pitch + kHaloWidth;
    const size_t high_row =
        static_cast<size_t>(decomposition.local_nx) * pitch + kHaloWidth;
    const size_t low_column = static_cast<size_t>(kHaloWidth) * pitch + kHaloWidth;
    const size_t high_column = static_cast<size_t>(kHaloWidth) * pitch + decomposition.local_ny;

    // The boundary data and the interior kernel access disjoint cells, so these
    // device-to-host copies can overlap the interior update on capable GPUs.
    if (decomposition.x_lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.x_low_send, state.energy + low_row,
                                   row_bytes, cudaMemcpyDeviceToHost,
                                   communication_stream));
    }
    if (decomposition.x_higher != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.x_high_send, state.energy + high_row,
                                   row_bytes, cudaMemcpyDeviceToHost,
                                   communication_stream));
    }
    if (decomposition.y_lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.y_low_send, sizeof(val_t),
                                     state.energy + low_column,
                                     static_cast<size_t>(pitch) * sizeof(val_t),
                                     sizeof(val_t), decomposition.local_nx,
                                     cudaMemcpyDeviceToHost, communication_stream));
    }
    if (decomposition.y_higher != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(buffers.y_high_send, sizeof(val_t),
                                     state.energy + high_column,
                                     static_cast<size_t>(pitch) * sizeof(val_t),
                                     sizeof(val_t), decomposition.local_nx,
                                     cudaMemcpyDeviceToHost, communication_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(communication_stream));

    MPI_Request requests[8];
    int request_count = 0;
    if (decomposition.x_lower != MPI_PROC_NULL) {
        MPI_Irecv(buffers.x_low_recv, decomposition.local_ny, MPI_DOUBLE,
                  decomposition.x_lower, kTagXToHigher, decomposition.communicator,
                  &requests[request_count++]);
        MPI_Isend(buffers.x_low_send, decomposition.local_ny, MPI_DOUBLE,
                  decomposition.x_lower, kTagXToLower, decomposition.communicator,
                  &requests[request_count++]);
    }
    if (decomposition.x_higher != MPI_PROC_NULL) {
        MPI_Irecv(buffers.x_high_recv, decomposition.local_ny, MPI_DOUBLE,
                  decomposition.x_higher, kTagXToLower, decomposition.communicator,
                  &requests[request_count++]);
        MPI_Isend(buffers.x_high_send, decomposition.local_ny, MPI_DOUBLE,
                  decomposition.x_higher, kTagXToHigher, decomposition.communicator,
                  &requests[request_count++]);
    }
    if (decomposition.y_lower != MPI_PROC_NULL) {
        MPI_Irecv(buffers.y_low_recv, decomposition.local_nx, MPI_DOUBLE,
                  decomposition.y_lower, kTagYToHigher, decomposition.communicator,
                  &requests[request_count++]);
        MPI_Isend(buffers.y_low_send, decomposition.local_nx, MPI_DOUBLE,
                  decomposition.y_lower, kTagYToLower, decomposition.communicator,
                  &requests[request_count++]);
    }
    if (decomposition.y_higher != MPI_PROC_NULL) {
        MPI_Irecv(buffers.y_high_recv, decomposition.local_nx, MPI_DOUBLE,
                  decomposition.y_higher, kTagYToLower, decomposition.communicator,
                  &requests[request_count++]);
        MPI_Isend(buffers.y_high_send, decomposition.local_nx, MPI_DOUBLE,
                  decomposition.y_higher, kTagYToHigher, decomposition.communicator,
                  &requests[request_count++]);
    }
    if (request_count > 0) MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    if (decomposition.x_lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(state.energy + kHaloWidth, buffers.x_low_recv,
                                   row_bytes, cudaMemcpyHostToDevice,
                                   communication_stream));
    }
    if (decomposition.x_higher != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(state.energy +
                                       static_cast<size_t>(decomposition.local_nx + 1) * pitch +
                                       kHaloWidth,
                                   buffers.x_high_recv, row_bytes,
                                   cudaMemcpyHostToDevice, communication_stream));
    }
    if (decomposition.y_lower != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(state.energy + static_cast<size_t>(kHaloWidth) * pitch,
                                     static_cast<size_t>(pitch) * sizeof(val_t),
                                     buffers.y_low_recv, sizeof(val_t), sizeof(val_t),
                                     decomposition.local_nx, cudaMemcpyHostToDevice,
                                     communication_stream));
    }
    if (decomposition.y_higher != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy2DAsync(state.energy + static_cast<size_t>(kHaloWidth) * pitch +
                                         decomposition.local_ny + 1,
                                     static_cast<size_t>(pitch) * sizeof(val_t),
                                     buffers.y_high_recv, sizeof(val_t), sizeof(val_t),
                                     decomposition.local_nx, cudaMemcpyHostToDevice,
                                     communication_stream));
    }
    CUDA_CHECK(cudaEventRecord(halos_ready, communication_stream));

}

void runSimulation(const Decomposition& decomposition, DeviceState& state,
                   HaloBuffers& buffers, int n_iterations) {
    cudaStream_t compute_stream = nullptr;
    cudaStream_t communication_stream = nullptr;
    cudaEvent_t halos_ready = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready, cudaEventDisableTiming));

    for (int iteration = 0; iteration < n_iterations; ++iteration) {
        launchInterior(decomposition, state, compute_stream);
        exchangeHalos(decomposition, state, buffers, communication_stream, halos_ready);
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, halos_ready, 0));
        launchBoundary(decomposition, state, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        state.swap();
    }

    CUDA_CHECK(cudaEventDestroy(halos_ready));
    CUDA_CHECK(cudaStreamDestroy(communication_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
}

void copyInteriorToHost(const Decomposition& decomposition, const DeviceState& state,
                        val_t* host_energy, val_t* host_flux) {
    const size_t device_pitch = static_cast<size_t>(decomposition.pitch()) * sizeof(val_t);
    const size_t host_pitch = static_cast<size_t>(decomposition.local_ny) * sizeof(val_t);
    const size_t first_interior =
        static_cast<size_t>(kHaloWidth) * decomposition.pitch() + kHaloWidth;
    CUDA_CHECK(cudaMemcpy2D(host_energy, host_pitch, state.energy + first_interior,
                             device_pitch, host_pitch, decomposition.local_nx,
                             cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(host_flux, host_pitch, state.total_flux + first_interior,
                             device_pitch, host_pitch, decomposition.local_nx,
                             cudaMemcpyDeviceToHost));
}

uint64_t computeLocalHash(const Decomposition& decomposition,
                          const std::vector<val_t>& energy,
                          const std::vector<val_t>& total_flux) {
    uint64_t hash = 0;
#pragma omp parallel for reduction(^ : hash) schedule(static)
    for (int local_x = 0; local_x < decomposition.local_nx; ++local_x) {
        uint64_t row_hash = 0;
        for (int local_y = 0; local_y < decomposition.local_ny; ++local_y) {
            const size_t local_index =
                static_cast<size_t>(local_x) * decomposition.local_ny + local_y;
            const uint64_t global_index =
                static_cast<uint64_t>(decomposition.offset_x + local_x) *
                    decomposition.global_n +
                decomposition.offset_y + local_y;
            uint64_t energy_bits = 0;
            uint64_t flux_bits = 0;
            std::memcpy(&energy_bits, &energy[local_index], sizeof(energy_bits));
            std::memcpy(&flux_bits, &total_flux[local_index], sizeof(flux_bits));
            row_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            row_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
        hash ^= row_hash;
    }
    return hash;
}

bool validateResults(const Decomposition& decomposition,
                     const std::vector<val_t>& energy,
                     const std::vector<val_t>& total_flux) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    const size_t count = energy.size();

#pragma omp parallel for reduction(+ : local_energy_sum, local_flux_sum) \
    reduction(max : local_energy_max) reduction(min : local_energy_min) schedule(static)
    for (size_t i = 0; i < count; ++i) {
        local_energy_sum += energy[i];
        local_flux_sum += total_flux[i];
        local_energy_max = std::max(local_energy_max, energy[i]);
        local_energy_min = std::min(local_energy_min, energy[i]);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM,
                  decomposition.communicator);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM,
                  decomposition.communicator);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX,
                  decomposition.communicator);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN,
                  decomposition.communicator);

    if (decomposition.rank != 0) return true;

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum) || !std::isfinite(energy_max) ||
        !std::isfinite(energy_min)) {
        std::printf("  ERROR: Flux results are not finite\n");
        return false;
    }
    std::printf("  Validation: PASSED\n");
    return true;
}

std::vector<val_t> gatherEnergy(const Decomposition& decomposition,
                                const std::vector<val_t>& local_energy) {
    const int local_count = static_cast<int>(local_energy.size());
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<int> metadata;
    if (decomposition.rank == 0) {
        counts.resize(decomposition.size);
        displacements.resize(decomposition.size);
        metadata.resize(static_cast<size_t>(decomposition.size) * 4);
    }

    const int local_metadata[4] = {decomposition.offset_x, decomposition.offset_y,
                                   decomposition.local_nx, decomposition.local_ny};
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               decomposition.communicator);
    MPI_Gather(local_metadata, 4, MPI_INT, metadata.data(), 4, MPI_INT, 0,
               decomposition.communicator);

    int gathered_count = 0;
    if (decomposition.rank == 0) {
        for (int rank = 0; rank < decomposition.size; ++rank) {
            displacements[rank] = gathered_count;
            gathered_count += counts[rank];
        }
    }
    std::vector<val_t> packed(decomposition.rank == 0 ? gathered_count : 0);
    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, packed.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0,
                decomposition.communicator);

    if (decomposition.rank != 0) return {};

    const size_t global_count =
        static_cast<size_t>(decomposition.global_n) * decomposition.global_n;
    std::vector<val_t> global_energy(global_count);
#pragma omp parallel for schedule(static)
    for (int rank = 0; rank < decomposition.size; ++rank) {
        const int offset_x = metadata[4 * rank];
        const int offset_y = metadata[4 * rank + 1];
        const int local_nx = metadata[4 * rank + 2];
        const int local_ny = metadata[4 * rank + 3];
        for (int local_x = 0; local_x < local_nx; ++local_x) {
            const val_t* source = packed.data() + displacements[rank] +
                                  static_cast<size_t>(local_x) * local_ny;
            val_t* destination = global_energy.data() +
                                 static_cast<size_t>(offset_x + local_x) *
                                     decomposition.global_n +
                                 offset_y;
            std::memcpy(destination, source, static_cast<size_t>(local_ny) * sizeof(val_t));
        }
    }
    return global_energy;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n_elems_root = 512;
    int n_iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool parse_ok = true;
    bool show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            parse_ok = false;
            if (world_rank == 0) std::printf("Unknown option: %s\n", argv[i]);
        }
    }

    if (show_help || !parse_ok || n_elems_root <= 0 || n_iterations < 0 ||
        static_cast<uint64_t>(n_elems_root) * n_elems_root >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (world_rank == 0) {
            if (n_elems_root <= 0 || n_iterations < 0) {
                std::printf("Grid size must be positive and iterations must be non-negative\n");
            } else if (static_cast<uint64_t>(n_elems_root) * n_elems_root >
                       static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                std::printf("Grid contains too many elements for MPI count arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (show_help && parse_ok) ? 0 : 1;
    }

    int requested_ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &requested_ranks);
    const int active_ranks = activeRankCount(requested_ranks, n_elems_root);
    MPI_Comm active_communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &active_communicator);
    if (active_communicator == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    Decomposition decomposition = makeDecomposition(active_communicator, n_elems_root);
    MPI_Comm_free(&active_communicator);

    selectDevice(decomposition);
    const size_t local_count =
        static_cast<size_t>(decomposition.local_nx) * decomposition.local_ny;
    const size_t global_count =
        static_cast<size_t>(n_elems_root) * n_elems_root;

    if (decomposition.rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root,
                    n_elems_root, global_count);
        std::printf("Iterations: %d\n", n_iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel execution: %d MPI ranks, Cartesian grid %dx%d, up to %d OpenMP threads/rank, CUDA enabled\n\n",
                    decomposition.size, decomposition.dimensions[0],
                    decomposition.dimensions[1], omp_get_max_threads());
        std::printf("Building distributed unstructured mesh...\n");
        const size_t static_memory = global_count * (sizeof(uint64_t) * (2 + 8) +
                                                     sizeof(val_t) * 8);
        const size_t dynamic_memory = global_count * sizeof(val_t) * 4;
        std::printf("Global memory footprint: %.2f MB (static-equivalent: %.2f MB, dynamic: %.2f MB)\n\n",
                    (static_memory + dynamic_memory) / (1024.0 * 1024.0),
                    static_memory / (1024.0 * 1024.0),
                    dynamic_memory / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    DeviceState state;
    HaloBuffers buffers;
    state.allocate(decomposition.elementCountWithHalo());
    buffers.allocate(decomposition);

    MPI_Barrier(decomposition.communicator);
    const auto start = std::chrono::steady_clock::now();
    runSimulation(decomposition, state, buffers, n_iterations);
    const auto end = std::chrono::steady_clock::now();
    const double local_milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double duration_milliseconds = 0.0;
    MPI_Reduce(&local_milliseconds, &duration_milliseconds, 1, MPI_DOUBLE, MPI_MAX,
               0, decomposition.communicator);

    std::vector<val_t> local_energy(local_count);
    std::vector<val_t> local_flux(local_count);
    copyInteriorToHost(decomposition, state, local_energy.data(), local_flux.data());

    const uint64_t local_hash = computeLocalHash(decomposition, local_energy, local_flux);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               decomposition.communicator);

    if (decomposition.rank == 0) {
        const int measured_iterations = std::max(n_iterations - 1, 1);
        const double safe_duration = std::max(duration_milliseconds, 1.0e-12);
        const double time_per_iteration = safe_duration / measured_iterations;
        const double giga_elements_per_second =
            (static_cast<double>(measured_iterations) * global_count) /
            (safe_duration / 1000.0) / 1.0e9;
        std::printf("Computation time: %.3f ms\n", duration_milliseconds);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(result_hash));
    }

    if (print_results_requested) {
        const std::vector<val_t> global_energy = gatherEnergy(decomposition, local_energy);
        if (decomposition.rank == 0) print_results(global_energy, "ElementEnergy");
    }

    const bool valid = !validate || validateResults(decomposition, local_energy, local_flux);

    buffers.release();
    state.release();
    MPI_Comm_free(&decomposition.communicator);
    MPI_Finalize();
    return valid ? 0 : 1;
}
