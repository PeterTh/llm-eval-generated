#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
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

using idx_t = uint64_t;
using val_t = double;

static void cudaCheck(cudaError_t error, const char* operation, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: CUDA error at line %d (%s): %s\n",
                     rank, line, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

static void mpiCheck(int error, const char* operation, int line) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0, rank = 0;
        MPI_Error_string(error, message, &length);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: MPI error at line %d (%s): %.*s\n",
                     rank, line, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation, __LINE__)
#define MPI_CHECK(operation) mpiCheck((operation), #operation, __LINE__)

static size_t bytesFor(size_t count, size_t element_size) {
    if (count > std::numeric_limits<size_t>::max() / element_size) {
        std::fprintf(stderr, "Requested mesh allocation is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    return count * element_size;
}

struct Partition {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[2] = {1, 1};
    int x0 = 0, y0 = 0, height = 0, width = 0;
    // -x, +x, -y, +y; opposite directions are side ^ 1.
    int neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
};

static void splitRange(int n, int parts, int coordinate, int& start, int& count) {
    count = n / parts + (coordinate < n % parts);
    start = coordinate * (n / parts) + std::min(coordinate, n % parts);
}

// Choose a balanced Cartesian grid with no empty partitions. Extra ranks wait
// in MPI_Finalize when a very small mesh cannot use all launched processes.
static std::array<int, 2> processGrid(int n, int ranks) {
    std::array<int, 2> best = {1, 1};
    int best_count = 1, best_difference = INT_MAX;
    for (int x = 1; x <= std::min(n, ranks); ++x) {
        const int y = std::min(n, ranks / x);
        const int count = x * y;
        const int difference = std::abs(x - y);
        if (count > best_count || (count == best_count && difference < best_difference)) {
            best = {x, y};
            best_count = count;
            best_difference = difference;
        }
    }
    return best;
}

static bool cudaAwareMPI() {
    int supported = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    supported = MPIX_Query_cuda_support();
#endif
    // Other MPI implementations do not have a portable capability query.
    // Set this to 1 only when that MPI runtime supports device buffers.
    if (const char* value = std::getenv("UNSTRUCTURED_CUDA_AWARE_MPI")) {
        supported = std::strcmp(value, "1") == 0;
    }
    return supported != 0;
}

struct World {
    Partition partition;
    int grid_size;
    int host_threads;
    size_t count, pitch, padded_count, halo_count;
    size_t offsets[4];
    int lengths[4];
    int neighbor_mask = 0;
    bool cuda_aware;
    int current = 0;
    // Structure of arrays: only energy is exchanged; cumulative flux is local.
    val_t* energy[2] = {nullptr, nullptr};
    val_t* flux = nullptr;
    unsigned char* material = nullptr;
    val_t *device_send = nullptr, *device_recv = nullptr;
    val_t *host_send = nullptr, *host_recv = nullptr;
    cudaStream_t compute_stream = nullptr, exchange_stream = nullptr;
    cudaEvent_t step_done = nullptr, send_ready = nullptr, halo_ready = nullptr;
    MPI_Request receives[4], sends[4];
    int receive_count = 0, send_count = 0;
    std::vector<val_t> host_energy, host_flux;

    World(const Partition& p, int n, bool direct)
        : partition(p), grid_size(n),
          count(static_cast<size_t>(p.height) * p.width),
          pitch((static_cast<size_t>(p.width) + 2 + 31) / 32 * 32),
          padded_count((static_cast<size_t>(p.height) + 2) * pitch),
          halo_count(2 * (static_cast<size_t>(p.width) + p.height)),
          cuda_aware(direct) {
        int max_threads = omp_get_max_threads();
        if (!std::getenv("OMP_NUM_THREADS")) max_threads = std::min(max_threads, 32);
        host_threads = static_cast<int>(std::max<size_t>(1,
            std::min<size_t>(max_threads, (count + 32767) / 32768)));
        offsets[0] = 0;
        offsets[1] = p.width;
        offsets[2] = 2 * static_cast<size_t>(p.width);
        offsets[3] = offsets[2] + p.height;
        lengths[0] = lengths[1] = p.width;
        lengths[2] = lengths[3] = p.height;
        for (int side = 0; side < 4; ++side) {
            receives[side] = sends[side] = MPI_REQUEST_NULL;
            if (p.neighbors[side] != MPI_PROC_NULL) neighbor_mask |= 1 << side;
        }
    }

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    ~World() {
        for (int i = 0; i < receive_count; ++i) MPI_Request_free(&receives[i]);
        for (int i = 0; i < send_count; ++i) MPI_Request_free(&sends[i]);
        if (step_done) cudaEventDestroy(step_done);
        if (send_ready) cudaEventDestroy(send_ready);
        if (halo_ready) cudaEventDestroy(halo_ready);
        if (compute_stream) cudaStreamDestroy(compute_stream);
        if (exchange_stream) cudaStreamDestroy(exchange_stream);
        for (auto pointer : energy) if (pointer) cudaFree(pointer);
        if (flux) cudaFree(flux);
        if (material) cudaFree(material);
        if (device_send) cudaFree(device_send);
        if (device_recv) cudaFree(device_recv);
        if (host_send) cudaFreeHost(host_send);
        if (host_recv) cudaFreeHost(host_recv);
    }
};

// The original builder only creates a four-neighbor square mesh with unit
// connection weights and coefficient 0.8. Its connectivity is therefore
// implicit in the tile and its one-cell halo, rather than a large adjacency
// record per cell. Material IDs retain the original corner assignment order.
static void buildSquare2D(World& world) {
    const auto& p = world.partition;
    std::vector<unsigned char> materials(world.count);
#pragma omp parallel for schedule(static) num_threads(world.host_threads)
    for (size_t i = 0; i < world.count; ++i) {
        const int x = p.x0 + static_cast<int>(i / p.width);
        const int y = p.y0 + static_cast<int>(i % p.width);
        unsigned char id = 0;
        if ((x == 0 || x == world.grid_size - 1) &&
            (y == 0 || y == world.grid_size - 1)) {
            id = (x == y) ? 1 : 2;
        }
        materials[i] = id;
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&world.compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&world.exchange_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&world.step_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&world.send_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&world.halo_ready, cudaEventDisableTiming));
    const size_t energy_bytes = bytesFor(world.padded_count, sizeof(val_t));
    for (auto& pointer : world.energy) {
        CUDA_CHECK(cudaMalloc(&pointer, energy_bytes));
        CUDA_CHECK(cudaMemsetAsync(pointer, 0, energy_bytes, world.compute_stream));
    }
    CUDA_CHECK(cudaMalloc(&world.flux, bytesFor(world.count, sizeof(val_t))));
    CUDA_CHECK(cudaMemsetAsync(world.flux, 0, bytesFor(world.count, sizeof(val_t)),
                              world.compute_stream));
    CUDA_CHECK(cudaMalloc(&world.material, world.count));
    CUDA_CHECK(cudaMemcpyAsync(world.material, materials.data(), world.count,
                              cudaMemcpyHostToDevice, world.compute_stream));

    if (world.neighbor_mask) {
        const size_t halo_bytes = bytesFor(world.halo_count, sizeof(val_t));
        CUDA_CHECK(cudaMalloc(&world.device_send, halo_bytes));
        CUDA_CHECK(cudaMalloc(&world.device_recv, halo_bytes));
        if (!world.cuda_aware) {
            CUDA_CHECK(cudaMallocHost(&world.host_send, halo_bytes));
            CUDA_CHECK(cudaMallocHost(&world.host_recv, halo_bytes));
            std::memset(world.host_recv, 0, halo_bytes);
        }
        val_t* send_buffer = world.cuda_aware ? world.device_send : world.host_send;
        val_t* recv_buffer = world.cuda_aware ? world.device_recv : world.host_recv;
        // Fixed buffers allow persistent MPI requests to be reused every step.
        for (int side = 0; side < 4; ++side) {
            if (p.neighbors[side] == MPI_PROC_NULL) continue;
            MPI_CHECK(MPI_Recv_init(recv_buffer + world.offsets[side], world.lengths[side],
                      MPI_DOUBLE, p.neighbors[side], side ^ 1, p.comm,
                      &world.receives[world.receive_count++]));
            MPI_CHECK(MPI_Send_init(send_buffer + world.offsets[side], world.lengths[side],
                      MPI_DOUBLE, p.neighbors[side], side, p.comm,
                      &world.sends[world.send_count++]));
        }
    }
    CUDA_CHECK(cudaEventRecord(world.step_done, world.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(world.compute_stream));
}

// Keep the original subtraction, multiplication and accumulation order.
// Folding 0.8 * 0.25 into 0.2 or fusing the accumulation changes rounding.
__device__ __forceinline__ val_t computeFlux(val_t center, val_t other) {
    return (other - center) * 0.8 * 1.0 * 0.25;
}

template <bool CheckEdges>
__device__ __forceinline__ void updateElement(
    const val_t* __restrict__ input, val_t* __restrict__ output,
    val_t* __restrict__ accumulated_flux, const unsigned char* __restrict__ materials,
    size_t pitch, int width, int x, int y, int x0, int y0, int n) {
    const size_t dense = static_cast<size_t>(x) * width + y;
    const size_t index = (static_cast<size_t>(x) + 1) * pitch + y + 1;
    const val_t center = input[index];
    val_t total = 0.0;
    if (CheckEdges) {
        const unsigned char material = materials[dense];
        total = material == 1 ? 0.5 : (material == 2 ? -0.5 : 0.0);
    }
    // Same neighbor order as the original: +x, -x, +y, -y.
    if (!CheckEdges || x0 + x < n - 1) total += computeFlux(center, input[index + pitch]);
    if (!CheckEdges || x0 + x > 0) total += computeFlux(center, input[index - pitch]);
    if (!CheckEdges || y0 + y < n - 1) total += computeFlux(center, input[index + 1]);
    if (!CheckEdges || y0 + y > 0) total += computeFlux(center, input[index - 1]);
    output[index] = center + total;
    accumulated_flux[dense] += fabs(total);
}

template <bool Interior>
__global__ void updateTile(const val_t* __restrict__ input, val_t* __restrict__ output,
                           val_t* __restrict__ flux, const unsigned char* __restrict__ materials,
                           size_t pitch, int height, int width, int x0, int y0, int n) {
    const size_t y = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + (Interior ? 1 : 0);
    const size_t end_x = height - (Interior ? 1 : 0);
    if (y >= static_cast<size_t>(width - (Interior ? 1 : 0))) return;
    for (size_t x = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + (Interior ? 1 : 0);
         x < end_x; x += static_cast<size_t>(blockDim.y) * gridDim.y) {
        updateElement<!Interior>(input, output, flux, materials, pitch, width,
                                 static_cast<int>(x), static_cast<int>(y), x0, y0, n);
    }
}

__global__ void updateBoundary(const val_t* __restrict__ input, val_t* __restrict__ output,
                               val_t* __restrict__ flux, const unsigned char* __restrict__ materials,
                               size_t pitch, int height, int width, int x0, int y0, int n,
                               size_t perimeter) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < perimeter; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int x, y;
        if (height == 1) {
            x = 0; y = static_cast<int>(i);
        } else if (width == 1) {
            x = static_cast<int>(i); y = 0;
        } else if (i < 2 * static_cast<size_t>(width)) {
            x = i < static_cast<size_t>(width) ? 0 : height - 1;
            y = static_cast<int>(i % width);
        } else {
            const size_t j = i - 2 * static_cast<size_t>(width);
            x = static_cast<int>(j / 2) + 1;
            y = (j & 1) ? width - 1 : 0;
        }
        updateElement<true>(input, output, flux, materials, pitch, width, x, y, x0, y0, n);
    }
}

__global__ void packHalo(const val_t* __restrict__ energy, val_t* __restrict__ buffer,
                         size_t pitch, int height, int width, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t index;
        if (i < static_cast<size_t>(width)) index = pitch + i + 1;
        else if (i < 2 * static_cast<size_t>(width)) index = static_cast<size_t>(height) * pitch + i - width + 1;
        else if (i < 2 * static_cast<size_t>(width) + height) index = (i - 2 * static_cast<size_t>(width) + 1) * pitch + 1;
        else index = (i - 2 * static_cast<size_t>(width) - height + 1) * pitch + width;
        buffer[i] = energy[index];
    }
}

__global__ void unpackHalo(val_t* __restrict__ energy, const val_t* __restrict__ buffer,
                           size_t pitch, int height, int width, size_t count, int mask) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t index;
        int side;
        if (i < static_cast<size_t>(width)) {
            side = 0; index = i + 1;
        } else if (i < 2 * static_cast<size_t>(width)) {
            side = 1; index = (static_cast<size_t>(height) + 1) * pitch + i - width + 1;
        } else if (i < 2 * static_cast<size_t>(width) + height) {
            side = 2; index = (i - 2 * static_cast<size_t>(width) + 1) * pitch;
        } else {
            side = 3; index = (i - 2 * static_cast<size_t>(width) - height + 1) * pitch + width + 1;
        }
        if (mask & (1 << side)) energy[index] = buffer[i];
    }
}

static unsigned int linearBlocks(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
}

static dim3 tileBlocks(int height, int width) {
    return dim3(static_cast<unsigned int>((static_cast<size_t>(width) + 31) / 32),
                static_cast<unsigned int>(std::min<size_t>((static_cast<size_t>(height) + 7) / 8, 65535)));
}

static void runSimulation(World& world, int iterations) {
    const auto& p = world.partition;
    const dim3 threads(32, 8);
    const size_t perimeter = p.height == 1 ? p.width : (p.width == 1 ? p.height :
        2 * static_cast<size_t>(p.width) + 2 * static_cast<size_t>(p.height - 2));

    for (int iteration = 0; iteration < iterations; ++iteration) {
        val_t* input = world.energy[world.current];
        val_t* output = world.energy[world.current ^ 1];
        if (!world.neighbor_mask) {
            updateTile<false><<<tileBlocks(p.height, p.width), threads, 0, world.compute_stream>>>(
                input, output, world.flux, world.material, world.pitch,
                p.height, p.width, p.x0, p.y0, world.grid_size);
        } else {
            MPI_CHECK(MPI_Startall(world.receive_count, world.receives));
            CUDA_CHECK(cudaStreamWaitEvent(world.exchange_stream, world.step_done, 0));
            packHalo<<<linearBlocks(world.halo_count), 256, 0, world.exchange_stream>>>(
                input, world.device_send, world.pitch, p.height, p.width, world.halo_count);
            if (!world.cuda_aware) {
                CUDA_CHECK(cudaMemcpyAsync(world.host_send, world.device_send,
                    bytesFor(world.halo_count, sizeof(val_t)), cudaMemcpyDeviceToHost, world.exchange_stream));
            }
            CUDA_CHECK(cudaEventRecord(world.send_ready, world.exchange_stream));

            // These cells never read a halo. CUDA runs them while MPI progresses
            // on the main thread; all MPI calls obey MPI_THREAD_FUNNELED.
            if (p.height > 2 && p.width > 2) {
                updateTile<true><<<tileBlocks(p.height - 2, p.width - 2), threads, 0, world.compute_stream>>>(
                    input, output, world.flux, world.material, world.pitch,
                    p.height, p.width, p.x0, p.y0, world.grid_size);
            }
            CUDA_CHECK(cudaEventSynchronize(world.send_ready));
            MPI_CHECK(MPI_Startall(world.send_count, world.sends));
            MPI_CHECK(MPI_Waitall(world.receive_count, world.receives, MPI_STATUSES_IGNORE));
            if (!world.cuda_aware) {
                CUDA_CHECK(cudaMemcpyAsync(world.device_recv, world.host_recv,
                    bytesFor(world.halo_count, sizeof(val_t)), cudaMemcpyHostToDevice, world.exchange_stream));
            }
            unpackHalo<<<linearBlocks(world.halo_count), 256, 0, world.exchange_stream>>>(
                input, world.device_recv, world.pitch, p.height, p.width, world.halo_count, world.neighbor_mask);
            CUDA_CHECK(cudaEventRecord(world.halo_ready, world.exchange_stream));
            CUDA_CHECK(cudaStreamWaitEvent(world.compute_stream, world.halo_ready, 0));
            updateBoundary<<<linearBlocks(perimeter), 256, 0, world.compute_stream>>>(
                input, output, world.flux, world.material, world.pitch,
                p.height, p.width, p.x0, p.y0, world.grid_size, perimeter);
            CUDA_CHECK(cudaEventRecord(world.step_done, world.compute_stream));
            MPI_CHECK(MPI_Waitall(world.send_count, world.sends, MPI_STATUSES_IGNORE));
            // Receives may be restarted only after unpack has consumed their
            // buffers. The boundary update can still run during the next step.
            CUDA_CHECK(cudaEventSynchronize(world.halo_ready));
        }
        CUDA_CHECK(cudaGetLastError());
        world.current ^= 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(world.compute_stream));
}

static void downloadResults(World& world) {
    world.host_energy.resize(world.count);
    world.host_flux.resize(world.count);
    CUDA_CHECK(cudaMemcpy2D(world.host_energy.data(), bytesFor(world.partition.width, sizeof(val_t)),
        world.energy[world.current] + world.pitch + 1, bytesFor(world.pitch, sizeof(val_t)),
        bytesFor(world.partition.width, sizeof(val_t)), world.partition.height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(world.host_flux.data(), world.flux, bytesFor(world.count, sizeof(val_t)),
                          cudaMemcpyDeviceToHost));
}

static uint64_t computeHash(const World& world) {
    uint64_t result = 0;
    const auto& p = world.partition;
#pragma omp parallel for schedule(static) num_threads(world.host_threads) reduction(^:result)
    for (size_t i = 0; i < world.count; ++i) {
        uint64_t energy_bits, flux_bits;
        // memcpy avoids the original type-punning/strict-aliasing violation.
        std::memcpy(&energy_bits, &world.host_energy[i], sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.host_flux[i], sizeof(flux_bits));
        const idx_t global = (static_cast<idx_t>(p.x0) + i / p.width) * world.grid_size + p.y0 + i % p.width;
        result ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global = 0;
    MPI_CHECK(MPI_Reduce(&result, &global, 1, MPI_UINT64_T, MPI_BXOR, 0, p.comm));
    return global;
}

static bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
#pragma omp parallel for schedule(static) num_threads(world.host_threads) \
    reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.count; ++i) {
        energy_sum += world.host_energy[i];
        flux_sum += world.host_flux[i];
        energy_max = std::max(energy_max, world.host_energy[i]);
        energy_min = std::min(energy_min, world.host_energy[i]);
    }
    val_t local_sums[2] = {energy_sum, flux_sum}, sums[2], maximum = 0.0, minimum = 0.0;
    const auto& p = world.partition;
    MPI_CHECK(MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, p.comm));
    MPI_CHECK(MPI_Reduce(&energy_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, p.comm));
    MPI_CHECK(MPI_Reduce(&energy_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, p.comm));
    int valid = 1;
    if (p.rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n  Flux sum: %.2f\n", sums[0], sums[1]);
        std::printf("  Energy range: [%.6f, %.6f]\n", minimum, maximum);
        if (!std::isfinite(sums[0])) {
            std::printf("  ERROR: Energy sum is not finite\n"); valid = 0;
        } else if (std::abs(sums[0]) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(sums[1])) {
            std::printf("  ERROR: Flux sum is not finite\n"); valid = 0;
        }
        if (!std::isfinite(maximum) || !std::isfinite(minimum)) {
            std::printf("  ERROR: Energy extrema are not finite\n"); valid = 0;
        }
        if (valid) std::printf("  Validation: PASSED\n");
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, p.comm));
    return valid != 0;
}

// Only -r gathers the complete energy field, in original global row order.
// Derived datatypes avoid int-sized total element counts and displacements.
static void printResults(const World& world) {
    const auto& p = world.partition;
    if (p.rank != 0) {
        MPI_Datatype row, tile;
        MPI_CHECK(MPI_Type_contiguous(p.width, MPI_DOUBLE, &row));
        MPI_CHECK(MPI_Type_contiguous(p.height, row, &tile));
        MPI_CHECK(MPI_Type_commit(&tile));
        MPI_CHECK(MPI_Send(world.host_energy.data(), 1, tile, 0, 10, p.comm));
        MPI_CHECK(MPI_Type_free(&tile));
        MPI_CHECK(MPI_Type_free(&row));
        return;
    }
    const size_t total = static_cast<size_t>(world.grid_size) * world.grid_size;
    bytesFor(total, sizeof(val_t));
    std::vector<val_t> energies(total);
#pragma omp parallel for schedule(static) num_threads(world.host_threads)
    for (int x = 0; x < p.height; ++x) {
        std::copy_n(world.host_energy.data() + static_cast<size_t>(x) * p.width, p.width,
                    energies.data() + static_cast<size_t>(p.x0 + x) * world.grid_size + p.y0);
    }
    for (int peer = 1; peer < p.size; ++peer) {
        int coordinates[2], starts[2], sizes[2];
        MPI_CHECK(MPI_Cart_coords(p.comm, peer, 2, coordinates));
        splitRange(world.grid_size, p.dims[0], coordinates[0], starts[0], sizes[0]);
        splitRange(world.grid_size, p.dims[1], coordinates[1], starts[1], sizes[1]);
        const int global_sizes[2] = {world.grid_size, world.grid_size};
        MPI_Datatype tile;
        MPI_CHECK(MPI_Type_create_subarray(2, global_sizes, sizes, starts, MPI_ORDER_C, MPI_DOUBLE, &tile));
        MPI_CHECK(MPI_Type_commit(&tile));
        MPI_CHECK(MPI_Recv(energies.data(), 1, tile, peer, 10, p.comm, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Type_free(&tile));
    }
    print_results(energies, "ElementEnergy");
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\nOptions:\n", program);
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 0 || parsed > INT_MAX) return false;
    value = static_cast<int>(parsed);
    return true;
}

static int parseOptions(int argc, char** argv, int options[4]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "-i") == 0) {
            const int option = std::strcmp(argv[i], "-n") == 0 ? 0 : 1;
            if (++i == argc || !parseInteger(argv[i], options[option]) ||
                (option == 0 && options[0] == 0)) {
                std::fprintf(stderr, "Invalid grid size or iteration count\n");
                printUsage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) options[2] = 1;
        else if (std::strcmp(argv[i], "-r") == 0) options[3] = 1;
        else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 2;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int options[4] = {512, 10, 0, 0};
    int parse_status = rank == 0 ? parseOptions(argc, argv, options) : 0;
    MPI_CHECK(MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD));
    if (parse_status) {
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    int exit_status = 0;
    try {
        const int n = options[0], iterations = options[1];
        const auto dims = processGrid(n, ranks);
        MPI_Comm active = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED,
                                rank, &active));
        if (active != MPI_COMM_NULL) {
            Partition p;
            p.dims[0] = dims[0]; p.dims[1] = dims[1];
            const int periods[2] = {0, 0};
            MPI_CHECK(MPI_Cart_create(active, 2, p.dims, periods, 0, &p.comm));
            MPI_CHECK(MPI_Comm_set_errhandler(p.comm, MPI_ERRORS_RETURN));
            MPI_CHECK(MPI_Comm_free(&active));
            MPI_CHECK(MPI_Comm_rank(p.comm, &p.rank));
            MPI_CHECK(MPI_Comm_size(p.comm, &p.size));
            int coordinates[2];
            MPI_CHECK(MPI_Cart_coords(p.comm, p.rank, 2, coordinates));
            splitRange(n, p.dims[0], coordinates[0], p.x0, p.height);
            splitRange(n, p.dims[1], coordinates[1], p.y0, p.width);
            MPI_CHECK(MPI_Cart_shift(p.comm, 0, 1, &p.neighbors[0], &p.neighbors[1]));
            MPI_CHECK(MPI_Cart_shift(p.comm, 1, 1, &p.neighbors[2], &p.neighbors[3]));

            MPI_Comm shared;
            MPI_CHECK(MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, p.rank, MPI_INFO_NULL, &shared));
            int local_rank = 0, devices = 0;
            MPI_CHECK(MPI_Comm_rank(shared, &local_rank));
            CUDA_CHECK(cudaGetDeviceCount(&devices));
            if (devices == 0) {
                std::fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            CUDA_CHECK(cudaSetDevice(local_rank % devices));
            MPI_CHECK(MPI_Comm_free(&shared));
            const int local_direct = cudaAwareMPI() ? 1 : 0;
            int direct = 0;
            MPI_CHECK(MPI_Allreduce(&local_direct, &direct, 1, MPI_INT, MPI_MIN, p.comm));

            {
                World world(p, n, direct != 0);
                const idx_t total_elements = static_cast<idx_t>(n) * n;
                if (p.rank == 0) {
                    std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
                    std::printf("============================================\n");
                    std::printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, total_elements);
                    std::printf("Iterations: %d\nValidation: %s\n", iterations, options[2] ? "enabled" : "disabled");
                    std::printf("Parallelism: %d MPI ranks (%d x %d), up to %d OpenMP threads/rank, CUDA\n",
                                p.size, p.dims[0], p.dims[1], world.host_threads);
                    std::printf("Halo transport: %s\n\n", direct ? "CUDA-aware MPI" : "pinned host buffers");
                    std::printf("Building unstructured mesh...\n");
                }
                buildSquare2D(world);
                const double static_memory = static_cast<double>(total_elements);
                const double local_dynamic = static_cast<double>(
                    2 * bytesFor(world.padded_count, sizeof(val_t)) + bytesFor(world.count, sizeof(val_t))) +
                    (world.neighbor_mask ? (direct ? 2.0 : 4.0) * bytesFor(world.halo_count, sizeof(val_t)) : 0.0);
                double dynamic_memory = 0.0;
                MPI_CHECK(MPI_Reduce(&local_dynamic, &dynamic_memory, 1, MPI_DOUBLE, MPI_SUM, 0, p.comm));
                if (p.rank == 0) {
                    constexpr double mb = 1024.0 * 1024.0;
                    std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                                (static_memory + dynamic_memory) / mb, static_memory / mb, dynamic_memory / mb);
                    std::printf("Running simulation...\n");
                }
                MPI_CHECK(MPI_Barrier(p.comm));
                const double start = MPI_Wtime();
                runSimulation(world, iterations);
                const double elapsed = MPI_Wtime() - start;
                double duration = 0.0;
                MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, p.comm));
                downloadResults(world);
                const uint64_t result_hash = computeHash(world);
                if (p.rank == 0) {
                    const double giga_elements = duration > 0.0 ?
                        static_cast<double>(iterations) * total_elements / duration / 1e9 : 0.0;
                    std::printf("Computation time: %.3f ms\nPerformance:\n", duration * 1000.0);
                    std::printf("  Time per iteration: %.4f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
                    std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements);
                    std::printf("  Performance: %.4f GFLOPS\n", giga_elements * 22.0);
                    std::printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
                }
                if (options[3]) printResults(world);
                if (options[2] && !validateResults(world)) exit_status = 1;
            }
            MPI_CHECK(MPI_Comm_free(&p.comm));
        }
        // Include inactive ranks in the final program status.
        int global_status = 0;
        MPI_CHECK(MPI_Allreduce(&exit_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        exit_status = global_status;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_status = 1;
    }
    MPI_Finalize();
    return exit_status;
}
