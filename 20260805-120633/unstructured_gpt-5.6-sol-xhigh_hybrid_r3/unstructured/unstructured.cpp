#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

// These are retained to report the logical mesh footprint of the original
// unstructured representation.  The generated square mesh is regular, so the
// accelerator kernel reconstructs its four connections instead of loading
// this comparatively large structure from global memory.
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct Decomposition {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[2] = {0, 0};
    int coords[2] = {0, 0};
    int x_start = 0;
    int y_start = 0;
    int nx = 0;
    int ny = 0;
    int x_minus = MPI_PROC_NULL;
    int x_plus = MPI_PROC_NULL;
    int y_minus = MPI_PROC_NULL;
    int y_plus = MPI_PROC_NULL;
};

struct DeviceDomain {
    val_t* energy = nullptr;
    val_t* energy_swap = nullptr;
    val_t* total_flux = nullptr;
    val_t* total_flux_swap = nullptr;
    val_t* send_left = nullptr;
    val_t* send_right = nullptr;
    val_t* recv_left = nullptr;
    val_t* recv_right = nullptr;
    val_t* host_halos = nullptr;
    val_t* host_send_top = nullptr;
    val_t* host_send_bottom = nullptr;
    val_t* host_recv_top = nullptr;
    val_t* host_recv_bottom = nullptr;
    val_t* host_send_left = nullptr;
    val_t* host_send_right = nullptr;
    val_t* host_recv_left = nullptr;
    val_t* host_recv_right = nullptr;
    cudaStream_t interior_stream = nullptr;
    cudaStream_t boundary_stream = nullptr;
    cudaEvent_t interior_ready[2] = {nullptr, nullptr};
    size_t pitch = 0;
    size_t allocation_elements = 0;
    bool cuda_aware_mpi = false;
};

static int world_rank = 0;

[[noreturn]] void abortWithMessage(const char* kind, const char* expression,
                                   const char* detail, const char* file, int line) {
    std::fprintf(stderr, "Rank %d: %s failure at %s:%d: %s (%s)\n",
                 world_rank, kind, file, line, expression, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        abortWithMessage("CUDA", expression, cudaGetErrorString(error), file, line);
    }
}

void checkMpi(int error, const char* expression, const char* file, int line) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(error, message, &length);
        abortWithMessage("MPI", expression, message, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

__device__ __forceinline__ val_t externalFlow(int global_x, int global_y,
                                              int n_elems_root) {
    const int last = n_elems_root - 1;
    if ((global_x == 0 && global_y == 0) ||
        (global_x == last && global_y == last)) {
        return 0.5;
    }
    if ((global_x == 0 && global_y == last) ||
        (global_x == last && global_y == 0)) {
        return -0.5;
    }
    return 0.0;
}

__device__ __forceinline__ val_t computeFluxDevice(val_t center,
                                                   val_t neighbor) {
    // Explicit round-to-nearest operations preserve the original CPU
    // expression's operation boundaries; in particular, CUDA must not fuse
    // the final multiply with the running flux sum.
    const val_t difference = __dsub_rn(neighbor, center);
    const val_t transferred = __dmul_rn(difference, 0.8);
    return __dmul_rn(transferred, 0.25);
}

__device__ __forceinline__ void updatePoint(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ accumulated_flux,
    val_t* __restrict__ energy_swap,
    val_t* __restrict__ accumulated_flux_swap,
    val_t* __restrict__ send_left,
    val_t* __restrict__ send_right,
    int local_x, int local_y, int nx, int ny, int pitch,
    int x_start, int y_start, int n_elems_root) {
    const int global_x = x_start + local_x - 1;
    const int global_y = y_start + local_y - 1;
    const size_t index = static_cast<size_t>(local_x) * pitch + local_y;
    const val_t center = energy[index];
    val_t flux = externalFlow(global_x, global_y, n_elems_root);

    // Keep the original connection order: +x, -x, +y, -y.
    if (global_x + 1 < n_elems_root) {
        flux = __dadd_rn(flux, computeFluxDevice(center, energy[index + pitch]));
    }
    if (global_x > 0) {
        flux = __dadd_rn(flux, computeFluxDevice(center, energy[index - pitch]));
    }
    if (global_y + 1 < n_elems_root) {
        flux = __dadd_rn(flux, computeFluxDevice(center, energy[index + 1]));
    }
    if (global_y > 0) {
        flux = __dadd_rn(flux, computeFluxDevice(center, energy[index - 1]));
    }

    const val_t new_energy = __dadd_rn(center, flux);
    energy_swap[index] = new_energy;
    accumulated_flux_swap[index] =
        __dadd_rn(accumulated_flux[index], fabs(flux));

    // Packing the next iteration's noncontiguous column halos here removes a
    // separate read of each boundary column.
    if (local_y == 1) {
        send_left[local_x - 1] = new_energy;
    }
    if (local_y == ny) {
        send_right[local_x - 1] = new_energy;
    }
}

constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

__global__ void updateInteriorKernel(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ accumulated_flux,
    val_t* __restrict__ energy_swap,
    val_t* __restrict__ accumulated_flux_swap,
    int nx, int ny, int pitch) {
    __shared__ val_t tile[TILE_Y + 2][TILE_X + 2];

    const int block_row = 2 + static_cast<int>(blockIdx.y) * TILE_Y;
    const int block_col = 2 + static_cast<int>(blockIdx.x) * TILE_X;
    const int linear_thread = static_cast<int>(threadIdx.y) * TILE_X + threadIdx.x;
    constexpr int tile_elements = (TILE_Y + 2) * (TILE_X + 2);
    constexpr int block_threads = TILE_X * TILE_Y;

    for (int item = linear_thread; item < tile_elements; item += block_threads) {
        const int tile_row = item / (TILE_X + 2);
        const int tile_col = item % (TILE_X + 2);
        const int local_x = block_row + tile_row - 1;
        const int local_y = block_col + tile_col - 1;
        val_t value = 0.0;
        if (local_x >= 1 && local_x <= nx && local_y >= 1 && local_y <= ny) {
            value = energy[static_cast<size_t>(local_x) * pitch + local_y];
        }
        tile[tile_row][tile_col] = value;
    }
    __syncthreads();

    const int local_x = block_row + threadIdx.y;
    const int local_y = block_col + threadIdx.x;
    if (local_x >= nx || local_y >= ny) {
        return;
    }

    const int shared_x = threadIdx.y + 1;
    const int shared_y = threadIdx.x + 1;
    const val_t center = tile[shared_x][shared_y];
    val_t flux = 0.0;
    flux = __dadd_rn(flux, computeFluxDevice(center, tile[shared_x + 1][shared_y]));
    flux = __dadd_rn(flux, computeFluxDevice(center, tile[shared_x - 1][shared_y]));
    flux = __dadd_rn(flux, computeFluxDevice(center, tile[shared_x][shared_y + 1]));
    flux = __dadd_rn(flux, computeFluxDevice(center, tile[shared_x][shared_y - 1]));

    const size_t index = static_cast<size_t>(local_x) * pitch + local_y;
    energy_swap[index] = __dadd_rn(center, flux);
    accumulated_flux_swap[index] =
        __dadd_rn(accumulated_flux[index], fabs(flux));
}

__global__ void updateBoundaryKernel(
    const val_t* __restrict__ energy,
    const val_t* __restrict__ accumulated_flux,
    val_t* __restrict__ energy_swap,
    val_t* __restrict__ accumulated_flux_swap,
    val_t* __restrict__ send_left,
    val_t* __restrict__ send_right,
    int boundary_elements, int nx, int ny, int pitch,
    int x_start, int y_start, int n_elems_root) {
    const int item = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= boundary_elements) {
        return;
    }

    int local_x;
    int local_y;
    if (nx == 1) {
        local_x = 1;
        local_y = item + 1;
    } else if (item < ny) {
        local_x = 1;
        local_y = item + 1;
    } else if (item < 2 * ny) {
        local_x = nx;
        local_y = item - ny + 1;
    } else {
        const int side_count = (ny == 1) ? 1 : 2;
        const int side_item = item - 2 * ny;
        local_x = side_item / side_count + 2;
        local_y = (side_count == 1 || side_item % 2 == 0) ? 1 : ny;
    }

    updatePoint(energy, accumulated_flux, energy_swap, accumulated_flux_swap,
                send_left, send_right, local_x, local_y, nx, ny, pitch,
                x_start, y_start, n_elems_root);
}

__global__ void packColumnsKernel(const val_t* __restrict__ energy,
                                  val_t* __restrict__ send_left,
                                  val_t* __restrict__ send_right,
                                  int nx, int ny, int pitch) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (row <= nx) {
        send_left[row - 1] = energy[static_cast<size_t>(row) * pitch + 1];
        send_right[row - 1] = energy[static_cast<size_t>(row) * pitch + ny];
    }
}

__global__ void unpackColumnsKernel(val_t* __restrict__ energy,
                                    const val_t* __restrict__ recv_left,
                                    const val_t* __restrict__ recv_right,
                                    int nx, int ny, int pitch) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (row <= nx) {
        energy[static_cast<size_t>(row) * pitch] = recv_left[row - 1];
        energy[static_cast<size_t>(row) * pitch + ny + 1] = recv_right[row - 1];
    }
}

int partitionExtent(int global_extent, int parts, int coordinate) {
    const int base = global_extent / parts;
    const int remainder = global_extent % parts;
    return base + (coordinate < remainder ? 1 : 0);
}

int partitionStart(int global_extent, int parts, int coordinate) {
    const int base = global_extent / parts;
    const int remainder = global_extent % parts;
    return coordinate * base + std::min(coordinate, remainder);
}

Decomposition makeDecomposition(int n_elems_root) {
    Decomposition domain;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &domain.size));
    MPI_CHECK(MPI_Dims_create(domain.size, 2, domain.dims));
    if (domain.dims[0] > n_elems_root || domain.dims[1] > n_elems_root) {
        if (world_rank == 0) {
            std::fprintf(stderr,
                         "The %d x %d MPI process grid exceeds the %d x %d mesh.\n",
                         domain.dims[0], domain.dims[1], n_elems_root, n_elems_root);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int periods[2] = {0, 0};
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 2, domain.dims, periods, 0,
                              &domain.cart));
    MPI_CHECK(MPI_Comm_rank(domain.cart, &domain.rank));
    MPI_CHECK(MPI_Cart_coords(domain.cart, domain.rank, 2, domain.coords));
    MPI_CHECK(MPI_Cart_shift(domain.cart, 0, 1, &domain.x_minus, &domain.x_plus));
    MPI_CHECK(MPI_Cart_shift(domain.cart, 1, 1, &domain.y_minus, &domain.y_plus));

    domain.nx = partitionExtent(n_elems_root, domain.dims[0], domain.coords[0]);
    domain.ny = partitionExtent(n_elems_root, domain.dims[1], domain.coords[1]);
    domain.x_start = partitionStart(n_elems_root, domain.dims[0], domain.coords[0]);
    domain.y_start = partitionStart(n_elems_root, domain.dims[1], domain.coords[1]);
    return domain;
}

template <typename T>
void allocateDevice(T** pointer, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)));
}

bool cudaAwareMpiAvailable() {
    const char* override_value = std::getenv("UNSTRUCTURED_CUDA_AWARE_MPI");
    if (override_value != nullptr) {
        return std::atoi(override_value) != 0;
    }
#if defined(OPEN_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA)
    return MPIX_Query_cuda_support() != 0;
#else
    // Unknown MPI implementations use the portable pinned-memory path.  A
    // CUDA-aware installation can opt into its direct path with the environment
    // variable above without requiring a rebuild.
    return false;
#endif
}

DeviceDomain allocateDeviceDomain(const Decomposition& decomposition,
                                  bool cuda_aware_mpi) {
    DeviceDomain device;
    device.cuda_aware_mpi = cuda_aware_mpi;
    device.pitch = static_cast<size_t>(decomposition.ny) + 2;
    device.allocation_elements =
        (static_cast<size_t>(decomposition.nx) + 2) * device.pitch;

    allocateDevice(&device.energy, device.allocation_elements);
    allocateDevice(&device.energy_swap, device.allocation_elements);
    allocateDevice(&device.total_flux, device.allocation_elements);
    allocateDevice(&device.total_flux_swap, device.allocation_elements);
    allocateDevice(&device.send_left, decomposition.nx);
    allocateDevice(&device.send_right, decomposition.nx);
    allocateDevice(&device.recv_left, decomposition.nx);
    allocateDevice(&device.recv_right, decomposition.nx);
    const bool has_neighbor =
        decomposition.x_minus != MPI_PROC_NULL ||
        decomposition.x_plus != MPI_PROC_NULL ||
        decomposition.y_minus != MPI_PROC_NULL ||
        decomposition.y_plus != MPI_PROC_NULL;
    if (!device.cuda_aware_mpi && has_neighbor) {
        const size_t halo_elements =
            4 * static_cast<size_t>(decomposition.ny) +
            4 * static_cast<size_t>(decomposition.nx);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&device.host_halos),
                                  halo_elements * sizeof(val_t)));
        val_t* halo = device.host_halos;
        device.host_send_top = halo;
        halo += decomposition.ny;
        device.host_send_bottom = halo;
        halo += decomposition.ny;
        device.host_recv_top = halo;
        halo += decomposition.ny;
        device.host_recv_bottom = halo;
        halo += decomposition.ny;
        device.host_send_left = halo;
        halo += decomposition.nx;
        device.host_send_right = halo;
        halo += decomposition.nx;
        device.host_recv_left = halo;
        halo += decomposition.nx;
        device.host_recv_right = halo;
        std::fill(device.host_halos, device.host_halos + halo_elements, 0.0);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.interior_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&device.boundary_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&device.interior_ready[0],
                                        cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&device.interior_ready[1],
                                        cudaEventDisableTiming));

    CUDA_CHECK(cudaMemsetAsync(device.energy, 0,
                               device.allocation_elements * sizeof(val_t),
                               device.boundary_stream));
    CUDA_CHECK(cudaMemsetAsync(device.energy_swap, 0,
                               device.allocation_elements * sizeof(val_t),
                               device.boundary_stream));
    CUDA_CHECK(cudaMemsetAsync(device.total_flux, 0,
                               device.allocation_elements * sizeof(val_t),
                               device.boundary_stream));
    CUDA_CHECK(cudaMemsetAsync(device.total_flux_swap, 0,
                               device.allocation_elements * sizeof(val_t),
                               device.boundary_stream));
    CUDA_CHECK(cudaMemsetAsync(device.recv_left, 0,
                               static_cast<size_t>(decomposition.nx) * sizeof(val_t),
                               device.boundary_stream));
    CUDA_CHECK(cudaMemsetAsync(device.recv_right, 0,
                               static_cast<size_t>(decomposition.nx) * sizeof(val_t),
                               device.boundary_stream));

    constexpr int threads = 256;
    const int blocks = (decomposition.nx + threads - 1) / threads;
    packColumnsKernel<<<blocks, threads, 0, device.boundary_stream>>>(
        device.energy, device.send_left, device.send_right,
        decomposition.nx, decomposition.ny, static_cast<int>(device.pitch));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(device.interior_ready[0], device.boundary_stream));
    CUDA_CHECK(cudaEventRecord(device.interior_ready[1], device.boundary_stream));
    CUDA_CHECK(cudaStreamSynchronize(device.boundary_stream));
    return device;
}

void freeDeviceDomain(DeviceDomain& device) {
    CUDA_CHECK(cudaEventDestroy(device.interior_ready[0]));
    CUDA_CHECK(cudaEventDestroy(device.interior_ready[1]));
    CUDA_CHECK(cudaStreamDestroy(device.interior_stream));
    CUDA_CHECK(cudaStreamDestroy(device.boundary_stream));
    CUDA_CHECK(cudaFree(device.energy));
    CUDA_CHECK(cudaFree(device.energy_swap));
    CUDA_CHECK(cudaFree(device.total_flux));
    CUDA_CHECK(cudaFree(device.total_flux_swap));
    CUDA_CHECK(cudaFree(device.send_left));
    CUDA_CHECK(cudaFree(device.send_right));
    CUDA_CHECK(cudaFree(device.recv_left));
    CUDA_CHECK(cudaFree(device.recv_right));
    if (device.host_halos != nullptr) {
        CUDA_CHECK(cudaFreeHost(device.host_halos));
    }
}

int boundaryElementCount(int nx, int ny) {
    if (nx == 1) {
        return ny;
    }
    return 2 * ny + std::max(nx - 2, 0) * (ny == 1 ? 1 : 2);
}

void runSimulation(DeviceDomain& device, const Decomposition& decomposition,
                   int n_elems_root, int n_iters) {
    constexpr int TAG_X_MINUS = 100;
    constexpr int TAG_X_PLUS = 101;
    constexpr int TAG_Y_MINUS = 102;
    constexpr int TAG_Y_PLUS = 103;
    constexpr int boundary_threads = 256;
    const int pitch = static_cast<int>(device.pitch);
    const int boundary_elements = boundaryElementCount(decomposition.nx, decomposition.ny);
    const int boundary_blocks =
        (boundary_elements + boundary_threads - 1) / boundary_threads;
    int current_slot = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Only the local perimeter is needed by MPI.  It is produced on a
        // separate stream so communication can start while the prior interior
        // work (which is ordered on interior_stream) is still completing.
        if (!device.cuda_aware_mpi) {
            const size_t row_bytes =
                static_cast<size_t>(decomposition.ny) * sizeof(val_t);
            const size_t column_bytes =
                static_cast<size_t>(decomposition.nx) * sizeof(val_t);
            if (decomposition.x_minus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    device.host_send_top, device.energy + pitch + 1, row_bytes,
                    cudaMemcpyDeviceToHost, device.boundary_stream));
            }
            if (decomposition.x_plus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    device.host_send_bottom,
                    device.energy +
                        static_cast<size_t>(decomposition.nx) * pitch + 1,
                    row_bytes, cudaMemcpyDeviceToHost, device.boundary_stream));
            }
            if (decomposition.y_minus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(device.host_send_left, device.send_left,
                                           column_bytes, cudaMemcpyDeviceToHost,
                                           device.boundary_stream));
            }
            if (decomposition.y_plus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(device.host_send_right, device.send_right,
                                           column_bytes, cudaMemcpyDeviceToHost,
                                           device.boundary_stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device.boundary_stream));

        val_t* recv_top = device.cuda_aware_mpi
                              ? device.energy + 1
                              : device.host_recv_top;
        val_t* recv_bottom =
            device.cuda_aware_mpi
                ? device.energy +
                      static_cast<size_t>(decomposition.nx + 1) * pitch + 1
                : device.host_recv_bottom;
        val_t* recv_left =
            device.cuda_aware_mpi ? device.recv_left : device.host_recv_left;
        val_t* recv_right =
            device.cuda_aware_mpi ? device.recv_right : device.host_recv_right;
        const val_t* send_top = device.cuda_aware_mpi
                                    ? device.energy + pitch + 1
                                    : device.host_send_top;
        const val_t* send_bottom =
            device.cuda_aware_mpi
                ? device.energy +
                      static_cast<size_t>(decomposition.nx) * pitch + 1
                : device.host_send_bottom;
        const val_t* send_left =
            device.cuda_aware_mpi ? device.send_left : device.host_send_left;
        const val_t* send_right =
            device.cuda_aware_mpi ? device.send_right : device.host_send_right;

        MPI_Request requests[8];
        int request_count = 0;
        if (decomposition.x_minus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_top, decomposition.ny, MPI_DOUBLE,
                                decomposition.x_minus, TAG_X_PLUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.x_plus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_bottom, decomposition.ny, MPI_DOUBLE,
                                decomposition.x_plus, TAG_X_MINUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.y_minus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_left, decomposition.nx, MPI_DOUBLE,
                                decomposition.y_minus, TAG_Y_PLUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.y_plus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Irecv(recv_right, decomposition.nx, MPI_DOUBLE,
                                decomposition.y_plus, TAG_Y_MINUS,
                                decomposition.cart, &requests[request_count++]));
        }

        if (decomposition.x_minus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(send_top, decomposition.ny, MPI_DOUBLE,
                                decomposition.x_minus, TAG_X_MINUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.x_plus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(send_bottom, decomposition.ny, MPI_DOUBLE,
                                decomposition.x_plus, TAG_X_PLUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.y_minus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(send_left, decomposition.nx, MPI_DOUBLE,
                                decomposition.y_minus, TAG_Y_MINUS,
                                decomposition.cart, &requests[request_count++]));
        }
        if (decomposition.y_plus != MPI_PROC_NULL) {
            MPI_CHECK(MPI_Isend(send_right, decomposition.nx, MPI_DOUBLE,
                                decomposition.y_plus, TAG_Y_PLUS,
                                decomposition.cart, &requests[request_count++]));
        }

        if (decomposition.nx > 2 && decomposition.ny > 2) {
            const dim3 threads(TILE_X, TILE_Y);
            const dim3 blocks((decomposition.ny - 2 + TILE_X - 1) / TILE_X,
                              (decomposition.nx - 2 + TILE_Y - 1) / TILE_Y);
            updateInteriorKernel<<<blocks, threads, 0, device.interior_stream>>>(
                device.energy, device.total_flux, device.energy_swap,
                device.total_flux_swap, decomposition.nx, decomposition.ny, pitch);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(device.interior_ready[1 - current_slot],
                                   device.interior_stream));

        MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));

        // A perimeter update reads the immediately adjacent interior of the
        // current buffer.  Track readiness per ping-pong buffer so this wait
        // does not unnecessarily serialize it behind this iteration's
        // independent interior work.
        CUDA_CHECK(cudaStreamWaitEvent(device.boundary_stream,
                                       device.interior_ready[current_slot], 0));

        if (!device.cuda_aware_mpi) {
            const size_t row_bytes =
                static_cast<size_t>(decomposition.ny) * sizeof(val_t);
            const size_t column_bytes =
                static_cast<size_t>(decomposition.nx) * sizeof(val_t);
            if (decomposition.x_minus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(device.energy + 1,
                                           device.host_recv_top, row_bytes,
                                           cudaMemcpyHostToDevice,
                                           device.boundary_stream));
            }
            if (decomposition.x_plus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(
                    device.energy +
                        static_cast<size_t>(decomposition.nx + 1) * pitch + 1,
                    device.host_recv_bottom, row_bytes, cudaMemcpyHostToDevice,
                    device.boundary_stream));
            }
            if (decomposition.y_minus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(device.recv_left,
                                           device.host_recv_left, column_bytes,
                                           cudaMemcpyHostToDevice,
                                           device.boundary_stream));
            }
            if (decomposition.y_plus != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(device.recv_right,
                                           device.host_recv_right, column_bytes,
                                           cudaMemcpyHostToDevice,
                                           device.boundary_stream));
            }
        }

        const int column_blocks =
            (decomposition.nx + boundary_threads - 1) / boundary_threads;
        if (decomposition.y_minus != MPI_PROC_NULL ||
            decomposition.y_plus != MPI_PROC_NULL) {
            unpackColumnsKernel<<<column_blocks, boundary_threads, 0,
                                  device.boundary_stream>>>(
                device.energy, device.recv_left, device.recv_right,
                decomposition.nx, decomposition.ny, pitch);
        }
        updateBoundaryKernel<<<boundary_blocks, boundary_threads, 0,
                               device.boundary_stream>>>(
            device.energy, device.total_flux, device.energy_swap,
            device.total_flux_swap, device.send_left, device.send_right,
            boundary_elements, decomposition.nx, decomposition.ny, pitch,
            decomposition.x_start, decomposition.y_start, n_elems_root);
        CUDA_CHECK(cudaGetLastError());

        std::swap(device.energy, device.energy_swap);
        std::swap(device.total_flux, device.total_flux_swap);
        current_slot = 1 - current_slot;
    }

    CUDA_CHECK(cudaStreamSynchronize(device.interior_stream));
    CUDA_CHECK(cudaStreamSynchronize(device.boundary_stream));
}

void copyLocalResults(const DeviceDomain& device,
                      const Decomposition& decomposition,
                      std::vector<val_t>& energy,
                      std::vector<val_t>& flux) {
    const size_t local_elements =
        static_cast<size_t>(decomposition.nx) * decomposition.ny;
    energy.resize(local_elements);
    flux.resize(local_elements);
    const size_t row_bytes = static_cast<size_t>(decomposition.ny) * sizeof(val_t);
    const val_t* energy_start = device.energy + device.pitch + 1;
    const val_t* flux_start = device.total_flux + device.pitch + 1;
    CUDA_CHECK(cudaMemcpy2D(energy.data(), row_bytes, energy_start,
                            device.pitch * sizeof(val_t), row_bytes,
                            decomposition.nx, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(flux.data(), row_bytes, flux_start,
                            device.pitch * sizeof(val_t), row_bytes,
                            decomposition.nx, cudaMemcpyDeviceToHost));
}

uint64_t computeDistributedHash(const std::vector<val_t>& energy,
                                const std::vector<val_t>& flux,
                                const Decomposition& decomposition,
                                int n_elems_root) {
    uint64_t local_hash = 0;
    const int64_t count = static_cast<int64_t>(energy.size());

#pragma omp parallel for schedule(static) reduction(^ : local_hash)
    for (int64_t local_index = 0; local_index < count; ++local_index) {
        const int local_x = static_cast<int>(local_index / decomposition.ny);
        const int local_y = static_cast<int>(local_index % decomposition.ny);
        const uint64_t global_index =
            static_cast<uint64_t>(decomposition.x_start + local_x) * n_elems_root +
            decomposition.y_start + local_y;
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &energy[local_index], sizeof(energy_bits));
        std::memcpy(&flux_bits, &flux[local_index], sizeof(flux_bits));
        local_hash ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T,
                         MPI_BXOR, 0, decomposition.cart));
    return global_hash;
}

std::vector<ElementDynamic> gatherResults(const std::vector<val_t>& local_energy,
                                          const std::vector<val_t>& local_flux,
                                          const Decomposition& decomposition,
                                          int n_elems_root) {
    if (local_energy.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage("MPI", "MPI_Gatherv count", "local domain is too large",
                         __FILE__, __LINE__);
    }

    const int local_count = static_cast<int>(local_energy.size());
    const int metadata[4] = {decomposition.x_start, decomposition.y_start,
                             decomposition.nx, decomposition.ny};
    std::vector<int> all_metadata;
    std::vector<int> counts;
    if (decomposition.rank == 0) {
        all_metadata.resize(static_cast<size_t>(decomposition.size) * 4);
        counts.resize(decomposition.size);
    }
    MPI_CHECK(MPI_Gather(metadata, 4, MPI_INT, all_metadata.data(), 4, MPI_INT,
                         0, decomposition.cart));
    MPI_CHECK(MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                         0, decomposition.cart));

    std::vector<int> displacements;
    std::vector<val_t> gathered_energy;
    std::vector<val_t> gathered_flux;
    if (decomposition.rank == 0) {
        displacements.resize(decomposition.size);
        int total = 0;
        for (int rank = 0; rank < decomposition.size; ++rank) {
            displacements[rank] = total;
            if (counts[rank] > std::numeric_limits<int>::max() - total) {
                abortWithMessage("MPI", "MPI_Gatherv displacement",
                                 "global result is too large", __FILE__, __LINE__);
            }
            total += counts[rank];
        }
        gathered_energy.resize(total);
        gathered_flux.resize(total);
    }

    MPI_CHECK(MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                          gathered_energy.data(), counts.data(), displacements.data(),
                          MPI_DOUBLE, 0, decomposition.cart));
    MPI_CHECK(MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE,
                          gathered_flux.data(), counts.data(), displacements.data(),
                          MPI_DOUBLE, 0, decomposition.cart));

    std::vector<ElementDynamic> global_results;
    if (decomposition.rank == 0) {
        global_results.resize(static_cast<size_t>(n_elems_root) * n_elems_root);

        // All OpenMP workers cooperate on each MPI block.  This avoids races
        // and retains global row-major ordering for output and validation.
#pragma omp parallel
        {
            for (int rank = 0; rank < decomposition.size; ++rank) {
                const int x_start = all_metadata[4 * rank];
                const int y_start = all_metadata[4 * rank + 1];
                const int nx = all_metadata[4 * rank + 2];
                const int ny = all_metadata[4 * rank + 3];
#pragma omp for schedule(static)
                for (int local_index = 0; local_index < counts[rank]; ++local_index) {
                    const int local_x = local_index / ny;
                    const int local_y = local_index % ny;
                    const size_t global_index =
                        static_cast<size_t>(x_start + local_x) * n_elems_root +
                        y_start + local_y;
                    const int gathered_index = displacements[rank] + local_index;
                    global_results[global_index] =
                        ElementDynamic{gathered_energy[gathered_index],
                                       gathered_flux[gathered_index]};
                }
                (void)nx;
            }
        }
    }
    return global_results;
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& element : elements) {
        energy_sum += element.current_energy;
        flux_sum += element.total_flux;
        energy_max = std::max(element.current_energy, energy_max);
        energy_min = std::min(element.current_energy, energy_min);
    }

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
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
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

int main(int argc, char** argv) {
    int provided_thread_level = 0;
    const int init_error =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (init_error != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required FUNNELED thread level.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool arguments_valid = true;
    bool help_requested = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help_requested = true;
        } else {
            if (world_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
        }
    }

    if (n_elems_root <= 0 || n_iters < 0) {
        if (world_rank == 0) {
            std::fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        }
        arguments_valid = false;
    }
    if (help_requested || !arguments_valid) {
        if (world_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank = 0;
    int local_size = 1;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        abortWithMessage("CUDA", "cudaGetDeviceCount", "no CUDA devices found",
                         __FILE__, __LINE__);
    }
    const int device_id = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device_id));

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int threads_per_rank =
            std::max(1, std::min(32, omp_get_num_procs() / local_size));
        omp_set_num_threads(threads_per_rank);
    }

    Decomposition decomposition = makeDecomposition(n_elems_root);
    const bool cuda_aware_mpi = cudaAwareMpiAvailable();
    DeviceDomain device = allocateDeviceDomain(decomposition, cuda_aware_mpi);

    const uint64_t n_elems =
        static_cast<uint64_t>(n_elems_root) * n_elems_root;
    const size_t logical_static_mem = n_elems * sizeof(ElementStatic);
    const size_t logical_dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
    const uint64_t local_device_mem =
        (4 * device.allocation_elements + 4 * static_cast<size_t>(decomposition.nx)) *
        sizeof(val_t);
    uint64_t max_device_mem = 0;
    MPI_CHECK(MPI_Reduce(&local_device_mem, &max_device_mem, 1, MPI_UINT64_T,
                         MPI_MAX, 0, decomposition.cart));

    if (world_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %" PRIu64 " elements\n",
                    n_elems_root, n_elems_root, n_elems);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks (%d x %d), up to %d OpenMP threads/rank\n",
                    decomposition.size, decomposition.dims[0], decomposition.dims[1],
                    omp_get_max_threads());
        std::printf("CUDA devices/node: %d (%s on rank 0)\n", device_count,
                    device_properties.name);
        std::printf("MPI halo transport: %s\n",
                    cuda_aware_mpi ? "direct CUDA-aware device buffers"
                                   : "portable pinned-host staging");
        std::printf("Building distributed unstructured mesh...\n");
        std::printf("Logical memory: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                    (logical_static_mem + logical_dynamic_mem) / (1024.0 * 1024.0),
                    logical_static_mem / (1024.0 * 1024.0),
                    logical_dynamic_mem / (1024.0 * 1024.0));
        std::printf("Maximum CUDA memory/rank: %.2f MB\n\n",
                    max_device_mem / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_CHECK(MPI_Barrier(decomposition.cart));
    const double start = MPI_Wtime();
    runSimulation(device, decomposition, n_elems_root, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE,
                         MPI_MAX, 0, decomposition.cart));

    std::vector<val_t> local_energy;
    std::vector<val_t> local_flux;
    copyLocalResults(device, decomposition, local_energy, local_flux);
    const uint64_t result_hash = computeDistributedHash(
        local_energy, local_flux, decomposition, n_elems_root);

    if (world_rank == 0) {
        const double duration_ms = elapsed_seconds * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elements_per_second =
            elapsed_seconds > 0.0
                ? (static_cast<double>(n_measured_iters) * n_elems) /
                      elapsed_seconds / 1.0e9
                : 0.0;
        const double gflops = giga_elements_per_second * 22.0;

        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }

    std::vector<ElementDynamic> global_results;
    if (print_results_requested || validate) {
        global_results = gatherResults(local_energy, local_flux,
                                       decomposition, n_elems_root);
    }

    int exit_status = 0;
    if (world_rank == 0 && print_results_requested) {
        std::vector<val_t> energy_data(global_results.size());
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(global_results.size()); ++i) {
            energy_data[i] = global_results[i].current_energy;
        }
        print_results(energy_data, "ElementEnergy");
    }
    if (world_rank == 0 && validate && !validateResults(global_results)) {
        exit_status = 1;
    }
    MPI_CHECK(MPI_Bcast(&exit_status, 1, MPI_INT, 0, decomposition.cart));

    freeDeviceDomain(device);
    MPI_CHECK(MPI_Comm_free(&decomposition.cart));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    MPI_CHECK(MPI_Finalize());
    return exit_status;
}
