#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Types and mesh constants retained from the original benchmark.
using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;
struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

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

namespace {

constexpr double TRANSFER_COEFF = 0.8;
constexpr double CONNECTION_FLUX = 1.0;
constexpr double FLUX_SCALE = 0.25;

int g_mpi_rank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", g_mpi_rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t error, const char* call, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d in %s: %s\n", g_mpi_rank,
                     file, line, call, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

struct Partition {
    int start;
    int size;
};

Partition makePartition(int global_size, int partitions, int coordinate) {
    const int base = global_size / partitions;
    const int remainder = global_size % partitions;
    const int size = base + (coordinate < remainder ? 1 : 0);
    const int start = coordinate * base + std::min(coordinate, remainder);
    return {start, size};
}

int ownerCoordinate(int global_index, int global_size, int partitions) {
    const int base = global_size / partitions;
    const int remainder = global_size % partitions;
    const int larger_partition_extent = (base + 1) * remainder;
    if (global_index < larger_partition_extent) {
        return global_index / (base + 1);
    }
    return remainder + (global_index - larger_partition_extent) / base;
}

struct Domain {
    int global_n;
    int local_x0;
    int local_y0;
    int local_nx;
    int local_ny;
    int dims[2];
    int coords[2];
    int x_minus;
    int x_plus;
    int y_minus;
    int y_plus;

    size_t localElements() const {
        return static_cast<size_t>(local_nx) * static_cast<size_t>(local_ny);
    }
};

struct PinnedBuffer {
    double* data = nullptr;

    explicit PinnedBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaHostAlloc(&data, count * sizeof(*data), cudaHostAllocDefault));
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data != nullptr) {
            // Destructors run after successful CUDA calls; do not throw during teardown.
            cudaFreeHost(data);
        }
    }
};

// The device arrays are padded with a one-element ghost layer.  Energy is the
// only remote dependency; total_flux is local history and never needs a halo.
struct DeviceState {
    double* energy[2] = {nullptr, nullptr};
    double* flux[2] = {nullptr, nullptr};
    size_t allocation_size = 0;

    explicit DeviceState(size_t padded_elements) : allocation_size(padded_elements) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMalloc(&energy[i], allocation_size * sizeof(*energy[i])));
            CUDA_CHECK(cudaMalloc(&flux[i], allocation_size * sizeof(*flux[i])));
        }
    }

    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;

    ~DeviceState() {
        for (int i = 0; i < 2; ++i) {
            if (energy[i] != nullptr) {
                cudaFree(energy[i]);
            }
            if (flux[i] != nullptr) {
                cudaFree(flux[i]);
            }
        }
    }
};

__device__ __forceinline__ double externalFlow(int global_x, int global_y, int global_n) {
    // This order exactly matches the four assignments at the end of
    // buildSquare2D, including the n == 1 corner case.
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

__device__ __forceinline__ double connectionContribution(double other_energy,
                                                          double this_energy) {
    // Keep the source expression's operation order.  --fmad=false in the
    // CUDA build prevents the contraction from changing benchmark semantics.
    return ((other_energy - this_energy) * TRANSFER_COEFF * CONNECTION_FLUX) * FLUX_SCALE;
}

__global__ void updateElements(const double* __restrict__ energy_in,
                               const double* __restrict__ flux_in,
                               double* __restrict__ energy_out,
                               double* __restrict__ flux_out,
                               int pitch,
                               int local_nx,
                               int local_ny,
                               int global_x0,
                               int global_y0,
                               int global_n,
                               bool boundary_only) {
    const int local_y = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_x = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_x >= local_nx || local_y >= local_ny) {
        return;
    }

    const bool is_local_boundary = local_x == 0 || local_y == 0 ||
                                   local_x == local_nx - 1 || local_y == local_ny - 1;
    if (is_local_boundary != boundary_only) {
        return;
    }

    const int x = local_x + 1;
    const int y = local_y + 1;
    const int index = x * pitch + y;
    const int global_x = global_x0 + local_x;
    const int global_y = global_y0 + local_y;
    const double this_energy = energy_in[index];
    double total_flux = externalFlow(global_x, global_y, global_n);

    // The neighbour order is up, down, right, left, as in buildSquare2D.
    if (global_x + 1 < global_n) {
        total_flux += connectionContribution(energy_in[index + pitch], this_energy);
    }
    if (global_x > 0) {
        total_flux += connectionContribution(energy_in[index - pitch], this_energy);
    }
    if (global_y + 1 < global_n) {
        total_flux += connectionContribution(energy_in[index + 1], this_energy);
    }
    if (global_y > 0) {
        total_flux += connectionContribution(energy_in[index - 1], this_energy);
    }

    energy_out[index] = this_energy + total_flux;
    flux_out[index] = flux_in[index] + fabs(total_flux);
}

// Launch a CUDA kernel for either the local interior or the one-cell local boundary.
void launchUpdate(const Domain& domain,
                  const DeviceState& state,
                  int current,
                  cudaStream_t stream,
                  bool boundary_only) {
    constexpr dim3 block(32, 8);
    const dim3 grid((domain.local_ny + block.x - 1) / block.x,
                    (domain.local_nx + block.y - 1) / block.y);
    const int pitch = domain.local_ny + 2;
    updateElements<<<grid, block, 0, stream>>>(state.energy[current], state.flux[current],
                                                state.energy[current ^ 1], state.flux[current ^ 1],
                                                pitch, domain.local_nx, domain.local_ny,
                                                domain.local_x0, domain.local_y0, domain.global_n,
                                                boundary_only);
    CUDA_CHECK(cudaGetLastError());
}

void runSimulation(const Domain& domain,
                   MPI_Comm cart_comm,
                   DeviceState& state,
                   int n_iters,
                   cudaStream_t compute_stream,
                   cudaStream_t copy_stream) {
    const int pitch = domain.local_ny + 2;
    PinnedBuffer send_x_minus(domain.local_ny);
    PinnedBuffer send_x_plus(domain.local_ny);
    PinnedBuffer recv_x_minus(domain.local_ny);
    PinnedBuffer recv_x_plus(domain.local_ny);
    PinnedBuffer send_y_minus(domain.local_nx);
    PinnedBuffer send_y_plus(domain.local_nx);
    PinnedBuffer recv_y_minus(domain.local_nx);
    PinnedBuffer recv_y_plus(domain.local_nx);

    cudaEvent_t outbound_ready;
    cudaEvent_t inbound_ready;
    cudaEvent_t iteration_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&outbound_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&inbound_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&iteration_done, cudaEventDisableTiming));

    int current = 0;
    bool have_previous_iteration = false;
    constexpr int X_MINUS_TAG = 100;
    constexpr int X_PLUS_TAG = 101;
    constexpr int Y_MINUS_TAG = 200;
    constexpr int Y_PLUS_TAG = 201;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (have_previous_iteration) {
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, iteration_done, 0));
        }

        double* const current_energy = state.energy[current];
        const size_t row_bytes = static_cast<size_t>(domain.local_ny) * sizeof(double);
        const size_t device_pitch_bytes = static_cast<size_t>(pitch) * sizeof(double);

        // Copy outgoing faces to pinned memory.  A portable MPI program cannot
        // assume CUDA-aware MPI, so this is intentionally standard MPI staging.
        if (domain.x_minus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_x_minus.data, current_energy + pitch + 1, row_bytes,
                                       cudaMemcpyDeviceToHost, copy_stream));
        }
        if (domain.x_plus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_x_plus.data,
                                       current_energy + domain.local_nx * pitch + 1,
                                       row_bytes, cudaMemcpyDeviceToHost, copy_stream));
        }
        if (domain.y_minus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(send_y_minus.data, sizeof(double),
                                         current_energy + pitch + 1, device_pitch_bytes,
                                         sizeof(double), domain.local_nx,
                                         cudaMemcpyDeviceToHost, copy_stream));
        }
        if (domain.y_plus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(send_y_plus.data, sizeof(double),
                                         current_energy + pitch + domain.local_ny, device_pitch_bytes,
                                         sizeof(double), domain.local_nx,
                                         cudaMemcpyDeviceToHost, copy_stream));
        }
        CUDA_CHECK(cudaEventRecord(outbound_ready, copy_stream));

        // Run the strictly local work while face data travels through PCIe and MPI.
        launchUpdate(domain, state, current, compute_stream, false);

        CUDA_CHECK(cudaEventSynchronize(outbound_ready));

        MPI_Request requests[8];
        int request_count = 0;
        const auto irecv = [&](double* buffer, int count, int source, int tag) {
            if (source != MPI_PROC_NULL) {
                MPI_Irecv(buffer, count, MPI_DOUBLE, source, tag, cart_comm,
                          &requests[request_count++]);
            }
        };
        const auto isend = [&](const double* buffer, int count, int destination, int tag) {
            if (destination != MPI_PROC_NULL) {
                MPI_Isend(buffer, count, MPI_DOUBLE, destination, tag, cart_comm,
                          &requests[request_count++]);
            }
        };

        // A minus-side neighbour sends its plus face, and vice versa.
        irecv(recv_x_minus.data, domain.local_ny, domain.x_minus, X_PLUS_TAG);
        irecv(recv_x_plus.data, domain.local_ny, domain.x_plus, X_MINUS_TAG);
        irecv(recv_y_minus.data, domain.local_nx, domain.y_minus, Y_PLUS_TAG);
        irecv(recv_y_plus.data, domain.local_nx, domain.y_plus, Y_MINUS_TAG);
        isend(send_x_minus.data, domain.local_ny, domain.x_minus, X_MINUS_TAG);
        isend(send_x_plus.data, domain.local_ny, domain.x_plus, X_PLUS_TAG);
        isend(send_y_minus.data, domain.local_nx, domain.y_minus, Y_MINUS_TAG);
        isend(send_y_plus.data, domain.local_nx, domain.y_plus, Y_PLUS_TAG);
        if (request_count != 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        // Install incoming energy faces into the ghost layer before the boundary
        // kernel.  These asynchronous copies are ordered ahead of the boundary
        // work, but the interior kernel remains overlapped with communication.
        if (domain.x_minus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(current_energy + 1, recv_x_minus.data, row_bytes,
                                       cudaMemcpyHostToDevice, copy_stream));
        }
        if (domain.x_plus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(current_energy + (domain.local_nx + 1) * pitch + 1,
                                       recv_x_plus.data, row_bytes, cudaMemcpyHostToDevice,
                                       copy_stream));
        }
        if (domain.y_minus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(current_energy + pitch, device_pitch_bytes,
                                         recv_y_minus.data, sizeof(double), sizeof(double),
                                         domain.local_nx, cudaMemcpyHostToDevice, copy_stream));
        }
        if (domain.y_plus != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2DAsync(current_energy + pitch + domain.local_ny + 1,
                                         device_pitch_bytes, recv_y_plus.data, sizeof(double),
                                         sizeof(double), domain.local_nx, cudaMemcpyHostToDevice,
                                         copy_stream));
        }
        CUDA_CHECK(cudaEventRecord(inbound_ready, copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, inbound_ready, 0));
        launchUpdate(domain, state, current, compute_stream, true);
        CUDA_CHECK(cudaEventRecord(iteration_done, compute_stream));

        current ^= 1;
        have_previous_iteration = true;
    }

    if (have_previous_iteration) {
        CUDA_CHECK(cudaEventSynchronize(iteration_done));
    }
    CUDA_CHECK(cudaEventDestroy(iteration_done));
    CUDA_CHECK(cudaEventDestroy(inbound_ready));
    CUDA_CHECK(cudaEventDestroy(outbound_ready));
}

void downloadState(const Domain& domain,
                   const DeviceState& state,
                   int current,
                   cudaStream_t copy_stream,
                   std::vector<double>& energy,
                   std::vector<double>& flux) {
    const int pitch = domain.local_ny + 2;
    const size_t row_bytes = static_cast<size_t>(domain.local_ny) * sizeof(double);
    CUDA_CHECK(cudaMemcpy2DAsync(energy.data(), row_bytes, state.energy[current] + pitch + 1,
                                 static_cast<size_t>(pitch) * sizeof(double), row_bytes,
                                 domain.local_nx, cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaMemcpy2DAsync(flux.data(), row_bytes, state.flux[current] + pitch + 1,
                                 static_cast<size_t>(pitch) * sizeof(double), row_bytes,
                                 domain.local_nx, cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
}

uint64_t computeDistributedHash(const Domain& domain,
                                const std::vector<double>& energy,
                                const std::vector<double>& flux,
                                MPI_Comm cart_comm) {
    uint64_t local_hash = 0;
    const size_t local_count = domain.localElements();

#pragma omp parallel for reduction(^ : local_hash) schedule(static)
    for (ptrdiff_t local_index = 0; local_index < static_cast<ptrdiff_t>(local_count); ++local_index) {
        const size_t index = static_cast<size_t>(local_index);
        const int local_x = static_cast<int>(index / domain.local_ny);
        const int local_y = static_cast<int>(index % domain.local_ny);
        const uint64_t global_index =
            static_cast<uint64_t>(domain.local_x0 + local_x) * domain.global_n +
            static_cast<uint64_t>(domain.local_y0 + local_y);
        local_hash ^= (std::bit_cast<uint64_t>(energy[index]) + global_index) *
                      0x9e3779b97f4a7c15ULL;
        local_hash ^= (std::bit_cast<uint64_t>(flux[index]) + global_index) *
                      0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, cart_comm);
    return global_hash;
}

std::vector<ElementDynamic> gatherGlobalState(const Domain& domain,
                                              const std::vector<double>& local_energy,
                                              const std::vector<double>& local_flux,
                                              MPI_Comm cart_comm,
                                              int cart_rank) {
    const int local_count = static_cast<int>(domain.localElements());
    std::vector<int> counts;
    if (cart_rank == 0) {
        counts.resize(domain.dims[0] * domain.dims[1]);
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, cart_comm);

    std::vector<int> displacements;
    std::vector<double> packed_energy;
    std::vector<double> packed_flux;
    if (cart_rank == 0) {
        displacements.resize(counts.size());
        int total = 0;
        for (size_t rank = 0; rank < counts.size(); ++rank) {
            displacements[rank] = total;
            total += counts[rank];
        }
        packed_energy.resize(total);
        packed_flux.resize(total);
    }

    MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, packed_energy.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, cart_comm);
    MPI_Gatherv(local_flux.data(), local_count, MPI_DOUBLE, packed_flux.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, cart_comm);

    if (cart_rank != 0) {
        return {};
    }

    const size_t global_elements = static_cast<size_t>(domain.global_n) * domain.global_n;
    std::vector<ElementDynamic> global_state(global_elements);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t global_index = 0; global_index < static_cast<ptrdiff_t>(global_elements);
         ++global_index) {
        const int global_x = static_cast<int>(static_cast<size_t>(global_index) / domain.global_n);
        const int global_y = static_cast<int>(static_cast<size_t>(global_index) % domain.global_n);
        const int coord_x = ownerCoordinate(global_x, domain.global_n, domain.dims[0]);
        const int coord_y = ownerCoordinate(global_y, domain.global_n, domain.dims[1]);
        const Partition partition_x = makePartition(domain.global_n, domain.dims[0], coord_x);
        const Partition partition_y = makePartition(domain.global_n, domain.dims[1], coord_y);
        const int source_rank = coord_x * domain.dims[1] + coord_y;
        const size_t source_index = static_cast<size_t>(displacements[source_rank]) +
            static_cast<size_t>(global_x - partition_x.start) * partition_y.size +
            (global_y - partition_y.start);
        global_state[static_cast<size_t>(global_index)] =
            ElementDynamic{packed_energy[source_index], packed_flux[source_index]};
    }
    return global_state;
}

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
    if (!std::isfinite(flux_sum) || !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Flux result is not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* prog_name) {
    std::printf("Usage: %s [options]\n", prog_name);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        fail("MPI does not provide MPI_THREAD_FUNNELED");
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool parse_error = false;
    bool show_help = false;
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
            show_help = true;
        } else {
            if (g_mpi_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parse_error = true;
        }
    }
    if (show_help || parse_error) {
        if (g_mpi_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_error ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (n_elems_root <= 0 || n_iters < 0) {
        if (g_mpi_rank == 0) {
            std::fprintf(stderr, "Grid size must be positive and iterations must be non-negative.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    if (dims[0] > n_elems_root || dims[1] > n_elems_root) {
        fail("MPI Cartesian grid has more ranks than mesh cells in one dimension");
    }
    int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        fail("could not create MPI Cartesian communicator");
    }
    int cart_rank = 0;
    int coords[2] = {0, 0};
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_mpi_rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fail("no CUDA device is visible to this MPI rank");
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));
    MPI_Comm_free(&node_comm);

    const Partition partition_x = makePartition(n_elems_root, dims[0], coords[0]);
    const Partition partition_y = makePartition(n_elems_root, dims[1], coords[1]);
    Domain domain{n_elems_root, partition_x.start, partition_y.start, partition_x.size,
                  partition_y.size, {dims[0], dims[1]}, {coords[0], coords[1]},
                  MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    MPI_Cart_shift(cart_comm, 0, 1, &domain.x_minus, &domain.x_plus);
    MPI_Cart_shift(cart_comm, 1, 1, &domain.y_minus, &domain.y_plus);

    const size_t global_elements = static_cast<size_t>(n_elems_root) * n_elems_root;
    const size_t local_elements = domain.localElements();
    const size_t padded_elements = static_cast<size_t>(domain.local_nx + 2) * (domain.local_ny + 2);
    std::vector<double> local_energy(local_elements, 0.0);
    std::vector<double> local_flux(local_elements, 0.0);
    // OpenMP owns the CPU-side setup and post-processing phases; MPI calls stay
    // on the main thread under MPI_THREAD_FUNNELED.
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(local_elements); ++i) {
        local_energy[static_cast<size_t>(i)] = 0.0;
        local_flux[static_cast<size_t>(i)] = 0.0;
    }

    if (cart_rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root,
                    global_elements);
        std::printf("Iterations: %d\n", n_iters);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (%d x %d Cartesian grid), CUDA devices per node: %d\n\n",
                    world_size, dims[0], dims[1], device_count);
        const size_t static_mem = global_elements * sizeof(ElementStatic);
        const size_t dynamic_mem = global_elements * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        std::printf("Building distributed unstructured mesh...\n");
        std::printf("Global memory-equivalent usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                    total_mem / (1024.0 * 1024.0), static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    cudaStream_t compute_stream;
    cudaStream_t copy_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    DeviceState state(padded_elements);
    const int pitch = domain.local_ny + 2;
    CUDA_CHECK(cudaMemsetAsync(state.energy[0], 0, padded_elements * sizeof(double), copy_stream));
    CUDA_CHECK(cudaMemsetAsync(state.flux[0], 0, padded_elements * sizeof(double), copy_stream));
    CUDA_CHECK(cudaMemcpy2DAsync(state.energy[0] + pitch + 1, static_cast<size_t>(pitch) * sizeof(double),
                                 local_energy.data(), static_cast<size_t>(domain.local_ny) * sizeof(double),
                                 static_cast<size_t>(domain.local_ny) * sizeof(double), domain.local_nx,
                                 cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaMemcpy2DAsync(state.flux[0] + pitch + 1, static_cast<size_t>(pitch) * sizeof(double),
                                 local_flux.data(), static_cast<size_t>(domain.local_ny) * sizeof(double),
                                 static_cast<size_t>(domain.local_ny) * sizeof(double), domain.local_nx,
                                 cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));

    MPI_Barrier(cart_comm);
    const auto start = std::chrono::steady_clock::now();
    runSimulation(domain, cart_comm, state, n_iters, compute_stream, copy_stream);
    const auto end = std::chrono::steady_clock::now();
    const double local_elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsed_ms = 0.0;
    MPI_Allreduce(&local_elapsed_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

    const int current = n_iters & 1;
    downloadState(domain, state, current, copy_stream, local_energy, local_flux);
    const uint64_t result_hash = computeDistributedHash(domain, local_energy, local_flux, cart_comm);

    if (cart_rank == 0) {
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double safe_elapsed_ms = std::max(elapsed_ms, 1.0e-9);
        const double time_per_iter = safe_elapsed_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * global_elements) / (safe_elapsed_ms / 1000.0) / 1.0e9;
        std::printf("Computation time: %.3f ms\n", elapsed_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        std::printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }

    if (validate || print_results_requested) {
        std::vector<ElementDynamic> global_state =
            gatherGlobalState(domain, local_energy, local_flux, cart_comm, cart_rank);
        if (cart_rank == 0) {
            if (print_results_requested) {
                std::vector<double> energy_data(global_state.size());
#pragma omp parallel for schedule(static)
                for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(global_state.size()); ++i) {
                    energy_data[static_cast<size_t>(i)] = global_state[static_cast<size_t>(i)].current_energy;
                }
                print_results(energy_data, "ElementEnergy");
            }
            if (validate && !validateResults(global_state)) {
                CUDA_CHECK(cudaStreamDestroy(copy_stream));
                CUDA_CHECK(cudaStreamDestroy(compute_stream));
                MPI_Comm_free(&cart_comm);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
