#include <algorithm>
#include <cinttypes>
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

using idx_t = uint64_t;
using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr val_t FLUX_SCALE = 0.25;
constexpr int TAG_TO_PREVIOUS = 100;
constexpr int TAG_TO_NEXT = 101;

[[noreturn]] void fatalError(const char* what, const char* file, int line) {
    int initialized = 0;
    int rank = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s at %s:%d\n", rank, what, file, line);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    char buffer[MPI_MAX_ERROR_STRING + 256];
    std::snprintf(buffer, sizeof(buffer), "MPI call %s failed: %.*s", expression,
                  length, message);
    fatalError(buffer, file, line);
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), "CUDA call %s failed: %s", expression,
                  cudaGetErrorString(status));
    fatalError(buffer, file, line);
}

#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

struct Slab {
    int first_row;
    int rows;
};

Slab getSlab(int n, int rank, int ranks) {
    const int quotient = n / ranks;
    const int remainder = n % ranks;
    return Slab{rank * quotient + std::min(rank, remainder),
                quotient + (rank < remainder ? 1 : 0)};
}

__device__ __forceinline__ val_t externalFlow(int global_row, int column, int n) {
    if (n == 1) {
        return 0.5;
    }
    if (global_row == 0) {
        if (column == 0) {
            return 0.5;
        }
        if (column == n - 1) {
            return -0.5;
        }
    } else if (global_row == n - 1) {
        if (column == 0) {
            return -0.5;
        }
        if (column == n - 1) {
            return 0.5;
        }
    }
    return 0.0;
}

__device__ __forceinline__ void updateElement(
    const val_t* __restrict__ energy, const val_t* __restrict__ flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_flux,
    const val_t* __restrict__ previous_halo,
    const val_t* __restrict__ next_halo, int n, int local_rows,
    int first_global_row, int local_row, int column) {
    const size_t index = static_cast<size_t>(local_row) * n + column;
    const int global_row = first_global_row + local_row;
    const val_t center = energy[index];
    val_t total = externalFlow(global_row, column, n);

    // Preserve the original connection order: +row, -row, +column, -column.
    if (global_row + 1 < n) {
        const val_t neighbor = local_row + 1 < local_rows
                                   ? energy[index + n]
                                   : next_halo[column];
        total += (neighbor - center) * TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }
    if (global_row > 0) {
        const val_t neighbor = local_row > 0 ? energy[index - n]
                                              : previous_halo[column];
        total += (neighbor - center) * TRANSFER_COEFF * CONNECTION_FLUX * FLUX_SCALE;
    }
    if (column + 1 < n) {
        total += (energy[index + 1] - center) * TRANSFER_COEFF *
                 CONNECTION_FLUX * FLUX_SCALE;
    }
    if (column > 0) {
        total += (energy[index - 1] - center) * TRANSFER_COEFF *
                 CONNECTION_FLUX * FLUX_SCALE;
    }

    next_energy[index] = center + total;
    next_flux[index] = flux[index] + fabs(total);
}

__global__ void updateInteriorKernel(
    const val_t* __restrict__ energy, const val_t* __restrict__ flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_flux,
    int n, int local_rows, int first_global_row) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_row = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= n || local_row >= local_rows - 1) {
        return;
    }
    updateElement(energy, flux, next_energy, next_flux, nullptr, nullptr, n,
                  local_rows, first_global_row, local_row, column);
}

__global__ void updateSingleRankKernel(
    const val_t* __restrict__ energy, const val_t* __restrict__ flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_flux, int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= n || row >= n) {
        return;
    }
    updateElement(energy, flux, next_energy, next_flux, nullptr, nullptr, n, n,
                  0, row, column);
}

__global__ void updateBoundaryKernel(
    const val_t* __restrict__ energy, const val_t* __restrict__ flux,
    val_t* __restrict__ next_energy, val_t* __restrict__ next_flux,
    const val_t* __restrict__ previous_halo,
    const val_t* __restrict__ next_halo, int n, int local_rows,
    int first_global_row) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int boundary_rows = local_rows == 1 ? 1 : 2;
    if (linear >= n * boundary_rows) {
        return;
    }
    const int side = linear / n;
    const int column = linear - side * n;
    const int local_row = side == 0 ? 0 : local_rows - 1;
    updateElement(energy, flux, next_energy, next_flux, previous_halo,
                  next_halo, n, local_rows, first_global_row, local_row,
                  column);
}

class DeviceSimulation {
public:
    DeviceSimulation(int n, Slab slab, int rank, int ranks)
        : n_(n), slab_(slab), rank_(rank), ranks_(ranks),
          elements_(static_cast<size_t>(n) * slab.rows) {
        const size_t local_bytes = elements_ * sizeof(val_t);
        const size_t halo_bytes = static_cast<size_t>(n_) * sizeof(val_t);

        CUDA_CHECK(cudaMalloc(&energy_, local_bytes));
        CUDA_CHECK(cudaMalloc(&flux_, local_bytes));
        CUDA_CHECK(cudaMalloc(&next_energy_, local_bytes));
        CUDA_CHECK(cudaMalloc(&next_flux_, local_bytes));
        CUDA_CHECK(cudaMalloc(&previous_halo_, halo_bytes));
        CUDA_CHECK(cudaMalloc(&next_halo_, halo_bytes));
        CUDA_CHECK(cudaHostAlloc(&halo_storage_, 4 * halo_bytes,
                                 cudaHostAllocPortable));

        send_previous_ = halo_storage_;
        send_next_ = send_previous_ + n_;
        receive_previous_ = send_next_ + n_;
        receive_next_ = receive_previous_ + n_;

        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&communication_stream_,
                                              cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&halos_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&iteration_ready_, cudaEventDisableTiming));

        CUDA_CHECK(cudaMemsetAsync(energy_, 0, local_bytes, compute_stream_));
        CUDA_CHECK(cudaMemsetAsync(flux_, 0, local_bytes, compute_stream_));
        CUDA_CHECK(cudaEventRecord(iteration_ready_, compute_stream_));
    }

    DeviceSimulation(const DeviceSimulation&) = delete;
    DeviceSimulation& operator=(const DeviceSimulation&) = delete;

    ~DeviceSimulation() {
        cudaEventDestroy(iteration_ready_);
        cudaEventDestroy(halos_ready_);
        cudaStreamDestroy(communication_stream_);
        cudaStreamDestroy(compute_stream_);
        cudaFreeHost(halo_storage_);
        cudaFree(next_halo_);
        cudaFree(previous_halo_);
        cudaFree(next_flux_);
        cudaFree(next_energy_);
        cudaFree(flux_);
        cudaFree(energy_);
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(iteration_ready_));
    }

    void run(int iterations) {
        const dim3 interior_threads(32, 8);
        if (ranks_ == 1) {
            const dim3 blocks((n_ + interior_threads.x - 1) / interior_threads.x,
                              (n_ + interior_threads.y - 1) / interior_threads.y);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                updateSingleRankKernel<<<blocks, interior_threads, 0,
                                         compute_stream_>>>(
                    energy_, flux_, next_energy_, next_flux_, n_);
                CUDA_CHECK(cudaGetLastError());
                std::swap(energy_, next_energy_);
                std::swap(flux_, next_flux_);
            }
            CUDA_CHECK(cudaEventRecord(iteration_ready_, compute_stream_));
            synchronize();
            return;
        }

        constexpr int boundary_threads = 256;
        const dim3 interior_blocks((n_ + interior_threads.x - 1) /
                                       interior_threads.x,
                                   slab_.rows > 2
                                       ? (slab_.rows - 2 + interior_threads.y - 1) /
                                             interior_threads.y
                                       : 0);
        const int boundary_rows = slab_.rows == 1 ? 1 : 2;
        const int boundary_blocks =
            (n_ * boundary_rows + boundary_threads - 1) / boundary_threads;
        const size_t halo_bytes = static_cast<size_t>(n_) * sizeof(val_t);

        for (int iteration = 0; iteration < iterations; ++iteration) {
            CUDA_CHECK(cudaStreamWaitEvent(communication_stream_, iteration_ready_, 0));

            MPI_Request requests[4];
            int request_count = 0;
            if (rank_ > 0) {
                MPI_CHECK(MPI_Irecv(receive_previous_, n_, MPI_DOUBLE, rank_ - 1,
                                    TAG_TO_NEXT, MPI_COMM_WORLD,
                                    &requests[request_count++]));
                CUDA_CHECK(cudaMemcpyAsync(send_previous_, energy_, halo_bytes,
                                           cudaMemcpyDeviceToHost,
                                           communication_stream_));
            }
            if (rank_ + 1 < ranks_) {
                MPI_CHECK(MPI_Irecv(receive_next_, n_, MPI_DOUBLE, rank_ + 1,
                                    TAG_TO_PREVIOUS, MPI_COMM_WORLD,
                                    &requests[request_count++]));
                CUDA_CHECK(cudaMemcpyAsync(
                    send_next_, energy_ + static_cast<size_t>(slab_.rows - 1) * n_,
                    halo_bytes, cudaMemcpyDeviceToHost, communication_stream_));
            }

            if (slab_.rows > 2) {
                updateInteriorKernel<<<interior_blocks, interior_threads, 0,
                                       compute_stream_>>>(
                    energy_, flux_, next_energy_, next_flux_, n_, slab_.rows,
                    slab_.first_row);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaStreamSynchronize(communication_stream_));
            if (rank_ > 0) {
                MPI_CHECK(MPI_Isend(send_previous_, n_, MPI_DOUBLE, rank_ - 1,
                                    TAG_TO_PREVIOUS, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (rank_ + 1 < ranks_) {
                MPI_CHECK(MPI_Isend(send_next_, n_, MPI_DOUBLE, rank_ + 1,
                                    TAG_TO_NEXT, MPI_COMM_WORLD,
                                    &requests[request_count++]));
            }
            if (request_count != 0) {
                MPI_CHECK(MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE));
            }

            if (rank_ > 0) {
                CUDA_CHECK(cudaMemcpyAsync(previous_halo_, receive_previous_,
                                           halo_bytes, cudaMemcpyHostToDevice,
                                           communication_stream_));
            }
            if (rank_ + 1 < ranks_) {
                CUDA_CHECK(cudaMemcpyAsync(next_halo_, receive_next_, halo_bytes,
                                           cudaMemcpyHostToDevice,
                                           communication_stream_));
            }
            CUDA_CHECK(cudaEventRecord(halos_ready_, communication_stream_));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, halos_ready_, 0));

            updateBoundaryKernel<<<boundary_blocks, boundary_threads, 0,
                                   compute_stream_>>>(
                energy_, flux_, next_energy_, next_flux_, previous_halo_,
                next_halo_, n_, slab_.rows, slab_.first_row);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(iteration_ready_, compute_stream_));

            std::swap(energy_, next_energy_);
            std::swap(flux_, next_flux_);
        }
        synchronize();
    }

    std::vector<ElementDynamic> copyResultToHost() {
        std::vector<ElementDynamic> result(elements_);
        std::vector<val_t> host_energy(elements_);
        std::vector<val_t> host_flux(elements_);
        CUDA_CHECK(cudaMemcpy(host_energy.data(), energy_, elements_ * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_flux.data(), flux_, elements_ * sizeof(val_t),
                              cudaMemcpyDeviceToHost));

#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(elements_); ++i) {
            result[static_cast<size_t>(i)] =
                ElementDynamic{host_energy[static_cast<size_t>(i)],
                               host_flux[static_cast<size_t>(i)]};
        }
        return result;
    }

private:
    int n_;
    Slab slab_;
    int rank_;
    int ranks_;
    size_t elements_;
    val_t* energy_ = nullptr;
    val_t* flux_ = nullptr;
    val_t* next_energy_ = nullptr;
    val_t* next_flux_ = nullptr;
    val_t* previous_halo_ = nullptr;
    val_t* next_halo_ = nullptr;
    val_t* halo_storage_ = nullptr;
    val_t* send_previous_ = nullptr;
    val_t* send_next_ = nullptr;
    val_t* receive_previous_ = nullptr;
    val_t* receive_next_ = nullptr;
    cudaStream_t compute_stream_{};
    cudaStream_t communication_stream_{};
    cudaEvent_t halos_ready_{};
    cudaEvent_t iteration_ready_{};
};

uint64_t computeLocalHash(const std::vector<ElementDynamic>& elements,
                          idx_t global_offset) {
    uint64_t result = 0;
#pragma omp parallel for schedule(static) reduction(^ : result)
    for (int64_t local_index = 0;
         local_index < static_cast<int64_t>(elements.size()); ++local_index) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits,
                    &elements[static_cast<size_t>(local_index)].current_energy,
                    sizeof(energy_bits));
        std::memcpy(&flux_bits,
                    &elements[static_cast<size_t>(local_index)].total_flux,
                    sizeof(flux_bits));
        const uint64_t global_index =
            global_offset + static_cast<uint64_t>(local_index);
        result ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

bool validateResults(const std::vector<ElementDynamic>& local_elements,
                     int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

#pragma omp parallel for schedule(static) reduction(+ : local_energy_sum, local_flux_sum) \
    reduction(max : local_energy_max) reduction(min : local_energy_min)
    for (int64_t i = 0; i < static_cast<int64_t>(local_elements.size()); ++i) {
        const ElementDynamic& element = local_elements[static_cast<size_t>(i)];
        local_energy_sum += element.current_energy;
        local_flux_sum += element.total_flux;
        local_energy_max = std::max(local_energy_max, element.current_energy);
        local_energy_min = std::min(local_energy_min, element.current_energy);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_CHECK(MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
                         MPI_COMM_WORLD));

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(energy_sum) > energy_epsilon) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return valid != 0;
}

void printDistributedResults(const std::vector<ElementDynamic>& local_elements,
                             int n, int rank, int ranks) {
    const int local_count = static_cast<int>(local_elements.size());
    std::vector<val_t> local_energy(local_elements.size());
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(local_elements.size()); ++i) {
        local_energy[static_cast<size_t>(i)] =
            local_elements[static_cast<size_t>(i)].current_energy;
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<val_t> global_energy;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r) {
            const Slab slab = getSlab(n, r, ranks);
            counts[r] = slab.rows * n;
            displacements[r] = slab.first_row * n;
        }
        global_energy.resize(static_cast<size_t>(n) * n);
    }

    MPI_CHECK(MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE,
                          rank == 0 ? global_energy.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                          0, MPI_COMM_WORLD));
    if (rank == 0) {
        print_results(global_energy, "ElementEnergy");
    }
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

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &provided_thread_level));
    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        fatalError("MPI does not provide MPI_THREAD_FUNNELED", __FILE__, __LINE__);
    }

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool print_results_requested = false;
    bool arguments_valid = true;
    bool show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            arguments_valid = false;
        }
    }
    if (show_help || !arguments_valid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }
    if (n <= 0 || iterations < 0 || n < ranks ||
        static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Grid size must be positive, at least the MPI rank count, "
                         "and contain at most INT_MAX elements; iterations must be "
                         "non-negative.\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm node_communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &node_communicator));
    int node_rank = 0;
    MPI_CHECK(MPI_Comm_rank(node_communicator, &node_rank));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fatalError("no CUDA-capable device is available", __FILE__, __LINE__);
    }
    const int device = node_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));

    const Slab slab = getSlab(n, rank, ranks);
    const int64_t total_elements = static_cast<int64_t>(n) * n;
    const size_t local_dynamic_memory =
        static_cast<size_t>(slab.rows) * n * sizeof(val_t) * 4;
    const size_t local_halo_memory = static_cast<size_t>(n) * sizeof(val_t) * 6;

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %" PRId64 " elements\n", n, n,
                    total_elements);
        std::printf("Iterations: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "CUDA GPUs\n", ranks, omp_get_max_threads());
        std::printf("Rank 0 device: %s\n\n", device_properties.name);
        std::printf("Building distributed unstructured mesh...\n");
        std::printf("Rank 0 accelerator memory: %.2f MB (state: %.2f MB, halos: "
                    "%.2f MB)\n\n",
                    (local_dynamic_memory + local_halo_memory) /
                        (1024.0 * 1024.0),
                    local_dynamic_memory / (1024.0 * 1024.0),
                    local_halo_memory / (1024.0 * 1024.0));
    }

    double local_elapsed = 0.0;
    std::vector<ElementDynamic> local_result;
    {
        DeviceSimulation simulation(n, slab, rank, ranks);
        simulation.synchronize();
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (rank == 0) {
            std::printf("Running simulation...\n");
        }
        const double start = MPI_Wtime();
        simulation.run(iterations);
        local_elapsed = MPI_Wtime() - start;
        local_result = simulation.copyResultToHost();
    }
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    const idx_t global_offset = static_cast<idx_t>(slab.first_row) * n;
    const uint64_t local_hash = computeLocalHash(local_result, global_offset);
    uint64_t global_hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        const double elapsed_ms = elapsed * 1000.0;
        const int measured_iterations = std::max(iterations, 1);
        const double time_per_iteration = elapsed_ms / measured_iterations;
        const double giga_elements_per_second =
            elapsed > 0.0
                ? (static_cast<double>(iterations) * total_elements) / elapsed / 1e9
                : 0.0;
        const double gflops = giga_elements_per_second * 22.0;
        std::printf("Computation time: %.3f ms\n", elapsed_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", gflops);
        std::printf("  Result hash: %016" PRIX64 "\n\n", global_hash);
    }

    if (print_results_requested) {
        printDistributedResults(local_result, n, rank, ranks);
    }

    bool valid = true;
    if (validate) {
        valid = validateResults(local_result, rank);
    }

    MPI_CHECK(MPI_Comm_free(&node_communicator));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
