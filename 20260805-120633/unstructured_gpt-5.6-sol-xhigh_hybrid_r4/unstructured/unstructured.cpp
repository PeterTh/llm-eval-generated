#include <algorithm>
#include <cerrno>
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

namespace {

constexpr val_t TRANSFER_COEFF = 0.8;
constexpr val_t INFLOW = 0.5;
constexpr val_t OUTFLOW = -0.5;
constexpr int HALO_TO_PREVIOUS = 100;
constexpr int HALO_TO_NEXT = 101;

struct Options {
    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int print_results = 0;
};

struct Partition {
    int first_row;
    int rows;
};

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseInteger(const char* text, int minimum, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Return 0 for success, 1 for an error, and 2 when help was requested.
int parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseInteger(argv[++i], 1, options.n_elems_root)) {
                std::fprintf(stderr, "Invalid grid size: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseInteger(argv[++i], 0, options.n_iters)) {
                std::fprintf(stderr, "Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.print_results = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 2;
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    const int largest_root = static_cast<int>(
        std::sqrt(static_cast<double>(std::numeric_limits<int>::max())));
    if (options.n_elems_root > largest_root) {
        std::fprintf(stderr,
                     "Grid root is too large for MPI element counts (maximum %d)\n",
                     largest_root);
        return 1;
    }
    return 0;
}

Partition partitionRows(int n, int ranks, int rank) {
    const int base = n / ranks;
    const int remainder = n % ranks;
    return {rank * base + std::min(rank, remainder),
            base + (rank < remainder ? 1 : 0)};
}

[[noreturn]] void abortMpi(MPI_Comm communicator, const char* expression,
                           int error, const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(communicator, &rank);
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: MPI failure at %s:%d: %s: %.*s\n", rank,
                 file, line, expression, length, message);
    MPI_Abort(communicator, error);
    std::abort();
}

[[noreturn]] void abortCuda(MPI_Comm communicator, const char* expression,
                            cudaError_t error, const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d: %s: %s\n", rank,
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(communicator, static_cast<int>(error));
    std::abort();
}

#define MPI_CHECK(comm, call)                                                   \
    do {                                                                        \
        const int mpi_check_error = (call);                                     \
        if (mpi_check_error != MPI_SUCCESS) {                                   \
            abortMpi((comm), #call, mpi_check_error, __FILE__, __LINE__);       \
        }                                                                       \
    } while (false)

#define CUDA_CHECK(comm, call)                                                  \
    do {                                                                        \
        const cudaError_t cuda_check_error = (call);                            \
        if (cuda_check_error != cudaSuccess) {                                  \
            abortCuda((comm), #call, cuda_check_error, __FILE__, __LINE__);     \
        }                                                                       \
    } while (false)

__device__ __forceinline__ val_t connectionFlux(val_t center, val_t neighbor) {
    return (neighbor - center) * TRANSFER_COEFF * 1.0 * 0.25;
}

__device__ __forceinline__ val_t externalFlow(int row, int column, int n) {
    if ((row == 0 && column == 0) ||
        (row == n - 1 && column == n - 1)) {
        return INFLOW;
    }
    if ((row == 0 && column == n - 1) ||
        (row == n - 1 && column == 0)) {
        return OUTFLOW;
    }
    return 0.0;
}

__device__ __forceinline__ void updateElement(
    const val_t* __restrict__ current, val_t* __restrict__ next,
    val_t* __restrict__ accumulated_flux, int n, int first_global_row,
    int local_row, int column) {
    const size_t stride = static_cast<size_t>(n);
    const size_t cell = static_cast<size_t>(local_row) * stride + column;
    const int global_row = first_global_row + local_row - 1;
    const val_t center = current[cell];
    val_t total_flux = externalFlow(global_row, column, n);

    // Match the original connectivity order: +x, -x, +y, -y.
    if (global_row + 1 < n) {
        total_flux += connectionFlux(center, current[cell + stride]);
    }
    if (global_row > 0) {
        total_flux += connectionFlux(center, current[cell - stride]);
    }
    if (column + 1 < n) {
        total_flux += connectionFlux(center, current[cell + 1]);
    }
    if (column > 0) {
        total_flux += connectionFlux(center, current[cell - 1]);
    }

    const size_t local_cell = static_cast<size_t>(local_row - 1) * stride + column;
    next[cell] = center + total_flux;
    accumulated_flux[local_cell] += fabs(total_flux);
}

__global__ __launch_bounds__(256) void updateRowsKernel(
    const val_t* __restrict__ current, val_t* __restrict__ next,
    val_t* __restrict__ accumulated_flux, int n, int first_global_row,
    int first_local_row) {
    const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= n) {
        return;
    }
    const int local_row = first_local_row + static_cast<int>(blockIdx.y);
    updateElement(current, next, accumulated_flux, n, first_global_row,
                  local_row, column);
}

__global__ __launch_bounds__(256) void updateBoundaryKernel(
    const val_t* __restrict__ current, val_t* __restrict__ next,
    val_t* __restrict__ accumulated_flux, int n, int local_rows,
    int first_global_row) {
    const int column = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= n) {
        return;
    }
    const int boundary = static_cast<int>(blockIdx.y);
    const int local_row = boundary == 0 ? 1 : local_rows;
    updateElement(current, next, accumulated_flux, n, first_global_row,
                  local_row, column);
}

class DistributedSimulation {
public:
    DistributedSimulation(MPI_Comm communicator, int n, Partition partition)
        : communicator_(communicator), n_(n), partition_(partition),
          row_bytes_(static_cast<size_t>(n) * sizeof(val_t)),
          local_elements_(static_cast<size_t>(partition.rows) * n),
          grid_elements_(static_cast<size_t>(partition.rows + 2) * n) {
        int rank = 0;
        MPI_CHECK(communicator_, MPI_Comm_rank(communicator_, &rank));
        previous_rank_ = rank == 0 ? MPI_PROC_NULL : rank - 1;
        int size = 0;
        MPI_CHECK(communicator_, MPI_Comm_size(communicator_, &size));
        next_rank_ = rank + 1 == size ? MPI_PROC_NULL : rank + 1;

        CUDA_CHECK(communicator_, cudaStreamCreateWithFlags(&compute_stream_,
                                                            cudaStreamNonBlocking));
        CUDA_CHECK(communicator_, cudaStreamCreateWithFlags(&halo_stream_,
                                                            cudaStreamNonBlocking));
        CUDA_CHECK(communicator_, cudaMalloc(&current_, grid_elements_ * sizeof(val_t)));
        CUDA_CHECK(communicator_, cudaMalloc(&next_, grid_elements_ * sizeof(val_t)));
        CUDA_CHECK(communicator_,
                   cudaMalloc(&accumulated_flux_, local_elements_ * sizeof(val_t)));
        CUDA_CHECK(communicator_, cudaMemset(current_, 0, grid_elements_ * sizeof(val_t)));
        CUDA_CHECK(communicator_, cudaMemset(next_, 0, grid_elements_ * sizeof(val_t)));
        CUDA_CHECK(communicator_,
                   cudaMemset(accumulated_flux_, 0, local_elements_ * sizeof(val_t)));

        CUDA_CHECK(communicator_, cudaHostAlloc(&halo_storage_, 4 * row_bytes_,
                                                cudaHostAllocPortable));
        val_t* rows = static_cast<val_t*>(halo_storage_);
        send_previous_ = rows;
        send_next_ = rows + n_;
        receive_previous_ = rows + 2 * n_;
        receive_next_ = rows + 3 * n_;

        if (previous_rank_ != MPI_PROC_NULL) {
            MPI_CHECK(communicator_,
                      MPI_Recv_init(receive_previous_, n_, MPI_DOUBLE,
                                    previous_rank_, HALO_TO_NEXT, communicator_,
                                    &halo_requests_[halo_request_count_++]));
            MPI_CHECK(communicator_,
                      MPI_Send_init(send_previous_, n_, MPI_DOUBLE,
                                    previous_rank_, HALO_TO_PREVIOUS,
                                    communicator_,
                                    &halo_requests_[halo_request_count_++]));
        }
        if (next_rank_ != MPI_PROC_NULL) {
            MPI_CHECK(communicator_,
                      MPI_Recv_init(receive_next_, n_, MPI_DOUBLE, next_rank_,
                                    HALO_TO_PREVIOUS, communicator_,
                                    &halo_requests_[halo_request_count_++]));
            MPI_CHECK(communicator_,
                      MPI_Send_init(send_next_, n_, MPI_DOUBLE, next_rank_,
                                    HALO_TO_NEXT, communicator_,
                                    &halo_requests_[halo_request_count_++]));
        }
    }

    DistributedSimulation(const DistributedSimulation&) = delete;
    DistributedSimulation& operator=(const DistributedSimulation&) = delete;

    ~DistributedSimulation() {
        // Destructors run only on a healthy path; errors are handled at each operation.
        for (int i = 0; i < halo_request_count_; ++i) {
            MPI_Request_free(&halo_requests_[i]);
        }
        if (halo_storage_ != nullptr) {
            cudaFreeHost(halo_storage_);
        }
        if (accumulated_flux_ != nullptr) {
            cudaFree(accumulated_flux_);
        }
        if (next_ != nullptr) {
            cudaFree(next_);
        }
        if (current_ != nullptr) {
            cudaFree(current_);
        }
        if (halo_stream_ != nullptr) {
            cudaStreamDestroy(halo_stream_);
        }
        if (compute_stream_ != nullptr) {
            cudaStreamDestroy(compute_stream_);
        }
    }

    size_t allocatedBytes() const {
        return (2 * grid_elements_ + local_elements_) * sizeof(val_t) +
               4 * row_bytes_;
    }

    void run(int iterations) {
        constexpr int threads = 256;
        const int interior_rows = std::max(partition_.rows - 2, 0);
        const int boundary_rows = partition_.rows == 1 ? 1 : 2;
        const unsigned int column_blocks =
            static_cast<unsigned int>((n_ + threads - 1) / threads);

        // A single-rank run has no communication dependency and needs one
        // launch per timestep instead of separate interior/boundary launches.
        if (previous_rank_ == MPI_PROC_NULL && next_rank_ == MPI_PROC_NULL) {
            const dim3 blocks(column_blocks,
                              static_cast<unsigned int>(partition_.rows));
            for (int iteration = 0; iteration < iterations; ++iteration) {
                updateRowsKernel<<<blocks, threads, 0, compute_stream_>>>(
                    current_, next_, accumulated_flux_, n_, partition_.first_row,
                    1);
                CUDA_CHECK(communicator_, cudaGetLastError());
                CUDA_CHECK(communicator_,
                           cudaStreamSynchronize(compute_stream_));
                std::swap(current_, next_);
            }
            return;
        }

        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (previous_rank_ != MPI_PROC_NULL) {
                CUDA_CHECK(communicator_,
                           cudaMemcpyAsync(send_previous_, current_ + n_, row_bytes_,
                                           cudaMemcpyDeviceToHost, halo_stream_));
            }
            if (next_rank_ != MPI_PROC_NULL) {
                const size_t last_row = static_cast<size_t>(partition_.rows) * n_;
                CUDA_CHECK(communicator_,
                           cudaMemcpyAsync(send_next_, current_ + last_row, row_bytes_,
                                           cudaMemcpyDeviceToHost, halo_stream_));
            }

            // Interior work is independent of incoming halo rows and overlaps staging/MPI.
            if (interior_rows != 0) {
                const dim3 blocks(column_blocks,
                                  static_cast<unsigned int>(interior_rows));
                updateRowsKernel<<<blocks, threads, 0, compute_stream_>>>(
                    current_, next_, accumulated_flux_, n_, partition_.first_row,
                    2);
                CUDA_CHECK(communicator_, cudaGetLastError());
            }

            CUDA_CHECK(communicator_, cudaStreamSynchronize(halo_stream_));

            MPI_CHECK(communicator_,
                      MPI_Startall(halo_request_count_, halo_requests_));
            MPI_CHECK(communicator_,
                      MPI_Waitall(halo_request_count_, halo_requests_,
                                  MPI_STATUSES_IGNORE));

            if (previous_rank_ != MPI_PROC_NULL) {
                CUDA_CHECK(communicator_,
                           cudaMemcpyAsync(current_, receive_previous_, row_bytes_,
                                           cudaMemcpyHostToDevice, halo_stream_));
            }
            if (next_rank_ != MPI_PROC_NULL) {
                const size_t bottom_halo =
                    static_cast<size_t>(partition_.rows + 1) * n_;
                CUDA_CHECK(communicator_,
                           cudaMemcpyAsync(current_ + bottom_halo, receive_next_,
                                           row_bytes_, cudaMemcpyHostToDevice,
                                           halo_stream_));
            }

            const dim3 boundary_blocks(
                column_blocks, static_cast<unsigned int>(boundary_rows));
            updateBoundaryKernel<<<boundary_blocks, threads, 0, halo_stream_>>>(
                current_, next_, accumulated_flux_, n_, partition_.rows,
                partition_.first_row);
            CUDA_CHECK(communicator_, cudaGetLastError());

            CUDA_CHECK(communicator_, cudaStreamSynchronize(compute_stream_));
            CUDA_CHECK(communicator_, cudaStreamSynchronize(halo_stream_));
            std::swap(current_, next_);
        }
    }

    void copyResults(std::vector<val_t>& energy,
                     std::vector<val_t>& accumulated_flux) const {
        energy.resize(local_elements_);
        accumulated_flux.resize(local_elements_);
        CUDA_CHECK(communicator_,
                   cudaMemcpy(energy.data(), current_ + n_,
                              local_elements_ * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(communicator_,
                   cudaMemcpy(accumulated_flux.data(), accumulated_flux_,
                              local_elements_ * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

private:
    MPI_Comm communicator_;
    int n_;
    Partition partition_;
    int previous_rank_ = MPI_PROC_NULL;
    int next_rank_ = MPI_PROC_NULL;
    size_t row_bytes_;
    size_t local_elements_;
    size_t grid_elements_;
    val_t* current_ = nullptr;
    val_t* next_ = nullptr;
    val_t* accumulated_flux_ = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t halo_stream_ = nullptr;
    void* halo_storage_ = nullptr;
    val_t* send_previous_ = nullptr;
    val_t* send_next_ = nullptr;
    val_t* receive_previous_ = nullptr;
    val_t* receive_next_ = nullptr;
    MPI_Request halo_requests_[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                                     MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int halo_request_count_ = 0;
};

uint64_t bitsOf(val_t value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "Unexpected floating-point size");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t computeDistributedHash(const std::vector<val_t>& energy,
                                const std::vector<val_t>& flux,
                                size_t first_global_element) {
    uint64_t local_hash = 0;
    const size_t count = energy.size();
#pragma omp parallel for schedule(static) reduction(^ : local_hash)
    for (size_t local_index = 0; local_index < count; ++local_index) {
        const size_t global_index = first_global_element + local_index;
        local_hash ^=
            (bitsOf(energy[local_index]) + global_index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^=
            (bitsOf(flux[local_index]) + global_index) * 0xbf58476d1ce4e5b9ULL;
    }
    return local_hash;
}

void gatherField(const std::vector<val_t>& local, std::vector<val_t>& global,
                 int n, int ranks, int rank, MPI_Comm communicator) {
    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        counts.resize(ranks);
        displacements.resize(ranks);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r) {
            const Partition partition = partitionRows(n, ranks, r);
            counts[r] = partition.rows * n;
            displacements[r] = partition.first_row * n;
        }
        global.resize(static_cast<size_t>(n) * n);
    }
    MPI_CHECK(communicator,
              MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                          rank == 0 ? global.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                          communicator));
}

bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
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

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = 0;
    int init_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                                     &provided_thread_level);
    if (init_error != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return 1;
    }

    int world_rank = 0;
    int world_size = 0;
    MPI_CHECK(MPI_COMM_WORLD, MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_COMM_WORLD, MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_COMM_WORLD,
              MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) {
            std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    Options options;
    int parse_result = 0;
    if (world_rank == 0) {
        parse_result = parseOptions(argc, argv, options);
    }
    MPI_CHECK(MPI_COMM_WORLD,
              MPI_Bcast(&parse_result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (parse_result != 0) {
        MPI_Finalize();
        return parse_result == 2 ? 0 : 1;
    }
    MPI_CHECK(MPI_COMM_WORLD,
              MPI_Bcast(&options, sizeof(options), MPI_BYTE, 0, MPI_COMM_WORLD));

    const int active_ranks = std::min(world_size, options.n_elems_root);
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_COMM_WORLD,
              MPI_Comm_split(MPI_COMM_WORLD,
                             world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                             world_rank, &communicator));
    if (communicator == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_CHECK(communicator,
              MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN));

    int rank = 0;
    int ranks = 0;
    MPI_CHECK(communicator, MPI_Comm_rank(communicator, &rank));
    MPI_CHECK(communicator, MPI_Comm_size(communicator, &ranks));

    MPI_Comm node_communicator = MPI_COMM_NULL;
    MPI_CHECK(communicator,
              MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &node_communicator));
    int local_rank = 0;
    int local_size = 0;
    MPI_CHECK(node_communicator,
              MPI_Comm_rank(node_communicator, &local_rank));
    MPI_CHECK(node_communicator,
              MPI_Comm_size(node_communicator, &local_size));

    int device_count = 0;
    CUDA_CHECK(communicator, cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA-capable devices are available\n");
        }
        MPI_Abort(communicator, 1);
        return 1;
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(communicator, cudaSetDevice(device));
    CUDA_CHECK(communicator, cudaFree(nullptr));

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int threads_per_rank = std::max(1, omp_get_num_procs() / local_size);
        omp_set_num_threads(threads_per_rank);
    }

    const Partition partition =
        partitionRows(options.n_elems_root, ranks, rank);
    const size_t global_elements =
        static_cast<size_t>(options.n_elems_root) * options.n_elems_root;

    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %zu elements\n",
                    options.n_elems_root, options.n_elems_root, global_elements);
        std::printf("Iterations: %d\n", options.n_iters);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("Parallel execution: %d MPI rank(s), OpenMP host threads, "
                    "%d CUDA device(s)/node\n\n",
                    ranks, device_count);
        if (local_size > device_count) {
            std::printf("WARNING: %d local MPI ranks share %d CUDA devices\n\n",
                        local_size, device_count);
        }
        std::printf("Building distributed mesh...\n");
    }

    int result = 0;
    {
        DistributedSimulation simulation(communicator, options.n_elems_root,
                                         partition);
        const unsigned long long local_bytes = simulation.allocatedBytes();
        unsigned long long global_bytes = 0;
        MPI_CHECK(communicator,
                  MPI_Reduce(&local_bytes, &global_bytes, 1,
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator));
        if (rank == 0) {
            std::printf("Aggregate accelerator/staging memory: %.2f MB\n\n",
                        global_bytes / (1024.0 * 1024.0));
            std::printf("Running simulation...\n");
        }

        MPI_CHECK(communicator, MPI_Barrier(communicator));
        const double start = MPI_Wtime();
        simulation.run(options.n_iters);
        const double local_seconds = MPI_Wtime() - start;
        double elapsed_seconds = 0.0;
        MPI_CHECK(communicator,
                  MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE,
                             MPI_MAX, 0, communicator));

        std::vector<val_t> local_energy;
        std::vector<val_t> local_flux;
        simulation.copyResults(local_energy, local_flux);

        const size_t first_global_element =
            static_cast<size_t>(partition.first_row) * options.n_elems_root;
        const uint64_t local_hash = computeDistributedHash(
            local_energy, local_flux, first_global_element);
        uint64_t global_hash = 0;
        MPI_CHECK(communicator,
                  MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
                             0, communicator));

        std::vector<val_t> global_energy;
        std::vector<val_t> global_flux;
        if (options.print_results || options.validate) {
            gatherField(local_energy, global_energy, options.n_elems_root, ranks,
                        rank, communicator);
        }
        if (options.validate) {
            gatherField(local_flux, global_flux, options.n_elems_root, ranks, rank,
                        communicator);
        }

        if (rank == 0) {
            const double duration_ms = elapsed_seconds * 1000.0;
            std::printf("Computation time: %.3f ms\n", duration_ms);
            const int measured_iterations = std::max(options.n_iters, 1);
            const double time_per_iteration = duration_ms / measured_iterations;
            const double giga_elements_per_second =
                elapsed_seconds > 0.0
                    ? (static_cast<double>(options.n_iters) * global_elements) /
                          elapsed_seconds / 1.0e9
                    : 0.0;
            const double gflops = giga_elements_per_second * 22.0;
            std::printf("Performance:\n");
            std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
            std::printf("  Elements/sec: %.4f GigaElements/s\n",
                        giga_elements_per_second);
            std::printf("  Performance: %.4f GFLOPS\n", gflops);
            std::printf("  Result hash: %016llX\n\n",
                        static_cast<unsigned long long>(global_hash));

            if (options.print_results) {
                print_results(global_energy, "ElementEnergy");
            }
            if (options.validate && !validateResults(global_energy, global_flux)) {
                result = 1;
            }
        }
        MPI_CHECK(communicator,
                  MPI_Bcast(&result, 1, MPI_INT, 0, communicator));
    }

    MPI_CHECK(node_communicator, MPI_Comm_free(&node_communicator));
    MPI_CHECK(communicator, MPI_Comm_free(&communicator));
    MPI_Finalize();
    return result;
}
