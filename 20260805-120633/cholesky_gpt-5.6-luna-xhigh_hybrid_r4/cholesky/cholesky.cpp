#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous row blocks.  A block step consists
// of a small panel factorization followed by a GPU triangular solve and a
// distributed trailing update.  Keeping the panel width fixed amortizes MPI
// collectives and CUDA launch overhead while retaining a simple, scalable
// ownership scheme.
constexpr std::size_t kBlockSize = 128;
constexpr unsigned kUpdateTileX = 32;
constexpr unsigned kUpdateTileY = 8;

[[noreturn]] void abortWorld(const char* message, const int error_code = 1) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, error_code);
    std::abort();
}

[[noreturn]] void abortWorld(const char* operation, const cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_status__ = (call);                              \
        if (cuda_status__ != cudaSuccess) {                                    \
            abortWorld(#call, cuda_status__);                                  \
        }                                                                       \
    } while (false)

#define CUDA_KERNEL_CHECK()                                                     \
    do {                                                                        \
        const cudaError_t cuda_launch_status__ = cudaGetLastError();            \
        if (cuda_launch_status__ != cudaSuccess) {                              \
            abortWorld("CUDA kernel launch", cuda_launch_status__);            \
        }                                                                       \
        const cudaError_t cuda_sync_status__ = cudaDeviceSynchronize();         \
        if (cuda_sync_status__ != cudaSuccess) {                                \
            abortWorld("CUDA kernel execution", cuda_sync_status__);           \
        }                                                                       \
    } while (false)

__global__ void solvePanelKernel(double* matrix,
                                 const double* diagonal,
                                 const std::size_t n,
                                 const std::size_t row_begin,
                                 const std::size_t local_rows,
                                 const std::size_t k,
                                 const std::size_t block_size) {
    const std::size_t local_row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                   threadIdx.x;
    if (local_row >= local_rows) {
        return;
    }

    const std::size_t global_row = row_begin + local_row;
    if (global_row < k + block_size) {
        return;
    }

    // Solve x * L^T = panel_row, one independent row per CUDA thread.
    for (std::size_t q = 0; q < block_size; ++q) {
        double value = matrix[local_row * n + k + q];
        for (std::size_t p = 0; p < q; ++p) {
            value -= matrix[local_row * n + k + p] * diagonal[q * block_size + p];
        }
        matrix[local_row * n + k + q] = value / diagonal[q * block_size + q];
    }
}

__global__ void updateTrailingBlockKernel(double* matrix,
                                          const double* panel,
                                          const std::size_t n,
                                          const std::size_t row_begin,
                                          const std::size_t local_rows,
                                          const std::size_t k,
                                          const std::size_t block_size) {
    // Tile the rank-k update in shared memory.  Each output element reuses
    // the same two panel rows, so this substantially reduces global-memory
    // traffic compared with loading both rows for every output thread.
    __shared__ double row_panel[kUpdateTileY][kUpdateTileX];
    __shared__ double column_panel[kUpdateTileX][kUpdateTileX];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const std::size_t local_row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + ty;
    const std::size_t global_row = row_begin + local_row;
    const std::size_t global_col = k + block_size +
                                   static_cast<std::size_t>(blockIdx.x) * blockDim.x + tx;

    double dot = 0.0;
    const bool valid_row = local_row < local_rows && global_row < n &&
                           global_row >= k + block_size;
    const bool valid_column = global_col < n && global_col >= k + block_size;
    for (std::size_t q_base = 0; q_base < block_size; q_base += kUpdateTileX) {
        const std::size_t row_q = q_base + tx;
        row_panel[ty][tx] = (valid_row && row_q < block_size)
                                ? panel[global_row * block_size + row_q]
                                : 0.0;

        for (std::size_t column_q = ty; column_q < kUpdateTileX; column_q += kUpdateTileY) {
            const std::size_t q = q_base + column_q;
            column_panel[tx][column_q] = (valid_column && q < block_size)
                                             ? panel[global_col * block_size + q]
                                             : 0.0;
        }
        __syncthreads();

        const std::size_t remaining_q = block_size - q_base;
        const std::size_t q_count = remaining_q < kUpdateTileX ? remaining_q : kUpdateTileX;
        for (std::size_t q = 0; q < q_count; ++q) {
            dot += row_panel[ty][q] * column_panel[tx][q];
        }
        __syncthreads();
    }

    if (valid_row && valid_column && global_col <= global_row) {
        matrix[local_row * n + global_col] -= dot;
    }
}

__global__ void zeroUpperKernel(double* matrix,
                                const std::size_t n,
                                const std::size_t row_begin,
                                const std::size_t local_rows) {
    const std::size_t local_row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                   threadIdx.x;
    if (local_row >= local_rows) {
        return;
    }

    const std::size_t first_upper = row_begin + local_row + 1;
    for (std::size_t col = first_upper; col < n; ++col) {
        matrix[local_row * n + col] = 0.0;
    }
}

void generatePositiveDefiniteMatrix(std::vector<double>& matrix, const std::size_t n) {
    // Preserve the original deterministic input.  The random stream is kept
    // serial so that -r remains reproducible; the O(n^3) product is threaded.
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (double& value : b) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += b[static_cast<std::size_t>(i) * n + k] * b[j * n + k];
            }
            matrix[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        matrix[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)] +=
            static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original,
                      const std::size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += lower[static_cast<std::size_t>(i) * n + k] *
                       lower[j * n + k];
            }
            reconstructed[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double max_relative_error = 0.0;
    #pragma omp parallel for reduction(max:max_error,max_relative_error) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n * n); ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const double error = std::fabs(reconstructed[index] - original[index]);
        max_error = std::max(max_error, error);
        const double relative = error / (std::fabs(original[index]) + 1e-10);
        max_relative_error = std::max(max_relative_error, relative);
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", max_relative_error);
    if (max_relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

std::size_t parseMatrixSize(const char* text) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return 0;
    }
    return static_cast<std::size_t>(parsed);
}

void checkMpiIntegerRange(const std::vector<std::size_t>& row_offsets,
                          const std::size_t n,
                          const int world_size) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n * n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        abortWorld("Matrix dimensions exceed the MPI count range");
    }
    for (int rank = 0; rank < world_size; ++rank) {
        const std::size_t rows = row_offsets[static_cast<std::size_t>(rank) + 1] -
                                 row_offsets[static_cast<std::size_t>(rank)];
        if (rows * n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            abortWorld("A distributed row block exceeds the MPI count range");
        }
    }
}

bool distributedCholesky(std::vector<double>& local_matrix,
                         const std::size_t n,
                         const std::size_t row_begin,
                         const std::size_t local_rows,
                         const std::vector<int>& row_counts,
                         const std::vector<int>& row_displacements,
                         const int rank,
                         const int world_size) {
    const std::size_t matrix_elements = std::max<std::size_t>(1, local_matrix.size());
    double* device_matrix = nullptr;
    double* device_diagonal = nullptr;
    double* device_panel = nullptr;

    CUDA_CHECK(cudaMalloc(&device_matrix, matrix_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_diagonal, kBlockSize * kBlockSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_panel, n * kBlockSize * sizeof(double)));
    if (local_rows != 0) {
        CUDA_CHECK(cudaMemcpy(device_matrix,
                              local_matrix.data(),
                              local_matrix.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    std::vector<int> panel_counts(static_cast<std::size_t>(world_size));
    std::vector<int> panel_displacements(static_cast<std::size_t>(world_size));
    std::vector<int> factor_counts(static_cast<std::size_t>(world_size));
    std::vector<int> factor_displacements(static_cast<std::size_t>(world_size));
    std::vector<double> diagonal(kBlockSize * kBlockSize);
    std::vector<double> local_panel;
    std::vector<double> local_factors;
    std::vector<double> gathered_panel;
    std::vector<double> global_factors(n * kBlockSize);

    bool success = true;
    for (std::size_t k = 0; k < n && success; k += kBlockSize) {
        const std::size_t block_size = std::min(kBlockSize, n - k);
        const std::size_t panel_elements = local_rows * block_size;
        local_panel.resize(panel_elements);
        local_factors.resize(panel_elements);

        for (int process = 0; process < world_size; ++process) {
            const std::size_t process_rows =
                static_cast<std::size_t>(row_counts[static_cast<std::size_t>(process)]);
            const std::size_t process_offset =
                static_cast<std::size_t>(row_displacements[static_cast<std::size_t>(process)]);
            const std::size_t count = process_rows * block_size;
            const std::size_t displacement = process_offset * block_size;
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                displacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                abortWorld("Panel dimensions exceed the MPI count range");
            }
            panel_counts[static_cast<std::size_t>(process)] = static_cast<int>(count);
            panel_displacements[static_cast<std::size_t>(process)] =
                static_cast<int>(displacement);
            factor_counts[static_cast<std::size_t>(process)] = static_cast<int>(count);
            factor_displacements[static_cast<std::size_t>(process)] =
                static_cast<int>(displacement);
        }

        if (local_rows != 0) {
            CUDA_CHECK(cudaMemcpy2D(local_panel.data(),
                                    block_size * sizeof(double),
                                    device_matrix + k,
                                    n * sizeof(double),
                                    block_size * sizeof(double),
                                    local_rows,
                                    cudaMemcpyDeviceToHost));
        }

        if (rank == 0) {
            gathered_panel.resize(n * block_size);
        }
        MPI_Gatherv(local_panel.data(),
                    static_cast<int>(panel_elements),
                    MPI_DOUBLE,
                    rank == 0 ? gathered_panel.data() : nullptr,
                    panel_counts.data(),
                    panel_displacements.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            std::fill(diagonal.begin(), diagonal.begin() + block_size * block_size, 0.0);
            for (std::size_t i = 0; i < block_size; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    double sum = gathered_panel[(k + i) * block_size + j];
                    for (std::size_t p = 0; p < j; ++p) {
                        sum -= gathered_panel[(k + i) * block_size + p] *
                               gathered_panel[(k + j) * block_size + p];
                    }
                    if (i == j) {
                        if (sum <= 0.0) {
                            success = false;
                            break;
                        }
                        diagonal[i * block_size + j] = std::sqrt(sum);
                        gathered_panel[(k + i) * block_size + j] =
                            diagonal[i * block_size + j];
                    } else {
                        diagonal[i * block_size + j] =
                            sum / diagonal[j * block_size + j];
                        gathered_panel[(k + i) * block_size + j] =
                            diagonal[i * block_size + j];
                    }
                }
                if (!success) {
                    break;
                }
            }
        }

        int success_flag = success ? 1 : 0;
        MPI_Bcast(&success_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        success = success_flag != 0;
        if (!success) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at block starting %zu\n", k);
            }
            break;
        }

        MPI_Bcast(diagonal.data(),
                  static_cast<int>(block_size * block_size),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(device_diagonal,
                              diagonal.data(),
                              block_size * block_size * sizeof(double),
                              cudaMemcpyHostToDevice));

        const unsigned solve_threads = 256;
        const unsigned solve_blocks =
            static_cast<unsigned>((local_rows + solve_threads - 1) / solve_threads);
        if (solve_blocks != 0) {
            solvePanelKernel<<<solve_blocks, solve_threads>>>(device_matrix,
                                                               device_diagonal,
                                                               n,
                                                               row_begin,
                                                               local_rows,
                                                               k,
                                                               block_size);
            CUDA_KERNEL_CHECK();
        }

        if (local_rows != 0) {
            CUDA_CHECK(cudaMemcpy2D(local_factors.data(),
                                    block_size * sizeof(double),
                                    device_matrix + k,
                                    n * sizeof(double),
                                    block_size * sizeof(double),
                                    local_rows,
                                    cudaMemcpyDeviceToHost));
        }

        // The diagonal block was factored on rank 0.  Install the common
        // diagonal-block result into whichever row block owns each row.
        #pragma omp parallel for schedule(static)
        for (std::int64_t local = 0; local < static_cast<std::int64_t>(local_rows); ++local) {
            const std::size_t global_row = row_begin + static_cast<std::size_t>(local);
            double* row = local_factors.data() + static_cast<std::size_t>(local) * block_size;
            if (global_row < k || global_row >= k + block_size) {
                if (global_row < k) {
                    std::fill(row, row + block_size, 0.0);
                }
                continue;
            }
            const std::size_t diagonal_row = global_row - k;
            for (std::size_t column = 0; column < block_size; ++column) {
                row[column] = column <= diagonal_row
                                  ? diagonal[diagonal_row * block_size + column]
                                  : 0.0;
            }
        }

        // Make the completed panel authoritative in device memory.  This is
        // needed both for the final result and for the next block's Schur
        // complement, while retaining the original row-major layout.
        if (local_rows != 0) {
            CUDA_CHECK(cudaMemcpy2D(device_matrix + k,
                                    n * sizeof(double),
                                    local_factors.data(),
                                    block_size * sizeof(double),
                                    block_size * sizeof(double),
                                    local_rows,
                                    cudaMemcpyHostToDevice));
        }

        MPI_Allgatherv(local_factors.data(),
                       static_cast<int>(panel_elements),
                       MPI_DOUBLE,
                       global_factors.data(),
                       factor_counts.data(),
                       factor_displacements.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(device_panel,
                              global_factors.data(),
                              n * block_size * sizeof(double),
                              cudaMemcpyHostToDevice));

        const std::size_t trailing = n - (k + block_size);
        const unsigned grid_x = static_cast<unsigned>((trailing + kUpdateTileX - 1) /
                                                       kUpdateTileX);
        const unsigned grid_y = static_cast<unsigned>((local_rows + kUpdateTileY - 1) /
                                                       kUpdateTileY);
        if (grid_x != 0 && grid_y != 0) {
            updateTrailingBlockKernel<<<dim3(grid_x, grid_y),
                                        dim3(kUpdateTileX, kUpdateTileY)>>>(
                device_matrix,
                device_panel,
                n,
                row_begin,
                local_rows,
                k,
                block_size);
            CUDA_KERNEL_CHECK();
        }
    }

    if (success) {
        const unsigned zero_threads = 256;
        const unsigned zero_blocks =
            static_cast<unsigned>((local_rows + zero_threads - 1) / zero_threads);
        if (zero_blocks != 0) {
            zeroUpperKernel<<<zero_blocks, zero_threads>>>(device_matrix,
                                                           n,
                                                           row_begin,
                                                           local_rows);
            CUDA_KERNEL_CHECK();
        }
        if (local_rows != 0) {
            CUDA_CHECK(cudaMemcpy(local_matrix.data(),
                                  device_matrix,
                                  local_matrix.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
    }

    CUDA_CHECK(cudaFree(device_panel));
    CUDA_CHECK(cudaFree(device_diagonal));
    CUDA_CHECK(cudaFree(device_matrix));
    return success;
}

int main(int argc, char** argv) {
    std::size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = parseMatrixSize(argv[++i]);
            if (n == 0) {
                std::printf("Invalid matrix size\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int provided_thread_level = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level) != MPI_SUCCESS) {
        return 1;
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        abortWorld("MPI implementation did not provide MPI_THREAD_FUNNELED");
    }

    omp_set_dynamic(0);

    MPI_Comm shared_memory_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        rank,
                        MPI_INFO_NULL,
                        &shared_memory_comm);
    int local_rank = 0;
    MPI_Comm_rank(shared_memory_comm, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        abortWorld("No CUDA device is visible to this MPI rank");
    }
    const int selected_device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(selected_device));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    std::vector<std::size_t> row_offsets(static_cast<std::size_t>(world_size) + 1, 0);
    for (int process = 0; process < world_size; ++process) {
        const std::size_t process_rows = n / static_cast<std::size_t>(world_size) +
                                         (static_cast<std::size_t>(process) <
                                          n % static_cast<std::size_t>(world_size));
        row_offsets[static_cast<std::size_t>(process) + 1] =
            row_offsets[static_cast<std::size_t>(process)] + process_rows;
    }
    checkMpiIntegerRange(row_offsets, n, world_size);

    std::vector<int> row_counts(static_cast<std::size_t>(world_size));
    std::vector<int> row_displacements(static_cast<std::size_t>(world_size));
    std::vector<int> matrix_counts(static_cast<std::size_t>(world_size));
    std::vector<int> matrix_displacements(static_cast<std::size_t>(world_size));
    for (int process = 0; process < world_size; ++process) {
        const std::size_t process_rows =
            row_offsets[static_cast<std::size_t>(process) + 1] -
            row_offsets[static_cast<std::size_t>(process)];
        const std::size_t process_begin = row_offsets[static_cast<std::size_t>(process)];
        row_counts[static_cast<std::size_t>(process)] = static_cast<int>(process_rows);
        row_displacements[static_cast<std::size_t>(process)] = static_cast<int>(process_begin);
        matrix_counts[static_cast<std::size_t>(process)] =
            static_cast<int>(process_rows * n);
        matrix_displacements[static_cast<std::size_t>(process)] =
            static_cast<int>(process_begin * n);
    }

    const std::size_t local_begin = row_offsets[static_cast<std::size_t>(rank)];
    const std::size_t local_rows =
        row_offsets[static_cast<std::size_t>(rank) + 1] - local_begin;
    std::vector<double> matrix;
    std::vector<double> original;
    if (rank == 0) {
        matrix.resize(n * n);
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(matrix, n);
        if (validate) {
            original = matrix;
        }
    }

    std::vector<double> local_matrix(local_rows * n);
    MPI_Scatterv(rank == 0 ? matrix.data() : nullptr,
                 matrix_counts.data(),
                 matrix_displacements.data(),
                 MPI_DOUBLE,
                 local_matrix.data(),
                 static_cast<int>(local_matrix.size()),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(local_matrix,
                                             n,
                                             local_begin,
                                             local_rows,
                                             row_counts,
                                             row_displacements,
                                             rank,
                                             world_size);
    const double local_seconds = MPI_Wtime() - start;
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds,
               &elapsed_seconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    int success_flag = success ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &success_flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (success_flag == 0) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&shared_memory_comm);
        MPI_Finalize();
        return 1;
    }

    std::vector<int> gathered_matrix_counts(static_cast<std::size_t>(world_size));
    std::vector<int> gathered_matrix_displacements(static_cast<std::size_t>(world_size));
    for (int process = 0; process < world_size; ++process) {
        gathered_matrix_counts[static_cast<std::size_t>(process)] =
            matrix_counts[static_cast<std::size_t>(process)];
        gathered_matrix_displacements[static_cast<std::size_t>(process)] =
            matrix_displacements[static_cast<std::size_t>(process)];
    }
    if (rank == 0) {
        matrix.resize(n * n);
    }
    MPI_Gatherv(local_matrix.data(),
                static_cast<int>(local_matrix.size()),
                MPI_DOUBLE,
                rank == 0 ? matrix.data() : nullptr,
                gathered_matrix_counts.data(),
                gathered_matrix_displacements.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed_seconds * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / std::max(elapsed_seconds, 1e-12) / 1e9;
        std::printf("Computation time: %ld ms\n", duration_ms);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (print_results_requested) {
            print_results(matrix, "CholeskyL");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(matrix, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            success_flag = valid ? 1 : 0;
        }
    }

    MPI_Bcast(&success_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&shared_memory_comm);
    MPI_Finalize();
    return success_flag == 0 ? 1 : 0;
}
