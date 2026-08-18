#include <algorithm>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

constexpr int BLOCK_SIZE = 128;

[[noreturn]] void abort_all(MPI_Comm communicator, const char* operation,
                            const char* detail, int error_code = 1) {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(communicator, error_code);
    std::abort();
}

void check_cuda(cudaError_t status, const char* expression, MPI_Comm communicator) {
    if (status != cudaSuccess) {
        abort_all(communicator, expression, cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* expression, MPI_Comm communicator) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abort_all(communicator, expression, "cuBLAS returned an error");
    }
}

void check_cusolver(cusolverStatus_t status, const char* expression,
                    MPI_Comm communicator) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        abort_all(communicator, expression, "cuSOLVER returned an error");
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call, MPI_COMM_WORLD)
#define CUBLAS_CHECK(call) check_cublas((call), #call, MPI_COMM_WORLD)
#define CUSOLVER_CHECK(call) check_cusolver((call), #call, MPI_COMM_WORLD)

__global__ void zero_upper_kernel(double* matrix, std::size_t columns,
                                  std::size_t rows, std::size_t first_row) {
    const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                              + threadIdx.x;
    const std::size_t element_count = rows * columns;
    if (element >= element_count) {
        return;
    }

    const std::size_t row = element / columns;
    const std::size_t column = element - row * columns;
    if (first_row + row < column) {
        matrix[element] = 0.0;
    }
}

long long as_openmp_index(std::size_t value, MPI_Comm communicator) {
    if (value > static_cast<std::size_t>(LLONG_MAX)) {
        abort_all(communicator, "matrix size", "too large for OpenMP loop indexing");
    }
    return static_cast<long long>(value);
}

int as_mpi_count(std::size_t value, MPI_Comm communicator) {
    if (value > static_cast<std::size_t>(INT_MAX)) {
        abort_all(communicator, "MPI message size", "matrix is too large for MPI counts");
    }
    return static_cast<int>(value);
}

std::size_t checked_square(std::size_t value, MPI_Comm communicator) {
    if (value != 0 && value > std::numeric_limits<std::size_t>::max() / value) {
        abort_all(communicator, "matrix allocation", "matrix size overflows size_t");
    }
    return value * value;
}

std::size_t parse_matrix_size(const char* text, MPI_Comm communicator) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '\0' || text[0] == '-' || end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        abort_all(communicator, "command line", "-n requires a non-negative integer");
    }
    return static_cast<std::size_t>(parsed);
}

void generate_positive_definite_matrix(std::vector<double>& matrix, std::size_t n,
                                       MPI_Comm communicator) {
    // Keep the original deterministic rand_r stream, while parallelizing the
    // expensive matrix product.  Every reduction has the same k order as the
    // reference implementation.
    const std::size_t elements = checked_square(n, communicator);
    std::vector<double> random_matrix(elements);
    unsigned int seed = 42;
    for (std::size_t i = 0; i < elements; ++i) {
        random_matrix[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    const long long n_omp = as_openmp_index(n, communicator);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_omp; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += random_matrix[static_cast<std::size_t>(i) * n + k]
                     * random_matrix[j * n + k];
            }
            matrix[static_cast<std::size_t>(i) * n + j] = sum;
        }
        matrix[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)] +=
            static_cast<double>(n);
    }
}

bool validate_cholesky(const std::vector<double>& factor,
                       const std::vector<double>& original, std::size_t n,
                       MPI_Comm communicator) {
    const std::size_t elements = checked_square(n, communicator);
    std::vector<double> reconstructed(elements);
    const long long n_omp = as_openmp_index(n, communicator);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_omp; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += factor[static_cast<std::size_t>(i) * n + k]
                     * factor[j * n + k];
            }
            reconstructed[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

    double max_error = 0.0;
    double relative_error = 0.0;
    const long long elements_omp = as_openmp_index(elements, communicator);
#pragma omp parallel for reduction(max:max_error,relative_error) schedule(static)
    for (long long index = 0; index < elements_omp; ++index) {
        const std::size_t i = static_cast<std::size_t>(index);
        const double error = std::fabs(reconstructed[i] - original[i]);
        max_error = std::max(max_error, error);
        relative_error = std::max(relative_error,
                                   error / (std::fabs(original[i]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int owner_of_range(std::size_t first_row, std::size_t row_count,
                   const std::vector<std::size_t>& row_starts,
                   const std::vector<std::size_t>& row_counts) {
    for (std::size_t rank = 0; rank < row_starts.size(); ++rank) {
        if (first_row >= row_starts[rank] &&
            first_row + row_count <= row_starts[rank] + row_counts[rank]) {
            return static_cast<int>(rank);
        }
    }
    return -1;
}

} // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    std::size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;
    bool show_help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = parse_matrix_size(argv[++i], MPI_COMM_WORLD);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (show_help) {
        if (rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    const std::size_t elements = checked_square(n, MPI_COMM_WORLD);
    if (elements > static_cast<std::size_t>(INT_MAX)) {
        abort_all(MPI_COMM_WORLD, "MPI message size",
                  "this implementation requires each MPI row message to fit in int");
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_communicator);
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abort_all(MPI_COMM_WORLD, "CUDA initialization", "no CUDA accelerator is visible");
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device));

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA device %d (%s)\n",
                    world_size, omp_get_max_threads(), device, device_properties.name);
    }

    std::vector<std::size_t> row_counts(static_cast<std::size_t>(world_size));
    std::vector<std::size_t> row_starts(static_cast<std::size_t>(world_size));
    const std::size_t base_rows = n / static_cast<std::size_t>(world_size);
    const std::size_t extra_rows = n % static_cast<std::size_t>(world_size);
    std::size_t next_row = 0;
    for (int r = 0; r < world_size; ++r) {
        row_starts[static_cast<std::size_t>(r)] = next_row;
        row_counts[static_cast<std::size_t>(r)] =
            base_rows + (static_cast<std::size_t>(r) < extra_rows ? 1 : 0);
        next_row += row_counts[static_cast<std::size_t>(r)];
    }

    std::vector<int> scatter_counts(static_cast<std::size_t>(world_size));
    std::vector<int> scatter_displacements(static_cast<std::size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        scatter_counts[static_cast<std::size_t>(r)] =
            as_mpi_count(row_counts[static_cast<std::size_t>(r)] * n, MPI_COMM_WORLD);
        scatter_displacements[static_cast<std::size_t>(r)] =
            as_mpi_count(row_starts[static_cast<std::size_t>(r)] * n, MPI_COMM_WORLD);
    }

    std::vector<double> matrix;
    std::vector<double> original;
    if (rank == 0) {
        matrix.resize(elements);
        std::printf("Generating positive definite matrix...\n");
        generate_positive_definite_matrix(matrix, n, MPI_COMM_WORLD);
        if (validate) {
            original = matrix;
        }
    }

    const std::size_t local_rows = row_counts[static_cast<std::size_t>(rank)];
    const std::size_t local_elements = local_rows * n;
    std::vector<double> local_matrix(local_elements);
    MPI_Scatterv(rank == 0 ? matrix.data() : nullptr, scatter_counts.data(),
                 scatter_displacements.data(), MPI_DOUBLE,
                 local_matrix.data(), scatter_counts[static_cast<std::size_t>(rank)],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* device_matrix = nullptr;
    CUDA_CHECK(cudaMalloc(&device_matrix,
                          std::max<std::size_t>(1, local_elements * sizeof(double))));
    if (local_elements != 0) {
        CUDA_CHECK(cudaMemcpy(device_matrix, local_matrix.data(),
                              local_elements * sizeof(double), cudaMemcpyHostToDevice));
    }

    const int maximum_block = std::max(1, static_cast<int>(std::min<std::size_t>(BLOCK_SIZE, n)));
    const std::size_t maximum_block_elements =
        static_cast<std::size_t>(std::max(1, maximum_block)) *
        static_cast<std::size_t>(std::max(1, maximum_block));
    double* device_diagonal = nullptr;
    double* device_panel = nullptr;
    double* device_workspace = nullptr;
    int* device_info = nullptr;
    CUDA_CHECK(cudaMalloc(&device_diagonal,
                          maximum_block_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_panel,
                          std::max<std::size_t>(1, n * static_cast<std::size_t>(maximum_block))
                              * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_workspace, maximum_block_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_info, sizeof(int)));

    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    int workspace_elements = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, std::max(1, maximum_block),
        device_diagonal, std::max(1, maximum_block), &workspace_elements));
    if (workspace_elements <= 0) {
        abort_all(MPI_COMM_WORLD, "cuSOLVER workspace query", "invalid workspace size");
    }
    CUDA_CHECK(cudaFree(device_workspace));
    CUDA_CHECK(cudaMalloc(&device_workspace,
                          static_cast<std::size_t>(workspace_elements) * sizeof(double)));

    std::vector<double> panel_send(local_rows * static_cast<std::size_t>(maximum_block));
    std::vector<double> panel_all(n * static_cast<std::size_t>(maximum_block));
    std::vector<double> diagonal_send(maximum_block_elements);
    std::vector<double> diagonal_host(maximum_block_elements);

    std::vector<int> panel_counts(static_cast<std::size_t>(world_size));
    std::vector<int> panel_displacements(static_cast<std::size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        panel_counts[static_cast<std::size_t>(r)] = as_mpi_count(
            row_counts[static_cast<std::size_t>(r)] * static_cast<std::size_t>(maximum_block),
            MPI_COMM_WORLD);
        panel_displacements[static_cast<std::size_t>(r)] = as_mpi_count(
            row_starts[static_cast<std::size_t>(r)] * static_cast<std::size_t>(maximum_block),
            MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    for (std::size_t k = 0; k < n; k += static_cast<std::size_t>(maximum_block)) {
        const int block = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(maximum_block), n - k));
        const std::size_t block_elements = static_cast<std::size_t>(block) * block;
        const int diagonal_owner = owner_of_range(k, static_cast<std::size_t>(block),
                                                  row_starts, row_counts);

        if (diagonal_owner >= 0) {
            if (rank == diagonal_owner) {
                const std::size_t local_offset = k - row_starts[static_cast<std::size_t>(rank)];
                CUDA_CHECK(cudaMemcpy2D(
                    diagonal_host.data(), static_cast<std::size_t>(block) * sizeof(double),
                    device_matrix + local_offset * n + k, n * sizeof(double),
                    static_cast<std::size_t>(block) * sizeof(double), block,
                    cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(device_diagonal, diagonal_host.data(),
                                      block_elements * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CUSOLVER_CHECK(cusolverDnDpotrf(
                    solver, CUBLAS_FILL_MODE_UPPER, block, device_diagonal, block,
                    device_workspace, workspace_elements, device_info));
                int info = 0;
                CUDA_CHECK(cudaMemcpy(&info, device_info, sizeof(info),
                                      cudaMemcpyDeviceToHost));
                if (info != 0) {
                    std::fprintf(stderr, "Rank %d: matrix is not positive definite at block %zu\n",
                                 rank, k);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                CUDA_CHECK(cudaMemcpy(diagonal_host.data(), device_diagonal,
                                      block_elements * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(diagonal_host.data(), as_mpi_count(block_elements, MPI_COMM_WORLD),
                      MPI_DOUBLE, diagonal_owner, MPI_COMM_WORLD);
        } else {
            // A non-divisible row distribution can split a diagonal block.  Gather
            // its row slices, factor it on every GPU, and keep the same path valid
            // for any matrix size and MPI rank count.
            std::vector<int> diagonal_counts(static_cast<std::size_t>(world_size));
            std::vector<int> diagonal_displacements(static_cast<std::size_t>(world_size));
            const std::size_t local_start = row_starts[static_cast<std::size_t>(rank)];
            const std::size_t local_end = local_start + local_rows;
            const std::size_t intersection_start = std::max(k, local_start);
            const std::size_t intersection_end = std::min(
                k + static_cast<std::size_t>(block), local_end);
            const std::size_t local_diagonal_rows =
                intersection_end > intersection_start ? intersection_end - intersection_start : 0;

            for (int r = 0; r < world_size; ++r) {
                const std::size_t r_start = row_starts[static_cast<std::size_t>(r)];
                const std::size_t r_end = r_start + row_counts[static_cast<std::size_t>(r)];
                const std::size_t first = std::max(k, r_start);
                const std::size_t last = std::min(k + static_cast<std::size_t>(block), r_end);
                const std::size_t rows = last > first ? last - first : 0;
                diagonal_counts[static_cast<std::size_t>(r)] =
                    as_mpi_count(rows * static_cast<std::size_t>(block), MPI_COMM_WORLD);
                diagonal_displacements[static_cast<std::size_t>(r)] =
                    as_mpi_count((first - k) * static_cast<std::size_t>(block), MPI_COMM_WORLD);
            }

            if (local_diagonal_rows != 0) {
                const std::size_t local_offset = intersection_start - local_start;
                CUDA_CHECK(cudaMemcpy2D(
                    diagonal_send.data(), static_cast<std::size_t>(block) * sizeof(double),
                    device_matrix + local_offset * n + k, n * sizeof(double),
                    static_cast<std::size_t>(block) * sizeof(double), local_diagonal_rows,
                    cudaMemcpyDeviceToHost));
            }
            MPI_Allgatherv(diagonal_send.data(),
                           diagonal_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE,
                           diagonal_host.data(), diagonal_counts.data(),
                           diagonal_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            CUDA_CHECK(cudaMemcpy(device_diagonal, diagonal_host.data(),
                                  block_elements * sizeof(double), cudaMemcpyHostToDevice));
            CUSOLVER_CHECK(cusolverDnDpotrf(
                solver, CUBLAS_FILL_MODE_UPPER, block, device_diagonal, block,
                device_workspace, workspace_elements, device_info));
            int info = 0;
            CUDA_CHECK(cudaMemcpy(&info, device_info, sizeof(info), cudaMemcpyDeviceToHost));
            int global_info = 0;
            MPI_Allreduce(&info, &global_info, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (global_info != 0) {
                if (rank == 0) {
                    std::fprintf(stderr, "Matrix is not positive definite at block %zu\n", k);
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            CUDA_CHECK(cudaMemcpy(diagonal_host.data(), device_diagonal,
                                  block_elements * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // Put the newly factored diagonal rows into each rank's row partition.
        const std::size_t local_start = row_starts[static_cast<std::size_t>(rank)];
        const std::size_t local_end = local_start + local_rows;
        const std::size_t diagonal_start = std::max(k, local_start);
        const std::size_t diagonal_end = std::min(k + static_cast<std::size_t>(block), local_end);
        if (diagonal_end > diagonal_start) {
            const std::size_t rows = diagonal_end - diagonal_start;
            const std::size_t source_row = diagonal_start - k;
            const std::size_t destination_row = diagonal_start - local_start;
            CUDA_CHECK(cudaMemcpy2D(
                device_matrix + destination_row * n + k, n * sizeof(double),
                device_diagonal + source_row * static_cast<std::size_t>(block),
                static_cast<std::size_t>(block) * sizeof(double),
                static_cast<std::size_t>(block) * sizeof(double), rows,
                cudaMemcpyDeviceToDevice));
        }

        // Solve L(i,k:k+b) L(k:k+b,k:k+b)^T = A(i,k:k+b) for this
        // rank's rows.  The row-major panel is viewed as a column-major
        // transposed matrix by cuBLAS.
        const std::size_t solve_start = std::max(k + static_cast<std::size_t>(block), local_start);
        if (solve_start < local_end) {
            const int solve_rows = static_cast<int>(local_end - solve_start);
            const std::size_t local_offset = solve_start - local_start;
            const double one = 1.0;
            CUBLAS_CHECK(cublasDtrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, block, solve_rows, &one, device_diagonal, block,
                device_matrix + local_offset * n + k, static_cast<int>(n)));
        }

        // Exchange the current panel so every GPU can perform its local
        // trailing update without another matrix replication.
        if (local_rows != 0) {
            std::fill(panel_send.begin(), panel_send.end(), 0.0);
            CUDA_CHECK(cudaMemcpy2D(
                panel_send.data(), static_cast<std::size_t>(maximum_block) * sizeof(double),
                device_matrix + k, n * sizeof(double),
                static_cast<std::size_t>(block) * sizeof(double), local_rows,
                cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(panel_send.data(), panel_counts[static_cast<std::size_t>(rank)],
                       MPI_DOUBLE, panel_all.data(), panel_counts.data(),
                       panel_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(device_panel, panel_all.data(),
                              n * static_cast<std::size_t>(maximum_block) * sizeof(double),
                              cudaMemcpyHostToDevice));

        const std::size_t update_start = std::max(k + static_cast<std::size_t>(block), local_start);
        if (update_start < local_end && k + static_cast<std::size_t>(block) < n) {
            const int update_rows = static_cast<int>(local_end - update_start);
            const int update_columns = static_cast<int>(n - k - block);
            const std::size_t local_offset = update_start - local_start;
            const std::size_t panel_row = k + static_cast<std::size_t>(block);
            const double alpha = -1.0;
            const double beta = 1.0;
            // C(row-major) -= X(row-major) Y(row-major)^T, expressed as
            // C^T -= Y^T X^T for cuBLAS's column-major interface.
            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_T, CUBLAS_OP_N, update_columns, update_rows, block,
                &alpha, device_panel + panel_row * static_cast<std::size_t>(maximum_block),
                maximum_block,
                device_matrix + local_offset * n + k, static_cast<int>(n),
                &beta, device_matrix + local_offset * n + k + block, static_cast<int>(n)));
        }
    }

    if (local_elements != 0) {
        const std::size_t total_elements = local_elements;
        const unsigned int grid = static_cast<unsigned int>((total_elements + 255) / 256);
        zero_upper_kernel<<<grid, 256>>>(device_matrix, n, local_rows,
                                          row_starts[static_cast<std::size_t>(rank)]);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_end_time = MPI_Wtime();

    if (local_elements != 0) {
        CUDA_CHECK(cudaMemcpy(local_matrix.data(), device_matrix,
                              local_elements * sizeof(double), cudaMemcpyDeviceToHost));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_seconds = local_end_time - start_time;
    double elapsed_seconds = 0.0;
    MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> gather_counts = scatter_counts;
    std::vector<int> gather_displacements = scatter_displacements;
    MPI_Gatherv(local_matrix.data(), scatter_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE,
                rank == 0 ? matrix.data() : nullptr, gather_counts.data(),
                gather_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed_seconds * 1000.0);
        std::printf("Computation time: %ld ms\n", duration_ms);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = elapsed_seconds > 0.0
                            ? operations / elapsed_seconds / 1.0e9
                            : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (print_results_requested) {
            print_results(matrix, "CholeskyL");
        }
    }

    int exit_code = 0;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        if (validate_cholesky(matrix, original, n, MPI_COMM_WORLD)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exit_code = 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUBLAS_CHECK(cublasDestroy(blas));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaFree(device_info));
    CUDA_CHECK(cudaFree(device_workspace));
    CUDA_CHECK(cudaFree(device_panel));
    CUDA_CHECK(cudaFree(device_diagonal));
    CUDA_CHECK(cudaFree(device_matrix));
    MPI_Comm_free(&local_communicator);
    MPI_Finalize();
    return exit_code;
}
