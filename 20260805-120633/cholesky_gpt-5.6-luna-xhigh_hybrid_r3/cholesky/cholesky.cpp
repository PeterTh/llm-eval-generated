#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// The block size is large enough to make the CUDA trailing update compute
// bound for the benchmark sizes, while keeping the MPI panel messages small.
constexpr int kBlockSize = 128;
constexpr int kTileSize = 16;

[[noreturn]] void abortWithMessage(MPI_Comm communicator, const char* message) {
    std::fprintf(stderr, "%s\n", message);
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortWithCudaError(MPI_Comm communicator, const char* operation,
                                     cudaError_t error) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

void checkCuda(MPI_Comm communicator, const char* operation, cudaError_t error) {
    if (error != cudaSuccess) {
        abortWithCudaError(communicator, operation, error);
    }
}

void checkCudaKernel(MPI_Comm communicator, const char* operation) {
    checkCuda(communicator, operation, cudaGetLastError());
    checkCuda(communicator, operation, cudaDeviceSynchronize());
}

struct Distribution {
    int n;
    int world_size;
    int block_count;
    std::vector<std::vector<int>> rank_rows;
    std::vector<int> block_owner;
    std::vector<int> block_size;
    std::vector<int> block_local_offset;

    Distribution(int matrix_size, int process_count)
        : n(matrix_size),
          world_size(process_count),
          block_count((matrix_size + kBlockSize - 1) / kBlockSize),
          rank_rows(static_cast<std::size_t>(process_count)),
          block_owner(static_cast<std::size_t>(block_count)),
          block_size(static_cast<std::size_t>(block_count)),
          block_local_offset(static_cast<std::size_t>(block_count)) {
        for (int block = 0; block < block_count; ++block) {
            const int owner = block % world_size;
            const int first_row = block * kBlockSize;
            const int rows = std::min(kBlockSize, n - first_row);
            block_owner[static_cast<std::size_t>(block)] = owner;
            block_size[static_cast<std::size_t>(block)] = rows;
            block_local_offset[static_cast<std::size_t>(block)] =
                static_cast<int>(rank_rows[static_cast<std::size_t>(owner)].size());
            for (int row = first_row; row < first_row + rows; ++row) {
                rank_rows[static_cast<std::size_t>(owner)].push_back(row);
            }
        }
    }
};

__global__ void factorDiagonalBlock(double* matrix, std::size_t leading_dimension,
                                    int block_size, int* factorization_error) {
    const int thread = static_cast<int>(threadIdx.x);

    // One CUDA block owns the diagonal panel. The factorization is inherently
    // sequential in its columns, but all rows below the current diagonal are
    // independent and are evaluated concurrently.
    for (int column = 0; column < block_size; ++column) {
        if (thread == column) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double value = matrix[static_cast<std::size_t>(thread) * leading_dimension + k];
                sum += value * value;
            }
            const std::size_t diagonal =
                static_cast<std::size_t>(thread) * leading_dimension + column;
            const double value = matrix[diagonal] - sum;
            if (value <= 0.0) {
                atomicExch(factorization_error, column + 1);
                matrix[diagonal] = 1.0;
            } else {
                matrix[diagonal] = sqrt(value);
            }
        }
        __syncthreads();

        if (thread > column && thread < block_size) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += matrix[static_cast<std::size_t>(thread) * leading_dimension + k] *
                       matrix[static_cast<std::size_t>(column) * leading_dimension + k];
            }
            matrix[static_cast<std::size_t>(thread) * leading_dimension + column] =
                (matrix[static_cast<std::size_t>(thread) * leading_dimension + column] - sum) /
                matrix[static_cast<std::size_t>(column) * leading_dimension + column];
        }
        __syncthreads();
    }
}

__global__ void solvePanelRows(double* matrix, std::size_t leading_dimension,
                               const int* global_rows, int local_rows, int n, int panel_start,
                               int panel_size, const double* diagonal_block) {
    const int local_row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (local_row >= local_rows) {
        return;
    }

    const int global_row = global_rows[local_row];
    if (global_row < panel_start + panel_size) {
        return;
    }

    double* row = matrix + static_cast<std::size_t>(local_row) * leading_dimension;
    for (int column = 0; column < panel_size; ++column) {
        double sum = row[panel_start + column];
        for (int k = 0; k < column; ++k) {
            sum -= row[panel_start + k] * diagonal_block[column * panel_size + k];
        }
        row[panel_start + column] = sum / diagonal_block[column * panel_size + column];
    }
    (void)n;
}

__global__ void packPanelRows(const double* matrix, std::size_t leading_dimension,
                              int first_local_row, int panel_rows, int panel_start,
                              int panel_size, double* packed_panel) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int elements = panel_rows * panel_size;
    if (index >= elements) {
        return;
    }

    const int row = index / panel_size;
    const int column = index % panel_size;
    packed_panel[index] = matrix[static_cast<std::size_t>(first_local_row + row) *
                                     leading_dimension + panel_start + column];
}

__global__ void trailingUpdate(double* matrix, std::size_t leading_dimension,
                               const int* global_rows, int local_rows, int n, int panel_start,
                               int panel_size, int panel_end, const double* panel,
                               int panel_stride) {
    __shared__ double left_tile[kTileSize][kTileSize];
    __shared__ double right_tile[kTileSize][kTileSize];

    const int local_row = static_cast<int>(blockIdx.y * kTileSize + threadIdx.y);
    const int column = static_cast<int>(blockIdx.x * kTileSize + threadIdx.x);
    const bool valid_row = local_row < local_rows && global_rows[local_row] > panel_end;
    const int global_row = valid_row ? global_rows[local_row] : -1;

    double sum = 0.0;
    for (int tile = 0; tile < panel_size; tile += kTileSize) {
        const int left_column = tile + static_cast<int>(threadIdx.x);
        if (valid_row && left_column < panel_size) {
            left_tile[threadIdx.y][threadIdx.x] =
                matrix[static_cast<std::size_t>(local_row) * leading_dimension + panel_start +
                       left_column];
        } else {
            left_tile[threadIdx.y][threadIdx.x] = 0.0;
        }

        const int right_panel_column = tile + static_cast<int>(threadIdx.y);
        if (column < n && right_panel_column < panel_size) {
            right_tile[threadIdx.y][threadIdx.x] =
                panel[static_cast<std::size_t>(column) * panel_stride + right_panel_column];
        } else {
            right_tile[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int tile_width = min(kTileSize, panel_size - tile);
        for (int k = 0; k < tile_width; ++k) {
            sum += left_tile[threadIdx.y][k] * right_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (valid_row && column > panel_end && column <= global_row && column < n) {
        matrix[static_cast<std::size_t>(local_row) * leading_dimension + column] -= sum;
    }
}

__global__ void zeroUpperTriangle(double* matrix, std::size_t leading_dimension,
                                   const int* global_rows, int local_rows, int n) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int elements = local_rows * n;
    if (index >= elements) {
        return;
    }

    const int local_row = index / n;
    const int column = index % n;
    if (column > global_rows[local_row]) {
        matrix[static_cast<std::size_t>(local_row) * leading_dimension + column] = 0.0;
    }
}

void generatePositiveDefiniteMatrix(std::vector<double>& matrix, int n) {
    std::vector<double> random_matrix(static_cast<std::size_t>(n) * n);
    unsigned int seed = 42;

    // Keep the original deterministic generator and its exact row-wise dot
    // product order, while parallelizing independent output rows.
    for (double& value : random_matrix) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += random_matrix[static_cast<std::size_t>(row) * n + k] *
                       random_matrix[static_cast<std::size_t>(column) * n + k];
            }
            matrix[static_cast<std::size_t>(row) * n + column] = sum;
        }
        matrix[static_cast<std::size_t>(row) * n + row] += n;
    }
}

void distributeMatrix(const std::vector<double>& root_matrix, std::vector<double>& local_matrix,
                      const Distribution& distribution, int rank, MPI_Comm communicator) {
    const std::size_t local_rows =
        distribution.rank_rows[static_cast<std::size_t>(rank)].size();
    local_matrix.resize(local_rows * static_cast<std::size_t>(distribution.n));

    if (rank == 0) {
        const auto& own_rows = distribution.rank_rows[0];
        #pragma omp parallel for schedule(static)
        for (std::size_t row = 0; row < own_rows.size(); ++row) {
            std::memcpy(local_matrix.data() + row * distribution.n,
                        root_matrix.data() + static_cast<std::size_t>(own_rows[row]) * distribution.n,
                        static_cast<std::size_t>(distribution.n) * sizeof(double));
        }

        for (int destination = 1; destination < distribution.world_size; ++destination) {
            const auto& rows = distribution.rank_rows[static_cast<std::size_t>(destination)];
            std::vector<int> block_lengths(rows.size(), distribution.n);
            std::vector<MPI_Aint> displacements(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                displacements[i] = static_cast<MPI_Aint>(
                    static_cast<std::size_t>(rows[i]) * distribution.n * sizeof(double));
            }

            MPI_Datatype row_type = MPI_DATATYPE_NULL;
            if (!rows.empty()) {
                if (MPI_Type_create_hindexed(static_cast<int>(rows.size()), block_lengths.data(),
                                             displacements.data(), MPI_DOUBLE, &row_type) != MPI_SUCCESS) {
                    abortWithMessage(communicator, "MPI failed to create the row distribution type");
                }
                MPI_Type_commit(&row_type);
                if (MPI_Send(root_matrix.data(), 1, row_type, destination, 701, communicator) !=
                    MPI_SUCCESS) {
                    abortWithMessage(communicator, "MPI failed while distributing matrix rows");
                }
                MPI_Type_free(&row_type);
            } else if (MPI_Send(nullptr, 0, MPI_DOUBLE, destination, 701, communicator) !=
                       MPI_SUCCESS) {
                abortWithMessage(communicator, "MPI failed while distributing empty matrix rows");
            }
        }
    } else {
        if (MPI_Recv(local_matrix.data(), static_cast<int>(local_matrix.size()), MPI_DOUBLE, 0, 701,
                     communicator, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            abortWithMessage(communicator, "MPI failed while receiving matrix rows");
        }
    }
}

void gatherMatrix(const std::vector<double>& local_matrix, std::vector<double>& root_matrix,
                  const Distribution& distribution, int rank, MPI_Comm communicator) {
    const int n = distribution.n;
    const auto& own_rows = distribution.rank_rows[static_cast<std::size_t>(rank)];

    if (rank == 0) {
        root_matrix.assign(static_cast<std::size_t>(n) * n, 0.0);
        #pragma omp parallel for schedule(static)
        for (std::size_t row = 0; row < own_rows.size(); ++row) {
            std::memcpy(root_matrix.data() + static_cast<std::size_t>(own_rows[row]) * n,
                        local_matrix.data() + row * n, static_cast<std::size_t>(n) * sizeof(double));
        }

        std::vector<double> received;
        for (int source = 1; source < distribution.world_size; ++source) {
            const auto& rows = distribution.rank_rows[static_cast<std::size_t>(source)];
            received.resize(rows.size() * static_cast<std::size_t>(n));
            if (MPI_Recv(received.data(), static_cast<int>(received.size()), MPI_DOUBLE, source, 702,
                         communicator, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                abortWithMessage(communicator, "MPI failed while gathering matrix rows");
            }
            #pragma omp parallel for schedule(static)
            for (std::size_t row = 0; row < rows.size(); ++row) {
                std::memcpy(root_matrix.data() + static_cast<std::size_t>(rows[row]) * n,
                            received.data() + row * n, static_cast<std::size_t>(n) * sizeof(double));
            }
        }
    } else if (MPI_Send(local_matrix.data(), static_cast<int>(local_matrix.size()), MPI_DOUBLE, 0, 702,
                       communicator) != MPI_SUCCESS) {
        abortWithMessage(communicator, "MPI failed while sending final matrix rows");
    }
}

bool validateCholesky(const std::vector<double>& factor, const std::vector<double>& original, int n) {
    std::vector<double> reconstructed(static_cast<std::size_t>(n) * n, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += factor[static_cast<std::size_t>(row) * n + k] *
                       factor[static_cast<std::size_t>(column) * n + k];
            }
            reconstructed[static_cast<std::size_t>(row) * n + column] = sum;
        }
    }

    double max_error = 0.0;
    double relative_error = 0.0;
    #pragma omp parallel for reduction(max:max_error, relative_error) schedule(static)
    for (std::size_t index = 0; index < original.size(); ++index) {
        const double error = std::fabs(reconstructed[index] - original[index]);
        max_error = std::max(max_error, error);
        relative_error = std::max(relative_error, error / (std::fabs(original[index]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

bool distributedCholesky(std::vector<double>& local_matrix, const Distribution& distribution,
                         int rank, MPI_Comm communicator, double& elapsed_seconds) {
    const int n = distribution.n;
    const int local_rows = static_cast<int>(distribution.rank_rows[static_cast<std::size_t>(rank)].size());
    const std::size_t local_elements = static_cast<std::size_t>(local_rows) * n;

    int device_count = 0;
    checkCuda(communicator, "cudaGetDeviceCount", cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        abortWithMessage(communicator, "The hybrid benchmark requires at least one CUDA device per node");
    }

    MPI_Comm local_communicator = MPI_COMM_NULL;
    if (MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &local_communicator) != MPI_SUCCESS) {
        abortWithMessage(communicator, "MPI_Comm_split_type failed");
    }
    int local_rank = 0;
    MPI_Comm_rank(local_communicator, &local_rank);
    checkCuda(communicator, "cudaSetDevice", cudaSetDevice(local_rank % device_count));
    checkCuda(communicator, "cudaFree(0)", cudaFree(nullptr));

    double* device_matrix = nullptr;
    int* device_global_rows = nullptr;
    double* device_diagonal = nullptr;
    double* device_panel_send = nullptr;
    double* device_panel_global = nullptr;
    int* device_error = nullptr;
    const std::size_t matrix_bytes = std::max<std::size_t>(1, local_elements) * sizeof(double);
    const std::size_t row_bytes = std::max<std::size_t>(1, static_cast<std::size_t>(local_rows)) *
                                  sizeof(int);
    const std::size_t panel_send_bytes =
        std::max<std::size_t>(1, static_cast<std::size_t>(local_rows) * kBlockSize) * sizeof(double);
    const std::size_t panel_global_bytes =
        std::max<std::size_t>(1, static_cast<std::size_t>(n) * kBlockSize) * sizeof(double);

    checkCuda(communicator, "cudaMalloc(matrix)", cudaMalloc(reinterpret_cast<void**>(&device_matrix), matrix_bytes));
    checkCuda(communicator, "cudaMalloc(global rows)",
              cudaMalloc(reinterpret_cast<void**>(&device_global_rows), row_bytes));
    checkCuda(communicator, "cudaMalloc(diagonal block)",
              cudaMalloc(reinterpret_cast<void**>(&device_diagonal),
                         static_cast<std::size_t>(kBlockSize) * kBlockSize * sizeof(double)));
    checkCuda(communicator, "cudaMalloc(panel send)",
              cudaMalloc(reinterpret_cast<void**>(&device_panel_send), panel_send_bytes));
    checkCuda(communicator, "cudaMalloc(global panel)",
              cudaMalloc(reinterpret_cast<void**>(&device_panel_global), panel_global_bytes));
    checkCuda(communicator, "cudaMalloc(error flag)",
              cudaMalloc(reinterpret_cast<void**>(&device_error), sizeof(int)));

    if (local_elements != 0) {
        checkCuda(communicator, "cudaMemcpy(matrix)",
                  cudaMemcpy(device_matrix, local_matrix.data(), local_elements * sizeof(double),
                             cudaMemcpyHostToDevice));
    }
    if (local_rows != 0) {
        checkCuda(communicator, "cudaMemcpy(global rows)",
                  cudaMemcpy(device_global_rows,
                             distribution.rank_rows[static_cast<std::size_t>(rank)].data(), row_bytes,
                             cudaMemcpyHostToDevice));
    }

    std::vector<double> diagonal(static_cast<std::size_t>(kBlockSize) * kBlockSize);
    std::vector<double> packed_panel(
        std::max<std::size_t>(1, static_cast<std::size_t>(local_rows) * kBlockSize));
    std::vector<double> received_panel(
        std::max<std::size_t>(1, static_cast<std::size_t>(n) * kBlockSize));
    std::vector<double> global_panel(
        std::max<std::size_t>(1, static_cast<std::size_t>(n) * kBlockSize), 0.0);
    std::vector<int> panel_counts(static_cast<std::size_t>(distribution.world_size));
    std::vector<int> panel_displacements(static_cast<std::size_t>(distribution.world_size));
    std::vector<int> first_local_row(static_cast<std::size_t>(distribution.world_size));

    MPI_Barrier(communicator);
    const double start = MPI_Wtime();
    bool success = true;

    for (int block = 0; block < distribution.block_count; ++block) {
        const int panel_start = block * kBlockSize;
        const int panel_size = distribution.block_size[static_cast<std::size_t>(block)];
        const int panel_end = panel_start + panel_size - 1;
        const int owner = distribution.block_owner[static_cast<std::size_t>(block)];
        const int owner_offset = distribution.block_local_offset[static_cast<std::size_t>(block)];

        int factorization_error = 0;
        if (rank == owner) {
            checkCuda(communicator, "cudaMemset(factorization error)",
                      cudaMemset(device_error, 0, sizeof(int)));
            factorDiagonalBlock<<<1, panel_size>>>(
                device_matrix + static_cast<std::size_t>(owner_offset) * n + panel_start, n,
                panel_size, device_error);
            checkCudaKernel(communicator, "factorDiagonalBlock");
            checkCuda(communicator, "cudaMemcpy(factorization error)",
                      cudaMemcpy(&factorization_error, device_error, sizeof(int), cudaMemcpyDeviceToHost));
            if (factorization_error == 0) {
                checkCuda(communicator, "cudaMemcpy(diagonal block)",
                          cudaMemcpy2D(diagonal.data(), static_cast<std::size_t>(panel_size) * sizeof(double),
                                       device_matrix + static_cast<std::size_t>(owner_offset) * n + panel_start,
                                       static_cast<std::size_t>(n) * sizeof(double),
                                       static_cast<std::size_t>(panel_size) * sizeof(double), panel_size,
                                       cudaMemcpyDeviceToHost));
            }
        }

        MPI_Bcast(&factorization_error, 1, MPI_INT, owner, communicator);
        if (factorization_error != 0) {
            if (rank == 0) {
                std::fprintf(stderr, "Non-positive diagonal at matrix index %d\n",
                             panel_start + factorization_error - 1);
            }
            success = false;
            break;
        }
        MPI_Bcast(diagonal.data(), panel_size * panel_size, MPI_DOUBLE, owner, communicator);
        checkCuda(communicator, "cudaMemcpy(diagonal block)",
                  cudaMemcpy(device_diagonal, diagonal.data(),
                             static_cast<std::size_t>(panel_size) * panel_size * sizeof(double),
                             cudaMemcpyHostToDevice));

        if (local_rows != 0) {
            const int threads = 256;
            const int blocks = (local_rows + threads - 1) / threads;
            solvePanelRows<<<blocks, threads>>>(device_matrix, n, device_global_rows, local_rows, n,
                                                panel_start, panel_size, device_diagonal);
            checkCudaKernel(communicator, "solvePanelRows");
        }

        int total_panel_elements = 0;
        for (int process = 0; process < distribution.world_size; ++process) {
            const auto& rows = distribution.rank_rows[static_cast<std::size_t>(process)];
            const auto first = std::lower_bound(rows.begin(), rows.end(), panel_start);
            first_local_row[static_cast<std::size_t>(process)] =
                static_cast<int>(first - rows.begin());
            panel_counts[static_cast<std::size_t>(process)] =
                static_cast<int>((rows.size() - static_cast<std::size_t>(first_local_row[process])) *
                                 panel_size);
            panel_displacements[static_cast<std::size_t>(process)] = total_panel_elements;
            total_panel_elements += panel_counts[static_cast<std::size_t>(process)];
        }

        const int local_panel_rows = static_cast<int>(
            distribution.rank_rows[static_cast<std::size_t>(rank)].size() -
            static_cast<std::size_t>(first_local_row[static_cast<std::size_t>(rank)]));
        if (local_panel_rows != 0) {
            const int threads = 256;
            const int elements = local_panel_rows * panel_size;
            const int blocks = (elements + threads - 1) / threads;
            packPanelRows<<<blocks, threads>>>(
                device_matrix, n, first_local_row[static_cast<std::size_t>(rank)], local_panel_rows,
                panel_start, panel_size, device_panel_send);
            checkCudaKernel(communicator, "packPanelRows");
            checkCuda(communicator, "cudaMemcpy(panel rows)",
                      cudaMemcpy(packed_panel.data(), device_panel_send,
                                 static_cast<std::size_t>(elements) * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        }

        if (MPI_Allgatherv(packed_panel.data(), panel_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE,
                           received_panel.data(), panel_counts.data(), panel_displacements.data(),
                           MPI_DOUBLE, communicator) != MPI_SUCCESS) {
            abortWithMessage(communicator, "MPI_Allgatherv failed for the Cholesky panel");
        }

        #pragma omp parallel for schedule(static)
        for (int process = 0; process < distribution.world_size; ++process) {
            for (int row = first_local_row[static_cast<std::size_t>(process)];
                 row < static_cast<int>(distribution.rank_rows[static_cast<std::size_t>(process)].size());
                 ++row) {
                const int global_row =
                    distribution.rank_rows[static_cast<std::size_t>(process)][static_cast<std::size_t>(row)];
                const int source_row = row - first_local_row[static_cast<std::size_t>(process)];
                for (int column = 0; column < panel_size; ++column) {
                    global_panel[static_cast<std::size_t>(global_row) * kBlockSize + column] =
                        received_panel[static_cast<std::size_t>(panel_displacements[process]) +
                                       static_cast<std::size_t>(source_row) * panel_size + column];
                }
            }
        }
        checkCuda(communicator, "cudaMemcpy(global panel)",
                  cudaMemcpy(device_panel_global, global_panel.data(), panel_global_bytes,
                             cudaMemcpyHostToDevice));

        if (local_rows != 0) {
            const dim3 threads(kTileSize, kTileSize);
            const dim3 blocks(static_cast<unsigned int>((n + kTileSize - 1) / kTileSize),
                              static_cast<unsigned int>((local_rows + kTileSize - 1) / kTileSize));
            trailingUpdate<<<blocks, threads>>>(device_matrix, n, device_global_rows, local_rows, n,
                                                panel_start, panel_size, panel_end,
                                                device_panel_global, kBlockSize);
            checkCudaKernel(communicator, "trailingUpdate");
        }
    }

    if (success && local_rows != 0) {
        const int threads = 256;
        const int blocks = (local_rows * n + threads - 1) / threads;
        zeroUpperTriangle<<<blocks, threads>>>(device_matrix, n, device_global_rows, local_rows, n);
        checkCudaKernel(communicator, "zeroUpperTriangle");
        checkCuda(communicator, "cudaMemcpy(final matrix)",
                  cudaMemcpy(local_matrix.data(), device_matrix, local_elements * sizeof(double),
                             cudaMemcpyDeviceToHost));
    }

    elapsed_seconds = MPI_Wtime() - start;
    checkCuda(communicator, "cudaFree(matrix)", cudaFree(device_matrix));
    checkCuda(communicator, "cudaFree(global rows)", cudaFree(device_global_rows));
    checkCuda(communicator, "cudaFree(diagonal block)", cudaFree(device_diagonal));
    checkCuda(communicator, "cudaFree(panel send)", cudaFree(device_panel_send));
    checkCuda(communicator, "cudaFree(global panel)", cudaFree(device_panel_global));
    checkCuda(communicator, "cudaFree(error flag)", cudaFree(device_error));
    MPI_Comm_free(&local_communicator);
    return success;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level) != MPI_SUCCESS ||
        provided_thread_level < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI does not provide the required FUNNELED thread level\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 512;
    bool validate = false;
    bool print_results_output = false;
    bool parse_error = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            char* end = nullptr;
            const long long parsed = std::strtoll(argv[++argument], &end, 10);
            if (end == argv[argument] || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                parse_error = true;
            } else {
                n = static_cast<int>(parsed);
            }
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            print_results_output = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            parse_error = true;
        }
    }

    if (parse_error) {
        if (rank == 0) {
            std::printf("Invalid command line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    Distribution distribution(n, world_size);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA block size %d\n",
                    world_size, omp_get_max_threads(), kBlockSize);
    }

    std::vector<double> matrix;
    std::vector<double> original;
    if (rank == 0) {
        matrix.resize(static_cast<std::size_t>(n) * n);
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(matrix, n);
        if (validate) {
            original = matrix;
        }
    }

    std::vector<double> local_matrix;
    distributeMatrix(matrix, local_matrix, distribution, rank, MPI_COMM_WORLD);
    if (rank == 0 && !validate && !print_results_output) {
        std::vector<double>().swap(matrix);
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    double local_elapsed = 0.0;
    const bool success = distributedCholesky(local_matrix, distribution, rank, MPI_COMM_WORLD,
                                             local_elapsed);
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    gatherMatrix(local_matrix, matrix, distribution, rank, MPI_COMM_WORLD);
    if (rank == 0) {
        const long computation_milliseconds =
            std::max<long>(1, static_cast<long>(elapsed * 1000.0));
        std::printf("Computation time: %ld ms\n", computation_milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / elapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (print_results_output) {
            print_results(matrix, "CholeskyL");
        }
    }

    int valid = 1;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        valid = validateCholesky(matrix, original, n) ? 1 : 0;
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
