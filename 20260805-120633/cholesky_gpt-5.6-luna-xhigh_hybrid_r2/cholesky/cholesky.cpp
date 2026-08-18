#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 256;

[[noreturn]] void mpiAbort(const char* operation, const char* detail, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                    \
        if (error__ != cudaSuccess) {                                          \
            mpiAbort(#call, cudaGetErrorString(error__), rank);                 \
        }                                                                       \
    } while (false)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        const cublasStatus_t status__ = (call);                                \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                \
            mpiAbort(#call, "cuBLAS returned an error", rank);                  \
        }                                                                       \
    } while (false)

// Factor one diagonal block in place. The panel is small enough that a single
// CUDA block can keep the factorization synchronized without host round trips.
// The column updates are distributed over the CUDA threads, while the diagonal
// dependency advances one column at a time.
__global__ void diagonalCholeskyKernel(double* matrix,
                                       const int matrixSize,
                                       const int localRowStart,
                                       const int diagonalStart,
                                       const int blockSize,
                                       int* failure) {
    const int thread = static_cast<int>(threadIdx.x);

    for (int column = 0; column < blockSize; ++column) {
        if (thread == 0) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double value = matrix[static_cast<size_t>(localRowStart + column) * matrixSize +
                                            diagonalStart + k];
                sum += value * value;
            }
            const size_t diagonal = static_cast<size_t>(localRowStart + column) * matrixSize +
                                    diagonalStart + column;
            const double value = matrix[diagonal] - sum;
            if (!(value > 0.0)) {
                *failure = column + 1;
                matrix[diagonal] = __longlong_as_double(0x7ff8000000000000LL);
            } else {
                matrix[diagonal] = sqrt(value);
            }
        }
        __syncthreads();

        if (thread > column && thread < blockSize) {
            const int row = thread;
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double left = matrix[static_cast<size_t>(localRowStart + row) * matrixSize +
                                           diagonalStart + k];
                const double right = matrix[static_cast<size_t>(localRowStart + column) * matrixSize +
                                            diagonalStart + k];
                sum += left * right;
            }
            const size_t element = static_cast<size_t>(localRowStart + row) * matrixSize +
                                   diagonalStart + column;
            matrix[element] = (matrix[element] - sum) /
                              matrix[static_cast<size_t>(localRowStart + column) * matrixSize +
                                     diagonalStart + column];
        }
        __syncthreads();
    }

    // The benchmark's result is a lower-triangular matrix. Clearing the upper
    // part here also makes every distributed block have the original layout.
    if (thread < blockSize) {
        for (int column = thread + 1; column < blockSize; ++column) {
            matrix[static_cast<size_t>(localRowStart + thread) * matrixSize +
                   diagonalStart + column] = 0.0;
        }
    }
}

struct DistributedLayout {
    int rank = 0;
    int worldSize = 1;
    int matrixSize = 0;
    std::vector<int> blockStart;
    std::vector<int> blockRows;
    std::vector<int> blockOwner;
    std::vector<int> localBlocks;
    std::vector<size_t> localOffsets;
    size_t localRows = 0;
};

DistributedLayout makeLayout(int matrixSize, int rank, int worldSize) {
    DistributedLayout layout;
    layout.rank = rank;
    layout.worldSize = worldSize;
    layout.matrixSize = matrixSize;

    const int blockCount = (matrixSize + kBlockSize - 1) / kBlockSize;
    layout.blockStart.resize(blockCount);
    layout.blockRows.resize(blockCount);
    layout.blockOwner.resize(blockCount);
    layout.localOffsets.assign(blockCount, std::numeric_limits<size_t>::max());

    for (int block = 0; block < blockCount; ++block) {
        layout.blockStart[block] = block * kBlockSize;
        layout.blockRows[block] = std::min(kBlockSize, matrixSize - layout.blockStart[block]);
        layout.blockOwner[block] = block % worldSize;
        if (layout.blockOwner[block] == rank) {
            layout.localBlocks.push_back(block);
            layout.localOffsets[block] = layout.localRows * static_cast<size_t>(matrixSize);
            layout.localRows += static_cast<size_t>(layout.blockRows[block]);
        }
    }
    return layout;
}

void ensureMpiCount(size_t count, const char* operation, int rank) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        mpiAbort(operation, "MPI count exceeds the implementation limit", rank);
    }
}

void generatePositiveDefiniteMatrix(std::vector<double>& matrix, size_t n) {
    // Keep the original deterministic rand_r stream and summation order. The
    // independent output entries are generated in parallel with OpenMP.
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += b[static_cast<size_t>(i) * n + k] * b[j * n + k];
            }
            matrix[static_cast<size_t>(i) * n + j] = sum;
        }
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        matrix[static_cast<size_t>(i) * n + static_cast<size_t>(i)] +=
            static_cast<double>(n);
    }
}

void clearUpperTriangle(std::vector<double>& matrix, size_t n) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const size_t row = static_cast<size_t>(i) * n;
        std::fill(matrix.begin() + row + static_cast<size_t>(i) + 1,
                  matrix.begin() + row + n, 0.0);
    }
}

bool distributedCholesky(std::vector<double>& localMatrix,
                         const DistributedLayout& layout,
                         int cudaDevice,
                         MPI_Comm communicator) {
    const int n = layout.matrixSize;
    const int rank = layout.rank;
    const size_t localElements = localMatrix.size();

    double* deviceMatrix = nullptr;
    double* devicePanel = nullptr;
    double* deviceDiagonal = nullptr;
    int* deviceFailure = nullptr;
    cublasHandle_t blas = nullptr;

    CUDA_CHECK(cudaSetDevice(cudaDevice));
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMatrix),
                          std::max<size_t>(1, localElements) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePanel),
                          std::max<size_t>(1, static_cast<size_t>(n) * kBlockSize) *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDiagonal),
                          static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceFailure), sizeof(int)));
    CUDA_CHECK(cudaMemcpy(deviceMatrix, localMatrix.data(),
                          localElements * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> diagonal(kBlockSize * kBlockSize);
    std::vector<double> panelHost(static_cast<size_t>(n) * kBlockSize);

    const int blockCount = static_cast<int>(layout.blockStart.size());
    for (int diagonalBlock = 0; diagonalBlock < blockCount; ++diagonalBlock) {
        const int diagonalStart = layout.blockStart[diagonalBlock];
        const int blockRows = layout.blockRows[diagonalBlock];

        if (layout.blockOwner[diagonalBlock] == rank) {
            CUDA_CHECK(cudaMemset(deviceFailure, 0, sizeof(int)));
            const int localRowStart = static_cast<int>(
                layout.localOffsets[diagonalBlock] / static_cast<size_t>(n));
            diagonalCholeskyKernel<<<1, kBlockSize>>>(
                deviceMatrix, n, localRowStart, diagonalStart, blockRows, deviceFailure);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int localFailure = 0;
            CUDA_CHECK(cudaMemcpy(&localFailure, deviceFailure, sizeof(int),
                                  cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy2D(
                diagonal.data(), static_cast<size_t>(blockRows) * sizeof(double),
                deviceMatrix + layout.localOffsets[diagonalBlock] + diagonalStart,
                static_cast<size_t>(n) * sizeof(double),
                static_cast<size_t>(blockRows) * sizeof(double), blockRows,
                cudaMemcpyDeviceToHost));
            if (localFailure != 0) {
                diagonal[0] = std::numeric_limits<double>::quiet_NaN();
            }
        }

        ensureMpiCount(static_cast<size_t>(blockRows) * blockRows, "MPI_Bcast", rank);
        MPI_Bcast(diagonal.data(), blockRows * blockRows, MPI_DOUBLE,
                  layout.blockOwner[diagonalBlock], communicator);

        int localFailure = std::isnan(diagonal[0]) ? 1 : 0;
        int globalFailure = 0;
        MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX, communicator);
        if (globalFailure != 0) {
            CUBLAS_CHECK(cublasDestroy(blas));
            CUDA_CHECK(cudaFree(deviceFailure));
            CUDA_CHECK(cudaFree(deviceDiagonal));
            CUDA_CHECK(cudaFree(devicePanel));
            CUDA_CHECK(cudaFree(deviceMatrix));
            return false;
        }

        CUDA_CHECK(cudaMemcpy(deviceDiagonal, diagonal.data(),
                              static_cast<size_t>(blockRows) * blockRows * sizeof(double),
                              cudaMemcpyHostToDevice));

        size_t localPanelElements = 0;
        for (int block : layout.localBlocks) {
            if (block > diagonalBlock) {
                localPanelElements += static_cast<size_t>(layout.blockRows[block]) * blockRows;
            }
        }
        ensureMpiCount(localPanelElements, "MPI_Allgatherv", rank);
        std::vector<double> localPanel(localPanelElements);

        size_t localPanelOffset = 0;
        const double one = 1.0;
        for (int block : layout.localBlocks) {
            if (block <= diagonalBlock) {
                continue;
            }
            const int rows = layout.blockRows[block];
            const int localRowStart = static_cast<int>(
                layout.localOffsets[block] / static_cast<size_t>(n));
            // The row-major block is viewed as a column-major transpose. With
            // the transposed upper triangle of L, this is L * X = B^T,
            // equivalent to X_row * L^T = B_row.
            CUBLAS_CHECK(cublasDtrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, blockRows, rows, &one, deviceDiagonal,
                blockRows, deviceMatrix + layout.localOffsets[block] + diagonalStart, n));

            CUDA_CHECK(cudaMemcpy2D(
                localPanel.data() + localPanelOffset, static_cast<size_t>(blockRows) * sizeof(double),
                deviceMatrix + layout.localOffsets[block] + diagonalStart,
                static_cast<size_t>(n) * sizeof(double),
                static_cast<size_t>(blockRows) * sizeof(double), rows,
                cudaMemcpyDeviceToHost));
            localPanelOffset += static_cast<size_t>(rows) * blockRows;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> receiveCounts(layout.worldSize, 0);
        std::vector<int> receiveDisplacements(layout.worldSize, 0);
        std::vector<size_t> rankPanelSizes(layout.worldSize, 0);
        for (int block = diagonalBlock + 1; block < blockCount; ++block) {
            rankPanelSizes[layout.blockOwner[block]] +=
                static_cast<size_t>(layout.blockRows[block]) * blockRows;
        }
        size_t totalPanelElements = 0;
        for (int process = 0; process < layout.worldSize; ++process) {
            ensureMpiCount(rankPanelSizes[process], "MPI_Allgatherv", rank);
            ensureMpiCount(totalPanelElements, "MPI_Allgatherv", rank);
            receiveCounts[process] = static_cast<int>(rankPanelSizes[process]);
            receiveDisplacements[process] = static_cast<int>(totalPanelElements);
            totalPanelElements += rankPanelSizes[process];
        }
        ensureMpiCount(totalPanelElements, "MPI_Allgatherv", rank);
        std::vector<double> gatheredPanel(totalPanelElements);
        MPI_Allgatherv(localPanel.empty() ? nullptr : localPanel.data(),
                       static_cast<int>(localPanelElements), MPI_DOUBLE,
                       gatheredPanel.empty() ? nullptr : gatheredPanel.data(),
                       receiveCounts.data(), receiveDisplacements.data(), MPI_DOUBLE,
                       communicator);

        std::vector<size_t> blockPanelOffsets(blockCount, 0);
        std::vector<size_t> rankCursors(layout.worldSize, 0);
        for (int block = diagonalBlock + 1; block < blockCount; ++block) {
            const int owner = layout.blockOwner[block];
            blockPanelOffsets[block] =
                static_cast<size_t>(receiveDisplacements[owner]) + rankCursors[owner];
            rankCursors[owner] += static_cast<size_t>(layout.blockRows[block]) * blockRows;
        }

        std::fill(panelHost.begin(), panelHost.begin() + static_cast<size_t>(n) * blockRows,
                  0.0);
#pragma omp parallel for schedule(static)
        for (long long block = diagonalBlock + 1; block < blockCount; ++block) {
            const size_t bytes = static_cast<size_t>(layout.blockRows[block]) * blockRows *
                                 sizeof(double);
            std::memcpy(panelHost.data() + static_cast<size_t>(layout.blockStart[block]) * blockRows,
                        gatheredPanel.data() + blockPanelOffsets[block], bytes);
        }
        CUDA_CHECK(cudaMemcpy(devicePanel, panelHost.data(),
                              static_cast<size_t>(n) * blockRows * sizeof(double),
                              cudaMemcpyHostToDevice));

        const double minusOne = -1.0;
        for (int block : layout.localBlocks) {
            if (block <= diagonalBlock) {
                continue;
            }
            const int rowsI = layout.blockRows[block];
            const int rowStart = layout.blockStart[block];
            const int columnStart = diagonalStart + blockRows;
            const int columns = rowStart + rowsI - columnStart;
            // In column-major view, the row-major C block is C^T and
            // P_trailing^T * P_i gives exactly the transposed update. Updating
            // the whole contiguous trailing range reduces the MPI/GPU launch
            // overhead to one GEMM per local block row.
            CUBLAS_CHECK(cublasDgemm(
                blas, CUBLAS_OP_T, CUBLAS_OP_N, columns, rowsI, blockRows,
                &minusOne,
                devicePanel + static_cast<size_t>(columnStart) * blockRows,
                blockRows,
                devicePanel + static_cast<size_t>(rowStart) * blockRows,
                blockRows,
                &one,
                deviceMatrix + layout.localOffsets[block] + columnStart,
                n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(localMatrix.data(), deviceMatrix,
                          localElements * sizeof(double), cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaFree(deviceFailure));
    CUDA_CHECK(cudaFree(deviceDiagonal));
    CUDA_CHECK(cudaFree(devicePanel));
    CUDA_CHECK(cudaFree(deviceMatrix));
    return true;
}

bool validateCholesky(const std::vector<double>& factor,
                      const std::vector<double>& original,
                      size_t n) {
    std::vector<double> reconstructed(n * n);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += factor[static_cast<size_t>(i) * n + k] *
                       factor[j * n + k];
            }
            reconstructed[static_cast<size_t>(i) * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relativeError = 0.0;
#pragma omp parallel for reduction(max : maxError, relativeError) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n * n); ++i) {
        const size_t index = static_cast<size_t>(i);
        const double error = std::fabs(reconstructed[index] - original[index]);
        maxError = std::max(maxError, error);
        relativeError = std::max(relativeError,
                                 error / (std::fabs(original[index]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
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

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        mpiAbort("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable", rank);
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                parseError = true;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            parseError = true;
        }
    }
    if (parseError) {
        if (rank == 0) {
            std::printf("Invalid command line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        mpiAbort("cudaGetDeviceCount", "the hybrid benchmark requires a CUDA device", rank);
    }
    const int cudaDevice = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(cudaDevice));
    CUDA_CHECK(cudaFree(nullptr));

    if (rank == 0) {
        std::printf("CUDA devices visible on this node: %d\n", deviceCount);
    }

    const DistributedLayout layout = makeLayout(static_cast<int>(n), rank, worldSize);
    std::vector<double> matrix;
    std::vector<double> original;
    std::vector<double> distributedInput;
    std::vector<int> counts(worldSize, 0);
    std::vector<int> displacements(worldSize, 0);

    size_t totalDistributedElements = 0;
    for (int process = 0; process < worldSize; ++process) {
        size_t processRows = 0;
        for (int block = 0; block < static_cast<int>(layout.blockStart.size()); ++block) {
            if (layout.blockOwner[block] == process) {
                processRows += static_cast<size_t>(layout.blockRows[block]);
            }
        }
        const size_t processElements = processRows * n;
        ensureMpiCount(processElements, "MPI_Scatterv", rank);
        ensureMpiCount(totalDistributedElements, "MPI_Scatterv", rank);
        counts[process] = static_cast<int>(processElements);
        displacements[process] = static_cast<int>(totalDistributedElements);
        totalDistributedElements += processElements;
    }
    ensureMpiCount(totalDistributedElements, "MPI_Scatterv", rank);

    if (rank == 0) {
        std::printf("Generating positive definite matrix...\n");
        matrix.resize(n * n);
        generatePositiveDefiniteMatrix(matrix, n);
        if (validate) {
            original = matrix;
        }
        clearUpperTriangle(matrix, n);

        distributedInput.resize(totalDistributedElements);
        for (int process = 0; process < worldSize; ++process) {
            size_t offset = static_cast<size_t>(displacements[process]);
            for (int block = 0; block < static_cast<int>(layout.blockStart.size()); ++block) {
                if (layout.blockOwner[block] != process) {
                    continue;
                }
                const size_t blockElements = static_cast<size_t>(layout.blockRows[block]) * n;
                std::memcpy(distributedInput.data() + offset,
                            matrix.data() + static_cast<size_t>(layout.blockStart[block]) * n,
                            blockElements * sizeof(double));
                offset += blockElements;
            }
        }
    }

    std::vector<double> localMatrix(layout.localRows * n);
    MPI_Scatterv(rank == 0 ? distributedInput.data() : nullptr, counts.data(),
                 displacements.data(), MPI_DOUBLE,
                 localMatrix.empty() ? nullptr : localMatrix.data(),
                 counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    distributedInput.clear();
    distributedInput.shrink_to_fit();
    if (rank == 0) {
        matrix.clear();
        matrix.shrink_to_fit();
    }

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = distributedCholesky(localMatrix, layout, cudaDevice, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    std::vector<double> distributedResult;
    if (rank == 0) {
        distributedResult.resize(totalDistributedElements);
    }
    MPI_Gatherv(localMatrix.empty() ? nullptr : localMatrix.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? distributedResult.data() : nullptr, counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        matrix.resize(n * n);
        for (int process = 0; process < worldSize; ++process) {
            size_t offset = static_cast<size_t>(displacements[process]);
            for (int block = 0; block < static_cast<int>(layout.blockStart.size()); ++block) {
                if (layout.blockOwner[block] != process) {
                    continue;
                }
                const size_t blockElements = static_cast<size_t>(layout.blockRows[block]) * n;
                std::memcpy(matrix.data() + static_cast<size_t>(layout.blockStart[block]) * n,
                            distributedResult.data() + offset,
                            blockElements * sizeof(double));
                offset += blockElements;
            }
        }
        const long durationMilliseconds = static_cast<long>(maximumElapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / std::max(maximumElapsed, 1e-12) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(matrix, "CholeskyL");
        }
    }

    bool valid = true;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        valid = validateCholesky(matrix, original, n);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    int validity = valid ? 1 : 0;
    MPI_Bcast(&validity, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return validity == 1 ? 0 : 1;
}
