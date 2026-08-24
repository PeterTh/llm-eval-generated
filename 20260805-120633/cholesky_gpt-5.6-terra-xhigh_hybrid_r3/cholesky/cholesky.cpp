#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Each rank owns a contiguous slab of rows.  The trailing update is therefore
// a local GEMM, while only the current panel is exchanged between ranks.
constexpr int kBlockSize = 256;

[[noreturn]] void fail(const char* message) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s\n", rank, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d: %s", expression, file,
                      line, cudaGetErrorString(status));
        fail(message);
    }
}

void checkCublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (cuBLAS status %d)",
                      expression, file, line, static_cast<int>(status));
        fail(message);
    }
}

void checkCusolver(cusolverStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (cuSOLVER status %d)",
                      expression, file, line, static_cast<int>(status));
        fail(message);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) checkCublas((expression), #expression, __FILE__, __LINE__)
#define CUSOLVER_CHECK(expression) checkCusolver((expression), #expression, __FILE__, __LINE__)

int rowBegin(int n, int rank, int ranks) {
    return static_cast<int>((static_cast<long long>(n) * rank) / ranks);
}

int rowEnd(int n, int rank, int ranks) {
    return static_cast<int>((static_cast<long long>(n) * (rank + 1)) / ranks);
}

// glibc rand_r advances its 32-bit LCG three times for each returned number.
// Jumping that recurrence gives every rank the exact portion of the original
// seed-42 B matrix without a serial random-number bottleneck.
std::uint32_t advanceRandState(std::uint32_t state, std::uint64_t steps) {
    std::uint32_t accumulatedMultiplier = 1;
    std::uint32_t accumulatedIncrement = 0;
    std::uint32_t multiplier = 1103515245U;
    std::uint32_t increment = 12345U;
    while (steps != 0) {
        if (steps & 1U) {
            accumulatedIncrement = multiplier * accumulatedIncrement + increment;
            accumulatedMultiplier *= multiplier;
        }
        increment = multiplier * increment + increment;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

// Preserve the original A = B * B^T + nI input exactly while distributing B
// by rows.  A ring exchanges one B slab at a time, so no rank stores a full
// second n-by-n input matrix.
void generatePositiveDefiniteLocal(std::vector<double>& matrix, int n, int globalRowBegin,
                                   int rank, int ranks) {
    const int localRows = static_cast<int>(matrix.size() / static_cast<size_t>(n));
    std::vector<double> localB(static_cast<size_t>(localRows) * n);
    const std::uint64_t firstElement = static_cast<std::uint64_t>(globalRowBegin) * n;
    unsigned int seed = advanceRandState(42U, 3U * firstElement);
    for (double& value : localB) {
        value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }

    const int maxRows = (n + ranks - 1) / ranks;
    std::vector<double> incomingA(static_cast<size_t>(maxRows) * n);
    std::vector<double> incomingB(static_cast<size_t>(maxRows) * n);
    const double* currentBlock = localB.data();
    double* receiveBlock = incomingA.data();
    int currentOwner = rank;

    for (int phase = 0; phase < ranks; ++phase) {
        const int columnBegin = rowBegin(n, currentOwner, ranks);
        const int columnEnd = rowEnd(n, currentOwner, ranks);
        const int columnRows = columnEnd - columnBegin;

#pragma omp parallel for schedule(static)
        for (int localRow = 0; localRow < localRows; ++localRow) {
            const double* bRow = localB.data() + static_cast<size_t>(localRow) * n;
            double* aRow = matrix.data() + static_cast<size_t>(localRow) * n;
            for (int blockRow = 0; blockRow < columnRows; ++blockRow) {
                const double* otherBRow = currentBlock + static_cast<size_t>(blockRow) * n;
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (int k = 0; k < n; ++k) {
                    sum += bRow[k] * otherBRow[k];
                }
                aRow[columnBegin + blockRow] = sum;
            }
        }

        if (phase + 1 < ranks) {
            const int destination = (rank + 1) % ranks;
            const int source = (rank + ranks - 1) % ranks;
            const int nextOwner = (currentOwner + ranks - 1) % ranks;
            const int nextRows = rowEnd(n, nextOwner, ranks) - rowBegin(n, nextOwner, ranks);
            MPI_Sendrecv(currentBlock, columnRows * n, MPI_DOUBLE, destination, 71,
                         receiveBlock, nextRows * n, MPI_DOUBLE, source, 71,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            currentBlock = receiveBlock;
            receiveBlock = receiveBlock == incomingA.data() ? incomingB.data() : incomingA.data();
            currentOwner = nextOwner;
        }
    }

#pragma omp parallel for schedule(static)
    for (int localRow = 0; localRow < localRows; ++localRow) {
        const int globalRow = globalRowBegin + localRow;
        matrix[static_cast<size_t>(localRow) * n + globalRow] += n;
    }
}

__global__ void zeroUpperTriangle(double* matrix, int localRows, int n, int globalRowBegin) {
    const int localRow = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow < localRows && column < n && column > globalRowBegin + localRow) {
        matrix[static_cast<size_t>(localRow) * n + column] = 0.0;
    }
}

void makeMatrixCounts(int n, int ranks, std::vector<int>& counts, std::vector<int>& displacements) {
    if (static_cast<long double>(n) * n > INT_MAX) {
        fail("matrix is too large for the MPI count interface used by this benchmark");
    }
    counts.resize(ranks);
    displacements.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        const int begin = rowBegin(n, rank, ranks);
        const int end = rowEnd(n, rank, ranks);
        counts[rank] = (end - begin) * n;
        displacements[rank] = begin * n;
    }
}

void makePanelCounts(int n, int panelEnd, int panelWidth, int ranks,
                     std::vector<int>& counts, std::vector<int>& displacements) {
    counts.resize(ranks);
    displacements.resize(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        const int begin = std::max(rowBegin(n, rank, ranks), panelEnd);
        const int end = rowEnd(n, rank, ranks);
        const int rows = std::max(0, end - begin);
        counts[rank] = rows * panelWidth;
        displacements[rank] = (begin - panelEnd) * panelWidth;
    }
}

bool validateCholesky(const std::vector<double>& localL, const std::vector<double>& localOriginal,
                      int n, int globalRowBegin, const std::vector<int>& matrixCounts,
                      const std::vector<int>& matrixDisplacements, int rank) {
    std::vector<double> fullL(static_cast<size_t>(n) * n);
    MPI_Allgatherv(localL.data(), static_cast<int>(localL.size()), MPI_DOUBLE, fullL.data(),
                   matrixCounts.data(), matrixDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    const int localRows = static_cast<int>(localL.size() / static_cast<size_t>(n));
    double localMaxError = 0.0;
    double localRelativeError = 0.0;

#pragma omp parallel for reduction(max : localMaxError, localRelativeError) schedule(static)
    for (int localRow = 0; localRow < localRows; ++localRow) {
        const int globalRow = globalRowBegin + localRow;
        for (int column = 0; column < n; ++column) {
            const int limit = std::min(globalRow, column);
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k <= limit; ++k) {
                sum += fullL[static_cast<size_t>(globalRow) * n + k] *
                       fullL[static_cast<size_t>(column) * n + k];
            }
            const double original = localOriginal[static_cast<size_t>(localRow) * n + column];
            const double error = std::abs(sum - original);
            localMaxError = std::max(localMaxError, error);
            localRelativeError = std::max(localRelativeError,
                                          error / (std::abs(original) + 1.0e-10));
        }
    }

    double maxError = 0.0;
    double relativeError = 0.0;
    MPI_Reduce(&localMaxError, &maxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localRelativeError, &relativeError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        std::printf("Max absolute error: %.10e\n", maxError);
        std::printf("Max relative error: %.10e\n", relativeError);
        valid = relativeError <= 1.0e-6;
        if (!valid) {
            std::printf("Validation failed: relative error too large\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the thread support required by OpenMP workers\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = 512;
    bool validate = false;
    bool printResults = false;
    bool usageOnly = false;
    bool argumentError = false;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "-n") == 0 && index + 1 < argc) {
            char* end = nullptr;
            const unsigned long value = std::strtoul(argv[++index], &end, 10);
            if (*argv[index] == '\0' || *end != '\0' || value == 0 || value > INT_MAX) {
                argumentError = true;
            } else {
                n = static_cast<int>(value);
            }
        } else if (std::strcmp(argv[index], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[index], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[index], "-h") == 0) {
            usageOnly = true;
        } else {
            argumentError = true;
        }
    }
    if (usageOnly || argumentError) {
        if (rank == 0) {
            if (argumentError) {
                std::printf("Invalid command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError ? 1 : 0;
    }

    std::vector<int> matrixCounts;
    std::vector<int> matrixDisplacements;
    makeMatrixCounts(n, ranks, matrixCounts, matrixDisplacements);

    const int localBegin = rowBegin(n, rank, ranks);
    const int localEnd = rowEnd(n, rank, ranks);
    const int localRows = localEnd - localBegin;
    const size_t localElements = static_cast<size_t>(localRows) * n;

    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    if (gpuCount == 0) {
        fail("no CUDA device is available");
    }
    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % gpuCount));
    MPI_Comm_free(&localCommunicator);

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("MPI ranks: %d, OpenMP max threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("CUDA: enabled (one GPU selected per node-local MPI rank)\n");
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> hostA(localElements);
    generatePositiveDefiniteLocal(hostA, n, localBegin, rank, ranks);
    std::vector<double> original;
    if (validate) {
        original = hostA;
    }

    cublasHandle_t cublasHandle = nullptr;
    cusolverDnHandle_t solverHandle = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUSOLVER_CHECK(cusolverDnCreate(&solverHandle));

    double* deviceA = nullptr;
    double* deviceDiagonal = nullptr;
    double* devicePanel = nullptr;
    double* deviceWork = nullptr;
    int* deviceInfo = nullptr;
    const size_t allocationElements = std::max<size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(&deviceA, allocationElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceDiagonal, static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&devicePanel, static_cast<size_t>(n) * kBlockSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceInfo, sizeof(int)));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), localElements * sizeof(double), cudaMemcpyHostToDevice));
    }
    hostA.clear();
    hostA.shrink_to_fit();

    int workspaceSize = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solverHandle, CUBLAS_FILL_MODE_UPPER, kBlockSize,
                                               deviceDiagonal, kBlockSize, &workspaceSize));
    CUDA_CHECK(cudaMalloc(&deviceWork, static_cast<size_t>(std::max(workspaceSize, 1)) * sizeof(double)));

    std::vector<int> diagonalCounts(ranks);
    std::vector<int> diagonalDisplacements(ranks);
    std::vector<int> panelCounts;
    std::vector<int> panelDisplacements;
    std::vector<double> diagonal(static_cast<size_t>(kBlockSize) * kBlockSize);

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double start = MPI_Wtime();

    bool success = true;
    for (int panelBegin = 0; panelBegin < n; panelBegin += kBlockSize) {
        const int panelWidth = std::min(kBlockSize, n - panelBegin);
        const int panelEnd = panelBegin + panelWidth;
        const int localDiagonalBegin = std::max(localBegin, panelBegin);
        const int localDiagonalEnd = std::min(localEnd, panelEnd);
        const int localDiagonalRows = std::max(0, localDiagonalEnd - localDiagonalBegin);
        std::vector<double> localDiagonal(static_cast<size_t>(localDiagonalRows) * panelWidth);

        if (localDiagonalRows != 0) {
            CUDA_CHECK(cudaMemcpy2D(localDiagonal.data(), static_cast<size_t>(panelWidth) * sizeof(double),
                                    deviceA + static_cast<size_t>(localDiagonalBegin - localBegin) * n + panelBegin,
                                    static_cast<size_t>(n) * sizeof(double),
                                    static_cast<size_t>(panelWidth) * sizeof(double), localDiagonalRows,
                                    cudaMemcpyDeviceToHost));
        }
        for (int process = 0; process < ranks; ++process) {
            const int begin = std::max(rowBegin(n, process, ranks), panelBegin);
            const int end = std::min(rowEnd(n, process, ranks), panelEnd);
            diagonalCounts[process] = std::max(0, end - begin) * panelWidth;
            diagonalDisplacements[process] = std::max(0, begin - panelBegin) * panelWidth;
        }
        MPI_Gatherv(localDiagonal.data(), static_cast<int>(localDiagonal.size()), MPI_DOUBLE,
                    diagonal.data(), diagonalCounts.data(), diagonalDisplacements.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        int panelIsPositiveDefinite = 1;
        if (rank == 0) {
            CUDA_CHECK(cudaMemcpy(deviceDiagonal, diagonal.data(),
                                  static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                                  cudaMemcpyHostToDevice));
            CUSOLVER_CHECK(cusolverDnDpotrf(solverHandle, CUBLAS_FILL_MODE_UPPER, panelWidth,
                                            deviceDiagonal, panelWidth, deviceWork, workspaceSize, deviceInfo));
            int factorizationInfo = 0;
            CUDA_CHECK(cudaMemcpy(&factorizationInfo, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (factorizationInfo != 0) {
                std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                            panelBegin + factorizationInfo - 1);
                panelIsPositiveDefinite = 0;
            } else {
                CUDA_CHECK(cudaMemcpy(diagonal.data(), deviceDiagonal,
                                      static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                for (int row = 0; row < panelWidth; ++row) {
                    for (int column = row + 1; column < panelWidth; ++column) {
                        diagonal[static_cast<size_t>(row) * panelWidth + column] = 0.0;
                    }
                }
            }
        }
        MPI_Bcast(&panelIsPositiveDefinite, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!panelIsPositiveDefinite) {
            success = false;
            break;
        }
        MPI_Bcast(diagonal.data(), panelWidth * panelWidth, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceDiagonal, diagonal.data(),
                              static_cast<size_t>(panelWidth) * panelWidth * sizeof(double),
                              cudaMemcpyHostToDevice));

        if (localDiagonalRows != 0) {
            CUDA_CHECK(cudaMemcpy2D(deviceA + static_cast<size_t>(localDiagonalBegin - localBegin) * n + panelBegin,
                                    static_cast<size_t>(n) * sizeof(double),
                                    diagonal.data() + static_cast<size_t>(localDiagonalBegin - panelBegin) * panelWidth,
                                    static_cast<size_t>(panelWidth) * sizeof(double),
                                    static_cast<size_t>(panelWidth) * sizeof(double), localDiagonalRows,
                                    cudaMemcpyHostToDevice));
        }

        const int localTrailingBegin = std::max(localBegin, panelEnd);
        const int localTrailingRows = std::max(0, localEnd - localTrailingBegin);
        const int trailingRows = n - panelEnd;
        if (localTrailingRows != 0) {
            const double one = 1.0;
            // Raw row-major data is viewed by cuBLAS as transposed.  The
            // UPPER/TRANS combination represents the row-major lower L tile.
            CUBLAS_CHECK(cublasDtrsm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, panelWidth, localTrailingRows,
                                     &one, deviceDiagonal, panelWidth,
                                     deviceA + static_cast<size_t>(localTrailingBegin - localBegin) * n + panelBegin,
                                     n));
        }

        if (trailingRows != 0) {
            std::vector<double> localPanel(static_cast<size_t>(localTrailingRows) * panelWidth);
            if (localTrailingRows != 0) {
                CUDA_CHECK(cudaMemcpy2D(localPanel.data(), static_cast<size_t>(panelWidth) * sizeof(double),
                                        deviceA + static_cast<size_t>(localTrailingBegin - localBegin) * n + panelBegin,
                                        static_cast<size_t>(n) * sizeof(double),
                                        static_cast<size_t>(panelWidth) * sizeof(double), localTrailingRows,
                                        cudaMemcpyDeviceToHost));
            }
            makePanelCounts(n, panelEnd, panelWidth, ranks, panelCounts, panelDisplacements);
            std::vector<double> fullPanel(static_cast<size_t>(trailingRows) * panelWidth);
            MPI_Allgatherv(localPanel.data(), static_cast<int>(localPanel.size()), MPI_DOUBLE,
                           fullPanel.data(), panelCounts.data(), panelDisplacements.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(devicePanel, fullPanel.data(), fullPanel.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));

            if (localTrailingRows != 0) {
                const double minusOne = -1.0;
                const double one = 1.0;
                // C = A^T: C <- C - L_panel * X^T.  This formulation keeps
                // the distributed matrix row-major without explicit transposes.
                CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         trailingRows, localTrailingRows, panelWidth, &minusOne,
                                         devicePanel, panelWidth,
                                         deviceA + static_cast<size_t>(localTrailingBegin - localBegin) * n + panelBegin,
                                         n, &one,
                                         deviceA + static_cast<size_t>(localTrailingBegin - localBegin) * n + panelEnd,
                                         n));
            }
        }
    }

    if (success) {
        constexpr dim3 block(32, 8);
        const dim3 grid((n + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);
        if (localRows != 0) {
            zeroUpperTriangle<<<grid, block>>>(deviceA, localRows, n, localBegin);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localL(localElements);
    if (success && localElements != 0) {
        CUDA_CHECK(cudaMemcpy(localL.data(), deviceA, localElements * sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(deviceInfo));
    CUDA_CHECK(cudaFree(deviceWork));
    CUDA_CHECK(cudaFree(devicePanel));
    CUDA_CHECK(cudaFree(deviceDiagonal));
    CUDA_CHECK(cudaFree(deviceA));
    CUSOLVER_CHECK(cusolverDnDestroy(solverHandle));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(std::llround(elapsedSeconds * 1000.0));
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = elapsedSeconds > 0.0 ? operations / elapsedSeconds / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if (printResults) {
        if (rank == 0) {
            fullL.resize(static_cast<size_t>(n) * n);
        }
        MPI_Gatherv(localL.data(), static_cast<int>(localL.size()), MPI_DOUBLE,
                    rank == 0 ? fullL.data() : nullptr, matrixCounts.data(), matrixDisplacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(fullL, "CholeskyL");
        }
    }

    int result = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateCholesky(localL, original, n, localBegin, matrixCounts,
                                            matrixDisplacements, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        result = valid ? 0 : 1;
    }

    MPI_Finalize();
    return result;
}
