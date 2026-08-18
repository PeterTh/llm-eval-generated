#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 64;

[[noreturn]] void abortRun(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation,
                      cudaGetErrorString(status));
        abortRun(message, rank);
    }
}

void checkCublas(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[256];
        std::snprintf(message, sizeof(message), "%s: cuBLAS status %d", operation,
                      static_cast<int>(status));
        abortRun(message, rank);
    }
}

__global__ void formMatrixKernel(const double* B, double* A, size_t n,
                                 size_t rowBegin, size_t localRows) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t localI = blockIdx.y * blockDim.y + threadIdx.y;
    if (localI >= localRows || j >= n) return;

    const size_t i = rowBegin + localI;
    double sum = 0.0;
    for (size_t k = 0; k < n; ++k) {
        sum += B[i * n + k] * B[j * n + k];
    }
    A[localI * n + j] = sum + (i == j ? static_cast<double>(n) : 0.0);
}

// The diagonal tile is small. Threads cooperate across its rows while the
// columns retain the required Cholesky dependency order.
__global__ void factorDiagonalKernel(double* A, size_t n, size_t localK,
                                     int blockSize, int* info) {
    for (int j = 0; j < blockSize; ++j) {
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (int t = 0; t < j; ++t) {
                const double value = A[(localK + j) * n + t];
                sum += value * value;
            }
            const double value = A[(localK + j) * n + j] - sum;
            if (!(value > 0.0)) {
                atomicMin(info, j);
            } else {
                A[(localK + j) * n + j] = sqrt(value);
            }
        }
        __syncthreads();
        if (*info != INT_MAX) return;

        for (int i = j + 1 + threadIdx.x; i < blockSize; i += blockDim.x) {
            double sum = 0.0;
            for (int t = 0; t < j; ++t) {
                sum += A[(localK + i) * n + t] * A[(localK + j) * n + t];
            }
            A[(localK + i) * n + j] =
                (A[(localK + i) * n + j] - sum) /
                A[(localK + j) * n + j];
        }
        __syncthreads();
    }
}

__global__ void panelSolveKernel(double* A, size_t n, size_t rowBegin,
                                 size_t firstRow, size_t localRows,
                                 size_t k, int blockSize,
                                 const double* diagonal) {
    const size_t localI = firstRow + blockIdx.x * blockDim.x + threadIdx.x;
    if (localI >= localRows) return;

    for (int j = 0; j < blockSize; ++j) {
        double value = A[localI * n + k + j];
        for (int t = 0; t < j; ++t) {
            value -= A[localI * n + k + t] * diagonal[j * blockSize + t];
        }
        A[localI * n + k + j] = value / diagonal[j * blockSize + j];
    }
    (void)rowBegin;
}

__global__ void packPanelKernel(const double* A, double* panel, size_t n,
                                size_t localRows, size_t k, int blockSize) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = localRows * static_cast<size_t>(blockSize);
    if (index >= count) return;
    const size_t localI = index / blockSize;
    const int j = static_cast<int>(index % blockSize);
    panel[index] = A[localI * n + k + j];
}

__global__ void zeroUpperKernel(double* A, size_t n, size_t rowBegin,
                                size_t localRows) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t localI = blockIdx.y * blockDim.y + threadIdx.y;
    if (localI < localRows && j < n && j > rowBegin + localI) {
        A[localI * n + j] = 0.0;
    }
}

size_t rowStart(int rank, int ranks, size_t n) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
}

int rowOwner(size_t row, int ranks, size_t n) {
    return static_cast<int>(((row + 1) * static_cast<size_t>(ranks) - 1) / n);
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseArguments(int argc, char** argv, size_t& n, bool& validate,
                    bool& printResults, bool& help, int rank) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return false;
            }
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for collapse(2) reduction(max : maxError, relError) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t last = std::min(i, j);
            for (size_t k = 0; k <= last; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = std::fabs(sum - original[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(original[i * n + j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) abortRun("MPI lacks required thread support", rank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    const bool argumentsOk =
        parseArguments(argc, argv, n, validate, printResults, help, rank);
    int allArgumentsOk = 0;
    const int localArgumentsOk = argumentsOk ? 1 : 0;
    MPI_Allreduce(&localArgumentsOk, &allArgumentsOk, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!allArgumentsOk || help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return allArgumentsOk ? 0 : 1;
    }
    if (n > static_cast<size_t>(INT_MAX / kBlockSize) ||
        n > std::numeric_limits<size_t>::max() / n) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for this MPI build\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) abortRun("no CUDA device is available", rank);
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    MPI_Comm_free(&localComm);
    cublasHandle_t blas;
    checkCublas(cublasCreate(&blas), "creating cuBLAS handle", rank);

    const size_t begin = rowStart(rank, ranks, n);
    const size_t end = rowStart(rank + 1, ranks, n);
    const size_t localRows = end - begin;
    const size_t localElements = localRows * n;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n",
                    ranks, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }

    // Keep rand_r's original deterministic stream; OpenMP performs the large,
    // independent conversion to doubles.
    std::vector<unsigned int> randomValues(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) randomValues[i] = rand_r(&seed);
    std::vector<double> B(n * n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = randomValues[i] / static_cast<double>(RAND_MAX) - 0.5;
    }
    randomValues.clear();
    randomValues.shrink_to_fit();

    double* dB = nullptr;
    double* dA = nullptr;
    checkCuda(cudaMalloc(&dB, std::max<size_t>(1, n * n) * sizeof(double)),
              "allocating B", rank);
    checkCuda(cudaMalloc(&dA, std::max<size_t>(1, localElements) * sizeof(double)),
              "allocating local A", rank);
    checkCuda(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice),
              "copying B", rank);
    B.clear();
    B.shrink_to_fit();

    const dim3 matrixThreads(16, 16);
    const dim3 matrixBlocks((n + 15) / 16, (localRows + 15) / 16);
    if (localRows != 0) {
        formMatrixKernel<<<matrixBlocks, matrixThreads>>>(dB, dA, n, begin, localRows);
        checkCuda(cudaGetLastError(), "launching matrix generation", rank);
        checkCuda(cudaDeviceSynchronize(), "generating matrix", rank);
    }
    checkCuda(cudaFree(dB), "freeing B", rank);

    std::vector<double> originalLocal;
    if (validate) {
        originalLocal.resize(localElements);
        checkCuda(cudaMemcpy(originalLocal.data(), dA, localElements * sizeof(double),
                             cudaMemcpyDeviceToHost), "saving original matrix", rank);
    }

    double* dDiagonal = nullptr;
    double* dLocalPanel = nullptr;
    double* dGlobalPanel = nullptr;
    int* dInfo = nullptr;
    checkCuda(cudaMalloc(&dDiagonal, kBlockSize * kBlockSize * sizeof(double)),
              "allocating diagonal tile", rank);
    checkCuda(cudaMalloc(&dLocalPanel,
                         std::max<size_t>(1, localRows * kBlockSize) * sizeof(double)),
              "allocating local panel", rank);
    checkCuda(cudaMalloc(&dGlobalPanel, n * kBlockSize * sizeof(double)),
              "allocating global panel", rank);
    checkCuda(cudaMalloc(&dInfo, sizeof(int)), "allocating factor status", rank);

    double* hostLocalPanel = nullptr;
    double* hostGlobalPanel = nullptr;
    checkCuda(cudaMallocHost(&hostLocalPanel,
                             std::max<size_t>(1, localRows * kBlockSize) * sizeof(double)),
              "allocating pinned local panel", rank);
    checkCuda(cudaMallocHost(&hostGlobalPanel, n * kBlockSize * sizeof(double)),
              "allocating pinned global panel", rank);
    std::vector<double> diagonal(kBlockSize * kBlockSize);
    std::vector<int> panelCounts(ranks);
    std::vector<int> panelOffsets(ranks);

    // Force CUDA context and cuBLAS kernel initialization outside the timed
    // factorization region; otherwise the first GEMM includes JIT/setup cost.
    const double warmupOne = 1.0;
    checkCuda(cudaMemset(dDiagonal, 0, 3 * sizeof(double)), "preparing GPU warmup", rank);
    checkCublas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                            &warmupOne, dDiagonal, 1, dDiagonal + 1, 1,
                            &warmupOne, dDiagonal + 2, 1),
                "warming up cuBLAS", rank);
    checkCuda(cudaDeviceSynchronize(), "warming up GPU", rank);

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
    bool success = true;
    size_t failedColumn = 0;

    for (size_t k = 0; k < n;) {
        const int owner = rowOwner(k, ranks, n);
        const size_t ownerEnd = rowStart(owner + 1, ranks, n);
        const int blockSize = static_cast<int>(
            std::min({static_cast<size_t>(kBlockSize), n - k, ownerEnd - k}));

        int info = INT_MAX;
        if (rank == owner) {
            checkCuda(cudaMemcpy(dInfo, &info, sizeof(int), cudaMemcpyHostToDevice),
                      "resetting factor status", rank);
            factorDiagonalKernel<<<1, 128>>>(dA + k, n, k - begin, blockSize, dInfo);
            checkCuda(cudaGetLastError(), "launching diagonal factorization", rank);
            checkCuda(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost),
                      "reading factor status", rank);
            if (info == INT_MAX) {
                checkCuda(cudaMemcpy2D(diagonal.data(), blockSize * sizeof(double),
                                       dA + (k - begin) * n + k, n * sizeof(double),
                                       blockSize * sizeof(double), blockSize,
                                       cudaMemcpyDeviceToHost),
                          "copying diagonal tile", rank);
            }
        }
        MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (info != INT_MAX) {
            success = false;
            failedColumn = k + static_cast<size_t>(info);
            break;
        }
        MPI_Bcast(diagonal.data(), blockSize * blockSize, MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dDiagonal, diagonal.data(),
                             blockSize * blockSize * sizeof(double),
                             cudaMemcpyHostToDevice), "copying diagonal tile to GPU", rank);

        const size_t firstLocal = std::max(k + static_cast<size_t>(blockSize), begin);
        const size_t firstLocalIndex = firstLocal < end ? firstLocal - begin : localRows;
        if (firstLocalIndex < localRows) {
            const int blocks = static_cast<int>((localRows - firstLocalIndex + 255) / 256);
            panelSolveKernel<<<blocks, 256>>>(dA, n, begin, firstLocalIndex, localRows,
                                              k, blockSize, dDiagonal);
            checkCuda(cudaGetLastError(), "launching panel solve", rank);
        }

        const size_t localPanelElements = localRows * static_cast<size_t>(blockSize);
        if (localPanelElements != 0) {
            const int blocks = static_cast<int>((localPanelElements + 255) / 256);
            packPanelKernel<<<blocks, 256>>>(dA, dLocalPanel, n, localRows, k, blockSize);
            checkCuda(cudaGetLastError(), "launching panel pack", rank);
            checkCuda(cudaMemcpy(hostLocalPanel, dLocalPanel,
                                 localPanelElements * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copying local panel", rank);
        }
        for (int r = 0; r < ranks; ++r) {
            panelCounts[r] = static_cast<int>((rowStart(r + 1, ranks, n) -
                                                rowStart(r, ranks, n)) * blockSize);
            panelOffsets[r] = static_cast<int>(rowStart(r, ranks, n) * blockSize);
        }
        MPI_Allgatherv(hostLocalPanel, static_cast<int>(localPanelElements), MPI_DOUBLE,
                       hostGlobalPanel, panelCounts.data(), panelOffsets.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dGlobalPanel, hostGlobalPanel,
                             n * static_cast<size_t>(blockSize) * sizeof(double),
                             cudaMemcpyHostToDevice), "copying global panel", rank);

        const size_t trailingStart = k + static_cast<size_t>(blockSize);
        const size_t firstUpdateRow = std::max(trailingStart, begin);
        if (trailingStart < n && firstUpdateRow < end) {
            // Row-major C -= P_local * P_global^T is column-major
            // C^T -= P_global * P_local^T. A single tuned GEMM is faster than
            // many triangular micro-kernels; upper entries are cleared later.
            const int updateColumns = static_cast<int>(n - trailingStart);
            const int updateRows = static_cast<int>(end - firstUpdateRow);
            const double minusOne = -1.0;
            const double one = 1.0;
            checkCublas(
                cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, updateColumns,
                            updateRows, blockSize, &minusOne,
                            dGlobalPanel + trailingStart * blockSize, blockSize,
                            dGlobalPanel + firstUpdateRow * blockSize, blockSize,
                            &one,
                            dA + (firstUpdateRow - begin) * n + trailingStart,
                            static_cast<int>(n)),
                "launching trailing GEMM", rank);
            checkCuda(cudaDeviceSynchronize(), "updating trailing matrix", rank);
        }
        k += static_cast<size_t>(blockSize);
    }

    if (success && localRows != 0) {
        zeroUpperKernel<<<matrixBlocks, matrixThreads>>>(dA, n, begin, localRows);
        checkCuda(cudaGetLastError(), "launching upper-triangle clear", rank);
        checkCuda(cudaDeviceSynchronize(), "clearing upper triangle", rank);
    }
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                        failedColumn);
            std::printf("Cholesky decomposition failed\n");
        }
        cudaFreeHost(hostGlobalPanel);
        cudaFreeHost(hostLocalPanel);
        cudaFree(dInfo);
        cudaFree(dGlobalPanel);
        cudaFree(dLocalPanel);
        cudaFree(dDiagonal);
        cudaFree(dA);
        cublasDestroy(blas);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", operations / elapsed / 1e9);
    }

    std::vector<double> localResult;
    std::vector<double> result;
    std::vector<double> original;
    if (validate || printResults) {
        if (localElements > static_cast<size_t>(INT_MAX))
            abortRun("local result exceeds MPI's count limit", rank);
        localResult.resize(localElements);
        checkCuda(cudaMemcpy(localResult.data(), dA, localElements * sizeof(double),
                             cudaMemcpyDeviceToHost), "copying result to host", rank);
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t count = (rowStart(r + 1, ranks, n) - rowStart(r, ranks, n)) * n;
            const size_t offset = rowStart(r, ranks, n) * n;
            if (count > static_cast<size_t>(INT_MAX) || offset > static_cast<size_t>(INT_MAX))
                abortRun("global result exceeds MPI's count limit", rank);
            counts[r] = static_cast<int>(count);
            offsets[r] = static_cast<int>(offset);
        }
        if (rank == 0) result.resize(n * n);
        MPI_Gatherv(localResult.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr, counts.data(), offsets.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            if (rank == 0) original.resize(n * n);
            MPI_Gatherv(originalLocal.data(), static_cast<int>(localElements), MPI_DOUBLE,
                        rank == 0 ? original.data() : nullptr, counts.data(), offsets.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) print_results(result, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            if (validateCholesky(result, original, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cudaFreeHost(hostGlobalPanel);
    cudaFreeHost(hostLocalPanel);
    cudaFree(dInfo);
    cudaFree(dGlobalPanel);
    cudaFree(dLocalPanel);
    cudaFree(dDiagonal);
    cudaFree(dA);
    cublasDestroy(blas);
    MPI_Finalize();
    return exitCode;
}
