#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous block rows.  Each MPI rank keeps its
// rows on one GPU, CUDA/cuBLAS/cuSOLVER execute the factorization and updates,
// and OpenMP is used for the host-side matrix generation and validation.
namespace {

constexpr size_t kBlockSize = 256;

[[noreturn]] void abortWithMessage(const char* where, const char* detail) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, where, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t cuda_status_ = (call);                                           \
        if (cuda_status_ != cudaSuccess) {                                                 \
            abortWithMessage(#call, cudaGetErrorString(cuda_status_));                     \
        }                                                                                   \
    } while (false)

#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        const cublasStatus_t cublas_status_ = (call);                                      \
        if (cublas_status_ != CUBLAS_STATUS_SUCCESS) {                                     \
            abortWithMessage(#call, "cuBLAS status is not CUBLAS_STATUS_SUCCESS");        \
        }                                                                                   \
    } while (false)

#define CUSOLVER_CHECK(call)                                                                \
    do {                                                                                    \
        const cusolverStatus_t cusolver_status_ = (call);                                  \
        if (cusolver_status_ != CUSOLVER_STATUS_SUCCESS) {                                 \
            abortWithMessage(#call, "cuSOLVER status is not CUSOLVER_STATUS_SUCCESS");    \
        }                                                                                   \
    } while (false)

#define MPI_CHECK(call)                                                                     \
    do {                                                                                    \
        const int mpi_status_ = (call);                                                    \
        if (mpi_status_ != MPI_SUCCESS) {                                                  \
            abortWithMessage(#call, "MPI call did not return MPI_SUCCESS");               \
        }                                                                                   \
    } while (false)

struct BlockRowPartition {
    size_t firstBlock;
    size_t lastBlock;
    size_t firstRow;
    size_t lastRow;
    size_t blockCount;
};

BlockRowPartition makePartition(size_t n, size_t blockSize, int rank, int ranks) {
    const size_t blocks = (n + blockSize - 1) / blockSize;
    const size_t firstBlock = blocks * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
    const size_t lastBlock = blocks * static_cast<size_t>(rank + 1) / static_cast<size_t>(ranks);
    return {firstBlock, lastBlock, std::min(n, firstBlock * blockSize),
            std::min(n, lastBlock * blockSize), blocks};
}

bool ownsBlock(const BlockRowPartition& partition, size_t block) {
    return block >= partition.firstBlock && block < partition.lastBlock;
}

int checkedMpiCount(size_t count, const char* what) {
    if (count > static_cast<size_t>(INT_MAX)) {
        abortWithMessage(what, "message size exceeds MPI's int count limit");
    }
    return static_cast<int>(count);
}

size_t checkedElements(size_t rows, size_t columns, const char* what) {
    if (rows != 0 && columns > std::numeric_limits<size_t>::max() / rows) {
        abortWithMessage(what, "allocation size overflows size_t");
    }
    return rows * columns;
}

void selectLocalGpu(int rank) {
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        abortWithMessage("cudaGetDeviceCount", "no CUDA accelerator is available");
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&localComm));
}

// Keep the original deterministic rand_r sequence.  Only rank zero produces
// it, then every rank receives the same B used to construct its local A rows.
void generateRandomMatrix(std::vector<double>& B) {
    unsigned int seed = 42;
    for (double& value : B) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
}

void generateLocalPositiveDefiniteMatrix(std::vector<double>& localA,
                                         const std::vector<double>& B,
                                         size_t n, size_t firstRow) {
    const size_t localRows = localA.size() / n;

#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(localRows); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        const double* const left = B.data() + globalRow * n;
        double* const output = localA.data() + static_cast<size_t>(localRow) * n;

        for (size_t column = 0; column < n; ++column) {
            const double* const right = B.data() + column * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t element = 0; element < n; ++element) {
                sum += left[element] * right[element];
            }
            output[column] = sum;
        }
        output[globalRow] += static_cast<double>(n);
    }
}

// CUDA performs the final formatting step so the gathered matrix has exactly
// the lower-triangular layout produced by the original benchmark.
__global__ void zeroUpperTriangle(double* matrix, size_t firstRow, size_t rows, size_t n) {
    const size_t total = rows * n;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const size_t localRow = index / n;
        const size_t column = index - localRow * n;
        if (column > firstRow + localRow) {
            matrix[index] = 0.0;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      size_t n) {
    double maxError = 0.0;
    double maxRelativeError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, maxRelativeError)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        const size_t i = static_cast<size_t>(row);
        for (size_t column = 0; column < n; ++column) {
            const size_t last = std::min(i, column);
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t element = 0; element <= last; ++element) {
                sum += L[i * n + element] * L[column * n + element];
            }
            const double error = std::fabs(sum - original[i * n + column]);
            maxError = std::max(maxError, error);
            maxRelativeError = std::max(
                maxRelativeError, error / (std::fabs(original[i * n + column]) + 1.0e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", maxRelativeError);
    if (maxRelativeError > 1.0e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, size_t* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
}

bool factorDistributed(double* deviceMatrix, const BlockRowPartition& partition, size_t n,
                       size_t blockSize, int rank, int ranks, cublasHandle_t cublas,
                       cusolverDnHandle_t solver, double* deviceDiagonal,
                       double* deviceSolverWork, int solverWorkSize,
                       int* deviceSolverInfo, double* deviceLocalPanel,
                       double* deviceGlobalPanel, std::vector<double>& hostDiagonal,
                       std::vector<double>& hostLocalPanel,
                       std::vector<double>& hostGlobalPanel) {
    const double minusOne = -1.0;
    const double one = 1.0;
    std::vector<int> receiveCounts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));

    for (size_t panelBlock = 0; panelBlock < partition.blockCount; ++panelBlock) {
        const size_t panelStart = panelBlock * blockSize;
        const size_t panelWidth = std::min(blockSize, n - panelStart);
        const size_t panelEnd = panelStart + panelWidth;
        // This is the inverse of makePartition's floor-based block ranges.
        // It also selects the non-empty owner correctly when there are more
        // MPI ranks than block rows.
        const int owner = static_cast<int>(
            ((panelBlock + 1) * static_cast<size_t>(ranks) - 1) / partition.blockCount);

        int localSuccess = 1;
        if (rank == owner) {
            const size_t localPanelStart = panelStart - partition.firstRow;
            CUDA_CHECK(cudaMemcpy2D(deviceDiagonal, panelWidth * sizeof(double),
                                    deviceMatrix + localPanelStart * n + panelStart,
                                    n * sizeof(double), panelWidth * sizeof(double), panelWidth,
                                    cudaMemcpyDeviceToDevice));
            CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER,
                                            static_cast<int>(panelWidth), deviceDiagonal,
                                            static_cast<int>(panelWidth), deviceSolverWork,
                                            solverWorkSize, deviceSolverInfo));
            int solverInfo = 0;
            CUDA_CHECK(cudaMemcpy(&solverInfo, deviceSolverInfo, sizeof(solverInfo),
                                  cudaMemcpyDeviceToHost));
            if (solverInfo != 0) {
                if (solverInfo > 0) {
                    std::fprintf(stderr,
                                 "Rank %d: Matrix is not positive definite at diagonal element %zu\n",
                                 rank, panelStart + static_cast<size_t>(solverInfo - 1));
                } else {
                    std::fprintf(stderr, "Rank %d: cuSOLVER received invalid argument %d\n", rank,
                                 -solverInfo);
                }
                localSuccess = 0;
            }
        }

        int globallySuccessful = 0;
        MPI_CHECK(MPI_Allreduce(&localSuccess, &globallySuccessful, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        if (globallySuccessful == 0) {
            return false;
        }

        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(hostDiagonal.data(), deviceDiagonal,
                                  panelWidth * panelWidth * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(hostDiagonal.data(), checkedMpiCount(panelWidth * panelWidth,
                                                                  "diagonal-panel broadcast"),
                           MPI_DOUBLE, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(deviceDiagonal, hostDiagonal.data(),
                              panelWidth * panelWidth * sizeof(double), cudaMemcpyHostToDevice));

        if (rank == owner) {
            const size_t localPanelStart = panelStart - partition.firstRow;
            CUDA_CHECK(cudaMemcpy2D(deviceMatrix + localPanelStart * n + panelStart,
                                    n * sizeof(double), deviceDiagonal,
                                    panelWidth * sizeof(double), panelWidth * sizeof(double),
                                    panelWidth, cudaMemcpyDeviceToDevice));
        }

        const size_t localStart = std::max(partition.firstRow, panelEnd);
        const size_t localRows = partition.lastRow > localStart ? partition.lastRow - localStart : 0;
        if (localRows != 0) {
            const size_t localOffset = localStart - partition.firstRow;
            CUDA_CHECK(cudaMemcpy2D(deviceLocalPanel, panelWidth * sizeof(double),
                                    deviceMatrix + localOffset * n + panelStart,
                                    n * sizeof(double), panelWidth * sizeof(double), localRows,
                                    cudaMemcpyDeviceToDevice));

            // In row-major memory L is seen by cuBLAS as L^T.  Solving with the
            // transposed upper view therefore performs A_ik * inv(L_kk^T).
            CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                     static_cast<int>(panelWidth), static_cast<int>(localRows),
                                     &one, deviceDiagonal, static_cast<int>(panelWidth),
                                     deviceLocalPanel, static_cast<int>(panelWidth)));
            CUDA_CHECK(cudaMemcpy2D(deviceMatrix + localOffset * n + panelStart,
                                    n * sizeof(double), deviceLocalPanel,
                                    panelWidth * sizeof(double), panelWidth * sizeof(double),
                                    localRows, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(hostLocalPanel.data(), deviceLocalPanel,
                                  localRows * panelWidth * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        const size_t trailingRows = n - panelEnd;
        for (int process = 0; process < ranks; ++process) {
            const BlockRowPartition processPartition =
                makePartition(n, blockSize, process, ranks);
            const size_t processStart = std::max(processPartition.firstRow, panelEnd);
            const size_t processRows = processPartition.lastRow > processStart
                                           ? processPartition.lastRow - processStart
                                           : 0;
            receiveCounts[static_cast<size_t>(process)] =
                checkedMpiCount(processRows * panelWidth, "panel allgather receive count");
            displacements[static_cast<size_t>(process)] = checkedMpiCount(
                (processStart - panelEnd) * panelWidth, "panel allgather displacement");
        }
        MPI_CHECK(MPI_Allgatherv(hostLocalPanel.data(),
                                 checkedMpiCount(localRows * panelWidth,
                                                 "panel allgather send count"),
                                 MPI_DOUBLE, hostGlobalPanel.data(), receiveCounts.data(),
                                 displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD));
        if (trailingRows != 0) {
            CUDA_CHECK(cudaMemcpy(deviceGlobalPanel, hostGlobalPanel.data(),
                                  trailingRows * panelWidth * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }

        // Only the lower blocks are updated.  Mapping row-major submatrices to
        // their transposed column-major views lets cuBLAS update contiguous GPU
        // block rows without an extra transpose or communication phase.
        for (size_t rowBlock = panelBlock + 1; rowBlock < partition.blockCount; ++rowBlock) {
            if (!ownsBlock(partition, rowBlock)) {
                continue;
            }
            const size_t rowStart = rowBlock * blockSize;
            const size_t rowWidth = std::min(blockSize, n - rowStart);
            const size_t rowOffset = rowStart - partition.firstRow;
            const size_t rowPanelOffset = (rowStart - panelEnd) * panelWidth;
            const double* const rowPanel = deviceGlobalPanel + rowPanelOffset;

            for (size_t columnBlock = panelBlock + 1; columnBlock < rowBlock; ++columnBlock) {
                const size_t columnStart = columnBlock * blockSize;
                const size_t columnWidth = std::min(blockSize, n - columnStart);
                const size_t columnPanelOffset = (columnStart - panelEnd) * panelWidth;
                const double* const columnPanel = deviceGlobalPanel + columnPanelOffset;
                CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                         static_cast<int>(columnWidth),
                                         static_cast<int>(rowWidth),
                                         static_cast<int>(panelWidth), &minusOne, columnPanel,
                                         static_cast<int>(panelWidth), rowPanel,
                                         static_cast<int>(panelWidth), &one,
                                         deviceMatrix + rowOffset * n + columnStart,
                                         static_cast<int>(n)));
            }

            CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     static_cast<int>(rowWidth), static_cast<int>(panelWidth),
                                     &minusOne, rowPanel, static_cast<int>(panelWidth), &one,
                                     deviceMatrix + rowOffset * n + rowStart,
                                     static_cast<int>(n)));
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI_Init_thread", "MPI_THREAD_FUNNELED support is required");
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool showHelp = false;
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            if (!parseSize(argv[++argument], &n)) {
                argumentError = true;
            }
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            showHelp = true;
        } else {
            argumentError = true;
        }
    }

    if (showHelp || argumentError) {
        if (rank == 0) {
            if (argumentError) {
                std::printf("Invalid command-line options\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large for the CUDA and MPI interfaces\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const size_t matrixElements = checkedElements(n, n, "matrix allocation");
    if (matrixElements > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix is too large for the MPI count interface\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    selectLocalGpu(rank);
    const size_t blockSize = std::min(kBlockSize, n);
    const BlockRowPartition partition = makePartition(n, blockSize, rank, ranks);
    const size_t localRows = partition.lastRow - partition.firstRow;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d, CUDA block size: %zu, OpenMP threads/rank: %d\n", ranks,
                    blockSize, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> randomMatrix(matrixElements);
    if (rank == 0) {
        generateRandomMatrix(randomMatrix);
    }
    MPI_CHECK(MPI_Bcast(randomMatrix.data(), checkedMpiCount(randomMatrix.size(), "B broadcast"),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD));

    std::vector<double> localMatrix(checkedElements(localRows, n, "local matrix allocation"));
    generateLocalPositiveDefiniteMatrix(localMatrix, randomMatrix, n, partition.firstRow);
    randomMatrix.clear();
    randomMatrix.shrink_to_fit();
    std::vector<double> localOriginal;
    if (validate) {
        localOriginal = localMatrix;
    }

    double* deviceMatrix = nullptr;
    double* deviceDiagonal = nullptr;
    double* deviceSolverWork = nullptr;
    int* deviceSolverInfo = nullptr;
    double* deviceLocalPanel = nullptr;
    double* deviceGlobalPanel = nullptr;
    const size_t localCapacity = std::max<size_t>(1, checkedElements(localRows, blockSize,
                                                                       "local panel allocation"));
    const size_t globalCapacity = std::max<size_t>(1, checkedElements(n, blockSize,
                                                                        "global panel allocation"));
    CUDA_CHECK(cudaMalloc(&deviceMatrix,
                          std::max<size_t>(1, checkedElements(localRows, n, "GPU matrix allocation")) *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceDiagonal, checkedElements(blockSize, blockSize,
                                                            "GPU diagonal allocation") * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceLocalPanel, localCapacity * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceGlobalPanel, globalCapacity * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceSolverInfo, sizeof(int)));

    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    int solverWorkSize = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               static_cast<int>(blockSize), deviceDiagonal,
                                               static_cast<int>(blockSize), &solverWorkSize));
    CUDA_CHECK(cudaMalloc(&deviceSolverWork,
                          std::max(1, solverWorkSize) * static_cast<int>(sizeof(double))));

    if (localRows != 0) {
        CUDA_CHECK(cudaMemcpy(deviceMatrix, localMatrix.data(),
                              checkedElements(localRows, n, "GPU matrix copy") * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    std::vector<double> hostDiagonal(checkedElements(blockSize, blockSize,
                                                       "host diagonal allocation"));
    std::vector<double> hostLocalPanel(localCapacity);
    std::vector<double> hostGlobalPanel(globalCapacity);

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    const bool success = factorDistributed(
        deviceMatrix, partition, n, blockSize, rank, ranks, cublas, solver, deviceDiagonal,
        deviceSolverWork, solverWorkSize, deviceSolverInfo, deviceLocalPanel, deviceGlobalPanel,
        hostDiagonal, hostLocalPanel, hostGlobalPanel);

    if (success && localRows != 0) {
        constexpr int threads = 256;
        const size_t elements = checkedElements(localRows, n, "upper-triangle kernel size");
        const int blocks = static_cast<int>(std::min<size_t>(65535, (elements + threads - 1) / threads));
        zeroUpperTriangle<<<blocks, threads>>>(deviceMatrix, partition.firstRow, localRows, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        CUSOLVER_CHECK(cusolverDnDestroy(solver));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUDA_CHECK(cudaFree(deviceSolverWork));
        CUDA_CHECK(cudaFree(deviceSolverInfo));
        CUDA_CHECK(cudaFree(deviceGlobalPanel));
        CUDA_CHECK(cudaFree(deviceLocalPanel));
        CUDA_CHECK(cudaFree(deviceDiagonal));
        CUDA_CHECK(cudaFree(deviceMatrix));
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(std::llround(seconds * 1000.0));
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n) / 3.0;
        const double gflops = seconds > 0.0 ? operations / seconds / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (localRows != 0) {
        const size_t elements = checkedElements(localRows, n, "result copy size");
        CUDA_CHECK(cudaMemcpy(localMatrix.data(), deviceMatrix, elements * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> rowReceiveCounts(static_cast<size_t>(ranks));
    std::vector<int> rowDisplacements(static_cast<size_t>(ranks));
    for (int process = 0; process < ranks; ++process) {
        const BlockRowPartition processPartition = makePartition(n, blockSize, process, ranks);
        rowReceiveCounts[static_cast<size_t>(process)] = checkedMpiCount(
            checkedElements(processPartition.lastRow - processPartition.firstRow, n,
                            "result gather count"),
            "result gather count");
        rowDisplacements[static_cast<size_t>(process)] = checkedMpiCount(
            checkedElements(processPartition.firstRow, n, "result gather displacement"),
            "result gather displacement");
    }

    std::vector<double> factor;
    std::vector<double> original;
    if (rank == 0 && (printResults || validate)) {
        factor.resize(checkedElements(n, n, "gathered factor allocation"));
        if (validate) {
            original.resize(checkedElements(n, n, "gathered original allocation"));
        }
    }
    if (printResults || validate) {
        MPI_CHECK(MPI_Gatherv(localMatrix.data(), checkedMpiCount(localMatrix.size(), "result send count"),
                              MPI_DOUBLE, factor.data(), rowReceiveCounts.data(),
                              rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (validate) {
            MPI_CHECK(MPI_Gatherv(localOriginal.data(),
                                  checkedMpiCount(localOriginal.size(), "original send count"),
                                  MPI_DOUBLE, original.data(), rowReceiveCounts.data(),
                                  rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        }
    }

    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(deviceSolverWork));
    CUDA_CHECK(cudaFree(deviceSolverInfo));
    CUDA_CHECK(cudaFree(deviceGlobalPanel));
    CUDA_CHECK(cudaFree(deviceLocalPanel));
    CUDA_CHECK(cudaFree(deviceDiagonal));
    CUDA_CHECK(cudaFree(deviceMatrix));

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        if (printResults) {
            print_results(factor, "CholeskyL");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateCholesky(factor, original, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}
