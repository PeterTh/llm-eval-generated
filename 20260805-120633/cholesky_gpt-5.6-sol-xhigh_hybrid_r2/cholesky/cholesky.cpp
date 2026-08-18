#include <algorithm>
#include <cerrno>
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
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Each MPI rank owns block-cyclic matrix rows.  The block size is also the
// algorithmic tile size, which makes every diagonal tile local to one rank.
constexpr int kBlockSize = 256;

[[noreturn]] void abortAll(const char* what, const char* detail, int line) {
    int initialized = 0;
    int rank = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s failed at line %d: %s\n", rank, what,
                 line, detail);
    std::fflush(stderr);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

void checkMpi(int status, const char* expression, int line) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    message[length] = '\0';
    abortAll(expression, message, line);
}

void checkCuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        abortAll(expression, cudaGetErrorString(status), line);
    }
}

void checkCublas(cublasStatus_t status, const char* expression, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuBLAS status %d",
                      static_cast<int>(status));
        abortAll(expression, message, line);
    }
}

void checkCusolver(cusolverStatus_t status, const char* expression, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuSOLVER status %d",
                      static_cast<int>(status));
        abortAll(expression, message, line);
    }
}

#define MPI_CHECK(call) checkMpi((call), #call, __LINE__)
#define CUDA_CHECK(call) checkCuda((call), #call, __LINE__)
#define CUBLAS_CHECK(call) checkCublas((call), #call, __LINE__)
#define CUSOLVER_CHECK(call) checkCusolver((call), #call, __LINE__)

size_t checkedProduct(size_t a, size_t b, const char* description) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        abortAll(description, "size overflow", __LINE__);
    }
    return a * b;
}

int ownerOfRow(size_t globalRow, int ranks) {
    return static_cast<int>((globalRow / kBlockSize) %
                            static_cast<size_t>(ranks));
}

size_t localIndexOfRow(size_t globalRow, int ranks) {
    const size_t block = globalRow / kBlockSize;
    return (block / static_cast<size_t>(ranks)) * kBlockSize +
           globalRow % kBlockSize;
}

size_t globalRowFromLocal(size_t localRow, int rank, int ranks) {
    const size_t localBlock = localRow / kBlockSize;
    const size_t globalBlock =
        localBlock * static_cast<size_t>(ranks) + static_cast<size_t>(rank);
    return globalBlock * kBlockSize + localRow % kBlockSize;
}

// Number of rows owned by rank in the half-open interval [0, limit).
size_t ownedRowsBefore(size_t limit, int rank, int ranks) {
    const size_t fullBlocks = limit / kBlockSize;
    const size_t partial = limit % kBlockSize;
    const size_t rankValue = static_cast<size_t>(rank);
    const size_t rankCount = static_cast<size_t>(ranks);
    size_t blocks = fullBlocks / rankCount;
    if (rankValue < fullBlocks % rankCount) {
        ++blocks;
    }
    size_t rows = blocks * kBlockSize;
    if (partial != 0 && fullBlocks % rankCount == rankValue) {
        rows += partial;
    }
    return rows;
}

__global__ void packDiagonalKernel(const double* matrix, size_t leadingDim,
                                   size_t localStart, size_t globalStart,
                                   double* diagonal, int blockSize) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < blockSize && column < blockSize) {
        // cuSOLVER is column-major; the distributed matrix is row-major.
        diagonal[row + static_cast<size_t>(column) * blockSize] =
            matrix[(localStart + row) * leadingDim + globalStart + column];
    }
}

__global__ void unpackDiagonalKernel(const double* diagonal, double* matrix,
                                     size_t leadingDim, size_t localStart,
                                     size_t globalStart, int blockSize) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < blockSize && column <= row) {
        matrix[(localStart + row) * leadingDim + globalStart + column] =
            diagonal[row + static_cast<size_t>(column) * blockSize];
    }
}

__global__ void reorderPanelKernel(const double* rankPacked,
                                   const int* packedRowForGlobal,
                                   double* globalOrdered, int rows,
                                   int blockSize) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && column < blockSize) {
        globalOrdered[static_cast<size_t>(row) * blockSize + column] =
            rankPacked[static_cast<size_t>(packedRowForGlobal[row]) *
                           blockSize +
                       column];
    }
}

__global__ void zeroUpperKernel(double* matrix, size_t n, size_t localRows,
                                int rank, int ranks) {
    const size_t column =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localRow =
        static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (localRow < localRows && column < n) {
        const size_t globalRow =
            (localRow / kBlockSize * static_cast<size_t>(ranks) +
             static_cast<size_t>(rank)) *
                kBlockSize +
            localRow % kBlockSize;
        if (column > globalRow) {
            matrix[localRow * n + column] = 0.0;
        }
    }
}

struct GpuResources {
    cudaStream_t stream{};
    cublasHandle_t blas{};
    cusolverDnHandle_t solver{};
    double* matrix = nullptr;
    double* diagonal = nullptr;
    double* workspace = nullptr;
    double* packedPanel = nullptr;
    double* orderedPanel = nullptr;
    int* deviceInfo = nullptr;
    int* panelOrder = nullptr;
    double* hostSend = nullptr;
    double* hostPacked = nullptr;
    double* hostDiagonal = nullptr;
    int workspaceSize = 0;

    GpuResources(size_t matrixElements, size_t panelElements,
                 size_t sendElements, size_t orderElements) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&blas));
        CUSOLVER_CHECK(cusolverDnCreate(&solver));
        CUBLAS_CHECK(cublasSetStream(blas, stream));
        CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matrix),
                              std::max<size_t>(matrixElements, 1) *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&diagonal),
                              static_cast<size_t>(kBlockSize) * kBlockSize *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&packedPanel),
                              std::max<size_t>(panelElements, 1) *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&orderedPanel),
                              std::max<size_t>(panelElements, 1) *
                                  sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInfo),
                              sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&panelOrder),
                              std::max<size_t>(orderElements, 1) *
                                  sizeof(int)));

        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
            solver, CUBLAS_FILL_MODE_LOWER, kBlockSize, diagonal,
            kBlockSize, &workspaceSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace),
                              std::max(workspaceSize, 1) * sizeof(double)));

        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostSend),
                                 std::max<size_t>(sendElements, 1) *
                                     sizeof(double),
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostPacked),
                                 std::max<size_t>(panelElements, 1) *
                                     sizeof(double),
                                 cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostDiagonal),
                                 static_cast<size_t>(kBlockSize) *
                                     kBlockSize * sizeof(double),
                                 cudaHostAllocPortable));

        // CUDA libraries lazily load kernels and choose algorithms on their
        // first calls.  Warm every operation used by the timed factorization
        // so the measurement reflects decomposition work, not initialization.
        std::fill_n(hostDiagonal,
                    static_cast<size_t>(kBlockSize) * kBlockSize, 0.0);
        for (int i = 0; i < kBlockSize; ++i) {
            hostDiagonal[i + static_cast<size_t>(i) * kBlockSize] = 1.0;
        }
        CUDA_CHECK(cudaMemsetAsync(matrix, 0, sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(packedPanel, 0,
                                   static_cast<size_t>(kBlockSize) *
                                       kBlockSize * sizeof(double),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(orderedPanel, 0,
                                   static_cast<size_t>(kBlockSize) *
                                       kBlockSize * sizeof(double),
                                   stream));
        CUDA_CHECK(cudaMemsetAsync(panelOrder, 0, sizeof(int), stream));
        CUDA_CHECK(cudaMemcpyAsync(
            diagonal, hostDiagonal,
            static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double),
            cudaMemcpyHostToDevice, stream));
        CUSOLVER_CHECK(cusolverDnDpotrf(
            solver, CUBLAS_FILL_MODE_LOWER, kBlockSize, diagonal, kBlockSize,
            workspace, workspaceSize, deviceInfo));
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDtrsm(
            blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, kBlockSize, 1, &one, diagonal, kBlockSize,
            packedPanel, kBlockSize));
        CUBLAS_CHECK(cublasDgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, kBlockSize, kBlockSize,
            kBlockSize, &one, packedPanel, kBlockSize, packedPanel,
            kBlockSize, &zero, orderedPanel, kBlockSize));
        packDiagonalKernel<<<1, 1, 0, stream>>>(matrix, 1, 0, 0, diagonal, 1);
        unpackDiagonalKernel<<<1, 1, 0, stream>>>(diagonal, matrix, 1, 0, 0,
                                                  1);
        reorderPanelKernel<<<1, 1, 0, stream>>>(packedPanel, panelOrder,
                                                orderedPanel, 1, 1);
        zeroUpperKernel<<<1, 1, 0, stream>>>(matrix, 1, 1, 0, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ~GpuResources() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        if (hostDiagonal != nullptr) cudaFreeHost(hostDiagonal);
        if (hostPacked != nullptr) cudaFreeHost(hostPacked);
        if (hostSend != nullptr) cudaFreeHost(hostSend);
        if (panelOrder != nullptr) cudaFree(panelOrder);
        if (deviceInfo != nullptr) cudaFree(deviceInfo);
        if (orderedPanel != nullptr) cudaFree(orderedPanel);
        if (packedPanel != nullptr) cudaFree(packedPanel);
        if (workspace != nullptr) cudaFree(workspace);
        if (diagonal != nullptr) cudaFree(diagonal);
        if (matrix != nullptr) cudaFree(matrix);
        if (solver != nullptr) cusolverDnDestroy(solver);
        if (blas != nullptr) cublasDestroy(blas);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }

    GpuResources(const GpuResources&) = delete;
    GpuResources& operator=(const GpuResources&) = delete;
};

// The random stream and the order of each dot product match the original
// benchmark.  MPI distributes rows and OpenMP computes the local row set.
void generateLocalPositiveDefiniteMatrix(std::vector<double>& localMatrix,
                                         size_t n, size_t localRows, int rank,
                                         int ranks) {
    const size_t fullElements = checkedProduct(n, n, "matrix allocation");
    std::vector<double> randomMatrix(fullElements);
    unsigned int seed = 42;
    for (size_t i = 0; i < fullElements; ++i) {
        randomMatrix[i] =
            (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow =
            globalRowFromLocal(localRow, rank, ranks);
        const double* left = randomMatrix.data() + globalRow * n;
        double* output = localMatrix.data() + localRow * n;
        for (size_t column = 0; column < n; ++column) {
            const double* right = randomMatrix.data() + column * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += left[k] * right[k];
            }
            output[column] = sum;
        }
        output[globalRow] += static_cast<double>(n);
    }
}

bool choleskyDecomposition(GpuResources& gpu, size_t n, size_t localRows,
                           int rank, int ranks) {
    std::vector<int> receiveCounts(ranks);
    std::vector<int> receiveDisplacements(ranks);
    std::vector<size_t> firstLocalRow(ranks);
    std::vector<int> hostOrder(n);
    const dim3 tile(16, 16);

    for (size_t k = 0; k < n; k += kBlockSize) {
        const int blockSize =
            static_cast<int>(std::min<size_t>(kBlockSize, n - k));
        const size_t belowPanel = k + static_cast<size_t>(blockSize);
        const int diagonalOwner = ownerOfRow(k, ranks);
        int factorInfo = 0;

        if (rank == diagonalOwner) {
            const size_t localStart = localIndexOfRow(k, ranks);
            const dim3 grid((blockSize + tile.x - 1) / tile.x,
                            (blockSize + tile.y - 1) / tile.y);
            packDiagonalKernel<<<grid, tile, 0, gpu.stream>>>(
                gpu.matrix, n, localStart, k, gpu.diagonal, blockSize);
            CUDA_CHECK(cudaGetLastError());
            CUSOLVER_CHECK(cusolverDnDpotrf(
                gpu.solver, CUBLAS_FILL_MODE_LOWER, blockSize, gpu.diagonal,
                blockSize, gpu.workspace, gpu.workspaceSize, gpu.deviceInfo));
            CUDA_CHECK(cudaMemcpyAsync(&factorInfo, gpu.deviceInfo,
                                       sizeof(int), cudaMemcpyDeviceToHost,
                                       gpu.stream));
            CUDA_CHECK(cudaMemcpyAsync(
                gpu.hostDiagonal, gpu.diagonal,
                static_cast<size_t>(blockSize) * blockSize * sizeof(double),
                cudaMemcpyDeviceToHost, gpu.stream));
            unpackDiagonalKernel<<<grid, tile, 0, gpu.stream>>>(
                gpu.diagonal, gpu.matrix, n, localStart, k, blockSize);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        }

        MPI_CHECK(MPI_Bcast(&factorInfo, 1, MPI_INT, diagonalOwner,
                            MPI_COMM_WORLD));
        if (factorInfo != 0) {
            if (rank == diagonalOwner) {
                if (factorInfo > 0) {
                    std::fprintf(stderr,
                                 "Matrix is not positive definite at "
                                 "diagonal element %zu\n",
                                 k + static_cast<size_t>(factorInfo - 1));
                } else {
                    std::fprintf(stderr,
                                 "cuSOLVER rejected POTRF argument %d\n",
                                 -factorInfo);
                }
            }
            return false;
        }

        MPI_CHECK(MPI_Bcast(gpu.hostDiagonal, blockSize * blockSize,
                            MPI_DOUBLE, diagonalOwner, MPI_COMM_WORLD));
        if (rank != diagonalOwner) {
            CUDA_CHECK(cudaMemcpyAsync(
                gpu.diagonal, gpu.hostDiagonal,
                static_cast<size_t>(blockSize) * blockSize * sizeof(double),
                cudaMemcpyHostToDevice, gpu.stream));
        }

        if (belowPanel == n) {
            continue;
        }

        const size_t localStart = ownedRowsBefore(belowPanel, rank, ranks);
        const size_t activeLocalRows = localRows - localStart;
        if (activeLocalRows > 0) {
            const double one = 1.0;
            CUBLAS_CHECK(cublasDtrsm(
                gpu.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, blockSize,
                static_cast<int>(activeLocalRows), &one, gpu.diagonal,
                blockSize, gpu.matrix + localStart * n + k,
                static_cast<int>(n)));
        }

        int displacementRows = 0;
        for (int process = 0; process < ranks; ++process) {
            firstLocalRow[process] =
                ownedRowsBefore(belowPanel, process, ranks);
            const size_t rows = ownedRowsBefore(n, process, ranks) -
                                firstLocalRow[process];
            if (rows > static_cast<size_t>(INT_MAX / blockSize)) {
                abortAll("MPI panel exchange", "message exceeds MPI limit",
                         __LINE__);
            }
            receiveCounts[process] = static_cast<int>(rows) * blockSize;
            receiveDisplacements[process] = displacementRows * blockSize;
            displacementRows += static_cast<int>(rows);
        }

        for (size_t globalRow = belowPanel; globalRow < n; ++globalRow) {
            const int process = ownerOfRow(globalRow, ranks);
            const size_t packedLocal =
                localIndexOfRow(globalRow, ranks) - firstLocalRow[process];
            hostOrder[globalRow - belowPanel] =
                receiveDisplacements[process] / blockSize +
                static_cast<int>(packedLocal);
        }

        if (activeLocalRows > 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                gpu.hostSend, static_cast<size_t>(blockSize) * sizeof(double),
                gpu.matrix + localStart * n + k, n * sizeof(double),
                static_cast<size_t>(blockSize) * sizeof(double),
                activeLocalRows, cudaMemcpyDeviceToHost, gpu.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream));

        MPI_CHECK(MPI_Allgatherv(
            gpu.hostSend, static_cast<int>(activeLocalRows) * blockSize,
            MPI_DOUBLE, gpu.hostPacked, receiveCounts.data(),
            receiveDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD));

        const size_t trailingRows = n - belowPanel;
        CUDA_CHECK(cudaMemcpyAsync(
            gpu.packedPanel, gpu.hostPacked,
            trailingRows * static_cast<size_t>(blockSize) * sizeof(double),
            cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu.panelOrder, hostOrder.data(),
                                   trailingRows * sizeof(int),
                                   cudaMemcpyHostToDevice, gpu.stream));
        const dim3 reorderGrid((blockSize + tile.x - 1) / tile.x,
                               (trailingRows + tile.y - 1) / tile.y);
        reorderPanelKernel<<<reorderGrid, tile, 0, gpu.stream>>>(
            gpu.packedPanel, gpu.panelOrder, gpu.orderedPanel,
            static_cast<int>(trailingRows), blockSize);
        CUDA_CHECK(cudaGetLastError());

        if (activeLocalRows > 0) {
            const double minusOne = -1.0;
            const double one = 1.0;
            const double* localPanel =
                gpu.packedPanel +
                static_cast<size_t>(receiveDisplacements[rank]);

            // Update only the lower block triangle.  Each call covers one
            // block-cyclic local row tile and all preceding trailing columns;
            // the small diagonal tile is updated in full so the next POTRF
            // can consume a symmetric tile.  This avoids nearly half of the
            // GEMM work and memory traffic of a full trailing-square update.
            size_t tileOffset = 0;
            while (tileOffset < activeLocalRows) {
                const size_t localTile = localStart + tileOffset;
                const size_t globalTile =
                    globalRowFromLocal(localTile, rank, ranks);
                const size_t tileRows =
                    std::min<size_t>(kBlockSize, n - globalTile);
                const size_t updateColumns =
                    globalTile + tileRows - belowPanel;

                // Row-major C -= P_local * P_global^T is column-major
                // C^T -= P_global * P_local^T to cuBLAS.
                CUBLAS_CHECK(cublasDgemm(
                    gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(updateColumns),
                    static_cast<int>(tileRows), blockSize, &minusOne,
                    gpu.orderedPanel, blockSize,
                    localPanel + tileOffset * blockSize, blockSize, &one,
                    gpu.matrix + localTile * n + belowPanel,
                    static_cast<int>(n)));
                tileOffset += tileRows;
            }
        }
    }

    if (localRows > 0) {
        const dim3 grid((n + tile.x - 1) / tile.x,
                        (localRows + tile.y - 1) / tile.y);
        zeroUpperKernel<<<grid, tile, 0, gpu.stream>>>(gpu.matrix, n,
                                                       localRows, rank, ranks);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    return true;
}

void sendLarge(const double* data, size_t count, int destination, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(
            std::min<size_t>(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Send(data, chunk, MPI_DOUBLE, destination, tag,
                           MPI_COMM_WORLD));
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveLarge(double* data, size_t count, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(
            std::min<size_t>(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Recv(data, chunk, MPI_DOUBLE, source, tag,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

// Reassemble block-cyclic rows only when output or validation was requested.
std::vector<double> gatherMatrix(const std::vector<double>& localMatrix,
                                 size_t n, size_t localRows, int rank,
                                 int ranks, int tag) {
    if (rank != 0) {
        sendLarge(localMatrix.data(), checkedProduct(localRows, n, "gather"),
                  0, tag);
        return {};
    }

    std::vector<double> result(checkedProduct(n, n, "gather result"));
    for (int source = 0; source < ranks; ++source) {
        const size_t sourceRows = ownedRowsBefore(n, source, ranks);
        std::vector<double> received;
        const double* sourceData = nullptr;
        if (source == 0) {
            sourceData = localMatrix.data();
        } else {
            received.resize(checkedProduct(sourceRows, n, "gather buffer"));
            receiveLarge(received.data(), received.size(), source, tag);
            sourceData = received.data();
        }

#pragma omp parallel for schedule(static)
        for (size_t localRow = 0; localRow < sourceRows; ++localRow) {
            const size_t globalRow =
                globalRowFromLocal(localRow, source, ranks);
            std::memcpy(result.data() + globalRow * n,
                        sourceData + localRow * n, n * sizeof(double));
        }
    }
    return result;
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relativeError)
    for (size_t row = 0; row < n; ++row) {
        for (size_t column = 0; column < n; ++column) {
            double sum = 0.0;
            const size_t terms = std::min(row, column) + 1;
            for (size_t k = 0; k < terms; ++k) {
                sum += lower[row * n + k] * lower[column * n + k];
            }
            const double error = std::fabs(sum - original[row * n + column]);
            maxError = std::max(maxError, error);
            relativeError =
                std::max(relativeError,
                         error / (std::fabs(original[row * n + column]) +
                                  1.0e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1.0e-6) {
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        abortAll("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable",
                 __LINE__);
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
                argumentsValid = false;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (!argumentsValid || showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return argumentsValid ? 0 : 1;
    }

    // Select GPUs by node-local rank.  More than one rank per GPU remains
    // functional, though one MPI rank per accelerator is recommended.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localRanks = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localRanks));

    // Avoid oversubscribing a node when the launcher has not provided an
    // explicit OpenMP policy.  An explicit OMP_NUM_THREADS always wins.
    omp_set_dynamic(0);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(
            std::max(1, omp_get_num_procs() / std::max(localRanks, 1)));
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortAll("cudaGetDeviceCount", "no CUDA accelerator is available",
                 __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    const size_t localRows = ownedRowsBefore(n, rank, ranks);
    const size_t localElements = checkedProduct(localRows, n, "local matrix");
    std::vector<double> localMatrix(localElements);

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), %d OpenMP "
                    "thread(s)/rank, CUDA tile %d\n",
                    ranks, omp_get_max_threads(), kBlockSize);
        std::printf("Generating positive definite matrix...\n");
    }

    generateLocalPositiveDefiniteMatrix(localMatrix, n, localRows, rank,
                                        ranks);
    std::vector<double> localOriginal;
    if (validate) {
        localOriginal = localMatrix;
    }

    const size_t panelElements = std::max(
        checkedProduct(n, kBlockSize, "panel"),
        static_cast<size_t>(kBlockSize) * kBlockSize);
    const size_t sendElements =
        checkedProduct(localRows, kBlockSize, "panel send buffer");
    GpuResources gpu(localElements, panelElements, sendElements, n);
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.matrix, localMatrix.data(),
                                   localElements * sizeof(double),
                                   cudaMemcpyHostToDevice, gpu.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    std::vector<double>().swap(localMatrix);

    if (rank == 0) {
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success =
        choleskyDecomposition(gpu, n, localRows, rank, ranks);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(std::llround(seconds * 1000.0));
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / std::max(seconds, 1.0e-12) / 1.0e9;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> lower;
    if (validate || printResults) {
        localMatrix.resize(localElements);
        if (localElements != 0) {
            CUDA_CHECK(cudaMemcpy(localMatrix.data(), gpu.matrix,
                                  localElements * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        lower = gatherMatrix(localMatrix, n, localRows, rank, ranks, 1701);
    }

    if (printResults && rank == 0) {
        print_results(lower, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        std::vector<double> original =
            gatherMatrix(localOriginal, n, localRows, rank, ranks, 1702);
        if (rank == 0) {
            std::printf("Validating result...\n");
            valid = validateCholesky(lower, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        int validInteger = valid ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&validInteger, 1, MPI_INT, 0, MPI_COMM_WORLD));
        valid = validInteger != 0;
    }

    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
