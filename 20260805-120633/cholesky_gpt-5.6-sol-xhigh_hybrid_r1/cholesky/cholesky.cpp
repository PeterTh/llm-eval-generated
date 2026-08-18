#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

[[noreturn]] void abortProgram(const char* subsystem, const char* expression,
                               const char* detail, const char* file, int line) {
    std::fprintf(stderr, "Rank %d: %s failure at %s:%d while evaluating %s: %s\n",
                 worldRank, subsystem, file, line, expression, detail);
    std::fflush(stderr);

    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalized(&finalized);
    }
    if (initialized && !finalized) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        abortProgram("CUDA", expression, cudaGetErrorString(status), file, line);
    }
}

void checkCublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char detail[64];
        std::snprintf(detail, sizeof(detail), "cuBLAS status %d", static_cast<int>(status));
        abortProgram("cuBLAS", expression, detail, file, line);
    }
}

void checkCusolver(cusolverStatus_t status, const char* expression, const char* file,
                   int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char detail[64];
        std::snprintf(detail, sizeof(detail), "cuSOLVER status %d", static_cast<int>(status));
        abortProgram("cuSOLVER", expression, detail, file, line);
    }
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char detail[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, detail, &length);
        detail[std::min(length, MPI_MAX_ERROR_STRING - 1)] = '\0';
        abortProgram("MPI", expression, detail, file, line);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkCublas((call), #call, __FILE__, __LINE__)
#define CUSOLVER_CHECK(call) checkCusolver((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(size_t count) {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFree(data_));
        }
        count_ = std::max<size_t>(count, 1);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { allocate(count); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    void allocate(size_t count) {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFreeHost(data_));
        }
        count_ = std::max<size_t>(count, 1);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

  private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

class CudaStream {
  public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() { cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }

  private:
    cudaStream_t stream_{};
};

class CublasHandle {
  public:
    explicit CublasHandle(cudaStream_t stream) {
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream));
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    ~CublasHandle() { cublasDestroy(handle_); }
    operator cublasHandle_t() const { return handle_; }

  private:
    cublasHandle_t handle_{};
};

class CusolverHandle {
  public:
    explicit CusolverHandle(cudaStream_t stream) {
        CUSOLVER_CHECK(cusolverDnCreate(&handle_));
        CUSOLVER_CHECK(cusolverDnSetStream(handle_, stream));
    }
    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;
    ~CusolverHandle() { cusolverDnDestroy(handle_); }
    operator cusolverDnHandle_t() const { return handle_; }

  private:
    cusolverDnHandle_t handle_{};
};

struct Block {
    int index;
    int globalRow;
    int localRow;
    int rows;
};

class BlockDistribution {
  public:
    BlockDistribution(int n, int blockSize, int rank, int ranks)
        : n_(n), blockSize_(blockSize), rank_(rank), ranks_(ranks),
          blockCount_((n + blockSize - 1) / blockSize) {
        int localRow = 0;
        for (int block = rank_; block < blockCount_; block += ranks_) {
            const int globalRow = block * blockSize_;
            const int rows = std::min(blockSize_, n_ - globalRow);
            localBlocks_.push_back({block, globalRow, localRow, rows});
            localRow += rows;
        }
        localRows_ = localRow;
    }

    int n() const { return n_; }
    int blockSize() const { return blockSize_; }
    int blockCount() const { return blockCount_; }
    int localRows() const { return localRows_; }
    int ranks() const { return ranks_; }
    const std::vector<Block>& localBlocks() const { return localBlocks_; }

    int blockRows(int block) const {
        return std::min(blockSize_, n_ - block * blockSize_);
    }

    int rowsOnRank(int rank) const {
        int rows = 0;
        for (int block = rank; block < blockCount_; block += ranks_) {
            rows += blockRows(block);
        }
        return rows;
    }

    const Block* localBlock(int blockIndex) const {
        for (const Block& block : localBlocks_) {
            if (block.index == blockIndex) {
                return &block;
            }
        }
        return nullptr;
    }

    int firstLocalRowAfter(int blockIndex) const {
        for (const Block& block : localBlocks_) {
            if (block.index > blockIndex) {
                return block.localRow;
            }
        }
        return localRows_;
    }

  private:
    int n_;
    int blockSize_;
    int rank_;
    int ranks_;
    int blockCount_;
    int localRows_ = 0;
    std::vector<Block> localBlocks_;
};

__global__ void addDiagonalBlock(double* matrix, int leadingDimension, int localRow,
                                 int globalRow, int rows, double diagonalValue) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        matrix[static_cast<size_t>(localRow + row) * leadingDimension + globalRow + row] +=
            diagonalValue;
    }
}

__global__ void zeroUpperBlock(double* matrix, int leadingDimension, int localRow,
                               int globalRow, int rows) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && column < leadingDimension && column > globalRow + row) {
        matrix[static_cast<size_t>(localRow + row) * leadingDimension + column] = 0.0;
    }
}

int checkedMpiCount(size_t count, const char* what) {
    if (count > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "%s exceeds the MPI-3 count limit\n", what);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
    return static_cast<int>(count);
}

void bcastDoubles(double* data, size_t count, int root, MPI_Comm communicator) {
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(
            std::min<size_t>(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data + offset, chunk, MPI_DOUBLE, root, communicator));
        offset += static_cast<size_t>(chunk);
    }
}

int selectGpuForLocalRank() {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localCommunicator));

    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortProgram("CUDA", "cudaGetDeviceCount", "no CUDA-capable device found", __FILE__,
                     __LINE__);
    }

    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    return device;
}

void generatePositiveDefiniteMatrix(DeviceBuffer<double>& localMatrix,
                                    const BlockDistribution& distribution,
                                    cublasHandle_t cublas, cudaStream_t stream) {
    const int n = distribution.n();
    const size_t matrixElements = static_cast<size_t>(n) * n;
    std::vector<double> hostB(matrixElements);

    if (worldRank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < matrixElements; ++i) {
            hostB[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }
    bcastDoubles(hostB.data(), matrixElements, 0, MPI_COMM_WORLD);

    DeviceBuffer<double> fullB(matrixElements);
    DeviceBuffer<double> localB(static_cast<size_t>(distribution.localRows()) * n);
    CUDA_CHECK(cudaMemcpyAsync(fullB.data(), hostB.data(), matrixElements * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    for (const Block& block : distribution.localBlocks()) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            localB.data() + static_cast<size_t>(block.localRow) * n,
            static_cast<size_t>(n) * sizeof(double),
            fullB.data() + static_cast<size_t>(block.globalRow) * n,
            static_cast<size_t>(n) * sizeof(double), static_cast<size_t>(n) * sizeof(double),
            block.rows, cudaMemcpyDeviceToDevice, stream));
    }

    // Row-major C = B_local * B^T is evaluated as the column-major operation
    // C^T = B * B_local^T, so the local block rows stay contiguous and distributed.
    if (distribution.localRows() > 0) {
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n,
                                 distribution.localRows(), n, &one, fullB.data(), n,
                                 localB.data(), n, &zero, localMatrix.data(), n));

        constexpr int threads = 256;
        for (const Block& block : distribution.localBlocks()) {
            const int grid = (block.rows + threads - 1) / threads;
            addDiagonalBlock<<<grid, threads, 0, stream>>>(
                localMatrix.data(), n, block.localRow, block.globalRow, block.rows,
                static_cast<double>(n));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<double> gatherMatrix(const DeviceBuffer<double>& localMatrix,
                                 const BlockDistribution& distribution,
                                 cudaStream_t stream) {
    const int n = distribution.n();
    const int ranks = distribution.ranks();
    const size_t localElements = static_cast<size_t>(distribution.localRows()) * n;
    PinnedBuffer<double> localHost(localElements);
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(localHost.data(), localMatrix.data(),
                                   localElements * sizeof(double), cudaMemcpyDeviceToHost,
                                   stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    size_t total = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        displacements[rank] = checkedMpiCount(total, "matrix gather displacement");
        counts[rank] = checkedMpiCount(
            static_cast<size_t>(distribution.rowsOnRank(rank)) * n, "matrix gather count");
        total += static_cast<size_t>(counts[rank]);
    }

    std::vector<double> packed;
    if (worldRank == 0) {
        packed.resize(total);
    }
    MPI_CHECK(MPI_Gatherv(localHost.data(), checkedMpiCount(localElements, "local matrix"),
                          MPI_DOUBLE, worldRank == 0 ? packed.data() : nullptr, counts.data(),
                          displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (worldRank != 0) {
        return {};
    }

    std::vector<double> matrix(static_cast<size_t>(n) * n);
#pragma omp parallel for schedule(static)
    for (int blockIndex = 0; blockIndex < distribution.blockCount(); ++blockIndex) {
        const int owner = blockIndex % ranks;
        const int globalRow = blockIndex * distribution.blockSize();
        const int rows = distribution.blockRows(blockIndex);
        const int localRow = (blockIndex / ranks) * distribution.blockSize();
        std::memcpy(matrix.data() + static_cast<size_t>(globalRow) * n,
                    packed.data() + displacements[owner] + static_cast<size_t>(localRow) * n,
                    static_cast<size_t>(rows) * n * sizeof(double));
    }
    return matrix;
}

bool choleskyDecomposition(DeviceBuffer<double>& localMatrix,
                           const BlockDistribution& distribution, cublasHandle_t cublas,
                           cusolverDnHandle_t cusolver, cudaStream_t stream) {
    const int n = distribution.n();
    const int blockSize = distribution.blockSize();
    const int ranks = distribution.ranks();

    DeviceBuffer<double> diagonal(static_cast<size_t>(blockSize) * blockSize);
    DeviceBuffer<double> panel(static_cast<size_t>(n) * blockSize);
    DeviceBuffer<int> deviceInfo(1);
    PinnedBuffer<double> hostDiagonal(static_cast<size_t>(blockSize) * blockSize);
    PinnedBuffer<double> hostPanelSend(
        static_cast<size_t>(distribution.localRows()) * blockSize);
    PinnedBuffer<double> hostPanelGather(static_cast<size_t>(n) * blockSize);
    PinnedBuffer<double> hostPanelOrdered(static_cast<size_t>(n) * blockSize);

    int workspaceElements = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER,
                                               blockSize, diagonal.data(), blockSize,
                                               &workspaceElements));
    DeviceBuffer<double> workspace(static_cast<size_t>(workspaceElements));

    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    std::vector<size_t> blockOffsets(distribution.blockCount());

    for (int panelBlock = 0; panelBlock < distribution.blockCount(); ++panelBlock) {
        const int panelColumn = panelBlock * blockSize;
        const int panelWidth = distribution.blockRows(panelBlock);
        const int owner = panelBlock % ranks;
        int factorInfo = 0;

        if (worldRank == owner) {
            const Block* diagonalBlock = distribution.localBlock(panelBlock);
            if (diagonalBlock == nullptr) {
                abortProgram("distribution", "localBlock(panelBlock)",
                             "diagonal block is not present on its owner", __FILE__, __LINE__);
            }
            double* diagonalInMatrix =
                localMatrix.data() + static_cast<size_t>(diagonalBlock->localRow) * n +
                panelColumn;
            CUSOLVER_CHECK(cusolverDnDpotrf(
                cusolver, CUBLAS_FILL_MODE_UPPER, panelWidth, diagonalInMatrix, n,
                workspace.data(), workspaceElements, deviceInfo.data()));
            CUDA_CHECK(cudaMemcpyAsync(&factorInfo, deviceInfo.data(), sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpy2DAsync(
                hostDiagonal.data(), static_cast<size_t>(panelWidth) * sizeof(double),
                diagonalInMatrix, static_cast<size_t>(n) * sizeof(double),
                static_cast<size_t>(panelWidth) * sizeof(double), panelWidth,
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        MPI_CHECK(MPI_Bcast(&factorInfo, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (factorInfo != 0) {
            if (worldRank == 0) {
                if (factorInfo > 0) {
                    std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                                panelColumn + factorInfo - 1);
                } else {
                    std::printf("Error: cuSOLVER reported invalid argument %d\n", -factorInfo);
                }
            }
            return false;
        }

        MPI_CHECK(MPI_Bcast(hostDiagonal.data(), panelWidth * panelWidth, MPI_DOUBLE, owner,
                            MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy2DAsync(
            diagonal.data(), static_cast<size_t>(panelWidth) * sizeof(double),
            hostDiagonal.data(), static_cast<size_t>(panelWidth) * sizeof(double),
            static_cast<size_t>(panelWidth) * sizeof(double), panelWidth,
            cudaMemcpyHostToDevice, stream));

        const int firstTrailingLocalRow = distribution.firstLocalRowAfter(panelBlock);
        const int trailingLocalRows = distribution.localRows() - firstTrailingLocalRow;
        if (trailingLocalRows > 0) {
            const double one = 1.0;
            // Row-major X = A_ik * inv(L_kk^T) becomes
            // X^T = inv(L_kk) * A_ik^T in the column-major cuBLAS view.
            CUBLAS_CHECK(cublasDtrsm(
                cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, panelWidth, trailingLocalRows, &one, diagonal.data(),
                panelWidth,
                localMatrix.data() + static_cast<size_t>(firstTrailingLocalRow) * n +
                    panelColumn,
                n));

            CUDA_CHECK(cudaMemcpy2DAsync(
                hostPanelSend.data(), static_cast<size_t>(panelWidth) * sizeof(double),
                localMatrix.data() + static_cast<size_t>(firstTrailingLocalRow) * n +
                    panelColumn,
                static_cast<size_t>(n) * sizeof(double),
                static_cast<size_t>(panelWidth) * sizeof(double), trailingLocalRows,
                cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::fill(counts.begin(), counts.end(), 0);
        for (int blockIndex = panelBlock + 1;
             blockIndex < distribution.blockCount(); ++blockIndex) {
            const int rank = blockIndex % ranks;
            counts[rank] += distribution.blockRows(blockIndex) * panelWidth;
        }
        int gatheredElements = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            displacements[rank] = gatheredElements;
            gatheredElements += counts[rank];
        }

        MPI_CHECK(MPI_Allgatherv(hostPanelSend.data(), trailingLocalRows * panelWidth,
                                 MPI_DOUBLE, hostPanelGather.data(), counts.data(),
                                 displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD));

        std::vector<size_t> cursor(displacements.begin(), displacements.end());
        for (int blockIndex = panelBlock + 1;
             blockIndex < distribution.blockCount(); ++blockIndex) {
            const int rank = blockIndex % ranks;
            blockOffsets[blockIndex] = cursor[rank];
            cursor[rank] +=
                static_cast<size_t>(distribution.blockRows(blockIndex)) * panelWidth;
        }

#pragma omp parallel for schedule(static)
        for (int blockIndex = panelBlock + 1;
             blockIndex < distribution.blockCount(); ++blockIndex) {
            const int globalRow = blockIndex * blockSize;
            const int rows = distribution.blockRows(blockIndex);
            std::memcpy(hostPanelOrdered.data() + static_cast<size_t>(globalRow) * panelWidth,
                        hostPanelGather.data() + blockOffsets[blockIndex],
                        static_cast<size_t>(rows) * panelWidth * sizeof(double));
        }

        if (gatheredElements > 0) {
            const int firstTrailingGlobalRow = panelColumn + panelWidth;
            const int trailingGlobalRows = n - firstTrailingGlobalRow;
            CUDA_CHECK(cudaMemcpyAsync(
                panel.data() + static_cast<size_t>(firstTrailingGlobalRow) * panelWidth,
                hostPanelOrdered.data() +
                    static_cast<size_t>(firstTrailingGlobalRow) * panelWidth,
                static_cast<size_t>(trailingGlobalRows) * panelWidth * sizeof(double),
                cudaMemcpyHostToDevice, stream));

            const double minusOne = -1.0;
            const double one = 1.0;
            for (const Block& block : distribution.localBlocks()) {
                if (block.index <= panelBlock) {
                    continue;
                }
                const int columns = block.globalRow + block.rows - firstTrailingGlobalRow;
                // Update only block-row entries on and below the block diagonal. This
                // preserves the O(n^3/3) Cholesky work while using large GPU GEMMs.
                CUBLAS_CHECK(cublasDgemm(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, columns, block.rows, panelWidth,
                    &minusOne,
                    panel.data() +
                        static_cast<size_t>(firstTrailingGlobalRow) * panelWidth,
                    panelWidth,
                    localMatrix.data() + static_cast<size_t>(block.localRow) * n +
                        panelColumn,
                    n, &one,
                    localMatrix.data() + static_cast<size_t>(block.localRow) * n +
                        firstTrailingGlobalRow,
                    n));
            }
        }
    }

    const dim3 threads(32, 8);
    for (const Block& block : distribution.localBlocks()) {
        const dim3 grid((n + threads.x - 1) / threads.x,
                        (block.rows + threads.y - 1) / threads.y);
        zeroUpperBlock<<<grid, threads, 0, stream>>>(localMatrix.data(), n, block.localRow,
                                                     block.globalRow, block.rows);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;

#pragma omp parallel for collapse(2) schedule(static) reduction(max : maxError, relativeError)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        for (long long column = 0; column < static_cast<long long>(n); ++column) {
            double sum = 0.0;
            const size_t terms = std::min(static_cast<size_t>(row),
                                          static_cast<size_t>(column)) +
                                 1;
            for (size_t k = 0; k < terms; ++k) {
                sum += lower[static_cast<size_t>(row) * n + k] *
                       lower[static_cast<size_t>(column) * n + k];
            }
            const size_t index = static_cast<size_t>(row) * n + column;
            const double error = std::fabs(sum - original[index]);
            maxError = std::max(maxError, error);
            relativeError =
                std::max(relativeError, error / (std::fabs(original[index]) + 1e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
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

bool parseSize(const char* value, size_t& result) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(INT_MAX)) {
        return false;
    }
    result = static_cast<size_t>(parsed);
    return result <= std::numeric_limits<size_t>::max() / result;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    size_t matrixSize = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentsValid = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], matrixSize)) {
                argumentsValid = false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const int n = static_cast<int>(matrixSize);
    // Small panels keep enough blocks available for MPI at modest sizes, while wider
    // panels make the large trailing updates compute-bound on modern accelerators.
    const int blockSize = std::min(n, n <= 1024 ? 128 : (n <= 2048 ? 256 : 512));
    const int device = selectGpuForLocalRank();

    int openmpThreads = 1;
#pragma omp parallel
    {
#pragma omp master
        openmpThreads = omp_get_num_threads();
    }

    int resultCode = EXIT_SUCCESS;
    {
        CudaStream stream;
        CublasHandle cublas(stream);
        CusolverHandle cusolver(stream);
        BlockDistribution distribution(n, blockSize, worldRank, ranks);
        DeviceBuffer<double> localMatrix(
            static_cast<size_t>(distribution.localRows()) * n);

        if (worldRank == 0) {
            std::printf("Cholesky Decomposition Benchmark\n");
            std::printf("Matrix size: %zu x %zu\n", matrixSize, matrixSize);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("Hybrid configuration: %d MPI rank(s), %d OpenMP thread(s)/rank, "
                        "CUDA device %d on rank 0\n",
                        ranks, openmpThreads, device);
            std::printf("Block size: %d\n", blockSize);
            std::printf("Generating positive definite matrix...\n");
        }

        generatePositiveDefiniteMatrix(localMatrix, distribution, cublas, stream);

        std::vector<double> original;
        if (validate) {
            original = gatherMatrix(localMatrix, distribution, stream);
        }

        if (worldRank == 0) {
            std::printf("Computing Cholesky decomposition...\n");
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        const bool success =
            choleskyDecomposition(localMatrix, distribution, cublas, cusolver, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD));

        if (!success) {
            if (worldRank == 0) {
                std::printf("Cholesky decomposition failed\n");
            }
            resultCode = EXIT_FAILURE;
        } else {
            if (worldRank == 0) {
                const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
                const double operations = static_cast<double>(n) * n * n / 3.0;
                const double gflops = operations / std::max(elapsed, 1e-12) / 1e9;
                std::printf("Computation time: %lld ms\n", milliseconds);
                std::printf("Performance: %.3f GFLOPS\n", gflops);
            }

            std::vector<double> lower;
            if (printResults || validate) {
                lower = gatherMatrix(localMatrix, distribution, stream);
            }
            if (worldRank == 0 && printResults) {
                print_results(lower, "CholeskyL");
            }
            if (worldRank == 0 && validate) {
                std::printf("Validating result...\n");
                if (validateCholesky(lower, original, matrixSize)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    resultCode = EXIT_FAILURE;
                }
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return resultCode;
}
