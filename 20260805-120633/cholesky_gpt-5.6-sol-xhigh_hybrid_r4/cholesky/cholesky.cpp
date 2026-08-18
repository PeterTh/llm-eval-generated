#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

// The matrix is distributed by block rows in a cyclic fashion.  A rank owns
// block rows rank, rank + nranks, ... and retains only those rows on its GPU.
// This balances the shrinking trailing update while keeping each local block
// row contiguous for strided-batched GEMMs.
constexpr int kSmallBlockSize = 512;
constexpr int kLargeBlockSize = 1024;

static int selectBlockSize(size_t n, int ranks) {
    if (n < static_cast<size_t>(kSmallBlockSize)) {
        // cuSOLVER/cuBLAS are most efficient with aligned leading dimensions,
        // without reserving multi-megabyte panels for tiny test matrices.
        return std::max(32, static_cast<int>((n + 31) / 32) * 32);
    }
    // Large tiles improve GEMM efficiency and reduce collective latency.  Keep
    // at least two block rows per rank to preserve trailing-update balance.
    if (n >= static_cast<size_t>(2LL * ranks * kLargeBlockSize)) {
        return kLargeBlockSize;
    }
    return kSmallBlockSize;
}

[[noreturn]] static void fatalError(const char* kind, const char* expression,
                                    int code, const char* file, int line) {
    int initialized = 0;
    int rank = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s failure (%d) in %s at %s:%d\n", rank,
                 kind, code, expression, file, line);
    std::fflush(stderr);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, code == 0 ? 1 : code);
    }
    std::abort();
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t status_ = (call);                                    \
        if (status_ != cudaSuccess)                                            \
            fatalError("CUDA", #call, static_cast<int>(status_), __FILE__,   \
                       __LINE__);                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        const cublasStatus_t status_ = (call);                                 \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                  \
            fatalError("cuBLAS", #call, static_cast<int>(status_), __FILE__, \
                       __LINE__);                                              \
    } while (0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        const cusolverStatus_t status_ = (call);                               \
        if (status_ != CUSOLVER_STATUS_SUCCESS)                                \
            fatalError("cuSOLVER", #call, static_cast<int>(status_),         \
                       __FILE__, __LINE__);                                    \
    } while (0)

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        const int status_ = (call);                                            \
        if (status_ != MPI_SUCCESS)                                            \
            fatalError("MPI", #call, status_, __FILE__, __LINE__);           \
    } while (0)

static size_t checkedProduct(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        fatalError("size overflow", "matrix allocation", 1, __FILE__,
                   __LINE__);
    }
    return a * b;
}

struct Distribution {
    size_t n;
    int rank;
    int ranks;
    int blockSize;
    int blocks;
    int paddedN;
    int localBlocks;
    size_t rowStride;
    double* deviceA = nullptr;

    Distribution(size_t matrixSize, int mpiRank, int mpiRanks)
        : n(matrixSize), rank(mpiRank), ranks(mpiRanks),
          blockSize(selectBlockSize(matrixSize, mpiRanks)) {
        blocks = static_cast<int>((n + blockSize - 1) / blockSize);
        paddedN = blocks * blockSize;
        localBlocks = rank < blocks ? 1 + (blocks - 1 - rank) / ranks : 0;
        rowStride = checkedProduct(static_cast<size_t>(blockSize),
                                   static_cast<size_t>(paddedN));
    }

    int owner(int blockRow) const { return blockRow % ranks; }

    int blockExtent(int block) const {
        const size_t begin = static_cast<size_t>(block) * blockSize;
        return static_cast<int>(std::min(n - begin,
                                         static_cast<size_t>(blockSize)));
    }

    double* blockRow(int globalBlock) const {
        return deviceA + static_cast<size_t>(globalBlock / ranks) * rowStride;
    }

    double* tile(int blockRowIndex, int blockColumn) const {
        return blockRow(blockRowIndex) +
               static_cast<size_t>(blockColumn) * blockSize * blockSize;
    }
};

// glibc rand_r advances the LCG three times per returned value.  Jumping to
// the beginning of each OpenMP thread's contiguous range preserves the exact
// matrix produced by the original serial generator while making generation
// parallel and race-free.
static uint32_t advanceLcg(uint32_t state, uint64_t steps) {
    uint32_t accumulatedMultiplier = 1;
    uint32_t accumulatedIncrement = 0;
    uint32_t multiplier = 1103515245U;
    uint32_t increment = 12345U;

    while (steps != 0) {
        if ((steps & 1U) != 0) {
            accumulatedIncrement =
                multiplier * accumulatedIncrement + increment;
            accumulatedMultiplier = multiplier * accumulatedMultiplier;
        }
        increment = (multiplier + 1U) * increment;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

static uint32_t compatibleRandR(uint32_t& state) {
    uint32_t result;
    state = state * 1103515245U + 12345U;
    result = (state / 65536U) % 2048U;
    state = state * 1103515245U + 12345U;
    result = (result << 10U) ^ ((state / 65536U) % 1024U);
    state = state * 1103515245U + 12345U;
    result = (result << 10U) ^ ((state / 65536U) % 1024U);
    return result;
}

static void generateRandomRange(double* values, size_t globalOffset,
                                size_t entries) {
#pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        const size_t begin = entries / threads * thread +
                             std::min<size_t>(entries % threads, thread);
        const size_t end = entries / threads * (thread + 1) +
                           std::min<size_t>(entries % threads, thread + 1);
        uint32_t state = advanceLcg(42U, 3ULL * (globalOffset + begin));
        for (size_t i = begin; i < end; ++i) {
            values[i] = static_cast<double>(compatibleRandR(state)) /
                            2147483647.0 -
                        0.5;
        }
    }
}

__global__ static void addDiagonal(double* blockRow, int globalRow,
                                   int rowCount, int leadingDimension,
                                   double diagonalValue) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowCount) {
        const int column = globalRow + row;
        blockRow[static_cast<size_t>(column) * leadingDimension + row] +=
            diagonalValue;
    }
}

static void generatePositiveDefiniteMatrix(
    Distribution& distribution, cublasHandle_t cublas,
    cudaStream_t computeStream, cudaStream_t communicationStream,
    bool cudaAwareMpi) {
    // B uses the same cyclic block-row distribution as A.  Broadcasting one
    // B block at a time avoids the O(n^2) per-rank replication that would
    // otherwise cap the largest matrix well before GPU factor storage does.
    const size_t bRowStride = checkedProduct(
        static_cast<size_t>(distribution.blockSize), distribution.n);
    const size_t localBEntries = checkedProduct(
        static_cast<size_t>(distribution.localBlocks), bRowStride);
    std::vector<double> hostLocalB(localBEntries, 0.0);
    for (int local = 0; local < distribution.localBlocks; ++local) {
        const int block = distribution.rank + local * distribution.ranks;
        const size_t globalRow =
            static_cast<size_t>(block) * distribution.blockSize;
        const size_t entries = checkedProduct(
            static_cast<size_t>(distribution.blockExtent(block)),
            distribution.n);
        generateRandomRange(hostLocalB.data() +
                                static_cast<size_t>(local) * bRowStride,
                            globalRow * distribution.n, entries);
    }

    double* deviceLocalB = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceLocalB,
                          std::max<size_t>(1, localBEntries) * sizeof(double)));
    if (localBEntries != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceLocalB, hostLocalB.data(),
                                   localBEntries * sizeof(double),
                                   cudaMemcpyHostToDevice, computeStream));
    }
    std::vector<double>().swap(hostLocalB);

    const size_t panelEntries = checkedProduct(
        static_cast<size_t>(distribution.blockSize), distribution.n);
    double* devicePanels[2] = {nullptr, nullptr};
    double* hostPanels[2] = {nullptr, nullptr};
    cudaEvent_t bufferFree[2];
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaMalloc(&devicePanels[buffer],
                              panelEntries * sizeof(double)));
        CUDA_CHECK(cudaHostAlloc(&hostPanels[buffer],
                                 panelEntries * sizeof(double),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&bufferFree[buffer],
                                            cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(bufferFree[buffer], computeStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    const double one = 1.0;
    const double zero = 0.0;
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    auto postBBlock = [&](int block, int buffer) {
        const int root = distribution.owner(block);
        const size_t count = checkedProduct(
            static_cast<size_t>(distribution.blockExtent(block)),
            distribution.n);
        if (count > INT_MAX) {
            fatalError("MPI", "B panel broadcast exceeds INT_MAX", 1,
                       __FILE__, __LINE__);
        }
        CUDA_CHECK(cudaEventSynchronize(bufferFree[buffer]));
        if (cudaAwareMpi) {
            if (distribution.rank == root) {
                const double* source =
                    deviceLocalB +
                    static_cast<size_t>(block / distribution.ranks) *
                        bRowStride;
                CUDA_CHECK(cudaMemcpyAsync(
                    devicePanels[buffer], source, count * sizeof(double),
                    cudaMemcpyDeviceToDevice, communicationStream));
                CUDA_CHECK(cudaStreamSynchronize(communicationStream));
            }
            MPI_CHECK(MPI_Ibcast(devicePanels[buffer], static_cast<int>(count),
                                 MPI_DOUBLE, root, MPI_COMM_WORLD,
                                 &requests[buffer]));
        } else {
            if (distribution.rank == root) {
                const double* source =
                    deviceLocalB +
                    static_cast<size_t>(block / distribution.ranks) *
                        bRowStride;
                CUDA_CHECK(cudaMemcpyAsync(
                    hostPanels[buffer], source, count * sizeof(double),
                    cudaMemcpyDeviceToHost, communicationStream));
                CUDA_CHECK(cudaStreamSynchronize(communicationStream));
            }
            MPI_CHECK(MPI_Ibcast(hostPanels[buffer], static_cast<int>(count),
                                 MPI_DOUBLE, root, MPI_COMM_WORLD,
                                 &requests[buffer]));
        }
    };

    postBBlock(0, 0);
    for (int j = 0; j < distribution.blocks; ++j) {
        const int buffer = j & 1;
        MPI_CHECK(MPI_Wait(&requests[buffer], MPI_STATUS_IGNORE));
        const int panelRows = distribution.blockExtent(j);
        const size_t panelCount =
            static_cast<size_t>(panelRows) * distribution.n;
        if (!cudaAwareMpi) {
            CUDA_CHECK(cudaMemcpyAsync(devicePanels[buffer],
                                       hostPanels[buffer],
                                       panelCount * sizeof(double),
                                       cudaMemcpyHostToDevice, computeStream));
            // Once this event completes, MPI may safely reuse the pinned host
            // buffer even while GEMMs continue reading its device counterpart.
            CUDA_CHECK(cudaEventRecord(bufferFree[buffer], computeStream));
        }

        int first = j;
        const int remainder = first % distribution.ranks;
        if (remainder <= distribution.rank) {
            first += distribution.rank - remainder;
        } else {
            first += distribution.ranks - (remainder - distribution.rank);
        }
        if (first < distribution.blocks) {
            const int batchCount =
                1 + (distribution.blocks - 1 - first) / distribution.ranks;
            const int last =
                first + (batchCount - 1) * distribution.ranks;
            const bool partialLast =
                last == distribution.blocks - 1 &&
                distribution.blockExtent(last) != distribution.blockSize;
            const int fullBatches = batchCount - (partialLast ? 1 : 0);
            const double* firstB =
                deviceLocalB +
                static_cast<size_t>(first / distribution.ranks) * bRowStride;

            if (fullBatches != 0) {
                CUBLAS_CHECK(cublasDgemmStridedBatched(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    distribution.blockSize, panelRows,
                    static_cast<int>(distribution.n), &one, firstB,
                    static_cast<int>(distribution.n),
                    static_cast<long long>(bRowStride), devicePanels[buffer],
                    static_cast<int>(distribution.n), 0, &zero,
                    distribution.tile(first, j), distribution.blockSize,
                    static_cast<long long>(distribution.rowStride),
                    fullBatches));
            }
            if (partialLast) {
                const int rows = distribution.blockExtent(last);
                const double* lastB =
                    deviceLocalB +
                    static_cast<size_t>(last / distribution.ranks) *
                        bRowStride;
                CUBLAS_CHECK(cublasDgemm(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, panelRows,
                    static_cast<int>(distribution.n), &one, lastB,
                    static_cast<int>(distribution.n), devicePanels[buffer],
                    static_cast<int>(distribution.n), &zero,
                    distribution.tile(last, j), distribution.blockSize));
            }
        }

        if (distribution.owner(j) == distribution.rank) {
            const int threads = 256;
            addDiagonal<<<(panelRows + threads - 1) / threads, threads, 0,
                          computeStream>>>(
                distribution.blockRow(j), j * distribution.blockSize,
                panelRows, distribution.blockSize,
                static_cast<double>(distribution.n));
            CUDA_CHECK(cudaGetLastError());
        }
        if (cudaAwareMpi) {
            // Direct MPI transport may reuse the device buffer only after all
            // consumers on the compute stream have completed.
            CUDA_CHECK(cudaEventRecord(bufferFree[buffer], computeStream));
        }
        if (j + 1 < distribution.blocks) {
            postBBlock(j + 1, buffer ^ 1);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaEventDestroy(bufferFree[buffer]));
        CUDA_CHECK(cudaFreeHost(hostPanels[buffer]));
        CUDA_CHECK(cudaFree(devicePanels[buffer]));
    }
    CUDA_CHECK(cudaFree(deviceLocalB));
}

static bool cudaAwareMpiEnabled() {
    bool enabled = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    enabled = MPIX_Query_cuda_support() != 0;
#endif
    // The override is useful for CUDA-aware MPIs that do not expose the
    // Open-MPI extension above.  Staging through pinned memory remains the
    // safe, portable default when support cannot be queried.
    if (const char* setting = std::getenv("CHOLESKY_CUDA_AWARE_MPI")) {
        enabled = std::strcmp(setting, "0") != 0;
    }
    return enabled;
}

static void launchTrailingUpdates(const Distribution& distribution, int k,
                                  int j, const double* panel,
                                  cublasHandle_t cublas) {
    const int panelRows = distribution.blockExtent(j);
    const int panelColumns = distribution.blockExtent(k);
    const double minusOne = -1.0;
    const double one = 1.0;

    if (distribution.owner(j) == distribution.rank) {
        CUBLAS_CHECK(cublasDsyrk(
            cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, panelRows,
            panelColumns, &minusOne, panel, distribution.blockSize, &one,
            distribution.tile(j, j), distribution.blockSize));
    }

    int first = j + 1;
    const int remainder = first % distribution.ranks;
    if (remainder <= distribution.rank) {
        first += distribution.rank - remainder;
    } else {
        first += distribution.ranks - (remainder - distribution.rank);
    }
    if (first >= distribution.blocks) {
        return;
    }

    int batchCount = 1 + (distribution.blocks - 1 - first) /
                             distribution.ranks;
    const int lastBlock = first + (batchCount - 1) * distribution.ranks;
    const bool partialLast =
        lastBlock == distribution.blocks - 1 &&
        distribution.blockExtent(lastBlock) != distribution.blockSize;
    const int fullBatches = batchCount - (partialLast ? 1 : 0);

    if (fullBatches > 0) {
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T, distribution.blockSize,
            panelRows, panelColumns, &minusOne, distribution.tile(first, k),
            distribution.blockSize,
            static_cast<long long>(distribution.rowStride), panel,
            distribution.blockSize, 0, &one, distribution.tile(first, j),
            distribution.blockSize,
            static_cast<long long>(distribution.rowStride), fullBatches));
    }

    if (partialLast) {
        const int rows = distribution.blockExtent(lastBlock);
        CUBLAS_CHECK(cublasDgemm(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T, rows, panelRows, panelColumns,
            &minusOne, distribution.tile(lastBlock, k), distribution.blockSize,
            panel, distribution.blockSize, &one,
            distribution.tile(lastBlock, j), distribution.blockSize));
    }
}

static bool choleskyDecomposition(Distribution& distribution,
                                  cublasHandle_t cublas,
                                  cusolverDnHandle_t cusolver,
                                  cudaStream_t computeStream,
                                  cudaStream_t communicationStream,
                                  bool cudaAwareMpi) {
    double* devicePanels[2] = {nullptr, nullptr};
    double* hostPanels[2] = {nullptr, nullptr};
    cudaEvent_t panelFree[2];
    const size_t panelEntries =
        static_cast<size_t>(distribution.blockSize) * distribution.blockSize;
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaMalloc(&devicePanels[buffer],
                              panelEntries * sizeof(double)));
        CUDA_CHECK(cudaHostAlloc(&hostPanels[buffer],
                                 panelEntries * sizeof(double),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelFree[buffer],
                                            cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(panelFree[buffer], computeStream));
    }

    int workspaceElements = 0;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    // The first local allocation is valid even on ranks with no block rows:
    // cuSOLVER only uses this pointer for its size query.
    double* sizeQueryPointer = distribution.deviceA;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cusolver, CUBLAS_FILL_MODE_LOWER, distribution.blockSize,
        sizeQueryPointer, distribution.blockSize, &workspaceElements));
    CUDA_CHECK(cudaMalloc(&workspace,
                          static_cast<size_t>(workspaceElements) *
                              sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceInfo, sizeof(int)));

    for (int k = 0; k < distribution.blocks; ++k) {
        const int root = distribution.owner(k);
        const int diagonalSize = distribution.blockExtent(k);
        int info = 0;
        if (distribution.rank == root) {
            CUSOLVER_CHECK(cusolverDnDpotrf(
                cusolver, CUBLAS_FILL_MODE_LOWER, diagonalSize,
                distribution.tile(k, k), distribution.blockSize, workspace,
                workspaceElements, deviceInfo));
            CUDA_CHECK(cudaMemcpyAsync(&info, deviceInfo, sizeof(int),
                                       cudaMemcpyDeviceToHost, computeStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, root, MPI_COMM_WORLD));
        if (info != 0) {
            if (distribution.rank == 0) {
                std::printf(
                    "Error: Matrix is not positive definite at diagonal "
                    "element %d\n",
                    k * distribution.blockSize + std::max(0, info - 1));
            }
            CUDA_CHECK(cudaFree(deviceInfo));
            CUDA_CHECK(cudaFree(workspace));
            for (int buffer = 0; buffer < 2; ++buffer) {
                CUDA_CHECK(cudaEventDestroy(panelFree[buffer]));
                CUDA_CHECK(cudaFreeHost(hostPanels[buffer]));
                CUDA_CHECK(cudaFree(devicePanels[buffer]));
            }
            return false;
        }

        const int diagonalCount = distribution.blockSize * diagonalSize;
        if (cudaAwareMpi) {
            if (distribution.rank == root) {
                CUDA_CHECK(cudaMemcpyAsync(
                    devicePanels[0], distribution.tile(k, k),
                    static_cast<size_t>(diagonalCount) * sizeof(double),
                    cudaMemcpyDeviceToDevice, communicationStream));
                CUDA_CHECK(cudaStreamSynchronize(communicationStream));
            }
            MPI_CHECK(MPI_Bcast(devicePanels[0], diagonalCount, MPI_DOUBLE,
                                root, MPI_COMM_WORLD));
        } else {
            if (distribution.rank == root) {
                CUDA_CHECK(cudaMemcpyAsync(
                    hostPanels[0], distribution.tile(k, k),
                    static_cast<size_t>(diagonalCount) * sizeof(double),
                    cudaMemcpyDeviceToHost, communicationStream));
                CUDA_CHECK(cudaStreamSynchronize(communicationStream));
            }
            MPI_CHECK(MPI_Bcast(hostPanels[0], diagonalCount, MPI_DOUBLE, root,
                                MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpyAsync(
                devicePanels[0], hostPanels[0],
                static_cast<size_t>(diagonalCount) * sizeof(double),
                cudaMemcpyHostToDevice, computeStream));
        }

        const double one = 1.0;
        for (int block = k + 1; block < distribution.blocks; ++block) {
            if (distribution.owner(block) != distribution.rank) {
                continue;
            }
            const int rows = distribution.blockExtent(block);
            CUBLAS_CHECK(cublasDtrsm(
                cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rows, diagonalSize, &one,
                devicePanels[0], distribution.blockSize,
                distribution.tile(block, k), distribution.blockSize));
        }
        // An owner must not expose a panel until its TRSM has completed.
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        if (k + 1 < distribution.blocks) {
            MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

            auto postPanel = [&](int j, int buffer) {
                const int panelRoot = distribution.owner(j);
                const int count = distribution.blockSize * diagonalSize;
                if (cudaAwareMpi) {
                    // MPI is about to write this device buffer, so make sure
                    // the update from two panel iterations ago released it.
                    CUDA_CHECK(cudaEventSynchronize(panelFree[buffer]));
                    if (distribution.rank == panelRoot) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            devicePanels[buffer], distribution.tile(j, k),
                            static_cast<size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToDevice, communicationStream));
                        CUDA_CHECK(
                            cudaStreamSynchronize(communicationStream));
                    }
                    MPI_CHECK(MPI_Ibcast(devicePanels[buffer], count,
                                         MPI_DOUBLE, panelRoot, MPI_COMM_WORLD,
                                         &requests[buffer]));
                } else {
                    if (distribution.rank == panelRoot) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            hostPanels[buffer], distribution.tile(j, k),
                            static_cast<size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToHost, communicationStream));
                        CUDA_CHECK(
                            cudaStreamSynchronize(communicationStream));
                    }
                    MPI_CHECK(MPI_Ibcast(hostPanels[buffer], count, MPI_DOUBLE,
                                         panelRoot, MPI_COMM_WORLD,
                                         &requests[buffer]));
                }
            };

            postPanel(k + 1, 0);
            for (int j = k + 1; j < distribution.blocks; ++j) {
                const int buffer = (j - (k + 1)) & 1;
                MPI_CHECK(MPI_Wait(&requests[buffer], MPI_STATUS_IGNORE));
                if (!cudaAwareMpi) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        devicePanels[buffer], hostPanels[buffer],
                        static_cast<size_t>(distribution.blockSize *
                                            diagonalSize) *
                            sizeof(double),
                        cudaMemcpyHostToDevice, computeStream));
                }
                launchTrailingUpdates(distribution, k, j,
                                      devicePanels[buffer], cublas);
                CUDA_CHECK(cudaEventRecord(panelFree[buffer], computeStream));

                if (j + 1 < distribution.blocks) {
                    postPanel(j + 1, buffer ^ 1);
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    CUDA_CHECK(cudaFree(deviceInfo));
    CUDA_CHECK(cudaFree(workspace));
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaEventDestroy(panelFree[buffer]));
        CUDA_CHECK(cudaFreeHost(hostPanels[buffer]));
        CUDA_CHECK(cudaFree(devicePanels[buffer]));
    }
    return true;
}

static int ownedRows(const Distribution& distribution, int rank) {
    int rows = 0;
    for (int block = rank; block < distribution.blocks;
         block += distribution.ranks) {
        rows += distribution.blockExtent(block);
    }
    return rows;
}

static std::vector<double> packLocalRows(const Distribution& distribution,
                                         const double* localMatrix) {
    const int rows = ownedRows(distribution, distribution.rank);
    std::vector<double> packed(checkedProduct(static_cast<size_t>(rows),
                                              distribution.n),
                               0.0);

#pragma omp parallel for schedule(static)
    for (int local = 0; local < distribution.localBlocks; ++local) {
        const int block = distribution.rank + local * distribution.ranks;
        const int extent = distribution.blockExtent(block);
        const size_t globalRow =
            static_cast<size_t>(block) * distribution.blockSize;
        const double* source =
            localMatrix + static_cast<size_t>(local) * distribution.rowStride;
        for (int row = 0; row < extent; ++row) {
            double* destination =
                packed.data() +
                (static_cast<size_t>(local) * distribution.blockSize + row) *
                    distribution.n;
            const size_t lastColumn = globalRow + row;
            for (size_t column = 0; column <= lastColumn; ++column) {
                destination[column] =
                    source[column * distribution.blockSize +
                           static_cast<size_t>(row)];
            }
        }
    }
    return packed;
}

static std::vector<double> gatherMatrix(const Distribution& distribution,
                                        const double* localMatrix) {
    std::vector<double> packed = packLocalRows(distribution, localMatrix);
    std::vector<int> counts(distribution.ranks);
    std::vector<int> displacements(distribution.ranks);
    long long total = 0;
    for (int rank = 0; rank < distribution.ranks; ++rank) {
        const long long count =
            static_cast<long long>(ownedRows(distribution, rank)) *
            static_cast<long long>(distribution.n);
        if (count > INT_MAX || total > INT_MAX) {
            fatalError("MPI", "MPI_Gatherv count exceeds INT_MAX", 1,
                       __FILE__, __LINE__);
        }
        counts[rank] = static_cast<int>(count);
        displacements[rank] = static_cast<int>(total);
        total += count;
    }
    if (total > INT_MAX) {
        fatalError("MPI", "MPI_Gatherv displacement exceeds INT_MAX", 1,
                   __FILE__, __LINE__);
    }

    std::vector<double> rankOrdered;
    if (distribution.rank == 0) {
        rankOrdered.resize(static_cast<size_t>(total));
    }
    MPI_CHECK(MPI_Gatherv(packed.data(), static_cast<int>(packed.size()),
                          MPI_DOUBLE, rankOrdered.data(), counts.data(),
                          displacements.data(), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD));

    std::vector<double> matrix;
    if (distribution.rank == 0) {
        matrix.resize(checkedProduct(distribution.n, distribution.n));
#pragma omp parallel for schedule(static)
        for (long long row = 0; row < static_cast<long long>(distribution.n);
             ++row) {
            const int block =
                static_cast<int>(row / distribution.blockSize);
            const int owner = distribution.owner(block);
            const int localBlock = block / distribution.ranks;
            const int rowInBlock =
                static_cast<int>(row % distribution.blockSize);
            const size_t sourceRow =
                static_cast<size_t>(localBlock) * distribution.blockSize +
                rowInBlock;
            std::memcpy(matrix.data() + static_cast<size_t>(row) *
                                            distribution.n,
                        rankOrdered.data() + displacements[owner] +
                            sourceRow * distribution.n,
                        distribution.n * sizeof(double));
        }
    }
    return matrix;
}

static bool validateCholesky(const std::vector<double>& lower,
                             const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;

#pragma omp parallel for schedule(dynamic, 1) reduction(max : maxError, relativeError)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        for (size_t column = 0; column <= static_cast<size_t>(row); ++column) {
            double reconstructed = 0.0;
            for (size_t k = 0; k <= column; ++k) {
                reconstructed += lower[static_cast<size_t>(row) * n + k] *
                                 lower[column * n + k];
            }
            const double expected =
                original[static_cast<size_t>(row) * n + column];
            const double error = std::fabs(reconstructed - expected);
            maxError = std::max(maxError, error);
            relativeError = std::max(
                relativeError, error / (std::fabs(expected) + 1.0e-10));
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

static void printUsage(const char* program) {
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

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI does not provide the required FUNNELED thread "
                         "support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentsValid = true;
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed == 0 ||
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
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return argumentsValid ? 0 : 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fatalError("CUDA", "no CUDA devices are visible", 1, __FILE__,
                   __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr)); // Create the context before timing.
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    cudaStream_t computeStream;
    cudaStream_t communicationStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&communicationStream, cudaStreamNonBlocking));
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUBLAS_CHECK(cublasSetStream(cublas, computeStream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, computeStream));

    Distribution distribution(n, rank, ranks);
    const size_t localEntries =
        checkedProduct(static_cast<size_t>(distribution.localBlocks),
                       distribution.rowStride);
    // cudaMalloc(0) is invalid.  Ranks with no rows retain a one-double
    // allocation so that all collective/control paths remain uniform.
    CUDA_CHECK(cudaMalloc(&distribution.deviceA,
                          std::max<size_t>(1, localEntries) * sizeof(double)));
    const bool cudaAwareMpi = cudaAwareMpiEnabled();

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), %d OpenMP "
                    "thread(s)/rank, CUDA block size %d\n",
                    ranks, omp_get_max_threads(), distribution.blockSize);
        std::printf("Generating positive definite matrix...\n");
    }

    generatePositiveDefiniteMatrix(distribution, cublas, computeStream,
                                   communicationStream, cudaAwareMpi);

    std::vector<double> originalLocal;
    if (validate) {
        originalLocal.resize(localEntries);
        if (localEntries != 0) {
            CUDA_CHECK(cudaMemcpy(originalLocal.data(), distribution.deviceA,
                                  localEntries * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
    }

    if (rank == 0) {
        std::printf("MPI panel transport: %s\n",
                    cudaAwareMpi ? "CUDA-aware device buffers"
                                 : "pinned host staging");
        std::printf("Computing Cholesky decomposition...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(
        distribution, cublas, cusolver, computeStream, communicationStream,
        cudaAwareMpi);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Allreduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX,
                            MPI_COMM_WORLD));

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        CUDA_CHECK(cudaFree(distribution.deviceA));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
        CUBLAS_CHECK(cublasDestroy(cublas));
        CUDA_CHECK(cudaStreamDestroy(communicationStream));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = operations / std::max(seconds, 1.0e-12) / 1.0e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> lower;
    if (validate || printResults) {
        const size_t rawEntries =
            checkedProduct(static_cast<size_t>(distribution.localBlocks),
                           distribution.rowStride);
        std::vector<double> localFactor(rawEntries);
        if (rawEntries != 0) {
            CUDA_CHECK(cudaMemcpy(localFactor.data(), distribution.deviceA,
                                  rawEntries * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        lower = gatherMatrix(distribution, localFactor.data());
    }

    if (printResults && rank == 0) {
        print_results(lower, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        std::vector<double> original =
            gatherMatrix(distribution, originalLocal.data());
        if (rank == 0) {
            std::printf("Validating result...\n");
            valid = validateCholesky(lower, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        int validInteger = valid ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&validInteger, 1, MPI_INT, 0, MPI_COMM_WORLD));
        valid = validInteger != 0;
    }

    CUDA_CHECK(cudaFree(distribution.deviceA));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(communicationStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
