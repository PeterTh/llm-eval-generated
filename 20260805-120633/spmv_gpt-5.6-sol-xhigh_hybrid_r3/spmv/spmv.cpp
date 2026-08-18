#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

namespace {

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = CUDA_BLOCK_SIZE / 32;

struct Options {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;
};

enum class KernelKind {
    Scalar,
    Warp,
    Block
};

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseIndex(const char* text, index_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

// Returns 0 on success, 1 on an error, and 2 for a help request.
int parseArguments(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.numRows)) {
                std::printf("Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.sparsity)) {
                std::printf("Invalid sparsity: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIndex(argv[++i], options.iterations)) {
                std::printf("Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            options.maxVal = std::strtod(argv[++i], &end);
            if (errno != 0 || end == argv[i] || *end != '\0') {
                std::printf("Invalid maximum value: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return 2;
        } else {
            std::printf("Unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }

    if (options.numRows == 0 || options.sparsity == 0 || options.iterations == 0) {
        std::printf("Matrix size, sparsity, and iteration count must all be non-zero.\n");
        return 1;
    }
    return 0;
}

void fill(double* values, index_t count, double maxVal) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxVal * (std::rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Preserve the original deterministic random CSR construction. Initialization is
// outside the measured region; retaining it keeps -r output comparable to the
// serial benchmark.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t nonzeros,
                      index_t dim) {
    index_t assigned = 0;
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * dim;
    const double probability = static_cast<double>(nonzeros) /
                               static_cast<double>(totalEntries);

    std::srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = assigned;
        for (index_t col = 0; col < dim; ++col) {
            const uint64_t linearIndex = static_cast<uint64_t>(row) * dim + col;
            const uint64_t entriesLeft = totalEntries - linearIndex;
            const uint64_t needed = static_cast<uint64_t>(nonzeros) - assigned;
            if (entriesLeft <= needed) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if ((assigned < nonzeros && randomValue <= probability) || fillRemaining) {
                cols[assigned++] = col;
            }
        }
    }
    rowDelimiters[dim] = nonzeros;
}

void spmvReference(const double* values, const index_t* cols,
                   const index_t* rowDelimiters, const double* vector,
                   index_t dim, double* output) {
#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(dim); ++row) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row];
             element < rowDelimiters[row + 1]; ++element) {
            sum += values[element] * vector[cols[element]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                            i, ref, res);
                return false;
            }
        } else {
            const double relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, ref, res, relativeError);
                return false;
            }
        }
    }
    return true;
}

void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", rank,
                     operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }
}

#define CUDA_CHECK(operation, rank) cudaCheck((operation), #operation, (rank))

__global__ void csrScalarKernel(const double* __restrict__ values,
                                const index_t* __restrict__ cols,
                                const index_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vector,
                                index_t rows, double* __restrict__ output) {
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows; row += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row];
             element < rowDelimiters[row + 1]; ++element) {
            sum += values[element] * vector[cols[element]];
        }
        output[row] = sum;
    }
}

__global__ void csrWarpKernel(const double* __restrict__ values,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vector,
                              index_t rows, double* __restrict__ output) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warpInBlock = threadIdx.x >> 5U;
    const index_t firstRow = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    const index_t rowStride = gridDim.x * WARPS_PER_BLOCK;

    for (index_t row = firstRow; row < rows; row += rowStride) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row] + lane;
             element < rowDelimiters[row + 1]; element += 32) {
            sum += values[element] * vector[cols[element]];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) {
            output[row] = sum;
        }
    }
}

__global__ void csrBlockKernel(const double* __restrict__ values,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vector,
                               index_t rows, double* __restrict__ output) {
    __shared__ double warpSums[WARPS_PER_BLOCK];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5U;
    for (index_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row] + threadIdx.x;
             element < rowDelimiters[row + 1]; element += blockDim.x) {
            sum += values[element] * vector[cols[element]];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum;
        }
        __syncthreads();

        if (warp == 0) {
            sum = lane < WARPS_PER_BLOCK ? warpSums[lane] : 0.0;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffU, sum, offset);
            }
            if (lane == 0) {
                output[row] = sum;
            }
        }
        __syncthreads();
    }
}

KernelKind selectKernel(index_t localRows, index_t localNonzeros) {
    const double average = localRows == 0
                               ? 0.0
                               : static_cast<double>(localNonzeros) / localRows;
    if (average < 16.0) {
        return KernelKind::Scalar;
    }
    if (average <= 512.0) {
        return KernelKind::Warp;
    }
    return KernelKind::Block;
}

const char* kernelName(KernelKind kind) {
    switch (kind) {
        case KernelKind::Scalar:
            return "thread-per-row";
        case KernelKind::Warp:
            return "warp-per-row";
        case KernelKind::Block:
            return "block-per-row";
    }
    return "unknown";
}

void launchSpmv(KernelKind kind, int residentBlockLimit,
                const double* deviceValues, const index_t* deviceCols,
                const index_t* deviceRows, const double* deviceVector,
                index_t rows, double* deviceOutput) {
    if (rows == 0) {
        return;
    }

    uint64_t requiredBlocks = 1;
    if (kind == KernelKind::Scalar) {
        requiredBlocks = (static_cast<uint64_t>(rows) + CUDA_BLOCK_SIZE - 1) /
                         CUDA_BLOCK_SIZE;
    } else if (kind == KernelKind::Warp) {
        requiredBlocks = (static_cast<uint64_t>(rows) + WARPS_PER_BLOCK - 1) /
                         WARPS_PER_BLOCK;
    } else {
        requiredBlocks = rows;
    }
    const int blocks = static_cast<int>(std::max<uint64_t>(
        1, std::min<uint64_t>(requiredBlocks, residentBlockLimit)));

    if (kind == KernelKind::Scalar) {
        csrScalarKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            deviceValues, deviceCols, deviceRows, deviceVector, rows, deviceOutput);
    } else if (kind == KernelKind::Warp) {
        csrWarpKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            deviceValues, deviceCols, deviceRows, deviceVector, rows, deviceOutput);
    } else {
        csrBlockKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            deviceValues, deviceCols, deviceRows, deviceVector, rows, deviceOutput);
    }
}

void bcastLarge(void* buffer, uint64_t count, MPI_Datatype datatype, int root,
                MPI_Comm communicator) {
    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    auto* bytes = static_cast<unsigned char*>(buffer);
    uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(
            count - offset, static_cast<uint64_t>(std::numeric_limits<int>::max())));
        MPI_Bcast(bytes + offset * static_cast<uint64_t>(typeSize), chunk, datatype,
                  root, communicator);
        offset += static_cast<uint64_t>(chunk);
    }
}

void sendLarge(const void* buffer, uint64_t count, MPI_Datatype datatype,
               int destination, int tag, MPI_Comm communicator) {
    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    const auto* bytes = static_cast<const unsigned char*>(buffer);
    uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(
            count - offset, static_cast<uint64_t>(std::numeric_limits<int>::max())));
        MPI_Send(bytes + offset * static_cast<uint64_t>(typeSize), chunk, datatype,
                 destination, tag, communicator);
        offset += static_cast<uint64_t>(chunk);
    }
}

void recvLarge(void* buffer, uint64_t count, MPI_Datatype datatype, int source,
               int tag, MPI_Comm communicator) {
    int typeSize = 0;
    MPI_Type_size(datatype, &typeSize);
    auto* bytes = static_cast<unsigned char*>(buffer);
    uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(
            count - offset, static_cast<uint64_t>(std::numeric_limits<int>::max())));
        MPI_Recv(bytes + offset * static_cast<uint64_t>(typeSize), chunk, datatype,
                 source, tag, communicator, MPI_STATUS_IGNORE);
        offset += static_cast<uint64_t>(chunk);
    }
}

void buildBalancedPartitions(const std::vector<index_t>& rowDelimiters,
                             index_t rows, index_t nonzeros, int ranks,
                             std::vector<index_t>& rowBounds,
                             std::vector<index_t>& nonzeroBounds) {
    rowBounds.resize(static_cast<size_t>(ranks) + 1);
    nonzeroBounds.resize(static_cast<size_t>(ranks) + 1);
    rowBounds.front() = 0;
    rowBounds.back() = rows;
    for (int rank = 1; rank < ranks; ++rank) {
        const index_t target = static_cast<index_t>(
            (static_cast<uint64_t>(nonzeros) * rank) / ranks);
        const auto boundary = std::lower_bound(rowDelimiters.begin(),
                                               rowDelimiters.end(), target);
        rowBounds[rank] = static_cast<index_t>(boundary - rowDelimiters.begin());
        if (rowBounds[rank] > rows) {
            rowBounds[rank] = rows;
        }
    }
    for (int rank = 0; rank <= ranks; ++rank) {
        nonzeroBounds[rank] = rowDelimiters[rowBounds[rank]];
    }
}

bool collectiveCsrCountsFit(index_t rows, index_t nonzeros) {
    return static_cast<uint64_t>(rows) + 1 <=
               static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
           nonzeros <= static_cast<index_t>(std::numeric_limits<int>::max());
}

void distributeCsr(const std::vector<double>& globalValues,
                   const std::vector<index_t>& globalCols,
                   const std::vector<index_t>& globalRows,
                   const std::vector<index_t>& rowBounds,
                   const std::vector<index_t>& nonzeroBounds, index_t totalRows,
                   index_t totalNonzeros, int rank, int ranks,
                   std::vector<double>& localValues,
                   std::vector<index_t>& localCols,
                   std::vector<index_t>& localRows) {
    const index_t rowBegin = rowBounds[rank];
    const index_t localRowCount = rowBounds[rank + 1] - rowBegin;
    const index_t nonzeroBegin = nonzeroBounds[rank];
    const index_t localNonzeroCount = nonzeroBounds[rank + 1] - nonzeroBegin;

    localValues.resize(localNonzeroCount);
    localCols.resize(localNonzeroCount);
    localRows.resize(static_cast<size_t>(localRowCount) + 1);

    if (collectiveCsrCountsFit(totalRows, totalNonzeros)) {
        std::vector<int> nonzeroCounts;
        std::vector<int> nonzeroDisplacements;
        std::vector<int> rowCounts;
        std::vector<int> rowDisplacements;
        if (rank == 0) {
            nonzeroCounts.resize(ranks);
            nonzeroDisplacements.resize(ranks);
            rowCounts.resize(ranks);
            rowDisplacements.resize(ranks);
            for (int peer = 0; peer < ranks; ++peer) {
                nonzeroCounts[peer] = static_cast<int>(nonzeroBounds[peer + 1] -
                                                       nonzeroBounds[peer]);
                nonzeroDisplacements[peer] = static_cast<int>(nonzeroBounds[peer]);
                rowCounts[peer] = static_cast<int>(rowBounds[peer + 1] -
                                                   rowBounds[peer] + 1);
                rowDisplacements[peer] = static_cast<int>(rowBounds[peer]);
            }
        }

        MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                     rank == 0 ? nonzeroCounts.data() : nullptr,
                     rank == 0 ? nonzeroDisplacements.data() : nullptr, MPI_DOUBLE,
                     localValues.data(), static_cast<int>(localNonzeroCount), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                     rank == 0 ? nonzeroCounts.data() : nullptr,
                     rank == 0 ? nonzeroDisplacements.data() : nullptr, MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNonzeroCount), MPI_UINT32_T,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? globalRows.data() : nullptr,
                     rank == 0 ? rowCounts.data() : nullptr,
                     rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                     localRows.data(), static_cast<int>(localRowCount + 1), MPI_UINT32_T,
                     0, MPI_COMM_WORLD);
    } else if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(localNonzeroCount); ++i) {
            localValues[i] = globalValues[static_cast<size_t>(nonzeroBegin) + i];
            localCols[i] = globalCols[static_cast<size_t>(nonzeroBegin) + i];
        }
#pragma omp parallel for schedule(static)
        for (long long i = 0; i <= static_cast<long long>(localRowCount); ++i) {
            localRows[i] = globalRows[static_cast<size_t>(rowBegin) + i];
        }

        for (int peer = 1; peer < ranks; ++peer) {
            const index_t peerNonzeros = nonzeroBounds[peer + 1] - nonzeroBounds[peer];
            const index_t peerRows = rowBounds[peer + 1] - rowBounds[peer];
            sendLarge(globalValues.data() + nonzeroBounds[peer], peerNonzeros,
                      MPI_DOUBLE, peer, 100, MPI_COMM_WORLD);
            sendLarge(globalCols.data() + nonzeroBounds[peer], peerNonzeros,
                      MPI_UINT32_T, peer, 101, MPI_COMM_WORLD);
            sendLarge(globalRows.data() + rowBounds[peer],
                      static_cast<uint64_t>(peerRows) + 1, MPI_UINT32_T, peer, 102,
                      MPI_COMM_WORLD);
        }
    } else {
        recvLarge(localValues.data(), localNonzeroCount, MPI_DOUBLE, 0, 100,
                  MPI_COMM_WORLD);
        recvLarge(localCols.data(), localNonzeroCount, MPI_UINT32_T, 0, 101,
                  MPI_COMM_WORLD);
        recvLarge(localRows.data(), static_cast<uint64_t>(localRowCount) + 1,
                  MPI_UINT32_T, 0, 102, MPI_COMM_WORLD);
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i <= static_cast<long long>(localRowCount); ++i) {
        localRows[i] -= nonzeroBegin;
    }
}

void gatherOutput(const std::vector<double>& localOutput,
                  const std::vector<index_t>& rowBounds, index_t totalRows,
                  int rank, int ranks, std::vector<double>& globalOutput) {
    const index_t localRows = rowBounds[rank + 1] - rowBounds[rank];
    if (totalRows <= static_cast<index_t>(std::numeric_limits<int>::max())) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int peer = 0; peer < ranks; ++peer) {
                counts[peer] = static_cast<int>(rowBounds[peer + 1] - rowBounds[peer]);
                displacements[peer] = static_cast<int>(rowBounds[peer]);
            }
        }
        MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    } else if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(localRows); ++i) {
            globalOutput[static_cast<size_t>(rowBounds[0]) + i] = localOutput[i];
        }
        for (int peer = 1; peer < ranks; ++peer) {
            const index_t peerRows = rowBounds[peer + 1] - rowBounds[peer];
            recvLarge(globalOutput.data() + rowBounds[peer], peerRows, MPI_DOUBLE,
                      peer, 400, MPI_COMM_WORLD);
        }
    } else {
        sendLarge(localOutput.data(), localRows, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Options options;
    int parseStatus = rank == 0 ? parseArguments(argc, argv, options) : 0;
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    MPI_Bcast(&options.numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&options.printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const uint64_t matrixEntries = static_cast<uint64_t>(options.numRows) *
                                   options.numRows;
    const uint64_t nonzeros64 = matrixEntries / options.sparsity;
    if (nonzeros64 > std::numeric_limits<index_t>::max()) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "The requested matrix has %llu nonzeros; the CSR index format "
                         "supports at most %u. Increase -s or reduce -n.\n",
                         static_cast<unsigned long long>(nonzeros64),
                         std::numeric_limits<index_t>::max());
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nonzeros = static_cast<index_t>(nonzeros64);

    int ompThreads = 1;
#pragma omp parallel
    {
#pragma omp single
        ompThreads = omp_get_num_threads();
    }
    int minThreads = 1;
    int maxThreads = 1;
    MPI_Reduce(&ompThreads, &minThreads, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ompThreads, &maxThreads, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    if (deviceQuery != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr, "MPI rank %d: no usable CUDA device: %s\n", rank,
                     cudaGetErrorString(deviceQuery));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);
    CUDA_CHECK(cudaFree(nullptr), rank);
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device), rank);
    const int residentBlockLimit = std::max(1, deviceProperties.multiProcessorCount * 8);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", options.numRows, options.numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", options.sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonzeros,
                    100.0 * (1.0 - static_cast<double>(nonzeros64) /
                                       static_cast<double>(matrixEntries)));
        std::printf("Iterations: %u\n", options.iterations);
        std::printf("Max value: %.2f\n", options.maxVal);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        if (minThreads == maxThreads) {
            std::printf("OpenMP threads per rank: %d\n", minThreads);
        } else {
            std::printf("OpenMP threads per rank: %d-%d\n", minThreads, maxThreads);
        }
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
    }

    std::vector<double> globalValues;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRows;
    std::vector<double> denseVector(options.numRows);
    std::vector<index_t> rowBounds(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> nonzeroBounds(static_cast<size_t>(ranks) + 1);

    if (rank == 0) {
        std::printf("Initializing data structures...\n");
        globalValues.resize(nonzeros);
        globalCols.resize(nonzeros);
        globalRows.resize(static_cast<size_t>(options.numRows) + 1);

        // The serial program starts rand() at the C library's default seed (1).
        // Set it explicitly so MPI/CUDA library initialization cannot perturb it.
        std::srand(1);
        fill(denseVector.data(), options.numRows, options.maxVal);
        fill(globalValues.data(), nonzeros, options.maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nonzeros,
                         options.numRows);
        buildBalancedPartitions(globalRows, options.numRows, nonzeros, ranks,
                                rowBounds, nonzeroBounds);
    }

    MPI_Bcast(rowBounds.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nonzeroBounds.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    bcastLarge(denseVector.data(), options.numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> localValues;
    std::vector<index_t> localCols;
    std::vector<index_t> localRows;
    distributeCsr(globalValues, globalCols, globalRows, rowBounds, nonzeroBounds,
                  options.numRows, nonzeros, rank, ranks, localValues, localCols,
                  localRows);

    std::vector<double> reference;
    if (rank == 0 && options.validate) {
        std::printf("Computing OpenMP reference solution...\n");
        reference.resize(options.numRows);
        spmvReference(globalValues.data(), globalCols.data(), globalRows.data(),
                      denseVector.data(), options.numRows, reference.data());
    }

    // The complete matrix is no longer needed after distribution/reference work.
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRows);
    }

    const index_t localRowCount = rowBounds[rank + 1] - rowBounds[rank];
    const index_t localNonzeroCount = nonzeroBounds[rank + 1] - nonzeroBounds[rank];
    std::vector<double> localOutput(localRowCount);

    double* deviceValues = nullptr;
    index_t* deviceCols = nullptr;
    index_t* deviceRows = nullptr;
    double* deviceVector = nullptr;
    double* deviceOutput = nullptr;
    if (localNonzeroCount != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceValues),
                              static_cast<size_t>(localNonzeroCount) * sizeof(double)),
                   rank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCols),
                              static_cast<size_t>(localNonzeroCount) * sizeof(index_t)),
                   rank);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRows),
                          (static_cast<size_t>(localRowCount) + 1) * sizeof(index_t)),
               rank);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVector),
                          static_cast<size_t>(options.numRows) * sizeof(double)),
               rank);
    if (localRowCount != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput),
                              static_cast<size_t>(localRowCount) * sizeof(double)),
                   rank);
    }

    if (localNonzeroCount != 0) {
        CUDA_CHECK(cudaMemcpy(deviceValues, localValues.data(),
                              static_cast<size_t>(localNonzeroCount) * sizeof(double),
                              cudaMemcpyHostToDevice),
                   rank);
        CUDA_CHECK(cudaMemcpy(deviceCols, localCols.data(),
                              static_cast<size_t>(localNonzeroCount) * sizeof(index_t),
                              cudaMemcpyHostToDevice),
                   rank);
    }
    CUDA_CHECK(cudaMemcpy(deviceRows, localRows.data(),
                          (static_cast<size_t>(localRowCount) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice),
               rank);
    CUDA_CHECK(cudaMemcpy(deviceVector, denseVector.data(),
                          static_cast<size_t>(options.numRows) * sizeof(double),
                          cudaMemcpyHostToDevice),
               rank);

    const KernelKind kernel = selectKernel(localRowCount, localNonzeroCount);
    if (rank == 0) {
        std::printf("Computing SpMV with MPI + OpenMP + CUDA (%s on rank 0)...\n",
                    kernelName(kernel));
    }

    // Warm up the context/kernel before the collective timed region.
    launchSpmv(kernel, residentBlockLimit, deviceValues, deviceCols, deviceRows,
               deviceVector, localRowCount, deviceOutput);
    CUDA_CHECK(cudaGetLastError(), rank);
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
    MPI_Barrier(MPI_COMM_WORLD);

    const double start = MPI_Wtime();
    for (index_t iteration = 0; iteration < options.iterations; ++iteration) {
        launchSpmv(kernel, residentBlockLimit, deviceValues, deviceCols, deviceRows,
                   deviceVector, localRowCount, deviceOutput);
    }
    CUDA_CHECK(cudaGetLastError(), rank);
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (options.validate || options.printResults) {
        if (localRowCount != 0) {
            CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput,
                                  static_cast<size_t>(localRowCount) * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       rank);
        }
    }

    CUDA_CHECK(cudaFree(deviceOutput), rank);
    CUDA_CHECK(cudaFree(deviceVector), rank);
    CUDA_CHECK(cudaFree(deviceRows), rank);
    CUDA_CHECK(cudaFree(deviceCols), rank);
    CUDA_CHECK(cudaFree(deviceValues), rank);
    MPI_Comm_free(&localCommunicator);

    if (rank == 0) {
        const double milliseconds = elapsedSeconds * 1.0e3;
        const double averageMilliseconds = milliseconds / options.iterations;
        const double gflops = elapsedSeconds > 0.0
                                  ? (2.0 * static_cast<double>(nonzeros) *
                                     options.iterations) /
                                        elapsedSeconds / 1.0e9
                                  : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n", averageMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if (options.validate || options.printResults) {
        std::vector<double> globalOutput;
        if (rank == 0) {
            globalOutput.resize(options.numRows);
        }
        gatherOutput(localOutput, rowBounds, options.numRows, rank, ranks,
                     globalOutput);

        if (rank == 0 && options.printResults) {
            print_results(globalOutput, "OutputVector");
        }
        if (rank == 0 && options.validate) {
            std::printf("Validating result...\n");
            if (verifyResults(reference.data(), globalOutput.data(), options.numRows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
