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
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

[[noreturn]] static void fatal(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::exit(1);
    }
}

// SplitMix64 provides inexpensive, reproducible independent values.  It lets
// every MPI rank construct its part of the matrix without communication.
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

static inline double randomValue(uint64_t key, double maxValue) {
    return maxValue * static_cast<double>(mix64(key) >> 11U) *
           (1.0 / 9007199254740992.0); // 2^-53
}

static bool parseIndex(const char* text, index_t& result) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || value > std::numeric_limits<index_t>::max()) {
        return false;
    }
    result = static_cast<index_t>(value);
    return true;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// One CUDA warp owns one CSR row. Loads are coalesced even when row lengths
// are irregular, and the sum stays in registers until the final store.
__global__ void spmvWarpKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const index_t* __restrict__ rowOffsets,
                               const double* __restrict__ vector,
                               index_t rows, double* __restrict__ output) {
    const index_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = thread >> 5U;
    const unsigned lane = threadIdx.x & 31U;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    const index_t begin = rowOffsets[row];
    const index_t end = rowOffsets[row + 1];
    for (index_t item = begin + lane; item < end; item += 32U) {
        sum += values[item] * vector[columns[item]];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

static void spmvCpu(const std::vector<double>& values,
                    const std::vector<index_t>& columns,
                    const std::vector<index_t>& rowOffsets,
                    const std::vector<double>& vector,
                    std::vector<double>& output) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t row = 0; row < static_cast<int64_t>(output.size()); ++row) {
        double sum = 0.0;
        for (index_t item = rowOffsets[row]; item < rowOffsets[row + 1]; ++item) {
            sum += values[item] * vector[columns[item]];
        }
        output[row] = sum;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        fatal("MPI does not provide the required MPI_THREAD_FUNNELED support", rank);
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxValue = 1.0;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            argumentsValid &= parseIndex(argv[++i], numRows);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            argumentsValid &= parseIndex(argv[++i], sparsity);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid &= parseIndex(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxValue = std::strtod(argv[++i], &end);
            argumentsValid &= end != argv[i] && *end == '\0' && std::isfinite(maxValue);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (help || !argumentsValid || numRows == 0 || sparsity == 0 || iterations == 0 ||
        maxValue < 0.0 || numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            if (!help) std::fprintf(stderr, "All counts must be positive, and max value must be non-negative.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help && argumentsValid ? 0 : 1;
    }

    const uint64_t square = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t globalItems64 = square / sparsity;
    if (globalItems64 > std::numeric_limits<index_t>::max()) {
        if (rank == 0) std::fprintf(stderr, "The CSR index space exceeds the 32-bit benchmark format.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t globalItems = static_cast<index_t>(globalItems64);

    // Contiguous, balanced row ownership makes final collection a single Gatherv.
    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / ranks);
    const index_t localRows = lastRow - firstRow;
    const index_t itemsPerRow = globalItems / numRows;
    const index_t extraRows = globalItems % numRows;
    const uint64_t firstItem64 = static_cast<uint64_t>(firstRow) * itemsPerRow +
                                 std::min(firstRow, extraRows);
    const uint64_t lastItem64 = static_cast<uint64_t>(lastRow) * itemsPerRow +
                                std::min(lastRow, extraRows);
    const index_t localItems = static_cast<index_t>(lastItem64 - firstItem64);

    std::vector<double> hostVector(numRows);
    std::vector<double> hostValues(localItems);
    std::vector<index_t> hostColumns(localItems);
    std::vector<index_t> hostRowOffsets(static_cast<size_t>(localRows) + 1);
    std::vector<double> hostOutput(localRows);

#pragma omp parallel for schedule(static)
    for (int64_t column = 0; column < static_cast<int64_t>(numRows); ++column) {
        hostVector[column] = randomValue(0x6a09e667f3bcc909ULL + column, maxValue);
    }

    // Each row uses an affine permutation of [0,n), yielding unique random-looking
    // columns with no rejection loop, including the dense (sparsity=1) case.
#pragma omp parallel for schedule(static)
    for (int64_t localRow = 0; localRow < static_cast<int64_t>(localRows); ++localRow) {
        const index_t globalRow = firstRow + static_cast<index_t>(localRow);
        const index_t rowItems = itemsPerRow + (globalRow < extraRows ? 1U : 0U);
        const index_t offset = static_cast<index_t>(
            static_cast<uint64_t>(globalRow) * itemsPerRow + std::min(globalRow, extraRows) - firstItem64);
        hostRowOffsets[localRow] = offset;

        index_t multiplier = static_cast<index_t>(mix64(0x243f6a8885a308d3ULL + globalRow) % numRows);
        if (multiplier == 0) multiplier = 1;
        while (std::gcd(multiplier, numRows) != 1U) {
            if (++multiplier == numRows) multiplier = 1;
        }
        const index_t shift = static_cast<index_t>(mix64(0x13198a2e03707344ULL + globalRow) % numRows);
        for (index_t item = 0; item < rowItems; ++item) {
            const index_t localItem = offset + item;
            hostColumns[localItem] = static_cast<index_t>(
                (static_cast<uint64_t>(multiplier) * item + shift) % numRows);
            hostValues[localItem] = randomValue(0xa4093822299f31d0ULL + firstItem64 + localItem,
                                                maxValue);
        }
    }
    hostRowOffsets[localRows] = localItems;

    // Bind ranks on the same node round-robin to its visible accelerators.
    MPI_Comm nodeCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &nodeCommunicator);
    int localRank = 0;
    MPI_Comm_rank(nodeCommunicator, &localRank);
    MPI_Comm_free(&nodeCommunicator);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) fatal("the hybrid benchmark requires at least one CUDA device", rank);
    const int device = localRank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice", rank);

    cudaDeviceProp deviceProperties{};
    cudaCheck(cudaGetDeviceProperties(&deviceProperties, device), "cudaGetDeviceProperties", rank);

    double* deviceValues = nullptr;
    double* deviceVector = nullptr;
    double* deviceOutput = nullptr;
    index_t* deviceColumns = nullptr;
    index_t* deviceRowOffsets = nullptr;
    const size_t allocatedItems = std::max<size_t>(localItems, 1);
    const size_t allocatedRows = std::max<size_t>(localRows, 1);
    cudaCheck(cudaMalloc(&deviceValues, allocatedItems * sizeof(double)), "cudaMalloc(values)", rank);
    cudaCheck(cudaMalloc(&deviceColumns, allocatedItems * sizeof(index_t)), "cudaMalloc(columns)", rank);
    cudaCheck(cudaMalloc(&deviceRowOffsets, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)),
              "cudaMalloc(row offsets)", rank);
    cudaCheck(cudaMalloc(&deviceVector, static_cast<size_t>(numRows) * sizeof(double)),
              "cudaMalloc(vector)", rank);
    cudaCheck(cudaMalloc(&deviceOutput, allocatedRows * sizeof(double)), "cudaMalloc(output)", rank);

    if (localItems != 0) {
        cudaCheck(cudaMemcpy(deviceValues, hostValues.data(), localItems * sizeof(double),
                             cudaMemcpyHostToDevice), "copy values", rank);
        cudaCheck(cudaMemcpy(deviceColumns, hostColumns.data(), localItems * sizeof(index_t),
                             cudaMemcpyHostToDevice), "copy columns", rank);
    }
    cudaCheck(cudaMemcpy(deviceRowOffsets, hostRowOffsets.data(),
                         (static_cast<size_t>(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice),
              "copy row offsets", rank);
    cudaCheck(cudaMemcpy(deviceVector, hostVector.data(), static_cast<size_t>(numRows) * sizeof(double),
                         cudaMemcpyHostToDevice), "copy vector", rank);

    const uint64_t warps = localRows;
    const int blocks = static_cast<int>((warps * 32U + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    if (blocks != 0) {
        spmvWarpKernel<<<blocks, CUDA_BLOCK_SIZE>>>(deviceValues, deviceColumns, deviceRowOffsets,
                                                    deviceVector, localRows, deviceOutput);
    }
    cudaCheck(cudaDeviceSynchronize(), "warm-up kernel", rank);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", globalItems,
                    100.0 * (1.0 - static_cast<double>(globalItems64) / static_cast<double>(square)));
        std::printf("Iterations: %u\n", iterations);
        std::printf("MPI ranks: %d; OpenMP threads/rank: %d; GPUs/node: %d\n",
                    ranks, omp_get_max_threads(), deviceCount);
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        if (blocks != 0) {
            spmvWarpKernel<<<blocks, CUDA_BLOCK_SIZE>>>(deviceValues, deviceColumns, deviceRowOffsets,
                                                        deviceVector, localRows, deviceOutput);
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "timed kernels", rank);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localRows != 0) {
        cudaCheck(cudaMemcpy(hostOutput.data(), deviceOutput, localRows * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy output", rank);
    }

    std::vector<int> receiveCounts;
    std::vector<int> receiveOffsets;
    std::vector<double> globalOutput;
    if (rank == 0) {
        receiveCounts.resize(ranks);
        receiveOffsets.resize(ranks);
        for (int process = 0; process < ranks; ++process) {
            const uint64_t begin = static_cast<uint64_t>(numRows) * process / ranks;
            const uint64_t end = static_cast<uint64_t>(numRows) * (process + 1) / ranks;
            receiveCounts[process] = static_cast<int>(end - begin);
            receiveOffsets[process] = static_cast<int>(begin);
        }
        globalOutput.resize(numRows);
    }
    MPI_Gatherv(hostOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? globalOutput.data() : nullptr,
                rank == 0 ? receiveCounts.data() : nullptr,
                rank == 0 ? receiveOffsets.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int validationPassed = 1;
    if (validate) {
        std::vector<double> reference(localRows);
        spmvCpu(hostValues, hostColumns, hostRowOffsets, hostVector, reference);
        int localPassed = 1;
#pragma omp parallel for reduction(&:localPassed) schedule(static)
        for (int64_t row = 0; row < static_cast<int64_t>(localRows); ++row) {
            const double ref = reference[row];
            const double result = hostOutput[row];
            const double error = std::abs(result - ref);
            const bool valid = std::abs(ref) < 1e-10 ? error <= MAX_RELATIVE_ERROR
                                                     : error / std::abs(ref) <= MAX_RELATIVE_ERROR;
            localPassed &= static_cast<int>(valid);
        }
        MPI_Allreduce(&localPassed, &validationPassed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double average = milliseconds / iterations;
        const double gflops = elapsed > 0.0
            ? (2.0 * static_cast<double>(globalItems64) * iterations) / elapsed / 1.0e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n", average);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(globalOutput, "OutputVector");
        if (validate) std::printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
    }

    cudaFree(deviceOutput);
    cudaFree(deviceVector);
    cudaFree(deviceRowOffsets);
    cudaFree(deviceColumns);
    cudaFree(deviceValues);
    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
