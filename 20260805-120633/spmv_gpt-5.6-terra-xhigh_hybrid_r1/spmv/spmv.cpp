#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS_PER_BLOCK = 8;
constexpr unsigned int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

namespace {

int g_rank = 0;

[[noreturn]] void abortWithMpi(const char* operation, const char* file, const int line,
                               const int error) {
    char errorString[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    std::fprintf(stderr, "Rank %d: %s failed at %s:%d: %.*s\n", g_rank, operation, file,
                 line, errorLength, errorString);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void mpiCheck(const int error, const char* operation, const char* file, const int line) {
    if (error != MPI_SUCCESS) {
        abortWithMpi(operation, file, line, error);
    }
}

[[noreturn]] void abortWithCuda(const cudaError_t error, const char* operation, const char* file,
                                const int line) {
    std::fprintf(stderr, "Rank %d: %s failed at %s:%d: %s\n", g_rank, operation, file, line,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

void cudaCheck(const cudaError_t error, const char* operation, const char* file, const int line) {
    if (error != cudaSuccess) {
        abortWithCuda(error, operation, file, line);
    }
}

#define MPI_CHECK(call) mpiCheck((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

template <typename T>
void mpiSendLarge(const T* data, size_t count, const MPI_Datatype datatype, const int destination,
                  const int tag) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t sent = 0;
    while (sent < count) {
        const int chunk = static_cast<int>(std::min(maxCount, count - sent));
        MPI_CHECK(MPI_Send(data + sent, chunk, datatype, destination, tag, MPI_COMM_WORLD));
        sent += static_cast<size_t>(chunk);
    }
}

template <typename T>
void mpiReceiveLarge(T* data, size_t count, const MPI_Datatype datatype, const int source,
                     const int tag) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t received = 0;
    while (received < count) {
        const int chunk = static_cast<int>(std::min(maxCount, count - received));
        MPI_CHECK(MPI_Recv(data + received, chunk, datatype, source, tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
        received += static_cast<size_t>(chunk);
    }
}

template <typename T>
void mpiBroadcastLarge(T* data, size_t count, const MPI_Datatype datatype) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t broadcast = 0;
    while (broadcast < count) {
        const int chunk = static_cast<int>(std::min(maxCount, count - broadcast));
        MPI_CHECK(MPI_Bcast(data + broadcast, chunk, datatype, 0, MPI_COMM_WORLD));
        broadcast += static_cast<size_t>(chunk);
    }
}

index_t rowStartForRank(const index_t rows, const int rank, const int ranks) {
    return static_cast<index_t>((static_cast<uint64_t>(rows) * static_cast<uint64_t>(rank)) /
                                static_cast<uint64_t>(ranks));
}

void fill(double* values, const offset_t count, const double maxValue) {
    for (offset_t i = 0; i < count; ++i) {
        values[i] = maxValue * (std::rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// This preserves the original CSR construction and its deterministic seed.
void initRandomMatrix(index_t* columns, offset_t* rowOffsets, const offset_t nonZeros,
                      const index_t dimension) {
    offset_t assigned = 0;
    const offset_t totalEntries = static_cast<offset_t>(dimension) * dimension;
    const double probability = static_cast<double>(nonZeros) / static_cast<double>(totalEntries);

    std::srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        rowOffsets[row] = assigned;
        for (index_t column = 0; column < dimension; ++column) {
            const offset_t entry = static_cast<offset_t>(row) * dimension + column;
            const offset_t entriesLeft = totalEntries - entry;
            const offset_t needed = nonZeros - assigned;
            if (entriesLeft <= needed) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if ((assigned < nonZeros && randomValue <= probability) || fillRemaining) {
                columns[assigned++] = column;
            }
        }
    }
    rowOffsets[dimension] = nonZeros;
}

void spmvCpuOpenMP(const double* values, const index_t* columns, const offset_t* rowOffsets,
                   const double* vector, const index_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        double sum = 0.0;
        for (offset_t entry = rowOffsets[row]; entry < rowOffsets[row + 1]; ++entry) {
            sum += values[entry] * vector[columns[entry]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double value = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(value) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i,
                            ref, value);
                return false;
            }
        } else {
            const double relativeError = std::abs((value - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, ref, value, relativeError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Configuration {
    index_t rows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxValue = 1.0;
    int validate = 0;
    int printResults = 0;
};

enum class ParseResult : int { success = 0, help = 1, error = 2 };

ParseResult parseArguments(const int argc, char** argv, Configuration& configuration) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            configuration.rows = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            configuration.sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            configuration.iterations = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            configuration.maxValue = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            configuration.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            configuration.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return ParseResult::help;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            return ParseResult::error;
        }
    }

    if (configuration.rows == 0 || configuration.sparsity == 0) {
        std::printf("Matrix size and sparsity must both be greater than zero.\n");
        return ParseResult::error;
    }
    return ParseResult::success;
}

void broadcastConfiguration(Configuration& configuration, int& parseResult) {
    MPI_CHECK(MPI_Bcast(&parseResult, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.maxValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.validate, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&configuration.printResults, 1, MPI_INT, 0, MPI_COMM_WORLD));
}

struct LocalMatrix {
    index_t rows = 0;
    std::vector<double> values;
    std::vector<index_t> columns;
    std::vector<offset_t> rowOffsets;
};

LocalMatrix distributeMatrix(const Configuration& configuration, const offset_t nonZeros,
                             std::vector<double>& vector, std::vector<double>& reference,
                             const int ranks) {
    const index_t firstRow = rowStartForRank(configuration.rows, g_rank, ranks);
    const index_t lastRow = rowStartForRank(configuration.rows, g_rank + 1, ranks);
    LocalMatrix local;
    local.rows = lastRow - firstRow;

    if (g_rank == 0) {
        std::vector<double> globalValues(static_cast<size_t>(nonZeros));
        std::vector<index_t> globalColumns(static_cast<size_t>(nonZeros));
        std::vector<offset_t> globalRowOffsets(static_cast<size_t>(configuration.rows) + 1);

        fill(vector.data(), configuration.rows, configuration.maxValue);
        fill(globalValues.data(), nonZeros, configuration.maxValue);
        initRandomMatrix(globalColumns.data(), globalRowOffsets.data(), nonZeros, configuration.rows);

        if (configuration.validate != 0) {
            reference.resize(configuration.rows);
            spmvCpuOpenMP(globalValues.data(), globalColumns.data(), globalRowOffsets.data(),
                           vector.data(), configuration.rows, reference.data());
        }

        for (int destination = 0; destination < ranks; ++destination) {
            const index_t destinationFirst = rowStartForRank(configuration.rows, destination, ranks);
            const index_t destinationLast = rowStartForRank(configuration.rows, destination + 1, ranks);
            const offset_t firstEntry = globalRowOffsets[destinationFirst];
            const offset_t entryCount = globalRowOffsets[destinationLast] - firstEntry;
            const size_t rowOffsetCount = static_cast<size_t>(destinationLast - destinationFirst) + 1;

            if (destination == 0) {
                local.values.assign(globalValues.begin() + static_cast<std::ptrdiff_t>(firstEntry),
                                    globalValues.begin() + static_cast<std::ptrdiff_t>(firstEntry + entryCount));
                local.columns.assign(globalColumns.begin() + static_cast<std::ptrdiff_t>(firstEntry),
                                     globalColumns.begin() + static_cast<std::ptrdiff_t>(firstEntry + entryCount));
                local.rowOffsets.assign(
                    globalRowOffsets.begin() + static_cast<std::ptrdiff_t>(destinationFirst),
                    globalRowOffsets.begin() + static_cast<std::ptrdiff_t>(destinationLast) + 1);
            } else {
                mpiSendLarge(globalRowOffsets.data() + destinationFirst, rowOffsetCount, MPI_UINT64_T,
                             destination, 100);
                if (entryCount != 0) {
                    mpiSendLarge(globalValues.data() + firstEntry, static_cast<size_t>(entryCount), MPI_DOUBLE,
                                 destination, 101);
                    mpiSendLarge(globalColumns.data() + firstEntry, static_cast<size_t>(entryCount), MPI_UINT32_T,
                                 destination, 102);
                }
            }
        }
    } else {
        local.rowOffsets.resize(static_cast<size_t>(local.rows) + 1);
        mpiReceiveLarge(local.rowOffsets.data(), local.rowOffsets.size(), MPI_UINT64_T, 0, 100);
        const offset_t entryCount = local.rowOffsets.back() - local.rowOffsets.front();
        local.values.resize(static_cast<size_t>(entryCount));
        local.columns.resize(static_cast<size_t>(entryCount));
        mpiReceiveLarge(local.values.data(), local.values.size(), MPI_DOUBLE, 0, 101);
        mpiReceiveLarge(local.columns.data(), local.columns.size(), MPI_UINT32_T, 0, 102);
    }

    mpiBroadcastLarge(vector.data(), vector.size(), MPI_DOUBLE);

    const offset_t firstEntry = local.rowOffsets.front();
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(local.rowOffsets.size()); ++row) {
        local.rowOffsets[static_cast<size_t>(row)] -= firstEntry;
    }
    return local;
}

template <typename T>
T* allocateDevice(const size_t count) {
    T* pointer = nullptr;
    const size_t allocationCount = std::max<size_t>(count, 1);
    if (allocationCount > std::numeric_limits<size_t>::max() / sizeof(T)) {
        abortWithCuda(cudaErrorMemoryAllocation, "device allocation size check", __FILE__, __LINE__);
    }
    CUDA_CHECK(cudaMalloc(&pointer, allocationCount * sizeof(T)));
    return pointer;
}

__global__ void spmvWarpKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const offset_t* __restrict__ rowOffsets,
                               const double* __restrict__ vector, double* __restrict__ output,
                               const index_t rows) {
    const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    const offset_t begin = rowOffsets[row];
    const offset_t end = rowOffsets[row + 1];
    for (offset_t entry = begin + lane; entry < end; entry += WARP_SIZE) {
        sum += values[entry] * vector[columns[entry]];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

int selectLocalGpu() {
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL,
                                  &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithCuda(cudaErrorNoDevice, "cudaGetDeviceCount", __FILE__, __LINE__);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    return device;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initResult = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initResult != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed.\n");
        return 1;
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    int ranks = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    Configuration configuration;
    int parseResult = static_cast<int>(ParseResult::success);
    if (g_rank == 0) {
        parseResult = static_cast<int>(parseArguments(argc, argv, configuration));
        if (parseResult != static_cast<int>(ParseResult::success)) {
            printUsage(argv[0]);
        }
    }
    broadcastConfiguration(configuration, parseResult);
    if (parseResult != static_cast<int>(ParseResult::success)) {
        MPI_CHECK(MPI_Finalize());
        return parseResult == static_cast<int>(ParseResult::help) ? 0 : 1;
    }

    const offset_t matrixEntries = static_cast<offset_t>(configuration.rows) * configuration.rows;
    const offset_t nonZeros = matrixEntries / configuration.sparsity;
    if (nonZeros > static_cast<offset_t>(std::numeric_limits<size_t>::max())) {
        if (g_rank == 0) {
            std::fprintf(stderr, "Matrix is too large for this platform's address space.\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    if (g_rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", configuration.rows, configuration.rows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", configuration.sparsity);
        std::printf("Non-zero elements: %llu (%.2f%% sparse)\n",
                    static_cast<unsigned long long>(nonZeros),
                    100.0 * (1.0 - static_cast<double>(nonZeros) / matrixEntries));
        std::printf("Iterations: %u\n", configuration.iterations);
        std::printf("Max value: %.2f\n", configuration.maxValue);
        std::printf("Validation: %s\n", configuration.validate != 0 ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        if (providedThreadLevel < MPI_THREAD_FUNNELED) {
            std::printf("MPI thread level is below MPI_THREAD_FUNNELED; MPI calls remain on the main thread.\n");
        }
        std::printf("Initializing data structures...\n");
    }

    std::vector<double> vector(configuration.rows);
    std::vector<double> reference;
    LocalMatrix local = distributeMatrix(configuration, nonZeros, vector, reference, ranks);

    const int device = selectLocalGpu();
    if (g_rank == 0) {
        std::printf("Computing SpMV on CUDA GPUs (root rank uses device %d)...\n", device);
    }

    double* deviceValues = allocateDevice<double>(local.values.size());
    index_t* deviceColumns = allocateDevice<index_t>(local.columns.size());
    offset_t* deviceRowOffsets = allocateDevice<offset_t>(local.rowOffsets.size());
    double* deviceVector = allocateDevice<double>(vector.size());
    double* deviceOutput = allocateDevice<double>(local.rows);

    if (!local.values.empty()) {
        CUDA_CHECK(cudaMemcpy(deviceValues, local.values.data(), local.values.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceColumns, local.columns.data(), local.columns.size() * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(deviceRowOffsets, local.rowOffsets.data(),
                          local.rowOffsets.size() * sizeof(offset_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVector, vector.data(), vector.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    // Match the original value-initialized host output when the iteration count is zero.
    CUDA_CHECK(cudaMemset(deviceOutput, 0, std::max<size_t>(local.rows, 1) * sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExecution = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));

    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<uint64_t>(local.rows) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    if (local.rows != 0) {
        // Capturing the fixed CSR launch avoids per-iteration host launch overhead.
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        spmvWarpKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            deviceValues, deviceColumns, deviceRowOffsets, deviceVector, deviceOutput, local.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExecution, graph, nullptr, nullptr, 0));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    for (index_t iteration = 0; iteration < configuration.iterations; ++iteration) {
        if (local.rows != 0) {
            CUDA_CHECK(cudaGraphLaunch(graphExecution, stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(endEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(endEvent));
    CUDA_CHECK(cudaGetLastError());

    float localMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&localMilliseconds, startEvent, endEvent));
    double elapsedMilliseconds = 0.0;
    const double localElapsedMilliseconds = static_cast<double>(localMilliseconds);
    MPI_CHECK(MPI_Reduce(&localElapsedMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    const bool needHostOutput = configuration.validate != 0 || configuration.printResults != 0;
    std::vector<double> localOutput;
    std::vector<double> output;
    if (needHostOutput) {
        localOutput.resize(local.rows);
        if (!localOutput.empty()) {
            CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput, localOutput.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        if (g_rank == 0) {
            output.resize(configuration.rows);
            std::copy(localOutput.begin(), localOutput.end(), output.begin());
            for (int source = 1; source < ranks; ++source) {
                const index_t sourceFirst = rowStartForRank(configuration.rows, source, ranks);
                const index_t sourceLast = rowStartForRank(configuration.rows, source + 1, ranks);
                mpiReceiveLarge(output.data() + sourceFirst, static_cast<size_t>(sourceLast - sourceFirst),
                                MPI_DOUBLE, source, 103);
            }
        } else {
            mpiSendLarge(localOutput.data(), localOutput.size(), MPI_DOUBLE, 0, 103);
        }
    }

    CUDA_CHECK(cudaEventDestroy(endEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    if (graphExecution != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExecution));
    }
    if (graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceVector));
    CUDA_CHECK(cudaFree(deviceRowOffsets));
    CUDA_CHECK(cudaFree(deviceColumns));
    CUDA_CHECK(cudaFree(deviceValues));

    int exitCode = 0;
    if (g_rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        const double averageTime = configuration.iterations == 0
                                       ? 0.0
                                       : elapsedMilliseconds / configuration.iterations;
        const double gflops = elapsedMilliseconds == 0.0
                                  ? 0.0
                                  : (2.0 * static_cast<double>(nonZeros) * configuration.iterations) /
                                        (elapsedMilliseconds / 1000.0) / 1.0e9;
        std::printf("Average time per iteration: %.3f ms\n", averageTime);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (configuration.printResults != 0) {
            print_results(output, "OutputVector");
        }
        if (configuration.validate != 0) {
            std::printf("Validating result...\n");
            if (verifyResults(reference.data(), output.data(), configuration.rows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
