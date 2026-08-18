#include <algorithm>
#include <cerrno>
#include <cmath>
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

using index_t = std::uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = CUDA_BLOCK_SIZE / CUDA_WARP_SIZE;

[[noreturn]] void abortMpi(const char* message, const int errorCode = 1) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s\n", rank, message);
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    }
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}

void checkMpi(const int status, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char error[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, error, &length);
    char message[MPI_MAX_ERROR_STRING + 128];
    std::snprintf(message, sizeof(message), "%s failed: %.*s", operation, length, error);
    abortMpi(message, status);
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    char message[512];
    std::snprintf(message, sizeof(message), "%s failed: %s", operation,
                  cudaGetErrorString(status));
    abortMpi(message, static_cast<int>(status));
}

#define MPI_CHECK(call) checkMpi((call), #call)
#define CUDA_CHECK(call) checkCuda((call), #call)

void fill(double* values, const index_t count, const double maxVal) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxVal * (std::rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t nnz,
                      const index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(nnz) /
                               (static_cast<double>(dim) * static_cast<double>(dim));

    std::srand(8675309);
    bool fillRemaining = false;
    const std::uint64_t totalEntries = static_cast<std::uint64_t>(dim) * dim;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = assigned;
        for (index_t col = 0; col < dim; ++col) {
            const std::uint64_t position = static_cast<std::uint64_t>(row) * dim + col;
            const std::uint64_t entriesLeft = totalEntries - position;
            const std::uint64_t needToAssign = nnz - assigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if ((assigned < nnz && randomValue <= probability) || fillRemaining) {
                cols[assigned++] = col;
            }
        }
    }
    rowDelimiters[dim] = nnz;
}

// The outer loop is independent; the inner loop intentionally retains CSR order so
// that the OpenMP reference has the same floating-point evaluation order per row.
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(dim); ++row) {
        double sum = 0.0;
        for (index_t item = rowDelimiters[row]; item < rowDelimiters[row + 1]; ++item) {
            sum += val[item] * vec[cols[item]];
        }
        out[row] = sum;
    }
}

__global__ void spmvThreadKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec, const index_t rows,
                                 double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (index_t item = rowDelimiters[row]; item < rowDelimiters[row + 1]; ++item) {
        sum += val[item] * __ldg(vec + cols[item]);
    }
    out[row] = sum;
}

__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec, const index_t rows,
                               double* __restrict__ out) {
    const int lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_SIZE;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (index_t item = rowDelimiters[row] + lane; item < rowDelimiters[row + 1];
         item += CUDA_WARP_SIZE) {
        sum += val[item] * __ldg(vec + cols[item]);
    }

#pragma unroll
    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

void launchSpmv(const double* dVal, const index_t* dCols, const index_t* dRows,
                const double* dVec, const index_t rows, const bool useWarpKernel,
                double* dOut, cudaStream_t stream) {
    if (rows == 0) {
        return;
    }

    if (useWarpKernel) {
        const unsigned int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        spmvWarpKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(dVal, dCols, dRows, dVec,
                                                               rows, dOut);
    } else {
        const unsigned int blocks = (rows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        spmvThreadKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(dVal, dCols, dRows, dVec,
                                                                 rows, dOut);
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i,
                            ref, res);
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

void printUsage(const char* program) {
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

bool parseIndex(const char* text, index_t* value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    *value = static_cast<index_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMpi("MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            argumentsValid = parseIndex(argv[++i], &numRows) && argumentsValid;
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            argumentsValid = parseIndex(argv[++i], &sparsity) && argumentsValid;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid = parseIndex(argv[++i], &iterations) && argumentsValid;
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = std::strtod(argv[++i], &end);
            argumentsValid = end != nullptr && *end == '\0' && std::isfinite(maxVal) &&
                             argumentsValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (worldRank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    if (!argumentsValid || numRows == 0 || sparsity == 0 || iterations == 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Matrix size, sparsity, and iteration count must be positive.\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    const std::uint64_t matrixEntries = static_cast<std::uint64_t>(numRows) * numRows;
    const std::uint64_t nnz64 = matrixEntries / sparsity;
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nnz64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "This MPI build supports at most INT_MAX rows and nonzero entries.\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nnz64);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortMpi("The hybrid benchmark requires at least one CUDA GPU per node");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    if (worldRank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
                    100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "%d CUDA device(s)/node\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("Initializing data structures...\n");
    }

    std::vector<double> values;
    std::vector<index_t> columns;
    std::vector<index_t> rowDelimiters;
    std::vector<double> vector(numRows);
    std::vector<double> output;
    std::vector<double> reference;

    if (worldRank == 0) {
        values.resize(nItems);
        columns.resize(nItems);
        rowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);
        output.resize(numRows);
        fill(vector.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rowDelimiters.data(), nItems, numRows);

        if (validate) {
            std::printf("Computing OpenMP reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), columns.data(), rowDelimiters.data(), vector.data(),
                    numRows, reference.data());
        }
    }

    MPI_CHECK(MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD));

    std::vector<int> rowStarts;
    std::vector<int> rowCounts;
    std::vector<int> nnzStarts;
    std::vector<int> nnzCounts;
    std::vector<int> delimiterCounts;
    if (worldRank == 0) {
        rowStarts.resize(worldSize + 1);
        rowCounts.resize(worldSize);
        nnzStarts.resize(worldSize);
        nnzCounts.resize(worldSize);
        delimiterCounts.resize(worldSize);
        rowStarts[0] = 0;
        rowStarts[worldSize] = static_cast<int>(numRows);
        for (int rank = 1; rank < worldSize; ++rank) {
            const index_t target = static_cast<index_t>(
                (static_cast<std::uint64_t>(nItems) * rank) / worldSize);
            const auto boundary = std::lower_bound(rowDelimiters.begin(),
                                                   rowDelimiters.end(), target);
            rowStarts[rank] = static_cast<int>(boundary - rowDelimiters.begin());
        }
        for (int rank = 0; rank < worldSize; ++rank) {
            rowCounts[rank] = rowStarts[rank + 1] - rowStarts[rank];
            nnzStarts[rank] = static_cast<int>(rowDelimiters[rowStarts[rank]]);
            nnzCounts[rank] = static_cast<int>(rowDelimiters[rowStarts[rank + 1]]) -
                              nnzStarts[rank];
            delimiterCounts[rank] = rowCounts[rank] + 1;
        }
    }

    int localRows = 0;
    int localNnzStart = 0;
    int localNnz = 0;
    MPI_CHECK(MPI_Scatter(rowCounts.data(), 1, MPI_INT, &localRows, 1, MPI_INT, 0,
                          MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatter(nnzStarts.data(), 1, MPI_INT, &localNnzStart, 1, MPI_INT, 0,
                          MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0,
                          MPI_COMM_WORLD));

    std::vector<double> localValues(localNnz);
    std::vector<index_t> localColumns(localNnz);
    std::vector<index_t> localRowDelimiters(static_cast<std::size_t>(localRows) + 1);
    std::vector<double> localOutput(localRows);

    MPI_CHECK(MPI_Scatterv(values.data(), nnzCounts.data(), nnzStarts.data(), MPI_DOUBLE,
                           localValues.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(columns.data(), nnzCounts.data(), nnzStarts.data(), MPI_UINT32_T,
                           localColumns.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(rowDelimiters.data(), delimiterCounts.data(), rowStarts.data(),
                           MPI_UINT32_T, localRowDelimiters.data(), localRows + 1,
                           MPI_UINT32_T, 0, MPI_COMM_WORLD));

    // Rebase and first-touch the rank-local CSR storage in parallel on the host.
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row <= localRows; ++row) {
        localRowDelimiters[row] -= static_cast<index_t>(localNnzStart);
        if (row < localRows) {
            localOutput[row] = 0.0;
        }
    }

    double* dValues = nullptr;
    index_t* dColumns = nullptr;
    index_t* dRowDelimiters = nullptr;
    double* dVector = nullptr;
    double* dOutput = nullptr;
    const std::size_t allocatedNnz = std::max<std::size_t>(1, localValues.size());
    const std::size_t allocatedRows = std::max<std::size_t>(1, localOutput.size());
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dValues), allocatedNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dColumns), allocatedNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRowDelimiters),
                          localRowDelimiters.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVector),
                          static_cast<std::size_t>(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dOutput), allocatedRows * sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(dValues, localValues.data(), localValues.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dColumns, localColumns.data(),
                               localColumns.size() * sizeof(index_t), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dRowDelimiters, localRowDelimiters.data(),
                               localRowDelimiters.size() * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dVector, vector.data(),
                               static_cast<std::size_t>(numRows) * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warm up the context and selected kernel before the measured region.
    const bool useWarpKernel = static_cast<double>(nItems) / numRows > 16.0;
    launchSpmv(dValues, dColumns, dRowDelimiters, dVector,
               static_cast<index_t>(localRows), useWarpKernel, dOutput, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // A graph removes per-iteration CPU launch latency, which is significant for small
    // rank-local partitions, while retaining every requested SpMV operation.
    cudaGraph_t iterationGraph = nullptr;
    cudaGraphExec_t iterationGraphExec = nullptr;
    if (localRows != 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (index_t iteration = 0; iteration < iterations; ++iteration) {
            launchSpmv(dValues, dColumns, dRowDelimiters, dVector,
                       static_cast<index_t>(localRows), useWarpKernel, dOutput, stream);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &iterationGraph));
        CUDA_CHECK(cudaGraphInstantiate(&iterationGraphExec, iterationGraph, nullptr, nullptr,
                                        0));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (worldRank == 0) {
        std::printf("Computing SpMV...\n");
    }
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    if (localRows != 0) {
        CUDA_CHECK(cudaGraphLaunch(iterationGraphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float localElapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&localElapsedMs, startEvent, stopEvent));
    float elapsedMs = 0.0F;
    MPI_CHECK(MPI_Reduce(&localElapsedMs, &elapsedMs, 1, MPI_FLOAT, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    CUDA_CHECK(cudaMemcpyAsync(localOutput.data(), dOutput,
                               localOutput.size() * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Gatherv(localOutput.data(), localRows, MPI_DOUBLE, output.data(),
                          rowCounts.data(), rowStarts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    int result = 0;
    if (worldRank == 0) {
        const double seconds = static_cast<double>(elapsedMs) / 1000.0;
        const double gflops = (2.0 * static_cast<double>(nItems) * iterations) / seconds / 1e9;
        const double averageMs = elapsedMs / iterations;
        std::printf("Computation time: %.3f ms\n", elapsedMs);
        std::printf("Average time per iteration: %.3f ms\n", averageMs);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(output, "OutputVector");
        }
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), output.data(), numRows);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    if (iterationGraphExec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(iterationGraphExec));
        CUDA_CHECK(cudaGraphDestroy(iterationGraph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dValues));
    CUDA_CHECK(cudaFree(dColumns));
    CUDA_CHECK(cudaFree(dRowDelimiters));
    CUDA_CHECK(cudaFree(dVector));
    CUDA_CHECK(cudaFree(dOutput));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return result;
}
