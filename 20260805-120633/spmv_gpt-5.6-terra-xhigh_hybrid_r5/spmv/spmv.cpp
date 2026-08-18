#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int CUDA_WARP_SIZE = 32;

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    if (rank == 0) {
        std::fprintf(stderr, "Error: %s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        if (rank == 0) {
            std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                         cudaGetErrorString(status));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(call, rank) checkCuda((call), #call, (rank))

void fill(double* values, const index_t count, const double maxVal) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// This retains the benchmark's original, deterministic random matrix layout.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t nnz,
                      const index_t dimension) {
    index_t assigned = 0;
    const std::uint64_t matrixEntries = static_cast<std::uint64_t>(dimension) * dimension;
    const double probability = static_cast<double>(nnz) / static_cast<double>(matrixEntries);

    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        rowDelimiters[row] = assigned;
        for (index_t col = 0; col < dimension; ++col) {
            const std::uint64_t position = static_cast<std::uint64_t>(row) * dimension + col;
            const std::uint64_t entriesLeft = matrixEntries - position;
            const index_t needToAssign = nnz - assigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < nnz && randomValue <= probability) || fillRemaining) {
                cols[assigned++] = col;
            }
        }
    }
    rowDelimiters[dimension] = nnz;
}

void spmvCpu(const double* values, const index_t* columns, const index_t* rowDelimiters,
             const double* vector, const index_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows); ++row) {
        double sum = 0.0;
        for (index_t entry = rowDelimiters[row]; entry < rowDelimiters[row + 1]; ++entry) {
            sum += values[entry] * vector[columns[entry]];
        }
        output[row] = sum;
    }
}

// One warp processes one CSR row.  This keeps the sequential row reduction
// local to a warp while loads of CSR values and column indices are coalesced.
__global__ void spmvCsrKernel(const double* __restrict__ values,
                              const index_t* __restrict__ columns,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vector,
                              const index_t rows, double* __restrict__ output) {
    const int lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
    const index_t warpsPerBlock = blockDim.x / CUDA_WARP_SIZE;
    const index_t row = static_cast<index_t>(blockIdx.x) * warpsPerBlock +
                        threadIdx.x / CUDA_WARP_SIZE;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    for (index_t entry = begin + lane; entry < end; entry += CUDA_WARP_SIZE) {
        sum += values[entry] * vector[columns[entry]];
    }

    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI does not provide the required thread support");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const std::uint64_t matrixEntries = static_cast<std::uint64_t>(numRows) * numRows;
    const std::uint64_t nnz64 = sparsity == 0 ? 0 : matrixEntries / sparsity;
    if (numRows == 0 || sparsity == 0 || numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nnz64 > static_cast<std::uint64_t>(std::numeric_limits<index_t>::max()) ||
        nnz64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(rank, "matrix dimensions or non-zero count exceed supported MPI/CSR limits");
    }
    const index_t nonZeros = static_cast<index_t>(nnz64);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        abortWithMessage(rank, "no CUDA device is visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonZeros,
                    100.0 * (1.0 - static_cast<double>(nonZeros) / matrixEntries));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads per rank: %d, CUDA devices per node: %d\n",
                    worldSize, omp_get_max_threads(), deviceCount);
    }

    const index_t rowBegin = static_cast<index_t>(
        static_cast<std::uint64_t>(numRows) * static_cast<unsigned int>(rank) / worldSize);
    const index_t rowEnd = static_cast<index_t>(
        static_cast<std::uint64_t>(numRows) * static_cast<unsigned int>(rank + 1) / worldSize);
    const index_t localRows = rowEnd - rowBegin;

    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> denseVector(numRows);
    std::vector<double> reference;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> valueCounts;
    std::vector<int> valueDisplacements;

    if (rank == 0) {
        globalValues.resize(nonZeros);
        globalColumns.resize(nonZeros);
        globalRowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);
        rowCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        valueCounts.resize(worldSize);
        valueDisplacements.resize(worldSize);

        std::printf("Initializing data structures...\n");
        fill(denseVector.data(), numRows, maxVal);
        fill(globalValues.data(), nonZeros, maxVal);
        initRandomMatrix(globalColumns.data(), globalRowDelimiters.data(), nonZeros, numRows);

        for (int process = 0; process < worldSize; ++process) {
            const index_t begin = static_cast<index_t>(
                static_cast<std::uint64_t>(numRows) * static_cast<unsigned int>(process) / worldSize);
            const index_t end = static_cast<index_t>(
                static_cast<std::uint64_t>(numRows) * static_cast<unsigned int>(process + 1) / worldSize);
            rowCounts[process] = static_cast<int>(end - begin);
            rowDisplacements[process] = static_cast<int>(begin);
            valueDisplacements[process] = static_cast<int>(globalRowDelimiters[begin]);
            valueCounts[process] = static_cast<int>(globalRowDelimiters[end] -
                                                    globalRowDelimiters[begin]);
        }

        if (validate) {
            std::printf("Computing OpenMP reference solution...\n");
            reference.resize(numRows);
            spmvCpu(globalValues.data(), globalColumns.data(), globalRowDelimiters.data(),
                    denseVector.data(), numRows, reference.data());
        }
    }

    MPI_Bcast(denseVector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localNonZerosInt = 0;
    int localNnzDisplacement = 0;
    MPI_Scatter(rank == 0 ? valueCounts.data() : nullptr, 1, MPI_INT, &localNonZerosInt, 1,
                MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? valueDisplacements.data() : nullptr, 1, MPI_INT,
                &localNnzDisplacement, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNonZeros = static_cast<index_t>(localNonZerosInt);

    std::vector<index_t> globalLocalRowStarts(localRows);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 globalLocalRowStarts.data(), static_cast<int>(localRows), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    std::vector<double> localValues(localNonZeros);
    std::vector<index_t> localColumns(localNonZeros);
    MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                 rank == 0 ? valueCounts.data() : nullptr,
                 rank == 0 ? valueDisplacements.data() : nullptr, MPI_DOUBLE,
                 localValues.data(), localNonZerosInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalColumns.data() : nullptr,
                 rank == 0 ? valueCounts.data() : nullptr,
                 rank == 0 ? valueDisplacements.data() : nullptr, MPI_UINT32_T,
                 localColumns.data(), localNonZerosInt, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // OpenMP is used by every rank to normalize its local CSR row offsets.
    std::vector<index_t> localRowDelimiters(static_cast<std::size_t>(localRows) + 1);
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(localRows); ++row) {
        localRowDelimiters[row] = globalLocalRowStarts[row] -
                                  static_cast<index_t>(localNnzDisplacement);
    }
    localRowDelimiters[localRows] = localNonZeros;

    // The root no longer needs its full matrix after distribution.
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRowDelimiters);
    std::vector<index_t>().swap(globalLocalRowStarts);

    double* deviceValues = nullptr;
    index_t* deviceColumns = nullptr;
    index_t* deviceRowDelimiters = nullptr;
    double* deviceVector = nullptr;
    double* deviceOutput = nullptr;
    const std::size_t valueBytes = std::max<std::size_t>(localNonZeros, 1) * sizeof(double);
    const std::size_t columnBytes = std::max<std::size_t>(localNonZeros, 1) * sizeof(index_t);
    const std::size_t rowBytes = (static_cast<std::size_t>(localRows) + 1) * sizeof(index_t);
    const std::size_t vectorBytes = static_cast<std::size_t>(numRows) * sizeof(double);
    const std::size_t outputBytes = std::max<std::size_t>(localRows, 1) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&deviceValues, valueBytes), rank);
    CUDA_CHECK(cudaMalloc(&deviceColumns, columnBytes), rank);
    CUDA_CHECK(cudaMalloc(&deviceRowDelimiters, rowBytes), rank);
    CUDA_CHECK(cudaMalloc(&deviceVector, vectorBytes), rank);
    CUDA_CHECK(cudaMalloc(&deviceOutput, outputBytes), rank);
    if (localNonZeros != 0) {
        CUDA_CHECK(cudaMemcpy(deviceValues, localValues.data(), static_cast<std::size_t>(localNonZeros) * sizeof(double),
                              cudaMemcpyHostToDevice), rank);
        CUDA_CHECK(cudaMemcpy(deviceColumns, localColumns.data(), static_cast<std::size_t>(localNonZeros) * sizeof(index_t),
                              cudaMemcpyHostToDevice), rank);
    }
    CUDA_CHECK(cudaMemcpy(deviceRowDelimiters, localRowDelimiters.data(), rowBytes,
                          cudaMemcpyHostToDevice), rank);
    CUDA_CHECK(cudaMemcpy(deviceVector, denseVector.data(), vectorBytes, cudaMemcpyHostToDevice), rank);
    CUDA_CHECK(cudaMemset(deviceOutput, 0, outputBytes), rank);

    const int warpsPerBlock = CUDA_THREADS_PER_BLOCK / CUDA_WARP_SIZE;
    const int gridBlocks = static_cast<int>((static_cast<std::uint64_t>(localRows) + warpsPerBlock - 1) /
                                            warpsPerBlock);
    if (iterations != 0 && gridBlocks != 0) {
        spmvCsrKernel<<<gridBlocks, CUDA_THREADS_PER_BLOCK>>>(
            deviceValues, deviceColumns, deviceRowDelimiters, deviceVector, localRows, deviceOutput);
        CUDA_CHECK(cudaGetLastError(), rank);
        CUDA_CHECK(cudaDeviceSynchronize(), rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CUDA_CHECK(cudaEventCreate(&startEvent), rank);
    CUDA_CHECK(cudaEventCreate(&stopEvent), rank);
    CUDA_CHECK(cudaEventRecord(startEvent), rank);
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        if (gridBlocks != 0) {
            spmvCsrKernel<<<gridBlocks, CUDA_THREADS_PER_BLOCK>>>(
                deviceValues, deviceColumns, deviceRowDelimiters, deviceVector, localRows, deviceOutput);
        }
    }
    CUDA_CHECK(cudaGetLastError(), rank);
    CUDA_CHECK(cudaEventRecord(stopEvent), rank);
    CUDA_CHECK(cudaEventSynchronize(stopEvent), rank);

    float localElapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&localElapsedMs, startEvent, stopEvent), rank);
    double elapsedMs = static_cast<double>(localElapsedMs);
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventDestroy(startEvent), rank);
    CUDA_CHECK(cudaEventDestroy(stopEvent), rank);

    std::vector<double> localOutput(localRows, 0.0);
    if (localRows != 0) {
        CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput, static_cast<std::size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost), rank);
    }

    std::vector<double> output;
    if (validate || printResults) {
        if (rank == 0) {
            output.resize(numRows);
        }
        MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? output.data() : nullptr,
                    rank == 0 ? rowCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(deviceValues), rank);
    CUDA_CHECK(cudaFree(deviceColumns), rank);
    CUDA_CHECK(cudaFree(deviceRowDelimiters), rank);
    CUDA_CHECK(cudaFree(deviceVector), rank);
    CUDA_CHECK(cudaFree(deviceOutput), rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMs);
        const double averageTimeMs = iterations == 0 ? 0.0 : elapsedMs / iterations;
        const double gflops = elapsedMs == 0.0 ? 0.0 :
            (2.0 * static_cast<double>(nonZeros) * iterations) / (elapsedMs / 1000.0) / 1.0e9;
        std::printf("Average time per iteration: %.3f ms\n", averageTimeMs);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(output, "OutputVector");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
            if (verifyResults(reference.data(), output.data(), numRows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
