#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int ROW_POINTER_TAG = 100;
constexpr int COLUMN_TAG = 101;
constexpr int VALUE_TAG = 102;
constexpr int OUTPUT_TAG = 103;

[[noreturn]] void abortWithMessage(const char* subsystem, const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s error: %s\n", rank, subsystem, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const char* expression) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING] = {};
        int messageLength = 0;
        MPI_Error_string(status, message, &messageLength);
        abortWithMessage("MPI", messageLength == 0 ? expression : message);
    }
}

void checkCuda(const cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        char message[256] = {};
        std::snprintf(message, sizeof(message), "%s: %s", expression, cudaGetErrorString(status));
        abortWithMessage("CUDA", message);
    }
}

void checkCusparse(const cusparseStatus_t status, const char* expression) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        char message[96] = {};
        std::snprintf(message, sizeof(message), "%s (status %d)", expression,
                      static_cast<int>(status));
        abortWithMessage("cuSPARSE", message);
    }
}

#define MPI_CHECK(call) checkMpi((call), #call)
#define CUDA_CHECK(call) checkCuda((call), #call)
#define CUSPARSE_CHECK(call) checkCusparse((call), #call)

// Keep the original random stream and matrix construction so that the input
// data, including the externally printed result, has the same semantics.
void fill(double* values, const index_t count, const double maxValue) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxValue * (std::rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* columns, index_t* rowPointers, const index_t nonZeros,
                      const index_t dimension) {
    index_t assigned = 0;
    const std::uint64_t matrixElements = static_cast<std::uint64_t>(dimension) * dimension;
    const double probability = static_cast<double>(nonZeros) /
                               static_cast<double>(matrixElements);

    std::srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        rowPointers[row] = assigned;
        for (index_t column = 0; column < dimension; ++column) {
            const std::uint64_t position = static_cast<std::uint64_t>(row) * dimension + column;
            const std::uint64_t entriesLeft = matrixElements - position;
            const index_t needToAssign = nonZeros - assigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
            if ((assigned < nonZeros && randomValue <= probability) || fillRemaining) {
                columns[assigned++] = column;
            }
        }
    }
    rowPointers[dimension] = nonZeros;
}

void spmvCpu(const double* values, const index_t* columns, const index_t* rowPointers,
             const double* vector, const index_t rows, double* output) {
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (index_t entry = rowPointers[row]; entry < rowPointers[row + 1]; ++entry) {
            sum += values[entry] * vector[columns[entry]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double expected = reference[i];
        const double actual = result[i];
        if (std::abs(expected) < 1e-10) {
            if (std::abs(actual) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i,
                            expected, actual);
                return false;
            }
        } else {
            const double relativeError = std::abs((actual - expected) / expected);
            if (relativeError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, expected, actual, relativeError);
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

struct Options {
    index_t rows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxValue = 1.0;
    bool validate = false;
    bool printResults = false;
};

bool parseOptions(const int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            options->rows = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            options->sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options->iterations = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            options->maxValue = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options->validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options->printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            return false;
        }
    }
    return true;
}

template <typename T>
void mpiBroadcastLarge(T* data, std::uint64_t count, const MPI_Datatype type, const int root) {
    constexpr std::uint64_t maxCount = std::numeric_limits<int>::max();
    for (std::uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(maxCount, count - offset));
        MPI_CHECK(MPI_Bcast(data + offset, chunk, type, root, MPI_COMM_WORLD));
        offset += static_cast<std::uint64_t>(chunk);
    }
}

template <typename T>
void mpiSendLarge(const T* data, std::uint64_t count, const MPI_Datatype type, const int destination,
                  const int tag) {
    constexpr std::uint64_t maxCount = std::numeric_limits<int>::max();
    for (std::uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(maxCount, count - offset));
        MPI_CHECK(MPI_Send(data + offset, chunk, type, destination, tag, MPI_COMM_WORLD));
        offset += static_cast<std::uint64_t>(chunk);
    }
}

template <typename T>
void mpiReceiveLarge(T* data, std::uint64_t count, const MPI_Datatype type, const int source,
                     const int tag) {
    constexpr std::uint64_t maxCount = std::numeric_limits<int>::max();
    for (std::uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(maxCount, count - offset));
        MPI_CHECK(MPI_Recv(data + offset, chunk, type, source, tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
        offset += static_cast<std::uint64_t>(chunk);
    }
}

std::vector<index_t> buildRowPartitions(const std::vector<index_t>& rowPointers,
                                        const index_t rows, const index_t nonZeros,
                                        const int ranks) {
    std::vector<index_t> partitions(static_cast<std::size_t>(ranks) + 1);
    partitions.front() = 0;
    partitions.back() = rows;
    for (int rank = 1; rank < ranks; ++rank) {
        const index_t target = static_cast<index_t>(
            (static_cast<std::uint64_t>(nonZeros) * rank) / static_cast<unsigned>(ranks));
        partitions[rank] = static_cast<index_t>(
            std::lower_bound(rowPointers.begin(), rowPointers.end(), target) - rowPointers.begin());
    }
    return partitions;
}

template <typename T>
T* allocateDevice(const std::size_t elements) {
    T* pointer = nullptr;
    // CUDA does not guarantee a useful result for a zero-byte allocation.
    const std::size_t allocationSize = std::max<std::size_t>(elements, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), allocationSize * sizeof(T)));
    return pointer;
}

int selectLocalGpu() {
    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                 &nodeCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage("CUDA", "no CUDA device is visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_CHECK(MPI_Comm_free(&nodeCommunicator));
    return device;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI", "MPI_THREAD_FUNNELED is not available");
    }

    Options options;
    const bool parsed = parseOptions(argc, argv, &options);
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        showHelp = showHelp || std::strcmp(argv[i], "-h") == 0;
    }
    if (!parsed) {
        if (rank == 0) {
            if (!showHelp) {
                std::printf("Invalid command line arguments.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const std::uint64_t matrixElements = static_cast<std::uint64_t>(options.rows) * options.rows;
    if (options.rows == 0 || options.sparsity == 0 || options.iterations == 0 ||
        matrixElements / options.sparsity > std::numeric_limits<std::int32_t>::max() ||
        options.rows > static_cast<index_t>(std::numeric_limits<std::int32_t>::max())) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix dimensions, sparsity, or iteration count are unsupported.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const index_t nonZeros = static_cast<index_t>(matrixElements / options.sparsity);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", options.rows, options.rows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", options.sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonZeros,
                    100.0 * (1.0 - static_cast<double>(nonZeros) / matrixElements));
        std::printf("Iterations: %u\n", options.iterations);
        std::printf("Max value: %.2f\n", options.maxValue);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Computing with MPI-distributed cuSPARSE on CUDA devices\n");
    }

    std::vector<double> vector(options.rows);
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowPointers;
    std::vector<double> reference;
    std::vector<index_t> rowPartitions(static_cast<std::size_t>(ranks) + 1);

    if (rank == 0) {
        std::printf("Initializing data structures...\n");
        globalValues.resize(nonZeros);
        globalColumns.resize(nonZeros);
        globalRowPointers.resize(static_cast<std::size_t>(options.rows) + 1);
        fill(vector.data(), options.rows, options.maxValue);
        fill(globalValues.data(), nonZeros, options.maxValue);
        initRandomMatrix(globalColumns.data(), globalRowPointers.data(), nonZeros, options.rows);
        rowPartitions = buildRowPartitions(globalRowPointers, options.rows, nonZeros, ranks);

        if (options.validate) {
            std::printf("Computing reference solution...\n");
            reference.resize(options.rows);
            spmvCpu(globalValues.data(), globalColumns.data(), globalRowPointers.data(), vector.data(),
                    options.rows, reference.data());
        }
    }

    mpiBroadcastLarge(vector.data(), options.rows, MPI_DOUBLE, 0);
    mpiBroadcastLarge(rowPartitions.data(), rowPartitions.size(), MPI_UINT32_T, 0);

    const index_t firstRow = rowPartitions[rank];
    const index_t lastRow = rowPartitions[rank + 1];
    const index_t localRows = lastRow - firstRow;
    std::vector<index_t> localRowPointers(static_cast<std::size_t>(localRows) + 1);

    if (rank == 0) {
        std::copy_n(globalRowPointers.data() + firstRow, static_cast<std::size_t>(localRows) + 1,
                    localRowPointers.data());
        for (int destination = 1; destination < ranks; ++destination) {
            const index_t begin = rowPartitions[destination];
            const index_t end = rowPartitions[destination + 1];
            const index_t rowCount = end - begin;
            mpiSendLarge(globalRowPointers.data() + begin, static_cast<std::uint64_t>(rowCount) + 1,
                         MPI_UINT32_T, destination, ROW_POINTER_TAG);
        }
    } else {
        mpiReceiveLarge(localRowPointers.data(), static_cast<std::uint64_t>(localRows) + 1, MPI_UINT32_T,
                        0, ROW_POINTER_TAG);
    }

    const index_t localNnz = localRowPointers.back() - localRowPointers.front();
    std::vector<double> localValues(localNnz);
    std::vector<index_t> localColumns(localNnz);
    const index_t firstNnz = localRowPointers.front();

    if (rank == 0) {
        // OpenMP is used on every rank to prepare the device-local CSR.  This
        // avoids serial host-side packing while keeping the original CSR order.
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(localNnz); ++i) {
            localValues[static_cast<std::size_t>(i)] = globalValues[firstNnz + i];
            localColumns[static_cast<std::size_t>(i)] = globalColumns[firstNnz + i];
        }
        for (int destination = 1; destination < ranks; ++destination) {
            const index_t beginRow = rowPartitions[destination];
            const index_t endRow = rowPartitions[destination + 1];
            const index_t beginNnz = globalRowPointers[beginRow];
            const index_t endNnz = globalRowPointers[endRow];
            const index_t count = endNnz - beginNnz;
            mpiSendLarge(globalColumns.data() + beginNnz, count, MPI_UINT32_T, destination, COLUMN_TAG);
            mpiSendLarge(globalValues.data() + beginNnz, count, MPI_DOUBLE, destination, VALUE_TAG);
        }
    } else {
        mpiReceiveLarge(localColumns.data(), localNnz, MPI_UINT32_T, 0, COLUMN_TAG);
        mpiReceiveLarge(localValues.data(), localNnz, MPI_DOUBLE, 0, VALUE_TAG);
    }

    const index_t rowPointerOffset = localRowPointers.front();
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row <= static_cast<std::int64_t>(localRows); ++row) {
        localRowPointers[static_cast<std::size_t>(row)] -= rowPointerOffset;
    }

    // The global CSR is no longer needed after each rank has its local slice.
    globalValues.clear();
    globalValues.shrink_to_fit();
    globalColumns.clear();
    globalColumns.shrink_to_fit();
    globalRowPointers.clear();
    globalRowPointers.shrink_to_fit();

    const int selectedDevice = selectLocalGpu();
    (void)selectedDevice;

    double* deviceValues = allocateDevice<double>(localValues.size());
    index_t* deviceColumns = allocateDevice<index_t>(localColumns.size());
    index_t* deviceRowPointers = allocateDevice<index_t>(localRowPointers.size());
    double* deviceVector = allocateDevice<double>(vector.size());
    double* deviceOutput = allocateDevice<double>(localRows);

    if (localNnz != 0) {
        CUDA_CHECK(cudaMemcpy(deviceValues, localValues.data(), localValues.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceColumns, localColumns.data(),
                              localColumns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(deviceRowPointers, localRowPointers.data(),
                          localRowPointers.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVector, vector.data(), vector.size() * sizeof(double),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t input = nullptr;
    cusparseDnVecDescr_t output = nullptr;
    void* workspace = nullptr;
    const double alpha = 1.0;
    const double beta = 0.0;
    constexpr cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;

    if (localRows != 0) {
        CUSPARSE_CHECK(cusparseCreate(&handle));
        CUSPARSE_CHECK(cusparseSetStream(handle, stream));
        CUSPARSE_CHECK(cusparseCreateCsr(&matrix, localRows, options.rows, localNnz, deviceRowPointers,
                                         deviceColumns, deviceValues, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&input, options.rows, deviceVector, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&output, localRows, deviceOutput, CUDA_R_64F));

        std::size_t workspaceSize = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, operation, &alpha, matrix, input, &beta, output,
                                               CUDA_R_64F, algorithm, &workspaceSize));
        if (workspaceSize != 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
        }
        // Prime the selected cuSPARSE algorithm before the timed region.
        CUSPARSE_CHECK(cusparseSpMV(handle, operation, &alpha, matrix, input, &beta, output, CUDA_R_64F,
                                    algorithm, workspace));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    float localMilliseconds = 0.0F;
    if (localRows != 0) {
        CUDA_CHECK(cudaEventRecord(startEvent, stream));
        for (index_t iteration = 0; iteration < options.iterations; ++iteration) {
            CUSPARSE_CHECK(cusparseSpMV(handle, operation, &alpha, matrix, input, &beta, output,
                                        CUDA_R_64F, algorithm, workspace));
        }
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        CUDA_CHECK(cudaEventElapsedTime(&localMilliseconds, startEvent, stopEvent));
    }

    float maximumMilliseconds = 0.0F;
    MPI_CHECK(MPI_Reduce(&localMilliseconds, &maximumMilliseconds, 1, MPI_FLOAT, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    std::vector<double> localOutput(localRows);
    std::vector<double> outputVector;
    if (options.validate || options.printResults) {
        if (localRows != 0) {
            CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput, localOutput.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        if (rank == 0) {
            outputVector.resize(options.rows);
            std::copy(localOutput.begin(), localOutput.end(), outputVector.begin() + firstRow);
            for (int source = 1; source < ranks; ++source) {
                const index_t begin = rowPartitions[source];
                const index_t end = rowPartitions[source + 1];
                mpiReceiveLarge(outputVector.data() + begin, end - begin, MPI_DOUBLE, source, OUTPUT_TAG);
            }
        } else {
            mpiSendLarge(localOutput.data(), localRows, MPI_DOUBLE, 0, OUTPUT_TAG);
        }
    }

    if (rank == 0) {
        const double elapsedMilliseconds = std::max(static_cast<double>(maximumMilliseconds), 1.0e-9);
        const double gflops = (2.0 * static_cast<double>(nonZeros) * options.iterations) /
                              (elapsedMilliseconds * 1.0e6);
        const double averageMilliseconds = elapsedMilliseconds / options.iterations;
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        std::printf("Average time per iteration: %.3f ms\n", averageMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (options.printResults) {
            print_results(outputVector, "OutputVector");
        }
    }

    int validationStatus = EXIT_SUCCESS;
    if (options.validate && rank == 0) {
        std::printf("Validating result...\n");
        if (verifyResults(reference.data(), outputVector.data(), options.rows)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            validationStatus = EXIT_FAILURE;
        }
    }
    MPI_CHECK(MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (matrix != nullptr) {
        CUSPARSE_CHECK(cusparseDestroySpMat(matrix));
        CUSPARSE_CHECK(cusparseDestroyDnVec(input));
        CUSPARSE_CHECK(cusparseDestroyDnVec(output));
        CUSPARSE_CHECK(cusparseDestroy(handle));
    }
    if (workspace != nullptr) {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceValues));
    CUDA_CHECK(cudaFree(deviceColumns));
    CUDA_CHECK(cudaFree(deviceRowPointers));
    CUDA_CHECK(cudaFree(deviceVector));
    CUDA_CHECK(cudaFree(deviceOutput));

    MPI_CHECK(MPI_Finalize());
    return validationStatus;
}
