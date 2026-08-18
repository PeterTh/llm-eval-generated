#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
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
constexpr std::size_t MPI_CHUNK_SIZE = 1U << 30;
constexpr cusparseSpMVAlg_t SPMV_ALGORITHM = CUSPARSE_SPMV_CSR_ALG1;

int worldRank = 0;

[[noreturn]] void abortMpi(const char* operation, const char* message) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", worldRank, operation, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, error, &length);
    abortMpi(operation, error);
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        abortMpi(operation, cudaGetErrorString(status));
    }
}

void checkCusparse(const cusparseStatus_t status, const char* operation) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        abortMpi(operation, cusparseGetErrorString(status));
    }
}

#define MPI_CHECK(call) checkMpi((call), #call)
#define CUDA_CHECK(call) checkCuda((call), #call)
#define CUSPARSE_CHECK(call) checkCusparse((call), #call)

template <typename T>
MPI_Datatype mpiDatatype();

template <>
MPI_Datatype mpiDatatype<index_t>() {
    return MPI_UINT32_T;
}

template <>
MPI_Datatype mpiDatatype<double>() {
    return MPI_DOUBLE;
}

template <typename T>
void mpiSendLarge(const T* data, const std::size_t count, const int destination, const int tag) {
    std::size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min(MPI_CHUNK_SIZE, count - offset));
        MPI_CHECK(MPI_Send(data + offset, chunk, mpiDatatype<T>(), destination, tag, MPI_COMM_WORLD));
        offset += static_cast<std::size_t>(chunk);
    }
}

template <typename T>
void mpiRecvLarge(T* data, const std::size_t count, const int source, const int tag) {
    std::size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min(MPI_CHUNK_SIZE, count - offset));
        MPI_CHECK(MPI_Recv(data + offset, chunk, mpiDatatype<T>(), source, tag,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += static_cast<std::size_t>(chunk);
    }
}

void fill(double* values, const std::size_t count, const double maxValue) {
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Preserve the benchmark's deterministic matrix construction while using
// 64-bit arithmetic for the position counters to avoid intermediate overflow.
void initRandomMatrix(index_t* columns, index_t* rowOffsets, const index_t nonzeros,
                      const index_t dimension) {
    index_t assigned = 0;
    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(dimension) * static_cast<std::uint64_t>(dimension);
    const double probability = static_cast<double>(nonzeros) / static_cast<double>(matrixEntries);

    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        rowOffsets[row] = assigned;
        for (index_t column = 0; column < dimension; ++column) {
            const std::uint64_t position =
                static_cast<std::uint64_t>(row) * dimension + column;
            const std::uint64_t entriesLeft = matrixEntries - position;
            const std::uint64_t needed = static_cast<std::uint64_t>(nonzeros - assigned);
            if (entriesLeft <= needed) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < nonzeros && randomValue <= probability) || fillRemaining) {
                columns[assigned++] = column;
            }
        }
    }
    rowOffsets[dimension] = nonzeros;
}

void spmvCpu(const double* values, const index_t* columns, const index_t* rowOffsets,
             const double* vector, const index_t dimension, double* output) {
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(dimension); ++row) {
        double sum = 0.0;
        for (index_t item = rowOffsets[row]; item < rowOffsets[row + 1]; ++item) {
            sum += values[item] * vector[columns[item]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
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

bool parseIndex(const char* text, index_t& value) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

std::vector<index_t> makeRowPartitions(const std::vector<index_t>& rowOffsets,
                                       const index_t rows, const index_t nonzeros,
                                       const int ranks) {
    std::vector<index_t> boundaries(static_cast<std::size_t>(ranks) + 1);
    boundaries.front() = 0;
    boundaries.back() = rows;

    if (nonzeros == 0) {
        for (int rank = 1; rank < ranks; ++rank) {
            boundaries[rank] = static_cast<index_t>(
                (static_cast<std::uint64_t>(rows) * rank) / ranks);
        }
        return boundaries;
    }

#pragma omp parallel for schedule(static)
    for (int rank = 1; rank < ranks; ++rank) {
        const std::uint64_t target =
            (static_cast<std::uint64_t>(nonzeros) * static_cast<std::uint64_t>(rank)) /
            static_cast<std::uint64_t>(ranks);
        const auto position = std::lower_bound(rowOffsets.begin(), rowOffsets.end(),
                                               static_cast<index_t>(target));
        boundaries[rank] = static_cast<index_t>(position - rowOffsets.begin());
    }
    return boundaries;
}

void distributeCsr(const std::vector<double>& globalValues,
                   const std::vector<index_t>& globalColumns,
                   const std::vector<index_t>& globalRowOffsets,
                   const std::vector<index_t>& rowBoundaries,
                   const std::vector<index_t>& nonzeroBoundaries,
                   std::vector<double>& localValues,
                   std::vector<index_t>& localColumns,
                   std::vector<index_t>& localRowOffsets,
                   const int ranks) {
    const index_t firstRow = rowBoundaries[worldRank];
    const index_t localRows = rowBoundaries[worldRank + 1] - firstRow;
    const index_t firstNonzero = nonzeroBoundaries[worldRank];
    const index_t localNonzeros = nonzeroBoundaries[worldRank + 1] - firstNonzero;

    localValues.resize(localNonzeros);
    localColumns.resize(localNonzeros);
    localRowOffsets.resize(static_cast<std::size_t>(localRows) + 1);

    constexpr int ROW_TAG = 101;
    constexpr int COLUMN_TAG = 102;
    constexpr int VALUE_TAG = 103;

    if (worldRank == 0) {
        std::copy_n(globalRowOffsets.data() + firstRow,
                    static_cast<std::size_t>(localRows) + 1, localRowOffsets.data());
        if (localNonzeros != 0) {
            std::copy_n(globalColumns.data() + firstNonzero, localNonzeros,
                        localColumns.data());
            std::copy_n(globalValues.data() + firstNonzero, localNonzeros,
                        localValues.data());
        }

        for (int rank = 1; rank < ranks; ++rank) {
            const index_t rankFirstRow = rowBoundaries[rank];
            const std::size_t rankRows = rowBoundaries[rank + 1] - rankFirstRow;
            const index_t rankFirstNonzero = nonzeroBoundaries[rank];
            const std::size_t rankNonzeros =
                nonzeroBoundaries[rank + 1] - rankFirstNonzero;
            mpiSendLarge(globalRowOffsets.data() + rankFirstRow, rankRows + 1,
                         rank, ROW_TAG);
            if (rankNonzeros != 0) {
                mpiSendLarge(globalColumns.data() + rankFirstNonzero, rankNonzeros,
                             rank, COLUMN_TAG);
                mpiSendLarge(globalValues.data() + rankFirstNonzero, rankNonzeros,
                             rank, VALUE_TAG);
            }
        }
    } else {
        mpiRecvLarge(localRowOffsets.data(), static_cast<std::size_t>(localRows) + 1,
                     0, ROW_TAG);
        if (localNonzeros != 0) {
            mpiRecvLarge(localColumns.data(), localNonzeros, 0, COLUMN_TAG);
            mpiRecvLarge(localValues.data(), localNonzeros, 0, VALUE_TAG);
        }
    }

    const index_t base = localRowOffsets.front();
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0;
         row <= static_cast<std::int64_t>(localRows); ++row) {
        localRowOffsets[static_cast<std::size_t>(row)] -= base;
    }
}

class GpuSpmv {
public:
    GpuSpmv(const std::vector<double>& values,
            const std::vector<index_t>& columns,
            const std::vector<index_t>& rowOffsets,
            const std::vector<double>& vector,
            const index_t rows, const index_t columnsCount)
        : rows_(rows), nonzeros_(static_cast<index_t>(values.size())) {
        if (rows_ == 0) {
            return;
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUSPARSE_CHECK(cusparseCreate(&handle_));
        CUSPARSE_CHECK(cusparseSetStream(handle_, stream_));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceValues_),
                              std::max<std::size_t>(1, values.size()) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceColumns_),
                              std::max<std::size_t>(1, columns.size()) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRowOffsets_),
                              rowOffsets.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVector_),
                              vector.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput_),
                              static_cast<std::size_t>(rows_) * sizeof(double)));

        if (nonzeros_ != 0) {
            CUDA_CHECK(cudaMemcpyAsync(deviceValues_, values.data(),
                                       values.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(deviceColumns_, columns.data(),
                                       columns.size() * sizeof(index_t),
                                       cudaMemcpyHostToDevice, stream_));
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceRowOffsets_, rowOffsets.data(),
                                   rowOffsets.size() * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(deviceVector_, vector.data(),
                                   vector.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream_));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &matrix_, static_cast<std::int64_t>(rows_),
            static_cast<std::int64_t>(columnsCount),
            static_cast<std::int64_t>(nonzeros_), deviceRowOffsets_, deviceColumns_,
            deviceValues_, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &input_, static_cast<std::int64_t>(columnsCount), deviceVector_, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &output_, static_cast<std::int64_t>(rows_), deviceOutput_, CUDA_R_64F));

        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, input_,
            &beta_, output_, CUDA_R_64F, SPMV_ALGORITHM, &workspaceSize_));
        if (workspaceSize_ != 0) {
            CUDA_CHECK(cudaMalloc(&workspace_, workspaceSize_));
        }
        synchronize();
    }

    GpuSpmv(const GpuSpmv&) = delete;
    GpuSpmv& operator=(const GpuSpmv&) = delete;

    ~GpuSpmv() {
        if (rows_ == 0) {
            return;
        }
        cusparseDestroyDnVec(output_);
        cusparseDestroyDnVec(input_);
        cusparseDestroySpMat(matrix_);
        cudaFree(workspace_);
        cudaFree(deviceOutput_);
        cudaFree(deviceVector_);
        cudaFree(deviceRowOffsets_);
        cudaFree(deviceColumns_);
        cudaFree(deviceValues_);
        cusparseDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void multiply() {
        if (rows_ == 0) {
            return;
        }
        CUSPARSE_CHECK(cusparseSpMV(
            handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matrix_, input_,
            &beta_, output_, CUDA_R_64F, SPMV_ALGORITHM, workspace_));
    }

    void synchronize() const {
        if (rows_ != 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    void copyOutput(std::vector<double>& output) const {
        output.resize(rows_);
        if (rows_ == 0) {
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(output.data(), deviceOutput_,
                                   static_cast<std::size_t>(rows_) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        synchronize();
    }

private:
    index_t rows_ = 0;
    index_t nonzeros_ = 0;
    cudaStream_t stream_ = nullptr;
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t matrix_ = nullptr;
    cusparseDnVecDescr_t input_ = nullptr;
    cusparseDnVecDescr_t output_ = nullptr;
    double* deviceValues_ = nullptr;
    index_t* deviceColumns_ = nullptr;
    index_t* deviceRowOffsets_ = nullptr;
    double* deviceVector_ = nullptr;
    double* deviceOutput_ = nullptr;
    void* workspace_ = nullptr;
    std::size_t workspaceSize_ = 0;
    double alpha_ = 1.0;
    double beta_ = 0.0;
};

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));

    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMpi("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable");
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
            argumentsValid = parseIndex(argv[++i], numRows) && argumentsValid;
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            argumentsValid = parseIndex(argv[++i], sparsity) && argumentsValid;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid = parseIndex(argv[++i], iterations) && argumentsValid;
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            maxValue = std::strtod(argv[++i], &end);
            argumentsValid = errno == 0 && *end == '\0' && std::isfinite(maxValue) &&
                             maxValue >= 0.0 && argumentsValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (help || !argumentsValid || numRows == 0 || sparsity == 0 || iterations == 0 ||
        numRows > static_cast<index_t>(INT_MAX)) {
        if (worldRank == 0) {
            if (!help && argumentsValid) {
                std::printf("Matrix size, sparsity, and iterations must be positive; matrix "
                            "size must not exceed INT_MAX.\n");
            }
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(numRows) * static_cast<std::uint64_t>(numRows);
    const std::uint64_t nonzeros64 = matrixEntries / sparsity;
    if (nonzeros64 > std::numeric_limits<index_t>::max()) {
        if (worldRank == 0) {
            std::fprintf(stderr, "The requested matrix exceeds the 32-bit CSR index range.\n");
        }
        MPI_CHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }
    const index_t nonzeros = static_cast<index_t>(nonzeros64);

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &nodeCommunicator));
    int nodeRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeCommunicator, &nodeRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortMpi("cudaGetDeviceCount", "no CUDA accelerator is available");
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    if (worldRank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonzeros,
                    100.0 * (1.0 - static_cast<double>(nonzeros) /
                                      static_cast<double>(matrixEntries)));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxValue);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "CUDA cuSPARSE\n", ranks, omp_get_max_threads());
        std::printf("Rank 0 accelerator: CUDA device %d (%s)\n", device,
                    deviceProperties.name);
        std::printf("Initializing data structures...\n");
    }

    std::vector<double> vector(numRows);
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowOffsets;
    if (worldRank == 0) {
        globalValues.resize(nonzeros);
        globalColumns.resize(nonzeros);
        globalRowOffsets.resize(static_cast<std::size_t>(numRows) + 1);
        fill(vector.data(), numRows, maxValue);
        fill(globalValues.data(), nonzeros, maxValue);
        initRandomMatrix(globalColumns.data(), globalRowOffsets.data(), nonzeros, numRows);
    }
    MPI_CHECK(MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD));

    std::vector<double> reference;
    if (validate && worldRank == 0) {
        std::printf("Computing reference solution with OpenMP...\n");
        reference.resize(numRows);
        spmvCpu(globalValues.data(), globalColumns.data(), globalRowOffsets.data(),
                vector.data(), numRows, reference.data());
    }

    std::vector<index_t> rowBoundaries(static_cast<std::size_t>(ranks) + 1);
    std::vector<index_t> nonzeroBoundaries(static_cast<std::size_t>(ranks) + 1);
    if (worldRank == 0) {
        rowBoundaries = makeRowPartitions(globalRowOffsets, numRows, nonzeros, ranks);
        for (int rank = 0; rank <= ranks; ++rank) {
            nonzeroBoundaries[rank] = globalRowOffsets[rowBoundaries[rank]];
        }
    }
    MPI_CHECK(MPI_Bcast(rowBoundaries.data(), ranks + 1, MPI_UINT32_T, 0,
                        MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(nonzeroBoundaries.data(), ranks + 1, MPI_UINT32_T, 0,
                        MPI_COMM_WORLD));

    std::vector<double> localValues;
    std::vector<index_t> localColumns;
    std::vector<index_t> localRowOffsets;
    distributeCsr(globalValues, globalColumns, globalRowOffsets, rowBoundaries,
                  nonzeroBoundaries, localValues, localColumns, localRowOffsets, ranks);

    // The root no longer needs the global CSR once validation and distribution finish.
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRowOffsets);

    const index_t localRows = rowBoundaries[worldRank + 1] - rowBoundaries[worldRank];
    if (localValues.size() > static_cast<std::size_t>(INT_MAX)) {
        abortMpi("CSR distribution", "a rank received more than INT_MAX nonzeros");
    }

    int localActive = localRows == 0 ? 0 : 1;
    int activeRanks = 0;
    MPI_CHECK(MPI_Reduce(&localActive, &activeRanks, 1, MPI_INT, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Computing SpMV on %d active accelerator(s)...\n", activeRanks);
    }

    int exitCode = EXIT_SUCCESS;
    {
        GpuSpmv gpu(localValues, localColumns, localRowOffsets, vector, localRows, numRows);

        // Warm up the CUDA context, cuSPARSE kernels, and GPU memory before timing.
        gpu.multiply();
        gpu.synchronize();
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        const double start = MPI_Wtime();
        for (index_t iteration = 0; iteration < iterations; ++iteration) {
            gpu.multiply();
        }
        gpu.synchronize();
        const double localSeconds = MPI_Wtime() - start;

        double elapsedSeconds = 0.0;
        MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD));

        std::vector<double> localOutput;
        gpu.copyOutput(localOutput);

        std::vector<double> output;
        std::vector<int> receiveCounts;
        std::vector<int> receiveOffsets;
        if (worldRank == 0) {
            output.resize(numRows);
            receiveCounts.resize(ranks);
            receiveOffsets.resize(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                receiveCounts[rank] = static_cast<int>(rowBoundaries[rank + 1] -
                                                       rowBoundaries[rank]);
                receiveOffsets[rank] = static_cast<int>(rowBoundaries[rank]);
            }
        }
        MPI_CHECK(MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                              worldRank == 0 ? output.data() : nullptr,
                              worldRank == 0 ? receiveCounts.data() : nullptr,
                              worldRank == 0 ? receiveOffsets.data() : nullptr,
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));

        if (worldRank == 0) {
            const double milliseconds = elapsedSeconds * 1000.0;
            const double averageMilliseconds = milliseconds / iterations;
            const double gflops = elapsedSeconds > 0.0
                                      ? (2.0 * static_cast<double>(nonzeros) * iterations) /
                                            elapsedSeconds / 1.0e9
                                      : 0.0;
            std::printf("Computation time: %.3f ms\n", milliseconds);
            std::printf("Average time per iteration: %.3f ms\n", averageMilliseconds);
            std::printf("Performance: %.3f GFLOPS\n", gflops);

            if (printResults) {
                print_results(output, "OutputVector");
            }

            if (validate) {
                std::printf("Validating result...\n");
                if (verifyResults(reference.data(), output.data(), numRows)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = EXIT_FAILURE;
                }
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&nodeCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
