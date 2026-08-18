#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

void mpiFail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void cudaCheck(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d for %s: %s\n", rank, file, line,
                     expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(const size_t count) { allocate(count); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    void allocate(const size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    void release() {
        if (ptr_ != nullptr) {
            // Cleanup must not hide an earlier error while the process is terminating.
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    T* ptr_ = nullptr;
};

// Each warp owns one CSR row.  This keeps the vector gather coalesced across
// the warp and avoids atomics or inter-block synchronization.
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rowOffsets,
                           const double* __restrict__ vector,
                           const index_t rowCount,
                           double* __restrict__ output) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const index_t row = static_cast<index_t>(blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE);

    if (row >= rowCount) {
        return;
    }

    double sum = 0.0;
    for (index_t entry = rowOffsets[row] + lane; entry < rowOffsets[row + 1]; entry += WARP_SIZE) {
        const index_t column = columns[entry];
        sum += values[entry] * vector[column];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

void fill(double* values, const index_t count, const double maxValue) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* columns, index_t* rowOffsets, const index_t nonzeros, const index_t dimension) {
    index_t assigned = 0;
    const double probability = static_cast<double>(nonzeros) /
                               (static_cast<double>(dimension) * static_cast<double>(dimension));

    // Keep the benchmark's deterministic matrix-generation sequence unchanged.
    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dimension; ++row) {
        rowOffsets[row] = assigned;
        for (index_t column = 0; column < dimension; ++column) {
            const index_t entriesLeft = (dimension * dimension) - ((row * dimension) + column);
            const index_t needed = nonzeros - assigned;
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

// This reference is deliberately parallelized by output row: all reductions
// retain their original per-row order, so it is suitable for validation.
void spmvCpu(const double* values, const index_t* columns, const index_t* rowOffsets,
             const double* vector, const index_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (index_t entry = rowOffsets[row]; entry < rowOffsets[row + 1]; ++entry) {
            sum += values[entry] * vector[columns[entry]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool valid = true;
#pragma omp parallel for schedule(static) reduction(&& : valid)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const bool entryValid = std::abs(ref) < 1e-10 ? std::abs(res) <= MAX_RELATIVE_ERROR
                                                       : std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
        valid = valid && entryValid;
    }

    if (!valid) {
        // Report one deterministic failure after the parallel comparison.
        for (index_t i = 0; i < size; ++i) {
            const double ref = reference[i];
            const double res = result[i];
            const bool entryValid = std::abs(ref) < 1e-10 ? std::abs(res) <= MAX_RELATIVE_ERROR
                                                           : std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
            if (!entryValid) {
                std::printf("Validation failed at local index %u: reference %.10e, got %.10e\n", i, ref, res);
                break;
            }
        }
    }
    return valid;
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

enum class ParseResult { valid, help, invalid };

ParseResult parseArguments(const int argc, char** argv, index_t& rows, index_t& sparsity,
                           index_t& iterations, double& maxValue, bool& validate, bool& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            rows = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxValue = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return ParseResult::help;
        } else {
            return ParseResult::invalid;
        }
    }
    return sparsity != 0 && iterations != 0 ? ParseResult::valid : ParseResult::invalid;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        mpiFail("MPI implementation does not provide the required thread support");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxValue = 1.0;
    bool validate = false;
    bool printResults = false;
    const ParseResult parseResult =
        parseArguments(argc, argv, numRows, sparsity, iterations, maxValue, validate, printResults);
    if (parseResult != ParseResult::valid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseResult == ParseResult::help ? 0 : 1;
    }

    const uint64_t nonzeros64 = (static_cast<uint64_t>(numRows) * numRows) / sparsity;
    if (nonzeros64 > std::numeric_limits<index_t>::max() ||
        numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nonzeros64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix dimensions exceed the supported MPI collective count range.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nonzeros = static_cast<index_t>(nonzeros64);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        mpiFail("no CUDA device is visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_Comm_free(&nodeComm);

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonzeros,
                    100.0 * (1.0 - static_cast<double>(nonzeros) /
                                       (static_cast<double>(numRows) * numRows)));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxValue);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d; CUDA device assignment: local-rank modulo visible devices\n", worldSize);
        std::printf("Initializing data structures...\n");
    }

    // Only rank zero owns the global CSR during setup.  Matrix generation is
    // kept serial and deterministic; the timed calculation is fully distributed.
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowOffsets;
    std::vector<double> vector(numRows);
    if (rank == 0) {
        globalValues.resize(nonzeros);
        globalColumns.resize(nonzeros);
        globalRowOffsets.resize(static_cast<size_t>(numRows) + 1);
        fill(vector.data(), numRows, maxValue);
        fill(globalValues.data(), nonzeros, maxValue);
        initRandomMatrix(globalColumns.data(), globalRowOffsets.data(), nonzeros, numRows);
    }
    MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t pastLastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / worldSize);
    const index_t localRows = pastLastRow -
                              static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / worldSize);

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nonzeroCounts;
    std::vector<int> nonzeroDisplacements;
    if (rank == 0) {
        rowCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        nonzeroCounts.resize(worldSize);
        nonzeroDisplacements.resize(worldSize);
        for (int process = 0; process < worldSize; ++process) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(process) * numRows) / worldSize);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(process + 1) * numRows) / worldSize);
            rowCounts[process] = static_cast<int>(end - begin + 1);
            rowDisplacements[process] = static_cast<int>(begin);
            nonzeroDisplacements[process] = static_cast<int>(globalRowOffsets[begin]);
            nonzeroCounts[process] = static_cast<int>(globalRowOffsets[end] - globalRowOffsets[begin]);
        }
    }

    std::vector<index_t> localRowOffsets(static_cast<size_t>(localRows) + 1);
    MPI_Scatterv(rank == 0 ? globalRowOffsets.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRowOffsets.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t nonzeroBase = localRowOffsets.front();
    const index_t localNonzeros = localRowOffsets.back() - nonzeroBase;

    // Normalize global row pointers to the local CSR range in parallel.
#pragma omp parallel for schedule(static)
    for (index_t row = 0; row <= localRows; ++row) {
        localRowOffsets[row] -= nonzeroBase;
    }

    std::vector<double> localValues(localNonzeros);
    std::vector<index_t> localColumns(localNonzeros);
    MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                 rank == 0 ? nonzeroCounts.data() : nullptr,
                 rank == 0 ? nonzeroDisplacements.data() : nullptr, MPI_DOUBLE,
                 localValues.data(), static_cast<int>(localNonzeros), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalColumns.data() : nullptr,
                 rank == 0 ? nonzeroCounts.data() : nullptr,
                 rank == 0 ? nonzeroDisplacements.data() : nullptr, MPI_UINT32_T,
                 localColumns.data(), static_cast<int>(localNonzeros), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    globalValues.clear();
    globalValues.shrink_to_fit();
    globalColumns.clear();
    globalColumns.shrink_to_fit();
    globalRowOffsets.clear();
    globalRowOffsets.shrink_to_fit();

    DeviceBuffer<double> deviceValues(localNonzeros);
    DeviceBuffer<index_t> deviceColumns(localNonzeros);
    DeviceBuffer<index_t> deviceRowOffsets(static_cast<size_t>(localRows) + 1);
    DeviceBuffer<double> deviceVector(numRows);
    DeviceBuffer<double> deviceOutput(localRows);
    if (localNonzeros != 0) {
        CUDA_CHECK(cudaMemcpy(deviceValues.get(), localValues.data(), localNonzeros * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceColumns.get(), localColumns.data(), localNonzeros * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(deviceRowOffsets.get(), localRowOffsets.data(),
                          (static_cast<size_t>(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    if (numRows != 0) {
        CUDA_CHECK(cudaMemcpy(deviceVector.get(), vector.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Prime the selected device before timing.  This excludes first-use CUDA
    // context and module-load latency while preserving the benchmark result.
    if (localRows != 0) {
        const unsigned int warmupBlocks = (localRows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        spmvKernel<<<warmupBlocks, THREADS_PER_BLOCK>>>(deviceValues.get(), deviceColumns.get(),
                                                        deviceRowOffsets.get(), deviceVector.get(), localRows,
                                                        deviceOutput.get());
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        std::printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));
    if (localRows != 0) {
        const unsigned int blocks = (localRows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        for (index_t iteration = 0; iteration < iterations; ++iteration) {
            spmvKernel<<<blocks, THREADS_PER_BLOCK>>>(deviceValues.get(), deviceColumns.get(),
                                                       deviceRowOffsets.get(), deviceVector.get(), localRows,
                                                       deviceOutput.get());
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float localElapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&localElapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    double elapsedMilliseconds = static_cast<double>(localElapsedMilliseconds);
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    std::vector<double> localOutput(localRows);
    if (localRows != 0) {
        CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput.get(), localRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        const double gflops = elapsedMilliseconds > 0.0
                                  ? (2.0 * static_cast<double>(nonzeros) * iterations) /
                                        (elapsedMilliseconds / 1000.0) / 1e9
                                  : 0.0;
        std::printf("Average time per iteration: %.3f ms\n", elapsedMilliseconds / iterations);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> globalOutput;
        std::vector<int> outputCounts;
        std::vector<int> outputDisplacements;
        if (rank == 0) {
            globalOutput.resize(numRows);
            outputCounts.resize(worldSize);
            outputDisplacements.resize(worldSize);
            for (int process = 0; process < worldSize; ++process) {
                const index_t begin = static_cast<index_t>((static_cast<uint64_t>(process) * numRows) / worldSize);
                const index_t end = static_cast<index_t>((static_cast<uint64_t>(process + 1) * numRows) / worldSize);
                outputCounts[process] = static_cast<int>(end - begin);
                outputDisplacements[process] = static_cast<int>(begin);
            }
        }
        MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr,
                    rank == 0 ? outputCounts.data() : nullptr,
                    rank == 0 ? outputDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalOutput, "OutputVector");
        }
    }

    int localValid = 1;
    if (validate) {
        std::vector<double> localReference(localRows);
        spmvCpu(localValues.data(), localColumns.data(), localRowOffsets.data(), vector.data(), localRows,
                localReference.data());
        localValid = verifyResults(localReference.data(), localOutput.data(), localRows) ? 1 : 0;
    }
    int globallyValid = 1;
    MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0 && validate) {
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return globallyValid ? 0 : 1;
}
