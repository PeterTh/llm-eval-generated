#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

[[noreturn]] void fail(const char* where, const char* detail) {
    std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, where, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const char* where) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, error, &length);
        fail(where, error);
    }
}

void checkCuda(const cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        fail(where, cudaGetErrorString(status));
    }
}

void checkCublas(const cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(where, "cuBLAS operation failed");
    }
}

// Generate pseudo-random values for matrix initialization.  This is identical
// to the original initialization, including the global (rather than local)
// row index used by ranks that own a subset of A.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

class PinnedBuffer {
  public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(const size_t elements) { allocate(elements); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            // There is no useful recovery path while unwinding program exit.
            cudaFreeHost(data_);
        }
    }

    void allocate(const size_t elements) {
        if (elements == 0) {
            return;
        }
        if (elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
            fail("host allocation", "matrix is too large");
        }
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&data_), elements * sizeof(double)),
                  "cudaMallocHost");
    }

    double* data() noexcept { return data_; }

  private:
    double* data_ = nullptr;
};

void initMatrixRows(double* const mat, const size_t N, const size_t firstRow,
                    const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(rowCount); ++localRow) {
        const size_t globalRow = firstRow + static_cast<size_t>(localRow);
        double* const row = mat + static_cast<size_t>(localRow) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// Validation uses the same summation order and deterministic input values as
// the serial version, while avoiding a second full copy of A on rank zero.
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t iPoint : checkPoints) {
        for (const size_t jPoint : checkPoints) {
            const size_t i = iPoint % N;
            const size_t j = jPoint % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f "
                            "(error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseMatrixSize(const char* text, size_t* const value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '\0' || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");

    int worldSize = 0;
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");
    if (provided < MPI_THREAD_FUNNELED) {
        fail("MPI_Init_thread", "MPI_THREAD_FUNNELED support is required");
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool invalidArguments = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseMatrixSize(argv[++i], &N)) {
                invalidArguments = true;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            invalidArguments = true;
        }
    }

    if (showHelp || invalidArguments) {
        if (worldRank == 0) {
            if (invalidArguments) {
                std::printf("Invalid command-line arguments\n");
            }
            printUsage(argv[0]);
        }
        checkMpi(MPI_Finalize(), "MPI_Finalize");
        return invalidArguments ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    // MPI counts and cuBLAS dimensions are int.  Keeping the complete result
    // within that range also prevents overflow in MPI_Gatherv displacements.
    if (N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N ||
        N * N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Matrix size is too large for this MPI/cuBLAS implementation\n");
        }
        checkMpi(MPI_Finalize(), "MPI_Finalize");
        return EXIT_FAILURE;
    }

    // One rank is bound to one visible GPU.  On a scheduler-managed run each
    // rank commonly sees a single GPU; modulo mapping remains valid when all
    // node GPUs are visible instead.
    MPI_Comm localComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                 MPI_INFO_NULL, &localComm),
             "MPI_Comm_split_type");
    int localRank = 0;
    checkMpi(MPI_Comm_rank(localComm, &localRank), "MPI_Comm_rank(local)");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fail("cudaGetDeviceCount", "no CUDA device is visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    checkMpi(MPI_Comm_free(&localComm), "MPI_Comm_free");

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(worldRank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(worldRank) * baseRows +
                            std::min(static_cast<size_t>(worldRank), extraRows);
    const size_t matrixElements = N * N;
    const size_t localElements = localRows * N;

    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("Initializing matrices...\n");
    }

    // A is distributed by contiguous rows. B is replicated, which avoids a
    // communication dependency in the DGEMM critical path and lets every GPU
    // execute an optimized square/rectangular DGEMM independently.
    PinnedBuffer hostA(localElements);
    PinnedBuffer hostB(matrixElements);
    initMatrixRows(hostA.data(), N, firstRow, localRows);
    initMatrixRows(hostB.data(), N, 0, N);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    if (localElements != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localElements * sizeof(double)),
                  "cudaMalloc(A)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), localElements * sizeof(double)),
                  "cudaMalloc(C)");
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), matrixElements * sizeof(double)),
              "cudaMalloc(B)");

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetStream(handle, stream), "cublasSetStream");
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    // Matrix construction is outside the serial benchmark's timed region, so
    // transfer the immutable inputs before timing the distributed DGEMMs too.
    checkCuda(cudaMemcpyAsync(deviceB, hostB.data(), matrixElements * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(B)");
    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(deviceA, hostA.data(), localElements * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(A)");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inputs)");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");

    if (worldRank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    checkCuda(cudaEventRecord(startEvent, stream), "cudaEventRecord(start)");

    if (localRows != 0) {
        // The row-major product C=A*B is the column-major product
        // C^T=B^T*A^T.  This mapping avoids expensive explicit transposes and
        // dispatches directly to the vendor-tuned CUDA DGEMM implementation.
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(N), static_cast<int>(localRows),
                                 static_cast<int>(N), &alpha, deviceB, static_cast<int>(N),
                                 deviceA, static_cast<int>(N), &beta, deviceC,
                                 static_cast<int>(N)),
                    "cublasDgemm");
    }

    checkCuda(cudaEventRecord(stopEvent, stream), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize(stop)");
    float localMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&localMilliseconds, startEvent, stopEvent),
              "cudaEventElapsedTime");

    double maximumMilliseconds = 0.0;
    const double localMillisecondsDouble = static_cast<double>(localMilliseconds);
    checkMpi(MPI_Reduce(&localMillisecondsDouble, &maximumMilliseconds, 1, MPI_DOUBLE, MPI_MAX,
                        0, MPI_COMM_WORLD),
             "MPI_Reduce(timing)");

    const bool needFullResult = printResults || validate;
    PinnedBuffer localResult(needFullResult ? localElements : 0);
    std::vector<double> result;
    std::vector<int> receiveCounts;
    std::vector<int> receiveDisplacements;
    if (needFullResult) {
        if (localElements != 0) {
            checkCuda(cudaMemcpyAsync(localResult.data(), deviceC, localElements * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync(C)");
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(result)");
        }

        if (worldRank == 0) {
            result.resize(matrixElements);
            receiveCounts.resize(static_cast<size_t>(worldSize));
            receiveDisplacements.resize(static_cast<size_t>(worldSize));
            size_t displacement = 0;
            for (int rank = 0; rank < worldSize; ++rank) {
                const size_t rows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
                const size_t count = rows * N;
                receiveCounts[static_cast<size_t>(rank)] = static_cast<int>(count);
                receiveDisplacements[static_cast<size_t>(rank)] = static_cast<int>(displacement);
                displacement += count;
            }
        }

        checkMpi(MPI_Gatherv(localResult.data(), static_cast<int>(localElements), MPI_DOUBLE,
                             worldRank == 0 ? result.data() : nullptr,
                             worldRank == 0 ? receiveCounts.data() : nullptr,
                             worldRank == 0 ? receiveDisplacements.data() : nullptr, MPI_DOUBLE, 0,
                             MPI_COMM_WORLD),
                 "MPI_Gatherv(C)");
    }

    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", maximumMilliseconds);
        const double gflops = maximumMilliseconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                                     static_cast<double>(N)) /
                                        (maximumMilliseconds * 1.0e6)
                                  : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(result, "MatrixC");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (!validateResult(result, N)) {
                std::printf("Validation: FAILED\n");
                checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE), "MPI_Abort(validation)");
                return EXIT_FAILURE;
            }
            std::printf("Validation: PASSED\n");
        }
    }

    checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy(stop)");
    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (deviceA != nullptr) {
        checkCuda(cudaFree(deviceA), "cudaFree(A)");
    }
    checkCuda(cudaFree(deviceB), "cudaFree(B)");
    if (deviceC != nullptr) {
        checkCuda(cudaFree(deviceC), "cudaFree(C)");
    }

    checkMpi(MPI_Finalize(), "MPI_Finalize");
    return EXIT_SUCCESS;
}
