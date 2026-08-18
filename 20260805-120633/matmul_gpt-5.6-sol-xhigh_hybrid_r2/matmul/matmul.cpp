#include <mpi.h>
#include <omp.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

[[noreturn]] void abortBenchmark(const char* operation, const char* detail) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", worldRank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        abortBenchmark(operation, cudaGetErrorString(status));
    }
}

const char* cublasStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
        default: return "unknown cuBLAS error";
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        abortBenchmark(operation, cublasStatusString(status));
    }
}

template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                      "cudaMallocHost");
        }
    }

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            // A destructor cannot usefully recover from a CUDA shutdown error.
            cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

struct RowBlock {
    std::size_t first;
    std::size_t count;
};

RowBlock rowsForRank(std::size_t n, int rank, int ranks) {
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t base = n / rankCount;
    const std::size_t remainder = n % rankCount;
    return {rankIndex * base + std::min(rankIndex, remainder),
            base + (rankIndex < remainder ? 1U : 0U)};
}

// Generate exactly the same deterministic values as the original benchmark.
inline double getPseudoRndValue(std::size_t n, std::size_t i, std::size_t j) noexcept {
    const std::size_t elements = n * n;
    return (((i + 1) * (i + j + 1) * 1299709) % elements) /
           static_cast<double>(elements);
}

void initMatrixRows(double* matrix, std::size_t n, std::size_t firstRow,
                    std::size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (std::int64_t localRow = 0; localRow < static_cast<std::int64_t>(rowCount);
         ++localRow) {
        const std::size_t i = firstRow + static_cast<std::size_t>(localRow);
        double* const row = matrix + static_cast<std::size_t>(localRow) * n;
        for (std::size_t j = 0; j < n; ++j) {
            row[j] = getPseudoRndValue(n, i, j);
        }
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    std::size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

bool parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || value == 0 ||
                value > std::numeric_limits<std::size_t>::max()) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return false;
            }
            options.n = static_cast<std::size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }

    if (options.n > std::numeric_limits<std::size_t>::max() / options.n ||
        options.n * options.n > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        options.n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Matrix size is too large for this CUDA/cuBLAS build\n");
        return false;
    }
    return true;
}

// Gather only for -r. The normal path uses a collective; the fallback keeps
// correctness when a rank's contiguous block exceeds MPI's int count limit.
void gatherResult(const double* localC, std::vector<double>& result, std::size_t n,
                  int rank, int ranks) {
    const RowBlock ownBlock = rowsForRank(n, rank, ranks);
    const std::size_t ownElements = ownBlock.count * n;
    const std::size_t totalElements = n * n;

    if (totalElements <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<std::size_t>(ranks));
            displacements.resize(static_cast<std::size_t>(ranks));
            for (int source = 0; source < ranks; ++source) {
                const RowBlock block = rowsForRank(n, source, ranks);
                counts[static_cast<std::size_t>(source)] = static_cast<int>(block.count * n);
                displacements[static_cast<std::size_t>(source)] = static_cast<int>(block.first * n);
            }
        }
        MPI_Gatherv(localC, static_cast<int>(ownElements), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return;
    }

    constexpr int tag = 1701;
    constexpr std::size_t maxChunk = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (rank == 0) {
        std::copy_n(localC, ownElements, result.data() + ownBlock.first * n);
        for (int source = 1; source < ranks; ++source) {
            const RowBlock block = rowsForRank(n, source, ranks);
            const std::size_t elements = block.count * n;
            for (std::size_t offset = 0; offset < elements; offset += maxChunk) {
                const int chunk = static_cast<int>(std::min(maxChunk, elements - offset));
                MPI_Recv(result.data() + block.first * n + offset, chunk, MPI_DOUBLE,
                         source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (std::size_t offset = 0; offset < ownElements; offset += maxChunk) {
            const int chunk = static_cast<int>(std::min(maxChunk, ownElements - offset));
            MPI_Send(localC + offset, chunk, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    }
}

bool validateDistributed(const double* localA, const double* b, const double* localC,
                         std::size_t n, const RowBlock& block, int rank, int ranks) {
    constexpr int checks = 25;
    int localFailure = checks;

#pragma omp parallel for schedule(static) reduction(min : localFailure)
    for (int check = 0; check < checks; ++check) {
        const std::size_t i = static_cast<std::size_t>(check / 5) % n;
        const std::size_t j = static_cast<std::size_t>(check % 5) % n;
        if (i < block.first || i >= block.first + block.count) {
            continue;
        }

        const std::size_t localRow = i - block.first;
        double expected = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            expected += localA[localRow * n + k] * b[k * n + j];
        }
        const double actual = localC[localRow * n + j];
        const double relativeError = std::abs((actual - expected) / (expected + 1.0e-10));
        if (relativeError > 1.0e-6 || !std::isfinite(relativeError)) {
            localFailure = std::min(localFailure, check);
        }
    }

    int globalFailure = checks;
    MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (globalFailure == checks) {
        return true;
    }

    const std::size_t failedI = static_cast<std::size_t>(globalFailure / 5) % n;
    const std::size_t failedJ = static_cast<std::size_t>(globalFailure % 5) % n;
    int owner = 0;
    for (; owner < ranks; ++owner) {
        const RowBlock ownerBlock = rowsForRank(n, owner, ranks);
        if (failedI >= ownerBlock.first && failedI < ownerBlock.first + ownerBlock.count) {
            break;
        }
    }

    double details[3] = {};
    if (rank == owner) {
        const std::size_t localRow = failedI - block.first;
        for (std::size_t k = 0; k < n; ++k) {
            details[0] += localA[localRow * n + k] * b[k * n + failedJ];
        }
        details[1] = localC[localRow * n + failedJ];
        details[2] = std::abs((details[1] - details[0]) / (details[0] + 1.0e-10));
    }
    MPI_Bcast(details, 3, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                    failedI, failedJ, details[0], details[1], details[2]);
    }
    return false;
}

int runBenchmark(const Options& options, int ranks) {
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    int localRanks = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        const int threadsPerRank = std::max(1, omp_get_num_procs() / localRanks);
        omp_set_num_threads(threadsPerRank);
    }
    omp_set_dynamic(0);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        abortBenchmark("CUDA initialization", "no CUDA-capable device is visible");
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    if (worldRank == 0 && localRanks > deviceCount) {
        std::fprintf(stderr,
                     "Warning: %d local MPI ranks share %d GPU(s); one rank per GPU is recommended.\n",
                     localRanks, deviceCount);
    }

    const std::size_t n = options.n;
    const std::size_t matrixElements = n * n;
    const RowBlock block = rowsForRank(n, worldRank, ranks);
    const std::size_t localElements = block.count * n;

    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, GPU(s)/node: %d\n",
                    ranks, omp_get_max_threads(), deviceCount);
        std::printf("Initializing matrices...\n");
    }

    PinnedBuffer<double> localA(localElements);
    PinnedBuffer<double> b(matrixElements);
    PinnedBuffer<double> localC(localElements);

    // Both independent loops are OpenMP-parallel and avoid broadcasting an
    // O(N^2) input matrix across the cluster interconnect.
    initMatrixRows(localA.data(), n, block.first, block.count);
    initMatrixRows(b.data(), n, 0, n);

    DeviceBuffer<double> deviceA(localElements);
    DeviceBuffer<double> deviceB(matrixElements);
    DeviceBuffer<double> deviceC(localElements);

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetStream(handle, stream), "cublasSetStream");
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    checkCublas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED), "cublasSetAtomicsMode");

    // Force lazy CUDA/cuBLAS kernel initialization before the measured region.
    // The real operation below overwrites deviceC because beta is zero.
    if (localElements != 0) {
        const double warmupAlpha = 1.0;
        const double warmupBeta = 0.0;
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                &warmupAlpha, deviceB.data(), 1, deviceA.data(), 1,
                                &warmupBeta, deviceC.data(), 1),
                    "cuBLAS warm-up");
        checkCuda(cudaStreamSynchronize(stream), "CUDA warm-up synchronization");
    }

    if (worldRank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    checkCuda(cudaMemcpyAsync(deviceB.data(), b.data(), matrixElements * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              "copying B to the GPU");
    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(deviceA.data(), localA.data(), localElements * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "copying A to the GPU");

        const double alpha = 1.0;
        const double beta = 0.0;
        // cuBLAS is column-major. Viewing the row-major buffers as transposes
        // computes C^T = B^T A^T without any explicit matrix transposition.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(n), static_cast<int>(block.count),
                                static_cast<int>(n), &alpha, deviceB.data(),
                                static_cast<int>(n), deviceA.data(), static_cast<int>(n),
                                &beta, deviceC.data(), static_cast<int>(n)),
                    "cublasDgemm");
        checkCuda(cudaMemcpyAsync(localC.data(), deviceC.data(), localElements * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  "copying C from the GPU");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (worldRank == 0) {
        const double operations = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                                  static_cast<double>(n);
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double gflops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (options.printResults) {
        if (worldRank == 0) {
            try {
                result.resize(matrixElements);
            } catch (const std::bad_alloc&) {
                abortBenchmark("result allocation", "insufficient host memory");
            }
        }
        gatherResult(localC.data(), result, n, worldRank, ranks);
        if (worldRank == 0) {
            print_results(result, "MatrixC");
        }
    }

    bool valid = true;
    if (options.validate) {
        if (worldRank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(localA.data(), b.data(), localC.data(), n, block,
                                    worldRank, ranks);
        if (worldRank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    MPI_Comm_free(&localComm);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    int threadSupport = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadSupport);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if (threadSupport < MPI_THREAD_FUNNELED) {
        abortBenchmark("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable");
    }

    Options options;
    int parseStatus = 1;
    if (worldRank == 0) {
        parseStatus = parseOptions(argc, argv, options) ? 1 : 0;
        if (!parseStatus || options.help) {
            printUsage(argv[0]);
        }
    }

    unsigned long long packedN = static_cast<unsigned long long>(options.n);
    int flags[4] = {parseStatus, options.validate ? 1 : 0,
                    options.printResults ? 1 : 0, options.help ? 1 : 0};
    MPI_Bcast(&packedN, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
    options.n = static_cast<std::size_t>(packedN);
    options.validate = flags[1] != 0;
    options.printResults = flags[2] != 0;
    options.help = flags[3] != 0;

    int status = EXIT_SUCCESS;
    if (flags[0] != 0 && !options.help) {
        status = runBenchmark(options, ranks);
    } else if (flags[0] == 0) {
        status = EXIT_FAILURE;
    }

    MPI_Finalize();
    return status;
}
