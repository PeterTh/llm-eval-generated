#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int TileSize = 32;

struct Options {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool valid = true;
};

// Generate pseudo-random values for matrix initialization.  The expression is
// intentionally unchanged from the reference implementation so that -r
// remains comparable across implementations.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                    const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    // The initialization is independent for every element and is consequently
    // a useful CPU-side OpenMP stage while the MPI ranks are being launched.
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

Options parseArguments(const int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0) {
                options.valid = false;
            } else {
                options.n = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            options.valid = false;
        }
    }

    return options;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

[[noreturn]] void mpiAbort(const char* message, const int errorCode = 1) {
    std::fprintf(stderr, "matmul: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

void cudaCheck(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d: %s",
                      expression, file, line, cudaGetErrorString(status));
        mpiAbort(message);
    }
}

#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t rows, const size_t N) {
    __shared__ double tileA[TileSize][TileSize];
    __shared__ double tileB[TileSize][TileSize];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TileSize + ty;
    const size_t col = static_cast<size_t>(blockIdx.x) * TileSize + tx;
    double sums[TileSize / 8] = {0.0, 0.0, 0.0, 0.0};

    // Each thread computes four rows of one output column.  This retains the
    // same k order for every element while using 256 threads per block rather
    // than 1024, which gives the SM more resident warps and better occupancy.
    for (size_t tileStart = 0; tileStart < N; tileStart += TileSize) {
        const size_t aColumn = tileStart + static_cast<size_t>(tx);
        for (int output = 0; output < TileSize / 8; ++output) {
            const size_t aRow = static_cast<size_t>(blockIdx.y) * TileSize +
                                static_cast<size_t>(ty + output * 8);
            const size_t bRow = tileStart + static_cast<size_t>(ty + output * 8);
            tileA[ty + output * 8][tx] = (aRow < rows && aColumn < N)
                                             ? A[aRow * N + aColumn]
                                             : 0.0;
            tileB[ty + output * 8][tx] = (bRow < N && col < N)
                                             ? B[bRow * N + col]
                                             : 0.0;
        }
        __syncthreads();

        const size_t tileWidth = min(static_cast<size_t>(TileSize), N - tileStart);
        for (size_t k = 0; k < tileWidth; ++k) {
            for (int output = 0; output < TileSize / 8; ++output) {
                const size_t rowInTile = static_cast<size_t>(ty + output * 8);
                sums[output] += tileA[rowInTile][k] * tileB[k][tx];
            }
        }
        __syncthreads();
    }

    for (int output = 0; output < TileSize / 8; ++output) {
        const size_t row = rowBase + static_cast<size_t>(output * 8);
        if (row < rows && col < N) {
            C[row * N + col] = sums[output];
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) /
                                             (expected + 1e-10));

            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

size_t checkedProduct(const size_t a, const size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        mpiAbort("matrix dimensions overflow the host address space");
    }
    return a * b;
}

void checkMpiSize(const size_t N) {
    // A contiguous row datatype lets MPI transfer a row without limiting the
    // number of matrix elements to INT_MAX.  MPI still uses an int for the
    // number of rows in Scatterv/Gatherv, so keep that conversion explicit.
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        mpiAbort("matrix dimension exceeds MPI's row-count limit");
    }
}

bool createRowDistribution(const size_t N, const int worldSize,
                           std::vector<int>& rowCounts,
                           std::vector<int>& rowDisplacements) {
    rowCounts.resize(static_cast<size_t>(worldSize));
    rowDisplacements.resize(static_cast<size_t>(worldSize));

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    size_t displacement = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        const size_t rows = baseRows +
                            (static_cast<size_t>(rank) < extraRows ? 1 : 0);
        if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        rowCounts[static_cast<size_t>(rank)] = static_cast<int>(rows);
        rowDisplacements[static_cast<size_t>(rank)] = static_cast<int>(displacement);
        displacement += rows;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        mpiAbort("MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    Options options = parseArguments(argc, argv);
    int valid = options.valid ? 1 : 0;
    int help = options.help ? 1 : 0;
    std::uint64_t n64 = static_cast<std::uint64_t>(options.n);
    int validate = options.validate ? 1 : 0;
    int printResults = options.printResults ? 1 : 0;

    // Rank zero is authoritative for command-line settings.  This also keeps
    // malformed invocations from leaving a subset of ranks in a collective.
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid || help) {
        if (rank == 0) {
            if (!valid) {
                std::printf("Invalid command line.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Bcast(&n64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t N = static_cast<size_t>(n64);
    if (N == 0 || static_cast<std::uint64_t>(N) != n64) {
        mpiAbort("matrix dimension is not representable on this platform");
    }
    checkMpiSize(N);
    const size_t matrixElements = checkedProduct(N, N);
    const size_t matrixBytes = checkedProduct(matrixElements, sizeof(double));

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    if (!createRowDistribution(N, worldSize, rowCounts, rowDisplacements)) {
        mpiAbort("matrix row distribution exceeds MPI's count limit");
    }

    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    const size_t localElements = checkedProduct(localRows, N);
    const size_t localBytes = checkedProduct(localElements, sizeof(double));

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI rank%s, %d OpenMP thread%s/rank, "
                    "one CUDA device/rank\n",
                    worldSize, worldSize == 1 ? "" : "s",
                    omp_get_max_threads(), omp_get_max_threads() == 1 ? "" : "s");
    }

    std::vector<double> A;
    std::vector<double> C;
    std::vector<double> B(matrixElements);
    if (rank == 0) {
        A.resize(matrixElements);
        C.resize(matrixElements);
        std::printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    std::vector<double> localA(localElements);
    std::vector<double> localC(localElements);

    MPI_Datatype matrixRowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &matrixRowType);
    MPI_Type_commit(&matrixRowType);

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    if (deviceQuery != cudaSuccess || deviceCount == 0) {
        mpiAbort("no CUDA device is available for this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaStream_t stream = nullptr;
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA),
                              localBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC),
                              localBytes));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB),
                          matrixBytes));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    MPI_Bcast(B.data(), static_cast<int>(N), matrixRowType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, rowCounts.data(),
                 rowDisplacements.data(), matrixRowType,
                 localA.data(), rowCounts[rank], matrixRowType, 0,
                 MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpyAsync(deviceB, B.data(),
                               matrixBytes,
                               cudaMemcpyHostToDevice, stream));
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceA, localA.data(),
                                   localBytes,
                                   cudaMemcpyHostToDevice, stream));

        const dim3 block(TileSize, TileSize / 4);
        const dim3 grid(static_cast<unsigned int>((N + TileSize - 1) / TileSize),
                        static_cast<unsigned int>((localRows + TileSize - 1) /
                                                  TileSize));
        matrixMultiplyKernel<<<grid, block, 0, stream>>>(
            deviceA, deviceB, deviceC, localRows, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(localC.data(), deviceC,
                                   localBytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Gatherv(localC.data(), rowCounts[rank], matrixRowType,
                rank == 0 ? C.data() : nullptr, rowCounts.data(),
                rowDisplacements.data(), matrixRowType, 0, MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceB));
    if (localRows > 0) {
        CUDA_CHECK(cudaFree(deviceA));
        CUDA_CHECK(cudaFree(deviceC));
    }
    MPI_Comm_free(&localCommunicator);
    MPI_Type_free(&matrixRowType);

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double seconds = std::max(elapsed, std::numeric_limits<double>::min());
        const double gflops = (2.0 * static_cast<double>(N) *
                               static_cast<double>(N) * static_cast<double>(N)) /
                              seconds / 1e9;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults != 0) {
            print_results(C, "MatrixC");
        }

        if (validate != 0) {
            std::printf("Validating result...\n");
            const bool validResult = validateResult(A, B, C, N);
            if (validResult) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return validResult ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
