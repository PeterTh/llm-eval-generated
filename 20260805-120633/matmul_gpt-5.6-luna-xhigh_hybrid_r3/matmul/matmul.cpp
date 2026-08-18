#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kTileColumns = 32;
constexpr int kTileRows = 8;

// Generate pseudo-random values for matrix initialization.  This is kept
// identical to the original implementation so every MPI rank can generate
// its own rows without communicating matrix A.
constexpr double getPseudoRndValue(const std::size_t N,
                                    const std::size_t i,
                                    const std::size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrixRows(std::vector<double>& mat,
                    const std::size_t N,
                    const std::size_t firstRow,
                    const std::size_t rowCount) {
    // Matrix initialization is deliberately parallel even though the actual
    // multiplication is performed by CUDA.  It avoids serial host-side setup
    // becoming visible at scale and honors the user's OpenMP configuration.
    #pragma omp parallel for schedule(static)
    for (std::int64_t localRow = 0;
         localRow < static_cast<std::int64_t>(rowCount);
         ++localRow) {
        const std::size_t globalRow = firstRow + static_cast<std::size_t>(localRow);
        const std::size_t rowStart = static_cast<std::size_t>(localRow) * N;
        for (std::size_t column = 0; column < N; ++column) {
            mat[rowStart + column] = getPseudoRndValue(N, globalRow, column);
        }
    }
}

void initMatrix(std::vector<double>& mat, const std::size_t N) {
    initMatrixRows(mat, N, 0, N);
}

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const int rows,
                                     const int N) {
    __shared__ double tileA[kTileRows][kTileColumns];
    __shared__ double tileB[kTileColumns][kTileColumns];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * kTileRows + ty;
    const int column = static_cast<int>(blockIdx.x) * kTileColumns + tx;

    double sum = 0.0;
    for (int tileStart = 0; tileStart < N; tileStart += kTileColumns) {
        const int aColumn = tileStart + tx;
        tileA[ty][tx] = (row < rows && aColumn < N)
                             ? A[static_cast<std::size_t>(row) * N + aColumn]
                             : 0.0;

        // There are four B rows for each of the eight thread rows.  This
        // fills the 32x32 B tile with coalesced accesses using a 32x8 block.
        for (int bRow = ty; bRow < kTileColumns; bRow += kTileRows) {
            const int bRowInMatrix = tileStart + bRow;
            tileB[bRow][tx] = (bRowInMatrix < N && column < N)
                                  ? B[static_cast<std::size_t>(bRowInMatrix) * N + column]
                                  : 0.0;
        }
        __syncthreads();

        const int validK = min(kTileColumns, N - tileStart);
        #pragma unroll
        for (int k = 0; k < kTileColumns; ++k) {
            if (k < validK) {
                sum += tileA[ty][k] * tileB[k][tx];
            }
        }
        __syncthreads();
    }

    if (row < rows && column < N) {
        C[static_cast<std::size_t>(row) * N + column] = sum;
    }
}

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    std::fprintf(stderr, "MPI rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error,
               const char* expression,
               const int rank) {
    if (error != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error in %s: %s",
                      expression, cudaGetErrorString(error));
        abortWithMessage(rank, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank)

void broadcastDoubles(double* data,
                      const std::size_t count,
                      const int root,
                      const MPI_Comm communicator) {
    constexpr std::size_t maxMpiCount =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t offset = 0; offset < count;) {
        const std::size_t remaining = count - offset;
        const int chunk = static_cast<int>(std::min(remaining, maxMpiCount));
        MPI_Bcast(data + offset, chunk, MPI_DOUBLE, root, communicator);
        offset += static_cast<std::size_t>(chunk);
    }
}

bool parseArguments(const int argc,
                    char** argv,
                    std::size_t& N,
                    bool& validate,
                    bool& printResults,
                    bool& showHelp) {
    N = 512;
    validate = false;
    printResults = false;
    showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed =
                std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || parsed == 0) {
                return false;
            }
            N = static_cast<std::size_t>(parsed);
            if (static_cast<unsigned long long>(N) != parsed) {
                return false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            return false;
        }
    }

    // The CUDA kernel and the MPI count/displacement metadata use int-sized
    // dimensions.  Values beyond this are not representable by one matrix in
    // this benchmark and are rejected before any allocation occurs.
    if (N > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (N > std::numeric_limits<std::size_t>::max() / N) {
        return false;
    }
    if ((N * N) > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return false;
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

bool validateResult(const std::vector<double>& A,
                   const std::vector<double>& B,
                   const std::vector<double>& C,
                   const std::size_t N) {
    constexpr std::size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (std::size_t pi = 0; pi < 5; ++pi) {
        for (std::size_t pj = 0; pj < 5; ++pj) {
            const std::size_t i = checkPoints[pi] % N;
            const std::size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void calculateRowPartition(const std::size_t N,
                           const int worldSize,
                           const int rank,
                           std::size_t& firstRow,
                           std::size_t& rowCount) {
    const std::size_t baseRows = N / static_cast<std::size_t>(worldSize);
    const std::size_t remainder = N % static_cast<std::size_t>(worldSize);
    firstRow = static_cast<std::size_t>(rank) * baseRows +
               std::min(static_cast<std::size_t>(rank), remainder);
    rowCount = baseRows + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
}

} // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    std::size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    int argumentStatus = 1;

    if (rank == 0) {
        argumentStatus = parseArguments(argc, argv, N, validate, printResults, showHelp) ? 0 : 1;
    }

    unsigned long long wireN = static_cast<unsigned long long>(N);
    int wireValidate = validate ? 1 : 0;
    int wirePrintResults = printResults ? 1 : 0;
    int wireShowHelp = showHelp ? 1 : 0;
    MPI_Bcast(&argumentStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wireN, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wireValidate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wirePrintResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&wireShowHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (argumentStatus != 0) {
        if (rank == 0) {
            std::printf("Invalid command line arguments.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    N = static_cast<std::size_t>(wireN);
    validate = wireValidate != 0;
    printResults = wirePrintResults != 0;
    showHelp = wireShowHelp != 0;

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    const std::size_t matrixElements = N * N;
    std::size_t firstRow = 0;
    std::size_t localRows = 0;
    calculateRowPartition(N, worldSize, rank, firstRow, localRows);
    const std::size_t localElements = localRows * N;

    // Select one GPU per local MPI rank.  MPI_COMM_TYPE_SHARED makes the
    // mapping restart at zero on each node, which is the required mapping for
    // multi-node accelerator jobs.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        abortWithMessage(rank, "no CUDA device is available");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    omp_set_dynamic(0);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA devices per node: %d\n", deviceCount);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localElements);
    std::vector<double> B(matrixElements);
    initMatrixRows(localA, N, firstRow, localRows);

    // B is identical on every rank.  Only rank zero initializes it; the
    // chunked broadcast also supports matrices larger than MPI's int count.
    if (rank == 0) {
        initMatrix(B, N);
    }
    broadcastDoubles(B.data(), matrixElements, 0, MPI_COMM_WORLD);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    const std::size_t safeLocalElements = std::max<std::size_t>(localElements, 1);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA),
                          safeLocalElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB),
                          matrixElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC),
                          safeLocalElements * sizeof(double)));

    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(deviceA, localA.data(),
                              localElements * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(),
                          matrixElements * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> localC(localElements);
    const bool needFullResult = validate || printResults;
    std::vector<double> C;
    if (rank == 0 && needFullResult) {
        C.resize(matrixElements);
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localRows > 0) {
        const dim3 block(static_cast<unsigned int>(kTileColumns),
                         static_cast<unsigned int>(kTileRows), 1);
        const dim3 grid(static_cast<unsigned int>((N + kTileColumns - 1) / kTileColumns),
                        static_cast<unsigned int>((localRows + kTileRows - 1) / kTileRows),
                        1);
        matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC,
                                              static_cast<int>(localRows),
                                              static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(localC.data(), deviceC,
                              localElements * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (needFullResult) {
        if (localElements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            abortWithMessage(rank, "local matrix result exceeds MPI_Gatherv count capacity");
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(static_cast<std::size_t>(worldSize));
            displacements.resize(static_cast<std::size_t>(worldSize));
            for (int source = 0; source < worldSize; ++source) {
                std::size_t sourceFirstRow = 0;
                std::size_t sourceRows = 0;
                calculateRowPartition(N, worldSize, source,
                                      sourceFirstRow, sourceRows);
                const std::size_t sourceElements = sourceRows * N;
                const std::size_t sourceDisplacement = sourceFirstRow * N;
                if (sourceElements > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                    sourceDisplacement > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    abortWithMessage(rank, "matrix is too large for MPI_Gatherv metadata");
                }
                receiveCounts[static_cast<std::size_t>(source)] =
                    static_cast<int>(sourceElements);
                displacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(sourceDisplacement);
            }
        }

        MPI_Gatherv(localC.data(), static_cast<int>(localElements), MPI_DOUBLE,
                    rank == 0 ? C.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    MPI_Comm_free(&localCommunicator);

    if (rank == 0) {
        const long durationMilliseconds = std::max<long>(
            1L, static_cast<long>(std::llround(elapsed * 1000.0)));
        std::printf("Computation time: %ld ms\n", durationMilliseconds);

        const double gflops = (2.0 * static_cast<double>(N) *
                               static_cast<double>(N) * static_cast<double>(N)) /
                              (elapsed > 0.0 ? elapsed : 1e-12) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            std::printf("Validating result...\n");
            std::vector<double> validationA(matrixElements);
            initMatrix(validationA, N);
            const bool valid = validateResult(validationA, B, C, N);
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
