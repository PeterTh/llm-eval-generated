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

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

[[noreturn]] void abortWithMessage(const int rank, const char* const message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const int rank, const char* const operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

void checkCublas(const cublasStatus_t status, const int rank, const char* const operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s failed with cuBLAS status %d\n", rank, operation,
                     static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

struct Partition {
    size_t offset;
    size_t count;
};

// A contiguous, balanced partition of [0, total).
Partition partition(const size_t total, const int parts, const int index) {
    const size_t base = total / static_cast<size_t>(parts);
    const size_t remainder = total % static_cast<size_t>(parts);
    const size_t count = base + (static_cast<size_t>(index) < remainder ? 1 : 0);
    const size_t offset = static_cast<size_t>(index) * base +
                          std::min(static_cast<size_t>(index), remainder);
    return {offset, count};
}

// MPI count arguments are int.  Keeping collective chunks below INT_MAX makes
// the benchmark usable for matrices larger than a single MPI count permits.
void broadcastDoubles(double* const values, const size_t count, const int root,
                      const MPI_Comm communicator) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        if (MPI_Bcast(values + offset, chunk, MPI_DOUBLE, root, communicator) != MPI_SUCCESS) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            abortWithMessage(rank, "MPI_Bcast failed");
        }
    }
}

void sendDoubles(const double* const values, const size_t count, const int destination,
                 const int tag, const MPI_Comm communicator, const int rank) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        if (MPI_Send(values + offset, chunk, MPI_DOUBLE, destination, tag, communicator) !=
            MPI_SUCCESS) {
            abortWithMessage(rank, "MPI_Send failed");
        }
    }
}

void receiveDoubles(double* const values, const size_t count, const int source, const int tag,
                    const MPI_Comm communicator, const int rank) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += maxChunk) {
        const int chunk = static_cast<int>(std::min(maxChunk, count - offset));
        if (MPI_Recv(values + offset, chunk, MPI_DOUBLE, source, tag, communicator,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            abortWithMessage(rank, "MPI_Recv failed");
        }
    }
}

// Each row of the process grid shares one A panel.  Only the rank at column
// zero generates it; OpenMP makes this host-side preparation parallel too.
void initializeRows(std::vector<double>& matrix, const size_t N, const Partition rows) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0; localRow < static_cast<long long>(rows.count); ++localRow) {
        const size_t globalRow = rows.offset + static_cast<size_t>(localRow);
        double* const row = matrix.data() + static_cast<size_t>(localRow) * N;
        for (size_t column = 0; column < N; ++column) {
            row[column] = getPseudoRndValue(N, globalRow, column);
        }
    }
}

// Each column of the process grid shares one B panel.  Only the rank at row
// zero generates it; B is stored compactly as an N by local-column matrix.
void initializeColumns(std::vector<double>& matrix, const size_t N, const Partition columns) {
#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(N); ++row) {
        double* const localRow = matrix.data() + static_cast<size_t>(row) * columns.count;
        for (size_t localColumn = 0; localColumn < columns.count; ++localColumn) {
            localRow[localColumn] =
                getPseudoRndValue(N, static_cast<size_t>(row), columns.offset + localColumn);
        }
    }
}

// Simple validation of the original mathematical operation.  The distributed
// inputs are generated from the same deterministic formula, so root can check
// any output element without reconstructing full A and B first.
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t pointI : checkPoints) {
        for (const size_t pointJ : checkPoints) {
            const size_t i = pointI % N;
            const size_t j = pointJ % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf(
                    "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                    i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* const progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Collect distributed rectangular C tiles only when the original interface
// needs the full matrix on rank zero (-r or -v).  The normal benchmark path
// keeps C distributed, avoiding an O(N^2) root-side allocation and transfer.
std::vector<double> gatherResult(const std::vector<double>& localC, const size_t N,
                                 const Partition rows, const Partition columns,
                                 const int rank, const int worldSize, const int dims[2],
                                 const MPI_Comm grid) {
    constexpr int resultTag = 991;
    const size_t localCount = rows.count * columns.count;

    if (rank != 0) {
        sendDoubles(localC.data(), localCount, 0, resultTag, grid, rank);
        return {};
    }

    std::vector<double> result(N * N);
    auto insertTile = [&result, N](const double* const tile, const Partition tileRows,
                                   const Partition tileColumns) {
        for (size_t row = 0; row < tileRows.count; ++row) {
            std::memcpy(result.data() + (tileRows.offset + row) * N + tileColumns.offset,
                        tile + row * tileColumns.count, tileColumns.count * sizeof(double));
        }
    };

    insertTile(localC.data(), rows, columns);
    for (int source = 1; source < worldSize; ++source) {
        int coordinates[2] = {0, 0};
        if (MPI_Cart_coords(grid, source, 2, coordinates) != MPI_SUCCESS) {
            abortWithMessage(rank, "MPI_Cart_coords failed");
        }

        const Partition sourceRows = partition(N, dims[0], coordinates[0]);
        const Partition sourceColumns = partition(N, dims[1], coordinates[1]);
        std::vector<double> tile(sourceRows.count * sourceColumns.count);
        receiveDoubles(tile.data(), tile.size(), source, resultTag, grid, rank);
        insertTile(tile.data(), sourceRows, sourceColumns);
    }

    return result;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool helpRequested = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            helpRequested = true;
        } else {
            argumentError = true;
        }
    }

    if (helpRequested || argumentError || N == 0) {
        if (rank == 0) {
            if (N == 0 && !helpRequested && !argumentError) {
                std::printf("Matrix size must be greater than zero\n");
            } else if (argumentError) {
                std::printf("Unknown or incomplete option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError || N == 0 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    int dims[2] = {0, 0};
    if (MPI_Dims_create(worldSize, 2, dims) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Dims_create failed");
    }
    const int periods[2] = {0, 0};
    MPI_Comm grid = MPI_COMM_NULL;
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid) != MPI_SUCCESS ||
        grid == MPI_COMM_NULL) {
        abortWithMessage(rank, "MPI_Cart_create failed");
    }

    int coordinates[2] = {0, 0};
    if (MPI_Cart_coords(grid, rank, 2, coordinates) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Cart_coords failed");
    }
    const Partition rows = partition(N, dims[0], coordinates[0]);
    const Partition columns = partition(N, dims[1], coordinates[1]);

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    if (MPI_Comm_split(grid, coordinates[0], coordinates[1], &rowCommunicator) != MPI_SUCCESS ||
        MPI_Comm_split(grid, coordinates[1], coordinates[0], &columnCommunicator) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Comm_split failed");
    }

    // Map ranks to devices locally.  In the normal one-rank-per-GPU launch,
    // localRank selects the matching accelerator; modulo also makes accidental
    // oversubscription deterministic instead of silently using device zero.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &localCommunicator) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Comm_split_type failed");
    }
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) {
        abortWithMessage(rank, "no CUDA device is visible to this MPI rank");
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "cudaSetDevice");

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI process grid: %d x %d\n", dims[0], dims[1]);
        std::printf("Initializing matrices...\n");
    }

    // A is replicated only within a process-grid row and B only within a
    // process-grid column.  This has O(N^2/sqrt(P)) input storage per rank for
    // the near-square grid selected by MPI_Dims_create.
    std::vector<double> localA(rows.count * N);
    std::vector<double> localB(N * columns.count);
    std::vector<double> localC(rows.count * columns.count);

    if (coordinates[1] == 0) {
        initializeRows(localA, N, rows);
    }
    broadcastDoubles(localA.data(), localA.size(), 0, rowCommunicator);

    if (coordinates[0] == 0) {
        initializeColumns(localB, N, columns);
    }
    broadcastDoubles(localB.data(), localB.size(), 0, columnCommunicator);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    const size_t aBytes = localA.size() * sizeof(double);
    const size_t bBytes = localB.size() * sizeof(double);
    const size_t cBytes = localC.size() * sizeof(double);
    if (aBytes != 0) {
        checkCuda(cudaMalloc(&deviceA, aBytes), rank, "cudaMalloc(A)");
        checkCuda(cudaMemcpy(deviceA, localA.data(), aBytes, cudaMemcpyHostToDevice), rank,
                  "copying A to device");
    }
    if (bBytes != 0) {
        checkCuda(cudaMalloc(&deviceB, bBytes), rank, "cudaMalloc(B)");
        checkCuda(cudaMemcpy(deviceB, localB.data(), bBytes, cudaMemcpyHostToDevice), rank,
                  "copying B to device");
    }
    if (cBytes != 0) {
        checkCuda(cudaMalloc(&deviceC, cBytes), rank, "cudaMalloc(C)");
    }

    cublasHandle_t cublas = nullptr;
    checkCublas(cublasCreate(&cublas), rank, "cublasCreate");

    // cuBLAS creates internal handles and selects kernels lazily.  Prime that
    // one-time work before the synchronized measurement so the reported rate
    // measures the distributed DGEMM, rather than CUDA/cuBLAS startup cost.
    if (rows.count != 0 && columns.count != 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(columns.count), static_cast<int>(rows.count),
                                static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(columns.count), deviceA, static_cast<int>(N),
                                &beta, deviceC, static_cast<int>(columns.count)),
                    rank, "warm-up cublasDgemm");
        checkCuda(cudaDeviceSynchronize(), rank, "warm-up cudaDeviceSynchronize");
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    checkCuda(cudaEventCreate(&startEvent), rank, "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stopEvent), rank, "cudaEventCreate(stop)");

    if (MPI_Barrier(grid) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Barrier failed");
    }
    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }

    checkCuda(cudaEventRecord(startEvent), rank, "cudaEventRecord(start)");
    if (rows.count != 0 && columns.count != 0) {
        // Row-major C = A * B is equivalent to the column-major product
        // C^T = B^T * A^T.  Swapping operands lets cuBLAS consume the compact
        // row-major panels directly, without an expensive transpose/copy.
        const double alpha = 1.0;
        const double beta = 0.0;
        checkCublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(columns.count), static_cast<int>(rows.count),
                                static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(columns.count), deviceA, static_cast<int>(N),
                                &beta, deviceC, static_cast<int>(columns.count)),
                    rank, "cublasDgemm");
    }
    checkCuda(cudaEventRecord(stopEvent), rank, "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stopEvent), rank, "cudaEventSynchronize(stop)");

    float localMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&localMilliseconds, startEvent, stopEvent), rank,
              "cudaEventElapsedTime");
    const double localMillisecondsDouble = static_cast<double>(localMilliseconds);
    double maximumMilliseconds = 0.0;
    if (MPI_Reduce(&localMillisecondsDouble, &maximumMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                   grid) != MPI_SUCCESS) {
        abortWithMessage(rank, "MPI_Reduce failed");
    }

    if (cBytes != 0) {
        checkCuda(cudaMemcpy(localC.data(), deviceC, cBytes, cudaMemcpyDeviceToHost), rank,
                  "copying C from device");
    }

    if (rank == 0) {
        const long roundedMilliseconds = static_cast<long>(std::llround(maximumMilliseconds));
        const double seconds = std::max(maximumMilliseconds / 1000.0,
                                        std::numeric_limits<double>::min());
        const double gflops =
            (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) /
            seconds / 1.0e9;
        std::printf("Computation time: %ld ms\n", roundedMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (printResults || validate) {
        result = gatherResult(localC, N, rows, columns, rank, worldSize, dims, grid);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0 && printResults) {
        print_results(result, "MatrixC");
    }
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(result, N)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = EXIT_FAILURE;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, grid);

    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);
    cublasDestroy(cublas);
    cudaFree(deviceC);
    cudaFree(deviceB);
    cudaFree(deviceA);
    MPI_Comm_free(&localCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return exitCode;
}
