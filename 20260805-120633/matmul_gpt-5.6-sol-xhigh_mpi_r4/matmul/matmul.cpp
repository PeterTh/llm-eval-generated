#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct Block {
    size_t begin;
    size_t size;
};

// Divide n elements as evenly as possible, including when parts > n.
Block getBlock(const size_t n, const int coordinate, const int parts) {
    const size_t quotient = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    return {coord * quotient + std::min(coord, remainder),
            quotient + (coord < remainder ? 1U : 0U)};
}

// Each process owns the A rows and B columns needed for its C tile.  The
// deterministic initializer lets every process construct those pieces in
// place, avoiding distribution collectives and a root-memory bottleneck.
void initLocalMatrices(std::vector<double>& localA,
                       std::vector<double>& localB, const size_t N,
                       const Block rows, const Block columns) {
    for (size_t localI = 0; localI < rows.size; ++localI) {
        const size_t globalI = rows.begin + localI;
        for (size_t k = 0; k < N; ++k) {
            localA[localI * N + k] = getPseudoRndValue(N, globalI, k);
        }
    }

    // B is stored as N x localColumns.  Thus B and C are contiguous in the
    // inner j loop of the multiplication kernel.
    for (size_t k = 0; k < N; ++k) {
        for (size_t localJ = 0; localJ < columns.size; ++localJ) {
            const size_t globalJ = columns.begin + localJ;
            localB[k * columns.size + localJ] =
                getPseudoRndValue(N, k, globalJ);
        }
    }
}

// Cache-blocked matrix multiplication for the process-local C tile.  k is
// ordered exactly as in the scalar implementation for every C element, while
// j is independent and can be vectorized without a floating-point reduction.
void matrixMultiply(const double* __restrict__ localA,
                    const double* __restrict__ localB,
                    double* __restrict__ localC, const size_t N,
                    const size_t localRows, const size_t localColumns) {
    constexpr size_t rowTile = 32;
    constexpr size_t columnTile = 128;
    constexpr size_t innerTile = 128;

    for (size_t ii = 0; ii < localRows; ii += rowTile) {
        const size_t iEnd = std::min(ii + rowTile, localRows);
        for (size_t jj = 0; jj < localColumns; jj += columnTile) {
            const size_t jEnd = std::min(jj + columnTile, localColumns);
            for (size_t kk = 0; kk < N; kk += innerTile) {
                const size_t kEnd = std::min(kk + innerTile, N);
                for (size_t k = kk; k < kEnd; ++k) {
                    const double* __restrict__ bRow =
                        localB + k * localColumns;
                    for (size_t i = ii; i < iEnd; ++i) {
                        const double a = localA[i * N + k];
                        double* __restrict__ cRow =
                            localC + i * localColumns;
#pragma omp simd
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += a * bRow[j];
                        }
                    }
                }
            }
        }
    }
}

bool validateLocalResult(const std::vector<double>& localA,
                         const std::vector<double>& localB,
                         const std::vector<double>& localC, const size_t N,
                         const Block rows, const Block columns,
                         const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (const size_t rowPoint : checkPoints) {
        const size_t globalI = rowPoint % N;
        if (globalI < rows.begin || globalI >= rows.begin + rows.size) {
            continue;
        }
        const size_t localI = globalI - rows.begin;

        for (const size_t columnPoint : checkPoints) {
            const size_t globalJ = columnPoint % N;
            if (globalJ < columns.begin ||
                globalJ >= columns.begin + columns.size) {
                continue;
            }
            const size_t localJ = globalJ - columns.begin;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += localA[localI * N + k] *
                            localB[k * columns.size + localJ];
            }

            const double actual =
                localC[localI * columns.size + localJ];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf(
                    "Validation failed on rank %d at (%zu, %zu): expected "
                    "%.10f, got %.10f (error: %.10e)\n",
                    rank, globalI, globalJ, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

// MPI uses int counts in the portable interfaces available on MPI-3 systems.
// Chunking also lets the benchmark handle rows wider than INT_MAX elements.
void sendLarge(const double* data, size_t count, const int destination,
               const int tag, MPI_Comm communicator) {
    constexpr size_t maxCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, maxCount));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, tag, communicator);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveLarge(double* data, size_t count, const int source, const int tag,
                  MPI_Comm communicator) {
    constexpr size_t maxCount =
        static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, maxCount));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, tag, communicator,
                 MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

// Assemble C only when the user requests result output.  Rows are received
// directly into their final locations, so rank zero needs no packed staging
// buffer and MPI int-count limits do not constrain the matrix size.
std::vector<double> gatherResult(const std::vector<double>& localC,
                                 const size_t N, const Block localRows,
                                 const Block localColumns, const int rank,
                                 const int worldSize, const int gridRows,
                                 const int gridColumns,
                                 MPI_Comm communicator) {
    constexpr int resultTag = 1701;

    if (rank != 0) {
        if (localColumns.size != 0) {
            for (size_t i = 0; i < localRows.size; ++i) {
                sendLarge(localC.data() + i * localColumns.size,
                          localColumns.size, 0, resultTag, communicator);
            }
        }
        return {};
    }

    std::vector<double> result(N * N);
    for (int source = 0; source < worldSize; ++source) {
        const int rowCoordinate = source / gridColumns;
        const int columnCoordinate = source % gridColumns;
        const Block rows = getBlock(N, rowCoordinate, gridRows);
        const Block columns = getBlock(N, columnCoordinate, gridColumns);

        if (columns.size == 0) {
            continue;
        }
        for (size_t i = 0; i < rows.size; ++i) {
            double* destination =
                result.data() + (rows.begin + i) * N + columns.begin;
            if (source == 0) {
                std::copy_n(localC.data() + i * localColumns.size,
                            columns.size, destination);
            } else {
                receiveLarge(destination, columns.size, source, resultTag,
                             communicator);
            }
        }
    }

    return result;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsed = std::atoll(argv[++i]);
            if (parsed <= 0) {
                argumentsValid = false;
            } else {
                N = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            if (!argumentsValid) {
                std::printf("Matrix size must be a positive integer.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    int grid[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, grid);
    const int rowCoordinate = rank / grid[1];
    const int columnCoordinate = rank % grid[1];
    const Block localRows = getBlock(N, rowCoordinate, grid[0]);
    const Block localColumns = getBlock(N, columnCoordinate, grid[1]);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d grid)\n", worldSize, grid[0],
                    grid[1]);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> localA(localRows.size * N);
    std::vector<double> localB(N * localColumns.size);
    std::vector<double> localC(localRows.size * localColumns.size, 0.0);
    initLocalMatrices(localA, localB, N, localRows, localColumns);

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiply(localA.data(), localB.data(), localC.data(), N,
                   localRows.size, localColumns.size);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(elapsed * 1000.0);
        const double n = static_cast<double>(N);
        const double gflops = (2.0 * n * n * n) / elapsed / 1e9;
        std::printf("Computation time: %lld ms\n", durationMs);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> result =
            gatherResult(localC, N, localRows, localColumns, rank, worldSize,
                         grid[0], grid[1], MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }

    int globallyValid = 1;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const int locallyValid =
            validateLocalResult(localA, localB, localC, N, localRows,
                                localColumns, rank)
                ? 1
                : 0;
        MPI_Allreduce(&locallyValid, &globallyValid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validation: %s\n",
                        globallyValid != 0 ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return globallyValid != 0 ? 0 : 1;
}
