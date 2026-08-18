#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate the same values as the original benchmark, without distributing a
// full input matrix from rank zero.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct Range {
    size_t begin;
    size_t size;
};

// Contiguous, almost equal partition. This also behaves correctly when there
// are more process rows/columns than matrix rows/columns.
Range partition(const size_t n, const int part, const int parts) {
    const size_t first = (n * static_cast<size_t>(part)) /
                         static_cast<size_t>(parts);
    const size_t last = (n * static_cast<size_t>(part + 1)) /
                        static_cast<size_t>(parts);
    return {first, last - first};
}

void initLocalInputs(std::vector<double>& A, std::vector<double>& B,
                     const size_t N, const Range rows, const Range cols) {
    for (size_t i = 0; i < rows.size; ++i) {
        for (size_t k = 0; k < N; ++k) {
            A[i * N + k] = getPseudoRndValue(N, rows.begin + i, k);
        }
    }
    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < cols.size; ++j) {
            B[k * cols.size + j] =
                getPseudoRndValue(N, k, cols.begin + j);
        }
    }
}

// Cache-tiled outer-product kernel. Each rank calculates its unique C tile.
// k remains ordered, preserving the original dot-product summation semantics.
void matrixMultiply(const std::vector<double>& A,
                    const std::vector<double>& B, std::vector<double>& C,
                    const size_t N, const size_t localRows,
                    const size_t localCols) {
    constexpr size_t rowTile = 16;
    constexpr size_t colTile = 128;

    std::fill(C.begin(), C.end(), 0.0);
    for (size_t ii = 0; ii < localRows; ii += rowTile) {
        const size_t iEnd = std::min(ii + rowTile, localRows);
        for (size_t jj = 0; jj < localCols; jj += colTile) {
            const size_t jEnd = std::min(jj + colTile, localCols);
            for (size_t k = 0; k < N; ++k) {
                const double* const b = B.data() + k * localCols;
                for (size_t i = ii; i < iEnd; ++i) {
                    const double aik = A[i * N + k];
                    double* const c = C.data() + i * localCols;
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += aik * b[j];
                    }
                }
            }
        }
    }
}

bool validateLocalResult(const std::vector<double>& C, const size_t N,
                         const Range rows, const Range cols) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;
    for (const size_t pi : checkPoints) {
        const size_t globalI = pi % N;
        if (globalI < rows.begin || globalI >= rows.begin + rows.size) {
            continue;
        }
        for (const size_t pj : checkPoints) {
            const size_t globalJ = pj % N;
            if (globalJ < cols.begin || globalJ >= cols.begin + cols.size) {
                continue;
            }
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, globalI, k) *
                            getPseudoRndValue(N, k, globalJ);
            }
            const double actual =
                C[(globalI - rows.begin) * cols.size + globalJ - cols.begin];
            const double error =
                std::abs((actual - expected) / (expected + 1e-10));
            valid = valid && error <= 1e-6;
        }
    }
    return valid;
}

void sendDoubles(const double* data, size_t count, const int destination) {
    while (count != 0) {
        const int chunk = static_cast<int>(
            std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveDoubles(double* data, size_t count, const int source) {
    while (count != 0) {
        const int chunk = static_cast<int>(
            std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

// Gather packed 2D tiles only when the user requests externally visible
// results. Keeping this outside the timed region avoids penalizing the kernel.
std::vector<double> gatherResult(const std::vector<double>& localC,
                                 const size_t N, const int rank,
                                 const int worldSize, const int processRows,
                                 const int processCols) {
    if (rank != 0) {
        sendDoubles(localC.data(), localC.size(), 0);
        return {};
    }

    std::vector<double> globalC(N * N);
    std::vector<double> incoming;
    for (int source = 0; source < worldSize; ++source) {
        const int rowCoordinate = source / processCols;
        const int colCoordinate = source % processCols;
        const Range rows = partition(N, rowCoordinate, processRows);
        const Range cols = partition(N, colCoordinate, processCols);
        const size_t count = rows.size * cols.size;
        const double* tile = nullptr;
        if (source == 0) {
            tile = localC.data();
        } else {
            incoming.resize(count);
            receiveDoubles(incoming.data(), count, source);
            tile = incoming.data();
        }
        for (size_t i = 0; i < rows.size; ++i) {
            std::copy_n(tile + i * cols.size, cols.size,
                        globalC.data() + (rows.begin + i) * N + cols.begin);
        }
    }
    return globalC;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
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
            errno = 0;
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argumentsValid = errno == 0 && end != argv[i] && *end == '\0' &&
                             value != 0 &&
                             value <= std::numeric_limits<size_t>::max();
            if (argumentsValid) {
                N = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
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
    if (N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) {
            std::fprintf(stderr, "Matrix size is too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int rowCoordinate = rank / dimensions[1];
    const int colCoordinate = rank % dimensions[1];
    const Range rows = partition(N, rowCoordinate, dimensions[0]);
    const Range cols = partition(N, colCoordinate, dimensions[1]);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d grid)\n", worldSize,
                    dimensions[0], dimensions[1]);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    try {
        A.resize(rows.size * N);
        B.resize(N * cols.size);
        C.resize(rows.size * cols.size);
        initLocalInputs(A, B, N, rows, cols);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Rank %d: insufficient memory for local matrices.\n",
                     rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows.size, cols.size);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        const double operations = 2.0 * static_cast<double>(N) *
                                  static_cast<double>(N) *
                                  static_cast<double>(N);
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n",
                    seconds > 0.0 ? operations / seconds / 1e9 : 0.0);
    }

    if (printResults) {
        std::vector<double> globalC = gatherResult(
            C, N, rank, worldSize, dimensions[0], dimensions[1]);
        if (rank == 0) {
            print_results(globalC, "MatrixC");
        }
    }

    int localValid = 1;
    if (validate) {
        localValid = validateLocalResult(C, N, rows, cols) ? 1 : 0;
    }
    int globallyValid = 1;
    MPI_Reduce(&localValid, &globallyValid, 1, MPI_INT, MPI_MIN, 0,
               MPI_COMM_WORLD);
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return validate && !globallyValid ? 1 : 0;
}
