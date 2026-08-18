#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate the same deterministic values as the original benchmark.  The
// global row is passed explicitly so ranks can initialize their own rows.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initRows(std::vector<double>& mat, const size_t N, const size_t firstRow) {
    const size_t rows = N == 0 ? 0 : mat.size() / N;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void transposeMatrix(const std::vector<double>& source, std::vector<double>& transposed,
                     const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            transposed[j * N + i] = source[i * N + j];
        }
    }
}

// Each dot product traverses k in the same order as the original code.  The
// transposed B makes both operands contiguous in the innermost loop.
void matrixMultiplyRows(const std::vector<double>& A, const std::vector<double>& BT,
                        std::vector<double>& C, const size_t rows, const size_t N) {
    for (size_t i = 0; i < rows; ++i) {
        const double* a = A.data() + i * N;
        double* c = C.data() + i * N;
        for (size_t j = 0; j < N; ++j) {
            const double* b = BT.data() + j * N;
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += a[k] * b[k];
            }
            c[j] = sum;
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void splitRows(const size_t N, const int rank, const int world,
                      size_t& firstRow, size_t& rows) {
    const size_t base = N / static_cast<size_t>(world);
    const size_t extra = N % static_cast<size_t>(world);
    rows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    firstRow = static_cast<size_t>(rank) * base +
               std::min(static_cast<size_t>(rank), extra);
}

static int mpiCount(const size_t elements) {
    if (elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(elements);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (N == 0 || N > std::numeric_limits<size_t>::max() / N) {
        if (rank == 0) printf("Matrix size must be a non-zero valid size\n");
        MPI_Finalize();
        return 1;
    }

    size_t firstRow = 0;
    size_t rows = 0;
    splitRows(N, rank, world, firstRow, rows);
    const size_t matrixElements = N * N;
    const int bCount = mpiCount(matrixElements);
    const int localCount = mpiCount(rows * N);
    if (bCount < 0 || localCount < 0) {
        if (rank == 0) printf("Matrix is too large for the MPI implementation\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank zero retains row-major B (needed for validation); all ranks
    // use the transposed copy for the computation.
    std::vector<double> B(rank == 0 ? matrixElements : 0);
    std::vector<double> BT(matrixElements);
    std::vector<double> A(rows * N);
    std::vector<double> C(rows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
        initRows(B, N, 0);
        transposeMatrix(B, BT, N);
    }
    MPI_Bcast(BT.data(), bCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    initRows(A, N, firstRow);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyRows(A, BT, C, rows, N);
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long computationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", computationMs);
        printf("Performance: %.3f GFLOPS\n",
               elapsed > 0.0 ? (2.0 * static_cast<double>(N) * N * N) /
                                  elapsed / 1e9 : 0.0);
    }

    std::vector<double> fullC;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(world);
            displacements.resize(world);
            for (int r = 0; r < world; ++r) {
                size_t begin = 0;
                size_t countRows = 0;
                splitRows(N, r, world, begin, countRows);
                counts[r] = mpiCount(countRows * N);
                displacements[r] = mpiCount(begin * N);
            }
            fullC.resize(matrixElements);
        }
        MPI_Gatherv(C.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? fullC.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0 && printResults) print_results(fullC, "MatrixC");
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        for (size_t pi = 0; pi < 5 && valid; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = pi % N;
                const size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += getPseudoRndValue(N, i, k) * B[k * N + j];
                }
                const double actual = fullC[i * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (relError > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    valid = 0;
                    break;
                }
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
