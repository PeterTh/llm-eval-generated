#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fills `rowCount` rows of an N-column matrix starting at global row `rowStart`
void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowStart + i, j);
        }
    }
}

// Row-block distribution: rank r owns rows [start, start + count) of C
struct RowBlock {
    size_t start;
    size_t count;
};

RowBlock rowBlock(const size_t N, const int rank, const int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    const size_t base = N / p;
    const size_t rem = N % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Computes C = A * B for the local `rows` rows of A (rows x N, row-major).
// Uses i-k-j order so the inner loop is contiguous and vectorizes, with four
// rows of C updated per pass over B and k tiled for cache reuse. Each C element
// still accumulates its k terms in ascending order, as in the naive triple loop.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t rows, const size_t N) {
    constexpr size_t kTile = 256;

    std::fill(C.begin(), C.begin() + rows * N, 0.0);
    for (size_t kk = 0; kk < N; kk += kTile) {
        const size_t kEnd = std::min(kk + kTile, N);
        size_t i = 0;
        for (; i + 4 <= rows; i += 4) {
            double* c0 = &C[(i + 0) * N];
            double* c1 = &C[(i + 1) * N];
            double* c2 = &C[(i + 2) * N];
            double* c3 = &C[(i + 3) * N];
            for (size_t k = kk; k < kEnd; ++k) {
                const double a0 = A[(i + 0) * N + k];
                const double a1 = A[(i + 1) * N + k];
                const double a2 = A[(i + 2) * N + k];
                const double a3 = A[(i + 3) * N + k];
                const double* b = &B[k * N];
                for (size_t j = 0; j < N; ++j) {
                    c0[j] += a0 * b[j];
                    c1[j] += a1 * b[j];
                    c2[j] += a2 * b[j];
                    c3[j] += a3 * b[j];
                }
            }
        }
        for (; i < rows; ++i) {
            double* c = &C[i * N];
            for (size_t k = kk; k < kEnd; ++k) {
                const double a = A[i * N + k];
                const double* b = &B[k * N];
                for (size_t j = 0; j < N; ++j) {
                    c[j] += a * b[j];
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Only the root gathers the full C matrix, needed for -r and -v (MPI counts are int)
    const bool gather = printResults || validate;
    if (gather && N * N > static_cast<size_t>(INT_MAX)) {
        if (root) {
            fprintf(stderr, "Matrix size %zu too large to gather with -r/-v\n", N);
        }
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Each rank holds its row block of A and C, plus a full copy of B
    const RowBlock block = rowBlock(N, rank, size);
    std::vector<double> A(block.count * N);
    std::vector<double> B(N * N);
    std::vector<double> C(block.count * N);

    initMatrix(A, N, block.start, block.count);
    initMatrix(B, N, 0, N);

    // Perform matrix multiplication
    if (root) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, block.count, N);

    auto end = std::chrono::high_resolution_clock::now();

    // The parallel run takes as long as its slowest rank
    long long localNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long maxNs = 0;
    MPI_Reduce(&localNs, &maxNs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full C matrix on the root
    std::vector<double> fullC;
    if (gather) {
        if (root) {
            fullC.resize(N * N);
        }
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const RowBlock rb = rowBlock(N, r, size);
            counts[r] = static_cast<int>(rb.count * N);
            displs[r] = static_cast<int>(rb.start * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(block.count * N), MPI_DOUBLE,
                    fullC.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(maxNs));
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(fullC, "MatrixC");
        }
    }

    // Validation
    int status = 0;
    if (root && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(B, fullC, N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    // Propagate the root's exit status to every rank
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
