#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cache blocking factor over the k dimension: a block of B rows stays hot while all local rows sweep it
constexpr size_t kBlock = 128;

// Sample positions checked during validation
constexpr size_t kCheckPoints[] = {0, 1, 2, 3, 4};

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// First row owned by `rank` under a balanced block distribution; rank == size yields N
constexpr size_t rowBegin(const size_t N, const int rank, const int size) noexcept {
    return N * static_cast<size_t>(rank) / static_cast<size_t>(size);
}

// Fills rows [rowBegin, rowEnd) of N columns; mat is indexed relative to rowBegin
void initRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Computes the local block C = A * B, where A and C hold `rows` rows and B is the full matrix.
// Each C[i][j] accumulates its k terms in ascending order, the same operation sequence as the original
// ijk kernel; the j loop is the innermost, so it vectorizes without reordering any sum.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t rows) {
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t kb = 0; kb < N; kb += kBlock) {
        const size_t kEnd = std::min(N, kb + kBlock);
        for (size_t i = 0; i < rows; ++i) {
            double* c = &C[i * N];
            for (size_t k = kb; k < kEnd; ++k) {
                const double a = A[i * N + k];
                const double* b = &B[k * N];
                for (size_t j = 0; j < N; ++j) {
                    c[j] += a * b[j];
                }
            }
        }
    }
}

// Simple validation: recompute sampled entries and compare.
// actual[pi][pj] holds C[i][j] for i = kCheckPoints[pi] % N, j = kCheckPoints[pj] % N.
bool validateResult(const size_t N, const double actual[5][5]) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = kCheckPoints[pi] % N;
            const size_t j = kCheckPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double relError = std::abs((actual[pi][pj] - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual[pi][pj], relError);
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

int run(int argc, char** argv, int rank, int size) {
    const bool root = rank == 0;
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks take the same path)
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Each rank owns a contiguous block of rows of A and C; B is needed in full
    const size_t r0 = rowBegin(N, rank, size);
    const size_t r1 = rowBegin(N, rank + 1, size);
    const size_t rows = r1 - r0;

    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rows * N);

    initRows(A, N, r0, r1);
    initRows(B, N, 0, N);

    // Perform matrix multiplication
    if (root) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiply(A, B, C, N, rows);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t begin = rowBegin(N, r, size);
            const size_t end = rowBegin(N, r + 1, size);
            counts[r] = static_cast<int>((end - begin) * N);
            displs[r] = static_cast<int>(begin * N);
        }

        std::vector<double> full(root ? N * N : 0);
        MPI_Gatherv(C.data(), static_cast<int>(rows * N), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            print_results(full, "MatrixC");
        }
    }

    // Validation
    if (validate) {
        // Each sampled entry is owned by exactly one rank; a sum-reduction (with zeros elsewhere) assembles them
        double sample[5][5] = {};
        double total[5][5] = {};
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = kCheckPoints[pi] % N;
                const size_t j = kCheckPoints[pj] % N;
                if (i >= r0 && i < r1) {
                    sample[pi][pj] = C[(i - r0) * N + j];
                }
            }
        }
        MPI_Reduce(&sample[0][0], &total[0][0], 25, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        int valid = 1;
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(N, total) ? 1 : 0;

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = run(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
