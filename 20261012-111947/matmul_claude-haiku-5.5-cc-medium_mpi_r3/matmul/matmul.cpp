#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cache blocking parameters for the local multiply kernel
constexpr size_t kBlockK = 128;
constexpr size_t kBlockJ = 512;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowBegin + rowCount) of an N x N matrix. The local
// buffer holds only those rows, so mat has rowCount * N elements.
void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowBegin + i, j);
        }
    }
}

// First row owned by rank r when N rows are split as evenly as possible across p ranks
constexpr size_t rowBegin(const int r, const int p, const size_t N) noexcept {
    return r * (N / p) + std::min(static_cast<size_t>(r), N % p);
}

// Number of rows owned by rank r
constexpr size_t rowCount(const int r, const int p, const size_t N) noexcept {
    return N / p + (static_cast<size_t>(r) < N % p ? 1 : 0);
}

// Local block multiply: C (rows x N) = A (rows x N) * B (N x N), all row-major.
// For every C element the products are still accumulated in increasing k order,
// exactly as in the sequential reference, so results are bit-identical for any
// process count. The loop nest is i-k-j so the inner loop is contiguous and vectorizes.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    std::fill(C, C + rows * N, 0.0);

    for (size_t k0 = 0; k0 < N; k0 += kBlockK) {
        const size_t k1 = std::min(k0 + kBlockK, N);
        for (size_t j0 = 0; j0 < N; j0 += kBlockJ) {
            const size_t j1 = std::min(j0 + kBlockJ, N);

            // Four rows of C at a time so each loaded B row is reused four times
            size_t i = 0;
            for (; i + 4 <= rows; i += 4) {
                double* c0 = C + (i + 0) * N;
                double* c1 = C + (i + 1) * N;
                double* c2 = C + (i + 2) * N;
                double* c3 = C + (i + 3) * N;
                const double* a0 = A + (i + 0) * N;
                const double* a1 = A + (i + 1) * N;
                const double* a2 = A + (i + 2) * N;
                const double* a3 = A + (i + 3) * N;
                for (size_t k = k0; k < k1; ++k) {
                    const double x0 = a0[k];
                    const double x1 = a1[k];
                    const double x2 = a2[k];
                    const double x3 = a3[k];
                    const double* b = B + k * N;
                    for (size_t j = j0; j < j1; ++j) {
                        const double bj = b[j];
                        c0[j] += x0 * bj;
                        c1[j] += x1 * bj;
                        c2[j] += x2 * bj;
                        c3[j] += x3 * bj;
                    }
                }
            }

            // Remaining rows
            for (; i < rows; ++i) {
                double* c = C + i * N;
                const double* a = A + i * N;
                for (size_t k = k0; k < k1; ++k) {
                    const double x = a[k];
                    const double* b = B + k * N;
                    for (size_t j = j0; j < j1; ++j) {
                        c[j] += x * b[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare.
// Takes the full result matrix C (N x N). A and B are regenerated from their
// closed-form values, which is identical to the initialized matrices.
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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

    // Parse command line arguments (identical on every rank)
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

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank owns a contiguous block of rows of A and of C. B is needed in full
    // by every rank and is generated locally, so no input data is communicated.
    const size_t myBegin = rowBegin(rank, size, N);
    const size_t myRows = rowCount(rank, size, N);

    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(myRows * N);

    // Initialize matrices
    if (root) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N, myBegin, myRows);
    initMatrix(B, N, 0, N);

    // Perform matrix multiplication
    if (root) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(A.data(), B.data(), C.data(), myRows, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The slowest rank determines the elapsed time
    long long localMs = duration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    bool valid = true;
    if (printResults || validate) {
        // Gather the full result matrix on rank 0
        std::vector<double> full;
        std::vector<int> counts;
        std::vector<int> displs;
        if (root) {
            full.resize(N * N);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                counts[r] = static_cast<int>(rowCount(r, size, N) * N);
                displs[r] = static_cast<int>(rowBegin(r, size, N) * N);
            }
        }
        MPI_Gatherv(C.data(), static_cast<int>(myRows * N), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(full, N);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
