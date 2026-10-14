#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Half-open range of matrix rows [begin, end)
struct RowRange {
    size_t begin;
    size_t end;
};

// Block distribution of N rows over `size` ranks
RowRange rowRange(const size_t N, const int rank, const int size) noexcept {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    return {N * r / p, N * (r + 1) / p};
}

// Fill rows [rows.begin, rows.end) of an N x N matrix. `mat` holds only those rows.
void initRows(std::vector<double>& mat, const size_t N, const RowRange rows) {
    for (size_t i = rows.begin; i < rows.end; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rows.begin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// C = A * B for the locally owned rows. A and C hold `localRows` rows, B is the full N x N matrix.
// The loops are tiled and ordered i-k-j for cache reuse and vectorization. Each C(i,j) still
// accumulates its k-terms in ascending order, so results are bit-identical to the naive product.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const size_t localRows) {
    constexpr size_t KB = 128;
    constexpr size_t JB = 256;

    std::fill(C.begin(), C.end(), 0.0);
    for (size_t jj = 0; jj < N; jj += JB) {
        const size_t jEnd = std::min(jj + JB, N);
        for (size_t kk = 0; kk < N; kk += KB) {
            const size_t kEnd = std::min(kk + KB, N);
            for (size_t i = 0; i < localRows; ++i) {
                double* c = &C[i * N];
                for (size_t k = kk; k < kEnd; ++k) {
                    const double a = A[i * N + k];
                    const double* b = &B[k * N];
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
// A(i,k) is identical to getPseudoRndValue(N, i, k), so it is recomputed here instead of stored.
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
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
            if (root) printUsage(argv[0]);
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank owns a contiguous block of rows of A and C. B is replicated on every rank.
    const RowRange rows = rowRange(N, rank, size);
    const size_t localRows = rows.end - rows.begin;

    // Allocate matrices
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);

    // Initialize matrices (deterministic, so every rank derives the same values it needs)
    if (root) printf("Initializing matrices...\n");
    initRows(A, N, rows);
    initRows(B, N, RowRange{0, N});

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiply(A, B, C, N, localRows);

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    // The parallel run takes as long as its slowest rank
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C on the root only when it is needed for output or validation
    const bool needFullC = printResults || validate;
    std::vector<double> Cfull;
    if (needFullC) {
        // Row-typed transfers keep counts in rows, so N*N may exceed INT_MAX
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const RowRange rr = rowRange(N, r, size);
            counts[r] = static_cast<int>(rr.end - rr.begin);
            displs[r] = static_cast<int>(rr.begin);
        }

        if (root) Cfull.resize(N * N);
        MPI_Gatherv(C.data(), static_cast<int>(localRows), rowType,
                    Cfull.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }

    if (root) {
        // Print results for external validation
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B, Cfull, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                return 1;
            }
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int status = runBenchmark(argc, argv, rank, size);
    // Every rank exits with the root's status, e.g. when validation fails on the root
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return status;
}
