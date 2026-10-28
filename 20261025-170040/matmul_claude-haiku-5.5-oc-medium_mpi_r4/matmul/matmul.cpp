#include <mpi.h>

#include <algorithm>
#include <chrono>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Row-block decomposition of N rows across `size` ranks.
// The first (N % size) ranks receive one extra row.
struct RowRange {
    size_t begin;
    size_t count;
};

RowRange rowRangeFor(const size_t N, const int rank, const int size) noexcept {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    const size_t base = N / p;
    const size_t rem = N % p;
    return RowRange{r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Computes rows [rowBegin, rowBegin + rowCount) of C = A * B.
// A and B are full NxN row-major matrices; C is a local rowCount x N row-major buffer.
//
// For every C[i][j] the products A[i][k]*B[k][j] are accumulated in strictly
// ascending k order starting from 0.0, exactly as in the sequential reference,
// so results are bitwise identical regardless of the number of ranks.
// Blocking and register reuse only change the order in which different (i, j)
// elements are visited, never the summation order of an individual element.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t N,
                        const size_t rowBegin, const size_t rowCount) {
    constexpr size_t BK = 128;  // k-block: B panel of BK x BJ stays cache resident
    constexpr size_t BJ = 256;  // j-block
    constexpr size_t RI = 4;    // rows of C processed together (reuses each B load 4x)

    std::fill(C, C + rowCount * N, 0.0);

    for (size_t kk = 0; kk < N; kk += BK) {
        const size_t kEnd = std::min(kk + BK, N);
        for (size_t jj = 0; jj < N; jj += BJ) {
            const size_t jEnd = std::min(jj + BJ, N);

            size_t i = 0;
            for (; i + RI <= rowCount; i += RI) {
                const double* a0 = A + (rowBegin + i + 0) * N;
                const double* a1 = A + (rowBegin + i + 1) * N;
                const double* a2 = A + (rowBegin + i + 2) * N;
                const double* a3 = A + (rowBegin + i + 3) * N;
                double* c0 = C + (i + 0) * N;
                double* c1 = C + (i + 1) * N;
                double* c2 = C + (i + 2) * N;
                double* c3 = C + (i + 3) * N;

                for (size_t k = kk; k < kEnd; ++k) {
                    const double* b = B + k * N;
                    const double x0 = a0[k];
                    const double x1 = a1[k];
                    const double x2 = a2[k];
                    const double x3 = a3[k];
                    for (size_t j = jj; j < jEnd; ++j) {
                        const double bj = b[j];
                        c0[j] += x0 * bj;
                        c1[j] += x1 * bj;
                        c2[j] += x2 * bj;
                        c3[j] += x3 * bj;
                    }
                }
            }
            for (; i < rowCount; ++i) {
                const double* a = A + (rowBegin + i) * N;
                double* c = C + i * N;
                for (size_t k = kk; k < kEnd; ++k) {
                    const double* b = B + k * N;
                    const double x = a[k];
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += x * b[j];
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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

int run(int argc, char** argv, int rank, int size) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const bool root = (rank == 0);

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
        printf("MPI ranks: %d\n", size);
    }

    // Every rank holds full A and B (deterministic initialization, no communication),
    // and computes only its own block of rows of C.
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    if (root) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    const RowRange range = rowRangeFor(N, rank, size);
    std::vector<double> Clocal(range.count * N);

    if (root) printf("Computing matrix multiplication...\n");

    // Synchronize so the timed region covers only the parallel computation
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(A.data(), B.data(), Clocal.data(), N, range.begin, range.count);

    auto end = std::chrono::high_resolution_clock::now();

    // Wall time of the parallel region is the slowest rank
    long long localNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long maxNs = 0;
    MPI_Reduce(&localNs, &maxNs, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

    // Gather full C on rank 0 only when it is needed for output or validation
    std::vector<double> C;
    if (printResults || validate) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts, displs;
        if (root) {
            C.resize(N * N);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const RowRange rr = rowRangeFor(N, r, size);
                counts[r] = static_cast<int>(rr.count);
                displs[r] = static_cast<int>(rr.begin);
            }
        }
        MPI_Gatherv(Clocal.data(), static_cast<int>(range.count), rowType,
                    root ? C.data() : nullptr, root ? counts.data() : nullptr,
                    root ? displs.data() : nullptr, rowType, 0, comm);
        MPI_Type_free(&rowType);
    }

    int status = 0;
    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::nanoseconds(maxNs));

        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    // Make the exit status consistent across all ranks
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    return status;
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
