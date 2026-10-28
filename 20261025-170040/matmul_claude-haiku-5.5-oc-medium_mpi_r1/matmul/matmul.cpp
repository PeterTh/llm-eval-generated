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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Contiguous block of rows [begin, begin + count) owned by one rank
struct RowRange {
    size_t begin;
    size_t count;
};

RowRange rowRangeFor(const size_t N, const int rank, const int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    const size_t base = N / p;
    const size_t rem = N % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Computes C = A * B for `rows` local rows.
//   A: rows x N slice of the full matrix (row-major)
//   B: full N x N matrix (row-major)
//   C: rows x N output, must be zero-initialized
// Each C element accumulates its k-terms in ascending k order, exactly as the
// naive triple loop does. Combined with -ffp-contract=off (see CMakeLists.txt),
// every term is a rounded multiply followed by a rounded add, so results are
// bitwise identical regardless of the number of MPI ranks.
// The k-dimension is blocked so a panel of B stays cache-resident while four
// rows of C are updated together, reusing each loaded B element four times.
void matrixMultiplyRows(const double* A, const double* B, double* C,
                        const size_t rows, const size_t N) {
    constexpr size_t KB = 64;

    for (size_t kk = 0; kk < N; kk += KB) {
        const size_t kEnd = std::min(kk + KB, N);

        size_t i = 0;
        for (; i + 4 <= rows; i += 4) {
            const double* a0 = A + (i + 0) * N;
            const double* a1 = A + (i + 1) * N;
            const double* a2 = A + (i + 2) * N;
            const double* a3 = A + (i + 3) * N;
            double* c0 = C + (i + 0) * N;
            double* c1 = C + (i + 1) * N;
            double* c2 = C + (i + 2) * N;
            double* c3 = C + (i + 3) * N;

            for (size_t k = kk; k < kEnd; ++k) {
                const double x0 = a0[k];
                const double x1 = a1[k];
                const double x2 = a2[k];
                const double x3 = a3[k];
                const double* b = B + k * N;
                for (size_t j = 0; j < N; ++j) {
                    const double bj = b[j];
                    c0[j] += x0 * bj;
                    c1[j] += x1 * bj;
                    c2[j] += x2 * bj;
                    c3[j] += x3 * bj;
                }
            }
        }

        for (; i < rows; ++i) {
            const double* a = A + i * N;
            double* c = C + i * N;
            for (size_t k = kk; k < kEnd; ++k) {
                const double x = a[k];
                const double* b = B + k * N;
                for (size_t j = 0; j < N; ++j) {
                    c[j] += x * b[j];
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

// Runs the benchmark on all ranks; only rank 0 prints. Returns the exit status.
int runBenchmark(const int rank, const int size, const size_t N,
                 const bool validate, const bool printResults) {
    const bool root = (rank == 0);
    MPI_Comm comm = MPI_COMM_WORLD;

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // A and B are deterministic, so every rank generates them locally rather
    // than receiving them over the network.
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    initMatrix(A, N);
    initMatrix(B, N);

    const RowRange myRows = rowRangeFor(N, rank, size);
    std::vector<double> Clocal(myRows.count * N, 0.0);

    if (root) {
        printf("Computing matrix multiplication...\n");
    }

    // Time the distributed multiplication: synchronize, compute, and report
    // the slowest rank's elapsed time.
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyRows(A.data() + myRows.begin * N, B.data(), Clocal.data(), myRows.count, N);

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double, std::milli>(maxMs));

        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C matrix on rank 0 only when it is needed for output or
    // validation. Row blocks are gathered as contiguous rows of N doubles, so
    // the MPI counts stay within int range for any N that fits in int.
    const bool needFullC = printResults || validate;
    std::vector<double> C;
    if (needFullC) {
        if (root) {
            C.resize(N * N);
        }

        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const RowRange rr = rowRangeFor(N, r, size);
            counts[r] = static_cast<int>(rr.count);
            displs[r] = static_cast<int>(rr.begin);
        }

        MPI_Gatherv(Clocal.data(), static_cast<int>(myRows.count), rowType,
                    C.data(), counts.data(), displs.data(), rowType, 0, comm);

        MPI_Type_free(&rowType);
    }

    if (!root) {
        return 0;
    }

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
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

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int status = 0;
    bool run = true;
    
    // Parse command line arguments (identically on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            run = false;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            run = false;
            break;
        }
    }

    if (run) {
        status = runBenchmark(rank, size, N, validate, printResults);
    }

    MPI_Finalize();
    return status;
}
