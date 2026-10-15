#include <mpi.h>

#include <algorithm>
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

// Computes rows [0, rows) of C for the local row block of A. Per-element
// accumulation order over k is identical to the naive kernel.
void matrixMultiply(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    constexpr size_t JB = 256;
    std::memset(C, 0, rows * N * sizeof(double));
    for (size_t jb = 0; jb < N; jb += JB) {
        const size_t jn = std::min(JB, N - jb);
        size_t i = 0;
        for (; i + 4 <= rows; i += 4) {
            const double* a0 = A + i * N;
            const double* a1 = a0 + N;
            const double* a2 = a1 + N;
            const double* a3 = a2 + N;
            double* __restrict c0 = C + i * N + jb;
            double* __restrict c1 = c0 + N;
            double* __restrict c2 = c1 + N;
            double* __restrict c3 = c2 + N;
            for (size_t k = 0; k < N; ++k) {
                const double* __restrict b = B + k * N + jb;
                const double x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
                for (size_t j = 0; j < jn; ++j) {
                    const double bv = b[j];
                    c0[j] += x0 * bv;
                    c1[j] += x1 * bv;
                    c2[j] += x2 * bv;
                    c3[j] += x3 * bv;
                }
            }
        }
        for (; i < rows; ++i) {
            const double* a0 = A + i * N;
            double* __restrict c0 = C + i * N + jb;
            for (size_t k = 0; k < N; ++k) {
                const double* __restrict b = B + k * N + jb;
                const double x0 = a0[k];
                for (size_t j = 0; j < jn; ++j) c0[j] += x0 * b[j];
            }
        }
    }
}

// Simple validation on rank 0: compute a few elements and compare.
// A entries are regenerated on the fly (A is distributed).
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Row block distribution
    std::vector<int> counts(size), displs(size);
    std::vector<size_t> rowStart(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        const size_t rows = N / size + (static_cast<size_t>(r) < N % size ? 1 : 0);
        rowStart[r + 1] = rowStart[r] + rows;
        counts[r] = static_cast<int>(rows * N);
        displs[r] = static_cast<int>(rowStart[r] * N);
    }
    const size_t myRows = rowStart[rank + 1] - rowStart[rank];
    const size_t myFirst = rowStart[rank];

    // Inputs are deterministic: each rank generates what it needs locally (no communication).
    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> Cloc(myRows * N);
    for (size_t i = 0; i < myRows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, myFirst + i, j);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiply(A.data(), B.data(), Cloc.data(), myRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int ret = 0;
    if (printResults || validate) {
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        MPI_Gatherv(Cloc.data(), static_cast<int>(myRows * N), MPI_DOUBLE, C.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(C, "MatrixC");
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(B, C, N)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    MPI_Finalize();
    return ret;
}
