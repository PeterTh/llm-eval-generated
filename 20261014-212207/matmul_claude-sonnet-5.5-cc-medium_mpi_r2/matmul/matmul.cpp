#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>

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

// Computes rows [0, rows) of C = A * B, where A is a local row block (rows x N).
// The k-summation order per element is ascending, as in the original code.
void matrixMultiply(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    constexpr size_t JB = 512;
    constexpr size_t KB = 128;
    std::memset(C, 0, rows * N * sizeof(double));
    for (size_t jb = 0; jb < N; jb += JB) {
        const size_t je = std::min(N, jb + JB);
        for (size_t kb = 0; kb < N; kb += KB) {
            const size_t ke = std::min(N, kb + KB);
            size_t i = 0;
            for (; i + 4 <= rows; i += 4) {
                double* __restrict c0 = C + (i + 0) * N;
                double* __restrict c1 = C + (i + 1) * N;
                double* __restrict c2 = C + (i + 2) * N;
                double* __restrict c3 = C + (i + 3) * N;
                const double* a0 = A + (i + 0) * N;
                const double* a1 = A + (i + 1) * N;
                const double* a2 = A + (i + 2) * N;
                const double* a3 = A + (i + 3) * N;
                for (size_t k = kb; k < ke; ++k) {
                    const double* __restrict b = B + k * N;
                    const double v0 = a0[k], v1 = a1[k], v2 = a2[k], v3 = a3[k];
                    for (size_t j = jb; j < je; ++j) {
                        const double bj = b[j];
                        c0[j] += v0 * bj;
                        c1[j] += v1 * bj;
                        c2[j] += v2 * bj;
                        c3[j] += v3 * bj;
                    }
                }
            }
            for (; i < rows; ++i) {
                double* __restrict c = C + i * N;
                const double* a = A + i * N;
                for (size_t k = kb; k < ke; ++k) {
                    const double* __restrict b = B + k * N;
                    const double v = a[k];
                    for (size_t j = jb; j < je; ++j) c[j] += v * b[j];
                }
            }
        }
    }
}

// Simple validation (rank 0): compute a few elements and compare
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
    }

    // Row-block distribution of A and C
    std::vector<int> rowStart(size + 1);
    for (int r = 0; r <= size; ++r) rowStart[r] = static_cast<int>(N * r / size);
    const size_t myRows = rowStart[rank + 1] - rowStart[rank];
    const size_t myStart = rowStart[rank];

    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(myRows * N);

    if (rank == 0) printf("Initializing matrices...\n");
    for (size_t i = 0; i < myRows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, myStart + i, j);
    // Each rank generates its slice of B, then allgather
    {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (rowStart[r + 1] - rowStart[r]) * static_cast<int>(N);
            displs[r] = rowStart[r] * static_cast<int>(N);
        }
        for (size_t i = 0; i < myRows; ++i)
            for (size_t j = 0; j < N; ++j) B[(myStart + i) * N + j] = getPseudoRndValue(N, myStart + i, j);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, B.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A.data(), B.data(), C.data(), myRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full C on rank 0 only if needed
    std::vector<double> fullC;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (rowStart[r + 1] - rowStart[r]) * static_cast<int>(N);
            displs[r] = rowStart[r] * static_cast<int>(N);
        }
        if (rank == 0) fullC.resize(N * N);
        MPI_Gatherv(C.data(), static_cast<int>(myRows * N), MPI_DOUBLE, fullC.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int ret = 0;
    if (rank == 0) {
        if (printResults) print_results(fullC, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(B, fullC, N)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    MPI_Finalize();
    return ret;
}
