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

// Computes rows [0, rows) of C = A * B, where A holds only the local rows.
// Per-element summation order (k ascending) matches the serial version.
void matrixMultiply(const double* A, const double* B, double* C,
                    const size_t rows, const size_t N) {
    constexpr size_t KB = 128, JB = 512;
    std::fill(C, C + rows * N, 0.0);
    for (size_t jj = 0; jj < N; jj += JB) {
        const size_t jEnd = std::min(jj + JB, N);
        for (size_t kk = 0; kk < N; kk += KB) {
            const size_t kEnd = std::min(kk + KB, N);
            size_t i = 0;
            for (; i + 4 <= rows; i += 4) {
                double* __restrict c0 = C + (i + 0) * N;
                double* __restrict c1 = C + (i + 1) * N;
                double* __restrict c2 = C + (i + 2) * N;
                double* __restrict c3 = C + (i + 3) * N;
                for (size_t k = kk; k < kEnd; ++k) {
                    const double a0 = A[(i + 0) * N + k], a1 = A[(i + 1) * N + k];
                    const double a2 = A[(i + 2) * N + k], a3 = A[(i + 3) * N + k];
                    const double* __restrict b = B + k * N;
                    #pragma GCC ivdep
                    for (size_t j = jj; j < jEnd; ++j) {
                        const double bv = b[j];
                        c0[j] += a0 * bv;
                        c1[j] += a1 * bv;
                        c2[j] += a2 * bv;
                        c3[j] += a3 * bv;
                    }
                }
            }
            for (; i < rows; ++i) {
                double* __restrict c0 = C + i * N;
                for (size_t k = kk; k < kEnd; ++k) {
                    const double a0 = A[i * N + k];
                    const double* __restrict b = B + k * N;
                    #pragma GCC ivdep
                    for (size_t j = jj; j < jEnd; ++j) c0[j] += a0 * b[j];
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    
    // Block-row distribution of C (and A); B is generated on every rank
    // since it is a deterministic function of the indices (no communication).
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t base = N / size, extra = N % size;
        counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < extra ? 1 : 0));
        displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
    }
    const size_t rows = counts[rank];
    const size_t row0 = displs[rank];

    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rows * N);
    
    if (rank == 0) printf("Initializing matrices...\n");
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = getPseudoRndValue(N, row0 + i, j);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            B[i * N + j] = getPseudoRndValue(N, i, j);
    
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    matrixMultiply(A.data(), B.data(), C.data(), rows, N);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    
    double elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long long ms = static_cast<long long>(max_elapsed * 1000.0);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", ms);
        double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather full C on rank 0 only when needed
    std::vector<double> Cfull;
    if (printResults || validate) {
        if (rank == 0) Cfull.resize(N * N);
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);
        MPI_Gatherv(C.data(), counts[rank], rowType, Cfull.data(), counts.data(),
                    displs.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }
    
    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(Cfull, "MatrixC");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B, Cfull, N);
            if (valid) {
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
