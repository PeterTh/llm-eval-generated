#include <chrono>
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

// Initialize rows [rowBegin, rowBegin + rows) of the matrix into mat (local storage)
void initRows(double* mat, const size_t N, const size_t rowBegin, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowBegin + i, j);
        }
    }
}

// C (rows x N) = A (rows x N) * B (N x N). Per-element summation order over k
// is sequential, identical to the reference implementation.
void matrixMultiply(const double* __restrict A, const double* __restrict B,
                    double* __restrict C, const size_t rows, const size_t N) {
    constexpr size_t JB = 64;
    constexpr size_t IB = 4;
    for (size_t jb = 0; jb < N; jb += JB) {
        const size_t jn = (N - jb < JB) ? (N - jb) : JB;
        size_t i = 0;
        if (jn == JB) {
            for (; i + IB <= rows; i += IB) {
                double acc[IB][JB] = {};
                for (size_t k = 0; k < N; ++k) {
                    const double* b = B + k * N + jb;
                    for (size_t r = 0; r < IB; ++r) {
                        const double a = A[(i + r) * N + k];
                        for (size_t j = 0; j < JB; ++j) acc[r][j] += a * b[j];
                    }
                }
                for (size_t r = 0; r < IB; ++r)
                    for (size_t j = 0; j < JB; ++j) C[(i + r) * N + jb + j] = acc[r][j];
            }
        }
        for (; i < rows; ++i) {
            double acc[JB] = {};
            for (size_t k = 0; k < N; ++k) {
                const double a = A[i * N + k];
                const double* b = B + k * N + jb;
                for (size_t j = 0; j < jn; ++j) acc[j] += a * b[j];
            }
            for (size_t j = 0; j < jn; ++j) C[i * N + jb + j] = acc[j];
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Row distribution (block, remainder spread over first ranks)
    std::vector<int> counts(nprocs), displs(nprocs);
    std::vector<size_t> rowStart(nprocs + 1, 0);
    for (int p = 0; p < nprocs; ++p) {
        const size_t rws = N / nprocs + (static_cast<size_t>(p) < N % nprocs ? 1 : 0);
        rowStart[p + 1] = rowStart[p] + rws;
        counts[p] = static_cast<int>(rws * N);
        displs[p] = static_cast<int>(rowStart[p] * N);
    }
    const size_t myRowBegin = rowStart[rank];
    const size_t myRows = rowStart[rank + 1] - myRowBegin;

    std::vector<double> A(myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(myRows * N);

    if (rank == 0) printf("Initializing matrices...\n");
    // A: local rows only. B: each rank generates its slice, then allgather.
    initRows(A.data(), N, myRowBegin, myRows);
    initRows(B.data() + myRowBegin * N, N, myRowBegin, myRows);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, B.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A.data(), B.data(), C.data(), myRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather full C on rank 0 only when needed
    std::vector<double> fullC;
    if (printResults || validate) {
        if (rank == 0) fullC.resize(N * N);
        MPI_Gatherv(C.data(), counts[rank], MPI_DOUBLE, fullC.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(fullC, "MatrixC");
        }

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
