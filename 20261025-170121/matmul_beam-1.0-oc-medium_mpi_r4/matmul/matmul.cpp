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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, const size_t rowCount) {
    for (size_t ii = 0; ii < rowCount; ++ii) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[ii * N + k] * B[k * N + j];
            }
            C[ii * N + j] = sum;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
            }
        }
    }
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    // broadcast flags as ints
    int v = validate ? 1 : 0, r = printResults ? 1 : 0;
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = v != 0; printResults = r != 0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribution of rows
    size_t base = N / size;
    size_t rem = N % size;
    size_t rowCount = base + (rank < (int)rem ? 1 : 0);

    // Allocate matrices
    std::vector<double> A(rowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rowCount * N);

    // Initialize A rows and B on rank 0 then scatter/bcast
    std::vector<double> Afull, Cfull;
    if (rank == 0) {
        Afull.assign(N * N, 0.0);
        initMatrix(Afull, N);
        initMatrix(B, N);
    }
    // broadcast B
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // scatter A rows
    std::vector<int> counts(size), displs(size);
    size_t off = 0;
    for (int p = 0; p < size; ++p) {
        size_t rc = base + (p < (int)rem ? 1 : 0);
        counts[p] = rc * N;
        displs[p] = off * N;
        off += rc;
    }
    MPI_Scatterv(rank == 0 ? Afull.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 A.data(), rowCount * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, rowCount);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // gather C
    std::vector<int> ccounts(size), cdispls(size);
    off = 0;
    for (int p = 0; p < size; ++p) {
        size_t rc = base + (p < (int)rem ? 1 : 0);
        ccounts[p] = rc * N;
        cdispls[p] = off * N;
        off += rc;
    }
    if (rank == 0) Cfull.assign(N * N, 0.0);
    MPI_Gatherv(C.data(), rowCount * N, MPI_DOUBLE,
                rank == 0 ? Cfull.data() : nullptr, ccounts.data(), cdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);
        double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(Cfull, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            // need A full
            bool valid = validateResult(Afull, B, Cfull, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
