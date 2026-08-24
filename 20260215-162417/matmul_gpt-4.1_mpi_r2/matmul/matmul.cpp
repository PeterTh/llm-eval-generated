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
                    std::vector<double>& C, const size_t N, int rank, int size) {
    size_t rows_per_proc = N / size;
    size_t start_row = rank * rows_per_proc;
    size_t end_row = (rank == size - 1) ? N : start_row + rows_per_proc;
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);
    
    // Initialize matrices (only root)
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    // Broadcast A and B to all processes
    MPI_Bcast(A.data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Each process computes its chunk
    std::vector<double> C_local(N * N, 0.0);
    matrixMultiply(A, B, C_local, N, rank, size);
    
    // Gather results to root
    size_t rows_per_proc = N / size;
    size_t start_row = rank * rows_per_proc;
    size_t end_row = (rank == size - 1) ? N : start_row + rows_per_proc;
    int *recvcounts = nullptr, *displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        for (int i = 0; i < size; ++i) {
            size_t srow = i * rows_per_proc;
            size_t erow = (i == size - 1) ? N : srow + rows_per_proc;
            recvcounts[i] = (erow - srow) * N;
            displs[i] = srow * N;
        }
    }
    MPI_Gatherv(C_local.data() + start_row * N, (end_row - start_row) * N, MPI_DOUBLE,
                C.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&local_duration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration / 1000.0) / 1e9;
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
            }
        }
        delete[] recvcounts;
        delete[] displs;
    }
    MPI_Finalize();
    return 0;
}
