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

// Initialize only rows [row_start, row_start+num_rows) of the matrix
void initMatrixRows(double* mat, const size_t N, const size_t row_start, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, row_start + i, j);
        }
    }
}

// Cache-optimized local matmul with loop tiling and B transposed
void matrixMultiplyLocal(const double* A, const double* Bt,
                         double* C, const size_t num_rows, const size_t N) {
    constexpr size_t TILE = 64;
    std::memset(C, 0, num_rows * N * sizeof(double));

    for (size_t ii = 0; ii < num_rows; ii += TILE) {
        const size_t i_end = std::min(ii + TILE, num_rows);
        for (size_t jj = 0; jj < N; jj += TILE) {
            const size_t j_end = std::min(jj + TILE, N);
            for (size_t kk = 0; kk < N; kk += TILE) {
                const size_t k_end = std::min(kk + TILE, N);
                for (size_t i = ii; i < i_end; ++i) {
                    const double* A_row = A + i * N;
                    double* C_row = C + i * N;
                    for (size_t j = jj; j < j_end; ++j) {
                        const double* Bt_row = Bt + j * N;
                        double sum = C_row[j];
                        for (size_t k = kk; k < k_end; ++k) {
                            sum += A_row[k] * Bt_row[k];
                        }
                        C_row[j] = sum;
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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

    int rank, nprocs;
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute row distribution: each rank gets a contiguous block of rows
    const size_t base_rows = N / static_cast<size_t>(nprocs);
    const size_t remainder = N % static_cast<size_t>(nprocs);
    const size_t my_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);



    // Compute sendcounts/displacements for Scatterv/Gatherv
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        sendcounts[r] = static_cast<int>(r_rows * N);
        size_t r_start = base_rows * static_cast<size_t>(r)
            + std::min(static_cast<size_t>(r), remainder);
        displs[r] = static_cast<int>(r_start * N);
    }

    // Full matrices on rank 0 only (for init, validation, output)
    std::vector<double> A, B_full, C;
    if (rank == 0) {
        A.resize(N * N);
        B_full.resize(N * N);
        C.resize(N * N);
        if (rank == 0) printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B_full, N);
    }

    // Local A rows
    std::vector<double> local_A(my_rows * N);

    // Scatter rows of A
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(),
                 MPI_DOUBLE, local_A.data(), static_cast<int>(my_rows * N),
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast B to all ranks (allocate on non-root)
    if (rank != 0) B_full.resize(N * N);
    MPI_Bcast(B_full.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Transpose B for cache-friendly access
    std::vector<double> Bt(N * N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            Bt[j * N + i] = B_full[i * N + j];

    // Free B_full on non-root to save memory
    if (rank != 0) { std::vector<double>().swap(B_full); }

    // Local C rows
    std::vector<double> local_C(my_rows * N);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(local_A.data(), Bt.data(), local_C.data(), my_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather C rows back to rank 0
    MPI_Gatherv(local_C.data(), static_cast<int>(my_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
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
            bool valid = validateResult(A, B_full, C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
