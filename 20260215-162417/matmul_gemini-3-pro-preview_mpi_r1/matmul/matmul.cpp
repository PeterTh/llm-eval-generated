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

// Optimized local matrix multiplication using transposed B
void matrixMultiplyLocal(const std::vector<double>& local_A, const std::vector<double>& BT, 
                    std::vector<double>& local_C, const size_t N, const size_t num_rows) {
    // local_A is num_rows x N
    // BT is N x N (transposed B)
    // local_C is num_rows x N
    
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            // Accessing BT row-wise (which is B column-wise) for cache efficiency
            for (size_t k = 0; k < N; ++k) {
                sum += local_A[i * N + k] * BT[j * N + k];
            }
            local_C[i * N + j] = sum;
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
    int validate_flag = 0;
    int printResults_flag = 0;
    
    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&N, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool validate = validate_flag;
    bool printResults = printResults_flag;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Determine row distribution
    int rows_per_rank = N / size;
    int remainder = N % size;
    int my_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    int my_offset = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    
    // Local part of A
    std::vector<double> local_A(my_rows * N);
    for (int i = 0; i < my_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            local_A[i * N + j] = getPseudoRndValue(N, my_offset + i, j);
        }
    }

    // Full BT (Transposed B)
    // B[i][j] is getPseudoRndValue(N, i, j).
    // BT[j][i] = B[i][j].
    // So BT[row][col] corresponds to B[col][row] = getPseudoRndValue(N, col, row).
    std::vector<double> BT(N * N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            BT[i * N + j] = getPseudoRndValue(N, j, i); // Notice swap of indices for BT
        }
    }
    
    // Output C
    std::vector<double> local_C(my_rows * N);

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyLocal(local_A, BT, local_C, N, my_rows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Prepare for gather
    std::vector<double> global_C;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    
    if (rank == 0) {
        global_C.resize(N * N);
        recvcounts.resize(size);
        displs.resize(size);
        
        int current_disp = 0;
        for (int i = 0; i < size; ++i) {
            int r_rows = rows_per_rank + (i < remainder ? 1 : 0);
            recvcounts[i] = r_rows * N;
            displs[i] = current_disp;
            current_disp += recvcounts[i];
        }
    }

    // Gather results
    // Use MPI_Gatherv because row counts can vary
    MPI_Gatherv(local_C.data(), my_rows * N, MPI_DOUBLE,
                global_C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    int ret_code = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(global_C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            
            // Need full A and B for validation
            std::vector<double> A(N * N);
            std::vector<double> B(N * N);
            
            initMatrix(A, N);
            initMatrix(B, N);
            
            bool valid = validateResult(A, B, global_C, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret_code = 1;
            }
        }
    }

    MPI_Finalize();
    return ret_code;
}
