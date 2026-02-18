#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

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
                    std::vector<double>& C, const size_t N, const size_t local_rows) {
    // Matrix multiplication for local rows with OpenMP optimization
    // A is local_rows × N
    // B is N × N  
    // C is local_rows × N
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int i = 0; i < (int)local_rows; ++i) {
        for (int j = 0; j < (int)N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < (int)N; ++k) {
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0)
    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
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
    
    // Broadcast N to all ranks
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate row distribution
    size_t rows_per_rank = N / mpi_size;
    size_t remaining_rows = N % mpi_size;
    size_t local_rows = (mpi_rank < (int)remaining_rows) ? rows_per_rank + 1 : rows_per_rank;
    size_t row_offset = mpi_rank * rows_per_rank + (mpi_rank < (int)remaining_rows ? mpi_rank : remaining_rows);
    
    // Allocate local matrices
    std::vector<double> local_A(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> local_C(local_rows * N);
    
    // Initialize matrices (deterministic, based on global indices)
    if (mpi_rank == 0) {
        printf("Initializing matrices...\n");
    }
    
    // Initialize B on all ranks
    initMatrix(B, N);
    
    // Initialize local_A on all ranks (using global row indices)
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t global_i = row_offset + i;
            local_A[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
    
    if (mpi_rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(local_A, B, local_C, N, local_rows);
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results on rank 0
    std::vector<double> global_C(mpi_rank == 0 ? N * N : 0);
    
    // Gather all local_C into global_C
    std::vector<int> recv_counts(mpi_size);
    std::vector<int> displacements(mpi_size);
    for (int i = 0; i < mpi_size; ++i) {
        size_t irows = N / mpi_size + ((i < (int)(N % mpi_size)) ? 1 : 0);
        recv_counts[i] = (int)(irows * N);
        displacements[i] = (i > 0) ? displacements[i-1] + recv_counts[i-1] : 0;
    }
    
    // Use NULL pointer for non-root ranks in MPI_Gatherv
    double* recv_buf = (mpi_rank == 0) ? global_C.data() : nullptr;
    MPI_Gatherv(local_C.data(), (int)(local_rows * N), MPI_DOUBLE,
                recv_buf, recv_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Validation and output on rank 0
    if (mpi_rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(global_C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            
            // Need full matrices for validation
            std::vector<double> full_A(N * N);
            std::vector<double> full_B = B;
            initMatrix(full_A, N);
            
            bool valid = validateResult(full_A, full_B, global_C, N);
            
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
