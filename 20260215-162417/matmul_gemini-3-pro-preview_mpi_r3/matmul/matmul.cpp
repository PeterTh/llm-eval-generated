#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <algorithm>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixLocal(std::vector<double>& mat, const size_t N, const size_t start_row, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

void initMatrixFull(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B, 
                    std::vector<double>& C_local, const size_t N, const size_t num_rows) {
    // Transpose B for better cache locality (access columns as rows)
    std::vector<double> B_T(N * N);
    for(size_t i=0; i<N; ++i) {
        for(size_t j=0; j<N; ++j) {
            B_T[j*N + i] = B[i*N + j];
        }
    }

    // Initialize C_local to 0
    std::fill(C_local.begin(), C_local.end(), 0.0);

    // Cache blocking parameters
    // L1 cache is typically 32KB per core.
    // L2 256KB-1MB.
    // L3 shared 8MB+.
    // A block size of 64 doubles is 512 bytes.
    // 64x64 block takes 32KB. Fitting 3 blocks (A, B, C) in L2 is easy.
    // With 64, A_block + B_block + C_block = 96KB. Fits in L2.
    // For L1, smaller block size like 32 might be better? Or relying on hardware prefetching.
    // Let's stick with 64.
    
    const size_t BLOCK_SIZE = 64;

    // Loop order for blocking:
    // i0, j0, k0 blocks.
    // Inside block:
    // For standard C = A * B, order usually i, k, j (to stream B row-wise).
    // Here we use B_T. C[i][j] = dot(A[i], B_T[j]).
    // The inner loop over k is a dot product.
    // To reuse A block and B block effectively:
    // With (i, j, k) block loop:
    // Load block A[i0..i_end, k0..k_end] into L1/L2.
    // Load block B_T[j0..j_end, k0..k_end] into L1/L2.
    // Compute partial C[i0..i_end, j0..j_end].
    
    for (size_t i0 = 0; i0 < num_rows; i0 += BLOCK_SIZE) {
        size_t i_end = std::min(i0 + BLOCK_SIZE, num_rows);
        for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            size_t j_end = std::min(j0 + BLOCK_SIZE, N);
            for (size_t k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                size_t k_end = std::min(k0 + BLOCK_SIZE, N);
                
                // Compute block
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t j = j0; j < j_end; ++j) {
                        double sum = 0.0;
                        for (size_t k = k0; k < k_end; ++k) {
                            sum += A_local[i * N + k] * B_T[j * N + k];
                        }
                        C_local[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    // We re-compute the expected value on the fly to avoid storing full A/B on rank 0
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                double a_ik = getPseudoRndValue(N, i, k);
                double b_kj = getPseudoRndValue(N, k, j);
                expected += a_ik * b_kj;
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

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine row distribution
    size_t rows_per_rank = N / world_size;
    size_t remainder = N % world_size;
    size_t local_rows = (world_rank < (int)remainder) ? rows_per_rank + 1 : rows_per_rank;
    size_t start_row = (world_rank < (int)remainder) ? 
                       world_rank * (rows_per_rank + 1) : 
                       remainder * (rows_per_rank + 1) + (world_rank - remainder) * rows_per_rank;

    // Allocate local matrices
    // A_local: local_rows x N
    // B: N x N (replicated)
    // C_local: local_rows x N
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N);
    
    // Initialize matrices
    if (world_rank == 0) printf("Initializing matrices...\n");
    initMatrixLocal(A_local, N, start_row, local_rows);
    initMatrixFull(B, N); // Every rank initializes full B. No comms needed because deterministic.
    
    // Sync before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (world_rank == 0) printf("Computing matrix multiplication...\n");
    double start_time = MPI_Wtime();
    
    matrixMultiplyLocal(A_local, B, C_local, N, local_rows);
    
    // Sync after computation
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double duration_sec = end_time - start_time;
    
    // Calculate GFLOPS
    // Only rank 0 reports, using the max time across all ranks to be conservative/correct
    double max_duration_sec;
    MPI_Reduce(&duration_sec, &max_duration_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", max_duration_sec * 1000.0);
        double gflops = (2.0 * N * N * N) / max_duration_sec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results if needed for validation or printing
    std::vector<double> C_global;
    if (validate || printResults) {
        if (world_rank == 0) C_global.resize(N * N);
        
        // Prepare counts and displs for Gatherv
        std::vector<int> recvcounts(world_size);
        std::vector<int> displs(world_size);
        
        // We need to gather these because local_rows varies
        // Gatherv expects counts in terms of elements, not rows. 
        
        // Gather the number of elements from each rank
        int local_elements = (int)(local_rows * N);
        MPI_Gather(&local_elements, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (world_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < world_size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(C_local.data(), local_elements, MPI_DOUBLE,
                   C_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        // Print results for external validation
        if (printResults) {
            // print_results expects a vector
            print_results(C_global, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(C_global, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                // We don't return 1 here because other ranks are waiting in Finalize
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
