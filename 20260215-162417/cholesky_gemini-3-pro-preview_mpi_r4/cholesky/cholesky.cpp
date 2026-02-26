#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (parallel row-cyclic)
bool choleskyDecomposition(std::vector<double>& local_A, const size_t n, int rank, int size) {
    std::vector<double> row_buffer(n);
    
    for (size_t j = 0; j < n; ++j) {
        int root = j % size;
        
        if (rank == root) {
            size_t local_idx = j / size;
            double* row_ptr = &local_A[local_idx * n];
            
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row_ptr[k] * row_ptr[k];
            }
            double val = row_ptr[j] - sum;
            
            if (val <= 0.0) {
                // Error: Matrix is not positive definite
                row_buffer[j] = NAN;
            } else {
                row_buffer[j] = sqrt(val);
                row_ptr[j] = row_buffer[j];
                // Copy computed part of row to buffer for broadcast
                for (size_t k = 0; k < j; ++k) row_buffer[k] = row_ptr[k];
            }
        }
        
        MPI_Bcast(row_buffer.data(), j + 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        if (std::isnan(row_buffer[j])) return false;
        
        // Update local rows i > j
        size_t first_i = j + 1;
        while(first_i < n && first_i % size != (size_t)rank) first_i++;
        
        for (size_t local_r = first_i / size; local_r < local_A.size() / n; ++local_r) {
            double* row_ptr = &local_A[local_r * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row_ptr[k] * row_buffer[k];
            }
            row_ptr[j] = (row_ptr[j] - sum) / row_buffer[j];
        }
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank, int size) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    size_t local_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size == (size_t)rank) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                A[local_idx * n + j] = sum;
            }
            A[local_idx * n + i] += n;
            local_idx++;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark (MPI Parallel)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t num_local_rows = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size == (size_t)rank) num_local_rows++;
    }
    
    std::vector<double> local_A(num_local_rows * n);
    
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(local_A, n, rank, size);
    
    std::vector<double> A_full;
    std::vector<double> A_orig;
    
    if (rank == 0 && validate) {
        A_orig.resize(n * n);
        generatePositiveDefiniteMatrix(A_orig, n, 0, 1); 
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(local_A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    if (validate || printResults) {
        if (rank == 0) A_full.resize(n * n);
        
        // Gather all rows to rank 0
        if (rank == 0) {
            size_t local_idx = 0;
            for (size_t i = 0; i < n; ++i) {
                int owner = i % size;
                if (owner == 0) {
                    memcpy(&A_full[i * n], &local_A[local_idx * n], n * sizeof(double));
                    local_idx++;
                } else {
                    MPI_Recv(&A_full[i * n], n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            size_t local_idx = 0;
            for (size_t i = 0; i < n; ++i) {
                if (i % size == (size_t)rank) {
                    MPI_Send(&local_A[local_idx * n], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    local_idx++;
                }
            }
        }
        
        if (rank == 0) {
            // Zero out upper triangle
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A_full[i * n + j] = 0.0;
                }
            }
            
            if (printResults) {
                print_results(A_full, "CholeskyL");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(A_full, A_orig, n);
                if (valid) printf("Validation: PASSED\n");
                else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
