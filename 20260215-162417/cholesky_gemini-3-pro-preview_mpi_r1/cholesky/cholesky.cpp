#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition using MPI
// Uses a 1D column-cyclic distribution for load balancing
// Global column j is stored on process (j % num_procs)

bool choleskyDecompositionMPI(std::vector<double>& local_A, const size_t n, int rank, int num_procs) {
    // local_A stores columns belonging to this rank in column-major order.
    // i.e., local_A[local_col * n + row]
    
    std::vector<double> current_col(n);
    size_t n_local_cols = local_A.size() / n;

    for (size_t k = 0; k < n; ++k) {
        int root = k % num_procs;
        
        // If I own column k
        if (rank == root) {
            size_t local_k = k / num_procs;
            
            // Calculate diagonal element
            double akk = local_A[local_k * n + k];
            
            if (akk <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu (rank %d, val %e)\n", k, rank, akk);
                return false;
            }
            
            double lkk = sqrt(akk);
            local_A[local_k * n + k] = lkk;
            
            // Scale the rest of the column
            for (size_t i = k + 1; i < n; ++i) {
                local_A[local_k * n + i] /= lkk;
            }
            
            // Copy column to buffer for broadcast
            // Only need to copy from k to n-1
            for (size_t i = k; i < n; ++i) {
                current_col[i] = local_A[local_k * n + i];
            }
        }
        
        // Broadcast column k from root
        // We broadcast starting from k to save bandwidth
        MPI_Bcast(&current_col[k], n - k, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        // Update local columns j > k
        // Iterate over all local columns
        for (size_t local_j = 0; local_j < n_local_cols; ++local_j) {
            size_t global_j = local_j * num_procs + rank;
            
            if (global_j <= k) continue;
            
            // Update A_{ij} for i >= global_j
            // A_{ij} -= L_{ik} * L_{jk}
            // L_{ik} is current_col[i]
            // L_{jk} is current_col[global_j] (since L is lower triangular, L_{jk} is at row global_j of col k)
            
            double L_jk = current_col[global_j];
            
            for (size_t i = global_j; i < n; ++i) {
                local_A[local_j * n + i] -= current_col[i] * L_jk;
            }
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (Row-Major)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

// Validation function
bool validateCholesky(const std::vector<double>& L_row_major, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T where L is lower triangular
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                // L stored in row major
                double L_ik = (i >= k) ? L_row_major[i * n + k] : 0.0;
                double L_jk = (j >= k) ? L_row_major[j * n + k] : 0.0;
                sum += L_ik * L_jk;
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
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
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
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate local columns count
    size_t n_local_cols = 0;
    for (size_t j = 0; j < n; ++j) {
        if ((int)(j % num_procs) == rank) n_local_cols++;
    }
    
    std::vector<double> local_A(n * n_local_cols);
    std::vector<double> A_orig;
    
    // Generation and Distribution
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        std::vector<double> A_full(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
        
        if (validate) {
            A_orig = A_full;
        }
        
        // Pack and send to other ranks
        // We iterate by rank to pack their data
        for (int p = 0; p < num_procs; ++p) {
            // Calculate size for rank p
            size_t p_cols_count = 0;
            for (size_t j = 0; j < n; ++j) if ((int)(j % num_procs) == p) p_cols_count++;
            
            std::vector<double> send_buf(n * p_cols_count);
            size_t buf_idx = 0;
            
            // Extract columns for rank p
            for (size_t j = 0; j < n; ++j) {
                if ((int)(j % num_procs) == p) {
                    // Copy column j from A_full (row-major) to send_buf (column-major)
                    for (size_t i = 0; i < n; ++i) {
                        send_buf[buf_idx * n + i] = A_full[i * n + j];
                    }
                    buf_idx++;
                }
            }
            
            if (p == 0) {
                local_A = send_buf;
            } else {
                MPI_Send(send_buf.data(), n * p_cols_count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_A.data(), n * n_local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecompositionMPI(local_A, n, rank, num_procs);
    
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
    
    // Gather and Validate
    if (validate || printResults) {
        std::vector<double> A_final_row_major;
        
        if (rank == 0) {
            A_final_row_major.resize(n * n);
            std::vector<double> recv_buf;
            
            for (int p = 0; p < num_procs; ++p) {
                size_t p_cols_count = 0;
                for (size_t j = 0; j < n; ++j) if ((int)(j % num_procs) == p) p_cols_count++;
                
                recv_buf.resize(n * p_cols_count);
                
                if (p == 0) {
                    recv_buf = local_A;
                } else {
                    MPI_Recv(recv_buf.data(), n * p_cols_count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                // Unpack into A_final_row_major
                size_t buf_idx = 0;
                for (size_t j = 0; j < n; ++j) {
                    if ((int)(j % num_procs) == p) {
                        for (size_t i = 0; i < n; ++i) {
                            // recv_buf is column-major: [col_idx * n + row]
                            // A_final is row-major: [row * n + col]
                            A_final_row_major[i * n + j] = recv_buf[buf_idx * n + i];
                        }
                        buf_idx++;
                    }
                }
            }
            
            if (printResults) {
                print_results(A_final_row_major, "CholeskyL");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(A_final_row_major, A_orig, n);
                if (valid) printf("Validation: PASSED\n");
                else printf("Validation: FAILED\n");
            }
        } else {
            MPI_Send(local_A.data(), n * n_local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}
