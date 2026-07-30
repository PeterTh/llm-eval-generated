#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

#define BLOCK_SIZE 128

// MPI-parallel Cholesky decomposition using blocked right-looking algorithm
// with 1D column-block distribution and optimized cache locality.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Column ownership: contiguous blocks
    size_t cols_per_proc = (n + nprocs - 1) / nprocs;
    size_t my_col_start = (size_t)rank * cols_per_proc;
    size_t my_col_end = std::min(my_col_start + cols_per_proc, n);
    size_t my_num_cols = my_col_end - my_col_start;

    // Local storage: only my columns, column-major layout
    std::vector<double> local_A(n * my_num_cols, 0.0);

    // Distribute initial matrix columns from root
    if (rank == 0) {
        for (int p = 0; p < nprocs; ++p) {
            size_t cs = (size_t)p * cols_per_proc;
            size_t ce = std::min(cs + cols_per_proc, n);
            size_t nc = ce - cs;
            
            std::vector<double> buf(n * nc);
            for (size_t j = 0; j < nc; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    buf[j * n + i] = A[i * n + (cs + j)];
                }
            }
            
            if (p == 0) {
                local_A = buf;
            } else {
                MPI_Send(buf.data(), (int)(n * nc), MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_A.data(), (int)(n * my_num_cols), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast buffer for computed panel
    std::vector<double> panel_buf(n * BLOCK_SIZE);

    // Blocked right-looking Cholesky
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (size_t bj = 0; bj < num_blocks; ++bj) {
        size_t j_start = bj * BLOCK_SIZE;
        size_t j_end = std::min(j_start + BLOCK_SIZE, n);
        size_t block_width = j_end - j_start;
        
        int owner = (int)(j_start / cols_per_proc);

        if (rank == owner) {
            // Factor the diagonal block (columns j_start to j_end-1)
            for (size_t j = j_start; j < j_end; ++j) {
                size_t local_j = j - my_col_start;
                
                // L[j,j] = sqrt(A[j,j])
                double diag = local_A[local_j * n + j];
                if (diag <= 0.0) {
                    return false;
                }
                double Ljj = sqrt(diag);
                local_A[local_j * n + j] = Ljj;

                // L[i,j] = A[i,j] / L[j,j] for i > j
                for (size_t i = j + 1; i < n; ++i) {
                    local_A[local_j * n + i] /= Ljj;
                }
                
                // Rank-1 update for remaining columns in this block
                for (size_t k = j + 1; k < j_end; ++k) {
                    size_t local_k = k - my_col_start;
                    double Lkj = local_A[local_j * n + k];
                    for (size_t i = j + 1; i < n; ++i) {
                        local_A[local_k * n + i] -= local_A[local_j * n + i] * Lkj;
                    }
                }
            }

            // Fill broadcast buffer with the panel
            for (size_t j = 0; j < block_width; ++j) {
                size_t local_j = j_start + j - my_col_start;
                for (size_t i = 0; i < j_start + j; ++i) panel_buf[j * n + i] = 0.0;
                for (size_t i = j_start + j; i < n; ++i) panel_buf[j * n + i] = local_A[local_j * n + i];
            }
        }

        // Broadcast panel to all processes
        MPI_Bcast(panel_buf.data(), (int)(n * block_width), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Rank-k update: each process updates its own columns k > j_end-1
        // Optimized loop order for better cache locality
        size_t ks = std::max(j_end, my_col_start);
        size_t ke = my_col_end;
        
        if (ks < ke) {
            // Outer loop over block columns for better cache reuse
            for (size_t j = 0; j < block_width; ++j) {
                double* panel_j = &panel_buf[j * n];
                
                // Update columns in my range
                for (size_t k = ks; k < ke; ++k) {
                    size_t local_k = k - my_col_start;
                    double Lkj = panel_j[k];
                    double* col_k = &local_A[local_k * n];
                    
                    // Update elements i >= max(j_start + j, k)
                    size_t i_start = std::max(j_start + j, k);
                    for (size_t i = i_start; i < n; ++i) {
                        col_k[i] -= panel_j[i] * Lkj;
                    }
                }
            }
        }
    }

    // Gather results back to root
    if (rank == 0) {
        for (int p = 1; p < nprocs; ++p) {
            size_t cs = (size_t)p * cols_per_proc;
            size_t ce = std::min(cs + cols_per_proc, n);
            size_t nc = ce - cs;
            
            std::vector<double> buf(n * nc);
            MPI_Recv(buf.data(), (int)(n * nc), MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (size_t j = 0; j < nc; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    A[i * n + (cs + j)] = buf[j * n + i];
                }
            }
        }
        
        // Copy local columns
        for (size_t j = 0; j < my_num_cols; ++j) {
            for (size_t i = 0; i < n; ++i) {
                A[i * n + (my_col_start + j)] = local_A[j * n + i];
            }
        }
        
        // Zero out upper triangle
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
    } else {
        MPI_Send(local_A.data(), (int)(n * my_num_cols), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    return true;
}

// Generate a symmetric positive definite matrix
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
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
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
            printUsage(argv[0]);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A;
        }
    }
    
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
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
