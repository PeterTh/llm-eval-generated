#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (row-cyclic distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& local_A, const size_t n, int rank, int size) {
    // local_A stores rows owned by this process: row i if i % size == rank
    // Stored contiguously: local_row_0, local_row_1, ...
    // Each local_row has n elements (columns 0 to n-1)

    std::vector<double> diag_row(n); // Buffer to store the current pivot row
    
    for (size_t k = 0; k < n; ++k) {
        int root = k % size;

        // 1. Pivot process computes diagonal and prepares the k-th row for broadcast
        if (rank == root) {
            size_t local_k = k / size;
            // Access element A[k][k]
            double val = local_A[local_k * n + k];
            
            // Subtract dot product of the row with itself (for previous columns)
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                double v = local_A[local_k * n + p];
                sum += v * v;
            }
            val -= sum;
            
            // Check positive definite
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu (val=%.5e)\n", k, val);
                // We need to notify others to abort? 
                // For simplicity in this benchmark, we might just abort or return false.
                // But let's assume valid input for now or handle gracefully.
                // Proper error handling in MPI requires communication.
            } else {
                val = sqrt(val);
                local_A[local_k * n + k] = val;
                
                // Scale the row k (elements k+1 to n-1 are 0 in L, so only update 0 to k-1? 
                // Wait, L is lower triangular. We need elements A[k][j] for j < k.
                // Actually, for Cholesky L * L^T, we compute column by column or row by row.
                // "Right-looking":
                // L_kk = sqrt(A_kk)
                // L_ik = A_ik / L_kk  (for i > k)
                // A_ij = A_ij - L_ik * L_jk (for i, j > k)
                
                // My local_A stores ROWS.
                // So I have A[i][0...n-1].
                // If I am root (owner of row k), I have A[k][...].
                // I compute L[k][k].
                // Then I need L[k][j] for j < k? No, those are already computed.
                // I need to broadcast the k-th row of L?
                // Wait, L is lower triangular. Row k has elements at 0, 1, ..., k.
                // A[k][j] for j > k is 0 in L.
                // So the "row k" of L is just L[k][0...k].
                // But for the update A[i][j] -= L[ik] * L[jk], we need L[ik] and L[jk].
                // This looks like we need COLUMN k.
                
                // Let's stick to the algorithm that matches the data layout (Row-Cyclic).
                // "Row-Cholesky" (Crout):
                // For i = 0 to n-1:
                //   For j = 0 to i:
                //     sum = ...
                //     L[i][j] = ...
                
                // If I own row i:
                // I need L[j][0...j] for all j < i.
                // This means I need *all previous rows* to compute my row i.
                // This implies broadcasting every completed row to everyone.
                
                // Algorithm:
                // For k = 0 to n-1:
                //   root = k % size
                //   If rank == root:
                //     finalize L[k][k] (diagonal)
                //     Prepare row k (elements 0 to k)
                //   Bcast row k (elements 0 to k) from root.
                //   Everyone stores row k? No, use it to update pending rows?
                //   
                //   If I own row i > k:
                //     I can compute L[i][k] ?
                //     L[i][k] = (A[i][k] - sum(L[i][p] * L[k][p] for p < k)) / L[k][k].
                //     I have L[i][p] (computed in my row).
                //     I just received L[k][p] (from Bcast).
                //     So I can compute L[i][k].
                
                //   So, at step k, we broadcast row k.
                //   Then everyone who owns rows i > k can compute the k-th element of their rows.
                
                //   This is perfect!
                //   Only need to broadcast k+1 doubles.
                
                //   Optimization:
                //   Only Bcast row k.
                //   Process with row i > k computes L[i][k] immediately.
                
                //   Does this work for scaling?
                //   L[i][k] depends on dot product of row i (partial) and row k (full).
                
                //   Yes.
                
            }
        }
        
        // Broadcast the pivot row k (elements 0 to k)
        // We can reuse a buffer.
        
        if (rank == root) {
            size_t local_k = k / size;
            std::copy(local_A.begin() + local_k * n, local_A.begin() + local_k * n + (k + 1), diag_row.begin());
        }
        
        // Broadcast row k
        MPI_Bcast(diag_row.data(), k + 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        // Now update local rows i > k
        // For each local row i (where global_i > k):
        //   L[i][k] = (A[i][k] - dot(L[i][0..k-1], L[k][0..k-1])) / L[k][k]
        
        // But wait, A[i][k] is modified in place?
        // Original code:
        // sum = 0; for p < k: sum += A[i][p] * A[k][p];
        // A[i][k] = (A[i][k] - sum) / A[k][k];
        
        // Yes, diag_row contains A[k][0...k].
        // diag_row[k] is L[k][k].
        
        double diag_val = diag_row[k];
        if (diag_val <= 0.0) return false; // Error handled by checking result later
        
        // Iterate over local rows
        size_t start_local_row = (k + 1) / size;
        if ((k + 1) % size > (size_t)rank) start_local_row++;
        // If rank <= (k+1)%size, we might start earlier?
        // Let's just iterate all local rows and check global index.
        
        for (size_t local_i = 0; local_i < local_A.size() / n; ++local_i) {
            size_t global_i = local_i * size + rank;
            if (global_i <= k) continue; // Already done
            
            double sum = 0.0;
            // Compute dot product of row i (up to k-1) and row k (up to k-1)
            // Vectorization friendly loop
            for (size_t p = 0; p < k; ++p) {
                sum += local_A[local_i * n + p] * diag_row[p];
            }
            
            local_A[local_i * n + k] = (local_A[local_i * n + k] - sum) / diag_val;
        }
    }
    
    // Zero out upper triangular part for local rows
    for (size_t local_i = 0; local_i < local_A.size() / n; ++local_i) {
        size_t global_i = local_i * size + rank;
        for (size_t j = global_i + 1; j < n; ++j) {
            local_A[local_i * n + j] = 0.0;
        }
    }
    
    return true;
}

// Generate a symmetric positive definite matrix (distributed)
void generatePositiveDefiniteMatrix(std::vector<double>& local_A, const size_t n, int rank, int size) {
    // Replicate B generation logic
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute owned rows of A
    // Row i of A = Row i of B * B^T
    // A[i][j] = dot(row i of B, row j of B)
    
    // Determine how many rows this rank owns
    size_t num_rows = n / size + (rank < (int)(n % size) ? 1 : 0);
    local_A.resize(num_rows * n);
    
    size_t local_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size != (size_t)rank) continue;
        
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            local_A[local_idx * n + j] = sum;
        }
        
        // Add diagonal dominance
        local_A[local_idx * n + i] += n;
        
        local_idx++;
    }
}

// Helper to gather distributed matrix to rank 0
std::vector<double> gatherMatrix(const std::vector<double>& local_A, const size_t n, int rank, int size) {
    std::vector<double> A_full;
    if (rank == 0) A_full.resize(n * n);
    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int my_count = local_A.size();
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    std::vector<double> gathered_buffer;
    // Upper bound size: each process has approx n*n/size
    // But Gatherv requires correct sizing.
    // Total size is sum(recvcounts).
    size_t total_recv = 0;
    if (rank == 0) {
        for(int c : recvcounts) total_recv += c;
        gathered_buffer.resize(total_recv);
    }
    
    MPI_Gatherv(local_A.data(), my_count, MPI_DOUBLE, 
                gathered_buffer.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
                
    if (rank == 0) {
        // Reconstruct A_full from cyclic parts
        for (int p = 0; p < size; ++p) {
            int count = recvcounts[p]; // number of doubles
            int num_rows = count / n;
            double* src = gathered_buffer.data() + displs[p];
            
            for (int r = 0; r < num_rows; ++r) {
                // Determine global row index
                // Process p owns: p, p+size, p+2*size...
                int global_row = p + r * size;
                if (global_row < (int)n) { // Safety check
                    std::memcpy(A_full.data() + global_row * n, src + r * n, n * sizeof(double));
                }
            }
        }
    }
    return A_full;
}

bool validateCholesky(const std::vector<double>& local_A, const size_t n, int rank, int size) {
    // Gather full matrix A on rank 0
    std::vector<double> A_full = gatherMatrix(local_A, n, rank, size);
    
    if (rank == 0) {
        // Now run sequential validation on A_full
        // We need original matrix too!
        // Re-generate original matrix on rank 0
        std::vector<double> A_orig(n * n);
        // ... (copy logic from generatePositiveDefiniteMatrix but sequential)
        {
            std::vector<double> B(n * n);
            unsigned int seed = 42;
            for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
                    A_orig[i * n + j] = sum;
                }
                A_orig[i * n + i] += n;
            }
        }
        
        // Use existing validation logic (adapted)
        // ... (copy validateCholesky body)
        // Compute L * L^T
        std::vector<double> reconstructed(n * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += A_full[i * n + k] * A_full[j * n + k];
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
    
    return true; // Non-root ranks always pass (they didn't check)
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
    
    // Parse command line arguments
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
        printf("Processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    
    // Allocate local matrix
    std::vector<double> local_A;
    
    generatePositiveDefiniteMatrix(local_A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(local_A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    if (printResults) {
        std::vector<double> A_full = gatherMatrix(local_A, n, rank, size);
        if (rank == 0) {
            print_results(A_full, "CholeskyL");
        }
    }
    
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(local_A, n, rank, size);
        if (rank == 0) {
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}

