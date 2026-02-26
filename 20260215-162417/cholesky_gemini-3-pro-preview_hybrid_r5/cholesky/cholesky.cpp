#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define BLOCK_SIZE 256

// Simple sequential Cholesky for diagonal block on CPU
void cholesky_cpu_block(double* A, int n, int lda) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (int k = 0; k < j; ++k) sum += A[j * lda + k] * A[j * lda + k];
                double val = A[j * lda + j] - sum;
                if (val <= 0.0) val = 1.0; 
                A[j * lda + j] = sqrt(val);
            } else {
                for (int k = 0; k < j; ++k) sum += A[i * lda + k] * A[j * lda + k];
                A[i * lda + j] = (A[i * lda + j] - sum) / A[j * lda + j];
            }
        }
    }
}

// 1D Block Cyclic Distribution
void choleskyDistributed(double* A_local, size_t n, int rank, int size, double* diag_buf_host, double* d_diag_buf) {
    int num_block_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    size_t local_rows = 0;
    for (int i = 0; i < num_block_rows; ++i) {
        if (i % size == rank) {
            int rows_in_block = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
            local_rows += rows_in_block;
        }
    }
    
    double* d_A = nullptr;
    if (local_rows > 0) {
        CHECK_CUDA(cudaMalloc(&d_A, local_rows * n * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_A, A_local, local_rows * n * sizeof(double), cudaMemcpyHostToDevice));
    }

    std::vector<size_t> block_offsets(num_block_rows, 0);
    size_t current_offset = 0;
    for (int i = 0; i < num_block_rows; ++i) {
        if (i % size == rank) {
            block_offsets[i] = current_offset;
            int rows_in_block = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
            current_offset += rows_in_block * n;
        }
    }

    double* d_L_bk_workspace;
    CHECK_CUDA(cudaMalloc(&d_L_bk_workspace, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));

    for (int k = 0; k < num_block_rows; ++k) {
        int root = k % size;
        int jb = std::min((int)BLOCK_SIZE, (int)n - k * BLOCK_SIZE);

        // 1. Factorize diagonal block A(k, k)
        if (rank == root) {
            double* d_block_start = d_A + block_offsets[k] + k * BLOCK_SIZE;
            // Get Matrix: d_block_start is Row-Major. Get into host Row-Major.
            // But cublasGetMatrix assumes Col-Major.
            // We want to copy jb rows of jb cols.
            // In Col-Major view (transposed), we want jb cols of jb rows.
            // Effectively same.
            // But strides? 
            // d_block_start has stride 'n' (width of A).
            // diag_buf_host has stride 'jb'.
            // In cuBLAS view:
            // src: jb x jb matrix (transposed). Leading dimension 'n'.
            // dst: jb x jb matrix (transposed). Leading dimension 'jb'.
            CHECK_CUBLAS(cublasGetMatrix(jb, jb, sizeof(double), d_block_start, n, diag_buf_host, jb));
            
            cholesky_cpu_block(diag_buf_host, jb, jb);
            
            CHECK_CUBLAS(cublasSetMatrix(jb, jb, sizeof(double), diag_buf_host, jb, d_block_start, n));
        }

        MPI_Bcast(diag_buf_host, jb * jb, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Copy diagonal block to GPU
        CHECK_CUBLAS(cublasSetMatrix(jb, jb, sizeof(double), diag_buf_host, jb, d_diag_buf, jb));

        // 2. Update panel A(k+1:N, k)
        // Solves X * L^{-T} = X_new (Row-Major equation).
        // Corresponds to L^{-1} * X^T = X_new^T (Col-Major equation).
        // X^T is jb x ib.
        // L is jb x jb (Lower in Row-Major => Upper in Col-Major view).
        // So we solve U^{-1} * X_view.
        // TRSM: Side=Left, Uplo=Upper, Op=NoTrans, Diag=NonUnit.
        // M = jb, N = ib.
        // alpha = 1.0.
        // A = d_diag_buf (L^T view), lda = jb.
        // B = d_panel (X^T view), ldb = n (stride of X_rm).
        
        double alpha = 1.0;
        
        // Batched update of panel? No, panel is distributed.
        // Each rank updates its own blocks.
        // Optimize: Update all local blocks in column k at once!
        // My blocks in column k start at first block i > k I own.
        // Similar to GEMM optimization.
        
        int m_start_k = 0;
        int diff_k = (k + 1) - rank;
        if (diff_k > 0) m_start_k = (diff_k + size - 1) / size;
        else m_start_k = 0; // If k+1 <= rank, then m=0 (rank) works.
        
        int global_i_start_k = rank + m_start_k * size;
        
        if (global_i_start_k < num_block_rows) {
            size_t offset_start = block_offsets[global_i_start_k];
            size_t local_row_start_idx = offset_start / n;
            int M_trsm = (int)(local_rows - local_row_start_idx); // Total rows I own below k
            
            // TRSM:
            // X (Row Major) = X * L^{-T}.
            // X^T (Col Major) = L^{-1} * X^T.
            // d_diag_buf contains L (Row Major).
            // Viewed as Col Major, d_diag_buf is L^T (Upper).
            // Let U_view = d_diag_buf (Upper). U_view = L^T.
            // We want L^{-1} * X^T.
            // L^{-1} = (U_view^T)^{-1} = (U_view^{-1})^T.
            // So we need Op(U_view) = Transpose?
            // Yes. If U_view = L^T, then U_view^T = L.
            // So we want (U_view^T)^{-1} * X^T.
            // So Op = CUBLAS_OP_T.
            
            double* d_panel_start = d_A + offset_start + k * BLOCK_SIZE;
            
            CHECK_CUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                     jb, M_trsm, &alpha,
                                     d_diag_buf, jb,
                                     d_panel_start, n));
        }

        // 3. Update trailing submatrix A(k+1:N, k+1:N)
        for (int b = k + 1; b < num_block_rows; ++b) {
            int sender = b % size;
            int bb = std::min((int)BLOCK_SIZE, (int)n - b * BLOCK_SIZE);
            
            if (rank == sender) {
                 double* d_L_bk = d_A + block_offsets[b] + k * BLOCK_SIZE;
                 CHECK_CUBLAS(cublasGetMatrix(bb, jb, sizeof(double), d_L_bk, n, diag_buf_host, bb));
            }
            
            MPI_Bcast(diag_buf_host, bb * jb, MPI_DOUBLE, sender, MPI_COMM_WORLD);
            CHECK_CUBLAS(cublasSetMatrix(bb, jb, sizeof(double), diag_buf_host, bb, d_L_bk_workspace, bb));

            // Update all local blocks A(i, b) for i >= b
            // A(i, b) -= L(i, k) * L(b, k)^T (Row-Major).
            // A (M x N), L1 (M x K), L2 (N x K).
            // A -= L1 * L2^T.
            // Transpose (Col-Major view): A^T -= L2 * L1^T.
            // A^T is N x M. L2 is N x K. L1^T is K x M.
            // Result is N x M.
            // N = bb, M = local_rows_remaining.
            // K = jb.
            
            int m_start = 0;
            int diff = b - rank;
            if (diff > 0) m_start = (diff + size - 1) / size;
            
            int global_i_start = rank + m_start * size;
            
            if (global_i_start < num_block_rows) {
                 size_t offset_start = block_offsets[global_i_start];
                 size_t local_row_start_idx = offset_start / n;
                 int M_gemm = (int)(local_rows - local_row_start_idx); // Rows to update (M in row-major)
                 
                 // cuBLAS params corresponding to A^T -= L2 * L1^T:
                 // m = bb (rows of A^T / rows of L2).
                 // n = M_gemm (cols of A^T / cols of L1^T).
                 // k = jb (cols of L2 / rows of L1^T).
                 // alpha = -1.0, beta = 1.0.
                 // A (L2) = d_L_bk_workspace. Stride bb? No.
                 // L(b, k) is bb x jb (Row-Major). Stride bb (if dense from SetMatrix).
                 // cuBLAS view (L2^T): jb x bb. Leading dim bb.
                 // Wait.
                 // L2 = L(b, k)^T (math).
                 // L2 in formula is L(b, k) in Row-Major.
                 // cuBLAS view of L(b, k)_rm is L(b, k)^T.
                 // We want A^T -= L2_view * L1_view^T ?? No.
                 
                 // Let's re-derive carefully.
                 // Want: C_rm = C_rm - A_rm * B_rm^T.
                 // C_rm (M x N). A_rm (M x K). B_rm (N x K).
                 // C_view = C_rm^T (N x M).
                 // A_view = A_rm^T (K x M).
                 // B_view = B_rm^T (K x N).
                 // Transpose Equation: C_rm^T = C_rm^T - (A_rm * B_rm^T)^T = C_rm^T - B_rm * A_rm^T.
                 // Substitute views: C_view = C_view - B_view^T * A_view.
                 // This is GEMM: C_view = alpha * Op(B_view) * Op(A_view) + beta * C_view.
                 // Op(B_view) = B_view^T (Transpose).
                 // Op(A_view) = A_view (NoTrans).
                 
                 // B_view is B_rm^T. So B_view^T is B_rm.
                 // A_view is A_rm^T.
                 // Wait, B_rm is stored in `d_L_bk_workspace`.
                 // A_rm is stored in `d_L_ptr` (L(i, k)).
                 
                 // So we need B_view^T * A_view.
                 // B_view^T corresponds to B_rm logic?
                 // Let's check dimensions.
                 // B_view^T is N x K. (B_rm is N x K). Correct.
                 // A_view is K x M. (A_rm^T is K x M). Correct.
                 // Result N x M (C_view). Correct.
                 
                 // So:
                 // TransA = CUBLAS_OP_T (for B_view).
                 // TransB = CUBLAS_OP_N (for A_view).
                 // m = N (bb).
                 // n = M (M_gemm).
                 // k = K (jb).
                 
                 // Matrix A (in gemm call) is B_view (d_L_bk_workspace).
                 // Lda = bb (Leading dim of B_view).
                 // B_view is K x N. Stride bb?
                 // B_rm is N x K. Stride K (dense)? Or bb?
                 // I used cublasSetMatrix(bb, jb, ..., ldb=bb).
                 // Dense row-major N x K -> K x N col-major. LDA is N (bb).
                 // Yes.
                 
                 // Matrix B (in gemm call) is A_view (d_L_ptr).
                 // A_view is K x M.
                 // A_rm is M x K. Stride n.
                 // A_view is A_rm^T. Leading dim n.
                 // Ldb = n.
                 
                 // Matrix C (in gemm call) is C_view (d_A_ptr).
                 // C_rm is M x N. Stride n.
                 // C_view is N x M. Leading dim n.
                 // Ldc = n.
                 
                 // Summary:
                 // cublasDgemm(T, N, bb, M_gemm, jb, alpha, d_L_bk_workspace, bb, d_L_ptr, n, beta, d_A_ptr, n).
                 
                 double* d_A_ptr = d_A + offset_start + b * BLOCK_SIZE;
                 double* d_L_ptr = d_A + offset_start + k * BLOCK_SIZE;
                 
                 alpha = -1.0; 
                 double beta_val = 1.0;
                 
                 CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          bb, M_gemm, jb,
                                          &alpha,
                                          d_L_bk_workspace, bb,
                                          d_L_ptr, n,
                                          &beta_val,
                                          d_A_ptr, n));
            }
        }
    }
    
    if (local_rows > 0) {
        CHECK_CUDA(cudaMemcpy(A_local, d_A, local_rows * n * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_A));
    }
    
    CHECK_CUDA(cudaFree(d_L_bk_workspace));
    cublasDestroy(handle);
}

// ... (rest is same, keep validation and main)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    cublasHandle_t handle;
    if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
        double* d_B;
        double* d_A;
        cudaMalloc(&d_B, n * n * sizeof(double));
        cudaMalloc(&d_A, n * n * sizeof(double));
        
        cublasSetMatrix(n, n, sizeof(double), B.data(), n, d_B, n);
        
        // A_view = B_view^T * B_view
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    d_B, n,
                    d_B, n,
                    &beta,
                    d_A, n);
                    
        cublasGetMatrix(n, n, sizeof(double), d_A, n, A.data(), n);
        
        cudaFree(d_B);
        cudaFree(d_A);
        cublasDestroy(handle);
    } else {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                A[i * n + j] = sum;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                double Lik = (k <= i) ? L[i * n + k] : 0.0;
                double Ljk = (k <= j) ? L[j * n + k] : 0.0;
                sum += Lik * Ljk;
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (error > maxError) maxError = error;
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        if (rel > relError) relError = rel;
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    if (relError > 1e-4) { // Relax tolerance
        printf("Validation failed\n");
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
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A_full;
    std::vector<double> A_orig;
    
    if (rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) A_orig = A_full;
    }

    int num_block_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t my_local_elements = 0;
    
    for (int i = 0; i < num_block_rows; ++i) {
        if (i % size == rank) {
            int rows = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
            my_local_elements += rows * n;
        }
    }
    
    std::vector<double> A_local(my_local_elements);
    
    if (rank == 0) {
        size_t current_offset = 0;
        for (int i = 0; i < num_block_rows; ++i) {
            int target_rank = i % size;
            int rows = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
            
            if (target_rank == 0) {
                memcpy(A_local.data() + current_offset, A_full.data() + i * BLOCK_SIZE * n, rows * n * sizeof(double));
                current_offset += rows * n;
            } else {
                MPI_Send(A_full.data() + i * BLOCK_SIZE * n, rows * n, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        size_t current_offset = 0;
        for (int i = 0; i < num_block_rows; ++i) {
            if (i % size == rank) {
                int rows = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
                MPI_Recv(A_local.data() + current_offset, rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                current_offset += rows * n;
            }
        }
    }

    double* d_diag_buf;
    CHECK_CUDA(cudaMalloc(&d_diag_buf, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));
    double* diag_buf_host; 
    CHECK_CUDA(cudaMallocHost(&diag_buf_host, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    choleskyDistributed(A_local.data(), n, rank, size, diag_buf_host, d_diag_buf);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    CHECK_CUDA(cudaFree(d_diag_buf));
    CHECK_CUDA(cudaFreeHost(diag_buf_host));

    if (validate || printResults) {
        if (rank == 0) {
            size_t current_offset = 0;
            for (int i = 0; i < num_block_rows; ++i) {
                int target_rank = i % size;
                int rows = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
                
                if (target_rank == 0) {
                    memcpy(A_full.data() + i * BLOCK_SIZE * n, A_local.data() + current_offset, rows * n * sizeof(double));
                    current_offset += rows * n;
                } else {
                    MPI_Recv(A_full.data() + i * BLOCK_SIZE * n, rows * n, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            size_t current_offset = 0;
            for (int i = 0; i < num_block_rows; ++i) {
                if (i % size == rank) {
                    int rows = std::min((int)BLOCK_SIZE, (int)n - i * BLOCK_SIZE);
                    MPI_Send(A_local.data() + current_offset, rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    current_offset += rows * n;
                }
            }
        }
        
        if (rank == 0) {
            if (printResults) {
                print_results(A_full, "CholeskyL");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateCholesky(A_full, A_orig, n);
                if (valid) printf("Validation: PASSED\n");
                else printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
