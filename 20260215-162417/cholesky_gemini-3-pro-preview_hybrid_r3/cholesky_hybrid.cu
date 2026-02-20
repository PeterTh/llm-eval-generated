#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSOLVER error at %s:%d\n", __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Helper to get local block index
int get_local_block_index(int global_block_idx, int rank, int size) {
    return (global_block_idx - rank) / size;
}

void cholesky_hybrid(int N, int B, int rank, int size, double* h_A_local, int max_local_rows) {
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // Matrix dimensions
    int n_blocks = (N + B - 1) / B;
    
    // Calculate actual local rows
    int local_rows_total = 0;
    for (int k = rank; k < n_blocks; k += size) {
        int block_height = std::min(B, N - k * B);
        local_rows_total += block_height;
    }

    // Allocate Device Memory
    double *d_A;
    size_t matrix_bytes = (size_t)local_rows_total * N * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_local, matrix_bytes, cudaMemcpyHostToDevice));

    // Workspaces
    double *d_diag;
    CUDA_CHECK(cudaMalloc((void**)&d_diag, B * B * sizeof(double)));
    
    double *h_diag = new double[B * B];
    
    // Panel buffer: Holds the full column k (L_{k:N, k}). 
    // Max size is N * B.
    double *d_panel_full;
    CUDA_CHECK(cudaMalloc((void**)&d_panel_full, (size_t)N * B * sizeof(double)));
    
    // Host buffers for gathering panel
    // Not strictly needed if we gather to d_panel_full via temp host buf, allocated dynamically

    // cuSolver workspace
    int Lwork;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, B, d_diag, B, &Lwork));
    double *d_work;
    CUDA_CHECK(cudaMalloc((void**)&d_work, Lwork * sizeof(double)));
    int *d_info;
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    // Constants for cuBLAS
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    // ----------------------------------------------------------------
    // MAIN LOOP
    // ----------------------------------------------------------------
    for (int k = 0; k < n_blocks; ++k) {
        int root = k % size;
        int current_B = std::min(B, N - k * B);
        
        // ============================================================
        // 1. Factor Diagonal Block (Owner only)
        // ============================================================
        if (rank == root) {
            int local_block_idx = get_local_block_index(k, rank, size);
            // Calculate exact row offset
            size_t row_offset = 0;
            for(int bb=0; bb<local_block_idx; ++bb) {
                 int r = bb * size + rank;
                 row_offset += std::min(B, N - r * B);
            }
            
            // Pointer to A_kk. Column k*B.
            // d_A is (local_rows x N).
            // Element (r, c) is at c * local_rows + r.
            // A_kk starts at local row `row_offset`, col `k*B`.
            double *d_A_kk = d_A + (size_t)(k * B) * local_rows_total + row_offset;
            
            // Copy to d_diag (contiguous)
            // Copy submatrix BxB. Src LDA = local_rows_total. Dst LDA = current_B.
            CUBLAS_CHECK(cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     current_B, current_B,
                                     &one, d_A_kk, local_rows_total,
                                     &zero, NULL, current_B,
                                     d_diag, current_B));
            
            // Factor
            CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, current_B, 
                                            d_diag, current_B, d_work, Lwork, d_info));
            
            // Copy back to A_kk (needed locally)
            CUBLAS_CHECK(cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     current_B, current_B,
                                     &one, d_diag, current_B,
                                     &zero, NULL, local_rows_total,
                                     d_A_kk, local_rows_total));
                                     
            // Copy to host for broadcast
            CUDA_CHECK(cudaMemcpy(h_diag, d_diag, current_B * current_B * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // ============================================================
        // 2. Broadcast Factored Block
        // ============================================================
        MPI_Bcast(h_diag, current_B * current_B, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        if (rank != root) {
            CUDA_CHECK(cudaMemcpy(d_diag, h_diag, current_B * current_B * sizeof(double), cudaMemcpyHostToDevice));
        }

        // ============================================================
        // 3. Update Panel (TRSM) - Local Computation
        // ============================================================
        // Each process updates its part of the column panel L_{i, k}
        // Iterate over local blocks > k
        
        int local_blks = (n_blocks - rank + size - 1) / size;
        
        for (int lb = 0; lb < local_blks; ++lb) {
            int global_block = lb * size + rank;
            if (global_block <= k) continue; 
            
            int block_h = std::min(B, N - global_block * B);
            
            // Calculate row_offset for this block
            size_t row_offset = 0;
            for(int bb=0; bb<lb; ++bb) {
                 int r = bb * size + rank;
                 row_offset += std::min(B, N - r * B);
            }

            // Pointer to A_{global_block, k}
            double *d_A_ik = d_A + (size_t)(k * B) * local_rows_total + row_offset;
            
            // TRSM
            // A_{ik} = A_{ik} * (L_kk^T)^-1
            CUBLAS_CHECK(cublasDtrsm(cublas_handle,
                                     CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                     block_h, current_B,
                                     &one,
                                     d_diag, current_B,
                                     d_A_ik, local_rows_total));
        }
        
        // ============================================================
        // 4. Gather Panel (All processes need the full column k)
        // ============================================================
        
        // Prepare send buffer
        // Size: Sum of heights of my blocks below k * current_B
        int my_panel_rows = 0;
        for (int lb = 0; lb < local_blks; ++lb) {
            int global_block = lb * size + rank;
            if (global_block <= k) continue;
            my_panel_rows += std::min(B, N - global_block * B);
        }
        
        std::vector<double> h_send_buf(my_panel_rows * current_B);
        
        // Copy data from device to send buffer
        int offset_buf = 0;
        for (int lb = 0; lb < local_blks; ++lb) {
            int global_block = lb * size + rank;
            if (global_block <= k) continue;
            
            int block_h = std::min(B, N - global_block * B);
            
            // Row offset logic
            size_t row_offset = 0;
            for(int bb=0; bb<lb; ++bb) {
                 int r = bb * size + rank;
                 row_offset += std::min(B, N - r * B);
            }
            
            double *d_src = d_A + (size_t)(k * B) * local_rows_total + row_offset;
            
            // Copy submatrix to host buffer (contiguous)
            // d_src stride is local_rows_total
            // h_send_buf stride is block_h (packed)
            CUBLAS_CHECK(cublasGetMatrix(block_h, current_B, sizeof(double),
                                         d_src, local_rows_total,
                                         h_send_buf.data() + offset_buf, block_h));
            offset_buf += block_h * current_B;
        }
        
        // MPI Parameters for Allgatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_send_count = my_panel_rows * current_B;
        MPI_Allgather(&my_send_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        int total_elems = 0;
        for(int i=0; i<size; ++i) {
            displs[i] = total_elems; // Displacement in doubles
            total_elems += recvcounts[i];
        }
        
        std::vector<double> h_recv_buf(total_elems > 0 ? total_elems : 1); 
        
        if (total_elems > 0) {
            MPI_Allgatherv(h_send_buf.data(), my_send_count, MPI_DOUBLE,
                           h_recv_buf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            
            // Copy to Device
            CUDA_CHECK(cudaMemcpy(d_panel_full, h_recv_buf.data(), total_elems * sizeof(double), cudaMemcpyHostToDevice));
        }

        // ============================================================
        // 5. Update Trailing Submatrix (SYRK/GEMM)
        // ============================================================
        
        for (int lb = 0; lb < local_blks; ++lb) {
            int i_blk = lb * size + rank; // Global row block index
            if (i_blk <= k) continue;
            
            int i_h = std::min(B, N - i_blk * B);
            
            // Calculate row_offset for i_blk
            size_t row_offset_i = 0;
            for(int bb=0; bb<lb; ++bb) {
                 int r = bb * size + rank;
                 row_offset_i += std::min(B, N - r * B);
            }
            
            // Pointer to L_{i,k} (My own panel part)
            // Use d_A
            double *d_Lik = d_A + (size_t)(k * B) * local_rows_total + row_offset_i;

            // Iterate over global block columns j_blk
            // We only update A_{i,j} for j <= i (Lower triangular).
            // So j_blk goes from k+1 to i_blk.
            for (int j_blk = k + 1; j_blk <= i_blk; ++j_blk) {
                int j_h = std::min(B, N - j_blk * B);
                
                // Find L_{j,k} in d_panel_full
                int owner = j_blk % size;
                
                // Calculate offset in d_panel_full
                // Offset = displs[owner] + (offset within owner's data)
                // Offset within owner: sum of heights of blocks owned by `owner` before `j_blk` (but after k) * current_B
                
                int offset_within_owner = 0;
                int start_blk_idx_owner = (owner <= k % size) ? (k / size + 1) : (k / size);
                if (owner < k % size) start_blk_idx_owner = k / size + 1; 
                
                // Iterate owner's blocks to find offset
                for (int bb = start_blk_idx_owner; ; ++bb) {
                    int global_b = bb * size + owner;
                    if (global_b >= j_blk) break;
                    if (global_b > k) {
                         offset_within_owner += std::min(B, N - global_b * B);
                    }
                }
                
                size_t offset_Ljk = displs[owner] + (size_t)offset_within_owner * current_B; // In doubles
                double *d_Ljk = d_panel_full + offset_Ljk;

                // Update A_{i,j}
                double *d_Aij = d_A + (size_t)(j_blk * B) * local_rows_total + row_offset_i;
                
                // GEMM
                // A_{ij} = A_{ij} - L_{ik} * L_{jk}^T
                // Dimensions: (i_h x j_h)
                // L_{ik} is (i_h x current_B)
                // L_{jk} is (j_h x current_B) packed column major.
                // We want L_{jk}^T (current_B x j_h).
                
                CUBLAS_CHECK(cublasDgemm(cublas_handle,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         i_h, j_h, current_B,
                                         &minus_one,
                                         d_Lik, local_rows_total,
                                         d_Ljk, j_h, // Stride of Ljk in packed panel is j_h (column height)
                                         &one,
                                         d_Aij, local_rows_total));
            }
        }
    }

    cudaFree(d_A);
    cudaFree(d_diag);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_panel_full);
    delete[] h_diag;
    
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) std::cout << "Usage: " << argv[0] << " N B" << std::endl;
        MPI_Finalize();
        return 0;
    }
    
    int N = atoi(argv[1]);
    int B = atoi(argv[2]);
    
    // Max local rows
    int max_local_rows = (N + B - 1) / size * B + B; // Upper bound
    double* h_A_local = new double[max_local_rows * N];
    
    // Initialize with dummy data (Identity + Random)
    // To be positive definite: A = I*N + Random
    // We only init local part
    int n_blocks = (N + B - 1) / B;
    int local_blks = (n_blocks - rank + size - 1) / size;
    
    // Simple init for compilation/check
    for(int i=0; i<max_local_rows*N; ++i) h_A_local[i] = 0.0;
    for(int lb=0; lb<local_blks; ++lb) {
        int global_blk = lb * size + rank;
        int h = std::min(B, N - global_blk * B);
        
        // This is just a placeholder to ensure memory is touched
        // Logic for proper initialization is complex for distributed block cyclic
    }
    
    // Call solver
    if (rank == 0) std::cout << "Starting Cholesky Hybrid N=" << N << " B=" << B << std::endl;
    cholesky_hybrid(N, B, rank, size, h_A_local, max_local_rows);
    if (rank == 0) std::cout << "Done." << std::endl;

    delete[] h_A_local;
    MPI_Finalize();
    return 0;
}
