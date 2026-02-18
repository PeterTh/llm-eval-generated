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

#include "../common/results_output.hpp"

// CUDA kernel for extracting column k elements
__global__ void extractColumnKernel(double* A, double* col_k, int n, int k, int rank, int size, double diag_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate global row index i owned by this rank
    int rem = (k + 1) % size;
    int offset = (rank - rem + size) % size;
    int start_i = k + 1 + offset;
    
    int i = start_i + idx * size;
    
    if (i < n) {
        // Since we launch only for owned rows, i corresponds to an owned row.
        double val = A[i * n + k] / diag_val;
        A[i * n + k] = val; // Update A in place
        col_k[i] = val;
    }
}

// CUDA kernel for updating trailing matrix
__global__ void updateTrailingMatrixKernel(double* A, const double* col_k, int n, int k, int rank, int size) {
    // 2D grid:
    // x: column index j relative to k+1
    // y: row index (local owned row index)
    
    // Shared memory for col_k[j] values used by this block
    // blockDim.x is 32, so we need 32 doubles.
    __shared__ double s_col_k_j[32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j_local = blockIdx.x * blockDim.x + tx;
    int j = j_local + (k + 1);
    
    // Load col_k[j] into shared memory
    // Only first warp (ty=0) loads shared memory to avoid redundant global loads
    if (ty == 0) {
        if (j < n) {
            s_col_k_j[tx] = col_k[j];
        } else {
            s_col_k_j[tx] = 0.0;
        }
    }
    
    __syncthreads();

    int row_idx = blockIdx.y * blockDim.y + ty;
    
    // Calculate global row index i owned by this rank
    int rem = (k + 1) % size;
    int offset = (rank - rem + size) % size;
    int start_i = k + 1 + offset;
    
    int i = start_i + row_idx * size;
    
    if (i < n && j <= i) {
        // Compute update
        // Each thread updates one element A[i][j]
        // Use s_col_k_j[tx] instead of global load col_k[j]
        A[i * n + j] -= col_k[i] * s_col_k_j[tx];
    }
}

// Simple Cholesky decomposition (parallel MPI+OpenMP+CUDA)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    double *d_A, *d_col_k;
    cudaError_t err;

    // Allocate and copy matrix to GPU
    err = cudaMalloc(&d_A, n * n * sizeof(double));
    if (err != cudaSuccess) { printf("CUDA malloc A failed\n"); return false; }
    err = cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA memcpy A failed\n"); cudaFree(d_A); return false; }

    // Allocate buffer for column vector k on GPU
    err = cudaMalloc(&d_col_k, n * sizeof(double));
    if (err != cudaSuccess) { printf("CUDA malloc col failed\n"); cudaFree(d_A); return false; }

    std::vector<double> host_col_k(n);
    double diag_val;

    for (size_t k = 0; k < n; ++k) {
        // 1. Process diagonal element A[k][k]
        int root = k % size;
        
        if (rank == root) {
            // Fetch A[k][k] from GPU
            cudaMemcpy(&diag_val, d_A + k * n + k, sizeof(double), cudaMemcpyDeviceToHost);
            
            if (diag_val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu (val=%.4e)\n", k, diag_val);
                diag_val = -1.0; // Signal error
            } else {
                diag_val = sqrt(diag_val);
                // Update on GPU
                cudaMemcpy(d_A + k * n + k, &diag_val, sizeof(double), cudaMemcpyHostToDevice);
            }
        }
        
        // Broadcast diagonal value
        MPI_Bcast(&diag_val, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Update diagonal on all ranks (needed for consistency if we copy back full A)
        // Although strictly speaking only the owner needs it for computation,
        // if we want A to be correct on all ranks at the end (or at least on rank 0),
        // we should keep them in sync or gather at end.
        // For now, let's update it on GPU for everyone so d_A stays consistent.
        cudaMemcpy(d_A + k * n + k, &diag_val, sizeof(double), cudaMemcpyHostToDevice);
        
        if (diag_val < 0.0) {
            cudaFree(d_A); cudaFree(d_col_k);
            return false;
        }

        // 2. Prepare column k (L update)
        // We need L[i][k] for all i > k.
        // L[i][k] = A[i][k] / L[k][k]
        // Each rank computes its part of the column.
        
        // Use CUDA kernel to extract column k elements
        int num_rows = n - k - 1;
        if (num_rows > 0) {
            // Optimization: First zero out the column buffer
            cudaMemset(d_col_k + k + 1, 0, num_rows * sizeof(double));
            
            // Only launch threads for owned rows
            int rem = (k + 1) % size;
            int offset = (rank - rem + size) % size;
            int start_i = k + 1 + offset;
            
            int num_owned = 0;
            if (start_i < (int)n) {
                 num_owned = (n - 1 - start_i) / size + 1;
            }
            
            if (num_owned > 0) {
                int threadsPerBlock = 256;
                int blocks = (num_owned + threadsPerBlock - 1) / threadsPerBlock;
                extractColumnKernel<<<blocks, threadsPerBlock>>>(d_A, d_col_k, n, k, rank, size, diag_val);
            }
            
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                 printf("Extract Kernel launch failed: %s\n", cudaGetErrorString(kerr));
                 return false;
            }
            
            // Copy column to host for Allreduce
            // Note: d_col_k contains 0.0 for unowned rows, and valid values for owned.
            cudaMemcpy(host_col_k.data() + k + 1, d_col_k + k + 1, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Allreduce ONLY the relevant part of the column
            // We reduce elements k+1 to n-1.
            MPI_Allreduce(MPI_IN_PLACE, host_col_k.data() + k + 1, num_rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Copy back to device (only relevant part)
            cudaMemcpy(d_col_k + k + 1, host_col_k.data() + k + 1, num_rows * sizeof(double), cudaMemcpyHostToDevice);
            
            // Launch kernel to update A[i][j] -= L[i][k] * L[j][k]
            
            // We can reuse num_owned calculated earlier as it is the same set of rows
            if (num_owned > 0) {
                // Use 2D grid for better parallelism
                dim3 threadsPerBlock(32, 16); // 512 threads per block
                
                // Grid X covers columns k+1 to n-1. range is n-(k+1) elements.
                int cols = n - (k + 1);
                // Grid Y covers owned rows. range is num_owned.
                dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (num_owned + threadsPerBlock.y - 1) / threadsPerBlock.y);
                            
                if (cols > 0) {
                    updateTrailingMatrixKernel<<<blocks, threadsPerBlock>>>(d_A, d_col_k, n, k, rank, size);
                }
                
                // Check kernel errors
                kerr = cudaGetLastError();
                if (kerr != cudaSuccess) {
                     // printf("Update Kernel launch failed: %s\n", cudaGetErrorString(kerr));
                }
            }
        }
    }
    
    // Copy result back - but only the parts we own are valid!
    // We need to gather the full matrix on rank 0 for validation/output.
    // Or we can just synchronize everything. 
    // Since this is a benchmark, usually we want the result on one node or distributed.
    // The validation expects it on rank 0.
    
    // First, copy local d_A to host A.
    // This gives us valid data for rows i where i % size == rank.
    cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Now we need to communicate to assemble the full matrix on rank 0.
    // Since rows are cyclic, we can't just use a simple Gatherv of blocks.
    // We'll just have every rank send its rows to rank 0.
    // For large N this might be slow, but it's necessary for the current validation logic.
    
    // Optimization: packing the owned rows into a buffer might be faster than many small sends.
    
    std::vector<double> local_rows;
    // Calculate number of rows owned by this rank
    size_t num_owned = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size == rank) num_owned++;
    }
    local_rows.resize(num_owned * n);
    
    // Pack owned rows
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % size == rank) {
            std::memcpy(&local_rows[idx * n], &A[i * n], n * sizeof(double));
            idx++;
        }
    }
    
    // Gather sizes
    std::vector<int> recvcounts(size);
    int my_count = (int)num_owned * n;
    MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> displs(size);
    std::vector<double> all_rows;
    
    if (rank == 0) {
        displs[0] = 0;
        int total_elements = recvcounts[0];
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
            total_elements += recvcounts[i];
        }
        all_rows.resize(total_elements);
    }
    
    // Gather all packed rows to rank 0
    MPI_Gatherv(local_rows.data(), my_count, MPI_DOUBLE, 
                all_rows.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
                
    // Unpack on rank 0
    if (rank == 0) {
        // We need to unpack 'all_rows' back into 'A' in the correct cyclic order.
        // The data in all_rows is ordered by rank: Rank 0's rows, then Rank 1's rows, etc.
        
        std::vector<int> rank_current_pos(size);
        for(int r=0; r<size; ++r) {
            rank_current_pos[r] = displs[r];
        }
        
        for (size_t i = 0; i < n; ++i) {
            int owner = i % size;
            // Copy row i from the owner's section in all_rows
            // Check bounds to be safe
            if (rank_current_pos[owner] + n <= all_rows.size()) {
                std::memcpy(&A[i * n], &all_rows[rank_current_pos[owner]], n * sizeof(double));
                rank_current_pos[owner] += n;
            }
        }
        
        // Zero out upper triangle for correctness of validation
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
            }
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_col_k);
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    
    // Generate random matrix B deterministically
    #pragma omp parallel for
    for (size_t i = 0; i < n * n; ++i) {
        unsigned int local_seed = i;
        B[i] = ((double)(local_seed * 1103515245 + 12345) / (double)2147483648) - 0.5;
    }
    
    // Compute A = B * B^T
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
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    // Use OpenMP
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
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
    
    // Check if error is within tolerance
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
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set CUDA device
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
    } else {
        if (rank == 0) printf("No CUDA devices found!\n");
        MPI_Finalize();
        return 1;
    }

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
        printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
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
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Validation only checks result on rank 0
            bool valid = validateCholesky(A, A_orig, n);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
