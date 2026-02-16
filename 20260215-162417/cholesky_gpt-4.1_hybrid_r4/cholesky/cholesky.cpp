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
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Blocked, hybrid parallel Cholesky using MPI, OpenMP, CUDA
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int block_size = 64; // Tune for your GPU/node
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CUDA setup
    cudaSetDevice(rank % 4); // crude device assignment, adjust as needed
    cublasHandle_t cublasH; cublasCreate(&cublasH);
    cusolverDnHandle_t cusolverH; cusolverDnCreate(&cusolverH);

    // Only lower triangle is used
    for (size_t k = 0; k < n; k += block_size) {
        size_t kb = std::min((size_t)block_size, n - k);
        int owner = (k / block_size) % size;

        // Diagonal block factorization (owner process, GPU)
        if (rank == owner) {
            double* dAkk;
            cudaMalloc((void**)&dAkk, kb * kb * sizeof(double));
            // Copy block to device
            for (size_t i = 0; i < kb; ++i)
                cudaMemcpy(dAkk + i * kb, &A[(k + i) * n + k], kb * sizeof(double), cudaMemcpyHostToDevice);
            int lwork = 0, info = 0;
            cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, kb, dAkk, kb, &lwork);
            double* work; cudaMalloc((void**)&work, lwork * sizeof(double));
            cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, kb, dAkk, kb, work, lwork, &info);
            cudaFree(work);
            // Copy back to host
            for (size_t i = 0; i < kb; ++i)
                cudaMemcpy(&A[(k + i) * n + k], dAkk + i * kb, kb * sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(dAkk);
            if (info != 0) {
                if (rank == 0) printf("Error: Matrix not positive definite at block %zu\n", k);
                cusolverDnDestroy(cusolverH); cublasDestroy(cublasH);
                return false;
            }
        }
        // Broadcast diagonal block to all
        for (size_t i = 0; i < kb; ++i)
            MPI_Bcast(&A[(k + i) * n + k], kb, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Panel update (blocks below diagonal)
        #pragma omp parallel for
        for (size_t i = k + block_size; i < n; i += block_size) {
            size_t ib = std::min(block_size, n - i);
            int owner_i = ((i / block_size) % size);
            if (rank != owner_i) continue;
            double* dAik; cudaMalloc((void**)&dAik, ib * kb * sizeof(double));
            // Copy block to device
            for (size_t ii = 0; ii < ib; ++ii)
                cudaMemcpy(dAik + ii * kb, &A[(i + ii) * n + k], kb * sizeof(double), cudaMemcpyHostToDevice);
            double* dAkk; cudaMalloc((void**)&dAkk, kb * kb * sizeof(double));
            for (size_t ii = 0; ii < kb; ++ii)
                cudaMemcpy(dAkk + ii * kb, &A[(k + ii) * n + k], kb * sizeof(double), cudaMemcpyHostToDevice);
            // Solve dAik = dAik * inv(Lkk^T)
            const double one = 1.0;
cublasDtrsm(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        ib, kb, &one, dAkk, kb, dAik, ib);
            // Copy back
            for (size_t ii = 0; ii < ib; ++ii)
                cudaMemcpy(&A[(i + ii) * n + k], dAik + ii * kb, kb * sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(dAik); cudaFree(dAkk);
        }
        // Allgather panel blocks
        for (size_t i = k + block_size; i < n; i += block_size) {
            size_t ib = std::min(block_size, n - i);
            int owner_i = ((i / block_size) % size);
            for (size_t ii = 0; ii < ib; ++ii)
                MPI_Bcast(&A[(i + ii) * n + k], kb, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);
        }
        // Trailing submatrix update (Schur complement)
        #pragma omp parallel for collapse(2)
        for (size_t i = k + block_size; i < n; i += block_size) {
            for (size_t j = k + block_size; j <= i; j += block_size) {
                size_t ib = std::min(block_size, n - i);
                size_t jb = std::min(block_size, n - j);
                int owner_ij = ((i / block_size) % size);
                if (rank != owner_ij) continue;
                double* dAij; cudaMalloc((void**)&dAij, ib * jb * sizeof(double));
                double* dAik; cudaMalloc((void**)&dAik, ib * kb * sizeof(double));
                double* dAjk; cudaMalloc((void**)&dAjk, jb * kb * sizeof(double));
                // Copy blocks
                for (size_t ii = 0; ii < ib; ++ii)
                    cudaMemcpy(dAik + ii * kb, &A[(i + ii) * n + k], kb * sizeof(double), cudaMemcpyHostToDevice);
                for (size_t jj = 0; jj < jb; ++jj)
                    cudaMemcpy(dAjk + jj * kb, &A[(j + jj) * n + k], kb * sizeof(double), cudaMemcpyHostToDevice);
                for (size_t ii = 0; ii < ib; ++ii)
                    cudaMemcpy(dAij + ii * jb, &A[(i + ii) * n + j], jb * sizeof(double), cudaMemcpyHostToDevice);
                // dAij -= dAik * dAjk^T
                const double minus1 = -1.0, one = 1.0;
                cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, kb, &minus1, dAik, ib, dAjk, jb, &one, dAij, ib);
                // Copy back
                for (size_t ii = 0; ii < ib; ++ii)
                    cudaMemcpy(&A[(i + ii) * n + j], dAij + ii * jb, jb * sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(dAij); cudaFree(dAik); cudaFree(dAjk);
            }
        }
        // Allgather trailing blocks
        for (size_t i = k + block_size; i < n; i += block_size) {
            for (size_t j = k + block_size; j <= i; j += block_size) {
                size_t ib = std::min(block_size, n - i);
                size_t jb = std::min(block_size, n - j);
                int owner_ij = ((i / block_size) % size);
                for (size_t ii = 0; ii < ib; ++ii)
                    MPI_Bcast(&A[(i + ii) * n + j], jb, MPI_DOUBLE, owner_ij, MPI_COMM_WORLD);
            }
        }
    }
    // Zero upper triangle
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;
    cusolverDnDestroy(cusolverH); cublasDestroy(cublasH);
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
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
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }
    // Broadcast matrix to all
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate && rank == 0) {
        A_orig = A; // Save original for validation
    }
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int global_success = 0;
    MPI_Allreduce(&success, &global_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!global_success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
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
                MPI_Finalize();
                return 0;
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
