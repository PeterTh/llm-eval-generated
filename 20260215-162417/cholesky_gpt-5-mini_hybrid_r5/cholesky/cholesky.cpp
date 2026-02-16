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

// Error checking for CUDA calls
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, -1);                                       \
        }                                                                        \
    } while (0)

// CUDA kernel: compute A = B * B^T (naive)
extern "C" __global__ void matMulSymKernel(const double* B, double* A, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += B[row * n + k] * B[col * n + k];
        }
        A[row * n + col] = sum;
    }
}

// Generate A = B * B^T using CUDA on rank 0
void generatePositiveDefiniteMatrixGPU(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double *d_B = nullptr, *d_A = nullptr;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    matMulSymKernel<<<grid, block>>>(d_B, d_A, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    // Add diagonal dominance
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    std::vector<double> reconstructed(n * n);

    // Parallelize reconstruction with OpenMP
    #pragma omp parallel for schedule(static)
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

// Parallel, MPI-aware Cholesky decomposition (row-cyclic ownership)
bool parallelCholesky(std::vector<double>& A, const size_t n, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (size_t j = 0; j < n; ++j) {
        int owner = j % size;
        if (rank == owner) {
            // compute diagonal
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += A[j * n + k] * A[j * n + k];
            double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0) fprintf(stderr, "Error: Matrix not positive definite at %zu\n", j);
                return false;
            }
            A[j * n + j] = sqrt(val);

            // compute column entries for i = j+1..n-1 in parallel with OpenMP
            #pragma omp parallel for schedule(static)
            for (int ii = (int)j + 1; ii < (int)n; ++ii) {
                double s = 0.0;
                for (size_t k = 0; k < j; ++k) s += A[ii * n + k] * A[j * n + k];
                A[ii * n + j] = (A[ii * n + j] - s) / A[j * n + j];
            }
        }

        // Broadcast column j from owner to all ranks (from row j..n-1)
        MPI_Bcast(&A[j * n + j], (int)(n - j), MPI_DOUBLE, owner, comm);

        // After receiving column j, other ranks do nothing extra now; they will use the broadcasted values when they become owners for later columns
    }

    // Zero out upper triangular part on rank 0
    // (All ranks have full matrix, but keep behavior similar to original)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
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
        printf("Cholesky Decomposition Benchmark (MPI=%d ranks, OpenMP threads=%d)\n", size, omp_get_max_threads());
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Generating positive definite matrix on GPU (rank 0)...\n");
        generatePositiveDefiniteMatrixGPU(A, n);
        if (validate) A_orig = A;
    }

    // Broadcast full matrix from rank 0 to all ranks
    MPI_Bcast(A.data(), (int)(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition (hybrid MPI+OpenMP+CUDA)...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    bool success = parallelCholesky(A, n, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double local_ms = (t1 - t0) * 1000.0;
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) fprintf(stderr, "Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time (max across ranks): %.0f ms\n", max_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(A, "CholeskyL");

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
