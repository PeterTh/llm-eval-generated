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

// Hybrid MPI + OpenMP + CUDA accelerated Cholesky benchmark
// Semantics preserved: performs full Cholesky on each MPI rank and validates on rank 0 if requested

#define CUDA_CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while(0)

// Simple CUDA kernel to compute A = B * B^T (Gram matrix)
extern "C" __global__ void gramKernel(const double* B, double* A, int n) {
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

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            // parallelize inner reduction over k
            #pragma omp parallel for reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            if (i == j) {
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    if (MPI::COMM_WORLD.Get_rank() == 0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    }
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
        // Zero out upper triangular part for this row
        #pragma omp parallel for
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    return true;
}

// Generate a symmetric positive definite matrix using GPU-accelerated Gram product
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int mpi_rank) {
    std::vector<double> B(n * n);
    unsigned int seed = 42 + mpi_rank;

    // Generate random matrix B (threaded)
    #pragma omp parallel
    {
        unsigned int local_seed = seed ^ (unsigned int)omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&local_seed) / (double)RAND_MAX) - 0.5;
        }
    }

    // Use CUDA to compute A = B * B^T
    double* d_B = nullptr;
    double* d_A = nullptr;
    size_t bytes = n * n * sizeof(double);

    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    // Launch 2D kernel
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gramKernel<<<grid, block>>>(d_B, d_A, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (parallelized)
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

    #pragma omp parallel for reduction(max:maxError,relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        if (error > maxError) maxError = error;
        if (rel > relError) relError = rel;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 for printing help)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI ranks: %d)\n", mpi_size);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // If CUDA is available, pick device based on MPI rank
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        int dev = mpi_rank % device_count;
        CUDA_CHECK(cudaSetDevice(dev));
    }

    // Configure OpenMP threads (let environment override)
    if (mpi_rank == 0) {
        int threads = omp_get_max_threads();
        printf("OpenMP threads: %d\n", threads);
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (mpi_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, mpi_rank);

    if (validate && mpi_rank == 0) {
        A_orig = A; // Save original on rank 0 for validation
    }

    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (mpi_rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation only on rank 0
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
