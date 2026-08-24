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

#define CUDA_CHECK(call)     do {         cudaError_t err = call;         if (err != cudaSuccess) {             fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));             MPI_Abort(MPI_COMM_WORLD, 1);         }     } while (0)

#define CUBLAS_CHECK(call)     do {         cublasStatus_t status = call;         if (status != CUBLAS_STATUS_SUCCESS) {             fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__);             MPI_Abort(MPI_COMM_WORLD, 1);         }     } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    if (N % size != 0) {
        if (rank == 0) printf("Error: N must be divisible by MPI size\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t rows_per_rank = N / size;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI Ranks: %d\n", size);
        printf("Rows per rank: %zu\n", rows_per_rank);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate host memory
    std::vector<double> A_local(rows_per_rank * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(rows_per_rank * N);
    
    // Initialize matrices
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    
    // Parallelize initialization with OpenMP
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows_per_rank; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t global_i = rank * rows_per_rank + i;
            A_local[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }

    initMatrix(B, N);

    MPI_Barrier(MPI_COMM_WORLD);

    // Create cublas handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, rows_per_rank * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, rows_per_rank * N * sizeof(double)));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), rows_per_rank * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));


    // Perform matrix multiplication: C = A * B
    // Since cuBLAS uses column-major storage, and we have row-major data,
    // we compute C^T = B^T * A^T.
    // Ideally, we interpret A as A^T (col-major), B as B^T (col-major), C as C^T (col-major).
    // So we compute C^T = B^T * A^T.
    // In cuBLAS parameters:
    // C (m x n) = alpha * op(A) * op(B) + beta * C
    // Here we want C_local (rows_per_rank x N)
    // In row-major, C_local is (rows_per_rank x N).
    // In col-major interpretation, it is (N x rows_per_rank) if we view it as C^T.
    // So we want result to be (N x rows_per_rank) in col-major.
    // So m = N, n = rows_per_rank, k = N.
    // We compute C^T = B^T * A^T.
    // B^T is (N x N) in col-major (from B row-major).
    // A^T is (N x rows_per_rank) in col-major (from A_local row-major).
    // So we multiply B^T (N x N) * A^T (N x rows_per_rank).
    // Wait. B is N x N. A_local is rows_per_rank x N.
    // A_local^T is N x rows_per_rank.
    // B^T * A^T -> (N x N) * (N x rows_per_rank) = (N x rows_per_rank). Correct.
    // So we pass B as first matrix, A as second matrix.
    // leading dimension of B^T is N.
    // leading dimension of A^T is N.
    // leading dimension of C^T is N.

    double alpha = 1.0;
    double beta = 0.0;
    
    // cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // We compute C^T = B^T * A^T
    // A -> d_B (which is B^T in col-major)
    // B -> d_A (which is A_local^T in col-major)
    // C -> d_C (which is C_local^T in col-major)
    // m = N (rows of B^T)
    // n = rows_per_rank (cols of A^T)
    // k = N (cols of B^T / rows of A^T)
    
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, rows_per_rank, N,
                             &alpha,
                             d_B, N,
                             d_A, N,
                             &beta,
                             d_C, N));

    // Sync device
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(C_local.data(), d_C, rows_per_rank * N * sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double gflops = (2.0 * N * N * N) / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> C_full;
    if (rank == 0 && (validate || printResults)) {
        C_full.resize(N * N);
    }

    if (validate || printResults) {
        MPI_Gather(C_local.data(), rows_per_rank * N, MPI_DOUBLE,
                   C_full.data(), rows_per_rank * N, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (printResults) {
            print_results(C_full, "MatrixC");
        }
        
        if (validate) {
            std::vector<double> A_full(N * N);
            initMatrix(A_full, N);
            
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B, C_full, N);
            
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
