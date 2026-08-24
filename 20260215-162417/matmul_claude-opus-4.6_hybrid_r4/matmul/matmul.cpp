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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                (int)stat); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixRows(double* mat, const size_t N, const size_t startRow, const size_t numRows) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, startRow + i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU round-robin across MPI ranks
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    CUDA_CHECK(cudaSetDevice(rank % num_gpus));

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int ret = 0;

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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d, GPUs: %d, OpenMP threads/rank: %d\n",
               nprocs, num_gpus, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute rows of A across MPI ranks
    size_t base_rows = N / static_cast<size_t>(nprocs);
    size_t remainder = N % static_cast<size_t>(nprocs);
    size_t my_start = static_cast<size_t>(rank) * base_rows +
                      (static_cast<size_t>(rank) < remainder ? static_cast<size_t>(rank) : remainder);
    size_t my_rows  = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    if (rank == 0) printf("Initializing matrices...\n");

    // Allocate pinned host memory for faster GPU transfers
    double *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    if (my_rows > 0) {
        CUDA_CHECK(cudaMallocHost(&h_A, my_rows * N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_C, my_rows * N * sizeof(double)));
    }
    CUDA_CHECK(cudaMallocHost(&h_B, N * N * sizeof(double)));

    // Initialize with OpenMP — deterministic, no MPI communication needed
    if (my_rows > 0) initMatrixRows(h_A, N, my_start, my_rows);
    initMatrixRows(h_B, N, 0, N);

    // cuBLAS setup
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (my_rows > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, my_rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, my_rows * N * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));

    // Transfer to GPU
    if (my_rows > 0) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, my_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // cuBLAS dgemm: row-major C = A * B  ↔  col-major C^T = B^T * A^T
    if (my_rows > 0) {
        const double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(N),
                                 static_cast<int>(my_rows),
                                 static_cast<int>(N),
                                 &alpha,
                                 d_B, static_cast<int>(N),
                                 d_A, static_cast<int>(N),
                                 &beta,
                                 d_C, static_cast<int>(N)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Copy result back to host
    if (my_rows > 0) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, my_rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather all rows of C on rank 0
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(r_rows * N);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + recvcounts[r - 1];
    }

    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(h_C, static_cast<int>(my_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        double gflops = (2.0 * N * N * N) / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A_full(N * N), B_full(N * N);
            initMatrix(A_full, N);
            initMatrix(B_full, N);
            bool valid = validateResult(A_full, B_full, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    // Cleanup
    if (d_A) CUDA_CHECK(cudaFree(d_A));
    if (d_C) CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUBLAS_CHECK(cublasDestroy(handle));
    if (h_A) CUDA_CHECK(cudaFreeHost(h_A));
    if (h_C) CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_B));

    MPI_Finalize();
    return ret;
}
