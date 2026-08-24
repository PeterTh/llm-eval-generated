#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrixRows(double* mat, const size_t N, const size_t start_row, const size_t num_rows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

void initMatrix(double* mat, const size_t N) {
    initMatrixRows(mat, N, 0, N);
}

// Simple validation: compute a single element and compare
bool validateResult(const double* A, const double* B,
                   const double* C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

    #pragma omp parallel for collapse(2) shared(valid)
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            if (!valid) continue;
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                #pragma omp critical
                {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                    valid = false;
                }
            }
        }
    }

    return valid;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Assign GPU to this rank (round-robin across available devices)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d, GPUs available: %d, OpenMP threads: %d\n",
               nprocs, numDevices, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution across MPI ranks
    size_t rows_per_rank = N / nprocs;
    size_t remainder = N % nprocs;
    size_t my_start_row = rank * rows_per_rank + std::min(static_cast<size_t>(rank), remainder);
    size_t my_num_rows = rows_per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Each rank initializes its local rows of A and the full B (deterministic, no broadcast needed)
    std::vector<double> A_local(my_num_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(my_num_rows * N, 0.0);

    if (rank == 0) printf("Initializing matrices...\n");

    initMatrixRows(A_local.data(), N, my_start_row, my_num_rows);
    initMatrix(B.data(), N);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (my_num_rows > 0) {
        // Allocate GPU memory
        double *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, my_num_rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, my_num_rows * N * sizeof(double)));

        // Copy matrices to GPU
        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), my_num_rows * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                              cudaMemcpyHostToDevice));

        // cuBLAS DGEMM: row-major C = A * B via col-major trick C^T = B^T * A^T
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const double alpha = 1.0, beta = 0.0;
        const int m = static_cast<int>(N);
        const int n = static_cast<int>(my_num_rows);
        const int k = static_cast<int>(N);
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha, d_B, m, d_A, k,
                                 &beta, d_C, m));

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(C_local.data(), d_C, my_num_rows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    // Gather results on rank 0 using MPI_Gatherv (ranks may have different row counts)
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_rows = rows_per_rank + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t r_start = r * rows_per_rank + std::min(static_cast<size_t>(r), remainder);
        recvcounts[r] = static_cast<int>(r_rows * N);
        displs[r] = static_cast<int>(r_start * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(my_num_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Reconstruct full A on rank 0 for validation
            std::vector<double> A_full(N * N);
            initMatrix(A_full.data(), N);

            bool valid = validateResult(A_full.data(), B.data(), C.data(), N);

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
