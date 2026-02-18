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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize only a block of rows [row_start, row_start + num_rows)
void initMatrixBlock(double* mat, const size_t N, const size_t row_start, const size_t num_rows) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, row_start + i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

    #pragma omp parallel for collapse(2) reduction(&:valid)
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
                valid = false;
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
    // Initialize MPI with thread support for OpenMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU to this rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d, GPUs per process: 1 (device %d), OpenMP threads: %d\n",
               nprocs, gpu_id, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution: each rank gets a contiguous block of rows
    size_t base_rows = N / nprocs;
    size_t remainder = N % nprocs;
    size_t my_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t my_row_start = 0;
    for (int r = 0; r < rank; ++r) {
        my_row_start += base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
    }

    // Prepare sendcounts/displs for MPI_Gatherv
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    {
        size_t offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t rows_r = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            sendcounts[r] = static_cast<int>(rows_r * N);
            displs[r] = static_cast<int>(offset);
            offset += rows_r * N;
        }
    }

    // Each rank initializes its own block of A and the full B
    std::vector<double> local_A(my_rows * N);
    std::vector<double> B(N * N);

    if (rank == 0) printf("Initializing matrices...\n");

    initMatrixBlock(local_A.data(), N, my_row_start, my_rows);
    initMatrix(B, N);

    // Allocate local C
    std::vector<double> local_C(my_rows * N, 0.0);

    // cuBLAS setup
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, my_rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, my_rows * N * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), my_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, my_rows * N * sizeof(double)));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // cuBLAS DGEMM: C = alpha * A * B + beta * C
    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // With row-major data: C(m x n) = A(m x k) * B(k x n)
    // In cuBLAS col-major: call with (n, m, k, B^T, A^T) -> C^T
    const double alpha = 1.0;
    const double beta = 0.0;
    int m = static_cast<int>(N);          // columns of C (and B)
    int n = static_cast<int>(my_rows);    // rows of C (and A)
    int k = static_cast<int>(N);          // shared dimension

    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_B, m,     // B treated as col-major B^T
        d_A, k,     // A treated as col-major A^T
        &beta,
        d_C, m));   // C treated as col-major C^T

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(local_C.data(), d_C, my_rows * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather results to rank 0
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(local_C.data(), static_cast<int>(my_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            // Reconstruct full A for validation
            std::vector<double> A(N * N);
            initMatrix(A, N);
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUBLAS_CHECK(cublasDestroy(handle));
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_B));
                CUDA_CHECK(cudaFree(d_C));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    MPI_Finalize();
    return 0;
}
