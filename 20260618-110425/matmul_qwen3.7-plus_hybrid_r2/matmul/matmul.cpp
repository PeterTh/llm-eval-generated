#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

    // All ranks parse the same command line (mpirun distributes identical argv)
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; i++) {
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
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local rank for GPU device assignment
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    // Select GPU device (round-robin if more ranks than GPUs)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));

    // 1D block distribution: distribute rows of A across MPI ranks
    size_t base_rows = N / size;
    size_t rem = N % size;
    size_t local_rows = base_rows + ((size_t)rank < rem ? 1 : 0);
    size_t start_row = (size_t)rank * base_rows + ((size_t)rank < rem ? (size_t)rank : rem);

    if (rank == 0) printf("Initializing matrices...\n");

    // Initialize local portion of A with OpenMP parallelism
    std::vector<double> local_A(local_rows * N);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < local_rows; i++) {
        for (size_t j = 0; j < N; j++) {
            local_A[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }

    // Initialize full B matrix with OpenMP (replicated on each rank)
    std::vector<double> B(N * N);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }

    // Create cuBLAS handle on the selected GPU
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate GPU memory and upload matrices
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (local_rows > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, local_rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, local_rows * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Release host A (no longer needed after upload)
    { std::vector<double>().swap(local_A); }

    // Warmup cuBLAS to trigger algorithm selection and workspace allocation
    if (local_rows > 0) {
        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  (int)N, (int)local_rows, (int)N,
                                  &alpha, d_B, (int)N, d_A, (int)N,
                                  &beta, d_C, (int)N));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");

    // Synchronize all ranks, then time the core GPU computation
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    if (local_rows > 0) {
        // Row-major C = A * B translated for cuBLAS column-major:
        // C^T (N x local_rows) = B^T (N x N) * A^T (N x local_rows)
        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  (int)N, (int)local_rows, (int)N,
                                  &alpha, d_B, (int)N, d_A, (int)N,
                                  &beta, d_C, (int)N));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double t_local = MPI_Wtime() - t_start;
    double t_max;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Download results from GPU
    std::vector<double> local_C(local_rows * N);
    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(local_C.data(), d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Free GPU resources
    if (d_A) CUDA_CHECK(cudaFree(d_A));
    if (d_B) CUDA_CHECK(cudaFree(d_B));
    if (d_C) CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));

    // Release host B (no longer needed)
    { std::vector<double>().swap(B); }

    // Gather all result rows to rank 0
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; r++) {
        size_t r_rows = base_rows + ((size_t)r < rem ? 1 : 0);
        size_t r_start = (size_t)r * base_rows + ((size_t)r < rem ? (size_t)r : rem);
        recvcounts[r] = (int)(r_rows * N);
        displs[r] = (int)(r_start * N);
    }

    MPI_Gatherv(local_C.data(), (int)(local_rows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    { std::vector<double>().swap(local_C); }

    // Rank 0 outputs results
    if (rank == 0) {
        long duration_ms = (long)(t_max * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        double gflops = (2.0 * N * N * N) / t_max / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

            for (size_t pi = 0; pi < 5 && valid; pi++) {
                for (size_t pj = 0; pj < 5 && valid; pj++) {
                    const size_t ci = checkPoints[pi] % N;
                    const size_t cj = checkPoints[pj] % N;

                    double expected = 0.0;
                    for (size_t k = 0; k < N; k++) {
                        expected += getPseudoRndValue(N, ci, k) * getPseudoRndValue(N, k, cj);
                    }

                    const double actual = C[ci * N + cj];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));

                    if (relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               ci, cj, expected, actual, relError);
                        valid = false;
                    }
                }
            }

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
