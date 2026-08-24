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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a contiguous block of rows [startRow, startRow+numRows) of an NxN matrix
void initMatrixRows(double* mat, const size_t startRow, const size_t numRows, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numRows; ++i) {
        const size_t globalRow = startRow + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// Validate result by recomputing checkpoint elements from the deterministic formula
bool validateResult(const size_t N, const std::vector<double>& C) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st_ = (call); \
    if (st_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(st_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResultsFlag = false;
    int showHelp = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResultsFlag = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
                printUsage(argv[0]);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                showHelp = 1;
            }
        }
    }

    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (showHelp) {
        MPI_Finalize();
        return 0;
    }

    // Broadcast parameters to all ranks
    unsigned long long N_ull = static_cast<unsigned long long>(N);
    MPI_Bcast(&N_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N_ull);

    int val_int = validate ? 1 : 0;
    int pr_int = printResultsFlag ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pr_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val_int != 0);
    printResultsFlag = (pr_int != 0);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute 1D row distribution across MPI ranks
    const size_t baseRows = N / static_cast<size_t>(nprocs);
    const size_t extraRows = N % static_cast<size_t>(nprocs);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t startRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);

    // Initialize local portion of A (OpenMP-parallelized)
    std::vector<double> A_local(localRows * N);
    if (localRows > 0 && N > 0) {
        initMatrixRows(A_local.data(), startRow, localRows, N);
    }

    // Initialize full B on every rank (OpenMP-parallelized)
    std::vector<double> B(N * N);
    if (N > 0) {
        initMatrixRows(B.data(), 0, N, N);
    }

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    // CUDA setup: allocate device memory and copy data
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    cublasHandle_t cublasH = nullptr;

    if (localRows > 0 && N > 0) {
        CUDA_CHECK(cudaMalloc(&d_A, localRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, localRows * N * sizeof(double)));
        CUBLAS_CHECK(cublasCreate(&cublasH));

        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), localRows * N * sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double),
                               cudaMemcpyHostToDevice));
    }

    // Free host copies of A and B that are no longer needed
    A_local.clear();
    A_local.shrink_to_fit();
    B.clear();
    B.shrink_to_fit();

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // cuBLAS DGEMM: C_local = A_local * B
    // Row-major trick: interpret row-major M(r×c) as column-major M^T(c×r).
    // C_rm = A_rm * B_rm  =>  C_rm^T = B_rm^T * A_rm^T  (column-major)
    if (localRows > 0 && N > 0) {
        const double alpha = 1.0;
        const double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(N),          // m
            static_cast<int>(localRows),  // n
            static_cast<int>(N),          // k
            &alpha,
            d_B, static_cast<int>(N),     // A_cublas = B^T (col-major N×N)
            d_A, static_cast<int>(N),     // B_cublas = A^T (col-major N×localRows)
            &beta,
            d_C, static_cast<int>(N)      // C_cublas = C^T (col-major N×localRows)
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result from GPU to host
    std::vector<double> C_local(localRows * N);
    if (localRows > 0 && N > 0) {
        CUDA_CHECK(cudaMemcpy(C_local.data(), d_C, localRows * N * sizeof(double),
                               cudaMemcpyDeviceToHost));
    }

    // Gather all C_local rows onto rank 0
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    {
        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
            recvcounts[r] = static_cast<int>(rows * N);
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Synchronize all ranks after timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_dur_ms = static_cast<long long>(duration.count());
    long long dur_ms = 0;
    MPI_Reduce(&local_dur_ms, &dur_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Cleanup CUDA resources
    if (cublasH) {
        cublasDestroy(cublasH);
    }
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    C_local.clear();
    C_local.shrink_to_fit();

    // Output results on rank 0
    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", dur_ms);

        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                         static_cast<double>(N)) /
                        (static_cast<double>(dur_ms) / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResultsFlag) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(N, C);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Broadcast exit code so all ranks return consistently
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
