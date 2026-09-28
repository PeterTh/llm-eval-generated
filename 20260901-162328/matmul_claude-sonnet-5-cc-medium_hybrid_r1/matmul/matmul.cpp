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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err_));                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                                \
    do {                                                                                   \
        cublasStatus_t st_ = (call);                                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__,      \
                    static_cast<int>(st_));                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                   \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Distributes rows of A (and thus C) across MPI ranks. Each rank picks a GPU
// (round-robin over the GPUs visible to its node, via a shared-memory
// communicator) and computes its row block with cuBLAS DGEMM. Results are
// gathered into the full C matrix on rank 0.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const int rank, const int numProcs) {
    // Determine this rank's GPU via a node-local (shared-memory) communicator,
    // so device assignment does not depend on MPI-implementation-specific
    // environment variables.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    const int device = deviceCount > 0 ? (localRank % deviceCount) : 0;
    CUDA_CHECK(cudaSetDevice(device));

    // Block row distribution across ranks (remainder spread over the first ranks).
    const size_t baseRows = N / static_cast<size_t>(numProcs);
    const size_t remainder = N % static_cast<size_t>(numProcs);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t startRow = static_cast<size_t>(rank) * baseRows +
                       std::min(static_cast<size_t>(rank), remainder);

    std::vector<double> localC(localRows * N);

    if (localRows > 0) {
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, localRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, localRows * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dA, A.data() + startRow * N, localRows * N * sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const double alpha = 1.0;
        const double beta = 0.0;
        const int m = static_cast<int>(localRows); // rows of A_local / C_local
        const int n = static_cast<int>(N);          // cols of B / C_local
        const int k = static_cast<int>(N);          // cols of A_local / rows of B

        // A_local, B and C_local are stored row-major. cuBLAS expects
        // column-major, so we compute C_local^T = B^T * A_local^T, which in
        // column-major terms is exactly the row-major C_local = A_local * B.
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k,
                                  &alpha,
                                  dB, n,
                                  dA, k,
                                  &beta,
                                  dC, n));

        CUDA_CHECK(cudaMemcpy(localC.data(), dC, localRows * N * sizeof(double),
                               cudaMemcpyDeviceToHost));

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    // Gather all row blocks into the full C matrix on rank 0.
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        recvCounts.resize(numProcs);
        displs.resize(numProcs);
        for (int r = 0; r < numProcs; ++r) {
            const size_t rows_r = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t start_r = static_cast<size_t>(r) * baseRows +
                                    std::min(static_cast<size_t>(r), remainder);
            recvCounts[r] = static_cast<int>(rows_r * N);
            displs[r] = static_cast<int>(start_r * N);
        }
    }

    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
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

    int rank = 0, numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int action = 0; // 0 = continue, 1 = exit success, 2 = exit failure

    // Parse command line arguments (rank 0 only, then broadcast the outcome).
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                action = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                action = 2;
                break;
            }
        }
    }

    struct { size_t N; int validate; int printResults; int action; } args{N, validate, printResults, action};
    MPI_Bcast(&args, sizeof(args), MPI_BYTE, 0, MPI_COMM_WORLD);
    N = args.N;
    validate = args.validate;
    printResults = args.printResults;
    action = args.action;

    if (action != 0) {
        MPI_Finalize();
        return action == 1 ? 0 : 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("Parallelization: MPI ranks=%d, OpenMP threads/rank=%d, GPUs visible on rank 0's node=%d\n",
               numProcs, omp_get_max_threads(), deviceCount);
    }

    // Allocate matrices. Every rank initializes its own full copies of A and B
    // (the initialization function is deterministic, so this avoids a broadcast
    // and lets OpenMP parallelize the fill on every rank). Only rank 0 needs the
    // full result matrix C; other ranks only ever hold their own row block.
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, rank, numProcs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDurationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

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
