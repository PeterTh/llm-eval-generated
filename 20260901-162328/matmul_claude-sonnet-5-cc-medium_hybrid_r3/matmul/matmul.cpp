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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                            \
    do {                                                                              \
        cublasStatus_t st__ = (call);                                                 \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, \
                    static_cast<int>(st__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                              \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a block of rows [rowOffset, rowOffset + numRows) of an N-wide matrix.
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t rowOffset,
                     const size_t numRows) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t li = 0; li < numRows; ++li) {
        for (size_t j = 0; j < N; ++j) {
            const size_t i = rowOffset + li;
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixBlock(mat, N, 0, N);
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

// Compute the row range [rowStart, rowEnd) of C assigned to a given MPI rank,
// distributing the N rows as evenly as possible across `size` ranks.
void computeRowRange(const size_t N, const int rank, const int size,
                     size_t& rowStart, size_t& rowCount) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    rowStart = r * base + std::min(r, rem);
    rowCount = base + (r < rem ? 1 : 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the local rank on this node so multiple ranks sharing a node
    // spread across distinct GPUs.
    int localRank = 0;
    if (const char* s = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
        localRank = std::atoi(s);
    } else if (const char* s2 = std::getenv("MV2_COMM_WORLD_LOCAL_RANK")) {
        localRank = std::atoi(s2);
    } else if (const char* s3 = std::getenv("SLURM_LOCALID")) {
        localRank = std::atoi(s3);
    } else {
        localRank = rank;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool shouldExit = false;
    int exitCode = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = true;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = true;
                exitCode = 1;
                break;
            }
        }
    }

    // Broadcast parsed configuration to all ranks
    {
        unsigned long long nBcast = N;
        int flags[4] = {validate ? 1 : 0, printResults ? 1 : 0, shouldExit ? 1 : 0, exitCode};
        MPI_Bcast(&nBcast, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(flags, 4, MPI_INT, 0, MPI_COMM_WORLD);
        N = static_cast<size_t>(nBcast);
        validate = flags[0] != 0;
        printResults = flags[1] != 0;
        shouldExit = flags[2] != 0;
        exitCode = flags[3];
    }

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d\n", size, deviceCount);
    }

    // Determine this rank's row block of C (and of A) to compute.
    size_t rowStart = 0, rowCount = 0;
    computeRowRange(N, rank, size, rowStart, rowCount);

    // Every rank needs the full B matrix, and only its own row block of A.
    // Both are pure functions of (N, i, j), so each rank regenerates them
    // locally in parallel with OpenMP instead of communicating them.
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    std::vector<double> A_local(rowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(rowCount * N);

    initMatrixBlock(A_local, N, rowStart, rowCount);
    initMatrix(B, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    if (rowCount > 0) {
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, rowCount * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rowCount * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dA, A_local.data(), rowCount * N * sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double),
                               cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const double alpha = 1.0;
        const double beta = 0.0;
        const int m = static_cast<int>(rowCount);
        const int n = static_cast<int>(N);
        const int k = static_cast<int>(N);

        // A_local, B and C_local are stored row-major. cuBLAS is column-major,
        // so we compute C^T = B^T * A^T (col-major), which is exactly the
        // row-major C = A * B in-place, without any explicit transposition.
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n,
                                  dA, k, &beta, dC, n));

        CUDA_CHECK(cudaMemcpy(C_local.data(), dC, rowCount * N * sizeof(double),
                               cudaMemcpyDeviceToHost));

        cublasDestroy(handle);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double endTime = MPI_Wtime();
    const double elapsed = endTime - startTime;
    double maxElapsed = elapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the row blocks of C back onto rank 0.
    std::vector<double> C;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        C.resize(N * N);
        recvCounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            size_t rs = 0, rc = 0;
            computeRowRange(N, r, size, rs, rc);
            recvCounts[r] = static_cast<int>(rc * N);
            displs[r] = static_cast<int>(rs * N);
        }
    }
    MPI_Gatherv(C_local.data(), static_cast<int>(rowCount * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Regenerate the full A matrix locally (cheap O(N^2) work) since
            // only rank 0's row block was materialized above.
            std::vector<double> A(N * N);
            initMatrix(A, N);
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
