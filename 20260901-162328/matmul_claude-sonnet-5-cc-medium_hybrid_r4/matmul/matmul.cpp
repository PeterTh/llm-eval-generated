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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (call);                                                      \
        if (_err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(_err));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t _st = (call);                                                     \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__,    \
                    static_cast<int>(_st));                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
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

// Initialize a contiguous block of rows [rowStart, rowStart + rowCount) of an N x N matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t rowCount) {
    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < rowCount; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute localC (rowCount x N) = localA (rowCount x N) * B (N x N) on the GPU using cuBLAS.
// Matrices are stored row-major; cuBLAS expects column-major, so we exploit the identity
// C = A*B  <=>  C^T = B^T * A^T, and since a row-major (m x n) matrix has the same memory
// layout as a column-major (n x m) matrix, calling dgemm with A and B swapped and
// dimensions swapped yields the correct row-major result directly.
void matrixMultiplyGPU(const std::vector<double>& localA, const std::vector<double>& B,
                        std::vector<double>& localC, const size_t rowCount, const size_t N,
                        cublasHandle_t handle) {
    if (rowCount == 0) {
        return;
    }

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, rowCount * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, rowCount * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dA, localA.data(), rowCount * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0;
    const double beta = 0.0;

    // Row-major localC(rowCount x N) = localA(rowCount x N) * B(N x N)
    // computed via column-major: C^T(N x rowCount) = B^T(N x N) * A^T(N x rowCount)
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rowCount), static_cast<int>(N),
                              &alpha,
                              dB, static_cast<int>(N),
                              dA, static_cast<int>(N),
                              &beta,
                              dC, static_cast<int>(N)));

    CUDA_CHECK(cudaMemcpy(localC.data(), dC, rowCount * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argError = false;

    // Parse command line arguments (identical on every rank, since argv is the same
    // for every process launched by mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            argError = true;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (argError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Select a GPU for this rank (round-robin across the GPUs visible on its node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA-capable devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", deviceCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute rows of A/C across ranks in contiguous blocks
    const size_t base = N / static_cast<size_t>(worldSize);
    const size_t rem = N % static_cast<size_t>(worldSize);
    const size_t localRowCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t localRowStart = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);

    // Allocate local matrices
    std::vector<double> localA(localRowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRowCount * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    // Matrix values are a pure function of (N, i, j), so each rank can generate its
    // own local rows of A and the full B matrix independently, without any communication
    initMatrixRows(localA, N, localRowStart, localRowCount);
    initMatrix(B, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    matrixMultiplyGPU(localA, B, localC, localRowCount, N, handle);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    const long durationMs = std::lround((t1 - t0) * 1000.0);

    // Gather the distributed row-blocks of C back onto rank 0
    std::vector<double> C;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        C.resize(N * N);
        recvCounts.resize(worldSize);
        displs.resize(worldSize);
        size_t offset = 0;
        for (int r = 0; r < worldSize; ++r) {
            const size_t rCount = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            recvCounts[r] = static_cast<int>(rCount * N);
            displs[r] = static_cast<int>(offset);
            offset += rCount * N;
        }
    }

    MPI_Gatherv(localC.data(), static_cast<int>(localRowCount * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    int validationResult = 0;

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            // Recompute the full A matrix on rank 0 for validation purposes
            std::vector<double> A(N * N);
            initMatrix(A, N);

            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                validationResult = 0;
            } else {
                printf("Validation: FAILED\n");
                validationResult = 1;
            }
        }
        MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    MPI_Finalize();
    return validationResult;
}
