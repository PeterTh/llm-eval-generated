#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow) {
    const size_t rows = mat.size() / N;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void blasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Contiguous blocks are balanced even when there are more ranks than rows.
size_t rowCount(size_t N, int ranks, int rank) {
    return N / ranks + (static_cast<size_t>(rank) < N % ranks ? 1 : 0);
}
size_t rowOffset(size_t N, int ranks, int rank) {
    return (N / ranks) * rank + std::min(static_cast<size_t>(rank), N % ranks);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || end == arg || *end || arg[0] == '-' || value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / value ||
                value * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        fflush(stdout);
    }

    const size_t rows = rowCount(N, ranks, rank);
    const size_t firstRow = rowOffset(N, ranks, rank);
    try {
        std::vector<double> A(rows * N);
        std::vector<double> B(rows ? N * N : 0);
        std::vector<double> C((printResults || validate) ? rows * N : 0);
        if (rows) {
            initMatrix(A, N, firstRow);
            initMatrix(B, N, 0);
        }

        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows) {
            int devices = 0;
            cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
            if (!devices) {
                fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), A.size() * sizeof(double)), "cudaMalloc A", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(double)), "cudaMalloc B", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), A.size() * sizeof(double)), "cudaMalloc C", rank);
            cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU", rank);
            cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU", rank);
            blasCheck(cublasCreate(&handle), "cublasCreate", rank);
            // Load and initialize the cuBLAS kernel before timing the multiply.
            const double alpha = 1.0, beta = 0.0;
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                  &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                  &beta, dC, static_cast<int>(N)), "warm up cublasDgemm", rank);
            cudaCheck(cudaDeviceSynchronize(), "warm up matrix multiplication", rank);
        }

        if (rank == 0) {
            printf("Computing matrix multiplication...\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (rows) {
            // Row-major C = A * B is column-major C^T = B^T * A^T.
            const double alpha = 1.0, beta = 0.0;
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                  &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                  &beta, dC, static_cast<int>(N)), "cublasDgemm", rank);
            cudaCheck(cudaDeviceSynchronize(), "matrix multiplication", rank);
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }

        if (rows && (printResults || validate)) {
            cudaCheck(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU", rank);
        }

        if (printResults) {
            // Chunk transfers to avoid MPI's int count limit for large matrices.
            if (rank == 0) {
                std::vector<double> result(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int source = 1; source < ranks; ++source) {
                    const size_t count = rowCount(N, ranks, source) * N;
                    const size_t offset = rowOffset(N, ranks, source) * N;
                    for (size_t pos = 0; pos < count;) {
                        const int chunk = static_cast<int>(std::min(count - pos, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + offset + pos, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        pos += chunk;
                    }
                }
                print_results(result, "MatrixC");
            } else {
                for (size_t pos = 0; pos < C.size();) {
                    const int chunk = static_cast<int>(std::min(C.size() - pos, static_cast<size_t>(INT_MAX)));
                    MPI_Send(C.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    pos += chunk;
                }
            }
        }

        int valid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            for (size_t pi = 0; pi < 5; ++pi) {
                const size_t i = pi % N;
                if (i < firstRow || i >= firstRow + rows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) {
                        expected += A[(i - firstRow) * N + k] * B[k * N + j];
                    }
                    const double actual = C[(i - firstRow) * N + j];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(actual) || relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, relError);
                        valid = 0;
                    }
                }
            }
            int allValid = 0;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = allValid;
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (rows) {
            blasCheck(cublasDestroy(handle), "cublasDestroy", rank);
            cudaCheck(cudaFree(dA), "cudaFree A", rank);
            cudaCheck(cudaFree(dB), "cudaFree B", rank);
            cudaCheck(cudaFree(dC), "cudaFree C", rank);
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "Rank %d: host matrix allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
