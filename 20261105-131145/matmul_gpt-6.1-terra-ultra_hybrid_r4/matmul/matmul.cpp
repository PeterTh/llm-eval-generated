#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Keep the original size_t arithmetic, including its unsigned overflow behavior.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void cublasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static size_t rowCount(size_t N, int rank, int ranks) {
    return N / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < N % static_cast<size_t>(ranks));
}

static size_t rowStart(size_t N, int rank, int ranks) {
    const size_t r = static_cast<size_t>(rank);
    return r * (N / static_cast<size_t>(ranks)) + std::min(r, N % static_cast<size_t>(ranks));
}

// MPI counts are int, while a matrix can contain more than INT_MAX elements.
static void transferRows(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || value[0] == '\0' || *end != '\0' || errno == ERANGE ||
                parsed > static_cast<unsigned long long>(INT_MAX) ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double))) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                parseStatus = 1;
                break;
            }
            N = static_cast<size_t>(parsed);
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
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus;
    }

    const size_t rows = rowCount(N, rank, ranks);
    const size_t firstRow = rowStart(N, rank, ranks);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    try {
        std::vector<double> A(rows * N), B(rows != 0 ? N * N : 0),
                            C((validate || printResults) ? rows * N : 0);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (rows != 0 ? N : 0); ++i) {
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
        }
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
        }

        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0, deviceCount = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
        if (deviceCount == 0) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

        double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows != 0) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceA), A.size() * sizeof(double)), "cudaMalloc A", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceB), B.size() * sizeof(double)), "cudaMalloc B", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceC), rows * N * sizeof(double)), "cudaMalloc C", rank);
            cublasCheck(cublasCreate(&handle), "cublasCreate", rank);
        }

        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (rows != 0) {
            cudaCheck(cudaMemcpy(deviceA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU", rank);
            cudaCheck(cudaMemcpy(deviceB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU", rank);
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A * B is column-major C^T = B^T * A^T.
            cublasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                    &alpha, deviceB, static_cast<int>(N), deviceA, static_cast<int>(N),
                                    &beta, deviceC, static_cast<int>(N)), "cublasDgemm", rank);
            if (validate || printResults) {
                cudaCheck(cudaMemcpy(C.data(), deviceC, C.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU", rank);
            } else {
                cudaCheck(cudaDeviceSynchronize(), "synchronize GPU", rank);
            }
        }
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
        }

        if (printResults) {
            if (rank == 0) {
                std::vector<double> result(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int source = 1; source < ranks; ++source) {
                    const size_t count = rowCount(N, source, ranks) * N;
                    if (count != 0) transferRows(result.data() + rowStart(N, source, ranks) * N, count, source, false);
                }
                print_results(result, "MatrixC");
            } else if (!C.empty()) {
                transferRows(C.data(), C.size(), 0, true);
            }
        }

        int valid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            for (size_t pi = 0; pi < 5 && valid && N != 0; ++pi) {
                const size_t i = pi % N;
                if (i < firstRow || i >= firstRow + rows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += A[(i - firstRow) * N + k] * B[k * N + j];
                    const double actual = C[(i - firstRow) * N + j];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(relError) || relError > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                        valid = 0;
                        break;
                    }
                }
            }
            int allValid = 0;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = allValid;
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (rows != 0) {
            cublasCheck(cublasDestroy(handle), "cublasDestroy", rank);
            cudaCheck(cudaFree(deviceA), "cudaFree A", rank);
            cudaCheck(cudaFree(deviceB), "cudaFree B", rank);
            cudaCheck(cudaFree(deviceC), "cudaFree C", rank);
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: host allocation or computation failed: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
