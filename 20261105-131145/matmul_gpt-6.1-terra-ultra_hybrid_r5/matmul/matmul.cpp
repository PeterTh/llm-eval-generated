#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Rows are contiguous and balanced, including when there are more ranks than rows.
static size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks ? 1 : 0);
}
static size_t rowStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
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
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || arg[0] == '\0' || *end != '\0' || value == 0 || value > INT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                exitCode = 1;
                break;
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
            exitCode = 1;
            break;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Matrix size is too large\n");
        exitCode = 1;
    }
    if (exitCode) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    const size_t rows = rowCount(N, rank, ranks);
    const size_t first = rowStart(N, rank, ranks);
    try {
        if (rank == 0) {
            printf("Matrix Multiplication Benchmark\n");
            printf("Matrix size: %zu x %zu\n", N, N);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing matrices...\n");
        }

        // A and B have the same initializer. The local A slab is therefore a
        // view into B, both on the host and on the device.
        std::unique_ptr<double[]> B(rows != 0 ? new double[N * N] : nullptr);
        std::vector<double> C(rows * N);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (rows != 0 ? N : 0); ++i) {
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
        }

        double *deviceB = nullptr, *deviceC = nullptr;
        cublasHandle_t handle = nullptr;
        if (rows != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), N * N * sizeof(double)), "cudaMalloc B");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), rows * N * sizeof(double)), "cudaMalloc C");
            checkBlas(cublasCreate(&handle), "cublasCreate");
        }
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (rows != 0) {
            checkCuda(cudaMemcpy(deviceB, B.get(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU");
            const double alpha = 1.0, beta = 0.0;
            // Column-major C^T = B^T * A^T produces the row-major output slab.
            checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                  &alpha, deviceB, static_cast<int>(N),
                                  deviceB + first * N, static_cast<int>(N),
                                  &beta, deviceC, static_cast<int>(N)), "cublasDgemm");
            checkCuda(cudaMemcpy(C.data(), deviceC, rows * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU");
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
            printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        }
        if (rows != 0) {
            checkBlas(cublasDestroy(handle), "cublasDestroy");
            checkCuda(cudaFree(deviceC), "cudaFree C");
            checkCuda(cudaFree(deviceB), "cudaFree B");
        }

        if (printResults) {
            std::vector<double> result;
            if (rank == 0) {
                result.resize(N * N);
                std::copy(C.begin(), C.end(), result.begin());
                for (int source = 1; source < ranks; ++source) {
                    size_t remaining = rowCount(N, source, ranks) * N;
                    size_t offset = rowStart(N, source, ranks) * N;
                    while (remaining != 0) {
                        const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        offset += count;
                        remaining -= count;
                    }
                }
                print_results(result, "MatrixC");
            } else {
                size_t offset = 0;
                while (offset < C.size()) {
                    const int count = static_cast<int>(std::min(C.size() - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += count;
                }
            }
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            int failedRank = ranks;
            unsigned long long position[2] = {0, 0};
            double details[3] = {0.0, 0.0, 0.0};
            for (size_t pi = 0; pi < 5 && failedRank == ranks; ++pi) {
                const size_t i = pi % N;
                if (i < first || i >= first + rows) continue;
                for (size_t pj = 0; pj < 5; ++pj) {
                    const size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k) expected += B[i * N + k] * B[k * N + j];
                    const double actual = C[(i - first) * N + j];
                    const double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (error > 1e-6 || !std::isfinite(actual)) {
                        failedRank = rank;
                        position[0] = i; position[1] = j;
                        details[0] = expected; details[1] = actual; details[2] = error;
                        break;
                    }
                }
            }
            int firstFailure = ranks;
            MPI_Allreduce(&failedRank, &firstFailure, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (firstFailure != ranks) {
                MPI_Bcast(position, 2, MPI_UNSIGNED_LONG_LONG, firstFailure, MPI_COMM_WORLD);
                MPI_Bcast(details, 3, MPI_DOUBLE, firstFailure, MPI_COMM_WORLD);
                if (rank == 0) {
                    printf("Validation failed at (%llu, %llu): expected %.10f, got %.10f (error: %.10e)\n",
                           position[0], position[1], details[0], details[1], details[2]);
                    printf("Validation: FAILED\n");
                }
                exitCode = 1;
            } else if (rank == 0) {
                printf("Validation: PASSED\n");
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return exitCode;
}
