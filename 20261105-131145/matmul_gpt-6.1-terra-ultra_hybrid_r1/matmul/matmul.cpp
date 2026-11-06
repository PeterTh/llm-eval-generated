#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
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
    printf("Run one MPI process per GPU; set OMP_NUM_THREADS for host initialization.\n");
}

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void blasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Send contiguous buffers in chunks because MPI's traditional count is an int.
static void sendDoubles(const double* data, size_t count, int destination) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static void receiveDoubles(double* data, size_t count, int source) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

static size_t rowsForRank(size_t n, int ranks, int rank) {
    return n / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < n % static_cast<size_t>(ranks));
}

static size_t firstRowForRank(size_t n, int ranks, int rank) {
    return static_cast<size_t>(rank) * (n / static_cast<size_t>(ranks)) +
           std::min(static_cast<size_t>(rank), n % static_cast<size_t>(ranks));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(INT_MAX) ||
                value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
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

    const size_t rows = rowsForRank(N, ranks, rank);
    const size_t firstRow = firstRowForRank(N, ranks, rank);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        fflush(stdout);
    }

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle = nullptr;
    try {
        if (rows) {
            int deviceCount = 0;
            cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
            if (deviceCount == 0) {
                fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
            std::vector<double> A(rows * N), B(N * N);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < N; ++j)
                    A[i * N + j] = getPseudoRndValue(N, firstRow + i, j);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                    B[i * N + j] = getPseudoRndValue(N, i, j);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), rows * N * sizeof(double)), "cudaMalloc A", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), N * N * sizeof(double)), "cudaMalloc B", rank);
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), rows * N * sizeof(double)), "cudaMalloc C", rank);
            cudaCheck(cudaMemcpy(dA, A.data(), rows * N * sizeof(double), cudaMemcpyHostToDevice), "copy A to device", rank);
            cudaCheck(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to device", rank);
            blasCheck(cublasCreate(&handle), "cublasCreate", rank);
            // Initialize cuBLAS's lazy runtime state before timing the multiplication.
            const double alpha = 1.0, beta = 0.0;
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                                 &alpha, dB, 1, dA, 1, &beta, dC, 1),
                      "cuBLAS warm-up", rank);
            cudaCheck(cudaDeviceSynchronize(), "cuBLAS warm-up", rank);
        }

        if (rank == 0) {
            printf("Computing matrix multiplication...\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (rows) {
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A * B is column-major C^T = B^T * A^T.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                                 &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                                 &beta, dC, static_cast<int>(N)), "cublasDgemm", rank);
            cudaCheck(cudaDeviceSynchronize(), "matrix multiplication", rank);
        }
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
            printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
        }

        const size_t checkedRows = firstRow < 5 ? std::min(rows, size_t{5} - firstRow) : 0;
        const size_t copyRows = printResults ? rows : (validate ? checkedRows : 0);
        std::vector<double> localC(copyRows * N);
        if (copyRows)
            cudaCheck(cudaMemcpy(localC.data(), dC, copyRows * N * sizeof(double), cudaMemcpyDeviceToHost), "copy C to host", rank);

        if (printResults) {
            if (rank == 0) {
                std::vector<double> C(N * N);
                std::copy(localC.begin(), localC.end(), C.begin());
                for (int source = 1; source < ranks; ++source) {
                    const size_t sourceRows = rowsForRank(N, ranks, source);
                    if (sourceRows)
                        receiveDoubles(C.data() + firstRowForRank(N, ranks, source) * N, sourceRows * N, source);
                }
                print_results(C, "MatrixC");
            } else if (rows) {
                sendDoubles(localC.data(), rows * N, 0);
            }
        }

        int valid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            for (size_t localI = 0; localI < checkedRows; ++localI) {
                const size_t i = firstRow + localI;
                for (size_t j = 0; j < std::min(N, size_t{5}); ++j) {
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double actual = localC[localI * N + j];
                    const double relError = std::abs((actual - expected) / (expected + 1e-10));
                    if (!(relError <= 1e-6)) {
                        fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, relError);
                        valid = 0;
                        break;
                    }
                }
                if (!valid) break;
            }
            int allValid = 0;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            valid = allValid;
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (handle) blasCheck(cublasDestroy(handle), "cublasDestroy", rank);
        if (dA) cudaCheck(cudaFree(dA), "cudaFree A", rank);
        if (dB) cudaCheck(cudaFree(dB), "cudaFree B", rank);
        if (dC) cudaCheck(cudaFree(dC), "cudaFree C", rank);
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
