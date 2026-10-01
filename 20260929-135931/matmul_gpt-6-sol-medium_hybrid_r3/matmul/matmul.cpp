#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Keep the original initialization formula and its row-major layout.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

[[noreturn]] void fail(int rank, const char* operation, int code) {
    std::fprintf(stderr, "Rank %d: %s failed (code %d)\n", rank, operation, code);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t result, int rank, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                     cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t result, int rank, const char* operation) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fail(rank, operation, static_cast<int>(result));
    }
}

void checkMpi(int result, int rank, const char* operation) {
    if (result != MPI_SUCCESS) {
        fail(rank, operation, result);
    }
}

// A is generated only for the rows owned by this MPI rank.
void initMatrix(double* mat, size_t N, size_t firstRow, size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < rows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        for (size_t j = 0; j < N; ++j) {
            mat[localRow * N + j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

// Preserve the original sample positions, including repeated positions for N < 5.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t firstRow,
                    size_t rows, int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        if (i < firstRow || i >= firstRow + rows) continue;
        const size_t localRow = i - firstRow;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[localRow * N + k] * B[k * N + j];
            }
            const double actual = C[localRow * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                std::printf("Rank %d: Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            rank, i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Avoid MPI's int count limit when collecting an optional full result.
void transferRows(double* data, size_t count, int peer, bool send,
                  int rank) {
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min(count - offset,
                                                   static_cast<size_t>(INT_MAX)));
        if (send) {
            checkMpi(MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 0,
                              MPI_COMM_WORLD), rank, "MPI_Send");
        } else {
            checkMpi(MPI_Recv(data + offset, chunk,
                              MPI_DOUBLE, peer, 0, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE), rank, "MPI_Recv");
        }
        offset += static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) fail(rank, "MPI thread support", provided);

    size_t N = 512;
    bool validate = false, printResults = false, help = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || value[0] == '-' ||
                parsed == 0 || parsed > static_cast<unsigned long long>(INT_MAX) ||
                parsed > std::numeric_limits<size_t>::max() / parsed) {
                argumentsValid = false;
            } else {
                N = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            argumentsValid = false;
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
        }
    }
    if (!argumentsValid || help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // Split by node so each rank chooses a GPU from those visible on its node.
    MPI_Comm localComm;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm), rank,
             "MPI_Comm_split_type");
    int localRank = 0;
    int localSize = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    // Respect an explicit thread setting; otherwise share node CPUs among ranks.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }
    int gpuCount = 0;
    checkCuda(cudaGetDeviceCount(&gpuCount), rank, "cudaGetDeviceCount");
    if (gpuCount == 0) fail(rank, "No CUDA device available", 0);
    checkCuda(cudaSetDevice(localRank % gpuCount), rank, "cudaSetDevice");

    const size_t baseRows = N / static_cast<size_t>(worldSize);
    const size_t extraRows = N % static_cast<size_t>(worldSize);
    const size_t rows = baseRows + (static_cast<size_t>(rank) < extraRows);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = rows * N;
    const size_t matrixElements = N * N;
    std::vector<double> A(localElements), B(rows ? matrixElements : 0),
                        C(localElements);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    initMatrix(A.data(), N, firstRow, rows);
    if (rows != 0) initMatrix(B.data(), N, 0, N);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cublasHandle_t handle = nullptr;
    if (rows != 0) {
        checkCuda(cudaMalloc(&deviceA, localElements * sizeof(double)), rank,
                  "cudaMalloc(A)");
        checkCuda(cudaMalloc(&deviceB, matrixElements * sizeof(double)), rank,
                  "cudaMalloc(B)");
        checkCuda(cudaMalloc(&deviceC, localElements * sizeof(double)), rank,
                  "cudaMalloc(C)");
        checkCublas(cublasCreate(&handle), rank, "cublasCreate");
    }

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), rank, "MPI_Barrier");
    const double start = MPI_Wtime();
    if (rows != 0) {
        checkCuda(cudaMemcpy(deviceA, A.data(), localElements * sizeof(double),
                             cudaMemcpyHostToDevice), rank, "copy A to GPU");
        checkCuda(cudaMemcpy(deviceB, B.data(), matrixElements * sizeof(double),
                             cudaMemcpyHostToDevice), rank, "copy B to GPU");
        const double alpha = 1.0, beta = 0.0;
        // cuBLAS is column-major. For row-major inputs, C^T = B^T * A^T.
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(N), static_cast<int>(rows),
                                static_cast<int>(N), &alpha, deviceB,
                                static_cast<int>(N), deviceA,
                                static_cast<int>(N), &beta, deviceC,
                                static_cast<int>(N)), rank, "cublasDgemm");
        checkCuda(cudaMemcpy(C.data(), deviceC, localElements * sizeof(double),
                             cudaMemcpyDeviceToHost), rank, "copy C from GPU");
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    checkMpi(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD), rank, "MPI_Reduce(time)");
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n",
                    static_cast<long>(maxElapsed * 1000.0));
        const double gflops = 2.0 * static_cast<double>(N) * N * N /
                              maxElapsed / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> fullC(matrixElements);
            std::copy(C.begin(), C.end(), fullC.begin());
            for (int peer = 1; peer < worldSize; ++peer) {
                const size_t peerRows = baseRows +
                    (static_cast<size_t>(peer) < extraRows);
                const size_t peerFirst = static_cast<size_t>(peer) * baseRows +
                    std::min(static_cast<size_t>(peer), extraRows);
                transferRows(fullC.data() + peerFirst * N, peerRows * N,
                             peer, false, rank);
            }
            print_results(fullC, "MatrixC");
        } else {
            transferRows(C.data(), localElements, 0, true, rank);
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(A, B, C, N, firstRow, rows, rank) ? 1 : 0;
        int allValid = 0;
        checkMpi(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN,
                               MPI_COMM_WORLD), rank, "MPI_Allreduce(validation)");
        valid = allValid;
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    if (handle) checkCublas(cublasDestroy(handle), rank, "cublasDestroy");
    if (deviceA) checkCuda(cudaFree(deviceA), rank, "cudaFree(A)");
    if (deviceB) checkCuda(cudaFree(deviceB), rank, "cudaFree(B)");
    if (deviceC) checkCuda(cudaFree(deviceC), rank, "cudaFree(C)");
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
