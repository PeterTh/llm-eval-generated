#include <mpi.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Keep the original initialization formula and its unsigned arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

[[noreturn]] void fail(const char* operation, int code, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed (code %d)\n", rank, operation, code);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t result, const char* operation, int rank) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

void checkCublas(cublasStatus_t result, const char* operation, int rank) {
    if (result != CUBLAS_STATUS_SUCCESS) fail(operation, static_cast<int>(result), rank);
}

void initMatrix(std::vector<double>& mat, size_t N, size_t firstRow, size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t globalRow = firstRow + row;
        for (size_t col = 0; col < N; ++col)
            mat[row * N + col] = getPseudoRndValue(N, globalRow, col);
    }
}

// Each MPI rank owns consecutive rows of C and the corresponding rows of A.
// B is initialized locally to avoid broadcasting an N*N matrix.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows, int rank) {
    if (rows == 0) return;

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t localBytes = rows * N * sizeof(double);
    const size_t fullBytes = N * N * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), localBytes), "cudaMalloc(A)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), fullBytes), "cudaMalloc(B)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), localBytes), "cudaMalloc(C)", rank);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate", rank);
    checkCuda(cudaMemcpy(deviceA, A.data(), localBytes, cudaMemcpyHostToDevice), "copy A to GPU", rank);
    checkCuda(cudaMemcpy(deviceB, B.data(), fullBytes, cudaMemcpyHostToDevice), "copy B to GPU", rank);

    // A and B are row major. Reversing their order in column major cuBLAS
    // computes the transpose of C in the same row major output buffer.
    const double alpha = 1.0, beta = 0.0;
    checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                            &alpha, deviceB, static_cast<int>(N), deviceA, static_cast<int>(N),
                            &beta, deviceC, static_cast<int>(N)), "cublasDgemm", rank);
    checkCuda(cudaMemcpy(C.data(), deviceC, localBytes, cudaMemcpyDeviceToHost), "copy C from GPU", rank);

    checkCublas(cublasDestroy(handle), "cublasDestroy", rank);
    checkCuda(cudaFree(deviceA), "cudaFree(A)", rank);
    checkCuda(cudaFree(deviceB), "cudaFree(B)", rank);
    checkCuda(cudaFree(deviceC), "cudaFree(C)", rank);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t firstRow, size_t rows, int rank) {
    // Match the original 5x5 sample, assigning each sampled element to its owner.
    int failures = 0;
#pragma omp parallel for collapse(2) reduction(+:failures) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = static_cast<size_t>(pi) % N;
            const size_t j = static_cast<size_t>(pj) % N;
            if (i < firstRow || i >= firstRow + rows) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[(i - firstRow) * N + k] * B[k * N + j];
            const double actual = C[(i - firstRow) * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
#pragma omp critical
                std::printf("Rank %d: Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            rank, i, j, expected, actual, relError);
                ++failures;
            }
        }
    }
    return failures == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (argv[i][0] == '-' || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(INT_MAX) ||
                value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max() / value / sizeof(double))) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Local rank gives each process a different visible GPU on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(16, omp_get_num_procs() / localSize)));
    int gpuCount = 0;
    checkCuda(cudaGetDeviceCount(&gpuCount), "cudaGetDeviceCount", rank);
    if (gpuCount == 0) fail("no CUDA GPU available", 0, rank);
    checkCuda(cudaSetDevice(localRank % gpuCount), "cudaSetDevice", rank);

    const size_t firstRow = N * static_cast<size_t>(rank) / static_cast<size_t>(worldSize);
    const size_t lastRow = N * static_cast<size_t>(rank + 1) / static_cast<size_t>(worldSize);
    const size_t rows = lastRow - firstRow;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    initMatrix(A, N, firstRow, rows);
    if (rows) initMatrix(B, N, 0, N);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    matrixMultiply(A, B, C, N, rows, rank);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    if (printResults) {
        // MPI counts are ints; transfer complete rows in chunks for large matrices.
        const size_t chunkRows = static_cast<size_t>(INT_MAX) / N;
        if (rank == 0) {
            std::vector<double> gathered(N * N);
            std::copy(C.begin(), C.end(), gathered.begin() + firstRow * N);
            for (int r = 1; r < worldSize; ++r) {
                const size_t begin = N * static_cast<size_t>(r) / static_cast<size_t>(worldSize);
                const size_t finish = N * static_cast<size_t>(r + 1) / static_cast<size_t>(worldSize);
                for (size_t row = begin; row < finish; row += chunkRows) {
                    const size_t countRows = std::min(chunkRows, finish - row);
                    MPI_Recv(gathered.data() + row * N, static_cast<int>(countRows * N),
                             MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            print_results(gathered, "MatrixC");
        } else {
            for (size_t row = 0; row < rows; row += chunkRows) {
                const size_t countRows = std::min(chunkRows, rows - row);
                MPI_Send(C.data() + row * N, static_cast<int>(countRows * N),
                         MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    int failed = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const int localFailed = !validateResult(A, B, C, N, firstRow, rows, rank);
        MPI_Allreduce(&localFailed, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) std::printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
