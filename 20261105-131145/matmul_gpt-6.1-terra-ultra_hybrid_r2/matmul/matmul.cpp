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
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void fail(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkBlas(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "Rank %d: %s failed (cuBLAS status %d)\n", rank, operation, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED", rank);

    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* arg = argv[++i];
            unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || end == arg || *end || arg[0] == '-' || value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", arg);
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

    const size_t base = N / static_cast<size_t>(ranks);
    const size_t remainder = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < remainder);
    const size_t firstRow = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);
    const size_t localElements = rows * N;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
        std::fflush(stdout);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    std::vector<double> A, B, C;
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cublasHandle_t handle = nullptr;
    if (rows) {
        int devices = 0;
        checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
        if (!devices) fail("No CUDA device available", rank);
        checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice", rank);
        try {
            A.resize(localElements);
            B.resize(N * N);
            C.resize(localElements);
        } catch (const std::exception&) {
            fail("Host matrix allocation failed", rank);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j)
                B[i * N + j] = getPseudoRndValue(N, i, j);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            std::copy_n(B.data() + (firstRow + i) * N, N, A.data() + i * N);

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dA), localElements * sizeof(double)), "cudaMalloc A", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dB), N * N * sizeof(double)), "cudaMalloc B", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dC), localElements * sizeof(double)), "cudaMalloc C", rank);
        checkBlas(cublasCreate(&handle), "cublasCreate", rank);
        // Load the selected cuBLAS kernel before timing; the measured operation
        // still includes the input and output transfers required for host data.
        checkCuda(cudaMemcpy(dA, A.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "warm up copy A", rank);
        checkCuda(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "warm up copy B", rank);
        const double alpha = 1.0, beta = 0.0;
        checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "warm up cublasDgemm", rank);
        checkCuda(cudaDeviceSynchronize(), "warm up matrix multiplication", rank);
    }

    if (rank == 0) { std::printf("Computing matrix multiplication...\n"); std::fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows) {
        checkCuda(cudaMemcpy(dA, A.data(), localElements * sizeof(double), cudaMemcpyHostToDevice), "copy A to GPU", rank);
        checkCuda(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice), "copy B to GPU", rank);
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)), "cublasDgemm", rank);
        checkCuda(cudaDeviceSynchronize(), "matrix multiplication", rank);
        checkCuda(cudaMemcpy(C.data(), dC, localElements * sizeof(double), cudaMemcpyDeviceToHost), "copy C from GPU", rank);
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rows) {
        checkBlas(cublasDestroy(handle), "cublasDestroy", rank);
        checkCuda(cudaFree(dA), "cudaFree A", rank);
        checkCuda(cudaFree(dB), "cudaFree B", rank);
        checkCuda(cudaFree(dC), "cudaFree C", rank);
    }
    if (rank == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
    }

    if (printResults) {
        // Chunk transfers so MPI's int count does not limit matrix size.
        std::vector<double> result;
        if (rank == 0) {
            try { result.resize(N * N); }
            catch (const std::exception&) { fail("Result allocation failed", rank); }
            std::copy(C.begin(), C.end(), result.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t sourceRows = base + (static_cast<size_t>(source) < remainder);
                const size_t sourceFirst = base * static_cast<size_t>(source) + std::min(static_cast<size_t>(source), remainder);
                const size_t count = sourceRows * N;
                for (size_t offset = 0; offset < count;) {
                    int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + sourceFirst * N + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(chunk);
                }
            }
            print_results(result, "MatrixC");
        } else {
            for (size_t offset = 0; offset < localElements;) {
                int chunk = static_cast<int>(std::min(localElements - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += static_cast<size_t>(chunk);
            }
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        for (size_t pi = 0; pi < 5 && valid; ++pi) {
            const size_t i = pi % N;
            if (i < firstRow || i >= firstRow + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = pj % N;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += A[(i - firstRow) * N + k] * B[k * N + j];
                const double actual = C[(i - firstRow) * N + j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(relError) || relError > 1e-6) {
                    std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                i, j, expected, actual, relError);
                    valid = 0;
                    break;
                }
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
