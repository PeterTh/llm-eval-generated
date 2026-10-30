#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void matmulKernel(const double* A, const double* B, double* C, size_t N, size_t rowStart, size_t numRows) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRows || j >= N) return;
    double sum = 0.0;
    for (size_t k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

void matrixMultiplyMPIOMP_CUDA(const std::vector<double>& A, const std::vector<double>& B,
                              std::vector<double>& C, const size_t N, int rank, int size) {
    size_t rowsPer = N / size;
    size_t rem = N % size;
    size_t start = rank * rowsPer + (rank < (int)rem ? rank : rem);
    size_t numRows = rowsPer + (rank < (int)rem ? 1 : 0);

    const double* Ablk = A.data() + start * N;
    double* Cblk = C.data() + start * N;

    double *dA, *dB, *dC;
    cudaMalloc(&dA, numRows * N * sizeof(double));
    cudaMalloc(&dB, N * N * sizeof(double));
    cudaMalloc(&dC, numRows * N * sizeof(double));
    cudaMemcpy(dA, Ablk, numRows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (numRows + 15) / 16);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N, start, numRows);
    cudaDeviceSynchronize();
    cudaMemcpy(Cblk, dC, numRows * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

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
                MPI_Finalize();
                return 0;
            }
        }
    }
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyMPIOMP_CUDA(A, B, C, N, rank, size);

    // gather C (each rank wrote its block into its own C? Actually C shared? No, each has full C zero; only block filled)
    // Need combine: use MPI_Allreduce sum
    MPI_Allreduce(MPI_IN_PLACE, C.data(), N * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(C, "MatrixC");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    MPI_Finalize();
    return 0;
}
