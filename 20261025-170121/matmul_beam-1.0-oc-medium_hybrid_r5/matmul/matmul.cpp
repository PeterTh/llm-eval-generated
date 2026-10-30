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

__global__ void matMulKernel(const double* A, const double* B, double* C,
                            size_t N, size_t rowOffset, size_t numRows) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t localI = blockIdx.y * blockDim.y + threadIdx.y;
    if (localI >= numRows || j >= N) return;
    size_t i = rowOffset + localI;
    double sum = 0.0;
    for (size_t k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[localI * N + j] = sum;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, int rank, int nprocs) {
    size_t rowsPer = N / nprocs;
    size_t rem = N % nprocs;
    size_t rowStart = (size_t)rank * rowsPer + (rank < (int)rem ? rank : rem);
    size_t numRows = rowsPer + (rank < (int)rem ? 1 : 0);

    double *dA, *dB, *dC;
    cudaMalloc(&dA, N * N * sizeof(double));
    cudaMalloc(&dB, N * N * sizeof(double));
    cudaMalloc(&dC, numRows * N * sizeof(double));
    cudaMemcpy(dA, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (numRows + 15) / 16);
    matMulKernel<<<grid, block>>>(dA, dB, dC, N, rowStart, numRows);
    cudaDeviceSynchronize();

    std::vector<double> localC(numRows * N);
    cudaMemcpy(localC.data(), dC, numRows * N * sizeof(double), cudaMemcpyDeviceToHost);

    // gather counts/displacements
    std::vector<int> counts(nprocs), displs(nprocs);
    size_t off = 0;
    for (int r = 0; r < nprocs; ++r) {
        size_t rp = N / nprocs + (r < (int)(N % nprocs) ? 1 : 0);
        counts[r] = (int)(rp * N);
        displs[r] = (int)(off * N);
        off += rp;
    }
    MPI_Gatherv(localC.data(), (int)(numRows * N), MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool ok = true;
#pragma omp parallel for
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
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
                ok = false;
            }
        }
    }
    return ok;
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);

    if (rank == 0) printf("Initializing matrices...\n");
    // all ranks init same (deterministic)
    initMatrix(A, N);
    initMatrix(B, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C, N, rank, nprocs);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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
