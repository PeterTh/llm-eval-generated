#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const double* A, const double* B, double* C, size_t N, size_t row_offset) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplyCUDA(const double* A, const double* B, double* C, size_t N, size_t local_rows) {
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, local_rows * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, local_rows * N * sizeof(double));
    cudaMemcpy(d_A, A, local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N, 0);
    cudaMemcpy(C, d_C, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void matrixMultiply(const double* A, const double* B, double* C, size_t N, size_t local_rows) {
    #pragma omp parallel for
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
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
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    size_t rows_per_proc = N / world_size;
    size_t remainder = N % world_size;
    size_t local_rows = rows_per_proc + (world_rank < remainder ? 1 : 0);
    size_t row_offset = world_rank * rows_per_proc + (world_rank < remainder ? world_rank : remainder);

    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N);
    std::vector<double> C;
    if (world_rank == 0) C.resize(N * N);

    // Initialize matrices
    if (world_rank == 0) {
        std::vector<double> A(N * N);
        initMatrix(A, N);
        initMatrix(B, N);
        // Scatter A
        int* sendcounts = new int[world_size];
        int* displs = new int[world_size];
        size_t offset = 0;
        for (int i = 0; i < world_size; ++i) {
            size_t rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = offset;
            offset += rows * N;
        }
        MPI_Scatterv(A.data(), sendcounts, displs, MPI_DOUBLE, A_local.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] sendcounts;
        delete[] displs;
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, A_local.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    matrixMultiplyCUDA(A_local.data(), B.data(), C_local.data(), N, local_rows);
    double t1 = MPI_Wtime();

    // Gather C
    if (world_rank == 0) {
        int* recvcounts = new int[world_size];
        int* displs = new int[world_size];
        size_t offset = 0;
        for (int i = 0; i < world_size; ++i) {
            size_t rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * N;
            displs[i] = offset;
            offset += rows * N;
        }
        MPI_Gatherv(C_local.data(), local_rows * N, MPI_DOUBLE, C.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] recvcounts;
        delete[] displs;
    } else {
        MPI_Gatherv(C_local.data(), local_rows * N, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        double duration = (t1 - t0) * 1000.0;
        printf("Computation time: %.0f ms\n", duration);
        double gflops = (2.0 * N * N * N) / ((t1 - t0)) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(C, "MatrixC");
        }
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            std::vector<double> Bv(N * N);
            initMatrix(A, N);
            initMatrix(Bv, N);
            bool valid = validateResult(A, Bv, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    MPI_Finalize();
    return 0;
}
