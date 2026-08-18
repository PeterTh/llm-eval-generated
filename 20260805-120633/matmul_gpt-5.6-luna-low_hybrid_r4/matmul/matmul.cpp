#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t rows, size_t N) {
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || col >= N) return;
    double sum = 0.0;
    for (size_t k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void cudaMultiply(const std::vector<double>& A, const std::vector<double>& B,
                  std::vector<double>& localC, size_t rows, size_t N) {
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytes = N * N * sizeof(double);
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, rows * N * sizeof(double));
    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);
    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matmulKernel<<<grid, block>>>(dA, dB, dC, rows, N);
    cudaDeviceSynchronize();
    cudaMemcpy(localC.data(), dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
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
    int rank = 0, worldSize = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 1;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Comm_free(&localComm); MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Comm_free(&localComm); MPI_Finalize(); return 1;
        }
    }
    
    if (N == 0) { if (rank == 0) fprintf(stderr, "Matrix size must be positive\n"); MPI_Comm_free(&localComm); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    const size_t rowBegin = (N * static_cast<size_t>(rank)) / static_cast<size_t>(worldSize);
    const size_t rowEnd = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(worldSize);
    const size_t localRows = rowEnd - rowBegin;
    std::vector<double> localC(localRows * N);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    cudaMultiply(A, B, localC, localRows, N);
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t first = (N * static_cast<size_t>(r)) / static_cast<size_t>(worldSize);
        const size_t last = (N * static_cast<size_t>(r + 1)) / static_cast<size_t>(worldSize);
        counts[r] = static_cast<int>((last - first) * N);
        displacements[r] = static_cast<int>(first * N);
    }
    MPI_Allgatherv(localC.data(), counts[rank], MPI_DOUBLE, C.data(), counts.data(),
                   displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    
    // Calculate GFLOPS
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Comm_free(&localComm); MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Comm_free(&localComm); MPI_Finalize(); return 1;
        }
    }
    
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
