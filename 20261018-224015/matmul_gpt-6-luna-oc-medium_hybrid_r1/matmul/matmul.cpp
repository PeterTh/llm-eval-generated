#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C,
                                     size_t N, size_t firstRow, size_t rows) {
    constexpr unsigned TILE = 16;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const size_t row = firstRow + blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < N; base += TILE) {
        const size_t ak = base + threadIdx.x;
        const size_t bk = base + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < firstRow + rows && ak < N) ? A[row * N + ak] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < TILE; ++k) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < firstRow + rows && col < N) C[(row - firstRow) * N + col] = sum;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t firstRow, size_t rows) {
    if (rows == 0) return;
    const size_t matrixBytes = N * N * sizeof(double);
    const size_t outputBytes = rows * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    auto checkCuda = [](cudaError_t error) {
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    };
    checkCuda(cudaMalloc(&dA, matrixBytes));
    checkCuda(cudaMalloc(&dB, matrixBytes));
    checkCuda(cudaMalloc(&dC, outputBytes));
    checkCuda(cudaMemcpy(dA, A.data(), matrixBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dB, B.data(), matrixBytes, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (rows + 15) / 16);
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N, firstRow, rows);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(C.data(), dC, outputBytes, cudaMemcpyDeviceToHost));
    cudaFree(dC); cudaFree(dB); cudaFree(dA);
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
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N > static_cast<size_t>(std::sqrt(INT_MAX))) { if (rank == 0) fprintf(stderr, "Invalid matrix size for MPI collectives\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    int devices = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&devices);
    if (deviceStatus != cudaSuccess || devices == 0) { fprintf(stderr, "Rank %d has no usable CUDA device\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    cudaSetDevice(localRank % devices);
    
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
    auto start = std::chrono::high_resolution_clock::now();
    const size_t firstRow = N * static_cast<size_t>(rank) / worldSize;
    const size_t endRow = N * static_cast<size_t>(rank + 1) / worldSize;
    const size_t rows = endRow - firstRow;
    std::vector<double> localC(rows * N);
    matrixMultiply(A, B, localC, N, firstRow, rows);
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        size_t begin = N * static_cast<size_t>(r) / worldSize;
        size_t end = N * static_cast<size_t>(r + 1) / worldSize;
        counts[r] = static_cast<int>((end - begin) * N);
        displacements[r] = static_cast<int>(begin * N);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(rows * N), MPI_DOUBLE, C.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(C.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Allreduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    std::chrono::duration<double> duration(maxSeconds);
    
    if (rank == 0) printf("Computation time: %.3f ms\n", duration.count() * 1000.0);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / duration.count() / 1e9;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    int validResult = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            validResult = 1;
        } else {
            printf("Validation: FAILED\n");
            validResult = 0;
        }
    }
    MPI_Bcast(&validResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return validResult ? 0 : 1;
}
