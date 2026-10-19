#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

__global__ void matmulKernel(const double* A, const double* B, double* C,
                             int rows, int N) {
    constexpr int T = 16;
    __shared__ double tileA[T][T], tileB[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    double sum = 0.0;
    for (int base = 0; base < N; base += T) {
        int ak = base + threadIdx.x, bk = base + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = row < rows && ak < N ? A[row * N + ak] : 0.0;
        tileB[threadIdx.y][threadIdx.x] = bk < N && col < N ? B[bk * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, int startRow, int rows) {
    double *dA, *dB, *dC;
    size_t aBytes = static_cast<size_t>(rows) * N * sizeof(double);
    size_t bBytes = N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dA, std::max(aBytes, sizeof(double))));
    CUDA_CHECK(cudaMalloc(&dB, bBytes));
    CUDA_CHECK(cudaMalloc(&dC, std::max(aBytes, sizeof(double))));
    if (rows) CUDA_CHECK(cudaMemcpy(dA, A.data() + static_cast<size_t>(startRow) * N, aBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bBytes, cudaMemcpyHostToDevice));
    if (rows) {
        dim3 block(16, 16), grid((N + 15) / 16, (rows + 15) / 16);
        matmulKernel<<<grid, block>>>(dA, dB, dC, rows, static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C.data() + static_cast<size_t>(startRow) * N, dC, aBytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
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
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (N == 0 || N > static_cast<size_t>(INT_MAX)) { if(rank==0) fprintf(stderr,"Invalid matrix size\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled"); }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank; MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) { fprintf(stderr,"Rank %d: no usable CUDA device\n",rank); MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);
    
    // Allocate matrices
    std::vector<double> A(N * N), B(N * N), C(N * N);
    
    // Initialize matrices
    if(rank==0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    if(rank==0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    int* counts = rank == 0 ? new int[world] : nullptr;
    int* displs = rank == 0 ? new int[world] : nullptr;
    for(int r=0;r<world;++r) if(rank==0) { int begin=(static_cast<long long>(N)*r)/world, end=(static_cast<long long>(N)*(r+1))/world; counts[r]=(end-begin)*N; displs[r]=begin*N; }
    int begin=(static_cast<long long>(N)*rank)/world, endRow=(static_cast<long long>(N)*(rank+1))/world;
    matrixMultiply(A, B, C, N, begin, endRow-begin);
    MPI_Gatherv(rank==0 ? MPI_IN_PLACE : C.data()+static_cast<size_t>(begin)*N,
                (endRow-begin)*N, MPI_DOUBLE, C.data(), counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if(rank==0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = duration.count() ? (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9 : 0.0;
    if(rank==0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank==0) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate && rank==0) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            delete[] counts; delete[] displs; MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            delete[] counts; delete[] displs; MPI_Finalize(); return 1;
        }
    }
    
    delete[] counts; delete[] displs; MPI_Finalize(); return 0;
}
