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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel for matrix multiplication using shared memory tiling
constexpr int TILE_SIZE = 32;

__global__ void matmulKernel(const double* __restrict__ A, 
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int rowsA, int N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < rowsA && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile of B
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < rowsA && col < N) {
        C[row * N + col] = sum;
    }
}

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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t rowsA, const size_t N) {
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    size_t sizeA = rowsA * N * sizeof(double);
    size_t sizeB = N * N * sizeof(double);
    size_t sizeC = rowsA * N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (rowsA + TILE_SIZE - 1) / TILE_SIZE);
    
    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on local rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0 only, then broadcast)
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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    
    // Broadcast parameters
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("CUDA devices: %d\n", deviceCount);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate row distribution
    std::vector<int> sendCounts(size);
    std::vector<int> displs(size);
    
    int baseRows = N / size;
    int extraRows = N % size;
    
    for (int i = 0; i < size; ++i) {
        int rows = baseRows + (i < extraRows ? 1 : 0);
        sendCounts[i] = rows * N;
        displs[i] = (i == 0) ? 0 : (displs[i-1] + sendCounts[i-1]);
    }
    
    int localRows = baseRows + (rank < extraRows ? 1 : 0);
    
    // Allocate matrices
    std::vector<double> A_full, B_full, C_full;
    if (rank == 0) {
        A_full.resize(N * N);
        B_full.resize(N * N);
        C_full.resize(N * N);
        
        // Initialize matrices
        if (rank == 0) printf("Initializing matrices...\n");
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                A_full[i * N + j] = getPseudoRndValue(N, i, j);
                B_full[i * N + j] = getPseudoRndValue(N, i, j);
            }
        }
    } else {
        B_full.resize(N * N);
    }
    
    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);
    
    // Distribute A matrix
    MPI_Scatterv(A_full.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), localRows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Broadcast B matrix to all ranks
    MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computing matrix multiplication...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform local matrix multiplication on GPU
    matrixMultiply(A_local, B_full, C_local, localRows, N);
    
    // Ensure all ranks complete
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather results at rank 0
    MPI_Gatherv(C_local.data(), localRows * N, MPI_DOUBLE,
                C_full.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B_full, C_full, N);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
