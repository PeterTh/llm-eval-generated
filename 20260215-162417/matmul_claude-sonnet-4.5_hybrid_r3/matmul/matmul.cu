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

// CUDA kernel for matrix multiplication using tiled approach with shared memory
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B, 
                             double* __restrict__ C, const size_t N, const size_t rows_per_rank) {
    const int TILE_SIZE = 16;
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    
    // Iterate over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A
        if (row < rows_per_rank && (tile * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile from B
        if (col < N && (tile * TILE_SIZE + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < rows_per_rank && col < N) {
        C[row * N + col] = sum;
    }
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, const size_t rows_per_rank) {
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    const size_t A_size = rows_per_rank * N * sizeof(double);
    const size_t B_size = N * N * sizeof(double);
    const size_t C_size = rows_per_rank * N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B, B_size));
    CUDA_CHECK(cudaMalloc(&d_C, C_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (rows_per_rank + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, rows_per_rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, C_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Simple validation: compute a few elements and compare (parallelized with OpenMP)
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    bool valid = true;
    #pragma omp parallel for collapse(2) reduction(&& : valid)
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
                #pragma omp critical
                {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                }
                valid = false;
            }
        }
    }
    
    return valid;
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
    
    // Set GPU device based on rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI Processes: %d\n", size);
        printf("CUDA Devices: %d\n", deviceCount);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate rows per rank
    const size_t rows_per_rank = N / size;
    const size_t remainder = N % size;
    const size_t my_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    const size_t my_start_row = rank * rows_per_rank + std::min((size_t)rank, remainder);
    
    // Allocate matrices - each rank needs its portion of A, all of B, and its portion of C
    std::vector<double> A_local(my_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(my_rows * N);
    std::vector<double> A_full, C_full; // Only used by rank 0
    
    if (rank == 0) {
        A_full.resize(N * N);
        C_full.resize(N * N);
    }
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        if (rank == 0) printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B, N);
    }
    
    // Broadcast matrix B to all ranks
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Scatter matrix A rows to all ranks
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t r_rows = rows_per_rank + (r < remainder ? 1 : 0);
        sendcounts[r] = r_rows * N;
        displs[r] = (r * rows_per_rank + std::min((size_t)r, remainder)) * N;
    }
    
    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform matrix multiplication on each rank
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A_local, B, C_local, N, my_rows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather results back to rank 0
    MPI_Gatherv(C_local.data(), my_rows * N, MPI_DOUBLE,
                C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    
    // Only rank 0 prints results
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
            bool valid = validateResult(A_full, B, C_full, N);
            
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
