#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// CUDA kernel: parallel matrix initialization
__global__ void initMatrixKernel(double* mat, const size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * N;
    if (idx < total) {
        size_t i = idx / N;
        size_t j = idx % N;
        mat[idx] = getPseudoRndValue(N, i, j);
    }
}

// Tile size for CUDA kernel
constexpr int TILE_DIM = 32;

// CUDA kernel: tiled shared-memory GEMM with double precision
// C = A * B, all matrices are N x N in row-major order
// Each block computes one TILE_DIM x TILE_DIM tile of C
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t N) {
    __shared__ double tileA[TILE_DIM][TILE_DIM];
    __shared__ double tileB[TILE_DIM][TILE_DIM];

    const int tid = threadIdx.x;
    const int tidj = threadIdx.y;

    const int row = blockIdx.y * TILE_DIM + tid;
    const int col = blockIdx.x * TILE_DIM + tidj;

    double sum = 0.0;

    for (int m = 0; m < (N + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load tiles into shared memory
        if (row < N && (m * TILE_DIM + tidj) < N)
            tileA[tid][tidj] = A[row * N + m * TILE_DIM + tidj];
        else
            tileA[tid][tidj] = 0.0;

        if (col < N && (m * TILE_DIM + tid) < N)
            tileB[tid][tidj] = B[(m * TILE_DIM + tid) * N + col];
        else
            tileB[tid][tidj] = 0.0;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += tileA[tid][k] * tileB[k][tidj];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse)
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
    
    // Print info only from rank 0
    if (rank == 0) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", deviceCount);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // 1D block decomposition: each rank owns a block of rows
    size_t localRows = (N + numRanks - 1) / numRanks;
    size_t localStartRow = rank * localRows;
    size_t localEndRow = std::min(localStartRow + localRows, N);
    size_t localRowCount = localEndRow - localStartRow;
    
    // Allocate local matrices:
    // A_local: localRowCount x N
    // B: N x N (replicated on all ranks)
    // C_local: localRowCount x N
    size_t localASize = localRowCount * N;
    size_t localCSize = localRowCount * N;
    size_t BSize = N * N;
    
    std::vector<double> h_A(localASize);
    std::vector<double> h_B(BSize);
    std::vector<double> h_C(localCSize, 0.0);
    
    // Initialize matrices with OpenMP parallelism
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    
    // Initialize local portion of A
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < localRowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            h_A[i * N + j] = getPseudoRndValue(N, localStartRow + i, j);
        }
    }
    
    // Initialize B (replicated on all ranks)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            h_B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
    
    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate GPU memory
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, localASize * sizeof(double));
    cudaMalloc(&d_B, BSize * sizeof(double));
    cudaMalloc(&d_C, localCSize * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_A, h_A.data(), localASize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), BSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, localCSize * sizeof(double));
    
    // Configure CUDA kernel launch parameters
    dim3 blockSize(TILE_DIM, TILE_DIM);  // 32x32 thread blocks
    dim3 gridSize((N + TILE_DIM - 1) / TILE_DIM,
                  (localRowCount + TILE_DIM - 1) / TILE_DIM);
    
    // Synchronize and start timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch CUDA kernel
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // Synchronize GPU
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy results back
    cudaMemcpy(h_C.data(), d_C, localCSize * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather all C results to rank 0 for validation/output
    std::vector<double> full_C(N * N);
    
    // Use MPI_Gatherv to gather variable-sized row blocks
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displacements(numRanks);
    
    int cum = 0;
    for (int r = 0; r < numRanks; ++r) {
        size_t rLocalRows = std::min((r + 1) * localRows, N) - r * localRows;
        recvCounts[r] = static_cast<int>(rLocalRows * N);
        displacements[r] = cum;
        cum += recvCounts[r];
    }
    
    MPI_Gatherv(h_C.data(), static_cast<int>(localCSize), MPI_DOUBLE,
                full_C.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Calculate total GFLOPS across all ranks
    double totalGflops = 0.0;
    double localGflops = 0.0;
    if (duration.count() > 0) {
        localGflops = (2.0 * static_cast<double>(localRowCount) * N * N) /
                      (duration.count() / 1000.0) / 1e9;
    }
    MPI_Reduce(&localGflops, &totalGflops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Print results only from rank 0
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);
        printf("Performance: %.3f GFLOPS\n", totalGflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(full_C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Reconstruct full A and B for validation
            std::vector<double> full_A(N * N);
            std::vector<double> full_B(N * N);
            initMatrix(full_A, N);
            initMatrix(full_B, N);
            
            bool valid = validateResult(full_A, full_B, full_C, N);
            
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
