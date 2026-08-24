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

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N, int rows_per_rank) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_per_rank && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
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


// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool all_valid = true;
    
    #pragma omp parallel for reduction(&:all_valid)
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
                all_valid = false;
            }
        }
    }
    
    return all_valid;
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
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set CUDA device based on local rank (assuming 1 GPU per rank or round-robin)
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        // Typically, local rank is needed, but we'll use global rank % num_devices
        // Ideally, we'd use MPI_Comm_split_type to find local rank
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int local_rank;
        MPI_Comm_rank(nodeComm, &local_rank);
        CHECK_CUDA(cudaSetDevice(local_rank % num_devices));
        MPI_Comm_free(&nodeComm);
    } else {
        if (rank == 0) printf("Warning: No CUDA devices found. Falling back might not work as intended.\n");
    }

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
            if (rank == 0) printUsage(argv[0]);
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
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine rows per rank
    size_t rows_per_rank = N / size;
    size_t remainder = N % size;
    // size_t start_row = rank * rows_per_rank + std::min((size_t)rank, remainder);
    size_t my_rows = rows_per_rank + (rank < (int)remainder ? 1 : 0);

    // Prepare displacements and counts for Scatterv/Gatherv
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        int disp = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (N / size + (i < (int)remainder ? 1 : 0)) * N;
            displs[i] = disp;
            disp += sendcounts[i];
        }
    }

    // Allocate matrices
    std::vector<double> A_full;
    std::vector<double> B_full(N * N);
    std::vector<double> C_full;
    
    std::vector<double> A_local(my_rows * N);
    std::vector<double> C_local(my_rows * N);

    if (rank == 0) {
        A_full.resize(N * N);
        C_full.resize(N * N);
        
        printf("Initializing matrices...\n");
        initMatrix(A_full, N);
        initMatrix(B_full, N);
    }

    // Broadcast B
    MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A
    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 
                 A_local.data(), my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // CUDA Computation
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, my_rows * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_C, my_rows * N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, A_local.data(), my_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_full.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (my_rows + block.y - 1) / block.y);
    
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N, my_rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C_local.data(), d_C, my_rows * N * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C
    MPI_Gatherv(C_local.data(), my_rows * N, MPI_DOUBLE, 
                C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
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
            }
        }
    }

    MPI_Finalize();
    return 0;
}
