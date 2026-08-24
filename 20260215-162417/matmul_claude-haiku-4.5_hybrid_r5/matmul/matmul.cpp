#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

#ifdef ENABLE_CUDA
bool tryGPUMultiply(const std::vector<double>& A, const std::vector<double>& B,
                   std::vector<double>& C, const size_t N) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        return false;
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Allocate GPU memory
    if (cudaMalloc(&d_A, N * N * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_B, N * N * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_C, N * N * sizeof(double)) != cudaSuccess) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        cublasDestroy(handle);
        return false;
    }

    // Copy matrices to GPU
    cudaMemcpy(d_A, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, N * N * sizeof(double));

    // Perform matrix multiplication: C = A * B
    double alpha = 1.0, beta = 0.0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return false;
    }

    // Copy result back to CPU
    cudaMemcpy(C.data(), d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return true;
}
#endif

void matrixMultiplyLocal(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N,
                         const size_t start_row, const size_t end_row) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiplyHybrid(const std::vector<double>& A, const std::vector<double>& B,
                          std::vector<double>& C, const size_t N,
                          int rank, int world_size) {
    // Distribute rows across MPI processes
    size_t rows_per_process = N / world_size;
    size_t remainder = N % world_size;
    
    size_t start_row = rank * rows_per_process + std::min(static_cast<size_t>(rank), remainder);
    size_t end_row = start_row + rows_per_process + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t my_rows = end_row - start_row;

    // Build scatter counts and displacements
    std::vector<int> send_counts(world_size);
    std::vector<int> send_disps(world_size);
    for (int p = 0; p < world_size; ++p) {
        size_t p_start = p * rows_per_process + std::min(static_cast<size_t>(p), remainder);
        size_t p_end = p_start + rows_per_process + (static_cast<size_t>(p) < remainder ? 1 : 0);
        send_counts[p] = (p_end - p_start) * N;
        send_disps[p] = p_start * N;
    }

    // Broadcast B to all processes
    std::vector<double> B_local = B;
    MPI_Bcast(B_local.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A rows
    std::vector<double> A_local(my_rows * N);
    MPI_Scatterv(A.data(), send_counts.data(), send_disps.data(), MPI_DOUBLE,
                 A_local.data(), my_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Compute local portion with OpenMP
    std::vector<double> C_local(my_rows * N, 0.0);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t local_i = 0; local_i < my_rows; ++local_i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A_local[local_i * N + k] * B_local[k * N + j];
            }
            C_local[local_i * N + j] = sum;
        }
    }

    // Gather results back to rank 0
    MPI_Gatherv(C_local.data(), my_rows * N, MPI_DOUBLE,
                C.data(), send_counts.data(), send_disps.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // Try GPU first if available
    #ifdef ENABLE_CUDA
    if (tryGPUMultiply(A, B, C, N)) {
        return;
    }
    #endif

    // Fall back to hybrid MPI+OpenMP
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size == 1) {
        // Single process: use OpenMP only
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    } else {
        matrixMultiplyHybrid(A, B, C, N, rank, world_size);
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
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (rank 0 only)
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", world_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        #ifdef ENABLE_CUDA
        int device_count;
        cudaGetDeviceCount(&device_count);
        printf("CUDA devices: %d\n", device_count);
        #endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    // Initialize matrices (all processes participate)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    // Synchronize before computation
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            
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
