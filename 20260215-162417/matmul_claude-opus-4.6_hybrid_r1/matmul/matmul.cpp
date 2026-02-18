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

#define TILE_SIZE 32

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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

// Tiled CUDA kernel with shared memory and padding to avoid bank conflicts
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, int N, int num_rows) {
    __shared__ double As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < num_rows && a_col < N) ? A[row * N + a_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < num_rows && col < N) {
        C[row * N + col] = sum;
    }
}

// Each MPI rank offloads its row-slice to GPU
void matrixMultiplyGPU(const double* A_local, const double* B, double* C_local,
                       size_t N, size_t num_rows) {
    double *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, num_rows * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, num_rows * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A_local, num_rows * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((static_cast<int>(N) + TILE_SIZE - 1) / TILE_SIZE,
                (static_cast<int>(num_rows) + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<<<blocks, threads>>>(d_A, d_B, d_C, static_cast<int>(N),
                                       static_cast<int>(num_rows));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_local, d_C, num_rows * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU round-robin across ranks
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

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
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), num_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // All ranks initialize full A and B deterministically (avoids large broadcast)
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Distribute rows across MPI ranks
    size_t base_rows = N / static_cast<size_t>(nprocs);
    size_t remainder = N % static_cast<size_t>(nprocs);
    size_t start_row = static_cast<size_t>(rank) * base_rows
                       + std::min(static_cast<size_t>(rank), remainder);
    size_t num_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    std::vector<double> C_local(num_rows * N, 0.0);

    // Timed computation
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyGPU(A.data() + start_row * N, B.data(), C_local.data(), N, num_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather all row-slices of C on rank 0
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t r_start = static_cast<size_t>(r) * base_rows
                         + std::min(static_cast<size_t>(r), remainder);
        recvcounts[r] = static_cast<int>(r_rows * N);
        displs[r] = static_cast<int>(r_start * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(num_rows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
