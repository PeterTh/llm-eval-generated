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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowOffset = 0) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowOffset + i, j);
        }
    }
}

// Simple validation: compute a few elements and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few deterministic positions
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
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Usage: %s [options]\n", progName);
        printf("Options:\n");
        printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
        printf("  -v           Enable validation\n");
        printf("  -r           Print results for external validation\n");
        printf("  -h           Show this help message\n");
    }
}

// Simple CUDA error check
#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

// Naive kernel that multiplies a local block of rows (rows x N) by full N x N B
extern "C" __global__ void matmul_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N, int rows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI first
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition rows across MPI ranks (block distribution)
    size_t baseRows = N / size;
    size_t remainder = N % size;
    size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    // compute row offset
    size_t rowOffset = 0;
    for (int r = 0; r < rank; ++r) rowOffset += baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);

    // Each rank initializes its local A block using deterministic function so no scatter is needed
    std::vector<double> A_local(localRows * N);
    initMatrix(A_local, N, rowOffset);

    // Each rank initializes full B locally (deterministic) to avoid communication; acceptable for benchmark
    std::vector<double> B(N * N);
    initMatrix(B, N, 0);

    // Prepare storage for local C
    std::vector<double> C_local(localRows * N);

    // Warmup / sync: ensure GPUs visible and device selected per rank (simple round-robin)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = deviceCount > 0 ? (rank % deviceCount) : 0;
    if (deviceCount > 0) CUDA_CHECK(cudaSetDevice(deviceId));

    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (deviceCount > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * localRows * N));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(double) * N * N));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(double) * localRows * N));

        // copy data to device (asynchronously where possible)
        CUDA_CHECK(cudaMemcpy(d_A, A_local.data(), sizeof(double) * localRows * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Launch CUDA kernel if available; otherwise fallback to OpenMP CPU multiply of local block
    if (deviceCount > 0) {
        const int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (localRows + TILE - 1) / TILE);
        // launch
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, static_cast<int>(N), static_cast<int>(localRows));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // copy back
        CUDA_CHECK(cudaMemcpy(C_local.data(), d_C, sizeof(double) * localRows * N, cudaMemcpyDeviceToHost));
    } else {
        // CPU fallback using OpenMP parallelization
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localRows; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k) sum += A_local[i * N + k] * B[k * N + j];
                C_local[i * N + j] = sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double localSec = localDuration.count() / 1000.0;

    // Get max time across ranks for correct GFLOPS calculation
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time (max across ranks): %.3f s\n", maxSec);
        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / maxSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // If printResults or validate requested, gather full C to rank 0
    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t rows_r = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(rows_r * N);
        displs[r] = static_cast<int>(( (r==0) ? 0 : displs[r-1] + recvcounts[r-1] ));
    }

    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results for external validation on rank 0
    if (rank == 0 && printResults) {
        print_results(C, "MatrixC");
    }

    // Validation on rank 0
    int finalExit = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        // Recreate full A and B deterministically for checking
        std::vector<double> A_full(N * N);
        std::vector<double> B_full(N * N);
        initMatrix(A_full, N, 0);
        initMatrix(B_full, N, 0);
        bool valid = validateResult(A_full, B_full, C, N);
        if (valid) {
            printf("Validation: PASSED\n");
            finalExit = 0;
        } else {
            printf("Validation: FAILED\n");
            finalExit = 1;
        }
    }

    // Cleanup device memory
    if (deviceCount > 0) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    MPI_Finalize();
    return finalExit;
}
