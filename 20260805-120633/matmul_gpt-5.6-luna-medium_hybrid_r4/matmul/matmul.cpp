#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
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
            mat[static_cast<size_t>(i) * N + j] =
                getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

[[noreturn]] void cudaError(const char* what, cudaError_t status) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C, size_t rows, size_t N) {
    constexpr int TILE = 32;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;

    for (size_t tile = 0; tile < (N + TILE - 1) / TILE; ++tile) {
        const size_t aCol = tile * TILE + threadIdx.x;
        const size_t bRow = tile * TILE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < rows && aCol < N) ? A[row * N + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

void matrixMultiplyCUDA(const std::vector<double>& localA,
                        const std::vector<double>& B,
                        std::vector<double>& localC, const size_t rows,
                        const size_t N) {
    if (rows == 0) return;
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t bytesA = rows * N * sizeof(double);
    const size_t bytesB = N * N * sizeof(double);
    const size_t bytesC = rows * N * sizeof(double);
    auto check = [](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) cudaError(msg, e);
    };
    check(cudaMalloc(&deviceA, bytesA), "cudaMalloc(A)");
    check(cudaMalloc(&deviceB, bytesB), "cudaMalloc(B)");
    check(cudaMalloc(&deviceC, bytesC), "cudaMalloc(C)");
    check(cudaMemcpy(deviceA, localA.data(), bytesA, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(deviceB, B.data(), bytesB, cudaMemcpyHostToDevice), "copy B");
    constexpr int TILE = 32;
    const dim3 block(TILE, TILE);
    const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                    static_cast<unsigned>((rows + TILE - 1) / TILE));
    matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, rows, N);
    check(cudaGetLastError(), "matmul kernel launch");
    check(cudaDeviceSynchronize(), "matmul kernel");
    check(cudaMemcpy(localC.data(), deviceC, bytesC, cudaMemcpyDeviceToHost), "copy C");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    matrixMultiplyCUDA(A, B, C, N, N);
}

void validateLocal(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t rows,
                   const size_t N, int& valid) {
#pragma omp parallel for schedule(static) reduction(&: valid)
    for (long long localI = 0; localI < static_cast<long long>(rows); ++localI) {
        for (size_t j = 0; j < N; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) expected += A[localI * N + k] * B[k * N + j];
            const double error = std::abs(C[localI * N + j] - expected);
            const double scale = std::max(1.0, std::abs(expected));
            if (error > 1e-6 * scale) valid = 0;
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
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "N must be in the range 1..%d\n", std::numeric_limits<int>::max());
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
    }

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device available: %s\n",
                                    cudaGetErrorString(deviceStatus));
        MPI_Finalize();
        return 1;
    }
    // Prefer a launcher-provided local rank so ranks on different nodes map
    // independently to their node's GPUs; fall back to the MPI rank.
    int localRank = rank;
    const char* localRankText = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!localRankText) localRankText = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (!localRankText) localRankText = std::getenv("SLURM_LOCALID");
    if (localRankText) localRank = std::atoi(localRankText);
    cudaError_t cudaStatus = cudaSetDevice(localRank % deviceCount);
    if (cudaStatus != cudaSuccess) cudaError("cudaSetDevice", cudaStatus);
    
    // Allocate matrices
    const size_t firstRow = (N * static_cast<size_t>(rank)) / static_cast<size_t>(ranks);
    const size_t endRow = (N * static_cast<size_t>(rank + 1)) / static_cast<size_t>(ranks);
    const size_t localRows = endRow - firstRow;
    std::vector<double> A(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(localRows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    // A is initialized by global row index so every MPI rank gets identical data.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localRows); ++i)
        for (size_t j = 0; j < N; ++j)
            A[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, firstRow + static_cast<size_t>(i), j);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyCUDA(A, B, C, localRows, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long localMilliseconds = duration.count();
    long long milliseconds = 0;
    MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate GFLOPS
    if (rank == 0) {
        printf("Computation time: %lld ms\n", milliseconds);
        double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> fullC;
        if (rank == 0) fullC.resize(N * N);
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(((N * static_cast<size_t>(r + 1)) / ranks -
                                          (N * static_cast<size_t>(r)) / ranks) * N);
            displacements[r] = static_cast<int>((N * static_cast<size_t>(r) / ranks) * N);
        }
        MPI_Gatherv(C.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                    fullC.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) print_results(fullC, "MatrixC");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = 1;
        validateLocal(A, B, C, localRows, N, localValid);
        int localFlag = localValid ? 1 : 0, globalFlag = 0;
        MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        bool valid = globalFlag != 0;
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
