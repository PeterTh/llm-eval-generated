#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

__global__ void matmulKernel(const double* A, const double* B, double* C,
                             size_t N, size_t rows, size_t rowStart) {
    constexpr int TILE = 16;
    __shared__ double aTile[TILE][TILE];
    __shared__ double bTile[TILE][TILE];
    const size_t row = blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < N; base += TILE) {
        const size_t ak = base + threadIdx.x;
        const size_t bk = base + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && ak < N) ? A[row * N + ak] : 0.0;
        bTile[threadIdx.y][threadIdx.x] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
    (void)rowStart;
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void initializeBlock(std::vector<double>& A, size_t N, size_t rowStart, size_t rows) {
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(rows); ++r)
        for (size_t j = 0; j < N; ++j) A[static_cast<size_t>(r) * N + j] = getPseudoRndValue(N, rowStart + static_cast<size_t>(r), j);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows, size_t rowStart) {
    if (rows == 0) return;
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dA, A.size() * sizeof(double)));
    cudaCheck(cudaMalloc(&dB, B.size() * sizeof(double)));
    cudaCheck(cudaMalloc(&dC, C.size() * sizeof(double)));
    cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    dim3 block(16, 16), grid((N + 15) / 16, (rows + 15) / 16);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N, rows, rowStart);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "MPI rank %d has no CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaSetDevice(rank % deviceCount);
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
    
    if (N == 0 || N > static_cast<size_t>(INT_MAX) || N * N > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Matrix size is out of supported range\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> B(N * N);
    if (rank == 0) printf("Initializing matrices...\n");
    #pragma omp parallel for schedule(static)
    for (long long x = 0; x < static_cast<long long>(N * N); ++x)
        B[static_cast<size_t>(x)] = getPseudoRndValue(N, static_cast<size_t>(x) / N, static_cast<size_t>(x) % N);

    std::vector<int> counts(world), displs(world), rowCounts(world), rowDispls(world);
    for (int p = 0; p < world; ++p) {
        size_t begin = N * static_cast<size_t>(p) / world;
        size_t finish = N * static_cast<size_t>(p + 1) / world;
        rowDispls[p] = static_cast<int>(begin);
        rowCounts[p] = static_cast<int>(finish - begin);
        displs[p] = static_cast<int>(begin * N);
        counts[p] = static_cast<int>((finish - begin) * N);
    }
    size_t rowStart = static_cast<size_t>(rowDispls[rank]);
    size_t rows = static_cast<size_t>(rowCounts[rank]);
    std::vector<double> A(rows * N), localC(rows * N);
    std::vector<double> C(rank == 0 ? N * N : 0);
    initializeBlock(A, N, rowStart, rows);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, localC, N, rows, rowStart);
    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localMs = duration.count(), maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) { MPI_Finalize(); return 0; }
    printf("Computation time: %lld ms\n", maxMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> fullA(N * N);
        initializeBlock(fullA, N, 0, N);
        bool valid = validateResult(fullA, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
