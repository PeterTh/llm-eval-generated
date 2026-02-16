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

// CUDA tiled kernel for double-precision matmul for local blocks
constexpr int TILE_DIM = 16;

__global__ void matmul_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                              int N, int A_rows) {
    __shared__ double sA[TILE_DIM][TILE_DIM];
    __shared__ double sB[TILE_DIM][TILE_DIM];

    int blockCol = blockIdx.x;
    int blockRow = blockIdx.y;

    int row = blockRow * TILE_DIM + threadIdx.y; // local row index in A (0..A_rows-1)
    int col = blockCol * TILE_DIM + threadIdx.x; // global column index

    double sum = 0.0;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        // Load A tile (A has A_rows rows)
        if (row < A_rows && aCol < N) sA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else sA[threadIdx.y][threadIdx.x] = 0.0;

        // Load B tile
        if (bRow < N && col < N) sB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < A_rows && col < N) C[row * N + col] = sum;
}

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

void initLocalMatrix(std::vector<double>& mat, const size_t N, const size_t rowOffset) {
    // rowOffset is the global start row index for this local matrix
    const size_t rows = mat.size() / N;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        size_t gi = rowOffset + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, gi, j);
        }
    }
}

bool validateResultDistributed(const std::vector<double>& A_full, const std::vector<double>& B_full,
                               const std::vector<double>& C_full, const size_t N) {
    // Reuse existing validation but parallelize checks with OpenMP
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            #pragma omp parallel for reduction(+:expected)
            for (size_t k = 0; k < N; ++k) {
                expected += A_full[i * N + k] * B_full[k * N + j];
            }

            const double actual = C_full[i * N + j];
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

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI+OpenMP+CUDA)\n");
        printf("World size: %d\n", world_size);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine rows per rank
    size_t baseRows = N / world_size;
    size_t remainder = N % world_size;
    size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t rowOffset = baseRows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    // Allocate local A and C, and full B on all ranks
    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);
    std::vector<double> B_full;
    if (rank == 0) B_full.resize(N * N);
    else B_full.resize(N * N); // allocate on all ranks for simplicity

    // Initialize local A and B (B will be filled on rank 0 then broadcast)
    initLocalMatrix(A_local, N, rowOffset);
    if (rank == 0) initLocalMatrix(B_full, N, 0);

    // Broadcast B to all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(B_full.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Warm-up CUDA device and create device buffers
    int deviceCount = 0;
    cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
    checkCuda(cerr, "cudaGetDeviceCount");
    int deviceId = rank % (deviceCount > 0 ? deviceCount : 1);
    if (deviceCount > 0) {
        checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");
    }

    // Allocate device memory
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytesA = localRows * N * sizeof(double);
    size_t bytesB = N * N * sizeof(double);
    size_t bytesC = localRows * N * sizeof(double);

    checkCuda(cudaMalloc(&dA, bytesA), "cudaMalloc dA");
    checkCuda(cudaMalloc(&dB, bytesB), "cudaMalloc dB");
    checkCuda(cudaMalloc(&dC, bytesC), "cudaMalloc dC");

    // Copy data to device
    checkCuda(cudaMemcpy(dA, A_local.data(), bytesA, cudaMemcpyHostToDevice), "cudaMemcpy A to dA");
    checkCuda(cudaMemcpy(dB, B_full.data(), bytesB, cudaMemcpyHostToDevice), "cudaMemcpy B to dB");

    // Synchronize and time the distributed multiply
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Launch kernel
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (localRows + TILE_DIM - 1) / TILE_DIM);
    matmul_kernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), static_cast<int>(localRows));
    checkCuda(cudaGetLastError(), "matmul_kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Copy result back
    checkCuda(cudaMemcpy(C_local.data(), dC, bytesC, cudaMemcpyDeviceToHost), "cudaMemcpy dC to C_local");

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Gather results to rank 0
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        size_t rows_r = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(rows_r * N);
        displs[r] = static_cast<int>((baseRows * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder)) * N);
    }

    std::vector<double> C_full;
    if (rank == 0) C_full.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                C_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double duration = t1 - t0;
        printf("Computation time: %.3f s\n", duration);
        double gflops = (2.0 * N * N * N) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(C_full, "MatrixC");

        if (validate) {
            printf("Validating result...\n");
            // Reconstruct full A on rank 0 for validation and run distributed validator
            std::vector<double> A_full(N * N);
            initLocalMatrix(A_full, N, 0);
            bool ok = validateResultDistributed(A_full, B_full, C_full, N);
            if (ok) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
