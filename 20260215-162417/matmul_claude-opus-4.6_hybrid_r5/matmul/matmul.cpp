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

#define TILE_SIZE 32

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                __LINE__, cudaGetErrorString(err));                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                 \
    }                                                                  \
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

// Tiled CUDA kernel with shared memory and padding to reduce bank conflicts
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int M, int N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < N) ? A[row * N + aCol] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Each MPI rank multiplies its local rows of A by full B on its GPU
void matrixMultiplyGPU(const double* localA, const double* B, double* localC,
                       size_t localRows, size_t N) {
    double *d_A, *d_B, *d_C;
    size_t sizeA = localRows * N * sizeof(double);
    size_t sizeB = N * N * sizeof(double);
    size_t sizeC = localRows * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, localA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((static_cast<unsigned>(N) + TILE_SIZE - 1) / TILE_SIZE,
              (static_cast<unsigned>(localRows) + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<<<grid, block>>>(d_A, d_B, d_C,
                                  static_cast<int>(localRows),
                                  static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(localC, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            #pragma omp parallel for reduction(+:expected)
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

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Assign GPU round-robin across ranks
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % numDevices));
    }

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               numProcs, omp_get_max_threads(), numDevices);
    }

    // Row distribution across MPI ranks
    size_t baseRows = N / numProcs;
    size_t extraRows = N % numProcs;
    size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);

    // Compute counts/displacements for Scatterv/Gatherv
    std::vector<int> sendcounts(numProcs), displs(numProcs);
    int offset = 0;
    for (int r = 0; r < numProcs; ++r) {
        size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        sendcounts[r] = static_cast<int>(rows * N);
        displs[r] = offset;
        offset += sendcounts[r];
    }

    // Allocate matrices
    std::vector<double> A, B(N * N), C;
    std::vector<double> localA(localRows * N);
    std::vector<double> localC(localRows * N, 0.0);

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);

        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Broadcast full B to all ranks
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A across ranks
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication on GPU
    if (rank == 0) printf("Computing matrix multiplication...\n");

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyGPU(localA.data(), B.data(), localC.data(), localRows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather result rows back to rank 0
    MPI_Gatherv(localC.data(), static_cast<int>(localRows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

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
