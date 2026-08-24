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

#define TILE_SIZE 16

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// CUDA tiled matrix multiplication kernel
// A is localRows x N (local portion), B is N x N (full), C is localRows x N (local portion)
__global__ void __launch_bounds__(TILE_SIZE * TILE_SIZE)
matmulCudaKernel(const double* __restrict__ A,
                  const double* __restrict__ B,
                  double* __restrict__ C,
                  const int N, const int localRows) {
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE_SIZE + threadIdx.x;
        const int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < localRows && aCol < N) ? A[row * N + aCol] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < localRows && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    const int n = static_cast<int>(N);
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool allValid = true;

    #pragma omp parallel for schedule(static) reduction(&&:allValid)
    for (int pi = 0; pi < 5; ++pi) {
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
                allValid = false;
            }
        }
    }

    return allValid;
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

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (same on all ranks)
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Setup CUDA device (one GPU per rank, round-robin if fewer GPUs)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Compute 1D row distribution across MPI ranks
    const int nInt = static_cast<int>(N);
    const int baseRows = nInt / nprocs;
    const int extraRows = nInt % nprocs;

    int localRows, rowOffset;
    if (rank < extraRows) {
        localRows = baseRows + 1;
        rowOffset = rank * (baseRows + 1);
    } else {
        localRows = baseRows;
        rowOffset = extraRows * (baseRows + 1) + (rank - extraRows) * baseRows;
    }

    // Each MPI process initializes its own rows of A (OpenMP parallel, deterministic)
    std::vector<double> localA(static_cast<size_t>(localRows) * N);
    std::vector<double> localC(static_cast<size_t>(localRows) * N, 0.0);
    {
        const int lr = localRows;
        const int ro = rowOffset;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < lr; ++i) {
            for (int j = 0; j < nInt; ++j) {
                localA[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, ro + i, j);
            }
        }
    }

    // Rank 0 initializes full B matrix (OpenMP parallel)
    std::vector<double> B(static_cast<size_t>(N) * N);
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(B, N);
    }

    // Broadcast B to all processes
    MPI_Bcast(B.data(), nInt * nInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 keeps full A for validation if needed
    std::vector<double> A_full;
    if (rank == 0 && validate) {
        A_full.resize(static_cast<size_t>(N) * N);
        initMatrix(A_full, N);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // GPU computation on each rank
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    if (localRows > 0) {
        // Allocate GPU device memory
        CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(localRows) * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(N) * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(localRows) * N * sizeof(double)));

        // Copy input matrices to GPU
        CUDA_CHECK(cudaMemcpy(d_A, localA.data(),
                              static_cast<size_t>(localRows) * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(),
                              static_cast<size_t>(N) * N * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Launch tiled CUDA kernel
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((nInt + TILE_SIZE - 1) / TILE_SIZE,
                  (localRows + TILE_SIZE - 1) / TILE_SIZE);
        matmulCudaKernel<<<grid, block>>>(d_A, d_B, d_C, nInt, localRows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(localC.data(), d_C,
                              static_cast<size_t>(localRows) * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Synchronize all processes after computation
    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    double localDuration = endTime - startTime;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Build Gatherv parameters for collecting results
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int rows = (r < extraRows) ? (baseRows + 1) : baseRows;
        int offset = (r < extraRows) ? (r * (baseRows + 1))
                                     : (extraRows * (baseRows + 1) + (r - extraRows) * baseRows);
        recvcounts[r] = rows * nInt;
        displs[r] = offset * nInt;
    }

    // Gather all results to rank 0
    std::vector<double> C;
    if (rank == 0) {
        C.resize(static_cast<size_t>(N) * N);
    }

    MPI_Gatherv(localC.data(), localRows * nInt, MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 reports results
    int exitCode = 0;
    if (rank == 0) {
        double duration_ms = maxDuration * 1000.0;
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        double gflops = (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Broadcast exit code to all processes
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup GPU resources
    if (d_A) CUDA_CHECK(cudaFree(d_A));
    if (d_B) CUDA_CHECK(cudaFree(d_B));
    if (d_C) CUDA_CHECK(cudaFree(d_C));

    MPI_Finalize();
    return exitCode;
}
