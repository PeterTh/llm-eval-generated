#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ============================================================================
// CUDA Kernel: Tiled Matrix Multiplication with Shared Memory
// ============================================================================
// Optimized block-based matmul using shared memory tiling.
// Each thread block computes a TILE_SIZE x TILE_SIZE tile of the output matrix.
// Tiles of A and B are loaded into shared memory for coalesced global memory
// access and reduced memory bandwidth pressure.
// ============================================================================
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int localM, int N) {
    constexpr int TILE_SIZE = 32;
    __shared__ double sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ double sharedB[TILE_SIZE][TILE_SIZE];

    // Global output coordinates for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Local tile coordinates
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    double sum = 0.0;

    // Iterate over tiles along the K dimension
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory with boundary checks
        if (row < localM && (t * TILE_SIZE + tx) < N) {
            sharedA[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0;
        }

        // Load tile of B into shared memory with boundary checks
        if (col < N && (t * TILE_SIZE + ty) < N) {
            sharedB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0;
        }

        __syncthreads();

        // Compute partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    // Write result to global memory (boundary-safe)
    if (row < localM && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Matrix Initialization (OpenMP Parallelized)
// ============================================================================
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

// ============================================================================
// Hybrid MPI + CUDA Matrix Multiplication
// ============================================================================
// Distribution: 1D row-wise block distribution of output matrix C across MPI
// ranks. Each MPI rank:
//   - Receives its assigned rows of A via MPI_Scatterv
//   - Receives full B via MPI_Bcast
//   - Computes its local block of C on GPU using tiled CUDA kernel
//   - Sends local C block back via MPI_Gatherv
// ============================================================================
void matrixMultiply(const std::vector<double>& A, std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Calculate row distribution: each rank gets ~N/numRanks rows
    int n = static_cast<int>(N);
    int baseRows = n / numRanks;
    int extraRows = n % numRanks;

    // Local dimensions for this rank
    int localM = baseRows + (rank < extraRows ? 1 : 0);

    // Local matrix storage (row-major, N columns)
    std::vector<double> localA(localM * N);
    std::vector<double> localC(localM * N, 0.0);

    // Scatterv: distribute rows of A to each rank
    {
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            int rrows = baseRows + (r < extraRows ? 1 : 0);
            counts[r] = rrows * n;
            displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
        }
        MPI_Scatterv(A.data(), counts.data(), displs.data(), MPI_DOUBLE,
                     localA.data(), localM * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Broadcast full B to all ranks
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // GPU computation
    if (localM > 0) {
        // Device pointers
        double *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, localM * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, localM * n * sizeof(double)));

        // Transfer input data to GPU
        CUDA_CHECK(cudaMemcpy(dA, localA.data(), localM * n * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double),
                              cudaMemcpyHostToDevice));

        // Launch tiled CUDA kernel
        constexpr int TILE = 32;
        dim3 blockDim(TILE, TILE);
        dim3 gridDim((n + TILE - 1) / TILE,
                     (localM + TILE - 1) / TILE);
        matmulKernel<<<gridDim, blockDim>>>(dA, dB, dC, localM, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Transfer result back to host
        CUDA_CHECK(cudaMemcpy(localC.data(), dC, localM * n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        // Cleanup GPU memory
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    // Gatherv: collect local C blocks into full C on root
    {
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            int rrows = baseRows + (r < extraRows ? 1 : 0);
            counts[r] = rrows * n;
            displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
        }
        MPI_Gatherv(localC.data(), localM * n, MPI_DOUBLE,
                    C.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation: compute a single element and compare
// ============================================================================
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, values must agree)
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

    // Print configuration (root only)
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Parallelization: Hybrid MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    // Initialize matrices (OpenMP parallelized) - all ranks initialize identically
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication (MPI + CUDA)
    if (rank == 0) printf("Computing matrix multiplication...\n");

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Report timing and performance (root only)
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (root only, C is gathered there)
    if (printResults && rank == 0) {
        print_results(C, "MatrixC");
    }

    // Validation (root only)
    if (validate && rank == 0) {
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

    MPI_Finalize();
    return 0;
}
