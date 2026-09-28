#include <algorithm>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

constexpr int TILE = 32;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fills `rowCount` rows of an NxN matrix starting at global row `rowStart`.
void initMatrix(std::vector<double>& mat, const size_t N, const size_t rowStart, const size_t rowCount) {
    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < rowCount; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled shared-memory matrix multiply kernel. Computes rowsLocal rows of C
// (a horizontal slab of the global result) from a matching slab of A and the
// full B matrix.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, size_t N, size_t rowsLocal) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const size_t row = blockIdx.y * static_cast<size_t>(TILE) + threadIdx.y;
    const size_t col = blockIdx.x * static_cast<size_t>(TILE) + threadIdx.x;

    double sum = 0.0;
    const size_t numTiles = (N + TILE - 1) / TILE;

    for (size_t t = 0; t < numTiles; ++t) {
        const size_t aCol = t * TILE + threadIdx.x;
        const size_t bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < rowsLocal && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rowsLocal && col < N) {
        C[row * N + col] = sum;
    }
}

// Distributes the N rows of C evenly (with remainder) across `size` MPI ranks.
void computeRowRange(const size_t N, const int size, const int rank, size_t& rowStart, size_t& rowCount) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    rowCount = base + (r < rem ? 1 : 0);
    rowStart = r * base + std::min(r, rem);
}

// Simple validation: compute a single element and compare against the values
// implied by the pseudo-random generator (equivalent to recomputing A and B).
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank under mpirun)
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

    // Bind this rank to a GPU (round-robin across the locally visible devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices visible: %d\n", size, deviceCount);
        printf("Initializing matrices...\n");
    }

    // Row-block decomposition of the output/A matrix across MPI ranks
    size_t rowStart = 0, rowCount = 0;
    computeRowRange(N, size, rank, rowStart, rowCount);

    // Each rank independently (re)generates the data it needs: its local
    // slab of A and the full B matrix. Both are pure functions of (N, i, j),
    // so this avoids any communication while producing bit-identical values
    // to the original single-process initMatrix.
    std::vector<double> A_local(rowCount * N);
    std::vector<double> B_full(N * N);
    std::vector<double> C_local(rowCount * N);

    initMatrix(A_local, N, rowStart, rowCount);
    initMatrix(B_full, N, 0, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, rowCount * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, rowCount * N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dA, A_local.data(), rowCount * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_full.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    if (rowCount > 0) {
        dim3 block(TILE, TILE);
        dim3 grid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                   static_cast<unsigned int>((rowCount + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, rowCount);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(C_local.data(), dC, rowCount * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed result onto rank 0 for reporting/validation
    std::vector<int> recvCounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t rs, rc;
        computeRowRange(N, size, r, rs, rc);
        recvCounts[r] = static_cast<int>(rc * N);
        displs[r] = static_cast<int>(rs * N);
    }

    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(rowCount * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (globalDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
