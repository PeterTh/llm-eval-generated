#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Tile size for the CUDA shared-memory matmul kernel.
constexpr int TILE_SIZE = 16;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_err));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize the full NxN matrix (used for the replicated B matrix and for
// recomputing A on rank 0 for validation/printing without extra MPI traffic).
void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize only the rows [rowStart, rowStart+rowsLocal) of an NxN matrix,
// storing them densely at the start of `mat`. Used so each MPI rank can
// build its slice of A locally (no communication needed since the values
// are a pure function of (N, i, j)).
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowStart,
                     const size_t rowsLocal) {
    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < rowsLocal; ++li) {
        const size_t i = rowStart + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled shared-memory matrix multiply: C_local = A_local * B
// A_local holds `rowsLocal` rows of the global A matrix, B is the full NxN
// matrix, and C_local receives the corresponding `rowsLocal` rows of C.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, int N, int rowsLocal) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE_SIZE + threadIdx.x;
        const int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < rowsLocal && aCol < N) ? A[static_cast<size_t>(row) * N + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsLocal && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool argError = false;

    // Parse command line arguments. mpirun replicates argv identically to
    // every rank, so each rank can parse independently without a broadcast.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            argError = true;
            break;
        }
    }

    if (argError) {
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA-capable devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices available: %d\n",
               numRanks, omp_get_max_threads(), deviceCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-block distribution of A/C across MPI ranks (as evenly as possible).
    std::vector<int> rowCounts(numRanks), rowOffsets(numRanks);
    const size_t base = N / static_cast<size_t>(numRanks);
    const size_t rem = N % static_cast<size_t>(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        rowCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
    }
    rowOffsets[0] = 0;
    for (int r = 1; r < numRanks; ++r) {
        rowOffsets[r] = rowOffsets[r - 1] + rowCounts[r - 1];
    }
    const size_t rowStart = static_cast<size_t>(rowOffsets[rank]);
    const size_t rowsLocal = static_cast<size_t>(rowCounts[rank]);

    // Each rank builds its own slice of A directly (no communication needed
    // since matrix entries are a pure function of N, i, j), and B is
    // initialized in full on every rank since it is needed in its entirety
    // for the local row-block multiply.
    std::vector<double> A_local(rowsLocal * N);
    std::vector<double> B(N * N);
    initMatrixRows(A_local, N, rowStart, rowsLocal);
    initMatrix(B, N);

    std::vector<double> C_local(rowsLocal * N);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, rowsLocal * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, rowsLocal * N * sizeof(double)));

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(dA, A_local.data(), rowsLocal * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

    if (rowsLocal > 0) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((static_cast<unsigned int>(N) + TILE_SIZE - 1) / TILE_SIZE,
                         (static_cast<unsigned int>(rowsLocal) + TILE_SIZE - 1) / TILE_SIZE);
        matmulKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N), static_cast<int>(rowsLocal));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(C_local.data(), dC, rowsLocal * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = duration.count();
    long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed C rows back onto rank 0 for reporting/validation.
    std::vector<int> recvCounts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        recvCounts[r] = rowCounts[r] * static_cast<int>(N);
        displs[r] = rowOffsets[r] * static_cast<int>(N);
    }
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }
    MPI_Gatherv(C_local.data(), static_cast<int>(rowsLocal * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    bool valid = true;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        const double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            // Recompute A in full locally (deterministic, OpenMP-parallel)
            // rather than gathering it over MPI.
            std::vector<double> A_full(N * N);
            initMatrix(A_full, N);
            valid = validateResult(A_full, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    int exitCode = 0;
    if (validate) {
        int validInt = (rank == 0) ? (valid ? 0 : 1) : 0;
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = validInt;
    }

    MPI_Finalize();
    return exitCode;
}
