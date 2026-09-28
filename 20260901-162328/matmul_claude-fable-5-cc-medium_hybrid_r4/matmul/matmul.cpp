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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),   \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

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

constexpr int TILE = 32;

// Tiled shared-memory matmul kernel: computes C_local = A_local * B, where
// A_local holds `rows` consecutive rows of A and C_local the matching rows of C.
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t N, const size_t rows) {
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + ty;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;

    double sum = 0.0;
    for (size_t t = 0; t < N; t += TILE) {
        const size_t ka = t + tx;
        const size_t kb = t + ty;
        sA[ty][tx] = (row < rows && ka < N) ? A[row * N + ka] : 0.0;
        sB[ty][tx] = (kb < N && col < N) ? B[kb * N + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < rows && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid MPI + CUDA matrix multiply: each rank computes a contiguous block of
// rows of C on its GPU; the result is gathered on rank 0.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const int rank, const int nranks) {
    // Balanced row distribution across ranks
    const size_t base = N / nranks;
    const size_t rem = N % nranks;
    const size_t myRows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t rowStart =
        static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (myRows > 0) {
        CUDA_CHECK(cudaMalloc(&dA, myRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, myRows * N * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dA, A.data() + rowStart * N, myRows * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 block(TILE, TILE);
        const dim3 grid(static_cast<unsigned>((N + TILE - 1) / TILE),
                        static_cast<unsigned>((myRows + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, myRows);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(C.data() + rowStart * N, dC, myRows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    // Gather the row blocks of C on rank 0
    std::vector<int> counts(nranks), displs(nranks);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nranks; ++r) {
        const size_t rRows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        const size_t rStart = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
        counts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rStart * N);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : C.data() + rowStart * N,
                static_cast<int>(myRows * N), MPI_DOUBLE, C.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            MPI_Finalize();
            return 1;
        }
    }

    // Bind each rank to a GPU (round-robin over the local devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n", nranks, deviceCount,
               omp_get_max_threads());
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices (deterministic, so every rank computes its own copy)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
