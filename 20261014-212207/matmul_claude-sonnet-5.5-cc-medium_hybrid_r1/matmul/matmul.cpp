#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowOffset, rowOffset+rows) of an N x N matrix directly on the device
__global__ void initKernel(double* mat, const size_t N, const size_t rowOffset, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / N, j = idx % N;
        mat[idx] = getPseudoRndValue(N, i + rowOffset, j);
    }
}

constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;

// C (rows x N) = A (rows x N) * B (N x N). Each thread owns a 4x4 register tile and
// accumulates over k in increasing order with unfused mul+add, so results are bit-identical
// to the serial code.
__global__ void __launch_bounds__(256) matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                                                    double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const size_t row0 = (size_t)blockIdx.y * BM;
    const size_t col0 = (size_t)blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int idx = threadIdx.x + s * 256;
            {
                const int ar = idx / BK, ac = idx % BK;
                const size_t gr = row0 + ar, gc = k0 + ac;
                As[ac][ar] = (gr < rows && gc < N) ? A[gr * N + gc] : 0.0;
            }
            {
                const int br = idx / BN, bc = idx % BN;
                const size_t gr = k0 + br, gc = col0 + bc;
                Bs[br][bc] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t gr = row0 + ty * TM + i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gc = col0 + tx + 16 * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Simple validation: compute a single element and compare (A and B regenerated on the host)
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool ok = true;

#pragma omp parallel for collapse(2) schedule(dynamic) reduction(&& : ok)
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
#pragma omp critical
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                ok = false;
            }
        }
    }
    return ok;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind each rank to a GPU on its node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // 1D row decomposition of A and C; B is needed in full by every rank.
    // Since the inputs are generated analytically, each rank builds its own data on its GPU
    // and no input communication is needed.
    std::vector<size_t> rowStart(size + 1);
    for (int r = 0; r <= size; ++r) rowStart[r] = N / size * r + std::min<size_t>(r, N % size);
    const size_t myOff = rowStart[rank], myRows = rowStart[rank + 1] - myOff;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    if (myRows) {
        CUDA_CHECK(cudaMalloc(&dA, myRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, myRows * N * sizeof(double)));
        initKernel<<<4096, 256>>>(dA, N, myOff, myRows);
    }
    initKernel<<<4096, 256>>>(dB, N, 0, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (myRows) {
        dim3 grid((N + BN - 1) / BN, (myRows + BM - 1) / BM);
        matmulKernel<<<grid, 256>>>(dA, dB, dC, myRows, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int rc = 0;
    if (printResults || validate) {
        // Gather C on rank 0 in bounded chunks (avoids MPI int count overflow)
        constexpr size_t chunkElems = size_t(1) << 26;
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        double* stage = nullptr;
        if (rank != 0 && myRows) CUDA_CHECK(cudaMallocHost(&stage, std::min(chunkElems, myRows * N) * sizeof(double)));
        const size_t rowsPerChunk = std::max<size_t>(1, chunkElems / std::max<size_t>(N, 1));

        if (rank == 0) {
            if (myRows) CUDA_CHECK(cudaMemcpy(C.data(), dC, myRows * N * sizeof(double), cudaMemcpyDeviceToHost));
            for (int r = 1; r < size; ++r) {
                for (size_t row = rowStart[r]; row < rowStart[r + 1]; row += rowsPerChunk) {
                    const size_t nr = std::min(rowsPerChunk, rowStart[r + 1] - row);
                    MPI_Recv(C.data() + row * N, (int)(nr * N), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            for (size_t row = 0; row < myRows; row += rowsPerChunk) {
                const size_t nr = std::min(rowsPerChunk, myRows - row);
                CUDA_CHECK(cudaMemcpy(stage, dC + row * N, nr * N * sizeof(double), cudaMemcpyDeviceToHost));
                MPI_Send(stage, (int)(nr * N), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            if (stage) cudaFreeHost(stage);
        }

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(C, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(C, N);
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
                rc = valid ? 0 : 1;
            }
        }
    }

    if (dA) cudaFree(dA);
    if (dC) cudaFree(dC);
    cudaFree(dB);
    MPI_Finalize();
    return rc;
}
