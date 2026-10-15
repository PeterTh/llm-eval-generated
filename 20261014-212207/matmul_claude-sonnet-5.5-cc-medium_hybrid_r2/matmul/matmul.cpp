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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [row0, row0 + rows) of the N x N matrix on the GPU
__global__ void initKernel(double* mat, const size_t N, const size_t row0, const size_t rows) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j < N && i < rows) mat[i * N + j] = getPseudoRndValue(N, row0 + i, j);
}

// Register-blocked shared-memory kernel. Each output element is accumulated with
// k in ascending order using FMA, matching the sequential reference loop.
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

__global__ void __launch_bounds__(256)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // A tile: BM x BK (store transposed); 4 elements per thread
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = threadIdx.x + l * 256;
            const int m = idx >> 3, kk = idx & 7;
            const size_t gr = rowBase + m, gk = k0 + kk;
            As[kk][m] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
        }
        // B tile: BK x BN
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = threadIdx.x + l * 256;
            const int kk = idx >> 7, n = idx & 127;
            const size_t gk = k0 + kk, gc = colBase + n;
            Bs[kk][n] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t gr = rowBase + ty + 16 * i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gc = colBase + tx + 16 * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Simple validation: compute a single element and compare.
// Only rows/columns 0..4 of A and B are needed; they are regenerated on the host.
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    int ok[25];
    double expectedV[25];

#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < 25; ++p) {
        const size_t i = checkPoints[p / 5] % N;
        const size_t j = checkPoints[p % 5] % N;
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) {
            expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
        }
        const double actual = C[i * N + j];
        const double relError = std::abs((actual - expected) / (expected + 1e-10));
        expectedV[p] = expected;
        ok[p] = !(relError > 1e-6);
    }

    for (int p = 0; p < 25; ++p) {
        if (!ok[p]) {
            const size_t i = checkPoints[p / 5] % N;
            const size_t j = checkPoints[p % 5] % N;
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expectedV[p]) / (expectedV[p] + 1e-10));
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                   i, j, expectedV[p], actual, relError);
            return false;
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
    int rank, nranks;
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

    // Bind each rank to a GPU (round-robin over the GPUs of its node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // 1D row decomposition of A and C; B is generated in full on every GPU
    // (deterministic generator, so no communication is needed for initialization).
    std::vector<size_t> rowStart(nranks + 1);
    for (int r = 0; r <= nranks; ++r) rowStart[r] = N / nranks * r + std::min<size_t>(r, N % nranks);
    const size_t row0 = rowStart[rank];
    const size_t rows = rowStart[rank + 1] - row0;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    if (rows > 0) {
        CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));
    }

    const bool gather = printResults || validate;
    std::vector<double> C;           // full result (rank 0, only if needed)
    std::vector<double> Cloc;        // local block (non-root ranks)
    double* hC = nullptr;
    if (rank == 0 && gather) {
        C.resize(N * N);
        hC = C.data();
    } else if (rows > 0 && gather) {
        Cloc.resize(rows * N);
        hC = Cloc.data();
    } else if (rows > 0) {
        // Result stays local when nobody needs it, but still copy it back to host
        Cloc.resize(rows * N);
        hC = Cloc.data();
    }

    {
        dim3 blk(256);
        dim3 gB((N + 255) / 256, N);
        initKernel<<<gB, blk>>>(dB, N, 0, N);
        if (rows > 0) {
            dim3 gA((N + 255) / 256, rows);
            initKernel<<<gA, blk>>>(dA, N, row0, rows);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (rows > 0) {
        dim3 grid((N + BN - 1) / BN, (rows + BM - 1) / BM);
        matmulKernel<<<grid, 256>>>(dA, dB, dC, rows, N);
        CUDA_CHECK(cudaGetLastError());
        // For rank 0 with the full result, hC already points at the right offset (row 0)
        CUDA_CHECK(cudaMemcpy(hC, dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (global_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the result on rank 0 (row blocks, chunked to stay within MPI int counts)
    if (gather) {
        constexpr size_t chunk = size_t(1) << 26;
        if (rank == 0) {
            for (int r = 1; r < nranks; ++r) {
                const size_t total = (rowStart[r + 1] - rowStart[r]) * N;
                double* dst = C.data() + rowStart[r] * N;
                for (size_t off = 0; off < total; off += chunk) {
                    MPI_Recv(dst + off, (int)std::min(chunk, total - off), MPI_DOUBLE, r, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            const size_t total = rows * N;
            for (size_t off = 0; off < total; off += chunk) {
                MPI_Send(Cloc.data() + off, (int)std::min(chunk, total - off), MPI_DOUBLE, 0, 0,
                         MPI_COMM_WORLD);
            }
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    int status = 0;
    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
