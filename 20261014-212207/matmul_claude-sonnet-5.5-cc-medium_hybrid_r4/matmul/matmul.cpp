#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Generate rows [row0, row0 + rows) of the matrix directly on the device
__global__ void initKernel(double* mat, const size_t N, const size_t row0, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        mat[idx] = getPseudoRndValue(N, row0 + idx / N, idx % N);
    }
}

constexpr int BM = 64, BN = 64, BK = 16, TX = 16, TY = 16, RM = BM / TY, RN = BN / TX;

// C[rows x N] = A[rows x N] * B[N x N]; each element accumulates sequentially over k (same
// order as the reference implementation) using fused multiply-add.
__global__ void __launch_bounds__(TX * TY) matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[RM][RN];
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) acc[i][j] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // A tile: BM x BK, stored transposed
#pragma unroll
        for (int l = 0; l < (BM * BK) / (TX * TY); ++l) {
            const int e = tid + l * TX * TY;
            const int r = e / BK, c = e % BK;
            const size_t gr = rowBase + r, gc = k0 + c;
            As[c][r] = (gr < rows && gc < N) ? A[gr * N + gc] : 0.0;
        }
        // B tile: BK x BN
#pragma unroll
        for (int l = 0; l < (BK * BN) / (TX * TY); ++l) {
            const int e = tid + l * TX * TY;
            const int r = e / BN, c = e % BN;
            const size_t gr = k0 + r, gc = colBase + c;
            Bs[r][c] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[RM], b[RN];
#pragma unroll
            for (int i = 0; i < RM; ++i) a[i] = As[k][ty + TY * i];
#pragma unroll
            for (int j = 0; j < RN; ++j) b[j] = Bs[k][tx + TX * j];
#pragma unroll
            for (int i = 0; i < RM; ++i)
#pragma unroll
                for (int j = 0; j < RN; ++j) acc[i][j] = __fma_rn(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < RM; ++i) {
        const size_t gr = rowBase + ty + TY * i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < RN; ++j) {
            const size_t gc = colBase + tx + TX * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Validation (rank 0): recompute a few elements on the host from the generating function
bool validateResult(const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool ok = true;

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
#pragma omp parallel for reduction(+ : expected) schedule(static)
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                ok = false;
                return ok;
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
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank = 0;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // 1D row-block decomposition of A and C; B is generated redundantly on every device
    // (generation is cheaper than communicating it).
    std::vector<size_t> rowStart(nprocs + 1);
    for (int r = 0; r <= nprocs; ++r) rowStart[r] = (N / nprocs) * r + std::min<size_t>(r, N % nprocs);
    const size_t row0 = rowStart[rank];
    const size_t rows = rowStart[rank + 1] - row0;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    if (rows > 0) {
        CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));
    }
    initKernel<<<1024, 256>>>(dB, N, 0, N);
    if (rows > 0) initKernel<<<1024, 256>>>(dA, N, row0, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (rows > 0) {
        dim3 block(TX, TY);
        dim3 grid((N + BN - 1) / BN, (rows + BM - 1) / BM);
        matmulKernel<<<grid, block>>>(dA, dB, dC, rows, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather C on rank 0 only when needed
    std::vector<double> C;
    if (printResults || validate) {
        constexpr size_t chunk = size_t(1) << 27;  // elements per message
        if (rank == 0) {
            double* hC = nullptr;
            CUDA_CHECK(cudaMallocHost(&hC, N * N * sizeof(double)));
            if (rows > 0) CUDA_CHECK(cudaMemcpy(hC, dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
            for (int r = 1; r < nprocs; ++r) {
                const size_t total = (rowStart[r + 1] - rowStart[r]) * N;
                double* dst = hC + rowStart[r] * N;
                for (size_t off = 0; off < total; off += chunk) {
                    const int cnt = (int)std::min(chunk, total - off);
                    MPI_Recv(dst + off, cnt, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            C.resize(N * N);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N * N; ++i) C[i] = hC[i];
            CUDA_CHECK(cudaFreeHost(hC));
        } else if (rows > 0) {
            const size_t total = rows * N;
            double* hC = nullptr;
            CUDA_CHECK(cudaMallocHost(&hC, total * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(hC, dC, total * sizeof(double), cudaMemcpyDeviceToHost));
            for (size_t off = 0; off < total; off += chunk) {
                const int cnt = (int)std::min(chunk, total - off);
                MPI_Send(hC + off, cnt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            CUDA_CHECK(cudaFreeHost(hC));
        }
    }

    if (dA) cudaFree(dA);
    if (dC) cudaFree(dC);
    cudaFree(dB);

    int ret = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(C, N)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
