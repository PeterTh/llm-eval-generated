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
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [row0, row0 + rows) of the N x N matrix directly on the device
__global__ void initKernel(double* __restrict__ mat, const size_t N, const size_t row0, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        mat[idx] = getPseudoRndValue(N, row0 + idx / N, idx % N);
    }
}

// C (rows x N) = A (rows x N) * B (N x N). Register-tiled; every element accumulates
// over k in increasing order with FMA, matching the sequential reference.
constexpr int TILE = 64;
constexpr int BK = 16;

__global__ void __launch_bounds__(256) matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                                                    double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE];

    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int t = threadIdx.x;
    const size_t rowBase = (size_t)blockIdx.y * TILE;
    const size_t colBase = (size_t)blockIdx.x * TILE;

    double acc[4][4] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = (t >> 4) + 16 * q;
            const int kk = t & 15;
            const size_t gr = rowBase + r, gk = k0 + kk;
            As[kk][r] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
            const int kb = (t >> 6) + 4 * q;
            const int c = t & 63;
            const size_t gkb = k0 + kb, gc = colBase + c;
            Bs[kb][c] = (gkb < N && gc < N) ? B[gkb * N + gc] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t gr = rowBase + ty + 16 * i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t gc = colBase + tx + 16 * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Simple validation: compute a single element and compare
// A and B are regenerated on the host from the initialization formula (only the needed elements).
bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool ok = true;

#pragma omp parallel for collapse(2) schedule(dynamic) reduction(&& : ok)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
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

// Send/receive in chunks so that counts always fit in an int
static void sendDoubles(const double* p, size_t n, int dest) {
    const size_t chunk = 1u << 28;
    for (size_t off = 0; off < n; off += chunk)
        MPI_Send(p + off, (int)std::min(chunk, n - off), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}
static void recvDoubles(double* p, size_t n, int src) {
    const size_t chunk = 1u << 28;
    for (size_t off = 0; off < n; off += chunk)
        MPI_Recv(p + off, (int)std::min(chunk, n - off), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
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

    // Bind each rank to a GPU of its node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
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

    // Row-block distribution of A and C; B is replicated (generated on each GPU, no communication)
    auto rowStart = [&](int r) { return N * (size_t)r / (size_t)size; };
    const size_t row0 = rowStart(rank);
    const size_t rows = rowStart(rank + 1) - row0;

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(N * N, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(rows * N, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, std::max<size_t>(rows * N, 1) * sizeof(double)));
    initKernel<<<4096, 256>>>(dB, N, 0, N);
    initKernel<<<4096, 256>>>(dA, N, row0, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (rows > 0) {
        dim3 grid((unsigned)((N + TILE - 1) / TILE), (unsigned)((rows + TILE - 1) / TILE));
        matmulKernel<<<grid, 256>>>(dA, dB, dC, rows, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int status = 0;
    if (printResults || validate) {
        // Collect C on rank 0
        if (rank == 0) {
            std::vector<double> C(N * N);
            if (rows > 0) CUDA_CHECK(cudaMemcpy(C.data(), dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
            for (int r = 1; r < size; ++r) {
                const size_t n = (rowStart(r + 1) - rowStart(r)) * N;
                recvDoubles(C.data() + rowStart(r) * N, n, r);
            }
            if (printResults) print_results(C, "MatrixC");
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(C, N)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        } else if (rows > 0) {
            std::vector<double> Cl(rows * N);
            CUDA_CHECK(cudaMemcpy(Cl.data(), dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
            sendDoubles(Cl.data(), Cl.size(), 0);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    MPI_Finalize();
    return status;
}
