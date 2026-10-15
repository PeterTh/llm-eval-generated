#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
                    __FILE__, __LINE__);                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill rows [row0, row0+rows) of the matrix on the device
__global__ void initKernel(double* mat, const size_t N, const size_t row0, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = row0 + idx / N;
        const size_t j = idx % N;
        mat[idx] = getPseudoRndValue(N, i, j);
    }
}

constexpr int TILE = 64;  // C tile per block
constexpr int BK = 16;    // k-slab depth
constexpr int TX = 16;    // threads per block dimension (each computes 4x4)

// C[rows x N] = A[rows x N] * B[N x N]; per-element k order is ascending (as in the original)
__global__ void __launch_bounds__(TX * TX)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N, const size_t rows) {
    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = (size_t)blockIdx.y * TILE;
    const size_t colBase = (size_t)blockIdx.x * TILE;

    double acc[4][4] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // load A tile (TILE x BK) transposed into As[k][i]
        for (int e = tid; e < TILE * BK; e += TX * TX) {
            const int kk = e % BK, ii = e / BK;
            const size_t r = rowBase + ii, k = k0 + kk;
            As[kk][ii] = (r < rows && k < N) ? A[r * N + k] : 0.0;
        }
        // load B tile (BK x TILE)
        for (int e = tid; e < TILE * BK; e += TX * TX) {
            const int jj = e % TILE, kk = e / TILE;
            const size_t k = k0 + kk, c = colBase + jj;
            Bs[kk][jj] = (k < N && c < N) ? B[k * N + c] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                a[x] = As[kk][ty + x * TX];
                b[x] = Bs[kk][tx + x * TX];
            }
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int y = 0; y < 4; ++y) acc[x][y] = __dadd_rn(acc[x][y], __dmul_rn(a[x], b[y]));
        }
        __syncthreads();
    }

#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const size_t r = rowBase + ty + x * TX;
        if (r >= rows) continue;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
            const size_t c = colBase + tx + y * TX;
            if (c < N) C[r * N + c] = acc[x][y];
        }
    }
}

// Simple validation: compute a single element and compare
// A and B elements are regenerated on the fly from the (deterministic) generator
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
    int rank, nprocs;
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU by node-local rank
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Row-block distribution of A and C; B is replicated (generated directly on each GPU)
    std::vector<size_t> rowStart(nprocs + 1);
    for (int r = 0; r <= nprocs; ++r) rowStart[r] = N * (size_t)r / nprocs;
    const size_t row0 = rowStart[rank];
    const size_t rows = rowStart[rank + 1] - row0;

    if (rank == 0) printf("Initializing matrices...\n");
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
    if (rows > 0) {
        CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));
    }
    initKernel<<<4096, 256>>>(dB, N, 0, N);
    if (rows > 0) initKernel<<<4096, 256>>>(dA, N, row0, rows);
    CUDA_CHECK(cudaGetLastError());

    // Host buffer for the gathered result (rank 0) / local block (others), pinned for fast copies
    std::vector<double> C;
    double* hC = nullptr;
    if (rank == 0) {
        C.resize(N * N);
        CUDA_CHECK(cudaHostRegister(C.data(), C.size() * sizeof(double), cudaHostRegisterPortable));
        hC = C.data();
    } else if (rows > 0) {
        CUDA_CHECK(cudaMallocHost(&hC, rows * N * sizeof(double)));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (rows > 0) {
        dim3 block(TX, TX);
        dim3 grid((unsigned)((N + TILE - 1) / TILE), (unsigned)((rows + TILE - 1) / TILE));
        matmulKernel<<<grid, block>>>(dA, dB, dC, N, rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hC, dC, rows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Gather row blocks on rank 0 (chunked to stay within MPI int counts)
    constexpr size_t CHUNK = size_t(1) << 28;
    if (rank == 0) {
        for (int r = 1; r < nprocs; ++r) {
            const size_t cnt = (rowStart[r + 1] - rowStart[r]) * N;
            double* dst = C.data() + rowStart[r] * N;
            for (size_t off = 0; off < cnt; off += CHUNK) {
                MPI_Recv(dst + off, (int)std::min(CHUNK, cnt - off), MPI_DOUBLE, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t off = 0; off < rows * N; off += CHUNK) {
            MPI_Send(hC + off, (int)std::min(CHUNK, rows * N - off), MPI_DOUBLE, 0, 0,
                     MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int rc = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
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
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
        CUDA_CHECK(cudaHostUnregister(C.data()));
    } else if (hC) {
        cudaFreeHost(hC);
    }

    cudaFree(dB);
    if (dA) cudaFree(dA);
    if (dC) cudaFree(dC);
    MPI_Finalize();
    return rc;
}
