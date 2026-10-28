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

// Abort the whole MPI job on any CUDA runtime error (avoids hangs in collectives).
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Host-parallel (OpenMP) initialization; each element is independent.
void initMatrix(std::vector<double>& mat, const size_t N, const int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: C[rows x N] = A[rows x N] * B[N x N], row-major, double precision.
//
// Shared-memory tiling with 4x4 register micro-tiles per thread. Every output
// element accumulates its k-terms strictly in ascending order, reproducing the
// arithmetic of the original sequential loop bit-for-bit:
//   - each product is rounded and then added (no contraction), and
//   - the final k-term is fused (FMA) when N is odd, as the host compiler
//     does for the scalar tail of the original loop.
// ---------------------------------------------------------------------------
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;
constexpr int BLOCK_X = TILE_N / THREAD_N;  // 16
constexpr int BLOCK_Y = TILE_M / THREAD_M;  // 16
constexpr int NTHREADS = BLOCK_X * BLOCK_Y; // 256

__global__ void __launch_bounds__(NTHREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[TILE_M][TILE_K + 1];
    __shared__ double Bs[TILE_K][TILE_N + 1];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE_M;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE_N;

    double acc[THREAD_M][THREAD_N];
#pragma unroll
    for (int m = 0; m < THREAD_M; ++m)
#pragma unroll
        for (int n = 0; n < THREAD_N; ++n) acc[m][n] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += TILE_K) {
        for (int t = tid; t < TILE_M * TILE_K; t += NTHREADS) {
            const int r = t / TILE_K;
            const int c = t % TILE_K;
            const size_t gr = rowBase + r;
            const size_t gk = k0 + c;
            As[r][c] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
        }
        for (int t = tid; t < TILE_K * TILE_N; t += NTHREADS) {
            const int r = t / TILE_N;
            const int c = t % TILE_N;
            const size_t gk = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

        const size_t remaining = N - k0;
        const int kmax = remaining < static_cast<size_t>(TILE_K) ? static_cast<int>(remaining) : TILE_K;
        // Odd N: the original's scalar tail (k = N-1) is fused; this tile holds that term iff it is the last one.
        const bool fuseLast = (N & 1) && (remaining <= static_cast<size_t>(TILE_K));
        for (int kk = 0; kk < kmax; ++kk) {
            double a[THREAD_M];
            double b[THREAD_N];
#pragma unroll
            for (int m = 0; m < THREAD_M; ++m) a[m] = As[threadIdx.y * THREAD_M + m][kk];
#pragma unroll
            for (int n = 0; n < THREAD_N; ++n) b[n] = Bs[kk][threadIdx.x * THREAD_N + n];
            if (fuseLast && kk == kmax - 1) {
#pragma unroll
                for (int m = 0; m < THREAD_M; ++m)
#pragma unroll
                    for (int n = 0; n < THREAD_N; ++n) acc[m][n] = __fma_rn(a[m], b[n], acc[m][n]);
            } else {
#pragma unroll
                for (int m = 0; m < THREAD_M; ++m)
#pragma unroll
                    for (int n = 0; n < THREAD_N; ++n)
                        acc[m][n] = __dadd_rn(acc[m][n], __dmul_rn(a[m], b[n]));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        const size_t gr = rowBase + threadIdx.y * THREAD_M + m;
        if (gr >= rows) continue;
#pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
            const size_t gc = colBase + threadIdx.x * THREAD_N + n;
            if (gc < N) C[gr * N + gc] = acc[m][n];
        }
    }
}

// Multiply a block of `rows` rows of A (full B) on one GPU; result written to hC.
void multiplyOnDevice(const int device, const double* hA, const double* hB, double* hC,
                      const size_t rows, const size_t N) {
    if (rows == 0 || N == 0) return;

    CUDA_CHECK(cudaSetDevice(device));

    const size_t aBytes = rows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, aBytes));
    CUDA_CHECK(cudaMalloc(&dB, bBytes));
    CUDA_CHECK(cudaMalloc(&dC, aBytes));

    CUDA_CHECK(cudaMemcpy(dA, hA, aBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bBytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned int>((N + TILE_N - 1) / TILE_N),
                    static_cast<unsigned int>((rows + TILE_M - 1) / TILE_M));
    matmulKernel<<<grid, block>>>(dA, dB, dC, rows, N);
    CUDA_CHECK(cudaGetLastError());

    // Synchronous D2H copy also waits for the kernel to finish.
    CUDA_CHECK(cudaMemcpy(hC, dC, aBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool exitEarly = false;
    int exitCode = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            exitEarly = true;
            break;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            exitEarly = true;
            break;
        }
    }

    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }
    
    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Topology: local (shared-memory) ranks and GPU assignment ----
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Each local rank drives a set of GPUs. If there are at least as many ranks as GPUs,
    // ranks share GPUs round-robin; otherwise GPUs are partitioned among local ranks.
    std::vector<int> myDevices;
    if (localSize >= numDevices) {
        myDevices.push_back(localRank % numDevices);
    } else {
        for (int d = 0; d < numDevices; ++d) {
            if (d % localSize == localRank) myDevices.push_back(d);
        }
    }
    const int myDevCount = static_cast<int>(myDevices.size());

    // Row partition across ranks, weighted by the number of GPUs each rank drives.
    std::vector<int> devCounts(worldSize);
    MPI_Allgather(&myDevCount, 1, MPI_INT, devCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    size_t totalWeight = 0;
    for (int r = 0; r < worldSize; ++r) totalWeight += static_cast<size_t>(devCounts[r]);
    std::vector<size_t> rowBounds(worldSize + 1, 0);
    {
        size_t prefix = 0;
        for (int r = 0; r < worldSize; ++r) {
            prefix += static_cast<size_t>(devCounts[r]);
            rowBounds[r + 1] = N * prefix / totalWeight;
        }
    }
    const size_t myRowBegin = rowBounds[worldRank];
    const size_t myRows = rowBounds[worldRank + 1] - myRowBegin;

    // Host threads: share the node's cores among local ranks.
    const int hostThreads = std::max(1, omp_get_num_procs() / localSize);

    // Allocate matrices (every rank deterministically generates the full inputs,
    // so no input communication is needed).
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    if (worldRank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, hostThreads);
    initMatrix(B, N, hostThreads);

    // Pin host buffers (read by all local GPUs) so H2D copies run at full PCIe speed.
    // Registration and context creation are done before timing so they are not measured.
    if (N > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), N * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterDefault));
    }

    // Create CUDA contexts before timing so context setup is not measured.
#pragma omp parallel for num_threads(myDevCount) schedule(static, 1)
    for (int t = 0; t < myDevCount; ++t) {
        CUDA_CHECK(cudaSetDevice(myDevices[t]));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // Perform matrix multiplication
    if (worldRank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Each local GPU computes a contiguous sub-block of this rank's rows.
#pragma omp parallel for num_threads(myDevCount) schedule(static, 1)
    for (int t = 0; t < myDevCount; ++t) {
        const size_t g0 = myRowBegin + myRows * static_cast<size_t>(t) / myDevCount;
        const size_t g1 = myRowBegin + myRows * static_cast<size_t>(t + 1) / myDevCount;
        multiplyOnDevice(myDevices[t], A.data() + g0 * N, B.data(), C.data() + g0 * N, g1 - g0, N);
    }

    // Gather the full C on rank 0 (row blocks are contiguous and ordered by rank).
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);
    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = static_cast<int>(rowBounds[r + 1] - rowBounds[r]);
        displs[r] = static_cast<int>(rowBounds[r]);
    }
    if (worldRank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), counts.data(), displs.data(), rowType,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(C.data() + myRowBegin * N, counts[worldRank], rowType, nullptr,
                    counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (worldRank == 0) {
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
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (N > 0) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
        CUDA_CHECK(cudaHostUnregister(B.data()));
    }

    MPI_Type_free(&rowType);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
