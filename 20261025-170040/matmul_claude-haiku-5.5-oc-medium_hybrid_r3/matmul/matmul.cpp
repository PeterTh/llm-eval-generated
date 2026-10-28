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

// Hybrid parallelization:
//   MPI    - rows of A/C are block-distributed across ranks (one rank per GPU or GPU group).
//   OpenMP - one host thread per GPU owned by a rank; also parallelizes matrix initialization.
//   CUDA   - tiled double-precision GEMM kernel on each device.
//
// Every C(i,j) is still accumulated sequentially over k = 0..N-1 with the same
// rounding as the original build, so results are bit-identical to the serial code.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill rows [rowBegin, rowBegin + rowCount) of an N-column matrix (OpenMP parallel).
void initRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rowCount; ++r) {
        for (size_t j = 0; j < N; ++j) {
            mat[r * N + j] = getPseudoRndValue(N, rowBegin + r, j);
        }
    }
}

// Tiled GEMM: C[M x N] = A[M x N] * B[N x N], all row-major.
// Block tile 64x64, k-tile 16, 16x16 threads, each thread computes a 4x4 register tile.
// Each output element accumulates over k in ascending order (same as the serial reference).
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void __launch_bounds__(256)
matmulTiled(const double* __restrict__ A, const double* __restrict__ B,
            double* __restrict__ C, const int M, const int N) {
    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            acc[m][n] = 0.0;

    for (int kt = 0; kt < N; kt += BK) {
        const int kl = min(BK, N - kt);

        for (int idx = tid; idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int c = idx % BK;
            const int gr = rowBase + r;
            As[r][c] = (gr < M && c < kl)
                           ? A[static_cast<size_t>(gr) * N + kt + c]
                           : 0.0;
        }
        for (int idx = tid; idx < BK * BN; idx += 256) {
            const int r = idx / BN;
            const int c = idx % BN;
            const int gc = colBase + c;
            Bs[r][c] = (r < kl && gc < N)
                           ? B[static_cast<size_t>(kt + r) * N + gc]
                           : 0.0;
        }
        __syncthreads();

        // The reference build accumulates with separately rounded multiply and add
        // (vectorized loop); only an odd-N trailing term is fused. Mirror that exactly.
        const bool fuseLast = (N & 1) && (kt + kl == N);
        const int kmain = fuseLast ? kl - 1 : kl;

        for (int kk = 0; kk < kmain; ++kk) {
            double a[TM];
            double b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[ty * TM + m][kk];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = Bs[kk][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] = __dadd_rn(acc[m][n], __dmul_rn(a[m], b[n]));
        }
        if (fuseLast) {
            double a[TM];
            double b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[ty * TM + m][kl - 1];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = Bs[kl - 1][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] = fma(a[m], b[n], acc[m][n]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int gr = rowBase + ty * TM + m;
        if (gr >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int gc = colBase + tx * TN + n;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[m][n];
        }
    }
}

// Block partition of `total` items over `parts` owners.
void partition(const size_t total, const int parts, const int idx,
               size_t& begin, size_t& count) {
    const size_t base = total / parts;
    const size_t rem = total % parts;
    const size_t u = static_cast<size_t>(idx);
    begin = u * base + std::min(u, rem);
    count = base + (u < rem ? 1 : 0);
}

// Ranks sharing a node split its GPUs; if ranks outnumber GPUs, ranks share round-robin.
std::vector<int> devicesForRank(const int localRank, const int localSize, const int numDevices) {
    std::vector<int> devs;
    if (localSize <= numDevices) {
        for (int d = localRank; d < numDevices; d += localSize) devs.push_back(d);
    } else {
        devs.push_back(localRank % numDevices);
    }
    return devs;
}

// Create CUDA contexts up front so context setup is not part of the timed region.
void warmUpDevices(const std::vector<int>& devs) {
    const int nd = static_cast<int>(devs.size());
    #pragma omp parallel for num_threads(nd) schedule(static, 1)
    for (int t = 0; t < nd; ++t) {
        CUDA_CHECK(cudaSetDevice(devs[t]));
        CUDA_CHECK(cudaFree(nullptr));
    }
}

// Compute the local rows of C (m x N) = hA (m x N) * hB (N x N) using all devices of this rank.
void multiplyLocalRows(const double* hA, const double* hB, double* hC,
                       const size_t m, const size_t N, const std::vector<int>& devs) {
    const int nd = static_cast<int>(devs.size());
    #pragma omp parallel for num_threads(nd) schedule(static, 1)
    for (int t = 0; t < nd; ++t) {
        size_t rowBegin = 0, rowCount = 0;
        partition(m, nd, t, rowBegin, rowCount);

        CUDA_CHECK(cudaSetDevice(devs[t]));
        if (rowCount == 0 || N == 0) continue;

        const size_t aBytes = rowCount * N * sizeof(double);
        const size_t bBytes = N * N * sizeof(double);

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, aBytes));
        CUDA_CHECK(cudaMalloc(&dB, bBytes));
        CUDA_CHECK(cudaMalloc(&dC, aBytes));

        CUDA_CHECK(cudaMemcpy(dA, hA + rowBegin * N, aBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bBytes, cudaMemcpyHostToDevice));

        const dim3 block(BN / TN, BM / TM);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((rowCount + BM - 1) / BM));
        matmulTiled<<<grid, block>>>(dA, dB, dC, static_cast<int>(rowCount), static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());

        // Synchronous copy-back also waits for the kernel to finish.
        CUDA_CHECK(cudaMemcpy(hC + rowBegin * N, dC, aBytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
}

// Simple validation: compute a few elements on the host and compare.
// A and B are regenerated on demand (identical values to the initialized matrices).
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Node-local rank, used to map ranks onto GPUs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const std::vector<int> devs = devicesForRank(localRank, localSize, numDevices);

    // Row block owned by this rank.
    size_t rowBegin = 0, rowCount = 0;
    partition(N, size, rank, rowBegin, rowCount);

    // Allocate and initialize local data
    if (rank == 0) printf("Initializing matrices...\n");
    std::vector<double> A(rowCount * N);
    std::vector<double> B(N * N);
    std::vector<double> C(rowCount * N);
    initRows(A.data(), N, rowBegin, rowCount);
    initRows(B.data(), N, 0, N);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    warmUpDevices(devs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    multiplyLocalRows(A.data(), B.data(), C.data(), rowCount, N, devs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full C on rank 0 only when it is needed for output or validation.
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rb = 0, rc = 0;
            partition(N, size, r, rb, rc);
            counts[r] = static_cast<int>(rc * N);
            displs[r] = static_cast<int>(rb * N);
        }
        std::vector<double> full;
        if (rank == 0) full.resize(N * N);
        MPI_Gatherv(C.data(), static_cast<int>(rowCount * N), MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "MatrixC");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, N);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    // Keep the exit status consistent across all ranks.
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ret;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ret = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return ret;
}
