#include <algorithm>
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

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA matrix multiplication.
//
// Decomposition:
//   * MPI: the rows of A (and hence of C) are split into contiguous blocks,
//     one block per rank. B is replicated (every rank generates it locally,
//     the initialization is a pure function of the indices, so no
//     communication is required).
//   * OpenMP: matrix initialization / validation on the host, and driving all
//     GPUs owned by a rank concurrently (one host thread per device).
//   * CUDA: the actual DGEMM, using a register-blocked, shared-memory tiled
//     kernel. A rank's row block is split again across the GPUs it owns.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                     \
        if (err_ != cudaSuccess) {                                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                       \
                    cudaGetErrorString(err_));                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                    \
        }                                                                                                    \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize the rows [rowBegin, rowEnd) of an NxN matrix into a compact buffer.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
#pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat.data() + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// --- CUDA kernel ------------------------------------------------------------

// Tile sizes: each block computes a BM x BN tile of C, each thread a TM x TN
// sub-tile held in registers. The per-thread elements are strided by BM/TM and
// BN/TN so that all shared memory accesses are bank-conflict free.
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 16;
static constexpr int TM = 4;
static constexpr int TN = 4;
static constexpr int THREADS = (BM / TM) * (BN / TN); // 256

// Computes C[0:M, 0:Ncols] = A[0:M, 0:K] * B[0:K, 0:Ncols]; A is stored with
// leading dimension K, B and C with leading dimension ld (they may be column
// strips of a larger matrix).
__global__ __launch_bounds__(THREADS) void dgemmKernel(const double* __restrict__ A, const double* __restrict__ B,
                                                       double* __restrict__ C, const int M, const int Ncols,
                                                       const int K, const int ld) {
    __shared__ double As[BK][BM]; // transposed A tile
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Element mapping of a thread inside the block tile (strided).
    const int tRow = tid / (BN / TN); // 0..15
    const int tCol = tid % (BN / TN); // 0..15

    // Cooperative load mapping.
    const int aLoadRow = tid / BK;    // 0..15, stride 16 -> covers BM = 64
    const int aLoadCol = tid % BK;    // 0..15
    const int bLoadRow = tid / BN;    // 0..3, stride 4 -> covers BK = 16
    const int bLoadCol = tid % BN;    // 0..63

    double acc[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            acc[m][n] = 0.0;
        }
    }

    for (int kt = 0; kt < K; kt += BK) {
#pragma unroll
        for (int r = 0; r < BM / (THREADS / BK); ++r) {
            const int row = aLoadRow + r * (THREADS / BK);
            const int gr = blockRow + row;
            const int gc = kt + aLoadCol;
            As[aLoadCol][row] = (gr < M && gc < K) ? A[static_cast<size_t>(gr) * K + gc] : 0.0;
        }
#pragma unroll
        for (int r = 0; r < BK / (THREADS / BN); ++r) {
            const int row = bLoadRow + r * (THREADS / BN);
            const int gr = kt + row;
            const int gc = blockCol + bLoadCol;
            Bs[row][bLoadCol] = (gr < K && gc < Ncols) ? B[static_cast<size_t>(gr) * ld + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double regA[TM];
            double regB[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                regA[m] = As[k][tRow + m * (BM / TM)];
            }
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                regB[n] = Bs[k][tCol + n * (BN / TN)];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = __dadd_rn(acc[m][n], __dmul_rn(regA[m], regB[n]));
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int gr = blockRow + tRow + m * (BM / TM);
        if (gr >= M) continue;
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int gc = blockCol + tCol + n * (BN / TN);
            if (gc < Ncols) {
                C[static_cast<size_t>(gr) * ld + gc] = acc[m][n];
            }
        }
    }
}

// Multiply the locally owned rows of A with the (replicated) B on all GPUs
// owned by this rank. localA holds localRows x N, localC receives localRows x N.
void matrixMultiplyLocal(const double* localA, const double* B, double* localC, const size_t N,
                         const size_t localRows, const std::vector<int>& devices) {
    if (localRows == 0 || devices.empty()) return;

    const int numDev = static_cast<int>(devices.size());

#pragma omp parallel for num_threads(numDev) schedule(static, 1)
    for (int d = 0; d < numDev; ++d) {
        const size_t begin = localRows * d / numDev;
        const size_t end = localRows * (d + 1) / numDev;
        const size_t rows = end - begin;
        if (rows == 0) continue;

        CUDA_CHECK(cudaSetDevice(devices[d]));

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * N * sizeof(double)));

        // The work is split into column strips of C so that the transfer of a
        // strip of B overlaps with the computation on the previous strip.
        constexpr int NUM_STREAMS = 4;
        const size_t stripWidth = ((N + NUM_STREAMS - 1) / NUM_STREAMS + BN - 1) / BN * BN;
        const int numStrips = static_cast<int>((N + stripWidth - 1) / stripWidth);

        cudaStream_t streams[NUM_STREAMS];
        for (int s = 0; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

        // A is needed by every strip: upload it once and let the others wait.
        cudaEvent_t aReady;
        CUDA_CHECK(cudaEventCreateWithFlags(&aReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaMemcpyAsync(dA, localA + begin * N, rows * N * sizeof(double), cudaMemcpyHostToDevice,
                                   streams[0]));
        CUDA_CHECK(cudaEventRecord(aReady, streams[0]));
        for (int s = 1; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamWaitEvent(streams[s], aReady, 0));

        for (int strip = 0; strip < numStrips; ++strip) {
            cudaStream_t stream = streams[strip % NUM_STREAMS];
            const size_t colBegin = strip * stripWidth;
            const size_t cols = std::min(stripWidth, N - colBegin);
            const size_t pitch = N * sizeof(double);

            CUDA_CHECK(cudaMemcpy2DAsync(dB + colBegin, pitch, B + colBegin, pitch, cols * sizeof(double), N,
                                         cudaMemcpyHostToDevice, stream));

            const dim3 block(THREADS);
            const dim3 grid(static_cast<unsigned>((cols + BN - 1) / BN), static_cast<unsigned>((rows + BM - 1) / BM));
            dgemmKernel<<<grid, block, 0, stream>>>(dA, dB + colBegin, dC + colBegin, static_cast<int>(rows),
                                                    static_cast<int>(cols), static_cast<int>(N),
                                                    static_cast<int>(N));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy2DAsync(localC + begin * N + colBegin, pitch, dC + colBegin, pitch,
                                         cols * sizeof(double), rows, cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventDestroy(aReady));
        for (int s = 0; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                // A is distributed; its entries are a pure function of (i, k).
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

// Determine the set of GPUs this rank drives: GPUs are distributed among the
// ranks sharing a node; if there are more ranks than GPUs they share devices,
// if there are fewer, each rank drives several GPUs.
std::vector<int> assignDevices() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    std::vector<int> devices;
    if (nodeSize >= deviceCount) {
        devices.push_back(nodeRank % deviceCount);
    } else {
        const int begin = nodeRank * deviceCount / nodeSize;
        const int end = (nodeRank + 1) * deviceCount / nodeSize;
        for (int d = begin; d < end; ++d) devices.push_back(d);
    }
    return devices;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    const std::vector<int> devices = assignDevices();
    // Initialize the CUDA contexts up front so that the measured region does
    // not include one-time driver setup.
    for (const int d : devices) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // Row decomposition of A and C over the ranks.
    std::vector<int> counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t begin = N * r / numRanks;
        const size_t end = N * (r + 1) / numRanks;
        counts[r] = static_cast<int>((end - begin) * N);
        displs[r] = static_cast<int>(begin * N);
    }
    const size_t rowBegin = N * rank / numRanks;
    const size_t rowEnd = N * (rank + 1) / numRanks;
    const size_t localRows = rowEnd - rowBegin;

    // Allocate matrices (A and C distributed by rows, B replicated)
    std::vector<double> localA(localRows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(localRows * N);
    std::vector<double> C(rank == 0 ? N * N : 0);

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(localA, N, rowBegin, rowEnd);
    initMatrixRows(B, N, 0, N);

    // Pinned host memory speeds up the PCIe transfers (best effort).
    const bool pinnedA = localA.empty() || cudaHostRegister(localA.data(), localA.size() * sizeof(double),
                                                            cudaHostRegisterDefault) == cudaSuccess;
    const bool pinnedB = cudaHostRegister(B.data(), B.size() * sizeof(double), cudaHostRegisterDefault) == cudaSuccess;
    const bool pinnedC = localC.empty() || cudaHostRegister(localC.data(), localC.size() * sizeof(double),
                                                            cudaHostRegisterDefault) == cudaSuccess;
    cudaGetLastError();

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(localA.data(), B.data(), localC.data(), N, localRows, devices);

    MPI_Gatherv(localC.data(), counts[rank], MPI_DOUBLE, rank == 0 ? C.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (pinnedA && !localA.empty()) cudaHostUnregister(localA.data());
    if (pinnedB) cudaHostUnregister(B.data());
    if (pinnedC && !localC.empty()) cudaHostUnregister(localC.data());
    cudaGetLastError();

    int exitCode = 0;
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
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
