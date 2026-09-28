#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__,         \
                   __LINE__, cudaGetErrorString(err_));                                      \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA matrix multiplication
//
// The rows of C are distributed over all visible GPUs; every GPU receives the
// full B, its slice of A and computes its slice of C with the kernel below.
//
// Block tile: BM x BN output elements, accumulated over BK-deep slices of the
// k dimension that are staged through shared memory. Each of the BLOCK_THREADS
// threads keeps a TM x TN sub-tile of the result in registers, so every pair of
// shared-memory values feeds TM*TN fused multiply-adds. The k loop of every
// output element still runs in increasing k order, matching the sequential
// reference summation order.
// ---------------------------------------------------------------------------

static constexpr int BM = 64;  // rows of C per block
static constexpr int BN = 64;  // columns of C per block
static constexpr int BK = 16;  // k-slice depth staged in shared memory
static constexpr int TM = 4;   // rows of C per thread
static constexpr int TN = 4;   // columns of C per thread
static constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);  // 256
static constexpr int SPAD = 1;  // shared-memory padding against bank conflicts
static constexpr int WARMUP_REPS = 8;   // warm-up launches (module load, clock ramp)
static constexpr int NUM_STREAMS = 4;   // concurrent transfer/compute pipelines
static constexpr int TARGET_CHUNKS = 16;  // row chunks per device

// Computes C[0:M, 0:N] = A[0:M, 0:N] * B[0:N, 0:N]; all matrices have row
// stride N, so M selects a horizontal slice of the output.
__global__ __launch_bounds__(BLOCK_THREADS) void matrixMultiplyKernel(
    const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
    const int M, const int N) {
    // A is staged transposed (k-major) so the inner loop reads a contiguous run
    // of rows for a fixed k; B is staged row-major in k.
    __shared__ double As[BK][BM + SPAD];
    __shared__ double Bs[BK][BN + SPAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Per-thread output origin inside the block tile.
    const int threadRow = threadIdx.y * TM;
    const int threadCol = threadIdx.x * TN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    // Global-load index decomposition (constant across the k loop).
    // A tile: BM rows x BK columns, BK consecutive doubles per row.
    // B tile: BK rows x BN columns, BN consecutive doubles per row.
    constexpr int A_LOADS = (BM * BK) / BLOCK_THREADS;
    constexpr int B_LOADS = (BK * BN) / BLOCK_THREADS;

    const bool fullTile = (blockRow + BM <= M) && (blockCol + BN <= N);

    for (int k0 = 0; k0 < N; k0 += BK) {
        const bool fullK = (k0 + BK <= N);

#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int e = tid + l * BLOCK_THREADS;
            const int ar = e / BK;
            const int ak = e % BK;
            const int gr = blockRow + ar;
            const int gk = k0 + ak;
            const bool in = (fullTile || gr < M) && (fullK || gk < N);
            As[ak][ar] = in ? A[static_cast<size_t>(gr) * N + gk] : 0.0;
        }

#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int e = tid + l * BLOCK_THREADS;
            const int bk = e / BN;
            const int bn = e % BN;
            const int gk = k0 + bk;
            const int gc = blockCol + bn;
            const bool in = (fullK || gk < N) && (fullTile || gc < N);
            Bs[bk][bn] = in ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[k][threadRow + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[k][threadCol + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + threadRow + i;
        if (!fullTile && row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = blockCol + threadCol + j;
            if (fullTile || col < N) {
                C[static_cast<size_t>(row) * N + col] = acc[i][j];
            }
        }
    }
}

// Devices used for the multiplication (all visible ones).
static int g_deviceCount = 0;

// One-time setup for every device: context creation, kernel module loading,
// transfer staging buffers and clock ramp-up, so that none of this one-time
// cost is attributed to the measured computation.
void initCuda() {
    CUDA_CHECK(cudaGetDeviceCount(&g_deviceCount));
    if (g_deviceCount < 1) {
        printf("No CUDA device available\n");
        exit(1);
    }

    constexpr size_t warmupN = 1024;
    std::vector<double> host(warmupN * warmupN, 1.0);
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid(warmupN / BN, warmupN / BM);

    for (int dev = 0; dev < g_deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        double* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, 3 * warmupN * warmupN * sizeof(double)));
        CUDA_CHECK(
            cudaMemcpy(d, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
        for (int rep = 0; rep < WARMUP_REPS; ++rep) {
            matrixMultiplyKernel<<<grid, block>>>(d, d + warmupN * warmupN,
                                                  d + 2 * warmupN * warmupN,
                                                  static_cast<int>(warmupN),
                                                  static_cast<int>(warmupN));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(
            cudaMemcpy(host.data(), d, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d));
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// Per-device state for the slice of output rows that device computes.
struct DeviceWork {
    int device = 0;
    size_t rowStart = 0;  // first row of C computed here
    size_t rowCount = 0;  // number of rows computed here
    double* dA = nullptr;  // rows [rowStart, rowStart+rowCount) of A
    double* dB = nullptr;  // full B
    double* dC = nullptr;  // rows [rowStart, rowStart+rowCount) of C
    cudaStream_t streams[NUM_STREAMS] = {};
    cudaEvent_t bReady = nullptr;
};

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bBytes = N * N * sizeof(double);

    // Split the output rows over the devices in whole block tiles; devices that
    // would receive no rows are skipped.
    const size_t blockRows = (N + BM - 1) / BM;
    const size_t devBlockRows =
        (blockRows + static_cast<size_t>(g_deviceCount) - 1) / static_cast<size_t>(g_deviceCount);

    std::vector<DeviceWork> work;
    for (int dev = 0; dev < g_deviceCount; ++dev) {
        const size_t start = std::min(N, static_cast<size_t>(dev) * devBlockRows * BM);
        const size_t count = std::min(devBlockRows * BM, N - start);
        if (count == 0) break;
        DeviceWork w;
        w.device = dev;
        w.rowStart = start;
        w.rowCount = count;
        work.push_back(w);
    }

    // Row chunks are whole block tiles so that every kernel launch keeps the
    // fast path for fully-covered tiles.
    const size_t chunkRows = ((devBlockRows + TARGET_CHUNKS - 1) / TARGET_CHUNKS) * BM;
    const dim3 block(BN / TN, BM / TM);

    // Issue all work asynchronously so that the devices run concurrently.
    for (DeviceWork& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));

        const size_t sliceBytes = w.rowCount * N * sizeof(double);
        CUDA_CHECK(cudaMalloc(&w.dA, sliceBytes));
        CUDA_CHECK(cudaMalloc(&w.dB, bBytes));
        CUDA_CHECK(cudaMalloc(&w.dC, sliceBytes));
        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamCreate(&w.streams[s]));
        }

        // B is needed in full by every output block, so it is uploaded first;
        // the A uploads, the multiplications and the C downloads of the
        // individual row chunks then overlap with each other across streams.
        CUDA_CHECK(cudaEventCreateWithFlags(&w.bReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaMemcpyAsync(w.dB, B.data(), bBytes, cudaMemcpyHostToDevice, w.streams[0]));
        CUDA_CHECK(cudaEventRecord(w.bReady, w.streams[0]));
        for (int s = 1; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamWaitEvent(w.streams[s], w.bReady, 0));
        }

        for (size_t row = 0, chunk = 0; row < w.rowCount; row += chunkRows, ++chunk) {
            cudaStream_t stream = w.streams[chunk % NUM_STREAMS];
            const size_t rows = std::min(chunkRows, w.rowCount - row);
            const size_t hostOffset = (w.rowStart + row) * N;
            const size_t devOffset = row * N;
            const size_t chunkBytes = rows * N * sizeof(double);

            CUDA_CHECK(cudaMemcpyAsync(w.dA + devOffset, A.data() + hostOffset, chunkBytes,
                                       cudaMemcpyHostToDevice, stream));

            const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                            static_cast<unsigned>((rows + BM - 1) / BM));
            matrixMultiplyKernel<<<grid, block, 0, stream>>>(w.dA + devOffset, w.dB,
                                                             w.dC + devOffset,
                                                             static_cast<int>(rows),
                                                             static_cast<int>(N));
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(C.data() + hostOffset, w.dC + devOffset, chunkBytes,
                                       cudaMemcpyDeviceToHost, stream));
        }
    }

    for (DeviceWork& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventDestroy(w.bReady));
        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamDestroy(w.streams[s]));
        }
        CUDA_CHECK(cudaFree(w.dA));
        CUDA_CHECK(cudaFree(w.dB));
        CUDA_CHECK(cudaFree(w.dC));
    }
    CUDA_CHECK(cudaSetDevice(0));
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Create the CUDA context, load the kernel module and warm up the driver's
    // transfer staging buffers up front, so that one-time initialization is not
    // attributed to the measured computation.
    initCuda();

    // Page-lock the host matrices (also one-time setup): the multiplication is
    // PCIe-transfer bound, and pinned buffers both raise the achieved bandwidth
    // several-fold and let the copies overlap with the kernels.
    if (N > 0) {
        const size_t bytes = N * N * sizeof(double);
        CUDA_CHECK(cudaHostRegister(A.data(), bytes, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(B.data(), bytes, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(C.data(), bytes, cudaHostRegisterDefault));
    }

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (N > 0) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
        CUDA_CHECK(cudaHostUnregister(B.data()));
        CUDA_CHECK(cudaHostUnregister(C.data()));
    }

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
