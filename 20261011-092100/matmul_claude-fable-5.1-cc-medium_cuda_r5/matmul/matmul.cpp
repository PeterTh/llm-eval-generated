#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
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
// CUDA kernel: tiled, register-blocked double-precision GEMM
//   C[M x N] = A[M x K] * B[K x N]   (row-major, leading dimension = ld)
// Each thread block computes a BM x BN tile of C; each thread a TM x TN sub-tile.
// Arbitrary sizes are supported via zero-padded bounds checks.
// ---------------------------------------------------------------------------
constexpr int BM = 64;   // rows of C per block
constexpr int BN = 64;   // cols of C per block
constexpr int BK = 16;   // k-depth per shared-memory stage
constexpr int TM = 4;    // rows of C per thread
constexpr int TN = 4;    // cols of C per thread
constexpr int THREADS_X = BN / TN;            // 16
constexpr int THREADS_Y = BM / TM;            // 16
constexpr int NTHREADS = THREADS_X * THREADS_Y;  // 256

__global__ void __launch_bounds__(NTHREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int M, const int N, const int K) {
    // As stored transposed ([k][m]) so that the inner loop reads consecutive m.
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x % THREADS_X;
    const int ty = threadIdx.x / THREADS_X;
    const int tid = threadIdx.x;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    // A tile: BM x BK = 1024 elements, 4 per thread; load coalesced along k.
    const int aRow = tid / BK;          // 0..15
    const int aCol = tid % BK;          // 0..15
    // B tile: BK x BN = 1024 elements, 4 per thread; load coalesced along n.
    const int bRow = tid / BN;          // 0..3
    const int bCol = tid % BN;          // 0..63

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int s = 0; s < (BM * BK) / NTHREADS; ++s) {
            const int r = aRow + s * (NTHREADS / BK);  // 0..63
            const int gr = rowBase + r;
            const int gk = k0 + aCol;
            As[aCol][r] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0;
        }
#pragma unroll
        for (int s = 0; s < (BK * BN) / NTHREADS; ++s) {
            const int r = bRow + s * (NTHREADS / BN);  // 0..15
            const int gk = k0 + r;
            const int gc = colBase + bCol;
            Bs[r][bCol] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx * TN + j;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU execution context: rows of C are partitioned across devices.
// Each device holds its slice of A, the full B, and its slice of C.
// ---------------------------------------------------------------------------
struct DeviceSlice {
    int device = 0;
    size_t rowStart = 0;
    size_t rows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
};

struct GpuContext {
    std::vector<DeviceSlice> slices;
    size_t N = 0;
};

static int selectDeviceCount(const size_t N) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        fprintf(stderr, "No CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    // Only split across GPUs when each slice is large enough for the extra
    // launch/transfer overhead to be amortized.
    constexpr size_t minRowsPerDevice = 1024;
    size_t usable = N / minRowsPerDevice;
    if (usable < 1) usable = 1;
    if (usable > static_cast<size_t>(count)) usable = static_cast<size_t>(count);
    return static_cast<int>(usable);
}

static void gpuSetup(GpuContext& ctx, const std::vector<double>& A, const std::vector<double>& B,
                     const size_t N) {
    ctx.N = N;
    const int ndev = selectDeviceCount(N);
    ctx.slices.resize(ndev);

    const size_t rowsPer = (N + ndev - 1) / ndev;
    for (int d = 0; d < ndev; ++d) {
        DeviceSlice& s = ctx.slices[d];
        s.device = d;
        s.rowStart = std::min(N, static_cast<size_t>(d) * rowsPer);
        s.rows = std::min(rowsPer, N - s.rowStart);

        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&s.dA, std::max<size_t>(1, s.rows * N) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.dC, std::max<size_t>(1, s.rows * N) * sizeof(double)));

        if (s.rows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(s.dA, A.data() + s.rowStart * N, s.rows * N * sizeof(double),
                                       cudaMemcpyHostToDevice, s.stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(s.dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice,
                                   s.stream));
    }
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamSynchronize(ctx.slices[d].stream));
    }
}

// Launch the GEMM on every device slice and wait for completion.
void matrixMultiply(GpuContext& ctx) {
    const size_t N = ctx.N;
    for (DeviceSlice& s : ctx.slices) {
        if (s.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(s.device));
        const dim3 block(NTHREADS);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((s.rows + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, s.stream>>>(s.dA, s.dB, s.dC, static_cast<int>(s.rows),
                                                   static_cast<int>(N), static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());
    }
    for (DeviceSlice& s : ctx.slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }
}

static void gpuDownload(GpuContext& ctx, std::vector<double>& C) {
    const size_t N = ctx.N;
    for (DeviceSlice& s : ctx.slices) {
        if (s.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMemcpyAsync(C.data() + s.rowStart * N, s.dC, s.rows * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, s.stream));
    }
    for (DeviceSlice& s : ctx.slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }
}

static void gpuTeardown(GpuContext& ctx) {
    for (DeviceSlice& s : ctx.slices) {
        cudaSetDevice(s.device);
        cudaFree(s.dA);
        cudaFree(s.dB);
        cudaFree(s.dC);
        cudaStreamDestroy(s.stream);
    }
    ctx.slices.clear();
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

    // Upload inputs to the GPU(s)
    GpuContext ctx;
    gpuSetup(ctx, A, B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Download result
    gpuDownload(ctx, C);
    gpuTeardown(ctx);
    
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
