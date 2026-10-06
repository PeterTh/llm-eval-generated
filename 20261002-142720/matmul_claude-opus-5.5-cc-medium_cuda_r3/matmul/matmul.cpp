#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Tiled DGEMM: each 256-thread block computes a BM x BN tile of C, each thread
// a TM x TN micro-tile. The k-loop runs in increasing order for every element,
// matching the summation order of the original sequential code.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN;  // 16
constexpr int THREADS_Y = BM / TM;  // 16
constexpr int NUM_THREADS = THREADS_X * THREADS_Y;
constexpr int A_LOADS = BM * BK / NUM_THREADS;
constexpr int B_LOADS = BK * BN / NUM_THREADS;

// A: rows x N slab, B: N x N, C: rows x N slab (all row-major, leading dim N)
__global__ void __launch_bounds__(NUM_THREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int rows, const int N) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double regA[A_LOADS];
    double regB[B_LOADS];

    auto loadGlobal = [&](const int k0) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int idx = tid + l * NUM_THREADS;
            const int r = idx / BK;
            const int c = idx % BK;
            const int gr = rowBase + r;
            const int gc = k0 + c;
            regA[l] = (gr < rows && gc < N) ? A[static_cast<size_t>(gr) * N + gc] : 0.0;
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int idx = tid + l * NUM_THREADS;
            const int r = idx / BN;
            const int c = idx % BN;
            const int gr = k0 + r;
            const int gc = colBase + c;
            regB[l] = (gr < N && gc < N) ? B[static_cast<size_t>(gr) * N + gc] : 0.0;
        }
    };
    auto storeShared = [&](const int buf) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int idx = tid + l * NUM_THREADS;
            As[buf][idx % BK][idx / BK] = regA[l];
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int idx = tid + l * NUM_THREADS;
            Bs[buf][idx / BN][idx % BN] = regB[l];
        }
    };

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    const int numTiles = (N + BK - 1) / BK;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int buf = t & 1;
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[buf][k][ty + i * THREADS_Y];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[buf][k][tx + j * THREADS_X];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (t + 1 < numTiles) {
            storeShared(buf ^ 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty + i * THREADS_Y;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx + j * THREADS_X;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// Per-GPU state: each device owns a contiguous slab of rows of A and C,
// plus a full copy of B. Row slabs are processed in chunks on two streams
// so that host<->device transfers overlap with computation.
struct GpuContext {
    int device = 0;
    size_t rowBegin = 0;
    size_t rowCount = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t bReady = nullptr;
};

class GpuMatmul {
  public:
    GpuMatmul(const size_t N) : N_(N) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            exit(EXIT_FAILURE);
        }
        // Give every GPU at least MIN_ROWS rows so tiny problems don't pay
        // multi-device overhead for no benefit.
        constexpr size_t MIN_ROWS = 256;
        size_t useDevices = std::max<size_t>(1, (N + MIN_ROWS - 1) / MIN_ROWS);
        useDevices = std::min<size_t>(useDevices, static_cast<size_t>(deviceCount));

        // Split rows in multiples of BM to keep tiles aligned
        const size_t tiles = (N + BM - 1) / BM;
        size_t tileBegin = 0;
        for (size_t d = 0; d < useDevices; ++d) {
            const size_t tileCount = tiles / useDevices + (d < tiles % useDevices ? 1 : 0);
            GpuContext ctx;
            ctx.device = static_cast<int>(d);
            ctx.rowBegin = std::min(N, tileBegin * BM);
            ctx.rowCount = std::min(N, (tileBegin + tileCount) * BM) - ctx.rowBegin;
            tileBegin += tileCount;
            if (ctx.rowCount == 0) continue;

            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaMalloc(&ctx.dA, ctx.rowCount * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&ctx.dB, N * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&ctx.dC, ctx.rowCount * N * sizeof(double)));
            for (auto& s : ctx.streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.bReady, cudaEventDisableTiming));
            // Warm-up launch: forces lazy module loading outside the timed region
            matmulKernel<<<1, dim3(THREADS_X, THREADS_Y), 0, ctx.streams[0]>>>(
                ctx.dA, ctx.dB, ctx.dC, 0, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(ctx.streams[0]));
            gpus_.push_back(ctx);
        }
    }

    ~GpuMatmul() {
        for (auto& ctx : gpus_) {
            cudaSetDevice(ctx.device);
            cudaFree(ctx.dA);
            cudaFree(ctx.dB);
            cudaFree(ctx.dC);
            for (auto& s : ctx.streams) cudaStreamDestroy(s);
            cudaEventDestroy(ctx.bReady);
        }
    }

    void multiply(const double* A, const double* B, double* C) {
        const size_t N = N_;
        const size_t bytesB = N * N * sizeof(double);
        // Chunk size in rows (multiple of BM): aim for a few chunks per GPU
        // so transfers overlap with compute, without making kernels too small.
        for (auto& ctx : gpus_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaMemcpyAsync(ctx.dB, B, bytesB, cudaMemcpyHostToDevice, ctx.streams[0]));
            CUDA_CHECK(cudaEventRecord(ctx.bReady, ctx.streams[0]));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.streams[1], ctx.bReady, 0));

            size_t chunkRows = ((ctx.rowCount / 4 + BM - 1) / BM) * BM;
            chunkRows = std::max<size_t>(chunkRows, 4 * BM);

            int si = 0;
            for (size_t r = 0; r < ctx.rowCount; r += chunkRows, si ^= 1) {
                const size_t rows = std::min(chunkRows, ctx.rowCount - r);
                const size_t off = r * N;
                const size_t bytes = rows * N * sizeof(double);
                cudaStream_t s = ctx.streams[si];
                CUDA_CHECK(cudaMemcpyAsync(ctx.dA + off, A + ctx.rowBegin * N + off, bytes,
                                           cudaMemcpyHostToDevice, s));
                const dim3 block(THREADS_X, THREADS_Y);
                const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                                static_cast<unsigned>((rows + BM - 1) / BM));
                matmulKernel<<<grid, block, 0, s>>>(ctx.dA + off, ctx.dB, ctx.dC + off,
                                                    static_cast<int>(rows), static_cast<int>(N));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(C + ctx.rowBegin * N + off, ctx.dC + off, bytes,
                                           cudaMemcpyDeviceToHost, s));
            }
        }
        for (auto& ctx : gpus_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

  private:
    size_t N_;
    std::vector<GpuContext> gpus_;
};

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
    
    // Set up GPUs (context creation, device buffers, pinned host memory)
    // outside of the timed region
    GpuMatmul gpu(N);
    if (N > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), N * N * sizeof(double), cudaHostRegisterPortable));
        CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterPortable));
        CUDA_CHECK(cudaHostRegister(C.data(), N * N * sizeof(double), cudaHostRegisterPortable));
    }
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (N > 0) gpu.multiply(A.data(), B.data(), C.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    if (N > 0) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
        CUDA_CHECK(cudaHostUnregister(B.data()));
        CUDA_CHECK(cudaHostUnregister(C.data()));
    }
    
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
