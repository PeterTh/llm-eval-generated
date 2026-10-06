#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int BM = 64;           // rows of C per block
constexpr int BN = 64;           // cols of C per block
constexpr int BK = 16;           // k-slice per iteration
constexpr int TX = 16;           // threads in x
constexpr int TY = 16;           // threads in y
constexpr int TM = BM / TY;      // rows per thread
constexpr int TN = BN / TX;      // cols per thread
constexpr int NTHREADS = TX * TY;
constexpr int A_LOADS = (BM * BK) / NTHREADS;
constexpr int B_LOADS = (BK * BN) / NTHREADS;

// C[rows x N] = A[rows x N] * B[N x N], row-major. Shared-memory tiled,
// register-blocked kernel with register prefetch of the next k-slice.
__global__ void __launch_bounds__(NTHREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int rows, const int N) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    double ra[A_LOADS];
    double rb[B_LOADS];

    auto loadGlobal = [&](const int k0) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            const int m = idx / BK;
            const int k = idx % BK;
            const int gr = rowBase + m;
            const int gk = k0 + k;
            ra[l] = (gr < rows && gk < N) ? A[(size_t)gr * N + gk] : 0.0;
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            const int k = idx / BN;
            const int n = idx % BN;
            const int gk = k0 + k;
            const int gc = colBase + n;
            rb[l] = (gk < N && gc < N) ? B[(size_t)gk * N + gc] : 0.0;
        }
    };
    auto storeShared = [&](const int buf) {
#pragma unroll
        for (int l = 0; l < A_LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            As[buf][idx % BK][idx / BK] = ra[l];
        }
#pragma unroll
        for (int l = 0; l < B_LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            Bs[buf][idx / BN][idx % BN] = rb[l];
        }
    };

    const int numTiles = (N + BK - 1) / BK;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][k][ty + i * TY];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][k][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (t + 1 < numTiles) storeShared(cur ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty + i * TY;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx + j * TX;
            if (gc < N) C[(size_t)gr * N + gc] = acc[i][j];
        }
    }
}

// Per-device execution context; rows of C are partitioned across GPUs.
struct DeviceCtx {
    int device = 0;
    size_t rowStart = 0;
    size_t rows = 0;
    cudaStream_t stream = nullptr;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
};

class GpuMatmul {
  public:
    GpuMatmul(const std::vector<double>& A, const std::vector<double>& B,
              std::vector<double>& C, const size_t N)
        : A_(A), B_(B), C_(C), N_(N) {
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount <= 0) {
            fprintf(stderr, "No CUDA devices available\n");
            exit(EXIT_FAILURE);
        }
        // Use enough GPUs so that each one gets a meaningful slab of rows.
        constexpr size_t minRowsPerDevice = 256;
        const size_t maxUseful = std::max<size_t>(1, (N + minRowsPerDevice - 1) / minRowsPerDevice);
        const int numDev = static_cast<int>(std::min<size_t>(devCount, maxUseful));

        // Partition rows in multiples of BM.
        const size_t rowTiles = (N + BM - 1) / BM;
        size_t tileStart = 0;
        for (int d = 0; d < numDev; ++d) {
            const size_t tiles = rowTiles / numDev + (static_cast<size_t>(d) < rowTiles % numDev ? 1 : 0);
            DeviceCtx ctx;
            ctx.device = d;
            ctx.rowStart = std::min(N, tileStart * BM);
            ctx.rows = std::min(N, (tileStart + tiles) * BM) - ctx.rowStart;
            tileStart += tiles;
            if (ctx.rows == 0) continue;
            ctxs_.push_back(ctx);
        }

        const size_t bytes = N * N * sizeof(double);
        if (bytes > 0) {
            // Pin host buffers for fast asynchronous transfers.
            pinned_ = cudaHostRegister(const_cast<double*>(A.data()), bytes, cudaHostRegisterDefault) == cudaSuccess &&
                      cudaHostRegister(const_cast<double*>(B.data()), bytes, cudaHostRegisterDefault) == cudaSuccess &&
                      cudaHostRegister(C.data(), bytes, cudaHostRegisterDefault) == cudaSuccess;
            if (!pinned_) {
                cudaGetLastError();
                cudaHostUnregister(const_cast<double*>(A.data()));
                cudaHostUnregister(const_cast<double*>(B.data()));
                cudaHostUnregister(C.data());
                cudaGetLastError();
            }
        }

        for (auto& ctx : ctxs_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&ctx.dA, ctx.rows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&ctx.dB, bytes));
            CUDA_CHECK(cudaMalloc(&ctx.dC, ctx.rows * N * sizeof(double)));
            CUDA_CHECK(cudaFuncSetAttribute(matmulKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
        }
        for (auto& ctx : ctxs_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    ~GpuMatmul() {
        for (auto& ctx : ctxs_) {
            cudaSetDevice(ctx.device);
            cudaFree(ctx.dA);
            cudaFree(ctx.dB);
            cudaFree(ctx.dC);
            cudaStreamDestroy(ctx.stream);
        }
        if (pinned_) {
            cudaHostUnregister(const_cast<double*>(A_.data()));
            cudaHostUnregister(const_cast<double*>(B_.data()));
            cudaHostUnregister(C_.data());
        }
    }

    // Upload inputs, compute C = A * B on all devices, download result.
    void run() {
        const size_t N = N_;
        for (auto& ctx : ctxs_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            const size_t slab = ctx.rows * N * sizeof(double);
            CUDA_CHECK(cudaMemcpyAsync(ctx.dA, A_.data() + ctx.rowStart * N, slab,
                                       cudaMemcpyHostToDevice, ctx.stream));
            CUDA_CHECK(cudaMemcpyAsync(ctx.dB, B_.data(), N * N * sizeof(double),
                                       cudaMemcpyHostToDevice, ctx.stream));
            const dim3 block(TX, TY);
            const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                            static_cast<unsigned>((ctx.rows + BM - 1) / BM));
            matmulKernel<<<grid, block, 0, ctx.stream>>>(ctx.dA, ctx.dB, ctx.dC,
                                                         static_cast<int>(ctx.rows),
                                                         static_cast<int>(N));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(C_.data() + ctx.rowStart * N, ctx.dC, slab,
                                       cudaMemcpyDeviceToHost, ctx.stream));
        }
        for (auto& ctx : ctxs_) {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }
    }

  private:
    const std::vector<double>& A_;
    const std::vector<double>& B_;
    std::vector<double>& C_;
    size_t N_;
    std::vector<DeviceCtx> ctxs_;
    bool pinned_ = false;
};

void matrixMultiply(GpuMatmul& gpu) {
    gpu.run();
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
    
    // Set up GPU resources (device buffers, pinned host memory) outside the timed region
    GpuMatmul gpu(A, B, C, N);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(gpu);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
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
