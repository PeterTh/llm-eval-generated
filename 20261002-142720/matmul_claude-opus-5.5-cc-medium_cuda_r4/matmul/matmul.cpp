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

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Tiling parameters: each block computes a BM x BN tile of C with
// TX x TY threads, each thread owning a TM x TN register sub-tile.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TX = 16;
constexpr int TY = 16;
constexpr int TM = BM / TY;
constexpr int TN = BN / TX;
constexpr int NTHREADS = TX * TY;

// C[rows x N] = A[rows x N] * B[N x N]  (row-major).
// Every output element accumulates over k in increasing order with fused
// multiply-add, the same sequence as the reference sequential loop.
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

    // Per-thread load registers (BM*BK / NTHREADS = 4 elements each)
    constexpr int LOADS = (BM * BK) / NTHREADS;
    double ra[LOADS], rb[LOADS];

    auto loadGlobal = [&](const int k0) {
#pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            // A tile: BM rows x BK cols, consecutive threads read along k
            const int ar = idx / BK, ac = idx % BK;
            const int gr = rowBase + ar, gk = k0 + ac;
            ra[l] = (gr < rows && gk < N) ? A[(size_t)gr * N + gk] : 0.0;
            // B tile: BK rows x BN cols, consecutive threads read along j
            const int br = idx / BN, bc = idx % BN;
            const int gkb = k0 + br, gc = colBase + bc;
            rb[l] = (gkb < N && gc < N) ? B[(size_t)gkb * N + gc] : 0.0;
        }
    };
    auto storeShared = [&](const int buf) {
#pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int idx = tid + l * NTHREADS;
            As[buf][idx % BK][idx / BK] = ra[l];
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
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][kk][ty + i * TY];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][kk][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }

        if (t + 1 < numTiles) {
            storeShared(cur ^ 1);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + ty + i * TY;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = colBase + tx + j * TX;
            if (c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

// Per-device work description: a contiguous slice of rows of A and C.
struct DeviceSlice {
    int device;
    size_t rowStart;
    size_t rows;
    cudaStream_t stream;
    double* dA;
    double* dB;
    double* dC;
};

class GpuMatmul {
  public:
    GpuMatmul(std::vector<double>& A, std::vector<double>& B, std::vector<double>& C,
              const size_t N)
        : A_(A), B_(B), C_(C), N_(N) {
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA devices found\n");
            exit(EXIT_FAILURE);
        }
        // Split rows across GPUs; keep at least MIN_ROWS rows per device so
        // small problems are not dominated by replicated transfers of B.
        constexpr size_t MIN_ROWS = 512;
        size_t nDev = std::min<size_t>(devCount, std::max<size_t>(1, N / MIN_ROWS));
        if (N == 0) nDev = 0;

        const size_t bytes = N * N * sizeof(double);
        if (bytes > 0) {
            pinned_ = cudaHostRegister(A.data(), bytes, cudaHostRegisterPortable) == cudaSuccess &&
                      cudaHostRegister(B.data(), bytes, cudaHostRegisterPortable) == cudaSuccess &&
                      cudaHostRegister(C.data(), bytes, cudaHostRegisterPortable) == cudaSuccess;
            if (!pinned_) {
                cudaHostUnregister(A.data());
                cudaHostUnregister(B.data());
                cudaHostUnregister(C.data());
                cudaGetLastError();
            }
        }

        size_t rowStart = 0;
        for (size_t d = 0; d < nDev; ++d) {
            const size_t rows = N / nDev + (d < N % nDev ? 1 : 0);
            DeviceSlice s{static_cast<int>(d), rowStart, rows, nullptr, nullptr, nullptr, nullptr};
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&s.dA, rows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&s.dB, bytes));
            CUDA_CHECK(cudaMalloc(&s.dC, rows * N * sizeof(double)));
            slices_.push_back(s);
            rowStart += rows;
        }
    }

    ~GpuMatmul() {
        for (auto& s : slices_) {
            cudaSetDevice(s.device);
            cudaFree(s.dA);
            cudaFree(s.dB);
            cudaFree(s.dC);
            cudaStreamDestroy(s.stream);
        }
        if (pinned_) {
            cudaHostUnregister(A_.data());
            cudaHostUnregister(B_.data());
            cudaHostUnregister(C_.data());
        }
    }

    // Transfers inputs, computes C = A * B on all devices, copies C back.
    void run() {
        const size_t N = N_;
        for (auto& s : slices_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaMemcpyAsync(s.dA, A_.data() + s.rowStart * N,
                                       s.rows * N * sizeof(double), cudaMemcpyHostToDevice,
                                       s.stream));
            CUDA_CHECK(cudaMemcpyAsync(s.dB, B_.data(), N * N * sizeof(double),
                                       cudaMemcpyHostToDevice, s.stream));
            const dim3 block(TX, TY);
            const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                            static_cast<unsigned>((s.rows + BM - 1) / BM));
            matmulKernel<<<grid, block, 0, s.stream>>>(s.dA, s.dB, s.dC,
                                                       static_cast<int>(s.rows),
                                                       static_cast<int>(N));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(C_.data() + s.rowStart * N, s.dC,
                                       s.rows * N * sizeof(double), cudaMemcpyDeviceToHost,
                                       s.stream));
        }
        for (auto& s : slices_) {
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
    }

  private:
    std::vector<double>& A_;
    std::vector<double>& B_;
    std::vector<double>& C_;
    size_t N_;
    bool pinned_ = false;
    std::vector<DeviceSlice> slices_;
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
    
    // Set up GPU resources (device contexts, buffers, pinned host memory)
    GpuMatmul gpu(A, B, C, N);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.run();
    
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
