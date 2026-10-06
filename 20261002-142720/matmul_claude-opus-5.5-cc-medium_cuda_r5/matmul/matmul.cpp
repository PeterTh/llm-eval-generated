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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Tile configuration: each 256-thread block computes a BM x BN tile of C,
// each thread a TM x TN register sub-tile (strided to avoid bank conflicts).
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TX = 16;  // threads along N
constexpr int TY = 16;  // threads along M
constexpr int TM = BM / TY;
constexpr int TN = BN / TX;
constexpr int NTHREADS = TX * TY;

// C[rows x colStart..colEnd) = A[rows x N] * B[N x colStart..colEnd);
// k is accumulated strictly in order (one FMA per step) for every element,
// matching the sequential reference.
__global__ void __launch_bounds__(NTHREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const int rows, const int N,
             const int colStart, const int colEnd) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = colStart + blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Load A tile (BM x BK), stored transposed in shared memory
#pragma unroll
        for (int l = tid; l < BM * BK; l += NTHREADS) {
            const int r = l / BK;
            const int c = l % BK;
            const int gr = rowBase + r;
            const int gc = k0 + c;
            As[c][r] = (gr < rows && gc < N) ? A[static_cast<size_t>(gr) * N + gc] : 0.0;
        }
        // Load B tile (BK x BN)
#pragma unroll
        for (int l = tid; l < BK * BN; l += NTHREADS) {
            const int r = l / BN;
            const int c = l % BN;
            const int gr = k0 + r;
            const int gc = colBase + c;
            Bs[r][c] = (gr < N && gc < colEnd) ? B[static_cast<size_t>(gr) * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty + i * TY];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty + i * TY;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + tx + j * TX;
            if (gc < colEnd) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// Per-device resources, created before the timed region
struct DeviceSlab {
    size_t rowStart = 0;
    size_t rows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t copyIn = nullptr;
    std::vector<cudaStream_t> compute;  // one per panel: lets panel kernels
                                        // overlap and fill each other's tail
    cudaStream_t copyOut = nullptr;
    std::vector<cudaEvent_t> bReady;  // B column panel p uploaded
    std::vector<cudaEvent_t> cReady;  // C column panel p computed
};

struct GpuContext {
    std::vector<DeviceSlab> devices;
    std::vector<size_t> panelStart;  // column panel boundaries (pipelining)
};

// Pick devices, create contexts/streams/buffers, load the kernel module and
// pin host buffers (setup work, kept outside the timed region)
GpuContext setupGpus(std::vector<double>& A, std::vector<double>& B,
                     std::vector<double>& C, const size_t N) {
    GpuContext ctx;
    if (N == 0) return ctx;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(EXIT_FAILURE);
    }
    // Use multiple GPUs only when each gets a sizeable share of row tiles
    const size_t rowTiles = (N + BM - 1) / BM;
    size_t nDev = std::min<size_t>(deviceCount, std::max<size_t>(1, rowTiles / 4));
    if (N < 512) nDev = 1;

    // Split columns into panels so that uploads/downloads overlap compute
    const size_t colTiles = (N + BN - 1) / BN;
    const size_t nPanels = std::min<size_t>(colTiles, N >= 2048 ? 8 : (N >= 512 ? 4 : 1));
    ctx.panelStart.resize(nPanels + 1);
    for (size_t p = 0; p <= nPanels; ++p) {
        ctx.panelStart[p] = std::min(N, (colTiles * p / nPanels) * BN);
    }

    ctx.devices.resize(nDev);
    for (size_t d = 0; d < nDev; ++d) {
        DeviceSlab& dev = ctx.devices[d];
        dev.rowStart = std::min(N, (rowTiles * d / nDev) * BM);
        dev.rows = std::min(N, (rowTiles * (d + 1) / nDev) * BM) - dev.rowStart;

        CUDA_CHECK(cudaSetDevice(static_cast<int>(d)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev.copyIn, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dev.copyOut, cudaStreamNonBlocking));
        dev.bReady.resize(nPanels);
        dev.cReady.resize(nPanels);
        dev.compute.resize(nPanels);
        for (size_t p = 0; p < nPanels; ++p) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&dev.compute[p], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&dev.bReady[p], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&dev.cReady[p], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaMalloc(&dev.dA, std::max<size_t>(1, dev.rows) * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev.dC, std::max<size_t>(1, dev.rows) * N * sizeof(double)));
        // Force (lazy) module loading of the kernel on this device
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, matmulKernel));
    }

    const size_t bytes = N * N * sizeof(double);
    CUDA_CHECK(cudaHostRegister(A.data(), bytes, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostRegister(B.data(), bytes, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostRegister(C.data(), bytes, cudaHostRegisterPortable));
    return ctx;
}

void teardownGpus(GpuContext& ctx, std::vector<double>& A, std::vector<double>& B,
                  std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    CUDA_CHECK(cudaHostUnregister(A.data()));
    CUDA_CHECK(cudaHostUnregister(B.data()));
    CUDA_CHECK(cudaHostUnregister(C.data()));
    for (size_t d = 0; d < ctx.devices.size(); ++d) {
        DeviceSlab& dev = ctx.devices[d];
        CUDA_CHECK(cudaSetDevice(static_cast<int>(d)));
        CUDA_CHECK(cudaFree(dev.dA));
        CUDA_CHECK(cudaFree(dev.dB));
        CUDA_CHECK(cudaFree(dev.dC));
        for (auto ev : dev.bReady) CUDA_CHECK(cudaEventDestroy(ev));
        for (auto ev : dev.cReady) CUDA_CHECK(cudaEventDestroy(ev));
        CUDA_CHECK(cudaStreamDestroy(dev.copyIn));
        for (auto s : dev.compute) CUDA_CHECK(cudaStreamDestroy(s));
        CUDA_CHECK(cudaStreamDestroy(dev.copyOut));
    }
}

// Each device computes a contiguous slab of rows of C. Work is pipelined over
// column panels: upload B panel p+1 while computing panel p while downloading
// the C panel p-1.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, const GpuContext& ctx) {
    if (N == 0) return;
    const size_t nPanels = ctx.panelStart.size() - 1;
    const size_t pitch = N * sizeof(double);

    for (size_t d = 0; d < ctx.devices.size(); ++d) {
        const DeviceSlab& dev = ctx.devices[d];
        if (dev.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(static_cast<int>(d)));

        CUDA_CHECK(cudaMemcpyAsync(dev.dA, A.data() + dev.rowStart * N, dev.rows * pitch,
                                   cudaMemcpyHostToDevice, dev.copyIn));
        for (size_t p = 0; p < nPanels; ++p) {
            const size_t c0 = ctx.panelStart[p];
            const size_t w = ctx.panelStart[p + 1] - c0;
            CUDA_CHECK(cudaMemcpy2DAsync(dev.dB + c0, pitch, B.data() + c0, pitch,
                                         w * sizeof(double), N, cudaMemcpyHostToDevice,
                                         dev.copyIn));
            CUDA_CHECK(cudaEventRecord(dev.bReady[p], dev.copyIn));
        }
        for (size_t p = 0; p < nPanels; ++p) {
            const size_t c0 = ctx.panelStart[p];
            const size_t w = ctx.panelStart[p + 1] - c0;
            CUDA_CHECK(cudaStreamWaitEvent(dev.compute[p], dev.bReady[p], 0));
            const dim3 block(TX, TY);
            const dim3 grid(static_cast<unsigned>((w + BN - 1) / BN),
                            static_cast<unsigned>((dev.rows + BM - 1) / BM));
            matmulKernel<<<grid, block, 0, dev.compute[p]>>>(
                dev.dA, dev.dB, dev.dC, static_cast<int>(dev.rows), static_cast<int>(N),
                static_cast<int>(c0), static_cast<int>(c0 + w));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(dev.cReady[p], dev.compute[p]));

            CUDA_CHECK(cudaStreamWaitEvent(dev.copyOut, dev.cReady[p], 0));
            CUDA_CHECK(cudaMemcpy2DAsync(C.data() + dev.rowStart * N + c0, pitch, dev.dC + c0,
                                         pitch, w * sizeof(double), dev.rows,
                                         cudaMemcpyDeviceToHost, dev.copyOut));
        }
    }
    for (size_t d = 0; d < ctx.devices.size(); ++d) {
        CUDA_CHECK(cudaSetDevice(static_cast<int>(d)));
        CUDA_CHECK(cudaStreamSynchronize(ctx.devices[d].copyOut));
    }
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
    
    // GPU setup (device contexts, streams, pinned host memory)
    GpuContext gpu = setupGpus(A, B, C, N);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, gpu);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    teardownGpus(gpu, A, B, C, N);

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
