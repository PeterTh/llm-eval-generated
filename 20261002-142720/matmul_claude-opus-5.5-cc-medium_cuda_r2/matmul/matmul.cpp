#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,     \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Tiling parameters: each 256-thread block computes a BM x BN tile of C,
// each thread a TM x TN register tile (strided so that loads/stores coalesce
// and shared memory reads are conflict-free).
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TX = 16;
constexpr int TY = 16;
constexpr int TM = BM / TY;
constexpr int TN = BN / TX;
constexpr int THREADS = TX * TY;

// C[rows x N] = A[rows x N] * B[N x N]. For each element the products are
// accumulated in the same k order and with the same rounding as the reference
// host build: products are rounded before being added (no contraction), except
// for the final term when N is odd, which the host compiler fuses into an FMA.
__global__ void __launch_bounds__(THREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    // Number of k terms accumulated with separate multiply/add
    const size_t kMain = N & ~static_cast<size_t>(1);

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (size_t k0 = 0; k0 < kMain; k0 += BK) {
        // Load A tile (BM x BK), stored transposed: As[k][m]
#pragma unroll
        for (int l = 0; l < (BM * BK) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int m = idx / BK;
            const int k = idx % BK;
            const size_t gr = rowBase + m;
            const size_t gk = k0 + k;
            As[k][m] = (gr < rows && gk < kMain) ? A[gr * N + gk] : 0.0;
        }
        // Load B tile (BK x BN)
#pragma unroll
        for (int l = 0; l < (BK * BN) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int k = idx / BN;
            const int n = idx % BN;
            const size_t gk = k0 + k;
            const size_t gc = colBase + n;
            Bs[k][n] = (gk < kMain && gc < N) ? B[gk * N + gc] : 0.0;
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
                for (int j = 0; j < TN; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }

    // Final (fused) term for odd N
    if (kMain < N) {
        const size_t kl = N - 1;
        double b[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t c = colBase + tx + j * TX;
            b[j] = (c < N) ? B[kl * N + c] : 0.0;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const size_t r = rowBase + ty + i * TY;
            const double a = (r < rows) ? A[r * N + kl] : 0.0;
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = fma(a, b[j], acc[i][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t r = rowBase + ty + i * TY;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t c = colBase + tx + j * TX;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}

struct DeviceSlice {
    int device;
    size_t rowStart;
    size_t rows;
    double* dA;
    double* dB;
    double* dC;
    cudaStream_t stream;
};

// Choose devices and row partitioning; allocate device buffers and streams.
std::vector<DeviceSlice> setupDevices(const size_t N) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(EXIT_FAILURE);
    }
    // Do not split small problems across more GPUs than useful
    constexpr size_t minRowsPerDevice = 256;
    size_t useDevices = std::max<size_t>(1, std::min<size_t>(deviceCount, N / minRowsPerDevice));

    std::vector<DeviceSlice> slices;
    // Row chunks in multiples of BM for balanced, full tiles
    const size_t tiles = (N + BM - 1) / BM;
    useDevices = std::min(useDevices, std::max<size_t>(tiles, 1));
    size_t tileStart = 0;
    for (size_t d = 0; d < useDevices; ++d) {
        const size_t tileCount = tiles / useDevices + (d < tiles % useDevices ? 1 : 0);
        const size_t rowStart = std::min(tileStart * BM, N);
        const size_t rowEnd = std::min((tileStart + tileCount) * BM, N);
        tileStart += tileCount;
        DeviceSlice s{static_cast<int>(d), rowStart, rowEnd - rowStart, nullptr, nullptr, nullptr, nullptr};
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        const size_t sliceBytes = std::max<size_t>(s.rows * N, 1) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&s.dA, sliceBytes));
        CUDA_CHECK(cudaMalloc(&s.dB, std::max<size_t>(N * N, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.dC, sliceBytes));
        slices.push_back(s);
    }
    return slices;
}

void releaseDevices(std::vector<DeviceSlice>& slices) {
    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaFree(s.dA));
        CUDA_CHECK(cudaFree(s.dB));
        CUDA_CHECK(cudaFree(s.dC));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    slices.clear();
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const std::vector<DeviceSlice>& slices) {
    const size_t rowBytes = N * sizeof(double);
    for (const auto& s : slices) {
        if (s.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMemcpyAsync(s.dA, A.data() + s.rowStart * N, s.rows * rowBytes,
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.dB, B.data(), N * rowBytes, cudaMemcpyHostToDevice, s.stream));
        const dim3 block(TX, TY);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((s.rows + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, s.stream>>>(s.dA, s.dB, s.dC, s.rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C.data() + s.rowStart * N, s.dC, s.rows * rowBytes,
                                   cudaMemcpyDeviceToHost, s.stream));
    }
    for (const auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
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
    
    // Set up GPUs (context creation, allocations) and pin host buffers
    std::vector<DeviceSlice> slices = setupDevices(N);
    const size_t matBytes = N * N * sizeof(double);
    if (matBytes > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), matBytes, cudaHostRegisterPortable));
        CUDA_CHECK(cudaHostRegister(B.data(), matBytes, cudaHostRegisterPortable));
        CUDA_CHECK(cudaHostRegister(C.data(), matBytes, cudaHostRegisterPortable));
    }
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, slices);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    if (matBytes > 0) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
        CUDA_CHECK(cudaHostUnregister(B.data()));
        CUDA_CHECK(cudaHostUnregister(C.data()));
    }
    releaseDevices(slices);
    
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
