#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                                  \
            exit(1);                                                                           \
        }                                                                                      \
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

// Tile sizes for the blocked GPU kernel: each 256-thread block computes a
// BM x BN output tile, each thread a TM x TN sub-tile held in registers.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN); // 256

// Computes C += A * B over the k range [kBegin, kEnd) for a horizontal slice of
// M rows: A and C point at the first row of the slice, B is the full NxN matrix.
// For kBegin == 0 the accumulator starts at zero, otherwise it continues from the
// partial sum already in C. Every thread therefore accumulates over k in strictly
// ascending order, matching the summation order of the sequential version.
__global__ __launch_bounds__(THREADS) void matmulKernel(const double* __restrict__ A,
                                                        const double* __restrict__ B,
                                                        double* __restrict__ C, const int N,
                                                        const int M, const int kBegin,
                                                        const int kEnd) {
    // As is stored transposed (k-major) and padded to avoid shared memory bank conflicts.
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM; // first row of this block's output tile
    const int col0 = blockIdx.x * BN; // first column of this block's output tile

    // Thread lane within the tile; the TM x TN sub-tile is strided by 16 so that
    // shared memory reads and global stores stay conflict-free / coalesced.
    const int tx = tid % 16;
    const int ty = tid / 16;

    // Global load mapping: 16 threads per row for A, 64 threads per row for B.
    const int aK = tid % BK;
    const int aM = tid / BK; // 0..15, strided by 16 over BM
    const int bN = tid % BN;
    const int bK = tid / BN; // 0..3, strided by 4 over BK

    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gRow = row0 + ty + i * 16;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gCol = col0 + tx + j * 16;
            acc[i][j] = (kBegin > 0 && gRow < M && gCol < N)
                            ? C[static_cast<size_t>(gRow) * N + gCol]
                            : 0.0;
        }
    }

    for (int kt = kBegin; kt < kEnd; kt += BK) {
        // Stage the A and B tiles in shared memory (out-of-range entries read as 0).
        #pragma unroll
        for (int r = 0; r < BM; r += 16) {
            const int m = aM + r;
            const int gRow = row0 + m;
            const int gCol = kt + aK;
            As[aK][m] = (gRow < M && gCol < N) ? A[static_cast<size_t>(gRow) * N + gCol] : 0.0;
        }
        #pragma unroll
        for (int r = 0; r < BK; r += THREADS / BN) {
            const int k = bK + r;
            const int gRow = kt + k;
            const int gCol = col0 + bN;
            Bs[k][bN] = (gRow < N && gCol < N) ? B[static_cast<size_t>(gRow) * N + gCol] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][ty + i * 16];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][tx + j * 16];
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
        const int gRow = row0 + ty + i * 16;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gCol = col0 + tx + j * 16;
            if (gCol < N) {
                C[static_cast<size_t>(gRow) * N + gCol] = acc[i][j];
            }
        }
    }
}

// Target size of one k panel of B; panels let the upload of B overlap the kernels.
constexpr size_t PANEL_TARGET_BYTES = 32u << 20;

// Per-GPU resources. Allocating device memory, creating streams and page-locking
// the host matrices are expensive one-off operations, so they happen before the
// benchmark, just like the allocation and initialization of the host matrices.
struct DeviceResources {
    int device = 0;
    size_t rowBegin = 0; // first row of C computed on this device
    size_t rows = 0;     // number of rows of C computed on this device
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t sUp{}, sComp{}, sDown{};
    cudaEvent_t panelReady{}, compDone{};
};

static std::vector<DeviceResources> g_devices;
static size_t g_panelRows = 0;
static std::vector<void*> g_registered;

// Page-locks a host matrix so that the GPUs can DMA from/to it directly. This is
// only an optimization: if it fails the transfers still work, just slower.
static void registerHostMemory(const void* ptr, const size_t bytes) {
    void* p = const_cast<void*>(ptr);
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) == cudaSuccess) {
        g_registered.push_back(p);
    } else {
        cudaGetLastError(); // clear the error, fall back to pageable transfers
    }
}

// Splits the rows of C evenly (in units of the BM tile) over the GPUs that are
// worth using and allocates their buffers.
void setupDevices(const size_t N, const double* A, const double* B, double* C) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA device available\n");
        exit(1);
    }
    if (N == 0) return;

    // Every GPU needs its own copy of B, so a GPU only pays off once its share of
    // the O(N^3) work outweighs that O(N^2) upload; below ~1024 rows per GPU the
    // transfers dominate.
    const int useDevices =
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(deviceCount, N / 1024)));
    printf("CUDA devices: %d of %d\n", useDevices, deviceCount);

    const size_t bytes = N * N * sizeof(double);
    registerHostMemory(A, bytes);
    registerHostMemory(B, bytes);
    registerHostMemory(C, bytes);

    g_panelRows = std::max<size_t>(PANEL_TARGET_BYTES / (N * sizeof(double)), 1);
    g_panelRows = std::min(((g_panelRows + BK - 1) / BK) * BK, N);

    const size_t tiles = (N + BM - 1) / BM;
    const size_t tilesPerDevice = (tiles + useDevices - 1) / useDevices;

    for (int d = 0; d < useDevices; ++d) {
        DeviceResources r;
        r.device = d;
        r.rowBegin = std::min(N, d * tilesPerDevice * BM);
        r.rows = std::min(N, r.rowBegin + tilesPerDevice * BM) - r.rowBegin;
        if (r.rows == 0) break;

        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&r.dA, r.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&r.dC, r.rows * N * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&r.sUp));
        CUDA_CHECK(cudaStreamCreate(&r.sComp));
        CUDA_CHECK(cudaStreamCreate(&r.sDown));
        CUDA_CHECK(cudaEventCreateWithFlags(&r.panelReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&r.compDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaDeviceSynchronize());
        g_devices.push_back(r);
    }
}

void teardownDevices() {
    for (auto& r : g_devices) {
        CUDA_CHECK(cudaSetDevice(r.device));
        CUDA_CHECK(cudaEventDestroy(r.panelReady));
        CUDA_CHECK(cudaEventDestroy(r.compDone));
        CUDA_CHECK(cudaStreamDestroy(r.sUp));
        CUDA_CHECK(cudaStreamDestroy(r.sComp));
        CUDA_CHECK(cudaStreamDestroy(r.sDown));
        CUDA_CHECK(cudaFree(r.dA));
        CUDA_CHECK(cudaFree(r.dB));
        CUDA_CHECK(cudaFree(r.dC));
    }
    g_devices.clear();
    for (void* p : g_registered) {
        CUDA_CHECK(cudaHostUnregister(p));
    }
    g_registered.clear();
}

// Computes this device's slice of C: upload its rows of A, then stream B panel by
// panel, running the kernel for one panel while the next one is still uploading.
static void multiplyRowRange(DeviceResources& r, const double* A, const double* B, double* C,
                             const size_t N) {
    CUDA_CHECK(cudaSetDevice(r.device));

    CUDA_CHECK(cudaMemcpyAsync(r.dA, A + r.rowBegin * N, r.rows * N * sizeof(double),
                               cudaMemcpyHostToDevice, r.sUp));

    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((r.rows + BM - 1) / BM));
    for (size_t k0 = 0; k0 < N; k0 += g_panelRows) {
        const size_t k1 = std::min(N, k0 + g_panelRows);
        CUDA_CHECK(cudaMemcpyAsync(r.dB + k0 * N, B + k0 * N, (k1 - k0) * N * sizeof(double),
                                   cudaMemcpyHostToDevice, r.sUp));
        // sUp is in-order, so this also covers the A upload and all earlier panels.
        CUDA_CHECK(cudaEventRecord(r.panelReady, r.sUp));
        CUDA_CHECK(cudaStreamWaitEvent(r.sComp, r.panelReady, 0));
        matmulKernel<<<grid, THREADS, 0, r.sComp>>>(r.dA, r.dB, r.dC, static_cast<int>(N),
                                                    static_cast<int>(r.rows),
                                                    static_cast<int>(k0), static_cast<int>(k1));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(r.compDone, r.sComp));
    CUDA_CHECK(cudaStreamWaitEvent(r.sDown, r.compDone, 0));
    CUDA_CHECK(cudaMemcpyAsync(C + r.rowBegin * N, r.dC, r.rows * N * sizeof(double),
                               cudaMemcpyDeviceToHost, r.sDown));
    CUDA_CHECK(cudaStreamSynchronize(r.sDown));
}

// Drives one host thread per GPU, each computing its own horizontal slice of C.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    std::vector<std::thread> workers;
    workers.reserve(g_devices.size());
    for (auto& r : g_devices) {
        workers.emplace_back(multiplyRowRange, std::ref(r), A.data(), B.data(), C.data(), N);
    }
    for (auto& w : workers) {
        w.join();
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
    
    // Set up the GPU contexts and buffers up front, so that (like the host
    // matrices above) only the multiplication itself is timed
    setupDevices(N, A.data(), B.data(), C.data());

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    teardownDevices();
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
