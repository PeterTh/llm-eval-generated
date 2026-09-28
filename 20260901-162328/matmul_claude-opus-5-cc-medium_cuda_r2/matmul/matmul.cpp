#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                              \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,  \
                   cudaGetErrorString(err_));                                                   \
            exit(1);                                                                            \
        }                                                                                       \
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

// --- GPU matrix multiplication -------------------------------------------------------------
//
// The rows of C are split across all available GPUs. Each device processes its row block in
// chunks of the k dimension so that the host -> device transfers of the next chunk overlap
// with the computation of the current one (double buffering, two streams).
//
// Each block computes a TILE x TILE tile of C from shared-memory tiles of A and B. The k loop
// is traversed strictly in increasing order (chunks in order, tiles in order within a chunk,
// and elements in order within a tile), and every chunk resumes from the running sum left in
// C by the previous one. Every element of C is therefore accumulated in exactly the same order
// as in the serial reference implementation, giving bit-identical results.

constexpr int TILE = 32;    // tile extent in all three dimensions
constexpr int BLOCK_Y = 8;  // threads in y; each thread handles TILE / BLOCK_Y rows
constexpr int ROWS_PER_THREAD = TILE / BLOCK_Y;
constexpr int K_CHUNK = 512;  // k-dimension chunk size (multiple of TILE) used for pipelining

// A holds the device's row block for the current k chunk with row pitch kc; B holds rows
// [k0, k0 + kc) of the full matrix; C is the device's row block of the result.
__global__ __launch_bounds__(TILE* BLOCK_Y) void matrixMultiplyKernel(
    const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
    const int N, const int rows, const int kc, const bool resume) {
    __shared__ double As[TILE][TILE + 1];  // +1 padding avoids shared-memory bank conflicts
    __shared__ double Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * TILE;  // first row of this C tile
    const int col = blockIdx.x * TILE + tx;

    double acc[ROWS_PER_THREAD];
#pragma unroll
    for (int u = 0; u < ROWS_PER_THREAD; ++u) {
        const int r = row0 + ty + u * BLOCK_Y;
        acc[u] = (resume && r < rows && col < N) ? C[static_cast<size_t>(r) * N + col] : 0.0;
    }

    for (int kt = 0; kt < kc; kt += TILE) {
        // Cooperatively load one tile of A and one tile of B (out-of-range entries are zero,
        // and adding those zero products leaves the running sums unchanged).
#pragma unroll
        for (int u = 0; u < ROWS_PER_THREAD; ++u) {
            const int r = ty + u * BLOCK_Y;
            const int ar = row0 + r;
            const int ac = kt + tx;
            As[r][tx] = (ar < rows && ac < kc) ? A[static_cast<size_t>(ar) * kc + ac] : 0.0;
            const int br = kt + r;
            Bs[r][tx] = (br < kc && col < N) ? B[static_cast<size_t>(br) * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
#pragma unroll
            for (int u = 0; u < ROWS_PER_THREAD; ++u) {
                acc[u] += As[ty + u * BLOCK_Y][k] * b;
            }
        }
        __syncthreads();
    }

    if (col < N) {
#pragma unroll
        for (int u = 0; u < ROWS_PER_THREAD; ++u) {
            const int r = row0 + ty + u * BLOCK_Y;
            if (r < rows) {
                C[static_cast<size_t>(r) * N + col] = acc[u];
            }
        }
    }
}

struct DeviceWork {
    int device = 0;
    int rowBegin = 0;
    int rowCount = 0;
    double* dA[2] = {nullptr, nullptr};  // row block of A for one k chunk (double buffered)
    double* dB[2] = {nullptr, nullptr};  // rows of B for one k chunk (double buffered)
    double* dC = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaStream_t compStream = nullptr;
    cudaEvent_t copyDone[2] = {nullptr, nullptr};
    cudaEvent_t compDone[2] = {nullptr, nullptr};
};

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    const std::vector<DeviceWork>& work) {
    const int n = static_cast<int>(N);
    const dim3 block(TILE, BLOCK_Y);

    for (const auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        const dim3 grid((n + TILE - 1) / TILE, (w.rowCount + TILE - 1) / TILE);

        int chunk = 0;
        for (int k0 = 0; k0 < n; k0 += K_CHUNK, ++chunk) {
            const int kc = std::min(K_CHUNK, n - k0);
            const int buf = chunk & 1;

            // Wait until the kernel that last read this buffer pair has finished.
            if (chunk >= 2) {
                CUDA_CHECK(cudaStreamWaitEvent(w.copyStream, w.compDone[buf], 0));
            }
            // Columns [k0, k0 + kc) of the device's row block of A.
            CUDA_CHECK(cudaMemcpy2DAsync(
                w.dA[buf], static_cast<size_t>(kc) * sizeof(double),
                A.data() + static_cast<size_t>(w.rowBegin) * N + k0, N * sizeof(double),
                static_cast<size_t>(kc) * sizeof(double), static_cast<size_t>(w.rowCount),
                cudaMemcpyHostToDevice, w.copyStream));
            // Rows [k0, k0 + kc) of B.
            CUDA_CHECK(cudaMemcpyAsync(w.dB[buf], B.data() + static_cast<size_t>(k0) * N,
                                       static_cast<size_t>(kc) * N * sizeof(double),
                                       cudaMemcpyHostToDevice, w.copyStream));
            CUDA_CHECK(cudaEventRecord(w.copyDone[buf], w.copyStream));

            CUDA_CHECK(cudaStreamWaitEvent(w.compStream, w.copyDone[buf], 0));
            matrixMultiplyKernel<<<grid, block, 0, w.compStream>>>(w.dA[buf], w.dB[buf], w.dC, n,
                                                                   w.rowCount, kc, k0 > 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(w.compDone[buf], w.compStream));
        }

        CUDA_CHECK(cudaMemcpyAsync(C.data() + static_cast<size_t>(w.rowBegin) * N, w.dC,
                                   static_cast<size_t>(w.rowCount) * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, w.compStream));
    }

    for (const auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaStreamSynchronize(w.compStream));
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

    // Set up the GPUs: split the rows of C evenly (in units of whole tiles) across devices.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable device found\n");
        return 1;
    }

    const size_t totalTiles = (N + TILE - 1) / TILE;
    const int usedDevices =
        static_cast<int>(std::min<size_t>(static_cast<size_t>(deviceCount), totalTiles));
    printf("Using %d CUDA device(s)\n", usedDevices);

    std::vector<DeviceWork> work;
    work.reserve(usedDevices);
    size_t tileBegin = 0;
    for (int d = 0; d < usedDevices; ++d) {
        const size_t tiles = totalTiles / usedDevices + (static_cast<size_t>(d) < totalTiles % usedDevices ? 1 : 0);
        DeviceWork w;
        w.device = d;
        w.rowBegin = static_cast<int>(tileBegin * TILE);
        w.rowCount = static_cast<int>(std::min(N, (tileBegin + tiles) * TILE) - tileBegin * TILE);
        tileBegin += tiles;

        const size_t kc = std::min<size_t>(K_CHUNK, N);
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&w.copyStream));
        CUDA_CHECK(cudaStreamCreate(&w.compStream));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&w.copyDone[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&w.compDone[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(&w.dA[b], static_cast<size_t>(w.rowCount) * kc * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&w.dB[b], kc * N * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&w.dC, static_cast<size_t>(w.rowCount) * N * sizeof(double)));
        work.push_back(w);
    }

    // Warm up every device (context and kernel module load) so that this one-off cost does not
    // land in the measured region. Passing rows == 0 makes the launch a no-op.
    for (const auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        matrixMultiplyKernel<<<dim3(1, 1), dim3(TILE, BLOCK_Y), 0, w.compStream>>>(
            w.dA[0], w.dB[0], w.dC, static_cast<int>(N), 0,
            static_cast<int>(std::min<size_t>(TILE, N)), false);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(w.compStream));
    }

    // Page-locking the host buffers speeds up the host <-> device transfers.
    if (N > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), N * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(C.data(), N * N * sizeof(double), cudaHostRegisterDefault));
    }

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N, work);

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
    for (const auto& w : work) {
        CUDA_CHECK(cudaSetDevice(w.device));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaFree(w.dA[b]));
            CUDA_CHECK(cudaFree(w.dB[b]));
            CUDA_CHECK(cudaEventDestroy(w.copyDone[b]));
            CUDA_CHECK(cudaEventDestroy(w.compDone[b]));
        }
        CUDA_CHECK(cudaFree(w.dC));
        CUDA_CHECK(cudaStreamDestroy(w.copyStream));
        CUDA_CHECK(cudaStreamDestroy(w.compStream));
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
