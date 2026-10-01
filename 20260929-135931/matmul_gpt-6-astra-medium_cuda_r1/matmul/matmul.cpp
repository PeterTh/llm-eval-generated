#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>

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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceMatrix {
public:
    explicit DeviceMatrix(size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }
    ~DeviceMatrix() { cudaFree(data); }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
    double* data = nullptr;
};

// Each warp accesses consecutive columns. Threads accumulate a small register
// tile, reusing both shared-memory operands across multiple output elements.
template <int Tile>
__global__ void multiplyTiles(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N) {
    constexpr int KTile = 16;
    constexpr int Rows = Tile / 8;
    constexpr int Cols = Tile / 32;
    __shared__ double aTile[Tile][KTile];
    __shared__ double bTile[KTile][Tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tid = y * 32 + x;
    const size_t tiles = (N + Tile - 1) / Tile;

    // Flattening the grid avoids the small grid.y limit on large matrices.
    for (size_t tile = blockIdx.x; tile < tiles * tiles; tile += gridDim.x) {
        const size_t row = (tile / tiles) * Tile;
        const size_t col = (tile % tiles) * Tile;
        double sums[Rows][Cols] = {};
        for (size_t base = 0; base < N; base += KTile) {
            #pragma unroll
            for (int p = tid; p < Tile * KTile; p += 256) {
                const int r = p / KTile, k = p % KTile;
                aTile[r][k] = row + r < N && base + k < N
                    ? A[(row + r) * N + base + k] : 0.0;
                const int bk = p / Tile, c = p % Tile;
                bTile[bk][c] = base + bk < N && col + c < N
                    ? B[(base + bk) * N + col + c] : 0.0;
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < KTile; ++k) {
                if (base + k < N) {
                    double av[Rows], bv[Cols];
                    #pragma unroll
                    for (int i = 0; i < Rows; ++i) av[i] = aTile[y + i * 8][k];
                    #pragma unroll
                    for (int j = 0; j < Cols; ++j) bv[j] = bTile[k][x + j * 32];
                    #pragma unroll
                    for (int i = 0; i < Rows; ++i) {
                        #pragma unroll
                        for (int j = 0; j < Cols; ++j) {
                            // Match the CPU's ordered, separately rounded multiply/add.
                            sums[i][j] = __dadd_rn(sums[i][j], __dmul_rn(av[i], bv[j]));
                        }
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < Rows; ++i) {
            #pragma unroll
            for (int j = 0; j < Cols; ++j) {
                const size_t r = row + y + i * 8, c = col + x + j * 32;
                if (r < N && c < N) C[r * N + c] = sums[i][j];
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    DeviceMatrix& deviceA, DeviceMatrix& deviceB, DeviceMatrix& deviceC) {
    const size_t bytes = N * N * sizeof(double);
    checkCuda(cudaMemcpy(deviceA.data, A.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    checkCuda(cudaMemcpy(deviceB.data, B.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");
    const size_t tileSize = N <= 128 ? 32 : 64;
    const size_t tiles = (N + tileSize - 1) / tileSize;
    const unsigned blocks = static_cast<unsigned>(std::min(tiles * tiles, size_t{65535}));
    const dim3 threads(32, 8);
    if (tileSize == 32) {
        multiplyTiles<32><<<blocks, threads>>>(deviceA.data, deviceB.data, deviceC.data, N);
    } else {
        multiplyTiles<64><<<blocks, threads>>>(deviceA.data, deviceB.data, deviceC.data, N);
    }
    checkCuda(cudaGetLastError(), "launch matrix multiplication");
    // The blocking copy also waits for the kernel and reports execution errors.
    checkCuda(cudaMemcpy(C.data(), deviceC.data, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");
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

int main(int argc, char** argv) try {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || value == end || *end != '\0' || *value == '-' ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Matrix size must be a positive integer");
            }
            N = static_cast<size_t>(parsed);
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
    
    if (N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        throw std::runtime_error("Matrix size is too large");
    }

    // Allocate matrices and initialize the CUDA context before timing, just as
    // host allocation and initialization are excluded from the original timer.
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    const size_t bytes = N * N * sizeof(double);
    DeviceMatrix deviceA(bytes), deviceB(bytes), deviceC(bytes);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, deviceA, deviceB, deviceC);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
} catch (const std::exception& error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
}
