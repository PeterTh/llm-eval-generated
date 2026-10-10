#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

namespace {
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A block produces a 64x64 output tile. Each of its 256 threads keeps
// a 4x4 output tile in registers, reusing each loaded operand four times.
constexpr unsigned Tile = 64;
constexpr unsigned Depth = 16;
constexpr unsigned Threads = 16;

__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    // Transpose A while loading so that both operands have contiguous
    // shared-memory accesses during accumulation. Padding avoids bank conflicts.
    __shared__ double aTile[Depth][Tile + 1];
    __shared__ double bTile[Depth][Tile];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * Threads + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * Tile;
    double accum[4][4] = {};

    for (size_t base = 0; base < N; base += Depth) {
        // Cooperative, coalesced global loads, including partial edge tiles.
        #pragma unroll
        for (unsigned offset = 0; offset < Tile * Depth; offset += Threads * Threads) {
            const unsigned index = tid + offset;
            const unsigned ar = index / Depth;
            const unsigned ak = index % Depth;
            const unsigned bk = index / Tile;
            const unsigned bc = index % Tile;
            aTile[ak][ar] = rowBase + ar < N && base + ak < N
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            bTile[bk][bc] = base + bk < N && colBase + bc < N
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < Depth; ++k) {
            double av[4], bv[4];
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                av[i] = aTile[k][ty + i * Threads];
                bv[i] = bTile[k][tx + i * Threads];
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    accum[i][j] += av[i] * bv[j];
                }
            }
        }
        // No thread may replace the tile until all threads have consumed it.
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const size_t row = rowBase + ty + i * Threads;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const size_t col = colBase + tx + j * Threads;
            if (row < N && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}
} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocate A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocate B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocate C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "upload A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "upload B");

    const size_t tiles = (N + Tile - 1) / Tile;
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "get device");
    checkCuda(cudaGetDeviceProperties(&properties, device), "get device properties");
    if (tiles > static_cast<size_t>(properties.maxGridSize[0]) ||
        tiles > static_cast<size_t>(properties.maxGridSize[1])) {
        fprintf(stderr, "Matrix size exceeds CUDA grid limits\n");
        std::exit(EXIT_FAILURE);
    }
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(tiles));
    multiplyKernel<<<grid, dim3(Threads, Threads)>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launch matrix multiplication");
    // This blocking copy also waits for the kernel, so timing includes completed
    // GPU computation and both transfer directions, and C is ready for validation.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "download C");
    checkCuda(cudaFree(deviceC), "free C");
    checkCuda(cudaFree(deviceB), "free B");
    checkCuda(cudaFree(deviceA), "free A");
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
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
    
    if (N != 0 && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    // Initialize the CUDA context outside the computation timer. CUDA is
    // required even for an empty matrix; there is no CPU multiplication fallback.
    checkCuda(cudaFree(nullptr), "initialize runtime");

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS
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
}
