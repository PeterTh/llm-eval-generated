#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA is mandatory: allocation, launch and execution errors are fatal.
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int tile = 32;

// Each block computes a 32x32 lower-triangular tile of X*X^T.
// Four accumulators per thread reuse coalesced shared-memory loads.
// Update mode reads only the completed panel, disjoint from its output.
template<bool update>
__global__ void gram(const double* X, double* A, size_t n,
                     size_t offset, size_t first, size_t last, double diagonal) {
    if (blockIdx.x > blockIdx.y) return;
    __shared__ double left[tile][tile + 1], right[tile][tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = offset + blockIdx.y * tile;
    const size_t col = offset + blockIdx.x * tile;
    double sums[2][2] = {};
    for (size_t k = first; k < last; k += tile) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = y + i * 16, c = x + j * 16;
                left[r][c] = row + r < n && k + c < last ? X[(row + r)*n + k + c] : 0.0;
                right[r][c] = col + r < n && k + c < last ? X[(col + r)*n + k + c] : 0.0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < tile; ++q) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    sums[i][j] += left[y + i*16][q] * right[x + j*16][q];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const size_t r = row + y + i*16, c = col + x + j*16;
            if (r < n && c <= r && c < n) {
                if (update) A[r*n + c] -= sums[i][j];
                else {
                    const double value = sums[i][j] + (r == c ? diagonal : 0.0);
                    A[r*n + c] = value;
                    if (r != c) A[c*n + r] = value;
                }
            }
        }
    }
}

// The only serial dependency is between pivots within the diagonal tile.
// All divisions and rank-one updates for each pivot execute cooperatively.
__global__ void factorDiagonal(double* A, size_t n, size_t base, int width, int* failure) {
    if (*failure >= 0) return;
    __shared__ double d[tile][tile + 1];
    __shared__ int bad;
    const int t = threadIdx.x;
    if (t == 0) bad = 0;
    for (int p = t; p < tile*tile; p += blockDim.x) {
        int r = p / tile, c = p % tile;
        d[r][c] = r < width && c <= r ? A[(base+r)*n + base+c] : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (t == 0) {
            if (!(d[k][k] > 0.0)) { bad = 1; *failure = static_cast<int>(base + k); }
            else d[k][k] = sqrt(d[k][k]);
        }
        __syncthreads();
        if (bad) return;
        if (t > k && t < width) d[t][k] /= d[k][k];
        __syncthreads();
        for (int p = t; p < tile*tile; p += blockDim.x) {
            int r = p / tile, c = p % tile;
            if (r < width && c > k && c <= r) d[r][c] -= d[r][k]*d[c][k];
        }
        __syncthreads();
    }
    for (int p = t; p < tile*tile; p += blockDim.x) {
        int r = p / tile, c = p % tile;
        if (r < width && c <= r) A[(base+r)*n + base+c] = d[r][c];
    }
}

__global__ void solvePanel(double* A, size_t n, size_t base, int* failure) {
    if (*failure >= 0) return;
    __shared__ double d[tile][tile + 1], p[tile][tile + 1];
    const int t = threadIdx.x;
    const size_t row = base + tile + blockIdx.x * tile;
    for (int idx = t; idx < tile*tile; idx += blockDim.x) {
        int r = idx / tile, c = idx % tile;
        d[r][c] = A[(base+r)*n + base+c];
        p[r][c] = row+r < n ? A[(row+r)*n + base+c] : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < tile; ++k) {
        if (t < tile) p[t][k] /= d[k][k];
        __syncthreads();
        for (int idx = t; idx < tile*tile; idx += blockDim.x) {
            int r = idx / tile, c = idx % tile;
            if (c > k) p[r][c] -= p[r][k]*d[c][k];
        }
        __syncthreads();
    }
    for (int idx = t; idx < tile*tile; idx += blockDim.x) {
        int r = idx / tile, c = idx % tile;
        if (row+r < n) A[(row+r)*n + base+c] = p[r][c];
    }
}

__global__ void clearUpper(double* A, size_t n) {
    for (size_t i = blockIdx.x*size_t(blockDim.x) + threadIdx.x;
         i < n*n; i += size_t(gridDim.x)*blockDim.x)
        if (i % n > i / n) A[i] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (!n) { cudaCheck(cudaFree(nullptr)); return true; }
    DeviceBuffer<double> matrix(n*n);
    DeviceBuffer<int> failure(1);
    cudaCheck(cudaMemcpy(matrix.data, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(failure.data, 0xff, sizeof(int)));
    for (size_t base = 0; base < n; base += tile) {
        const int width = static_cast<int>(std::min(size_t(tile), n-base));
        factorDiagonal<<<1, 256>>>(matrix.data, n, base, width, failure.data);
        if (base + tile < n) {
            unsigned blocks = static_cast<unsigned>((n-base-tile+tile-1)/tile);
            solvePanel<<<blocks, 256>>>(matrix.data, n, base, failure.data);
            gram<true><<<dim3(blocks, blocks), dim3(16,16)>>>(
                matrix.data, matrix.data, n, base+tile, base, base+tile, 0.0);
        }
    }
    clearUpper<<<static_cast<unsigned>(std::min(size_t(65535), (n*n+255)/256)), 256>>>(matrix.data, n);
    cudaCheck(cudaGetLastError());
    int bad;
    cudaCheck(cudaMemcpy(&bad, failure.data, sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(A.data(), matrix.data, n*n*sizeof(double), cudaMemcpyDeviceToHost));
    if (bad >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", bad);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    cudaCheck(cudaFree(nullptr));
    if (!n) return;
    std::vector<double> B(n*n);
    unsigned int seed = 42;
    // Preserve rand_r's original sequence exactly.
    for (size_t i = 0; i < n*n; ++i) B[i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    DeviceBuffer<double> input(n*n), output(n*n);
    cudaCheck(cudaMemcpy(input.data, B.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));
    unsigned blocks = static_cast<unsigned>((n+tile-1)/tile);
    gram<false><<<dim3(blocks,blocks), dim3(16,16)>>>(input.data, output.data, n, 0, 0, n, double(n));
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(A.data(), output.data, n*n*sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n*n);
    if (n) {
        DeviceBuffer<double> input(n*n), output(n*n);
        cudaCheck(cudaMemcpy(input.data, L.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));
        unsigned blocks = static_cast<unsigned>((n+tile-1)/tile);
        gram<false><<<dim3(blocks,blocks), dim3(16,16)>>>(input.data, output.data, n, 0, 0, n, 0.0);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(reconstructed.data(), output.data, n*n*sizeof(double), cudaMemcpyDeviceToHost));
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n*n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (!std::isfinite(error)) {
            printf("Validation failed: non-finite result\n");
            return false;
        }
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Guard size arithmetic and CUDA's two-dimensional grid limit.
    if (n > size_t(65535)*tile || (n && n > std::numeric_limits<size_t>::max()/sizeof(double)/n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
