#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

namespace {
constexpr int tile = 32;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Shared-memory diagonal factorization. All threads take the same failure path.
__global__ void factorDiagonal(double* a, size_t n, size_t offset, int* failure) {
    __shared__ double d[tile][tile + 1];
    const int t = threadIdx.x;
    const int width = static_cast<int>(min(size_t(tile), n - offset));
    if (*failure) return;
    for (int q = t; q < tile * tile; q += blockDim.x) {
        int r = q / tile, c = q % tile;
        d[r][c] = r < width && c <= r ? a[(offset + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (t == 0) {
            if (d[k][k] <= 0.0) *failure = static_cast<int>(offset + k + 1);
            else d[k][k] = sqrt(d[k][k]);
        }
        __syncthreads();
        if (*failure) return;
        if (t > k && t < width) d[t][k] /= d[k][k];
        __syncthreads();
        for (int q = t; q < tile * tile; q += blockDim.x) {
            int r = q / tile, c = q % tile;
            if (r < width && c > k && c <= r)
                d[r][c] -= d[r][k] * d[c][k];
        }
        __syncthreads();
    }
    for (int q = t; q < tile * tile; q += blockDim.x) {
        int r = q / tile, c = q % tile;
        if (r < width && c <= r) a[(offset + r) * n + offset + c] = d[r][c];
    }
}

// Solve each panel row against the small diagonal block in shared memory.
__global__ void solvePanel(double* a, size_t n, size_t offset, const int* failure) {
    __shared__ double d[tile][tile + 1];
    __shared__ double p[128][tile + 1];
    if (*failure) return;
    const int t = threadIdx.x;
    const size_t first = offset + tile + size_t(blockIdx.x) * 128;
    for (int q = t; q < tile * tile; q += 128) {
        int r = q / tile, c = q % tile;
        d[r][c] = a[(offset + r) * n + offset + c];
    }
    // Load consecutive columns together, then let one thread solve each row.
    for (int q = t; q < 128 * tile; q += 128) {
        int r = q / tile, c = q % tile;
        p[r][c] = first + r < n ? a[(first + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    if (first + t < n) {
        for (int j = 0; j < tile; ++j) {
            double v = p[t][j];
            for (int k = 0; k < j; ++k) v -= p[t][k] * d[j][k];
            p[t][j] = v / d[j][j];
        }
    }
    __syncthreads();
    for (int q = t; q < 128 * tile; q += 128) {
        int r = q / tile, c = q % tile;
        if (first + r < n) a[(first + r) * n + offset + c] = p[r][c];
    }
}

// Four output rows per thread reuse the shared tiles and keep accesses coalesced.
// Update=true performs the lower-triangular trailing update; otherwise form B B^T.
template<bool Update>
__global__ void gram(double* out, const double* b, size_t n,
                     size_t offset, const int* failure) {
    if (blockIdx.x > blockIdx.y) return;
    if (Update && *failure) return;
    __shared__ double left[tile][tile + 1], right[tile][tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = Update ? offset + tile : 0;
    const size_t row = base + size_t(blockIdx.y) * tile;
    const size_t col = base + size_t(blockIdx.x) * tile;
    double accum[4] = {0.0, 0.0, 0.0, 0.0};
    const size_t end = Update ? offset + tile : n;
    for (size_t k = Update ? offset : 0; k < end; k += tile) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            int r = y + 8 * v;
            left[r][x] = row + r < n && k + x < n ? b[(row + r) * n + k + x] : 0.0;
            right[r][x] = col + r < n && k + x < n ? b[(col + r) * n + k + x] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < tile; ++j) {
            double rv = right[x][j];
            #pragma unroll
            for (int v = 0; v < 4; ++v) accum[v] += left[y + 8 * v][j] * rv;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        const size_t r = row + y + 8 * v, c = col + x;
        if (r < n && c <= r) {
            if (Update) out[r * n + c] -= accum[v];
            else {
                out[r * n + c] = accum[v];
                if (r != c) out[c * n + r] = accum[v];
            }
        }
    }
}

__global__ void finishMatrix(double* a, size_t n, bool zeroUpper) {
    for (size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         q < n * n; q += size_t(blockDim.x) * gridDim.x) {
        size_t r = q / n, c = q % n;
        if (zeroUpper) { if (c > r) a[q] = 0.0; }
        else if (r == c) a[q] += n;
    }
}
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    DeviceBuffer<double> a(A.size());
    DeviceBuffer<int> failure(1);
    checkCuda(cudaMemcpy(a.data, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(failure.data, 0, sizeof(int)));
    for (size_t k = 0; k < n; k += tile) {
        factorDiagonal<<<1, 128>>>(a.data, n, k, failure.data);
        if (k + tile < n) {
            size_t remaining = n - k - tile;
            solvePanel<<<(remaining + 127) / 128, 128>>>(a.data, n, k, failure.data);
            unsigned blocks = (remaining + tile - 1) / tile;
            gram<true><<<dim3(blocks, blocks), dim3(32, 8)>>>(a.data, a.data, n, k, failure.data);
        }
    }
    if (n) finishMatrix<<<std::min(size_t(65535), (A.size() + 255) / 256), 256>>>(a.data, n, true);
    checkCuda(cudaGetLastError());
    int bad = 0;
    checkCuda(cudaMemcpy(&bad, failure.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", bad - 1);
        return false;
    }
    checkCuda(cudaMemcpy(A.data(), a.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    DeviceBuffer<double> b(B.size()), a(A.size());
    checkCuda(cudaMemcpy(b.data, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (n) {
        unsigned blocks = (n + tile - 1) / tile;
        gram<false><<<dim3(blocks, blocks), dim3(32, 8)>>>(a.data, b.data, n, 0, nullptr);
        finishMatrix<<<std::min(size_t(65535), (A.size() + 255) / 256), 256>>>(a.data, n, false);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(A.data(), a.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    DeviceBuffer<double> l(L.size()), product(L.size());
    checkCuda(cudaMemcpy(l.data, L.data(), L.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (n) {
        unsigned blocks = (n + tile - 1) / tile;
        gram<false><<<dim3(blocks, blocks), dim3(32, 8)>>>(product.data, l.data, n, 0, nullptr);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(reconstructed.data(), product.data, L.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
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
