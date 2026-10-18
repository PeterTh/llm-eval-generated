#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {
constexpr int tile = 32;

void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// One block owns a diagonal tile. All dependent columns stay in shared memory.
__global__ void factorDiagonal(double* a, size_t n, size_t offset, int* status) {
    __shared__ double d[tile][tile + 1];
    if (*status != INT_MAX) return;
    const int x = threadIdx.x;
    const int width = static_cast<int>(n - offset < tile ? n - offset : tile);
    for (int y = 0; y < width; ++y)
        if (x < width) d[y][x] = a[(offset + y) * n + offset + x];
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (x == k) {
            if (!(d[k][k] > 0.0)) *status = static_cast<int>(offset + k);
            else d[k][k] = sqrt(d[k][k]);
        }
        __syncthreads();
        if (*status != INT_MAX) return;
        if (x > k && x < width) d[x][k] /= d[k][k];
        __syncthreads();
        if (x > k && x < width)
            for (int y = x; y < width; ++y)
                d[y][x] -= d[y][k] * d[x][k];
        __syncthreads();
    }
    for (int y = 0; y < width; ++y)
        if (x <= y) a[(offset + y) * n + offset + x] = d[y][x];
}

// Independent rows solve X * L_diagonal^T = A_panel in parallel.
__global__ void solvePanel(double* a, size_t n, size_t offset, int* status) {
    __shared__ double d[tile][tile + 1];
    __shared__ double p[tile][tile + 1];
    if (*status != INT_MAX) return;
    const int x = threadIdx.x;
    const size_t base = offset + tile + static_cast<size_t>(blockIdx.x) * tile;
    for (int y = 0; y < tile; ++y) {
        d[y][x] = a[(offset + y) * n + offset + x];
        p[y][x] = base + y < n ? a[(base + y) * n + offset + x] : 0.0;
    }
    __syncthreads();
    if (base + x < n) {
        for (int k = 0; k < tile; ++k) {
            double value = p[x][k];
            for (int j = 0; j < k; ++j) value -= p[x][j] * d[k][j];
            p[x][k] = value / d[k][k];
        }
    }
    __syncthreads();
    for (int y = 0; y < tile; ++y)
        if (base + y < n) a[(base + y) * n + offset + x] = p[y][x];
}

// Shared-memory tiled Gram product. Each thread accumulates four outputs.
// Update mode computes only the lower trailing triangle after a panel solve;
// product mode generates B*B^T (and optionally adds n*I).
template <bool update, bool addDiagonal>
__global__ void gramProduct(const double* b, double* a, size_t n,
                            size_t offset, const int* status) {
    if (update && *status != INT_MAX) return;
    if (blockIdx.y < blockIdx.x) return;
    __shared__ double left[tile][tile + 1];
    __shared__ double right[tile][tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t start = update ? offset + tile : 0;
    const size_t row = start + static_cast<size_t>(blockIdx.y) * tile;
    const size_t col = start + static_cast<size_t>(blockIdx.x) * tile;
    double sums[4] = {0.0, 0.0, 0.0, 0.0};
    const size_t end = update ? offset + tile : n;
    for (size_t k0 = update ? offset : 0; k0 < end; k0 += tile) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = y + q * 8;
            left[r][x] = row + r < n && k0 + x < n ? b[(row + r) * n + k0 + x] : 0.0;
            right[r][x] = col + r < n && k0 + x < n ? b[(col + r) * n + k0 + x] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile; ++k) {
            const double v = right[x][k];
            #pragma unroll
            for (int q = 0; q < 4; ++q) sums[q] += left[y + q * 8][k] * v;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const size_t i = row + y + q * 8, j = col + x;
        if (i < n && j < n && i >= j) {
            if (update) a[i * n + j] -= sums[q];
            else {
                const double value = sums[q] + (addDiagonal && i == j ? static_cast<double>(n) : 0.0);
                a[i * n + j] = value;
                if (i != j) a[j * n + i] = value;
            }
        }
    }
}

__global__ void clearUpper(double* a, size_t n) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        if (index % n > index / n) a[index] = 0.0;
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (!n) return true;
    DeviceBuffer<double> a(n * n);
    DeviceBuffer<int> status(1);
    int info = INT_MAX;
    cudaCheck(cudaMemcpy(a.data, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(status.data, &info, sizeof(info), cudaMemcpyHostToDevice));
    // The default stream orders the three dependent stages without host barriers.
    for (size_t offset = 0; offset < n; offset += tile) {
        factorDiagonal<<<1, tile>>>(a.data, n, offset, status.data);
        if (n - offset > tile) {
            const unsigned blocks = static_cast<unsigned>((n - offset - 1) / tile);
            solvePanel<<<blocks, tile>>>(a.data, n, offset, status.data);
            gramProduct<true, false><<<dim3(blocks, blocks), dim3(32, 8)>>>(
                a.data, a.data, n, offset, status.data);
        }
    }
    clearUpper<<<static_cast<unsigned>(std::min<size_t>((n * n + 255) / 256, 65535)), 256>>>(a.data, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&info, status.data, sizeof(info), cudaMemcpyDeviceToHost));
    if (info != INT_MAX) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info);
        return false;
    }
    cudaCheck(cudaMemcpy(A.data(), a.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    if (!n) return;
    DeviceBuffer<double> b(n * n), a(n * n);
    cudaCheck(cudaMemcpy(b.data, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>((n + tile - 1) / tile);
    gramProduct<false, true><<<dim3(blocks, blocks), dim3(32, 8)>>>(b.data, a.data, n, 0, nullptr);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(A.data(), a.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    if (n) {
        DeviceBuffer<double> l(n * n), product(n * n);
        cudaCheck(cudaMemcpy(l.data, L.data(), L.size() * sizeof(double), cudaMemcpyHostToDevice));
        const unsigned blocks = static_cast<unsigned>((n + tile - 1) / tile);
        gramProduct<false, false><<<dim3(blocks, blocks), dim3(32, 8)>>>(l.data, product.data, n, 0, nullptr);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(reconstructed.data(), product.data, reconstructed.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (!std::isfinite(error)) {
            printf("Validation failed: non-finite result\n");
            return false;
        }
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
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
            char* end = nullptr;
            const char* argument = argv[++i];
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end || parsed > INT_MAX ||
                parsed > 65535ULL * tile ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
    
    // Initialize CUDA before timing; all numerical work always uses the GPU.
    cudaCheck(cudaFree(nullptr));

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
