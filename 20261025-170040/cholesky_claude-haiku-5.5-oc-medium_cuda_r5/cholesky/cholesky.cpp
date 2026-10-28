#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA-parallel Cholesky decomposition (unblocked, row-oriented)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Algorithm:
//   for j = 0..n-1:
//     L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
//     for i = j+1..n-1 (all rows in parallel):
//       L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
//
// Column j depends only on columns < j, so each column is one kernel launch.
// Within a launch, one warp handles one row i: the lanes split the k-sum and
// the partial sums are combined with a warp shuffle reduction. The working
// matrix is row-major, so the lanes read consecutive k values (coalesced).
// The diagonal L[j][j] is recomputed by each warp from the same data (a cheap,
// identical computation), so no extra launch or synchronization is needed.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 8;
constexpr unsigned kBlockThreads = kWarpSize * kWarpsPerBlock;
constexpr unsigned kGramBlock = 16;
constexpr unsigned kFinalizeBlock = 256;

// RAII wrapper for device memory
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
};

__device__ __forceinline__ double warpReduceSum(double v) {
    // Result is valid in lane 0
    for (unsigned offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// One column step of the factorization. Warp w of block b handles row
// i = j + b*kWarpsPerBlock + w (i >= j). Row i of the working matrix W holds
// A[i][*] on entry and L[i][*] (for columns < j) on exit.
__global__ void choleskyColumnKernel(double* __restrict__ W,
                                     double* __restrict__ diag,
                                     int* __restrict__ firstFail,
                                     size_t n, size_t j) {
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned warp = threadIdx.x / kWarpSize;
    const size_t i = j + static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (i >= n) return;  // warp-uniform: whole warp exits together

    const double* rowI = W + i * n;
    const double* rowJ = W + j * n;

    double dsum = 0.0;  // sum_{k<j} L[j][k]^2
    double sum = 0.0;   // sum_{k<j} L[i][k] * L[j][k]
    #pragma unroll 4
    for (size_t k = lane; k < j; k += kWarpSize) {
        const double ljk = rowJ[k];
        dsum += ljk * ljk;
        sum += rowI[k] * ljk;
    }
    dsum = warpReduceSum(dsum);
    sum = warpReduceSum(sum);
    // Broadcast so every lane holds the identical values
    dsum = __shfl_sync(0xffffffffu, dsum, 0);
    sum = __shfl_sync(0xffffffffu, sum, 0);

    // L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2). A[j][j] is still the
    // original value: W[j][j] is never written during the factorization.
    const double val = W[j * n + j] - dsum;
    const double d = sqrt(val);

    if (lane != 0) return;
    if (i == j) {
        diag[j] = d;
        if (val <= 0.0) {
            atomicMin(firstFail, static_cast<int>(j));
        }
        return;
    }

    W[i * n + j] = (W[i * n + j] - sum) / d;
}

// Final pass: zero the strict upper triangle and place the diagonal.
__global__ void finalizeKernel(double* __restrict__ W, const double* __restrict__ diag, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const size_t r = idx / n;
    const size_t c = idx % n;
    if (c > r) {
        W[idx] = 0.0;
    } else if (c == r) {
        W[idx] = diag[r];
    }
}

// Gram product: Y = X * X^T (row-major), i.e. Y[i][j] = sum_k X[i][k] * X[j][k]
__global__ void gramKernel(const double* __restrict__ X, double* __restrict__ Y, size_t n) {
    const size_t j = static_cast<size_t>(blockIdx.x) * kGramBlock + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * kGramBlock + threadIdx.y;
    if (i >= n || j >= n) return;

    double sum = 0.0;
    for (size_t k = 0; k < n; ++k) {
        sum += X[i * n + k] * X[j * n + k];
    }
    Y[i * n + j] = sum;
}

// Y = X * X^T computed on the GPU (row-major n x n matrices)
void gramProduct(const std::vector<double>& X, std::vector<double>& Y, const size_t n) {
    if (n == 0) return;
    DeviceBuffer<double> dX(n * n);
    DeviceBuffer<double> dY(n * n);
    CUDA_CHECK(cudaMemcpy(dX.get(), X.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(kGramBlock, kGramBlock);
    const dim3 grid(static_cast<unsigned>((n + kGramBlock - 1) / kGramBlock),
                    static_cast<unsigned>((n + kGramBlock - 1) / kGramBlock));
    gramKernel<<<grid, block>>>(dX.get(), dY.get(), n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(Y.data(), dY.get(), n * n * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order; on success it holds L (row-major, zero upper part)
    if (n == 0) return true;

    const size_t nn = n * n;
    DeviceBuffer<double> dW(nn);     // row-major working matrix
    DeviceBuffer<double> dDiag(n);   // diagonal of L
    DeviceBuffer<int> dFail(1);      // smallest failing diagonal index (n if none)

    CUDA_CHECK(cudaMemcpy(dW.get(), A.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    const int noFailure = static_cast<int>(n);
    CUDA_CHECK(cudaMemcpy(dFail.get(), &noFailure, sizeof(int), cudaMemcpyHostToDevice));

    for (size_t j = 0; j < n; ++j) {
        const size_t rows = n - j;
        const unsigned grid = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
        choleskyColumnKernel<<<grid, kBlockThreads>>>(dW.get(), dDiag.get(), dFail.get(), n, j);
        CUDA_CHECK(cudaGetLastError());
    }

    finalizeKernel<<<static_cast<unsigned>((nn + kFinalizeBlock - 1) / kFinalizeBlock),
                     kFinalizeBlock>>>(dW.get(), dDiag.get(), n);
    CUDA_CHECK(cudaGetLastError());

    int failIdx = noFailure;
    CUDA_CHECK(cudaMemcpy(&failIdx, dFail.get(), sizeof(int), cudaMemcpyDeviceToHost));

    if (failIdx < static_cast<int>(n)) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               static_cast<size_t>(failIdx));
        return false;
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dW.get(), nn * sizeof(double), cudaMemcpyDeviceToHost));
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

    // Compute A = B * B^T on the GPU
    gramProduct(B, A, n);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU
    gramProduct(L, reconstructed, n);

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
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

// Select the GPU and create the CUDA context up front so that context setup
// is not included in the timed decomposition.
void initCuda() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
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

    initCuda();

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
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
