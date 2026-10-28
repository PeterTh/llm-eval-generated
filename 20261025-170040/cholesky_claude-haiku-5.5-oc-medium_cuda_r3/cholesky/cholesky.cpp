#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition on the GPU (CUDA).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Algorithm: left-looking, column-oriented Cholesky. Column j depends only on
// columns 0..j-1, and every entry of column j is an independent dot product over
// k < j. One kernel is launched per column; within it, each row i >= j is handled
// by its own thread. Each dot product is accumulated in ascending k, the same
// order as the sequential reference, so results match it element for element.
//
// Device data is stored column-major so that a thread's access to L[i][k] for
// consecutive rows i is coalesced. The host converts to and from row-major.

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

constexpr unsigned int kBlockSize = 256;  // threads per block (one thread per row)
constexpr size_t kTileK = 1024;           // doubles of row j staged in shared memory

// Processes column j of L. Thread for row i (i >= j):
//   i == j : L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
//   i >  j : L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
// Also accumulates dsum[i] += L[i][j]^2 for i > j, which yields the running
// sum_{k<i} L[i][k]^2 needed by the diagonal step of column i (same order as the
// sequential code).
//
// L is column-major: element (r, c) lives at L[c * n + r].
__global__ void choleskyColumnKernel(double* __restrict__ L,
                                     const double* __restrict__ Adiag,
                                     double* __restrict__ dsum,
                                     int* __restrict__ failCol,
                                     const size_t n,
                                     const size_t j) {
    // Fast path: a previous column already found a non-positive pivot.
    if (*failCol < static_cast<int>(n)) {
        return;
    }

    extern __shared__ double rowj[];  // holds L[j][k] for the current k-tile

    const size_t i = j + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = i < n;

    // Diagonal pivot. Every thread computes the same value from data that is not
    // written during this kernel (Adiag and dsum[j]), so no synchronization is needed.
    const double val = Adiag[j] - dsum[j];
    const double Ljj = sqrt(val);

    if (i == j) {
        if (val <= 0.0) {
            atomicMin(failCol, static_cast<int>(j));
        }
        L[j * n + j] = Ljj;
    }

    // Dot products over k < j, staged through shared memory in tiles.
    // All threads in the block participate in every barrier.
    double sum = 0.0;
    for (size_t base = 0; base < j; base += kTileK) {
        const size_t len = min(kTileK, j - base);
        for (size_t t = threadIdx.x; t < len; t += blockDim.x) {
            rowj[t] = L[(base + t) * n + j];
        }
        __syncthreads();

        if (active && i > j) {
            const double* col = L + base * n + i;
            // Unrolling lets several loads be in flight; the accumulation order
            // (ascending k) is unchanged.
            #pragma unroll 8
            for (size_t t = 0; t < len; ++t) {
                sum += col[t * n] * rowj[t];
            }
        }
        __syncthreads();
    }

    if (active && i > j) {
        const double Lij = (L[j * n + i] - sum) / Ljj;
        L[j * n + i] = Lij;
        dsum[i] += Lij * Lij;
    }
}

}  // namespace

// Initializes the CUDA device and runtime so that context creation is not
// counted as part of the timed decomposition.
void initCudaDevice() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFree(nullptr));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order on input and output.
    if (n == 0) {
        return true;
    }

    // Host: row-major lower triangle -> column-major working layout.
    std::vector<double> Lcm(n * n, 0.0);
    std::vector<double> Adiag(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            Lcm[j * n + i] = A[i * n + j];
        }
        Adiag[i] = A[i * n + i];
    }

    double* dL = nullptr;
    double* dAdiag = nullptr;
    double* dsum = nullptr;
    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dAdiag, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dsum, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dL, Lcm.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAdiag, Adiag.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dsum, 0, n * sizeof(double)));
    const int noFailure = static_cast<int>(n);
    CUDA_CHECK(cudaMemcpy(dFail, &noFailure, sizeof(int), cudaMemcpyHostToDevice));

    const size_t sharedBytes = kTileK * sizeof(double);
    for (size_t j = 0; j < n; ++j) {
        const size_t rows = n - j;
        const unsigned int blocks = static_cast<unsigned int>((rows + kBlockSize - 1) / kBlockSize);
        choleskyColumnKernel<<<blocks, kBlockSize, sharedBytes>>>(dL, dAdiag, dsum, dFail, n, j);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(Lcm.data(), dL, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    int failCol = 0;
    CUDA_CHECK(cudaMemcpy(&failCol, dFail, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dAdiag));
    CUDA_CHECK(cudaFree(dsum));
    CUDA_CHECK(cudaFree(dFail));

    if (failCol < static_cast<int>(n)) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failCol);
        return false;
    }

    // Column-major working layout -> row-major, zeroing the upper triangle.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            A[i * n + j] = Lcm[j * n + i];
        }
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

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
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
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
    
    // Initialize the GPU before timing so context creation is not measured
    initCudaDevice();

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
