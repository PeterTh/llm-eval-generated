#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static inline bool cuda_check_(cudaError_t e, const char* file, int line, const char* call) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", file, line, call, cudaGetErrorString(e));
    return false;
}
#define CUDA_CHECK(call) cuda_check_((call), __FILE__, __LINE__, #call)

static __global__ void potrf_diag_kernel(double* A, int n, int k, int bs, int* err_flag, int* err_index) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int j = 0; j < bs; ++j) {
        const int row = k + j;

        double sum = 0.0;
        for (int p = 0; p < j; ++p) {
            const double l = A[row * n + (k + p)];
            sum += l * l;
        }

        double val = A[row * n + row] - sum;
        if (val <= 0.0) {
            if (err_flag) {
                *err_flag = 1;
                if (err_index) *err_index = row;
            }
            val = 1.0; // avoid NaNs; host will treat this as failure
        }

        const double diag = sqrt(val);
        A[row * n + row] = diag;

        for (int i = j + 1; i < bs; ++i) {
            const int r = k + i;
            double s = 0.0;
            for (int p = 0; p < j; ++p) {
                s += A[r * n + (k + p)] * A[row * n + (k + p)];
            }
            A[r * n + row] = (A[r * n + row] - s) / diag;
        }
    }
}

static __global__ void trsm_panel_kernel(double* A, int n, int k, int bs) {
    const int row = (k + bs) + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;

    for (int j = 0; j < bs; ++j) {
        double sum = 0.0;
        for (int p = 0; p < j; ++p) {
            sum += A[row * n + (k + p)] * A[(k + j) * n + (k + p)];
        }
        A[row * n + (k + j)] = (A[row * n + (k + j)] - sum) / A[(k + j) * n + (k + j)];
    }
}

static __global__ void syrk_update_kernel(double* A, int n, int k, int bs) {
    // Updates trailing lower triangle: A[i,j] -= sum_p A[i,k+p]*A[j,k+p]
    const int base = k + bs;
    const int j = base + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = base + (int)(blockIdx.y * blockDim.y + threadIdx.y);

    const bool valid_i = (i < n);
    const bool valid_j = (j < n);

    extern __shared__ double sh[];
    double* shI = sh;                     // [blockDim.y][bs]
    double* shJ = sh + blockDim.y * bs;   // [blockDim.x][bs]

    // Cache the panel rows for i-rows (y dimension)
    if (valid_i) {
        for (int p = threadIdx.x; p < bs; p += (int)blockDim.x) {
            shI[threadIdx.y * bs + p] = A[i * n + (k + p)];
        }
    }

    // Cache the panel rows for j-rows (x dimension)
    if (valid_j) {
        for (int p = threadIdx.y; p < bs; p += (int)blockDim.y) {
            shJ[threadIdx.x * bs + p] = A[j * n + (k + p)];
        }
    }

    __syncthreads();

    if (!valid_i || !valid_j || j > i) return;

    double sum = 0.0;
    const double* a = &shI[threadIdx.y * bs];
    const double* b = &shJ[threadIdx.x * bs];
    for (int p = 0; p < bs; ++p) {
        sum += a[p] * b[p];
    }

    A[i * n + j] -= sum;
}

static __global__ void zero_upper_kernel(double* A, int n) {
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= n || j >= n) return;
    if (j > i) A[i * n + j] = 0.0;
}

static bool choleskyDecompositionDevice(double* dA, int n, int* d_err_flag, int* d_err_index) {
    constexpr int BLOCK_K = 32;
    constexpr int TILE = 16;

    for (int k = 0; k < n; k += BLOCK_K) {
        const int bs = std::min(BLOCK_K, n - k);

        potrf_diag_kernel<<<1, 1>>>(dA, n, k, bs, d_err_flag, d_err_index);
        if (!CUDA_CHECK(cudaGetLastError())) return false;

        const int rows_below = n - (k + bs);
        if (rows_below > 0) {
            const int threads = 256;
            const int blocks = (rows_below + threads - 1) / threads;
            trsm_panel_kernel<<<blocks, threads>>>(dA, n, k, bs);
            if (!CUDA_CHECK(cudaGetLastError())) return false;

            dim3 block(TILE, TILE);
            const int dim = (rows_below + TILE - 1) / TILE;
            dim3 grid(dim, dim);
            const size_t shmem = (size_t)(2 * TILE * bs) * sizeof(double);
            syrk_update_kernel<<<grid, block, shmem>>>(dA, n, k, bs);
            if (!CUDA_CHECK(cudaGetLastError())) return false;
        }
    }

    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        zero_upper_kernel<<<grid, block>>>(dA, n);
        if (!CUDA_CHECK(cudaGetLastError())) return false;
    }

    return true;
}

// Blocked Cholesky decomposition (CUDA, right-looking)
// Decomposes SPD matrix A into L * L^T where L is lower triangular.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* dA = nullptr;
    int* d_err_flag = nullptr;
    int* d_err_index = nullptr;

    auto cleanup = [&]() {
        if (dA) CUDA_CHECK(cudaFree(dA));
        if (d_err_flag) CUDA_CHECK(cudaFree(d_err_flag));
        if (d_err_index) CUDA_CHECK(cudaFree(d_err_index));
        dA = nullptr;
        d_err_flag = nullptr;
        d_err_index = nullptr;
    };

    if (!CUDA_CHECK(cudaMalloc((void**)&dA, n * n * sizeof(double)))) {
        cleanup();
        return false;
    }
    if (!CUDA_CHECK(cudaMalloc((void**)&d_err_flag, sizeof(int)))) {
        cleanup();
        return false;
    }
    if (!CUDA_CHECK(cudaMalloc((void**)&d_err_index, sizeof(int)))) {
        cleanup();
        return false;
    }

    int h_err_flag = 0;
    if (!CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice))) {
        cleanup();
        return false;
    }
    if (!CUDA_CHECK(cudaMemcpy(d_err_flag, &h_err_flag, sizeof(int), cudaMemcpyHostToDevice))) {
        cleanup();
        return false;
    }

    if (!choleskyDecompositionDevice(dA, (int)n, d_err_flag, d_err_index)) {
        cleanup();
        return false;
    }
    if (!CUDA_CHECK(cudaDeviceSynchronize())) {
        cleanup();
        return false;
    }

    if (!CUDA_CHECK(cudaMemcpy(&h_err_flag, d_err_flag, sizeof(int), cudaMemcpyDeviceToHost))) {
        cleanup();
        return false;
    }
    if (h_err_flag != 0) {
        int idx = -1;
        (void)CUDA_CHECK(cudaMemcpy(&idx, d_err_index, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Error: Matrix is not positive definite at diagonal element %d\n", idx);
        cleanup();
        return false;
    }

    if (!CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost))) {
        cleanup();
        return false;
    }

    cleanup();
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
    
    // Perform Cholesky decomposition (GPU)
    printf("Computing Cholesky decomposition...\n");

    if (!CUDA_CHECK(cudaSetDevice(0))) {
        printf("Failed to select CUDA device\n");
        return 1;
    }
    (void)CUDA_CHECK(cudaFree(nullptr)); // initialize CUDA context

    // Allocate/copy outside the timed region
    double* dA = nullptr;
    int* d_err_flag = nullptr;
    int* d_err_index = nullptr;

    if (!CUDA_CHECK(cudaMalloc((void**)&dA, n * n * sizeof(double))) ||
        !CUDA_CHECK(cudaMalloc((void**)&d_err_flag, sizeof(int))) ||
        !CUDA_CHECK(cudaMalloc((void**)&d_err_index, sizeof(int)))) {
        if (dA) CUDA_CHECK(cudaFree(dA));
        if (d_err_flag) CUDA_CHECK(cudaFree(d_err_flag));
        if (d_err_index) CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    int h_err_flag = 0;
    if (!CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice)) ||
        !CUDA_CHECK(cudaMemcpy(d_err_flag, &h_err_flag, sizeof(int), cudaMemcpyHostToDevice))) {
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(d_err_flag));
        CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    (void)CUDA_CHECK(cudaFuncSetCacheConfig(syrk_update_kernel, cudaFuncCachePreferShared));

    auto start = std::chrono::high_resolution_clock::now();
    bool launched = choleskyDecompositionDevice(dA, (int)n, d_err_flag, d_err_index);
    bool synced = CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!launched || !synced) {
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(d_err_flag));
        CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    if (!CUDA_CHECK(cudaMemcpy(&h_err_flag, d_err_flag, sizeof(int), cudaMemcpyDeviceToHost))) {
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(d_err_flag));
        CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    if (h_err_flag != 0) {
        int idx = -1;
        (void)CUDA_CHECK(cudaMemcpy(&idx, d_err_index, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Error: Matrix is not positive definite at diagonal element %d\n", idx);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(d_err_flag));
        CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    if (!CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost))) {
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(d_err_flag));
        CUDA_CHECK(cudaFree(d_err_index));
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_err_flag));
    CUDA_CHECK(cudaFree(d_err_index));

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const long long ms = std::max(1LL, (long long)duration.count());
    double gflops = ops / (ms / 1000.0) / 1e9;
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
