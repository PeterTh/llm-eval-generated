#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cholesky decomposition on the GPU (CUDA): blocked right-looking algorithm.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Per panel: (1) factor the diagonal block, (2) triangular solve of the panel
// below it, (3) symmetric rank-NB update of the trailing matrix (FP64 tiled GEMM).

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int NB = 32;   // panel width
constexpr int TILE = 64; // GEMM tile (rows and cols)
constexpr int BK = 16;   // GEMM k step

// Factor the diagonal block (bs x bs, bs <= NB) in shared memory.
__global__ void potrfDiagKernel(double* A, const size_t n, const int k0, const int bs, int* status) {
    __shared__ double s[NB][NB + 1];
    const int t = threadIdx.x;
    double* base = A + (size_t)k0 * n + k0;
    for (int e = t; e < bs * bs; e += blockDim.x) {
        const int i = e / bs, j = e % bs;
        s[i][j] = base[(size_t)i * n + j];
    }
    __syncthreads();
    for (int j = 0; j < bs; ++j) {
        if (t == 0) {
            const double val = s[j][j];
            if (val <= 0.0) {
                atomicMin(status, k0 + j);
                s[j][j] = 1.0;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        const double d = s[j][j];
        for (int i = j + 1 + t; i < bs; i += blockDim.x) s[i][j] /= d;
        __syncthreads();
        for (int e = t; e < bs * bs; e += blockDim.x) {
            const int i = e / bs, c = e % bs;
            if (c > j && c <= i) s[i][c] -= s[i][j] * s[c][j];
        }
        __syncthreads();
    }
    for (int e = t; e < bs * bs; e += blockDim.x) {
        const int i = e / bs, j = e % bs;
        if (j <= i) base[(size_t)i * n + j] = s[i][j];
    }
}

// Panel solve: L21 = A21 * L11^{-T}; one thread per row.
__global__ void trsmKernel(double* A, const size_t n, const int k0, const int bs) {
    __shared__ double L[NB][NB];
    for (int e = threadIdx.x; e < bs * bs; e += blockDim.x) {
        const int i = e / bs, j = e % bs;
        L[i][j] = A[(size_t)(k0 + i) * n + k0 + j];
    }
    __syncthreads();
    const size_t row = (size_t)k0 + bs + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double* r = A + row * n + k0;
    double x[NB];
    #pragma unroll
    for (int j = 0; j < NB; ++j) x[j] = (j < bs) ? r[j] : 0.0;
    #pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < bs) {
            double sum = x[j];
            #pragma unroll
            for (int k = 0; k < j; ++k) sum -= x[k] * L[j][k];
            x[j] = sum / L[j][j];
        }
    }
    #pragma unroll
    for (int j = 0; j < NB; ++j) if (j < bs) r[j] = x[j];
}

enum GemmMode { MODE_SYRK = 0, MODE_GEN = 1, MODE_CHECK = 2 };

// C(i,j) op= sum_k P[i][k] * P[j][k]  (row-major, both operands row-major with k contiguous)
//  MODE_SYRK : C -= dot, lower tiles only
//  MODE_GEN  : C  = dot (+ diagonal shift n)
//  MODE_CHECK: compare dot against C, reduce max abs / rel error
template <int MODE>
__global__ void __launch_bounds__(256)
gemmNT(const double* __restrict__ P, const size_t ldp, double* C, const size_t ldc,
       const int m, const int K, double shift, unsigned long long* errs) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (MODE == MODE_SYRK && bj > bi) return;
    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE + 1];
    const int t = threadIdx.x;
    const int tx = t & 15, ty = t >> 4;
    const int lk = t & 15, lr = t >> 4;
    const int row0 = bi * TILE, col0 = bj * TILE;
    double acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int r = lr + 16 * p;
            const int k = k0 + lk;
            As[lk][r] = (row0 + r < m && k < K) ? P[(size_t)(row0 + r) * ldp + k] : 0.0;
            Bs[lk][r] = (col0 + r < m && k < K) ? P[(size_t)(col0 + r) * ldp + k] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int x = 0; x < 4; ++x) { a[x] = As[k][ty + 16 * x]; b[x] = Bs[k][tx + 16 * x]; }
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                #pragma unroll
                for (int y = 0; y < 4; ++y) acc[x][y] = fma(a[x], b[y], acc[x][y]);
        }
        __syncthreads();
    }
    double maxAbs = 0.0, maxRel = 0.0;
    #pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int i = row0 + ty + 16 * x;
        if (i >= m) continue;
        #pragma unroll
        for (int y = 0; y < 4; ++y) {
            const int j = col0 + tx + 16 * y;
            if (j >= m) continue;
            double* c = C + (size_t)i * ldc + j;
            if (MODE == MODE_SYRK) {
                if (j <= i) *c -= acc[x][y];
            } else if (MODE == MODE_GEN) {
                *c = acc[x][y] + (i == j ? shift : 0.0);
            } else {
                const double err = fabs(acc[x][y] - *c);
                maxAbs = fmax(maxAbs, err);
                maxRel = fmax(maxRel, err / (fabs(*c) + 1e-10));
            }
        }
    }
    if (MODE == MODE_CHECK) {
        // non-negative doubles order the same as their bit patterns
        atomicMax(&errs[0], (unsigned long long)__double_as_longlong(maxAbs));
        atomicMax(&errs[1], (unsigned long long)__double_as_longlong(maxRel));
    }
}

__global__ void zeroUpperKernel(double* A, const size_t n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    const size_t i = idx / n, j = idx % n;
    if (j > i) A[idx] = 0.0;
}

static inline dim3 tileGrid(int m) {
    const int t = (m + TILE - 1) / TILE;
    return dim3(t, t);
}

// d_A: device matrix (row-major). Returns false if not positive definite.
bool choleskyDecomposition(double* d_A, const size_t n) {
    int* d_status;
    CUDA_CHECK(cudaMalloc(&d_status, sizeof(int)));
    const int INT_OK = 0x7fffffff;
    CUDA_CHECK(cudaMemcpy(d_status, &INT_OK, sizeof(int), cudaMemcpyHostToDevice));

    for (size_t k0 = 0; k0 < n; k0 += NB) {
        const int bs = (int)std::min((size_t)NB, n - k0);
        const size_t k1 = k0 + bs;
        potrfDiagKernel<<<1, 256>>>(d_A, n, (int)k0, bs, d_status);
        if (k1 < n) {
            const int m = (int)(n - k1);
            trsmKernel<<<(m + 127) / 128, 128>>>(d_A, n, (int)k0, bs);
            const double* P = d_A + k1 * n + k0;
            gemmNT<MODE_SYRK><<<tileGrid(m), 256>>>(P, n, d_A + k1 * n + k1, n, m, bs, 0.0, nullptr);
        }
    }
    zeroUpperKernel<<<(unsigned)((n * n + 255) / 256), 256>>>(d_A, n);
    CUDA_CHECK(cudaGetLastError());

    int status;
    CUDA_CHECK(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_status));
    if (status != INT_OK) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", status);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly on the device
void generatePositiveDefiniteMatrix(double* d_A, const size_t n) {
    // Method: Create A = B * B^T where B is random, then add n to the diagonal
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    double* d_B;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    gemmNT<MODE_GEN><<<tileGrid((int)n), 256>>>(d_B, n, d_A, n, (int)n, (int)n, (double)n, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_B));
}

// Validate by computing L * L^T on the device and comparing with the original
bool validateCholesky(const double* d_L, const double* d_Aorig, const size_t n) {
    unsigned long long* d_errs;
    CUDA_CHECK(cudaMalloc(&d_errs, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_errs, 0, 2 * sizeof(unsigned long long)));
    gemmNT<MODE_CHECK><<<tileGrid((int)n), 256>>>(d_L, n, const_cast<double*>(d_Aorig), n,
                                                  (int)n, (int)n, 0.0, d_errs);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h[2];
    CUDA_CHECK(cudaMemcpy(h, d_errs, sizeof(h), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_errs));
    double maxError, relError;
    memcpy(&maxError, &h[0], sizeof(double));
    memcpy(&relError, &h[1], sizeof(double));

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (!(relError <= 1e-6)) {
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
    
    // Allocate matrix (device)
    const size_t bytes = n * n * sizeof(double);
    double *d_A = nullptr, *d_Aorig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d_A, n);

    if (validate) {
        CUDA_CHECK(cudaMalloc(&d_Aorig, bytes));
        CUDA_CHECK(cudaMemcpy(d_Aorig, d_A, bytes, cudaMemcpyDeviceToDevice)); // Save original for validation
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n);
    CUDA_CHECK(cudaDeviceSynchronize());

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
        std::vector<double> A(n * n);
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(d_A, d_Aorig, n);
        
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
