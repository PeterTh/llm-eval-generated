#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA Cholesky decomposition: A = L * L^T, L lower triangular.
//
// The reference is a sequential, unblocked, left-looking algorithm.  The GPU
// implementation keeps the exact same floating-point semantics: for every
// output element the partial sums are accumulated over k = 0 .. j-1 in
// increasing order, using a separate multiply + add for the first (j & ~3)
// terms and a fused multiply-add for the remaining tail.  This reproduces the
// reference result bit for bit.
//
// The columns are processed in panels of a runtime-selected width:
//   1) oldSumKernel   : S[i][j] = sum_{k < jb} L[i][k] * L[j][k]  (bulk work)
//   2) bandKernel     : factor the panel x panel diagonal block sequentially
//   3) trailingKernel : finish all rows below the panel; every block owns a
//                       set of rows and walks the panel columns in order
// Panels are processed from left to right, exactly like the reference.
// ---------------------------------------------------------------------------

namespace {

constexpr int KTILE        = 32;   // k tile of the tiled old-sum kernel
constexpr int BAND_THREADS = 256;
constexpr int OLDS_COLS    = 1;  // panel columns handled per thread
constexpr int TRAIL_ROWS   = 32;

__device__ __forceinline__ double d_mul_add(double a, double b, double c) {
    return __dadd_rn(__dmul_rn(a, b), c);
}

// 1) Old-column partial sums for a whole panel (the bulk of the work).
template <int TI>
__global__ void oldSumKernel(double* __restrict__ S,
                             const double* __restrict__ A,
                             int n, int jb, int panel, int sstride) {
    const int tid   = threadIdx.x;
    const int group = tid / TI;
    const int lane  = tid - group * TI;
    const int row0  = jb + blockIdx.x * TI;
    const int i     = row0 + lane;
    const int col0  = jb + group * OLDS_COLS;

    extern __shared__ double smem[];
    double* sA = smem;                       // [KTILE][TI+1]
    double* sB = smem + KTILE * (TI + 1);    // [KTILE][panel+1]

    double acc[OLDS_COLS];
#pragma unroll
    for (int c = 0; c < OLDS_COLS; ++c) acc[c] = 0.0;

    for (int kt = 0; kt < jb; kt += KTILE) {
        const int kkmax = min(KTILE, jb - kt);

        for (int idx = tid; idx < TI * kkmax; idx += blockDim.x) {
            const int r  = idx / kkmax;
            const int kk = idx - r * kkmax;
            const int ii = row0 + r;
            sA[kk * (TI + 1) + r] =
                (ii < n) ? A[(size_t)ii * n + (size_t)(kt + kk)] : 0.0;
        }
        for (int idx = tid; idx < panel * kkmax; idx += blockDim.x) {
            const int r  = idx / kkmax;
            const int kk = idx - r * kkmax;
            const int jj = jb + r;
            sB[kk * (panel + 1) + r] =
                (jj < n) ? A[(size_t)jj * n + (size_t)(kt + kk)] : 0.0;
        }
        __syncthreads();

        for (int kk = 0; kk < kkmax; ++kk) {
            const double av = sA[kk * (TI + 1) + lane];
            const double* b = sB + kk * (panel + 1) + group * OLDS_COLS;
#pragma unroll
            for (int c = 0; c < OLDS_COLS; ++c) {
                acc[c] = d_mul_add(av, b[c], acc[c]);
            }
        }
        __syncthreads();
    }

    if (i < n) {
        double* s = S + (size_t)(i - jb) * sstride + (col0 - jb);
#pragma unroll
        for (int c = 0; c < OLDS_COLS; ++c) s[c] = acc[c];
    }
}

// 2) Factor the diagonal band of the panel, column by column.
__global__ void bandKernel(const double* __restrict__ S,
                           double* __restrict__ A,
                           int n, int jb, int bl, int sstride,
                           int* __restrict__ errj) {
    extern __shared__ double smem[];
    double* band = smem;              // [bl][bl] factored values
    double* sorig = band + bl * bl;   // [bl][bl] original matrix block
    double* ssum = sorig + bl * bl;   // [bl][bl] old partial sums

    const int entries = bl * bl;
    for (int idx = threadIdx.x; idx < entries; idx += blockDim.x) {
        const int ii = idx / bl;
        const int jj = idx - ii * bl;
        if (jj <= ii) {
            sorig[idx] = A[(size_t)(jb + ii) * n + (jb + jj)];
            ssum[idx]  = S[(size_t)ii * sstride + jj];
        } else {
            sorig[idx] = 0.0;
            ssum[idx]  = 0.0;
        }
    }
    __syncthreads();

    for (int jj = 0; jj < bl; ++jj) {
        const int j   = jb + jj;
        const int lim = j & ~3;

        if (threadIdx.x == 0) {
            double dacc = ssum[jj * bl + jj];
            for (int k = jb; k < lim; ++k) {
                const double a = band[jj * bl + (k - jb)];
                dacc = d_mul_add(a, a, dacc);
            }
            for (int k = lim; k < j; ++k) {
                const double a = band[jj * bl + (k - jb)];
                dacc = fma(a, a, dacc);
            }
            double dval = sorig[jj * bl + jj] - dacc;
            if (dval <= 0.0) {
                atomicMin(errj, j + 1);
                dval = 0.0;
            }
            band[jj * bl + jj] = sqrt(dval);
        }
        __syncthreads();

        const double dj = band[jj * bl + jj];
        for (int ii = jj + 1 + threadIdx.x; ii < bl; ii += blockDim.x) {
            double acc = ssum[ii * bl + jj];
            for (int k = jb; k < lim; ++k) {
                acc = d_mul_add(band[ii * bl + (k - jb)],
                                band[jj * bl + (k - jb)], acc);
            }
            for (int k = lim; k < j; ++k) {
                acc = fma(band[ii * bl + (k - jb)],
                          band[jj * bl + (k - jb)], acc);
            }
            band[ii * bl + jj] = (sorig[ii * bl + jj] - acc) / dj;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < entries; idx += blockDim.x) {
        const int ii = idx / bl;
        const int jj = idx - ii * bl;
        if (jj <= ii) {
            A[(size_t)(jb + ii) * n + (jb + jj)] = band[idx];
        }
    }
}

// 3) Finalize all rows below the panel.
__global__ void trailingKernel(const double* __restrict__ S,
                               double* __restrict__ A,
                               int n, int jb, int bl, int panel,
                               int sstride) {
    extern __shared__ double smem[];
    double* srow  = smem;                        // [TRAIL_ROWS][panel]
    double* sband = srow + TRAIL_ROWS * panel;   // [panel][panel]

    const int lane = threadIdx.x;
    const int base = jb + bl + blockIdx.x * TRAIL_ROWS;

    for (int idx = lane; idx < bl * bl; idx += blockDim.x) {
        const int ii = idx / bl;
        const int jj = idx - ii * bl;
        sband[idx] = (jj <= ii)
                         ? A[(size_t)(jb + ii) * n + (jb + jj)]
                         : 0.0;
    }
    for (int idx = lane; idx < TRAIL_ROWS * bl; idx += blockDim.x) {
        const int r  = idx / bl;
        const int jj = idx - r * bl;
        const int i  = base + r;
        srow[idx] = (i < n) ? S[(size_t)(i - jb) * sstride + jj] : 0.0;
    }
    __syncthreads();

    const int i = base + lane;
    if (i < n) {
        double* r = srow + (size_t)lane * bl;
        for (int jj = 0; jj < bl; ++jj) {
            const int j   = jb + jj;
            const int lim = j & ~3;
            double acc = r[jj];
            for (int k = jb; k < lim; ++k) {
                acc = d_mul_add(r[k - jb], sband[jj * bl + (k - jb)], acc);
            }
            for (int k = lim; k < j; ++k) {
                acc = fma(r[k - jb], sband[jj * bl + (k - jb)], acc);
            }
            r[jj] = (A[(size_t)i * n + j] - acc) / sband[jj * bl + jj];
        }
        for (int jj = 0; jj < bl; ++jj) {
            A[(size_t)i * n + (jb + jj)] = r[jj];
        }
    }
}

__global__ void zeroUpperKernel(double* __restrict__ A, int n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * n) return;
    const int i = (int)(idx / n);
    const int j = (int)(idx % n);
    if (j > i) A[idx] = 0.0;
}

template <int TI>
void launchOldSumTI(double* S, const double* A, int n, int jb, int rows,
                    int panel, int sstride) {
    const int blocks = (rows + TI - 1) / TI;
    const size_t sh = (size_t)KTILE * (TI + 1) +
                      (size_t)KTILE * (panel + 1);
    oldSumKernel<TI><<<blocks, (panel / OLDS_COLS) * TI, sh * sizeof(double)>>>(
        S, A, n, jb, panel, sstride);
}

void launchOldSum(double* S, const double* A, int n, int jb, int rows,
                  int sms, int panel, int sstride) {
    if (rows >= 32 * sms) {
        launchOldSumTI<32>(S, A, n, jb, rows, panel, sstride);
    } else if (rows >= 16 * sms) {
        launchOldSumTI<16>(S, A, n, jb, rows, panel, sstride);
    } else if (rows >= 8 * sms) {
        launchOldSumTI<8>(S, A, n, jb, rows, panel, sstride);
    } else {
        launchOldSumTI<4>(S, A, n, jb, rows, panel, sstride);
    }
}

}  // namespace


bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    if (n > (size_t)2147483647) {
        printf("Matrix size too large\n");
        return false;
    }

    const int ni = (int)n;

    // Panel width: smaller panels keep the serial band factorization short
    // for small matrices, larger panels give better data reuse for big ones.
    int panel = 32;
    if (ni <= 640) {
        panel = 16;
    } else if (ni <= 1536) {
        panel = 20;
    }
    const int sstride = panel;

    double* dA = nullptr;
    double* dS = nullptr;
    int* d_err = nullptr;
    if (cudaMalloc(&dA, (size_t)n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&dS, (size_t)n * sstride * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_err, sizeof(int)) != cudaSuccess) {
        printf("CUDA allocation failed\n");
        return false;
    }

    cudaMemset(d_err, 0, sizeof(int));
    if (cudaMemcpyAsync(dA, A.data(), (size_t)n * n * sizeof(double),
                        cudaMemcpyHostToDevice, 0) != cudaSuccess) {
        printf("CUDA copy to device failed\n");
        cudaFree(dA);
        cudaFree(dS);
        cudaFree(d_err);
        return false;
    }

    int sms = 1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    }

    for (int jb = 0; jb < ni; jb += panel) {
        const int bl = min(panel, ni - jb);
        const int rows = ni - jb;

        launchOldSum(dS, dA, ni, jb, rows, sms, panel, sstride);

        const size_t bandBytes = (size_t)3 * bl * bl * sizeof(double);
        bandKernel<<<1, BAND_THREADS, bandBytes>>>(dS, dA, ni, jb, bl, sstride,
                                                   d_err);

        const int trailRows = ni - jb - bl;
        if (trailRows > 0) {
            const int blocks = (trailRows + TRAIL_ROWS - 1) / TRAIL_ROWS;
            const size_t srBytes = (size_t)(TRAIL_ROWS * panel +
                                            panel * panel) * sizeof(double);
            trailingKernel<<<blocks, TRAIL_ROWS, srBytes>>>(dS, dA, ni, jb, bl,
                                                            panel, sstride);
        }
    }

    {
        const size_t total = (size_t)n * n;
        const int threads = 256;
        const long long blocks =
            (long long)((total + (size_t)threads - 1) / (size_t)threads);
        zeroUpperKernel<<<(unsigned int)blocks, threads>>>(dA, ni);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(dA);
        cudaFree(dS);
        cudaFree(d_err);
        return false;
    }

    int errj = 0;
    cudaMemcpy(&errj, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(A.data(), dA, (size_t)n * n * sizeof(double),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    cudaFree(dA);
    cudaFree(dS);
    cudaFree(d_err);

    if (errj > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               errj - 1);
        return false;
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
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");

    // Initialize the CUDA context/runtime before the timed region so that
    // one-time device setup is not attributed to the decomposition.  Pinning
    // the host matrix makes the device transfers use the full PCIe bandwidth.
    cudaFree(nullptr);
    bool hostPinned = false;
    if (n > 0) {
        hostPinned = cudaHostRegister(A.data(),
                                      (size_t)n * n * sizeof(double),
                                      cudaHostRegisterDefault) == cudaSuccess;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    if (hostPinned) {
        cudaHostUnregister(A.data());
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", (long)duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double seconds = (double)duration_us.count() / 1e6;
    double gflops = (seconds > 0.0) ? ops / seconds / 1e9 : 0.0;
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
