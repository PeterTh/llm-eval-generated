#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }
}

#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

constexpr int kBlockSize = 64;   // Panel/block width
constexpr int kTile = 16;        // Tile for trailing update
constexpr int kTrsmThreads = 128;

template <int NB>
__global__ void chol_panel_kernel(double* __restrict__ A, int n, int k, int bk, int* __restrict__ info) {
    __shared__ double sA[NB * NB];

    // Load diagonal block into shared memory
    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        const int r = idx / bk;
        const int c = idx - r * bk;
        sA[r * NB + c] = A[(k + r) * n + (k + c)];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int j = 0; j < bk; ++j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                const double v = sA[j * NB + p];
                sum += v * v;
            }
            const double val = sA[j * NB + j] - sum;
            if (val <= 0.0) {
                *info = 1;
                return;
            }
            sA[j * NB + j] = sqrt(val);

            for (int i = j + 1; i < bk; ++i) {
                double s = 0.0;
                for (int p = 0; p < j; ++p) {
                    s += sA[i * NB + p] * sA[j * NB + p];
                }
                sA[i * NB + j] = (sA[i * NB + j] - s) / sA[j * NB + j];
            }

            // Zero upper triangle inside the panel for consistency
            for (int c = j + 1; c < bk; ++c) {
                sA[j * NB + c] = 0.0;
            }
        }
    }
    __syncthreads();

    // Store back
    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        const int r = idx / bk;
        const int c = idx - r * bk;
        A[(k + r) * n + (k + c)] = sA[r * NB + c];
    }
}

template <int NB>
__global__ void trsm_kernel(double* __restrict__ A, int n, int k, int bk) {
    __shared__ double sL[NB * NB];

    for (int idx = threadIdx.x; idx < bk * bk; idx += blockDim.x) {
        const int r = idx / bk;
        const int c = idx - r * bk;
        sL[r * NB + c] = A[(k + r) * n + (k + c)];
    }
    __syncthreads();

    const int row = (k + bk) + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;

    // Solve A[row, k:k+bk] = A[row, k:k+bk] * inv(L^T)
    for (int j = 0; j < bk; ++j) {
        double sum = 0.0;
        #pragma unroll
        for (int p = 0; p < NB; ++p) {
            if (p < j) {
                sum += A[row * n + (k + p)] * sL[j * NB + p];
            }
        }
        const double diag = sL[j * NB + j];
        A[row * n + (k + j)] = (A[row * n + (k + j)] - sum) / diag;
    }
}

template <int NB, int TILE>
__global__ void syrk_update_kernel(double* __restrict__ A, int n, int k, int bk) {
    const int base = k + bk;
    const int tile_i = (int)blockIdx.y;
    const int tile_j = (int)blockIdx.x;
    if (tile_i < tile_j) return; // lower triangle only

    const int i = base + tile_i * TILE + (int)threadIdx.y;
    const int j = base + tile_j * TILE + (int)threadIdx.x;
    const bool doUpdate = (i < n) && (j < n) && (i >= j);

    __shared__ double sLi[TILE * NB];
    __shared__ double sLj[TILE * NB];

    const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
    const int nthreads = (int)(blockDim.x * blockDim.y);

    // Load L21 rows for this i-tile and j-tile
    for (int idx = tid; idx < TILE * bk; idx += nthreads) {
        const int r = idx / bk;
        const int p = idx - r * bk;
        const int gi = base + tile_i * TILE + r;
        const int gj = base + tile_j * TILE + r;
        sLi[r * NB + p] = (gi < n) ? A[gi * n + (k + p)] : 0.0;
        sLj[r * NB + p] = (gj < n) ? A[gj * n + (k + p)] : 0.0;
    }
    __syncthreads();

    double acc = 0.0;
    #pragma unroll
    for (int p = 0; p < NB; ++p) {
        if (p < bk) {
            acc += sLi[(int)threadIdx.y * NB + p] * sLj[(int)threadIdx.x * NB + p];
        }
    }

    if (doUpdate) {
        A[i * n + j] -= acc;
    }
}

__global__ void zero_upper_kernel(double* __restrict__ A, int n) {
    const int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= n || j >= n) return;
    if (j > i) A[i * n + j] = 0.0;
}

} // namespace

// Blocked CUDA Cholesky decomposition.
// Decomposes SPD matrix A into L * L^T in-place (row-major), storing L in the lower triangle.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, double* kernelMs = nullptr) {
    if (n == 0) {
        if (kernelMs) *kernelMs = 0.0;
        return true;
    }

    double* dA = nullptr;
    int* dInfo = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dInfo, sizeof(int)));
    CUDA_CHECK(cudaMemset(dInfo, 0, sizeof(int)));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));

    const int nn = (int)n;
    for (int k = 0; k < nn; k += kBlockSize) {
        const int bk = ((k + kBlockSize) <= nn) ? kBlockSize : (nn - k);

        chol_panel_kernel<kBlockSize><<<1, 256>>>(dA, nn, k, bk, dInfo);

        const int base = k + bk;
        const int m = nn - base;
        if (m > 0) {
            const int trsmBlocks = (m + kTrsmThreads - 1) / kTrsmThreads;
            trsm_kernel<kBlockSize><<<trsmBlocks, kTrsmThreads>>>(dA, nn, k, bk);

            const int tiles = (m + kTile - 1) / kTile;
            dim3 grid((unsigned)tiles, (unsigned)tiles);
            dim3 block((unsigned)kTile, (unsigned)kTile);
            syrk_update_kernel<kBlockSize, kTile><<<grid, block>>>(dA, nn, k, bk);
        }
    }

    {
        const int threads = 16;
        dim3 block((unsigned)threads, (unsigned)threads);
        dim3 grid((unsigned)((nn + threads - 1) / threads), (unsigned)((nn + threads - 1) / threads));
        zero_upper_kernel<<<grid, block>>>(dA, nn);
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    CUDA_CHECK(cudaGetLastError());

    int hInfo = 0;
    CUDA_CHECK(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dInfo));
    CUDA_CHECK(cudaFree(dA));

    if (kernelMs) *kernelMs = (double)ms;
    return hInfo == 0;
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
            n = (size_t)atoi(argv[++i]);
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
    double kernelMs = 0.0;
    bool success = choleskyDecomposition(A, n, &kernelMs);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %.3f ms\n", kernelMs);

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * (double)n * (double)n / 3.0;
    double gflops = ops / (kernelMs / 1000.0) / 1e9;
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
