#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace cg = cooperative_groups;

namespace {

constexpr int kBlock = 32;
constexpr int kTile = 64;
constexpr int kTileK = 32;
constexpr int kMicro = 4;
constexpr int kThreads = 256;
constexpr int kPanelStride = kBlock + 1;
constexpr int kTileStride = kTileK + 1;
constexpr int kTrsmRows = 16;
constexpr int kRowStride = kBlock + 1;
constexpr int kPanelSmem = kBlock * kPanelStride;
constexpr int kGemmSmem = 2 * kTile * kTileStride;
constexpr int kTrsmSmem = kPanelSmem + kTrsmRows * kRowStride;
constexpr int kSmemCore = (kPanelSmem > kGemmSmem) ? kPanelSmem : kGemmSmem;
constexpr int kSmemDoubles = ((kSmemCore > kTrsmSmem) ? kSmemCore : kTrsmSmem) + kBlock;
constexpr int kFailInit = -2139062144;

__device__ __forceinline__ void gemmTile(double* A, int n, int k, int b,
                                         int row0, int col0,
                                         double* As, double* Bs) {
    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    double acc[kMicro][kMicro] = {};

    for (int kk = 0; kk < b; kk += kTileK) {
        const int klocal = min(kTileK, b - kk);

        for (int idx = tid; idx < kTile * klocal; idx += kThreads) {
            const int r = idx / klocal;
            const int c = idx - r * klocal;
            const int gr = row0 + r;
            const int gc = col0 + r;
            As[r * kTileStride + c] = (gr < n) ? A[(size_t)gr * n + k + kk + c] : 0.0;
            Bs[r * kTileStride + c] = (gc < n) ? A[(size_t)gc * n + k + kk + c] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < klocal; ++p) {
            double av[kMicro];
            double bv[kMicro];
#pragma unroll
            for (int i = 0; i < kMicro; ++i) {
                av[i] = As[(ty * kMicro + i) * kTileStride + p];
                bv[i] = Bs[(tx * kMicro + i) * kTileStride + p];
            }
#pragma unroll
            for (int i = 0; i < kMicro; ++i) {
#pragma unroll
                for (int j = 0; j < kMicro; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
        const int gr = row0 + ty * kMicro + i;
        if (gr < n) {
#pragma unroll
            for (int j = 0; j < kMicro; ++j) {
                const int gc = col0 + tx * kMicro + j;
                if (gc < n) {
                    A[(size_t)gr * n + gc] -= acc[i][j];
                }
            }
        }
    }
}

__global__ void choleskyKernel(double* A, int n, int* failFlag) {
    extern __shared__ double smem[];
    double* As = smem;
    double* Bs = smem + kTile * kTileStride;
    double* Lp = smem;
    double* rowbuf = smem + kPanelSmem;
    double* dinv = smem + kPanelSmem + kTrsmRows * kRowStride;

    const int tid = threadIdx.x;
    cg::grid_group grid = cg::this_grid();

    for (int k = 0; k < n; k += kBlock) {
        const int b = min(kBlock, n - k);
        const int m = n - k - b;

        if (blockIdx.x == 0) {
            for (int idx = tid; idx < b * b; idx += blockDim.x) {
                const int r = idx / b;
                const int c = idx - r * b;
                Lp[r * kPanelStride + c] = A[(size_t)(k + r) * n + k + c];
            }
            __syncthreads();

            double sumSq = 0.0;
            for (int j = 0; j < b; ++j) {
                if (tid == j) {
                    const double val = Lp[j * kPanelStride + j] - sumSq;
                    if (val <= 0.0) {
                        atomicMax(failFlag, -(k + j + 1));
                        Lp[j * kPanelStride + j] = 0.0;
                        dinv[j] = 0.0;
                    } else {
                        const double d = sqrt(val);
                        Lp[j * kPanelStride + j] = d;
                        dinv[j] = 1.0 / d;
                    }
                }
                __syncthreads();

                if (tid > j && tid < b) {
                    const int i = tid;
                    double* ai = Lp + i * kPanelStride;
                    const double* aj = Lp + j * kPanelStride;
                    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    int p = 0;
                    for (; p + 3 < j; p += 4) {
                        s0 += ai[p] * aj[p];
                        s1 += ai[p + 1] * aj[p + 1];
                        s2 += ai[p + 2] * aj[p + 2];
                        s3 += ai[p + 3] * aj[p + 3];
                    }
                    for (; p < j; ++p) {
                        s0 += ai[p] * aj[p];
                    }
                    const double sum = (s0 + s1) + (s2 + s3);
                    const double v = (ai[j] - sum) * dinv[j];
                    ai[j] = v;
                    sumSq += v * v;
                }
                __syncthreads();
            }

            for (int idx = tid; idx < b * b; idx += blockDim.x) {
                const int r = idx / b;
                const int c = idx - r * b;
                if (c <= r) {
                    A[(size_t)(k + r) * n + k + c] = Lp[r * kPanelStride + c];
                }
            }
        }
        grid.sync();

        if (m > 0) {
            for (int idx = tid; idx < b * b; idx += blockDim.x) {
                const int r = idx / b;
                const int c = idx - r * b;
                Lp[r * kPanelStride + c] = A[(size_t)(k + r) * n + k + c];
            }
            __syncthreads();
            if (tid < b) {
                dinv[tid] = 1.0 / Lp[tid * kPanelStride + tid];
            }
            __syncthreads();

            const int chunks = (m + kTrsmRows - 1) / kTrsmRows;
            for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
                const int row0 = chunk * kTrsmRows;
                const int nr = min(kTrsmRows, m - row0);

                for (int idx = tid; idx < nr * b; idx += blockDim.x) {
                    const int rr = idx / b;
                    const int cc = idx - rr * b;
                    rowbuf[rr * kRowStride + cc] =
                        A[(size_t)(k + b + row0 + rr) * n + k + cc];
                }
                __syncthreads();

                if (tid < nr) {
                    double* a = rowbuf + tid * kRowStride;
                    for (int j = 0; j < b; ++j) {
                        const double* lj = Lp + j * kPanelStride;
                        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                        int p = 0;
                        for (; p + 3 < j; p += 4) {
                            s0 += a[p] * lj[p];
                            s1 += a[p + 1] * lj[p + 1];
                            s2 += a[p + 2] * lj[p + 2];
                            s3 += a[p + 3] * lj[p + 3];
                        }
                        for (; p < j; ++p) {
                            s0 += a[p] * lj[p];
                        }
                        const double sum = (s0 + s1) + (s2 + s3);
                        a[j] = (a[j] - sum) * dinv[j];
                    }
                }
                __syncthreads();

                for (int idx = tid; idx < nr * b; idx += blockDim.x) {
                    const int rr = idx / b;
                    const int cc = idx - rr * b;
                    A[(size_t)(k + b + row0 + rr) * n + k + cc] =
                        rowbuf[rr * kRowStride + cc];
                }
                __syncthreads();
            }
        }
        grid.sync();

        if (m > 0) {
            const int ntiles = (m + kTile - 1) / kTile;
            const long long total = (long long)ntiles * (ntiles + 1) / 2;
            for (long long t = blockIdx.x; t < total; t += gridDim.x) {
                int tr = (int)((sqrt(8.0 * (double)t + 1.0) - 1.0) * 0.5);
                while ((long long)(tr + 1) * (tr + 2) / 2 <= t) {
                    ++tr;
                }
                while ((long long)tr * (tr + 1) / 2 > t) {
                    --tr;
                }
                const int tc = (int)(t - (long long)tr * (tr + 1) / 2);
                gemmTile(A, n, k, b, k + b + tr * kTile, k + b + tc * kTile, As, Bs);
            }
        }
        grid.sync();
    }

    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        for (int j = i + 1 + tid; j < n; j += blockDim.x) {
            A[(size_t)i * n + j] = 0.0;
        }
    }
}

}  // namespace

// Blocked Cholesky decomposition on the GPU:
// A = L * L^T, with L written to the lower triangle of A (row-major).
bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           double* d_A, int* d_failFlag) {
    if (n == 0) {
        return true;
    }

    int ni = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_failFlag, 0x80, sizeof(int));

    int device = 0;
    cudaGetDevice(&device);
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    int maxBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, choleskyKernel, kThreads, kSmemDoubles * sizeof(double));

    int maxGrid = maxBlocksPerSm * smCount;
    if (maxGrid < 1) {
        maxGrid = 1;
    }

    const int firstM = ni - std::min(kBlock, ni);
    int grid = maxGrid;
    if (firstM > 0) {
        const int ntiles = (firstM + kTile - 1) / kTile;
        const long long totalTiles = (long long)ntiles * (ntiles + 1) / 2;
        if (totalTiles < grid) {
            grid = static_cast<int>(totalTiles);
        }
    } else {
        grid = 1;
    }
    if (grid < 1) {
        grid = 1;
    }

    void* args[] = {&d_A, &ni, &d_failFlag};
    const cudaError_t launchStatus =
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(choleskyKernel), dim3(grid),
                                    dim3(kThreads), args, kSmemDoubles * sizeof(double), 0);
    if (launchStatus != cudaSuccess) {
        printf("Error: CUDA kernel launch failed: %s\n", cudaGetErrorString(launchStatus));
        return false;
    }

    int hFail = 0;
    cudaMemcpy(&hFail, d_failFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if (hFail != kFailInit) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", -hFail - 1);
        return false;
    }

    cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost);

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

    // Allocate device buffers and initialize the CUDA context before timing
    cudaFree(nullptr);
    cudaDeviceSynchronize();
    const size_t bytes = n > 0 ? n * n * sizeof(double) : sizeof(double);
    double* d_A = nullptr;
    int* d_failFlag = nullptr;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_failFlag, sizeof(int)) != cudaSuccess) {
        printf("Error: CUDA allocation failed\n");
        return 1;
    }

    {
        std::vector<double> warmup(1, 1.0);
        choleskyDecomposition(warmup, 1, d_A, d_failFlag);
    }
    cudaDeviceSynchronize();

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, d_A, d_failFlag);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();

    cudaFree(d_failFlag);
    cudaFree(d_A);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = (elapsedSeconds > 0.0) ? ops / elapsedSeconds / 1e9 : 0.0;
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
