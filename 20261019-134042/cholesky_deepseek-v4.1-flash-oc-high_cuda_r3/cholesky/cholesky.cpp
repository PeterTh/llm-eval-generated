#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// GPU (CUDA) Cholesky decomposition.
// Decomposes positive definite matrix A (row-major, host vector) into
// L * L^T where L is lower triangular. The factorization itself runs on the
// GPU using a blocked right-looking algorithm:
//   1. unblocked factorization of the current diagonal block,
//   2. triangular solve for the panel below the diagonal block,
//   3. rank-k update of the trailing submatrix.
// The result is written back into A with the strictly upper part set to zero,
// matching the semantics of the original sequential implementation.

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                   __FILE__, __LINE__);                                         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace {

constexpr int kBlockSize = 32;   // blocked Cholesky panel width
constexpr int kTile = 32;        // trailing update tile size
constexpr int kTileThreads = 16; // trailing update threads per tile dimension

// ---------------------------------------------------------------------------
// Matrix generation
// ---------------------------------------------------------------------------

__global__ void gramKernel(const double* __restrict__ B,
                           double* __restrict__ A, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) {
        return;
    }
    const double* Bi = B + static_cast<size_t>(i) * n;
    const double* Bj = B + static_cast<size_t>(j) * n;
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += Bi[k] * Bj[k];
    }
    if (i == j) {
        sum += static_cast<double>(n);
    }
    A[static_cast<size_t>(i) * n + j] = sum;
}

// Generate a symmetric positive definite matrix on the device:
// A = B * B^T + n * I, with B generated exactly like the reference code.
void generatePositiveDefiniteMatrixDevice(double* dA, const size_t n) {
    if (n == 0) {
        return;
    }
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    const int nn = static_cast<int>(n);
    dim3 block(kTileThreads, kTileThreads);
    dim3 grid((nn + block.x - 1) / block.x, (nn + block.y - 1) / block.y);
    gramKernel<<<grid, block>>>(dB, dA, nn);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Blocked Cholesky kernels
// ---------------------------------------------------------------------------

// Factor the nb x nb diagonal block starting at (p, p) using the classic
// unblocked column algorithm, kept inside one thread block.
__global__ void diagonalBlockKernel(double* __restrict__ A, int n, int p,
                                    int nb, int* __restrict__ failIdx) {
    __shared__ double sdata[kBlockSize];
    __shared__ int sfail;

    const int tid = threadIdx.x;
    if (tid == 0) {
        sfail = 0;
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        const int gj = p + j;

        double local = 0.0;
        for (int k = tid; k < j; k += blockDim.x) {
            const double v = A[static_cast<size_t>(gj) * n + p + k];
            local += v * v;
        }
        sdata[tid] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const double val =
                A[static_cast<size_t>(gj) * n + gj] - sdata[0];
            if (val <= 0.0) {
                atomicMin(failIdx, gj);
                sfail = 1;
            } else {
                A[static_cast<size_t>(gj) * n + gj] = sqrt(val);
            }
        }
        __syncthreads();
        if (sfail) {
            return;
        }

        const double diag = A[static_cast<size_t>(gj) * n + gj];
        for (int i = j + 1 + tid; i < nb; i += blockDim.x) {
            const int gi = p + i;
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += A[static_cast<size_t>(gi) * n + p + k] *
                       A[static_cast<size_t>(gj) * n + p + k];
            }
            A[static_cast<size_t>(gi) * n + gj] =
                (A[static_cast<size_t>(gi) * n + gj] - sum) / diag;
        }
        __syncthreads();
    }
}

// Solve L21 = A21 * L11^{-T} for the panel below the diagonal block.
// One warp handles one row; within a row the columns are solved in order.
__global__ void panelSolveKernel(double* __restrict__ A, int n, int p,
                                 int nb, int rowStart) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;
    const int r = rowStart + blockIdx.x * warpsPerBlock + warp;
    if (r >= n) {
        return;
    }

    const size_t rowOff = static_cast<size_t>(r) * n;
    for (int j = 0; j < nb; ++j) {
        const int gj = p + j;
        const size_t jOff = static_cast<size_t>(gj) * n;
        double partial = 0.0;
        for (int k = lane; k < j; k += 32) {
            partial += A[rowOff + p + k] * A[jOff + p + k];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        }
        if (lane == 0) {
            A[rowOff + gj] = (A[rowOff + gj] - partial) / A[jOff + gj];
        }
        __syncwarp();
    }
}

// Trailing submatrix update: A22 -= L21 * L21^T.
__global__ void trailingUpdateKernel(double* __restrict__ A, int n, int p,
                                     int nb, int start) {
    __shared__ double As[kTile][kTile + 1];
    __shared__ double Bs[kTile][kTile + 1];

    const int row0 = start + blockIdx.y * kTile;
    const int col0 = start + blockIdx.x * kTile;
    if (row0 >= n || col0 >= n) {
        return;
    }
    // Tiles strictly above the diagonal contain no lower-triangular entries.
    if (col0 > row0 + kTile - 1) {
        return;
    }

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r0 = row0 + ty * 2;
    const int c0 = col0 + tx * 2;

    double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;

    for (int kk = 0; kk < nb; kk += kTile) {
        const int kend = min(kTile, nb - kk);
        for (int idx = ty * kTileThreads + tx; idx < kTile * kTile;
             idx += kTileThreads * kTileThreads) {
            const int ii = idx >> 5;
            const int jj = idx & (kTile - 1);
            const int gr = row0 + ii;
            const int gk = p + kk + jj;
            As[ii][jj] = (gr < n && jj < kend)
                             ? A[static_cast<size_t>(gr) * n + gk]
                             : 0.0;
            const int gc = col0 + ii;
            Bs[ii][jj] = (gc < n && jj < kend)
                             ? A[static_cast<size_t>(gc) * n + gk]
                             : 0.0;
        }
        __syncthreads();

        for (int jj = 0; jj < kend; ++jj) {
            const double a0 = As[ty * 2][jj];
            const double a1 = As[ty * 2 + 1][jj];
            const double b0 = Bs[tx * 2][jj];
            const double b1 = Bs[tx * 2 + 1][jj];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    if (r0 < n && c0 < n && r0 >= c0) {
        A[static_cast<size_t>(r0) * n + c0] -= acc00;
    }
    if (r0 < n && c0 + 1 < n && r0 >= c0 + 1) {
        A[static_cast<size_t>(r0) * n + c0 + 1] -= acc01;
    }
    if (r0 + 1 < n && c0 < n && r0 + 1 >= c0) {
        A[static_cast<size_t>(r0 + 1) * n + c0] -= acc10;
    }
    if (r0 + 1 < n && c0 + 1 < n && r0 + 1 >= c0 + 1) {
        A[static_cast<size_t>(r0 + 1) * n + c0 + 1] -= acc11;
    }
}

__global__ void zeroUpperKernel(double* __restrict__ A, int n) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    if (idx >= total) {
        return;
    }
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx % n);
    if (j > i) {
        A[idx] = 0.0;
    }
}

// Runs the entire blocked factorization on the device. Returns false if a
// non-positive diagonal pivot is encountered (matrix not positive definite).
bool choleskyDecompositionDevice(double* dA, const size_t n) {
    if (n == 0) {
        return true;
    }
    const int nn = static_cast<int>(n);

    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    const int noFail = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &noFail, sizeof(int), cudaMemcpyHostToDevice));

    for (int p = 0; p < nn; p += kBlockSize) {
        const int nb = min(kBlockSize, nn - p);

        diagonalBlockKernel<<<1, kBlockSize>>>(dA, nn, p, nb, dFail);
        CUDA_CHECK(cudaGetLastError());

        const int rowStart = p + nb;
        if (rowStart < nn) {
            constexpr int kWarps = 8;
            const int blocks = (nn - rowStart + kWarps - 1) / kWarps;
            panelSolveKernel<<<blocks, kWarps * 32>>>(dA, nn, p, nb, rowStart);
            CUDA_CHECK(cudaGetLastError());

            const int tiles = (nn - rowStart + kTile - 1) / kTile;
            dim3 block(kTileThreads, kTileThreads);
            dim3 grid(tiles, tiles);
            trailingUpdateKernel<<<grid, block>>>(dA, nn, p, nb, rowStart);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    const int total = nn * nn;
    const int threads = 256;
    zeroUpperKernel<<<(total + threads - 1) / threads, threads>>>(dA, nn);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    int failIdx = -1;
    CUDA_CHECK(cudaMemcpy(&failIdx, dFail, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dFail));

    if (failIdx != INT_MAX) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               failIdx);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Validation on the GPU
// ---------------------------------------------------------------------------

__device__ __forceinline__ void atomicMaxDouble(double* addr, double value) {
    atomicMax(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(__double_as_longlong(value)));
}

__global__ void validateKernel(const double* __restrict__ L,
                               const double* __restrict__ A_orig, int n,
                               double* __restrict__ maxAbs,
                               double* __restrict__ maxRel) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) {
        return;
    }
    const double* Li = L + static_cast<size_t>(i) * n;
    const double* Lj = L + static_cast<size_t>(j) * n;
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += Li[k] * Lj[k];
    }
    const size_t idx = static_cast<size_t>(i) * n + j;
    const double a = A_orig[idx];
    const double error = fabs(sum - a);
    const double rel = error / (fabs(a) + 1e-10);
    atomicMaxDouble(maxAbs, error);
    atomicMaxDouble(maxRel, rel);
}

bool validateCholeskyDevice(const double* dL, const std::vector<double>& A_orig,
                            const size_t n) {
    if (n == 0) {
        printf("Max absolute error: 0.0000000000e+00\n");
        printf("Max relative error: 0.0000000000e+00\n");
        return true;
    }
    double* dAorig = nullptr;
    double* dMaxAbs = nullptr;
    double* dMaxRel = nullptr;
    CUDA_CHECK(cudaMalloc(&dAorig, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMaxAbs, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMaxRel, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dAorig, A_orig.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dMaxAbs, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(dMaxRel, 0, sizeof(double)));

    const int nn = static_cast<int>(n);
    dim3 block(kTileThreads, kTileThreads);
    dim3 grid((nn + block.x - 1) / block.x, (nn + block.y - 1) / block.y);
    validateKernel<<<grid, block>>>(dL, dAorig, nn, dMaxAbs, dMaxRel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double maxError = 0.0;
    double relError = 0.0;
    CUDA_CHECK(
        cudaMemcpy(&maxError, dMaxAbs, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(&relError, dMaxRel, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dAorig));
    CUDA_CHECK(cudaFree(dMaxAbs));
    CUDA_CHECK(cudaFree(dMaxRel));

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

} // namespace

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

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    double* dA = nullptr;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    }

    // Generate positive definite matrix (on the GPU)
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrixDevice(dA, n);

    if (validate && n > 0) {
        A_orig.resize(n * n);
        CUDA_CHECK(cudaMemcpy(A_orig.data(), dA, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionDevice(dA, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        if (dA != nullptr) {
            CUDA_CHECK(cudaFree(dA));
        }
        return 1;
    }

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
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
        bool valid = validateCholeskyDevice(dA, A_orig, n);

        if (dA != nullptr) {
            CUDA_CHECK(cudaFree(dA));
        }

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    if (dA != nullptr) {
        CUDA_CHECK(cudaFree(dA));
    }
    return 0;
}
