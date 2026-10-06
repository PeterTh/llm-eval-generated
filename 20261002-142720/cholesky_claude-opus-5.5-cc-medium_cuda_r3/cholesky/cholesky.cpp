#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

// Blocked Cholesky decomposition on the GPU (right-looking, tile size NB).
// The device matrix is padded to a multiple of NB; the padding is filled with
// the identity so that it factors trivially and does not affect the result.

constexpr int NB = 64;        // panel / tile size
constexpr int BK = 16;        // k-chunk of the tiled NT-GEMM kernel
constexpr int GEMM_THREADS = 256;

static size_t paddedSize(size_t n) { return (n + NB - 1) / NB * NB; }

enum GemmMode { GEMM_GENERATE = 0, GEMM_SYRK_UPDATE = 1, GEMM_VALIDATE = 2 };

struct ValidateArgs {
    const double* Aorig;
    unsigned long long* maxAbs;   // bit patterns of non-negative doubles
    unsigned long long* maxRel;
};

// Tiled kernel computing S[i][j] = sum_{k<K} X[i][k] * Y[j][k] for a 64x64 tile
// per block (each thread accumulates over k in ascending order), then applying
// a mode-specific epilogue:
//   GENERATE:     C[i][j] = S + (i == j && i < n ? n : 0)
//   SYRK_UPDATE:  C[i][j] -= S       (lower-triangular tiles only)
//   VALIDATE:     track max |S - Aorig[i][j]| and relative error for i,j < n
template <int MODE>
__global__ void __launch_bounds__(GEMM_THREADS)
ntGemmKernel(const double* __restrict__ X, const double* __restrict__ Y, int K, size_t ld,
             double* __restrict__ C, size_t n, const int* __restrict__ info, ValidateArgs va,
             int bjOff) {
    const int bi = blockIdx.y, bj = blockIdx.x + bjOff;
    if (MODE == GEMM_SYRK_UPDATE) {
        if (bj > bi) return;
        if (*info >= 0) return;
    }

    __shared__ double As[BK][NB + 1];
    __shared__ double Bs[BK][NB + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const size_t i0 = (size_t)bi * NB, j0 = (size_t)bj * NB;

    double acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0;

    const double* Xb = X + i0 * ld;
    const double* Yb = Y + j0 * ld;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int l = 0; l < (NB * BK) / GEMM_THREADS; ++l) {
            const int e = tid + l * GEMM_THREADS;
            const int row = e / BK, kk = e % BK;
            As[kk][row] = Xb[(size_t)row * ld + k0 + kk];
            Bs[kk][row] = Yb[(size_t)row * ld + k0 + kk];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[kk][ty + 16 * r];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

    if constexpr (MODE == GEMM_VALIDATE) {
        double localAbs = 0.0, localRel = 0.0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const size_t i = i0 + ty + 16 * r;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const size_t j = j0 + tx + 16 * c;
                if (i < n && j < n) {
                    const double a = va.Aorig[i * ld + j];
                    const double err = fabs(acc[r][c] - a);
                    localAbs = fmax(localAbs, err);
                    localRel = fmax(localRel, err / (fabs(a) + 1e-10));
                }
            }
        }
        // Block reduction
        __shared__ double redAbs[GEMM_THREADS], redRel[GEMM_THREADS];
        redAbs[tid] = localAbs;
        redRel[tid] = localRel;
        __syncthreads();
        for (int s = GEMM_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                redAbs[tid] = fmax(redAbs[tid], redAbs[tid + s]);
                redRel[tid] = fmax(redRel[tid], redRel[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicMax(va.maxAbs, (unsigned long long)__double_as_longlong(redAbs[0]));
            atomicMax(va.maxRel, (unsigned long long)__double_as_longlong(redRel[0]));
        }
    } else {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t i = i0 + ty + 16 * r;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t j = j0 + tx + 16 * c;
            double* p = C + i * ld + j;
            if (MODE == GEMM_GENERATE) {
                *p = acc[r][c] + ((i == j && i < n) ? (double)n : 0.0);
            } else {
                *p -= acc[r][c];
            }
        }
    }
    }
}

// Factor the NB x NB diagonal tile starting at (k, k) in shared memory.
// On failure, records the global index of the failing diagonal element in *info.
__global__ void __launch_bounds__(256) potrfDiagKernel(double* __restrict__ A, size_t ld, int k,
                                                       int* __restrict__ info) {
    if (*info >= 0) return;
    __shared__ double S[NB][NB + 1];
    __shared__ int failed;
    const int tid = threadIdx.x;
    double* T = A + (size_t)k * ld + k;

    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int r = e / NB, c = e % NB;
        S[r][c] = T[(size_t)r * ld + c];
    }
    if (tid == 0) failed = 0;
    __syncthreads();

    for (int c = 0; c < NB; ++c) {
        if (tid == 0) {
            const double val = S[c][c];
            if (val <= 0.0) {
                failed = 1;
                *info = k + c;
            } else {
                S[c][c] = sqrt(val);
            }
        }
        __syncthreads();
        if (failed) return;
        if (tid > c && tid < NB) S[tid][c] /= S[c][c];
        __syncthreads();
        // Rank-1 update of the remaining lower-triangular part
        const int m = NB - c - 1;
        for (int e = tid; e < m * m; e += blockDim.x) {
            const int r = c + 1 + e / m, cc = c + 1 + e % m;
            if (cc <= r) S[r][cc] -= S[r][c] * S[cc][c];
        }
        __syncthreads();
    }

    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int r = e / NB, c = e % NB;
        if (c <= r) T[(size_t)r * ld + c] = S[r][c];
    }
}

// Triangular solve for the panel below the diagonal tile:
//   L[i][k:k+NB] = A[i][k:k+NB] * L_kk^{-T}   for rows i in [k+NB, Np)
// Each block handles NB rows.
__global__ void __launch_bounds__(256) trsmPanelKernel(double* __restrict__ A, size_t ld, int k,
                                                       const int* __restrict__ info) {
    if (*info >= 0) return;
    extern __shared__ double smem[];
    double (*L)[NB + 1] = reinterpret_cast<double (*)[NB + 1]>(smem);
    double (*Xs)[NB + 1] = reinterpret_cast<double (*)[NB + 1]>(smem + NB * (NB + 1));
    const int tid = threadIdx.x;
    const double* D = A + (size_t)k * ld + k;
    double* P = A + ((size_t)k + NB + (size_t)blockIdx.x * NB) * ld + k;

    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int r = e / NB, c = e % NB;
        L[r][c] = (c <= r) ? D[(size_t)r * ld + c] : 0.0;
        Xs[r][c] = P[(size_t)r * ld + c];
    }
    __syncthreads();

    // Each group of 4 threads owns one row; the threads split the columns.
    const int row = tid / 4, lane = tid % 4;
    for (int c = 0; c < NB; ++c) {
        const double x = Xs[row][c] / L[c][c];
        __syncwarp();
        if (lane == 0) Xs[row][c] = x;
        for (int cc = c + 1 + lane; cc < NB; cc += 4) Xs[row][cc] -= x * L[cc][c];
        __syncwarp();
    }
    __syncthreads();

    for (int e = tid; e < NB * NB; e += blockDim.x) {
        const int r = e / NB, c = e % NB;
        P[(size_t)r * ld + c] = Xs[r][c];
    }
}

// Fill the padding of an n x n matrix stored with leading dimension ld:
// zero, or identity on the diagonal if identityPad is set.
__global__ void padFillKernel(double* __restrict__ A, size_t n, size_t ld, int identityPad) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j >= ld || (i < n && j < n)) return;
    A[i * ld + j] = (identityPad && i == j) ? 1.0 : 0.0;
}

// Zero the strictly upper-triangular part of the rows [r0, r0 + NB).
__global__ void zeroUpperKernel(double* __restrict__ A, size_t ld, size_t r0) {
    const size_t i = r0 + blockIdx.y;
    const size_t j = r0 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j > i && j < ld) A[i * ld + j] = 0.0;
}

// Upload an n x n host matrix into a freshly allocated padded device matrix.
static double* uploadPadded(const double* host, size_t n, size_t Np, bool identityPad,
                            cudaStream_t stream = 0) {
    double* d;
    CUDA_CHECK(cudaMalloc(&d, Np * Np * sizeof(double)));
    CUDA_CHECK(cudaMemcpy2DAsync(d, Np * sizeof(double), host, n * sizeof(double),
                                 n * sizeof(double), n, cudaMemcpyHostToDevice, stream));
    if (Np > n) {
        dim3 grid((unsigned)((Np + 255) / 256), (unsigned)Np);
        padFillKernel<<<grid, 256, 0, stream>>>(d, n, Np, identityPad ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }
    return d;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;
    const size_t Np = paddedSize(n);
    const int nt = (int)(Np / NB);

    // Streams: 'update' runs the trailing-matrix updates, 'panel' (high
    // priority) factors the next panel ahead of time (lookahead), 'copy'
    // streams finished row blocks of L back to the host.
    int loPrio, hiPrio;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPrio, &hiPrio));
    cudaStream_t update, panel, copy;
    CUDA_CHECK(cudaStreamCreateWithPriority(&update, cudaStreamNonBlocking, loPrio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&panel, cudaStreamNonBlocking, hiPrio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&copy, cudaStreamNonBlocking, loPrio));
    std::vector<cudaEvent_t> panelDone(nt), rowDone(nt), colReady(nt);
    for (int t = 0; t < nt; ++t) {
        CUDA_CHECK(cudaEventCreateWithFlags(&panelDone[t], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&rowDone[t], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&colReady[t], cudaEventDisableTiming));
    }

    int* dInfo;
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(dInfo, 0xff, sizeof(int), panel));   // -1: no failure
    double* dA = uploadPadded(A.data(), n, Np, true, panel);

    const size_t trsmSmem = 2 * NB * (NB + 1) * sizeof(double);
    CUDA_CHECK(cudaFuncSetAttribute(trsmPanelKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)trsmSmem));

    // Factor panel t: diagonal tile, then (after it) the row block t of L is
    // final, then the triangular solve for the block column below it.
    auto factorPanel = [&](int t) {
        const int k = t * NB;
        potrfDiagKernel<<<1, 256, 0, panel>>>(dA, Np, k, dInfo);
        zeroUpperKernel<<<dim3((unsigned)((Np - k + 255) / 256), NB), 256, 0, panel>>>(dA, Np, k);
        CUDA_CHECK(cudaEventRecord(rowDone[t], panel));
        const int rem = nt - t - 1;
        if (rem > 0) trsmPanelKernel<<<rem, 256, trsmSmem, panel>>>(dA, Np, k, dInfo);
        CUDA_CHECK(cudaEventRecord(panelDone[t], panel));
    };

    factorPanel(0);
    for (int t = 0; t < nt - 1; ++t) {
        const int k = t * NB;
        const int rem = nt - t - 1;
        const double* P = dA + (size_t)(k + NB) * Np + k;
        double* T = dA + (size_t)(k + NB) * Np + (k + NB);
        CUDA_CHECK(cudaStreamWaitEvent(update, panelDone[t], 0));
        // Update the next block column first so the next panel can start...
        ntGemmKernel<GEMM_SYRK_UPDATE><<<dim3(1, rem), GEMM_THREADS, 0, update>>>(
            P, P, NB, Np, T, n, dInfo, ValidateArgs{}, 0);
        CUDA_CHECK(cudaEventRecord(colReady[t], update));
        CUDA_CHECK(cudaStreamWaitEvent(panel, colReady[t], 0));
        factorPanel(t + 1);
        // ...while the rest of the trailing matrix is updated concurrently.
        if (rem > 1) {
            ntGemmKernel<GEMM_SYRK_UPDATE><<<dim3(rem - 1, rem), GEMM_THREADS, 0, update>>>(
                P, P, NB, Np, T, n, dInfo, ValidateArgs{}, 1);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // Copy finished row blocks back as soon as they are final.
    for (int t = 0; t < nt; ++t) {
        const size_t r0 = (size_t)t * NB;
        if (r0 >= n) break;
        const size_t rows = std::min((size_t)NB, n - r0);
        CUDA_CHECK(cudaStreamWaitEvent(copy, rowDone[t], 0));
        CUDA_CHECK(cudaMemcpy2DAsync(A.data() + r0 * n, n * sizeof(double), dA + r0 * Np,
                                     Np * sizeof(double), n * sizeof(double), rows,
                                     cudaMemcpyDeviceToHost, copy));
    }

    int info;
    CUDA_CHECK(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, copy));
    CUDA_CHECK(cudaDeviceSynchronize());
    const bool ok = info < 0;
    if (!ok) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info);
    }

    for (int t = 0; t < nt; ++t) {
        CUDA_CHECK(cudaEventDestroy(panelDone[t]));
        CUDA_CHECK(cudaEventDestroy(rowDone[t]));
        CUDA_CHECK(cudaEventDestroy(colReady[t]));
    }
    CUDA_CHECK(cudaStreamDestroy(update));
    CUDA_CHECK(cudaStreamDestroy(panel));
    CUDA_CHECK(cudaStreamDestroy(copy));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dInfo));
    return ok;
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
    if (n == 0) return;
    
    // Compute A = B * B^T on the GPU and add diagonal dominance (+n on the
    // diagonal) to ensure positive definiteness
    const size_t Np = paddedSize(n);
    const int nt = (int)(Np / NB);
    double* dA;
    CUDA_CHECK(cudaMalloc(&dA, Np * Np * sizeof(double)));
    double* dB = uploadPadded(B.data(), n, Np, false);
    ntGemmKernel<GEMM_GENERATE><<<dim3(nt, nt), GEMM_THREADS>>>(dB, dB, (int)Np, Np, dA, n,
                                                                 nullptr, ValidateArgs{}, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy2D(A.data(), n * sizeof(double), dA, Np * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;
    
    if (n > 0) {
        // Compute L * L^T on the GPU and compare with the original matrix
        const size_t Np = paddedSize(n);
        const int nt = (int)(Np / NB);
        unsigned long long* dErr;
        CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
        double* dL = uploadPadded(L.data(), n, Np, false);
        double* dOrig = uploadPadded(A_orig.data(), n, Np, false);
        ntGemmKernel<GEMM_VALIDATE><<<dim3(nt, nt), GEMM_THREADS>>>(
            dL, dL, (int)Np, Np, nullptr, n, nullptr, ValidateArgs{dOrig, dErr, dErr + 1}, 0);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long hErr[2];
        CUDA_CHECK(cudaMemcpy(hErr, dErr, sizeof(hErr), cudaMemcpyDeviceToHost));
        memcpy(&maxError, &hErr[0], sizeof(double));
        memcpy(&relError, &hErr[1], sizeof(double));
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dOrig));
        CUDA_CHECK(cudaFree(dErr));
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
