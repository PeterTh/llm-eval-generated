#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Distributed hybrid Cholesky decomposition (MPI + OpenMP + CUDA).
// A = L * L^T, A stored row-major on rank 0 for I/O, computed on GPUs.
//
// The matrix (padded to a multiple of NB with an identity block, which does not
// change the leading factor) is split into column panels of width NB. Panel j
// (rows j*NB..np, NB columns, compact row-major) is owned by rank j % P and
// lives on that rank's GPU. A right-looking blocked algorithm with one-panel
// lookahead is used: the owner of the next panel updates and factors it first,
// then broadcasts it (MPI, host staged) while every GPU updates its remaining
// trailing panels asynchronously.

#define NB 64
#define BK 16

#define CUDA_CHECK(x)                                                                 \
    do {                                                                              \
        cudaError_t e_ = (x);                                                         \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),       \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 2);                                             \
        }                                                                             \
    } while (0)

// C = alpha * A * B^T + beta * C for a 64x64 tile per block (all dims multiples of 64).
// mode 1: add n to diagonal entries with global index < n and 1 to padded ones.
__global__ void gemmNT(int K, double alpha, const double* __restrict__ A, size_t lda,
                       const double* __restrict__ B, size_t ldb, double beta,
                       double* __restrict__ C, size_t ldc, int mode, long grow0, long gcol0,
                       long n) {
    __shared__ double As[BK][NB + 1];
    __shared__ double Bs[BK][NB + 1];
    const int t = threadIdx.x;
    const int tx = t & 15, ty = t >> 4;
    A += (size_t)blockIdx.y * NB * lda;
    B += (size_t)blockIdx.x * NB * ldb;
    C += (size_t)blockIdx.y * NB * ldc + (size_t)blockIdx.x * NB;

    double acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty + 16 * i;
            As[tx][r] = A[(size_t)r * lda + k0 + tx];
            Bs[tx][r] = B[(size_t)r * ldb + k0 + tx];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][ty + 16 * i];
                b[i] = Bs[kk][tx + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty + 16 * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tx + 16 * j;
            double* p = C + (size_t)r * ldc + c;
            double v = alpha * acc[i][j];
            if (beta != 0.0) v += beta * *p;
            if (mode == 1) {
                const long gr = grow0 + (long)blockIdx.y * NB + r;
                const long gc = gcol0 + (long)blockIdx.x * NB + c;
                if (gr == gc) v += (gr < n) ? (double)n : 1.0;
            }
            *p = v;
        }
    }
}

// In-place Cholesky of the 64x64 diagonal block (row-major, ld = NB); zeroes its upper part.
// flag = 1 + failing local index, or left untouched on success.
__global__ void potrfDiag(double* P, int* flag) {
    __shared__ double S[NB][NB + 1];
    __shared__ int bad;
    const int t = threadIdx.x;
    for (int e = t; e < NB * NB; e += blockDim.x) S[e / NB][e % NB] = P[e];
    if (t == 0) bad = 0;
    __syncthreads();
    for (int j = 0; j < NB; ++j) {
        if (t == 0) {
            const double val = S[j][j];
            if (val <= 0.0) {
                bad = j + 1;
            } else {
                S[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (bad) break;
        if (t < NB && t > j) S[t][j] /= S[j][j];
        __syncthreads();
        for (int e = t; e < NB * NB; e += blockDim.x) {
            const int i = e / NB, c = e % NB;
            if (c > j && c <= i) S[i][c] -= S[i][j] * S[c][j];
        }
        __syncthreads();
    }
    if (t == 0 && bad) *flag = bad;
    for (int e = t; e < NB * NB; e += blockDim.x) {
        const int i = e / NB, c = e % NB;
        P[e] = (c <= i) ? S[i][c] : 0.0;
    }
}

// X <- X * L^{-T} for rows of X (row-major, ld = NB); one thread per row. L is the 64x64 diag block.
__global__ void trsmPanel(const double* __restrict__ L, double* X, int rows) {
    __shared__ double S[NB][NB];
    for (int e = threadIdx.x; e < NB * NB; e += blockDim.x) S[e / NB][e % NB] = L[e];
    __syncthreads();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double* xr = X + (size_t)row * NB;
    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) x[c] = xr[c];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        double s = x[c];
#pragma unroll
        for (int k = 0; k < c; ++k) s -= x[k] * S[c][k];
        x[c] = s / S[c][c];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) xr[c] = x[c];
}

// Validation error reduction over the leading n x n block (values are >= 0, so
// the bit pattern of a double orders like an unsigned integer).
__global__ void errorKernel(const double* R, const double* Aorig, size_t ld, long n,
                            unsigned long long* maxAbs, unsigned long long* maxRel) {
    const size_t total = (size_t)n * n;
    double ma = 0.0, mr = 0.0;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        const size_t i = e / n, j = e % n;
        const double o = Aorig[i * ld + j];
        const double err = fabs(R[i * ld + j] - o);
        ma = fmax(ma, err);
        mr = fmax(mr, err / (fabs(o) + 1e-10));
    }
    atomicMax(maxAbs, (unsigned long long)__double_as_longlong(ma));
    atomicMax(maxRel, (unsigned long long)__double_as_longlong(mr));
}

static void generateB(std::vector<double>& Bh, const size_t n, const size_t np) {
    // Same random sequence as the reference (inherently sequential), padded with zeros.
    Bh.assign(np * np, 0.0);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) Bh[i * np + j] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
}

static inline size_t panelRows(size_t np, size_t j) { return np - j * NB; }

// Validate L (n x n on rank 0, row-major) against the regenerated original on rank 0's GPU.
static bool validateCholesky(const std::vector<double>& L, const size_t n, const size_t np) {
    std::vector<double> Bh;
    generateB(Bh, n, np);
    const size_t sz = np * np * sizeof(double);
    double *dB, *dA, *dL, *dR;
    CUDA_CHECK(cudaMalloc(&dB, sz));
    CUDA_CHECK(cudaMalloc(&dA, sz));
    CUDA_CHECK(cudaMalloc(&dL, sz));
    CUDA_CHECK(cudaMalloc(&dR, sz));
    CUDA_CHECK(cudaMemcpy(dB, Bh.data(), sz, cudaMemcpyHostToDevice));
    Bh.clear();
    Bh.shrink_to_fit();
    CUDA_CHECK(cudaMemset(dL, 0, sz));
    CUDA_CHECK(cudaMemcpy2D(dL, np * sizeof(double), L.data(), n * sizeof(double),
                            n * sizeof(double), n, cudaMemcpyHostToDevice));
    dim3 grid(np / NB, np / NB);
    gemmNT<<<grid, 256>>>((int)np, 1.0, dB, np, dB, np, 0.0, dA, np, 1, 0, 0, (long)n);
    gemmNT<<<grid, 256>>>((int)np, 1.0, dL, np, dL, np, 0.0, dR, np, 0, 0, 0, (long)n);
    unsigned long long* dErr;
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
    errorKernel<<<1024, 256>>>(dR, dA, np, (long)n, dErr, dErr + 1);
    unsigned long long h[2];
    CUDA_CHECK(cudaMemcpy(h, dErr, sizeof(h), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dErr));
    double maxError, relError;
    memcpy(&maxError, &h[0], sizeof(double));
    memcpy(&relError, &h[1], sizeof(double));

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

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank (round-robin over the GPUs of a node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lr, ndev;
        MPI_Comm_rank(local, &lr);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lr % ndev));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const size_t np = ((n + NB - 1) / NB) * NB;
    const size_t nPanels = np / NB;
    const size_t maxRows = np;

    // ---- Local panel storage ----
    std::vector<size_t> off(nPanels, 0);
    size_t localElems = 0;
    for (size_t j = rank; j < nPanels; j += P) {
        off[j] = localElems;
        localElems += panelRows(np, j) * NB;
    }
    double* dLocal;
    CUDA_CHECK(cudaMalloc(&dLocal, std::max<size_t>(localElems, 1) * sizeof(double)));
    double* dRecv[2];
    double* hBuf[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dRecv[b], maxRows * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hBuf[b], (maxRows * NB + 1) * sizeof(double)));
    }
    int* dFlag;
    CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreate(&sA));
    CUDA_CHECK(cudaStreamCreate(&sB));

    // ---- Generate A = B*B^T + n*I directly into the owned panels (lower part) ----
    {
        std::vector<double> Bh;
        generateB(Bh, n, np);
        double* dB;
        CUDA_CHECK(cudaMalloc(&dB, np * np * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dB, Bh.data(), np * np * sizeof(double), cudaMemcpyHostToDevice));
        for (size_t j = rank; j < nPanels; j += P) {
            const double* Bj = dB + j * NB * np;
            dim3 grid(1, (unsigned)(panelRows(np, j) / NB));
            gemmNT<<<grid, 256, 0, sA>>>((int)np, 1.0, Bj, np, Bj, np, 0.0, dLocal + off[j], NB,
                                         1, (long)(j * NB), (long)(j * NB), (long)n);
        }
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaFree(dB));
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int failedAt = -1;

    // Factor locally owned panel j (already fully updated) and stage it in hBuf[slot] (+ flag)
    auto factorAndStage = [&](size_t j, int slot) {
        double* Pj = dLocal + off[j];
        const size_t M = panelRows(np, j);
        CUDA_CHECK(cudaMemsetAsync(dFlag, 0, sizeof(int), sA));
        potrfDiag<<<1, 256, 0, sA>>>(Pj, dFlag);
        if (M > NB) {
            const int rows = (int)(M - NB);
            trsmPanel<<<(rows + 127) / 128, 128, 0, sA>>>(Pj, Pj + NB * NB, rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(hBuf[slot], Pj, M * NB * sizeof(double), cudaMemcpyDeviceToHost, sA));
        int f = 0;
        CUDA_CHECK(cudaMemcpyAsync(&f, dFlag, sizeof(int), cudaMemcpyDeviceToHost, sA));
        CUDA_CHECK(cudaStreamSynchronize(sA));
        hBuf[slot][M * NB] = (double)f;
    };

    // Broadcast panel j from its owner; non-owners upload it. Returns device pointer of the panel.
    auto exchange = [&](size_t j, int slot) -> const double* {
        const int owner = (int)(j % P);
        const size_t M = panelRows(np, j);
        MPI_Bcast(hBuf[slot], (int)(M * NB + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        const int f = (int)hBuf[slot][M * NB];
        if (f && failedAt < 0) failedAt = (int)(j * NB) + f - 1;
        if (rank == owner) return dLocal + off[j];
        if (!f) {
            CUDA_CHECK(cudaMemcpyAsync(dRecv[slot], hBuf[slot], M * NB * sizeof(double),
                                       cudaMemcpyHostToDevice, sA));
        }
        return dRecv[slot];
    };

    // Panel 0
    if (rank == 0) factorAndStage(0, 0);
    const double* cur = exchange(0, 0);
    CUDA_CHECK(cudaStreamSynchronize(sA));

    for (size_t k = 0; k < nPanels && failedAt < 0; ++k) {
        const size_t nk = k + 1;
        const int nslot = (int)(nk & 1);
        // Lookahead: update and factor the next panel first
        if (nk < nPanels && (int)(nk % P) == rank) {
            const size_t M = panelRows(np, nk);
            const double* Ak = cur + (nk - k) * NB * NB;
            dim3 grid(1, (unsigned)(M / NB));
            gemmNT<<<grid, 256, 0, sA>>>(NB, -1.0, Ak, NB, Ak, NB, 1.0, dLocal + off[nk], NB, 0,
                                         0, 0, 0);
            factorAndStage(nk, nslot);
        }
        // Trailing update of remaining owned panels (async on sB)
        const size_t jfirst = nk + 1;
        const size_t j0 = jfirst + (((size_t)rank + P - jfirst % P) % P);
        for (size_t j = j0; j < nPanels; j += P) {
            const size_t M = panelRows(np, j);
            const double* Aj = cur + (j - k) * NB * NB;
            dim3 grid(1, (unsigned)(M / NB));
            gemmNT<<<grid, 256, 0, sB>>>(NB, -1.0, Aj, NB, Aj, NB, 1.0, dLocal + off[j], NB, 0, 0,
                                         0, 0);
        }
        if (nk < nPanels) {
            // Broadcast overlaps with the trailing update running on the GPU
            const double* nextPanel = exchange(nk, nslot);
            CUDA_CHECK(cudaStreamSynchronize(sA));
            CUDA_CHECK(cudaStreamSynchronize(sB));
            cur = nextPanel;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(sB));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failedAt >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failedAt);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---- Gather L on rank 0 (only needed for -r / -v) ----
    std::vector<double> A;
    if (printResults || validate) {
        if (rank == 0) A.assign(n * n, 0.0);
        std::vector<double> stage(maxRows * NB);
        for (size_t j = 0; j < nPanels; ++j) {
            const int owner = (int)(j % P);
            const size_t M = panelRows(np, j);
            if (rank == owner) {
                CUDA_CHECK(cudaMemcpy(stage.data(), dLocal + off[j], M * NB * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                if (rank != 0) MPI_Send(stage.data(), (int)(M * NB), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(stage.data(), (int)(M * NB), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
            if (rank == 0) {
                const size_t r0 = j * NB;
#pragma omp parallel for schedule(static)
                for (size_t r = r0; r < n; ++r) {
                    for (size_t c = 0; c < NB && r0 + c < n; ++c) {
                        A[r * n + r0 + c] = (r0 + c <= r) ? stage[(r - r0) * NB + c] : 0.0;
                    }
                }
            }
        }
    }

    bool ok = true;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, n, np);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ok = false;
            }
        }
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(dRecv[b]));
        CUDA_CHECK(cudaFreeHost(hBuf[b]));
    }
    CUDA_CHECK(cudaFree(dLocal));
    CUDA_CHECK(cudaFree(dFlag));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
    MPI_Finalize();
    return ok ? 0 : 1;
}
