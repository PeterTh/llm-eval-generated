#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (A = L * L^T).
//
// Distribution: right-looking blocked algorithm. The matrix is split into block
// columns of width NB which are distributed cyclically over the MPI ranks (one GPU
// per rank). In step k the owner of block column k factors the diagonal block and
// solves the panel on its GPU, then broadcasts the panel. Every rank updates its own
// trailing block columns on its GPU. The update of the next panel column is done
// first on a separate stream so the next panel factorization / broadcast overlaps
// with the bulk of the trailing update (look-ahead).
// OpenMP is used for host side work (matrix assembly for output).

static const int NB = 64;  // block width (must be 64, matches kernel tiling)

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 2);                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

// Computes a 64x64 tile: acc[i][j] = sum_k X[i0+i][k] * Y[j0+j][k]  (row-major X,Y)
// Block of 16x16 threads, each thread owns a 4x4 sub-tile (strided by 16).
__device__ __forceinline__ void tileDot(double acc[4][4], const double* __restrict__ X, long ldx,
                                        long xRows, const double* __restrict__ Y, long ldy,
                                        long yRows, long K, long i0, long j0) {
    __shared__ double As[16][64];
    __shared__ double Bs[16][64];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    for (long k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q;
            const int r = idx >> 4, kk = idx & 15;
            const long kg = k0 + kk;
            As[kk][r] = (i0 + r < xRows && kg < K) ? X[(i0 + r) * ldx + kg] : 0.0;
            Bs[kk][r] = (j0 + r < yRows && kg < K) ? Y[(j0 + r) * ldy + kg] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                a[r] = As[kk][ty + 16 * r];
                b[r] = Bs[kk][tx + 16 * r];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }
}

// Generate A = B*B^T + n*I directly into the local block-column storage.
// Local column m holds global block column J = me + P*m, stored full height (npad rows,
// ld = NB). Padding rows/columns (>= n) form an identity block.
__global__ void genKernel(const double* __restrict__ B, double* loc, long n, long npad, int P,
                          int me) {
    const long J = me + (long)P * blockIdx.y;
    const long j0 = J * NB;
    const long i0 = j0 + (long)blockIdx.x * 64;
    if (i0 >= npad) return;
    double acc[4][4] = {};
    tileDot(acc, B, n, n, B, n, n, n, i0, j0);
    double* C = loc + (long)blockIdx.y * npad * NB;
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const long i = i0 + threadIdx.y + 16 * r;
            const long j = j0 + threadIdx.x + 16 * c;
            double v;
            if (i < n && j < n)
                v = acc[r][c] + (i == j ? (double)n : 0.0);
            else
                v = (i == j) ? 1.0 : 0.0;
            C[i * NB + (j - j0)] = v;
        }
}

// Factor the 64x64 diagonal block at the top of the panel (single block).
__global__ void diagKernel(double* pan, int* status, long k0) {
    __shared__ double S[NB][NB + 1];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < NB * NB; idx += blockDim.x) S[idx / NB][idx % NB] = pan[idx];
    __syncthreads();
    for (int j = 0; j < NB; ++j) {
        if (tid == 0) {
            const double v = S[j][j];
            if (v <= 0.0 || v != v) {
                if (*status == 0) *status = (int)(k0 + j + 1);
                S[j][j] = 1.0;
            } else {
                S[j][j] = sqrt(v);
            }
        }
        __syncthreads();
        const double d = S[j][j];
        for (int r = j + 1 + tid; r < NB; r += blockDim.x) S[r][j] /= d;
        __syncthreads();
        for (int idx = tid; idx < NB * NB; idx += blockDim.x) {
            const int r = idx / NB, c = idx % NB;
            if (c > j && r >= c) S[r][c] -= S[r][j] * S[c][j];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < NB * NB; idx += blockDim.x) {
        const int r = idx / NB, c = idx % NB;
        pan[idx] = (c <= r) ? S[r][c] : 0.0;
    }
}

// Panel solve: rows below the diagonal block, X * L11^T = A21 (one thread per row).
__global__ void trsmKernel(double* pan, long rows) {
    __shared__ double S[NB][NB + 1];
    for (int idx = threadIdx.x; idx < NB * NB; idx += blockDim.x)
        S[idx / NB][idx % NB] = pan[idx];
    __syncthreads();
    const long r = NB + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    double* row = pan + r * NB;
    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) x[c] = row[c];
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        double s = x[c];
#pragma unroll
        for (int t = 0; t < c; ++t) s -= x[t] * S[c][t];
        x[c] = s / S[c][c];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) row[c] = x[c];
}

// Trailing update of local block columns m in [mFirst, mFirst+mCount):
//   C[i][c] -= sum_t pan[i-k0][t] * pan[J*NB+c-k0][t]    for i >= J*NB
__global__ void updateKernel(const double* __restrict__ pan, double* loc, long npad, long k0,
                             int P, int me, int mFirst) {
    const int m = mFirst + blockIdx.y;
    const long J = me + (long)P * m;
    const long j0 = J * NB;
    const long i0 = j0 + (long)blockIdx.x * 64;
    if (i0 >= npad) return;
    const long rows = npad - k0;
    double acc[4][4] = {};
    tileDot(acc, pan, NB, rows, pan, NB, rows, NB, i0 - k0, j0 - k0);
    double* C = loc + (long)m * npad * NB;
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const long i = i0 + threadIdx.y + 16 * r;
            const int cc = threadIdx.x + 16 * c;
            C[i * NB + cc] -= acc[r][c];
        }
}

// Validation: compare L*L^T with B*B^T + n*I for a slab of rows. Max stored as bit
// patterns of non-negative doubles (monotonic as unsigned integers).
__global__ void validateKernel(const double* __restrict__ L, const double* __restrict__ B, long n,
                               long rowBeg, long rowEnd, unsigned long long* maxAbs,
                               unsigned long long* maxRel) {
    const long i0 = rowBeg + (long)blockIdx.y * 64;
    const long j0 = (long)blockIdx.x * 64;
    double rec[4][4] = {}, org[4][4] = {};
    tileDot(rec, L, n, n, L, n, n, n, i0, j0);
    tileDot(org, B, n, n, B, n, n, n, i0, j0);
    double ea = 0.0, er = 0.0;
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const long i = i0 + threadIdx.y + 16 * r;
            const long j = j0 + threadIdx.x + 16 * c;
            if (i < rowEnd && j < n) {
                const double a = org[r][c] + (i == j ? (double)n : 0.0);
                const double e = fabs(rec[r][c] - a);
                ea = fmax(ea, e);
                er = fmax(er, e / (fabs(a) + 1e-10));
            }
        }
    atomicMax(maxAbs, (unsigned long long)__double_as_longlong(ea));
    atomicMax(maxRel, (unsigned long long)__double_as_longlong(er));
}

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static double bitsToDouble(unsigned long long b) {
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // One GPU per rank (round-robin among ranks on the same node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    const long nbk = (long)((n + NB - 1) / NB);
    const long npad = nbk * NB;
    const long nl = (long)n;
    const int nlocal = (rank < nbk) ? (int)((nbk - rank + P - 1) / P) : 0;
    const size_t colElems = (size_t)npad * NB;

    // Random matrix B (sequential generator => identical to the reference sequence)
    double* dB = nullptr;
    {
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    double* dLoc = nullptr;
    CUDA_CHECK(cudaMalloc(&dLoc, std::max(1, nlocal) * colElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(dLoc, 0, std::max(1, nlocal) * colElems * sizeof(double)));
    if (nlocal > 0) {
        dim3 blk(16, 16), grd((unsigned)(npad / 64), nlocal);
        genKernel<<<grd, blk>>>(dB, dLoc, nl, npad, P, rank);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!validate && dB) {
        CUDA_CHECK(cudaFree(dB));
        dB = nullptr;
    }

    // Work buffers
    double* dBuf[2];
    double* hBuf[2];
    int* dStatus;
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&dBuf[i], colElems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hBuf[i], (colElems + 1) * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&dStatus, sizeof(int)));
    CUDA_CHECK(cudaMemset(dStatus, 0, sizeof(int)));
    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
    cudaEvent_t evPanel, evB[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&evPanel, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evB[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evB[1], cudaEventDisableTiming));

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int failedAt = 0;
    for (long k = 0; k < nbk; ++k) {
        const int owner = (int)(k % P);
        const long k0 = k * NB;
        const long rows = npad - k0;
        const size_t cnt = (size_t)rows * NB;
        double* hb = hBuf[k & 1];
        double* pan = nullptr;

        if (rank == owner) {
            pan = dLoc + (size_t)(k / P) * colElems + (size_t)k0 * NB;
            diagKernel<<<1, 256, 0, sA>>>(pan, dStatus, k0);
            if (rows > NB) {
                const int tb = 64;
                trsmKernel<<<(unsigned)((rows - NB + tb - 1) / tb), tb, 0, sA>>>(pan, rows);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(evPanel, sA));
            int st = 0;
            if (P > 1) CUDA_CHECK(cudaMemcpyAsync(hb, pan, cnt * sizeof(double), cudaMemcpyDeviceToHost, sA));
            CUDA_CHECK(cudaMemcpyAsync(&st, dStatus, sizeof(int), cudaMemcpyDeviceToHost, sA));
            CUDA_CHECK(cudaStreamSynchronize(sA));
            hb[cnt] = (double)st;
        }
        if (P > 1) {
            MPI_Bcast(hb, (int)(cnt + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }
        failedAt = (int)hb[cnt];
        if (failedAt) break;

        if (rank != owner) {
            pan = dBuf[k & 1];
            if (k >= 2) CUDA_CHECK(cudaStreamWaitEvent(sA, evB[k & 1], 0));
            CUDA_CHECK(cudaMemcpyAsync(pan, hb, cnt * sizeof(double), cudaMemcpyHostToDevice, sA));
            CUDA_CHECK(cudaEventRecord(evPanel, sA));
        }
        if (k + 1 >= nbk) break;
        CUDA_CHECK(cudaStreamWaitEvent(sB, evPanel, 0));

        const int m0 = (k + 1 > rank) ? (int)((k + 1 - rank + P - 1) / P) : 0;
        int restFirst = m0, restCount = nlocal - m0;
        const unsigned rowTiles = (unsigned)((npad - (k + 1) * NB) / 64);
        dim3 blk(16, 16);
        if ((k + 1) % P == rank) {
            // look-ahead: update next panel column first on stream A
            if (k >= 1) CUDA_CHECK(cudaStreamWaitEvent(sA, evB[(k - 1) & 1], 0));
            updateKernel<<<dim3(rowTiles, 1), blk, 0, sA>>>(pan, dLoc, npad, k0, P, rank, m0);
            restFirst++;
            restCount--;
        }
        if (restCount > 0) {
            updateKernel<<<dim3(rowTiles, restCount), blk, 0, sB>>>(pan, dLoc, npad, k0, P, rank,
                                                                    restFirst);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evB[k & 1], sB));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failedAt) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failedAt - 1);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather L on rank 0 (only needed for output / validation)
    std::vector<double> A;
    if (printResults || validate) {
        std::vector<double> hLoc((size_t)std::max(1, nlocal) * colElems);
        CUDA_CHECK(cudaMemcpy(hLoc.data(), dLoc, hLoc.size() * sizeof(double), cudaMemcpyDeviceToHost));
        if (rank == 0) {
            A.assign(n * n, 0.0);
            std::vector<double> tmp(P > 1 ? colElems : 0);
            for (long J = 0; J < nbk; ++J) {
                const int o = (int)(J % P);
                const double* src;
                if (o == 0) {
                    src = hLoc.data() + (size_t)(J / P) * colElems;
                } else {
                    MPI_Recv(tmp.data(), (int)colElems, MPI_DOUBLE, o, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    src = tmp.data();
                }
                const long c0 = J * NB;
                const long w = std::min<long>(NB, nl - c0);
#pragma omp parallel for schedule(static)
                for (long i = c0; i < nl; ++i)
                    for (long c = 0; c < w; ++c) A[i * n + c0 + c] = src[i * NB + c];
            }
        } else {
            for (int m = 0; m < nlocal; ++m)
                MPI_Send(hLoc.data() + (size_t)m * colElems, (int)colElems, MPI_DOUBLE, 0, 0,
                         MPI_COMM_WORLD);
        }
    }

    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        // Broadcast L, each rank checks a slab of rows on its GPU
        if (rank != 0) A.resize(n * n);
        {
            const size_t total = n * n, chunk = 1u << 26;
            for (size_t off = 0; off < total; off += chunk)
                MPI_Bcast(A.data() + off, (int)std::min(chunk, total - off), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD);
        }
        double* dL;
        unsigned long long* dMax;
        CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dL, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&dMax, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dMax, 0, 2 * sizeof(unsigned long long)));
        const long tiles = nbk;
        const long tb = tiles / P, tr = tiles % P;
        const long tBeg = rank * tb + std::min<long>(rank, tr);
        const long tCnt = tb + (rank < tr ? 1 : 0);
        if (tCnt > 0) {
            validateKernel<<<dim3((unsigned)tiles, (unsigned)tCnt), dim3(16, 16)>>>(
                dL, dB, nl, tBeg * 64, std::min<long>(nl, (tBeg + tCnt) * 64), dMax, dMax + 1);
            CUDA_CHECK(cudaGetLastError());
        }
        unsigned long long hm[2];
        CUDA_CHECK(cudaMemcpy(hm, dMax, sizeof(hm), cudaMemcpyDeviceToHost));
        double loc[2] = {bitsToDouble(hm[0]), bitsToDouble(hm[1])}, glob[2];
        MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dMax));
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", glob[0]);
            printf("Max relative error: %.10e\n", glob[1]);
            if (glob[1] > 1e-6) {
                printf("Validation failed: relative error too large\n");
                printf("Validation: FAILED\n");
                rc = 1;
            } else {
                printf("Validation: PASSED\n");
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < 2; ++i) {
        cudaFree(dBuf[i]);
        cudaFreeHost(hBuf[i]);
    }
    cudaFree(dStatus);
    cudaFree(dLoc);
    if (dB) cudaFree(dB);
    MPI_Finalize();
    return rc;
}
