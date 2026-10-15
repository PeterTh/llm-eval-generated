#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (A = L * L^T, row-major).
//
// Distribution: the (zero-padded to a multiple of NB) matrix is split into block
// columns that are distributed cyclically over the MPI ranks; each rank drives one GPU
// and stores only the lower part of its block columns on the device.
// Right-looking blocked algorithm with one-step look-ahead:
//   - the owner of the next panel updates and factors it first, then broadcasts it
//     (non-blocking) while all ranks apply the current panel to their trailing columns.
// OpenMP is used for host-side work (final assembly of the result).

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),       \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------

// C[r][c] = beta*C[r][c] + alpha * sum_k X[r*ldx+k] * Y[c*ldy+k] for one 64x64 tile.
// Pointers are already offset to the tile origin; M, Nn are the remaining extents.
__device__ __forceinline__ void gemmNTTile(const double* __restrict__ X, size_t ldx,
                                           const double* __restrict__ Y, size_t ldy,
                                           double* __restrict__ C, size_t ldc, int M,
                                           int Nn, int K, double alpha, double beta) {
    __shared__ double Xs[16][65];
    __shared__ double Ys[16][65];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int idx = tid; idx < 64 * 16; idx += 256) {
            const int r = idx >> 4, kk = idx & 15;
            const bool kok = (k0 + kk) < K;
            Xs[kk][r] = (r < M && kok) ? X[(size_t)r * ldx + k0 + kk] : 0.0;
            Ys[kk][r] = (r < Nn && kok) ? Y[(size_t)r * ldy + k0 + kk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = Xs[kk][ty + 16 * i];
                b[i] = Ys[kk][tx + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty + 16 * i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tx + 16 * j;
            if (c >= Nn) continue;
            double* p = C + (size_t)r * ldc + c;
            *p = (beta == 0.0 ? 0.0 : beta * *p) + alpha * acc[i][j];
        }
    }
}

// General C = beta*C + alpha * X * Y^T (X: MxK, Y: NxK, both row-major).
__global__ void gemmNTKernel(const double* X, size_t ldx, const double* Y, size_t ldy,
                             double* C, size_t ldc, int M, int Nn, int K, double alpha,
                             double beta) {
    const int c0 = blockIdx.x * 64, r0 = blockIdx.y * 64;
    gemmNTTile(X + (size_t)r0 * ldx, ldx, Y + (size_t)c0 * ldy, ldy,
               C + (size_t)r0 * ldc + c0, ldc, M - r0, Nn - c0, K, alpha, beta);
}

// Trailing update of several local block columns with the received panel Pk:
//   column j (global): C -= Pk[rows >= j*nb] * Pk[rows j*nb..(j+1)*nb)]^T
__global__ void updateKernel(const double* Pk, int k, int rank, int nranks, int lstart,
                             int nb, int N, double* base, const size_t* offs) {
    const int l = lstart + blockIdx.y;
    const int j = rank + l * nranks;
    const int M = N - j * nb;
    const int r0 = blockIdx.x * 64;
    if (r0 >= M) return;
    const int c0 = blockIdx.z * 64;
    const double* Y = Pk + (size_t)(j - k) * nb * nb;
    gemmNTTile(Y + (size_t)r0 * nb, nb, Y + (size_t)c0 * nb, nb,
               base + offs[l] + (size_t)r0 * nb + c0, nb, M - r0, nb - c0, nb, -1.0, 1.0);
}

// Unblocked Cholesky of the nb x nb diagonal block (single thread block).
// status = 1-based index of the first non-positive pivot, 0 on success.
__global__ void potrfKernel(double* D, int nb, int* status) {
    __shared__ int fail;
    const int tid = threadIdx.x, bd = blockDim.x;
    if (tid == 0) fail = 0;
    __syncthreads();
    for (int c = 0; c < nb; ++c) {
        if (tid == 0) {
            const double v = D[(size_t)c * nb + c];
            if (v <= 0.0) {
                fail = c + 1;
            } else {
                D[(size_t)c * nb + c] = sqrt(v);
            }
        }
        __syncthreads();
        if (fail) break;
        const double d = D[(size_t)c * nb + c];
        for (int i = c + 1 + tid; i < nb; i += bd) D[(size_t)i * nb + c] /= d;
        __syncthreads();
        const int cnt = nb - c - 1;
        for (int t = tid; t < cnt * cnt; t += bd) {
            const int i = c + 1 + t / cnt, j = c + 1 + t % cnt;
            if (j <= i) D[(size_t)i * nb + j] -= D[(size_t)i * nb + c] * D[(size_t)j * nb + c];
        }
        __syncthreads();
    }
    if (tid == 0) *status = fail;
    if (!fail) {
        for (int t = tid; t < nb * nb; t += bd) {
            const int i = t / nb, j = t % nb;
            if (j > i) D[t] = 0.0;
        }
    }
}

// Solve X * L11^T = A21 in place, one row per thread. L11 is the nb x nb block at D.
__global__ void trsmKernel(const double* __restrict__ D, double* A21, int nb, int M) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    double* row = A21 + (size_t)i * nb;
    for (int j = 0; j < nb; ++j) {
        double s = row[j];
        const double* lj = D + (size_t)j * nb;
        for (int k = 0; k < j; ++k) s -= row[k] * lj[k];
        row[j] = s / lj[j];
    }
}

// Add n to the diagonal and pad with identity outside the n x n region.
__global__ void fixupKernel(double* C, size_t ldc, int rows, int cols, int i0, int j0, int n) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)rows * cols) return;
    const int r = (int)(t / cols), c = (int)(t % cols);
    const int gi = i0 + r, gj = j0 + c;
    double* p = C + (size_t)r * ldc + c;
    if (gi >= n || gj >= n) *p = (gi == gj) ? 1.0 : 0.0;
    else if (gi == gj) *p += (double)n;
}

__device__ __forceinline__ void atomicMaxPos(unsigned long long* addr, double v) {
    if (v != v) return;
    atomicMax(addr, (unsigned long long)__double_as_longlong(v));
}

__global__ void errorKernel(const double* R, const double* O, size_t cnt,
                            unsigned long long* maxAbs, unsigned long long* maxRel) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < cnt;
         i += (size_t)gridDim.x * blockDim.x) {
        const double e = fabs(R[i] - O[i]);
        atomicMaxPos(maxAbs, e);
        atomicMaxPos(maxRel, e / (fabs(O[i]) + 1e-10));
    }
}

// ---------------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------------

static void launchGemmNT(const double* X, size_t ldx, const double* Y, size_t ldy, double* C,
                         size_t ldc, int M, int Nn, int K, double alpha, double beta,
                         cudaStream_t s) {
    dim3 grid((Nn + 63) / 64, (M + 63) / 64);
    gemmNTKernel<<<grid, 256, 0, s>>>(X, ldx, Y, ldy, C, ldc, M, Nn, K, alpha, beta);
}

static void launchFixup(double* C, size_t ldc, int rows, int cols, int i0, int j0, int n,
                        cudaStream_t s) {
    size_t tot = (size_t)rows * cols;
    fixupKernel<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(C, ldc, rows, cols, i0, j0, n);
}

// Fill B with the same pseudo-random sequence as the original code.
static void generateB(std::vector<double>& B) {
    unsigned int seed = 42;
    for (size_t i = 0; i < B.size(); ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
}

static bool validateCholeskyGPU(const double* dL, const double* dB, const size_t n) {
    // Recompute the original matrix A = B*B^T + n*I and L*L^T on the GPU.
    double *dOrig = nullptr, *dRec = nullptr;
    CUDA_CHECK(cudaMalloc(&dOrig, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRec, n * n * sizeof(double)));
    launchGemmNT(dB, n, dB, n, dOrig, n, (int)n, (int)n, (int)n, 1.0, 0.0, 0);
    launchFixup(dOrig, n, (int)n, (int)n, 0, 0, (int)n, 0);
    launchGemmNT(dL, n, dL, n, dRec, n, (int)n, (int)n, (int)n, 1.0, 0.0, 0);
    unsigned long long* dErr;
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
    errorKernel<<<1024, 256>>>(dRec, dOrig, n * n, dErr, dErr + 1);
    unsigned long long h[2];
    CUDA_CHECK(cudaMemcpy(h, dErr, sizeof(h), cudaMemcpyDeviceToHost));
    double maxError, relError;
    memcpy(&maxError, &h[0], sizeof(double));
    memcpy(&relError, &h[1], sizeof(double));
    cudaFree(dOrig);
    cudaFree(dRec);
    cudaFree(dErr);

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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind one GPU per rank (by node-local rank).
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lr;
        MPI_Comm_rank(local, &lr);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lr % ndev));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Block size: large for efficiency, reduced so every rank owns several panels.
    int nb = 128;
    while (nb > 16 && (int)((n + nb - 1) / nb) < 4 * size) nb /= 2;
    const int np = (int)((n + nb - 1) / nb);
    const int N = np * nb;
    const int ni = (int)n;

    cudaStream_t sMain;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));

    // Local panels: global panel j (j % size == rank) -> local index l = j / size.
    // Panel j stores rows j*nb..N-1, nb columns, row-major.
    std::vector<size_t> offs;
    size_t localElems = 0;
    for (int j = rank; j < np; j += size) {
        offs.push_back(localElems);
        localElems += (size_t)(N - j * nb) * nb;
    }
    const int nloc = (int)offs.size();
    double* dLocal = nullptr;
    size_t* dOffs = nullptr;
    CUDA_CHECK(cudaMalloc(&dLocal, std::max<size_t>(localElems, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOffs, std::max(nloc, 1) * sizeof(size_t)));
    if (nloc) CUDA_CHECK(cudaMemcpy(dOffs, offs.data(), nloc * sizeof(size_t), cudaMemcpyHostToDevice));

    // Generate the owned columns of A = B*B^T + n*I directly on the GPU.
    double* dB = nullptr;
    {
        std::vector<double> B(n * n);
        generateB(B);
        CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    }
    for (int l = 0; l < nloc; ++l) {
        const int j = rank + l * size;
        const int rows = N - j * nb;
        double* C = dLocal + offs[l];
        const int rv = std::max(0, std::min(rows, ni - j * nb));
        const int cv = std::max(0, std::min(nb, ni - j * nb));
        if (rv > 0 && cv > 0)
            launchGemmNT(dB + (size_t)j * nb * n, n, dB + (size_t)j * nb * n, n, C, nb, rv, cv,
                         ni, 1.0, 0.0, sMain);
        launchFixup(C, nb, rows, nb, j * nb, j * nb, ni, sMain);
    }
    CUDA_CHECK(cudaStreamSynchronize(sMain));

    // Broadcast buffers (data followed by one status value) and device panel copies.
    const size_t bufElems = (size_t)N * nb + 1;
    double* hbuf[2];
    double* dbuf[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hbuf[b], bufElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dbuf[b], bufElems * sizeof(double)));
    }
    int* dStatus;
    CUDA_CHECK(cudaMalloc(&dStatus, sizeof(int)));

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Factor local panel l (diagonal block + sub-panel) and stage it in host buffer hb.
    auto factorPanel = [&](int l, double* hb) {
        const int j = rank + l * size;
        const int M = N - j * nb;
        double* P = dLocal + offs[l];
        potrfKernel<<<1, 256, 0, sMain>>>(P, nb, dStatus);
        if (M > nb)
            trsmKernel<<<(M - nb + 127) / 128, 128, 0, sMain>>>(P, P + (size_t)nb * nb, nb, M - nb);
        const size_t cnt = (size_t)M * nb;
        CUDA_CHECK(cudaMemcpyAsync(hb, P, cnt * sizeof(double), cudaMemcpyDeviceToHost, sMain));
        int st;
        CUDA_CHECK(cudaMemcpyAsync(&st, dStatus, sizeof(int), cudaMemcpyDeviceToHost, sMain));
        CUDA_CHECK(cudaStreamSynchronize(sMain));
        hb[cnt] = st ? (double)((size_t)j * nb + st) : 0.0;  // 1-based failing index
    };

    long failedAt = 0;
    if (rank == 0) factorPanel(0, hbuf[0]);
    MPI_Bcast(hbuf[0], (int)((size_t)N * nb + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int cur = 0;
    for (int k = 0; k < np; ++k) {
        const size_t cnt = (size_t)(N - k * nb) * nb;
        // The status sits after the data: only valid for the full-size first broadcast
        // (which sends N*nb+1 elements, so read at its own position).
        const size_t statPos = (k == 0) ? (size_t)N * nb : cnt;
        if (hbuf[cur][statPos] != 0.0) {
            failedAt = (long)hbuf[cur][statPos];
            break;
        }
        CUDA_CHECK(cudaMemcpyAsync(dbuf[cur], hbuf[cur], cnt * sizeof(double),
                                   cudaMemcpyHostToDevice, sMain));
        const int nxt = cur ^ 1;
        MPI_Request req = MPI_REQUEST_NULL;
        int lstart = 0;
        while (lstart < nloc && rank + lstart * size <= k) ++lstart;

        if (k + 1 < np) {
            const int o = (k + 1) % size;
            const size_t ncnt = (size_t)(N - (k + 1) * nb) * nb + 1;
            if (rank == o) {
                const int l = (k + 1) / size;  // == lstart
                dim3 g((N - (k + 1) * nb + 63) / 64, 1, (nb + 63) / 64);
                updateKernel<<<g, 256, 0, sMain>>>(dbuf[cur], k, rank, size, l, nb, N, dLocal, dOffs);
                factorPanel(l, hbuf[nxt]);
                lstart = l + 1;
            }
            MPI_Ibcast(hbuf[nxt], (int)ncnt, MPI_DOUBLE, o, MPI_COMM_WORLD, &req);
        }
        if (lstart < nloc) {
            const int jmin = rank + lstart * size;
            dim3 g((N - jmin * nb + 63) / 64, nloc - lstart, (nb + 63) / 64);
            updateKernel<<<g, 256, 0, sMain>>>(dbuf[cur], k, rank, size, lstart, nb, N, dLocal, dOffs);
        }
        CUDA_CHECK(cudaStreamSynchronize(sMain));
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        cur = nxt;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failedAt) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %ld\n", failedAt - 1);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the result on rank 0 only when it is needed.
    int ret = 0;
    if (printResults || validate) {
        std::vector<double> A;
        if (rank == 0) A.assign(n * n, 0.0);
        std::vector<double> tmp;
        for (int j = 0; j < np; ++j) {
            const int o = j % size;
            const size_t cnt = (size_t)(N - j * nb) * nb;
            const double* src = nullptr;
            if (rank == o) {
                CUDA_CHECK(cudaMemcpy(hbuf[0], dLocal + offs[j / size], cnt * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                src = hbuf[0];
                if (rank != 0) MPI_Send(hbuf[0], (int)cnt, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(hbuf[1], (int)cnt, MPI_DOUBLE, o, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                src = hbuf[1];
            }
            if (rank == 0) {
                const int rows = N - j * nb;
                const int c0 = j * nb;
#pragma omp parallel for schedule(static)
                for (int r = 0; r < rows; ++r) {
                    const int gi = c0 + r;
                    if (gi >= ni) continue;
                    for (int c = 0; c < nb && c0 + c < ni && c0 + c <= gi; ++c)
                        A[(size_t)gi * n + c0 + c] = src[(size_t)r * nb + c];
                }
            }
        }
        if (rank == 0) {
            if (printResults) print_results(A, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                double* dL;
                CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(dL, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
                cudaFree(dLocal);
                dLocal = nullptr;
                bool valid = validateCholeskyGPU(dL, dB, n);
                cudaFree(dL);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    for (int b = 0; b < 2; ++b) {
        cudaFreeHost(hbuf[b]);
        cudaFree(dbuf[b]);
    }
    cudaFree(dLocal);
    cudaFree(dB);
    cudaFree(dOffs);
    cudaFree(dStatus);
    cudaStreamDestroy(sMain);
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
