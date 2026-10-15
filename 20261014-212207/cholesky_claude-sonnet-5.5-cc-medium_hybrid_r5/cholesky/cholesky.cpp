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

// Distributed blocked Cholesky decomposition (A = L * L^T, L lower triangular).
//
// Parallelization:
//  - MPI:    1D block-cyclic distribution of block columns over ranks (one GPU per rank).
//            The factored panel is broadcast to all ranks every step.
//  - CUDA:   panel factorization (potrf + trsm), trailing-matrix update and matrix
//            generation run on the GPU with hand-written tiled kernels.
//  - OpenMP: host-side work (matrix gather/unpack, mirroring, validation).
// A right-looking algorithm with one-panel lookahead overlaps the panel broadcast
// with the trailing update.

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

static const int NB = 64;  // block size (kernels support widths up to 64)
#define TBM 64
#define TBK 8

// C(r, c) = beta * C(r, c) + alpha * sum_k X(r, k) * Y(c, k) for the block column owned
// locally as index lc. X and Y are rows of `base` (row-major, leading dim ld); the rows
// used are global rows J*NB + r (resp. + c) minus rowoff. C is stored row-major with
// width w, starting at global row J*NB. diagAdd is added to diagonal elements.
__global__ void __launch_bounds__(256)
blockColKernel(double* const* cols, const double* __restrict__ base, size_t ld, int K,
               int n, int nb, int P, int rank, int lc0, int rowoff,
               double alpha, double beta, double diagAdd) {
    const int lc = lc0 + blockIdx.y;
    const long J = rank + (long)lc * P;
    const int jn = (int)(J * nb);
    const int M = n - jn;
    const int w = min(nb, M);
    const int row0 = blockIdx.x * TBM;
    if (row0 >= M) return;

    __shared__ double sX[TBK][TBM + 1];
    __shared__ double sY[TBK][TBM + 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int idx = tid + 256 * s;
            const int r = idx / TBK, c = idx % TBK;
            const int kk = k0 + c;
            const int gr = row0 + r;
            sX[c][r] = (gr < M && kk < K) ? base[(size_t)(jn + gr - rowoff) * ld + kk] : 0.0;
            sY[c][r] = (r < w && kk < K) ? base[(size_t)(jn + r - rowoff) * ld + kk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TBK; ++kk) {
            double xr[4], yc[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) xr[i] = sX[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) yc[j] = sY[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(xr[i], yc[j], acc[i][j]);
        }
        __syncthreads();
    }

    double* C = cols[lc];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tx + 16 * j;
            if (r < M && c < w) {
                double v = alpha * acc[i][j];
                if (beta != 0.0) v += beta * C[(size_t)r * w + c];
                if (r == c) v += diagAdd;
                C[(size_t)r * w + c] = v;
            }
        }
    }
}

// Unblocked Cholesky of the w x w diagonal block at the top of the panel (row-major, ld = w).
// err receives (global index + 1) of a non-positive pivot, or stays untouched.
__global__ void potrfKernel(double* a, int w, double* err, int jn) {
    __shared__ double s[NB][NB + 1];
    __shared__ int bad;
    const int t = threadIdx.x;
    for (int idx = t; idx < w * w; idx += blockDim.x) s[idx / w][idx % w] = a[idx];
    if (t == 0) bad = 0;
    __syncthreads();
    for (int j = 0; j < w; ++j) {
        double v = 0.0;
        if (t >= j && t < w) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += s[t][k] * s[j][k];
            v = s[t][j] - sum;
            if (t == j) {
                if (v <= 0.0) {
                    bad = 1;
                    *err = (double)(jn + j + 1);
                } else {
                    s[j][j] = sqrt(v);
                }
            }
        }
        __syncthreads();
        if (bad) break;
        if (t > j && t < w) s[t][j] = v / s[j][j];
        __syncthreads();
    }
    if (!bad)
        for (int idx = t; idx < w * w; idx += blockDim.x) {
            const int i = idx / w, j = idx % w;
            if (j <= i) a[idx] = s[i][j];
        }
}

// Solve X * L11^T = A21 for the rows below the diagonal block (one thread per row).
__global__ void trsmKernel(double* a, int w, int M) {
    __shared__ double sL[NB][NB + 1];
    for (int idx = threadIdx.x; idx < w * w; idx += blockDim.x) sL[idx / w][idx % w] = a[idx];
    __syncthreads();
    const int r = w + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    double* row = a + (size_t)r * w;
    double x[NB];
    for (int c = 0; c < w; ++c) x[c] = row[c];
    for (int c = 0; c < w; ++c) {
        double sum = 0.0;
        for (int k = 0; k < c; ++k) sum += x[k] * sL[c][k];
        x[c] = (x[c] - sum) / sL[c][c];
    }
    for (int c = 0; c < w; ++c) row[c] = x[c];
}

struct Dist {
    int n = 0, nb = NB, nblk = 0;
    int P = 1, rank = 0, nloc = 0;
    std::vector<double*> cols;   // host copies of local block column pointers
    double** dcols = nullptr;    // device table of the same
    double* dbuf[2] = {nullptr, nullptr};  // device panel buffers
    double* hbuf[2] = {nullptr, nullptr};  // pinned host panel buffers (+1 for error flag)
    double* derr = nullptr;
    cudaStream_t stream{};
    cudaEvent_t evPanel{}, evH2D[2]{};

    int owner(int J) const { return J % P; }
    int width(int J) const { return std::min(nb, n - J * nb); }
    size_t rows(int J) const { return (size_t)(n - (size_t)J * nb); }
    // first locally owned block column index (local) with global index >= J
    int firstLocalFrom(int J) const {
        const int g = J + (((rank - J) % P) + P) % P;
        return g >= nblk ? nloc : g / P;
    }

    void init(int n_, int P_, int rank_) {
        n = n_; P = P_; rank = rank_;
        nblk = (n + nb - 1) / nb;
        nloc = nblk > rank ? (nblk - rank + P - 1) / P : 0;
        cols.resize(nloc);
        for (int lc = 0; lc < nloc; ++lc) {
            const int J = rank + lc * P;
            CUDA_CHECK(cudaMalloc(&cols[lc], rows(J) * width(J) * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&dcols, std::max(nloc, 1) * sizeof(double*)));
        if (nloc) CUDA_CHECK(cudaMemcpy(dcols, cols.data(), nloc * sizeof(double*), cudaMemcpyHostToDevice));
        const size_t pbytes = (size_t)n * nb * sizeof(double);
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMalloc(&dbuf[i], pbytes));
            CUDA_CHECK(cudaMallocHost(&hbuf[i], pbytes + sizeof(double)));
            CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(evH2D[i], 0));
        }
        CUDA_CHECK(cudaMalloc(&derr, sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanel, cudaEventDisableTiming));
    }

    void destroy() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (double* p : cols) cudaFree(p);
        cudaFree(dcols);
        cudaFree(derr);
        for (int i = 0; i < 2; ++i) {
            cudaFree(dbuf[i]);
            cudaFreeHost(hbuf[i]);
            cudaEventDestroy(evH2D[i]);
        }
        cudaEventDestroy(evPanel);
        cudaStreamDestroy(stream);
    }

    // Launch kernel over local block columns [lcBegin, lcEnd).
    void launchBlockCols(int lcBegin, int lcEnd, const double* base, size_t ld, int K, int rowoff,
                         double alpha, double beta, double diagAdd) {
        if (lcBegin >= lcEnd) return;
        const int Jfirst = rank + lcBegin * P;
        const int M = n - Jfirst * nb;
        dim3 grid((M + TBM - 1) / TBM, lcEnd - lcBegin);
        blockColKernel<<<grid, 256, 0, stream>>>(dcols, base, ld, K, n, nb, P, rank, lcBegin,
                                                 rowoff, alpha, beta, diagAdd);
        CUDA_CHECK(cudaGetLastError());
    }

    // Generate A = B*B^T + n*I for all locally owned block columns (lower part).
    void generate(const std::vector<double>& B) {
        double* dB;
        CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        launchBlockCols(0, nloc, dB, n, n, 0, 1.0, 0.0, (double)n);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(dB));
    }

    // Factor local block column J (diagonal block + panel solve) and stage it for broadcast.
    void factorPanel(int J) {
        const int lc = J / P;
        const int w = width(J);
        const size_t M = rows(J);
        double* a = cols[lc];
        double* h = hbuf[J & 1];
        CUDA_CHECK(cudaMemsetAsync(derr, 0, sizeof(double), stream));
        potrfKernel<<<1, 64, 0, stream>>>(a, w, derr, J * nb);
        if (M > (size_t)w) {
            const int bs = 128;
            trsmKernel<<<(int)((M - w + bs - 1) / bs), bs, 0, stream>>>(a, w, (int)M);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h, a, M * w * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h + M * w, derr, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(evPanel, stream));
        CUDA_CHECK(cudaMemcpyAsync(dbuf[J & 1], a, M * w * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    }

    // Make panel J available in dbuf[J&1] on all ranks. Returns false on factorization failure
    // (failingIndex receives the failing diagonal element).
    bool obtainPanel(int J, long& failingIndex) {
        const int w = width(J);
        const size_t M = rows(J);
        double* h = hbuf[J & 1];
        if (owner(J) == rank) {
            CUDA_CHECK(cudaEventSynchronize(evPanel));
        } else {
            CUDA_CHECK(cudaEventSynchronize(evH2D[J & 1]));  // previous H2D from this buffer is done
        }
        MPI_Bcast(h, (int)(M * w + 1), MPI_DOUBLE, owner(J), MPI_COMM_WORLD);
        if (owner(J) != rank) {
            CUDA_CHECK(cudaMemcpyAsync(dbuf[J & 1], h, M * w * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(evH2D[J & 1], stream));
        }
        const double e = h[M * w];
        if (e != 0.0) {
            failingIndex = (long)e - 1;
            return false;
        }
        return true;
    }

    bool factorize(long& failingIndex) {
        if (nblk == 0) return true;
        if (owner(0) == rank) factorPanel(0);
        if (!obtainPanel(0, failingIndex)) return false;
        for (int K = 0; K < nblk; ++K) {
            const int next = K + 1;
            const double* pnl = dbuf[K & 1];
            const int kb = width(K);
            const int rowoff = K * nb;
            int lcBulk = firstLocalFrom(K + 1);
            if (next < nblk && owner(next) == rank) {
                // lookahead: update the next panel column first, factor it, then the rest
                const int lcN = next / P;
                launchBlockCols(lcN, lcN + 1, pnl, kb, kb, rowoff, -1.0, 1.0, 0.0);
                factorPanel(next);
                lcBulk = lcN + 1;
            }
            launchBlockCols(lcBulk, nloc, pnl, kb, kb, rowoff, -1.0, 1.0, 0.0);
            if (next < nblk && !obtainPanel(next, failingIndex)) return false;
        }
        return true;
    }

    // Gather the (lower triangular) distributed matrix onto rank 0. If symmetric, mirror it to the
    // upper triangle; otherwise the upper triangle is left untouched (zero).
    void gather(std::vector<double>& out, bool symmetric) {
        std::vector<double> tmp;
        double* pin = nullptr;
        CUDA_CHECK(cudaMallocHost(&pin, (size_t)n * nb * sizeof(double)));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int J = 0; J < nblk; ++J) {
            const int w = width(J);
            const size_t M = rows(J);
            const int ow = owner(J);
            if (ow == rank) {
                CUDA_CHECK(cudaMemcpy(pin, cols[J / P], M * w * sizeof(double), cudaMemcpyDeviceToHost));
                if (rank != 0) MPI_Send(pin, (int)(M * w), MPI_DOUBLE, 0, J, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(pin, (int)(M * w), MPI_DOUBLE, ow, J, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank == 0) {
                const size_t jn = (size_t)J * nb;
                double* o = out.data();
                const size_t nn = n;
#pragma omp parallel for schedule(static)
                for (long r = 0; r < (long)M; ++r) {
                    const size_t i = jn + r;
                    for (int c = 0; c < w; ++c) {
                        const size_t j = jn + c;
                        if (j <= i) {
                            const double v = pin[(size_t)r * w + c];
                            o[i * nn + j] = v;
                            if (symmetric) o[j * nn + i] = v;
                        }
                    }
                }
            }
        }
        CUDA_CHECK(cudaFreeHost(pin));
    }
};

// Generate the random matrix B exactly as the sequential reference does.
static void generateB(std::vector<double>& B, const size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Both are symmetric and L is lower triangular, so only j <= i, k <= j is computed.
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        const double* Li = &L[i * n];
        for (size_t j = 0; j <= i; ++j) {
            const double* Lj = &L[j * n];
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) {
                sum += Li[k] * Lj[k];
            }
            const double a = A_orig[i * n + j];
            const double error = fabs(sum - a);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(a) + 1e-10));
        }
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

static void printUsage(const char* progName) {
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

    // One GPU per rank, assigned by node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Dist d;
    d.init((int)n, P, rank);

    // Generate positive definite matrix (distributed, on the GPUs)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    {
        std::vector<double> B(n * n);
        generateB(B, n);
        d.generate(B);
    }

    std::vector<double> A, A_orig;
    if (rank == 0) A.assign(n * n, 0.0);
    if (validate) {
        if (rank == 0) A_orig.assign(n * n, 0.0);
        d.gather(A_orig, true);  // save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    long failingIndex = -1;
    bool success = d.factorize(failingIndex);
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    MPI_Barrier(MPI_COMM_WORLD);

    const double t1 = MPI_Wtime();
    const long ms = (long)((t1 - t0) * 1000.0);

    int rc = 0;
    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %ld\n", failingIndex);
            printf("Cholesky decomposition failed\n");
        }
        rc = 1;
    } else {
        if (rank == 0) {
            printf("Computation time: %ld ms\n", ms);

            // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
            double ops = (double)n * n * n / 3.0;
            double gflops = ops / (ms / 1000.0) / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }

        d.gather(A, false);

        // Print results for external validation
        if (printResults && rank == 0) {
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            int valid = 1;
            if (rank == 0) {
                printf("Validating result...\n");
                valid = validateCholesky(A, A_orig, n) ? 1 : 0;
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            rc = valid ? 0 : 1;
        }
    }

    d.destroy();
    MPI_Finalize();
    return rc;
}
