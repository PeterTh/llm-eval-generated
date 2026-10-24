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

#define CHOL_CUDA_CHECK(call)                                                   \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__),       \
                   __FILE__, __LINE__);                                         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct PinnedBuffer {
    double* ptr = nullptr;
    explicit PinnedBuffer(size_t count = 0) { alloc(count); }
    ~PinnedBuffer() {
        if (ptr) cudaFreeHost(ptr);
    }
    void alloc(size_t count) {
        if (ptr) cudaFreeHost(ptr);
        ptr = nullptr;
        const size_t bytes = std::max<size_t>(1, count) * sizeof(double);
        CHOL_CUDA_CHECK(cudaHostAlloc((void**)&ptr, bytes, cudaHostAllocPortable));
    }
    double* data() { return ptr; }
};

__global__ void diagFactorKernel(double* A, size_t ld, size_t b, int* fail,
                                 unsigned long long* failIdx, size_t kb) {
    __shared__ int sFail;
    if (threadIdx.x == 0) sFail = 0;
    __syncthreads();

    for (size_t j = 0; j < b; ++j) {
        if (threadIdx.x == 0) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double v = A[j * ld + k];
                sum += v * v;
            }
            const double val = A[j * ld + j] - sum;
            if (val <= 0.0) {
                sFail = 1;
                *fail = 1;
                *failIdx = (unsigned long long)(kb + j);
                A[j * ld + j] = 0.0;
            } else {
                A[j * ld + j] = sqrt(val);
            }
        }
        __syncthreads();
        if (sFail) return;

        const double d = A[j * ld + j];
        for (size_t i = j + 1 + threadIdx.x; i < b; i += blockDim.x) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * ld + k] * A[j * ld + k];
            }
            A[i * ld + j] = (A[i * ld + j] - sum) / d;
        }
        __syncthreads();
    }

    for (size_t idx = threadIdx.x; idx < b * b; idx += blockDim.x) {
        const size_t r = idx / b;
        const size_t c = idx % b;
        if (c > r) A[r * ld + c] = 0.0;
    }
}

__global__ void trsmKernel(double* A, size_t ld, const double* L11, size_t b,
                           size_t startRow, size_t lrows, size_t kb) {
    constexpr int ROWS_PER_BLOCK = 16;
    extern __shared__ double sx[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row =
        startRow + (size_t)blockIdx.x * ROWS_PER_BLOCK + (size_t)warp;
    if (row >= lrows) return;

    double* x = sx + (size_t)warp * b;
    double* a = A + row * ld + kb;
    for (size_t j = 0; j < b; ++j) {
        const double* Lj = L11 + j * b;
        double part = 0.0;
        for (size_t t = (size_t)lane; t < j; t += 32) {
            part += Lj[t] * x[t];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_down_sync(0xffffffffu, part, off);
        }
        if (lane == 0) {
            const double v = (a[j] - part) / Lj[j];
            x[j] = v;
            a[j] = v;
        }
        __syncwarp();
    }
}

__global__ void updateKernel(double* A, size_t ld, const double* panelAll,
                             size_t b, size_t kb, size_t trailingStart,
                             const size_t* rowGlobal,
                             const int* blockPackedStart, size_t NB,
                             size_t rowStartLocal, size_t lrows, size_t n) {
    constexpr int TILE = 32;
    constexpr int BK = 32;
    __shared__ double As[TILE][BK + 1];
    __shared__ double Bs[TILE][BK + 1];

    const size_t colBase = trailingStart + (size_t)blockIdx.x * TILE;
    const size_t rowLocalBase = rowStartLocal + (size_t)blockIdx.y * TILE;

    if (colBase >= n) return;
    const size_t lastIdx = (rowLocalBase + (size_t)TILE < lrows) ? (rowLocalBase + (size_t)TILE) : lrows;
    const size_t gRowMax = rowGlobal[lastIdx - 1];
    if (colBase > gRowMax) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t r = rowLocalBase + (size_t)ty;
    const size_t c = colBase + (size_t)tx;
    const size_t gRow = (r < lrows) ? rowGlobal[r] : 0;
    const bool valid = (r < lrows) && (c < n) && (c <= gRow);
    double acc = valid ? A[r * ld + c] : 0.0;

    for (size_t kk = 0; kk < b; kk += BK) {
        const size_t kidx = kk + (size_t)tx;
        As[ty][tx] =
            (r < lrows && kidx < b) ? A[r * ld + kb + kidx] : 0.0;
        const size_t cRow = colBase + (size_t)ty;
        size_t bidx = 0;
        if (cRow < n) {
            const size_t bcol = cRow / NB;
            const size_t packedRow =
                (size_t)blockPackedStart[bcol] + (cRow - bcol * NB);
            bidx = packedRow * b + kidx;
        }
        Bs[ty][tx] =
            (cRow < n && kidx < b) ? panelAll[bidx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k2 = 0; k2 < BK; ++k2) {
            if (kk + (size_t)k2 < b) {
                acc -= As[ty][k2] * Bs[tx][k2];
            }
        }
        __syncthreads();
    }

    if (valid) A[r * ld + c] = acc;
}

__global__ void gemmKernel(const double* X, const double* Y, double* C,
                           size_t ldx, size_t ldy, size_t ldc,
                           const size_t* rowGlobalX, size_t M, size_t N,
                           size_t K) {
    constexpr int TILE = 32;
    constexpr int BK = 32;
    __shared__ double As[TILE][BK + 1];
    __shared__ double Bs[TILE][BK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t r = (size_t)blockIdx.y * TILE + (size_t)ty;
    const size_t c = (size_t)blockIdx.x * TILE + (size_t)tx;
    if (r >= M && c >= N) return;

    double acc = 0.0;
    for (size_t kk = 0; kk < K; kk += BK) {
        const size_t kidx = kk + (size_t)tx;
        const size_t xRow =
            (r < M) ? (rowGlobalX ? rowGlobalX[r] : r) : 0;
        As[ty][tx] = (r < M && kidx < K) ? X[xRow * ldx + kidx] : 0.0;
        const size_t cRow = (size_t)blockIdx.x * TILE + (size_t)ty;
        Bs[ty][tx] =
            (cRow < N && kidx < K) ? Y[cRow * ldy + kidx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k2 = 0; k2 < BK; ++k2) {
            if (kk + (size_t)k2 < K) {
                acc += As[ty][k2] * Bs[tx][k2];
            }
        }
        __syncthreads();
    }

    if (r < M && c < N) C[r * ldc + c] = acc;
}

__global__ void addDiagKernel(double* C, size_t ld, size_t rows,
                              const size_t* rowGlobal, double v) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) C[r * ld + rowGlobal[r]] += v;
}

static size_t blockRows(const size_t n, const size_t NB, const size_t b) {
    const size_t gb = b * NB;
    return std::min(NB, n - gb);
}

static void generatePositiveDefiniteMatrixLocal(std::vector<double>& A_local,
                                                const size_t n,
                                                const size_t lrows,
                                                double* dA, double* dB,
                                                const size_t* dRowGlobal) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    CHOL_CUDA_CHECK(
        cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    if (lrows > 0) {
        dim3 block(32, 32);
        dim3 grid((unsigned)((n + 31) / 32), (unsigned)((lrows + 31) / 32));
        gemmKernel<<<grid, block>>>(dB, dB, dA, n, n, n, dRowGlobal, lrows, n, n);
        CHOL_CUDA_CHECK(cudaGetLastError());

        const unsigned blocks = (unsigned)((lrows + 255) / 256);
        addDiagKernel<<<blocks, 256>>>(dA, n, lrows, dRowGlobal, (double)n);
        CHOL_CUDA_CHECK(cudaGetLastError());

        CHOL_CUDA_CHECK(cudaMemcpy(A_local.data(), dA,
                                   lrows * n * sizeof(double),
                                   cudaMemcpyDeviceToHost));
    }
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& A_orig,
                             const size_t n) {
    std::vector<double> reconstructed(n * n);

    double* dL = nullptr;
    double* dR = nullptr;
    CHOL_CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double),
                               cudaMemcpyHostToDevice));
    dim3 block(32, 32);
    dim3 grid((unsigned)((n + 31) / 32), (unsigned)((n + 31) / 32));
    gemmKernel<<<grid, block>>>(dL, dL, dR, n, n, n, nullptr, n, n, n);
    CHOL_CUDA_CHECK(cudaGetLastError());
    CHOL_CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR,
                               n * n * sizeof(double),
                               cudaMemcpyDeviceToHost));
    CHOL_CUDA_CHECK(cudaFree(dL));
    CHOL_CUDA_CHECK(cudaFree(dR));

    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError) \
    reduction(max : relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

static std::vector<double> gatherMatrix(const std::vector<double>& A_local,
                                        const size_t n, const size_t NB,
                                        const size_t m, const int rank,
                                        const int size, MPI_Comm comm) {
    std::vector<int> counts(size), displs(size);
    int total = 0;
    for (int r = 0; r < size; ++r) {
        size_t rows = 0;
        for (size_t b = (size_t)r; b < m; b += (size_t)size) {
            rows += blockRows(n, NB, b);
        }
        counts[r] = (int)(rows * n);
        displs[r] = total;
        total += counts[r];
    }

    std::vector<double> packed;
    if (rank == 0) packed.resize((size_t)total);
    MPI_Gatherv(A_local.data(), (int)(A_local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                comm);

    std::vector<double> full;
    if (rank == 0) {
        full.assign(n * n, 0.0);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < size; ++r) {
            size_t off = (size_t)displs[r];
            for (size_t b = (size_t)r; b < m; b += (size_t)size) {
                const size_t gb = b * NB;
                const size_t rows = blockRows(n, NB, b);
                std::memcpy(&full[gb * n], &packed[off],
                            rows * n * sizeof(double));
                off += rows * n;
            }
        }
    }
    return full;
}

static bool choleskyDistributed(
    double* dA, const size_t n, const size_t NB, const size_t m,
    const int rank, const int size, MPI_Comm comm,
    const std::vector<size_t>& blocks, const std::vector<size_t>& offsets,
    const size_t* dRowGlobal, double* panelLocal, double* panelPacked,
    size_t packedStride, int* dBlockPackedStart, double* dPanelAll,
    double* dL11, int* dFail, unsigned long long* dFailIdx) {
    const size_t lrows = offsets.back();
    const size_t nOwn = blocks.size();
    std::vector<double> L11Host(NB * NB);
    bool ok = true;

    for (size_t k = 0; k < m; ++k) {
        const size_t kb = k * NB;
        const size_t b = std::min(NB, n - kb);
        const size_t trailingStart = kb + b;
        const int owner = (int)(k % (size_t)size);
        const size_t ownerIdx = k / (size_t)size;

        int localFail = 0;
        unsigned long long localFailIdx = 0;

        if (rank == owner) {
            const size_t diagLocalRow = offsets[ownerIdx];
            CHOL_CUDA_CHECK(cudaMemset(dFail, 0, sizeof(int)));
            diagFactorKernel<<<1, 256>>>(dA + diagLocalRow * n + kb, n, b,
                                         dFail, dFailIdx, kb);
            CHOL_CUDA_CHECK(cudaGetLastError());
            CHOL_CUDA_CHECK(
                cudaMemcpy(&localFail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
            if (localFail) {
                CHOL_CUDA_CHECK(cudaMemcpy(&localFailIdx, dFailIdx,
                                           sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost));
            }
            CHOL_CUDA_CHECK(cudaMemcpy2D(
                L11Host.data(), b * sizeof(double),
                dA + diagLocalRow * n + kb, n * sizeof(double),
                b * sizeof(double), b, cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(L11Host.data(), (int)(b * b), MPI_DOUBLE, owner, comm);

        int globalFail = 0;
        unsigned long long globalFailIdx = 0;
        MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&localFailIdx, &globalFailIdx, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, comm);
        if (globalFail) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal "
                       "element %llu\n",
                       globalFailIdx);
            }
            ok = false;
            break;
        }

        CHOL_CUDA_CHECK(
            cudaMemcpy(dL11, L11Host.data(), b * b * sizeof(double),
                       cudaMemcpyHostToDevice));

        size_t firstGT = 0;
        while (firstGT < nOwn && blocks[firstGT] <= k) ++firstGT;

        if (firstGT < nOwn) {
            const size_t startLocal = offsets[firstGT];
            const size_t nrows = lrows - startLocal;
            constexpr int rowsPerBlock = 16;
            const int threads = rowsPerBlock * 32;
            const int grid = (int)((nrows + rowsPerBlock - 1) / rowsPerBlock);
            const size_t shmem = (size_t)rowsPerBlock * b * sizeof(double);
            trsmKernel<<<grid, threads, shmem>>>(dA, n, dL11, b, startLocal,
                                                 lrows, kb);
            CHOL_CUDA_CHECK(cudaGetLastError());
        }

        size_t firstGE = 0;
        while (firstGE < nOwn && blocks[firstGE] < k) ++firstGE;

        double* packedBuf = panelPacked + (k & 1) * packedStride;
        std::vector<int> blockPackedStart(m, 0);
        std::vector<size_t> rankRows(size, 0);
        size_t maxRows = 0;
        for (int r = 0; r < size; ++r) {
            size_t rows = 0;
            for (size_t bb = (size_t)r; bb < m; bb += (size_t)size) {
                if (bb >= k) rows += blockRows(n, NB, bb);
            }
            rankRows[r] = rows;
            maxRows = std::max(maxRows, rows);
        }
        for (int r = 0; r < size; ++r) {
            size_t localCum = 0;
            for (size_t bb = (size_t)r; bb < m; bb += (size_t)size) {
                if (bb < k) continue;
                blockPackedStart[bb] =
                    (int)((size_t)r * maxRows + localCum);
                localCum += blockRows(n, NB, bb);
            }
        }

        size_t packedRows = 0;
        if (firstGE < nOwn) {
            const size_t startLocal = offsets[firstGE];
            packedRows = lrows - startLocal;
            CHOL_CUDA_CHECK(cudaMemcpy2D(
                panelLocal, b * sizeof(double),
                dA + startLocal * n + kb, n * sizeof(double),
                b * sizeof(double), packedRows, cudaMemcpyDeviceToHost));
        }
        std::fill(panelLocal + packedRows * b, panelLocal + maxRows * b, 0.0);

        MPI_Allgather(panelLocal, (int)(maxRows * b), MPI_DOUBLE,
                      packedBuf, (int)(maxRows * b), MPI_DOUBLE, comm);

        const size_t paddedTotal = (size_t)size * maxRows * b;
        CHOL_CUDA_CHECK(
            cudaMemcpy(dBlockPackedStart, blockPackedStart.data(),
                       m * sizeof(int), cudaMemcpyHostToDevice));
        CHOL_CUDA_CHECK(cudaMemcpyAsync(dPanelAll, packedBuf,
                                        paddedTotal * sizeof(double),
                                        cudaMemcpyHostToDevice, 0));

        if (firstGT < nOwn) {
            const size_t rowStartLocal = offsets[firstGT];
            const size_t nrows = lrows - rowStartLocal;
            const size_t ncols = n - trailingStart;
            dim3 block(32, 32);
            dim3 grid((unsigned)((ncols + 31) / 32),
                      (unsigned)((nrows + 31) / 32));
            updateKernel<<<grid, block>>>(dA, n, dPanelAll, b, kb,
                                          trailingStart, dRowGlobal,
                                          dBlockPackedStart, NB,
                                          rowStartLocal, lrows, n);
            CHOL_CUDA_CHECK(cudaGetLastError());
        }
    }

    CHOL_CUDA_CHECK(cudaDeviceSynchronize());
    return ok;
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localRank = 0, localSize = 1;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        CHOL_CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    }

    int maxThreads = omp_get_max_threads();
    int procs = omp_get_num_procs();
    int suggested = std::max(1, procs / std::max(1, localSize));
    omp_set_num_threads(std::min(maxThreads, suggested));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
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

    if (n == 0) {
        if (rank == 0) printf("Matrix size must be positive\n");
        MPI_Finalize();
        return 1;
    }

    const size_t NB = (n < 512) ? 64 : 128;
    const size_t m = (n + NB - 1) / NB;

    std::vector<size_t> blocks, offsets;
    for (size_t b = (size_t)rank; b < m; b += (size_t)size) {
        blocks.push_back(b);
    }
    offsets.resize(blocks.size() + 1);
    offsets[0] = 0;
    for (size_t j = 0; j < blocks.size(); ++j) {
        offsets[j + 1] = offsets[j] + blockRows(n, NB, blocks[j]);
    }
    const size_t lrows = offsets.back();

    std::vector<size_t> rowGlobal(lrows);
    for (size_t j = 0; j < blocks.size(); ++j) {
        const size_t gb = blocks[j] * NB;
        const size_t rows = blockRows(n, NB, blocks[j]);
        for (size_t i = 0; i < rows; ++i) {
            rowGlobal[offsets[j] + i] = gb + i;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A_local(lrows * n);

    double* dA = nullptr;
    double* dB = nullptr;
    size_t* dRowGlobal = nullptr;
    const size_t dASize = std::max<size_t>(1, lrows * n);
    CHOL_CUDA_CHECK(cudaMalloc(&dA, dASize * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMalloc(&dRowGlobal, std::max<size_t>(1, lrows) * sizeof(size_t)));
    if (lrows > 0) {
        CHOL_CUDA_CHECK(cudaMemcpy(dRowGlobal, rowGlobal.data(),
                                   lrows * sizeof(size_t),
                                   cudaMemcpyHostToDevice));
    }

    generatePositiveDefiniteMatrixLocal(A_local, n, lrows, dA, dB, dRowGlobal);
    CHOL_CUDA_CHECK(cudaFree(dB));
    dB = nullptr;

    std::vector<double> A_orig;
    if (validate) {
        A_orig = gatherMatrix(A_local, n, NB, m, rank, size, MPI_COMM_WORLD);
    }

    double* dL11 = nullptr;
    double* dPanelAll = nullptr;
    int* dFail = nullptr;
    unsigned long long* dFailIdx = nullptr;
    const size_t panelCapRows = ((m + (size_t)size - 1) / (size_t)size) * NB;
    const size_t panelCapElems = panelCapRows * (size_t)size * NB;
    CHOL_CUDA_CHECK(cudaMalloc(&dL11, NB * NB * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMalloc(&dPanelAll, panelCapElems * sizeof(double)));
    CHOL_CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    CHOL_CUDA_CHECK(cudaMalloc(&dFailIdx, sizeof(unsigned long long)));

    PinnedBuffer panelLocal(std::max<size_t>(1, panelCapRows * NB));
    PinnedBuffer panelPacked(std::max<size_t>(1, 2 * panelCapElems));
    int* dBlockPackedStart = nullptr;
    CHOL_CUDA_CHECK(cudaMalloc(&dBlockPackedStart,
                               std::max<size_t>(1, m) * sizeof(int)));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDistributed(
        dA, n, NB, m, rank, size, MPI_COMM_WORLD, blocks, offsets, dRowGlobal,
        panelLocal.data(), panelPacked.data(), panelCapElems,
        dBlockPackedStart, dPanelAll, dL11, dFail, dFailIdx);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        cudaFree(dA);
        cudaFree(dRowGlobal);
        cudaFree(dBlockPackedStart);
        cudaFree(dL11);
        cudaFree(dPanelAll);
        cudaFree(dFail);
        cudaFree(dFailIdx);
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    if (lrows > 0) {
        CHOL_CUDA_CHECK(cudaMemcpy(A_local.data(), dA,
                                   lrows * n * sizeof(double),
                                   cudaMemcpyDeviceToHost));
    }

    std::vector<double> A_full =
        gatherMatrix(A_local, n, NB, m, rank, size, MPI_COMM_WORLD);

    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A_full[i * n + j] = 0.0;
            }
        }

        printf("Computation time: %ld ms\n", duration.count());
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }

            cudaFree(dA);
            cudaFree(dRowGlobal);
            cudaFree(dBlockPackedStart);
            cudaFree(dL11);
            cudaFree(dPanelAll);
            cudaFree(dFail);
            cudaFree(dFailIdx);
            MPI_Comm_free(&localComm);
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    cudaFree(dA);
    cudaFree(dRowGlobal);
    cudaFree(dBlockPackedStart);
    cudaFree(dL11);
    cudaFree(dPanelAll);
    cudaFree(dFail);
    cudaFree(dFailIdx);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
