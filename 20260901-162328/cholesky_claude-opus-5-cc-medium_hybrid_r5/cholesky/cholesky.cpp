#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Decomposes a positive definite matrix A into L * L^T where L is lower
// triangular and stored row-major (identical semantics to the original
// sequential reference implementation).
//
// Implementation notes
// --------------------
// The reference code stores A row-major and computes the lower triangular
// factor L.  A row-major lower-triangular L is bit-for-bit the same memory
// layout as a column-major upper-triangular R with A = R^T * R and R = L^T.
// The whole distributed factorization is therefore carried out in column-major
// "upper" form, which maps directly onto cuBLAS.
//
// The matrix is distributed over MPI ranks by block columns in a 1-D
// block-cyclic fashion (block column t is owned by rank t % size).  Only the
// upper trapezoidal part of each block column is stored, halving the memory
// footprint.  Every rank drives one GPU; the whole trailing-matrix update
// (DTRSM + DGEMM) runs in double precision on the GPU through cuBLAS, and the
// matrix never leaves GPU memory during the factorization.
//
// Each step is software pipelined across three CUDA streams:
//   * the next diagonal block is updated, factorized on the CPU and broadcast
//     first (lookahead), so the latency of the sequential panel is covered by
//     the GPU updating the remaining block rows,
//   * the freshly solved block row is then all-gathered while the GPU performs
//     the bulk of the trailing update,
//   * block columns are copied back and sent to rank 0 as soon as they are
//     final, so collecting the result costs (almost) nothing at the end.
//
// OpenMP parallelizes the remaining host-side work: matrix generation, the
// row-major reassembly of the result and the validation.

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),          \
                    __FILE__, __LINE__);                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t st_ = (call);                                                      \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                               \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, __LINE__);   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

static int g_rank = 0;
static int g_size = 1;

// ---------------------------------------------------------------------------
// Block column bookkeeping
// ---------------------------------------------------------------------------
struct BlockLayout {
    size_t n = 0;
    size_t b = 0;              // block size
    int nb = 0;                // number of block columns
    std::vector<size_t> rows;  // stored rows of block t: min(n, (t+1)*b)
    std::vector<size_t> cols;  // columns of block t
    std::vector<size_t> off;   // offset of block t in the local buffer (owner only)
    size_t localElems = 0;     // total local elements

    void build(size_t n_, size_t b_, int rank, int size) {
        n = n_;
        b = b_;
        nb = (int)((n + b - 1) / b);
        rows.resize(nb);
        cols.resize(nb);
        off.assign(nb, 0);
        size_t acc = 0;
        for (int t = 0; t < nb; ++t) {
            const size_t c0 = (size_t)t * b;
            rows[t] = std::min(n, c0 + b);
            cols[t] = rows[t] - c0;
            if (t % size == rank) {
                off[t] = acc;
                acc += rows[t] * cols[t];
            }
        }
        localElems = acc;
    }
};

// ---------------------------------------------------------------------------
// Diagonal block factorization: D = R^T * R with R upper triangular,
// column-major, in place.  Returns -1 on success, otherwise the index of the
// first non positive definite diagonal element.
// ---------------------------------------------------------------------------
// `rowbuf` is scratch space of at least m elements.
//
// The block is small (it fits in L2) and sits on the critical path of every
// iteration of the outer algorithm, so this is kept single threaded: an OpenMP
// fork/join per column would cost far more than the update itself.  Row j is
// gathered into contiguous scratch once so that the rank-1 update becomes a
// pure unit-stride AXPY that the compiler can vectorize.
static int factorUpperBlock(double* D, const size_t m, const size_t ld, double* rowbuf) {
    for (size_t j = 0; j < m; ++j) {
        const double d = D[j + j * ld];
        if (!(d > 0.0)) {
            return (int)j;
        }
        const double r = sqrt(d);
        D[j + j * ld] = r;

        for (size_t l = j + 1; l < m; ++l) {
            const double v = D[j + l * ld] / r;
            D[j + l * ld] = v;
            rowbuf[l] = v;
        }

        // Rank-1 update of the trailing upper triangle.
        for (size_t l = j + 1; l < m; ++l) {
            const double djl = rowbuf[l];
            double* __restrict__ col = D + l * ld;
            const double* __restrict__ row = rowbuf;
            for (size_t i = j + 1; i <= l; ++i) {
                col[i] -= row[i] * djl;
            }
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Generate the local block columns of the symmetric positive definite matrix.
//
// Mirrors the reference generator exactly: A = B * B^T + n * I with B filled
// from rand_r(seed = 42).  The k-summation order is preserved, so the produced
// matrix is bit-for-bit identical to the sequential version.
// ---------------------------------------------------------------------------
static void generateLocal(double* Aloc, const BlockLayout& L) {
    const size_t n = L.n;

    // Default-initialized (not zero filled): the random fill below touches
    // every page exactly once.
    std::unique_ptr<double[]> Bbuf(new double[n * n]);
    double* const B = Bbuf.get();
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Work list: four adjacent columns of one block column at a time.  Their
    // four rows of B stay resident in L2 while the rows of the other operand
    // stream past, which is what makes this memory bound kernel fast.
    struct Tile {
        int t;
        size_t j0;
        size_t nj;
    };
    std::vector<Tile> tiles;
    for (int t = 0; t < L.nb; ++t) {
        if (t % g_size != g_rank) continue;
        for (size_t j = 0; j < L.cols[t]; j += 4) {
            tiles.push_back({t, j, std::min<size_t>(4, L.cols[t] - j)});
        }
    }

    const long long ntiles = (long long)tiles.size();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long c = 0; c < ntiles; ++c) {
        const Tile& tile = tiles[(size_t)c];
        const size_t R = L.rows[tile.t];
        const size_t gj0 = (size_t)tile.t * L.b + tile.j0;
        double* const base = Aloc + L.off[tile.t] + tile.j0 * R;

        size_t i = 0;
        if (tile.nj == 4) {
            const double* __restrict__ c0 = B + (gj0 + 0) * n;
            const double* __restrict__ c1 = B + (gj0 + 1) * n;
            const double* __restrict__ c2 = B + (gj0 + 2) * n;
            const double* __restrict__ c3 = B + (gj0 + 3) * n;
            double* __restrict__ o0 = base;
            double* __restrict__ o1 = base + R;
            double* __restrict__ o2 = base + 2 * R;
            double* __restrict__ o3 = base + 3 * R;

            // 2x4 register blocking; every accumulator still sums over k in
            // ascending order, so each entry is bit-for-bit the reference value.
            for (; i + 2 <= R; i += 2) {
                const double* __restrict__ a0 = B + (i + 0) * n;
                const double* __restrict__ a1 = B + (i + 1) * n;
                double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
                double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    const double x0 = a0[k], x1 = a1[k];
                    const double y0 = c0[k], y1 = c1[k], y2 = c2[k], y3 = c3[k];
                    s00 += x0 * y0;
                    s01 += x0 * y1;
                    s02 += x0 * y2;
                    s03 += x0 * y3;
                    s10 += x1 * y0;
                    s11 += x1 * y1;
                    s12 += x1 * y2;
                    s13 += x1 * y3;
                }
                o0[i] = s00;
                o1[i] = s01;
                o2[i] = s02;
                o3[i] = s03;
                o0[i + 1] = s10;
                o1[i + 1] = s11;
                o2[i + 1] = s12;
                o3[i + 1] = s13;
            }
        }
        // Remaining rows (and short tiles at the end of a block column).
        for (; i < R; ++i) {
            const double* __restrict__ a0 = B + i * n;
            for (size_t j = 0; j < tile.nj; ++j) {
                const double* __restrict__ cj = B + (gj0 + j) * n;
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    s += a0[k] * cj[k];
                }
                base[j * R + i] = s;
            }
        }

        // Diagonal dominance (A[gj][gj] += n).
        for (size_t j = 0; j < tile.nj; ++j) {
            base[j * R + gj0 + j] += (double)n;
        }
    }
}

// ---------------------------------------------------------------------------
// Validation: compute L * L^T and compare against the original matrix.
// ---------------------------------------------------------------------------
static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                             const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(dynamic, 8) reduction(max : maxError, relError)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        const size_t i = (size_t)ii;
        const double* __restrict__ Li = L.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict__ Lj = L.data() + j * n;
            // L is lower triangular: entries beyond min(i, j) are exactly zero.
            const size_t kmax = std::min(i, j) + 1;
            double sum = 0.0;
            for (size_t k = 0; k < kmax; ++k) {
                sum += Li[k] * Lj[k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // --- Node-local setup: one GPU per rank, CPU cores split between ranks ---
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);

    // Give every node-local rank an equal, disjoint share of the node's cores.
    // This makes the hybrid code independent of the launcher's default binding
    // (which often pins a whole rank to a single core).
    {
        const int nCpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
        const int share = std::max(1, nCpus / localSize);
        const int first = (localRank * share) % std::max(1, nCpus);
        cpu_set_t* set = CPU_ALLOC(nCpus);
        const size_t setSize = CPU_ALLOC_SIZE(nCpus);
        CPU_ZERO_S(setSize, set);
        for (int c = 0; c < share; ++c) {
            CPU_SET_S((first + c) % nCpus, setSize, set);
        }
        if (sched_setaffinity(0, setSize, set) != 0 && g_rank == 0) {
            fprintf(stderr, "Warning: could not set CPU affinity\n");
        }
        CPU_FREE(set);

        if (getenv("OMP_NUM_THREADS") == nullptr) {
            omp_set_num_threads(share);
        }
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", g_size,
               omp_get_max_threads(), deviceCount);
    }

    if (n == 0) {
        if (g_rank == 0) {
            printf("Generating positive definite matrix...\n");
            printf("Computing Cholesky decomposition...\n");
            printf("Computation time: 0 ms\n");
            printf("Performance: 0.000 GFLOPS\n");
            const std::vector<double> empty;
            if (printResults) print_results(empty, "CholeskyL");
            if (validate) printf("Validating result...\nValidation: PASSED\n");
        }
        MPI_Finalize();
        return 0;
    }

    // --- Layout ---
    size_t b = (n >= 2048) ? 256 : 128;
    b = std::min(b, n);
    BlockLayout lay;
    lay.build(n, b, g_rank, g_size);
    const int nb = lay.nb;

    // --- Matrix generation (distributed, OpenMP parallel) ---
    if (g_rank == 0) printf("Generating positive definite matrix...\n");

    // Pinned staging area for the local block columns, so that all host/device
    // transfers of the factor are truly asynchronous.
    double* hostLoc = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hostLoc, std::max<size_t>(lay.localElems, 1) * sizeof(double),
                             cudaHostAllocDefault));
    generateLocal(hostLoc, lay);

    // Local element counts / displacements for the final (and optional
    // original-matrix) gather.
    std::vector<int> gCounts(g_size), gDispls(g_size);
    {
        int myCount = (int)lay.localElems;
        MPI_Allgather(&myCount, 1, MPI_INT, gCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int acc = 0;
        for (int p = 0; p < g_size; ++p) {
            gDispls[p] = acc;
            acc += gCounts[p];
        }
    }

    std::vector<double> A;       // full result (rank 0)
    std::vector<double> A_orig;  // original matrix (rank 0, validation only)
    std::vector<double> gBuf;    // gather receive buffer (rank 0)
    if (g_rank == 0) {
        size_t total = 0;
        for (int p = 0; p < g_size; ++p) total += (size_t)gCounts[p];
        gBuf.resize(total ? total : 1);
        // Allocated (and first-touched) up front so that neither the page
        // faults nor the zero fill land in the timed region.  The zero fill
        // also takes care of the upper triangular part of the result.
        A.assign(n * n, 0.0);
    }

    if (validate) {
        MPI_Gatherv(hostLoc, (int)lay.localElems, MPI_DOUBLE, gBuf.data(), gCounts.data(),
                    gDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (g_rank == 0) {
            A_orig.resize(n * n);
            for (int p = 0; p < g_size; ++p) {
                size_t src = (size_t)gDispls[p];
                for (int t = p; t < nb; t += g_size) {
                    const size_t rt = lay.rows[t];
                    const size_t ct = lay.cols[t];
#pragma omp parallel for schedule(static)
                    for (long long jj = 0; jj < (long long)ct; ++jj) {
                        const size_t gj = (size_t)t * lay.b + (size_t)jj;
                        const double* col = gBuf.data() + src + (size_t)jj * rt;
                        for (size_t i = 0; i < rt; ++i) {
                            A_orig[i * n + gj] = col[i];
                            A_orig[gj * n + i] = col[i];  // symmetric
                        }
                    }
                    src += rt * ct;
                }
            }
        }
    }

    // --- GPU / communication resources ---
    // Three streams: compute (cuBLAS), panel-row exchange transfers and result
    // transfers.  Keeping them apart lets the MPI exchange of one block row and
    // the streaming of finished block columns back to rank 0 overlap with the
    // GPU trailing update.
    cudaStream_t streamC, streamT, streamR;
    CUDA_CHECK(cudaStreamCreate(&streamC));
    CUDA_CHECK(cudaStreamCreate(&streamT));
    CUDA_CHECK(cudaStreamCreate(&streamR));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, streamC));

    cudaEvent_t evDiag, evTrsm, evPack, evW, evFin, evRes;
    CUDA_CHECK(cudaEventCreateWithFlags(&evDiag, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evTrsm, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evPack, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evW, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evFin, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evRes, cudaEventDisableTiming));

    double* dLoc = nullptr;
    double* dW = nullptr;  // two alternating block-row buffers of b x n
    double* dDiag = nullptr;
    CUDA_CHECK(cudaMalloc(&dLoc, std::max<size_t>(lay.localElems, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dW, 2 * b * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dDiag, b * b * sizeof(double)));

    double* hSend = nullptr;
    double* hRecv = nullptr;
    double* hDiag = nullptr;
    double* hBcast = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hSend, b * n * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hRecv, b * n * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hDiag, b * b * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hBcast, (b * b + 1) * sizeof(double), cudaHostAllocDefault));

    std::vector<double> rowbuf(b);
    std::vector<int> counts(g_size), displs(g_size);
    std::vector<size_t> srcOff(nb);

    // Offset of every block column inside rank 0's gather buffer.
    std::vector<size_t> blkSrc(nb, 0);
    {
        size_t acc = 0;
        for (int p = 0; p < g_size; ++p) {
            for (int t = p; t < nb; t += g_size) {
                blkSrc[t] = acc;
                acc += lay.rows[t] * lay.cols[t];
            }
        }
    }

    // Block columns are shipped to rank 0 as soon as they are final, so the
    // result collection overlaps with the rest of the factorization.
    std::vector<MPI_Request> resReq;
    if (g_rank == 0) {
        for (int t = 0; t < nb; ++t) {
            if (t % g_size == 0) continue;
            MPI_Request r;
            MPI_Irecv(gBuf.data() + blkSrc[t], (int)(lay.rows[t] * lay.cols[t]), MPI_DOUBLE,
                      t % g_size, t, MPI_COMM_WORLD, &r);
            resReq.push_back(r);
        }
    }

    const double one = 1.0;
    const double negone = -1.0;

    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (lay.localElems) {
        CUDA_CHECK(cudaMemcpyAsync(dLoc, hostLoc, lay.localElems * sizeof(double),
                                   cudaMemcpyHostToDevice, streamC));
    }

    int failIndex = -1;
    int cur = 0;          // index of the dW buffer holding the current block row
    int pendingSend = -1; // block column finalized but not yet handed to MPI

    // Copy a finished block column back to the host (overlapped) and remember
    // that it still has to be sent to rank 0.
    auto retireBlock = [&](int t) {
        CUDA_CHECK(cudaEventRecord(evFin, streamC));
        CUDA_CHECK(cudaStreamWaitEvent(streamR, evFin, 0));
        CUDA_CHECK(cudaMemcpyAsync(hostLoc + lay.off[t], dLoc + lay.off[t],
                                   lay.rows[t] * lay.cols[t] * sizeof(double),
                                   cudaMemcpyDeviceToHost, streamR));
        CUDA_CHECK(cudaEventRecord(evRes, streamR));
        pendingSend = t;
    };

    // Apply R[r,r]^-T to block row r of all locally owned block columns t > r
    // and start moving it to the host.
    auto exchangeStart = [&](int r) {
        const size_t br = lay.cols[r];
        for (int t = r + 1; t < nb; ++t) {
            if (t % g_size != g_rank) continue;
            CUBLAS_CHECK(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     CUBLAS_DIAG_NON_UNIT, (int)br, (int)lay.cols[t], &one, dDiag,
                                     (int)br, dLoc + lay.off[t] + (size_t)r * lay.b,
                                     (int)lay.rows[t]));
        }
        CUDA_CHECK(cudaEventRecord(evTrsm, streamC));
        CUDA_CHECK(cudaStreamWaitEvent(streamT, evTrsm, 0));
        size_t sOff = 0;
        for (int t = r + 1; t < nb; ++t) {
            if (t % g_size != g_rank) continue;
            CUDA_CHECK(cudaMemcpy2DAsync(hSend + sOff, br * sizeof(double),
                                         dLoc + lay.off[t] + (size_t)r * lay.b,
                                         lay.rows[t] * sizeof(double), br * sizeof(double),
                                         lay.cols[t], cudaMemcpyDeviceToHost, streamT));
            sOff += br * lay.cols[t];
        }
        CUDA_CHECK(cudaEventRecord(evPack, streamT));
    };

    // Complete the exchange of block row r: all-gather it and upload it into
    // the alternate dW buffer.  Runs while the GPU is busy with the bulk of the
    // trailing update.
    auto exchangeFinish = [&](int r) {
        const size_t br = lay.cols[r];
        CUDA_CHECK(cudaEventSynchronize(evPack));

        if (pendingSend >= 0) {
            CUDA_CHECK(cudaEventSynchronize(evRes));
            if (g_rank != 0) {
                MPI_Request s;
                MPI_Isend(hostLoc + lay.off[pendingSend],
                          (int)(lay.rows[pendingSend] * lay.cols[pendingSend]), MPI_DOUBLE, 0,
                          pendingSend, MPI_COMM_WORLD, &s);
                resReq.push_back(s);
            }
            pendingSend = -1;
        }

        int acc = 0;
        for (int p = 0; p < g_size; ++p) {
            int c = 0;
            for (int t = p; t < nb; t += g_size) {
                if (t > r) c += (int)(br * lay.cols[t]);
            }
            counts[p] = c;
            displs[p] = acc;
            acc += c;
        }
        if (acc == 0) return;

        MPI_Allgatherv(hSend, counts[g_rank], MPI_DOUBLE, hRecv, counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        for (int p = 0; p < g_size; ++p) {
            size_t off = (size_t)displs[p];
            for (int t = p; t < nb; t += g_size) {
                if (t <= r) continue;
                srcOff[t] = off;
                off += br * lay.cols[t];
            }
        }
        double* Wnext = dW + (size_t)(cur ^ 1) * b * n;
        for (int t = r + 1; t < nb; ++t) {
            CUDA_CHECK(cudaMemcpyAsync(Wnext + (size_t)t * lay.b * br, hRecv + srcOff[t],
                                       br * lay.cols[t] * sizeof(double), cudaMemcpyHostToDevice,
                                       streamT));
        }
        CUDA_CHECK(cudaEventRecord(evW, streamT));
        CUDA_CHECK(cudaStreamWaitEvent(streamC, evW, 0));
    };

    // Factor the very first diagonal block and broadcast it.
    {
        const size_t m0 = lay.cols[0];
        if (g_rank == 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(hDiag, m0 * sizeof(double), dLoc,
                                         lay.rows[0] * sizeof(double), m0 * sizeof(double), m0,
                                         cudaMemcpyDeviceToHost, streamC));
            CUDA_CHECK(cudaStreamSynchronize(streamC));
            const int bad = factorUpperBlock(hDiag, m0, m0, rowbuf.data());
            memcpy(hBcast, hDiag, m0 * m0 * sizeof(double));
            hBcast[m0 * m0] = (double)bad;
        }
        MPI_Bcast(hBcast, (int)(m0 * m0 + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (hBcast[m0 * m0] >= 0.0) {
            failIndex = (int)hBcast[m0 * m0];
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dDiag, hBcast, m0 * m0 * sizeof(double),
                                       cudaMemcpyHostToDevice, streamC));
            if (g_rank == 0) {
                CUDA_CHECK(cudaMemcpy2DAsync(dLoc, lay.rows[0] * sizeof(double), hBcast,
                                             m0 * sizeof(double), m0 * sizeof(double), m0,
                                             cudaMemcpyHostToDevice, streamC));
                retireBlock(0);
            }
            exchangeStart(0);
            exchangeFinish(0);
            cur ^= 1;  // block row 0 now lives in dW[cur]
        }
    }

    for (int k = 0; k + 1 < nb && failIndex < 0; ++k) {
        const size_t bk = lay.cols[k];
        const int next = k + 1;
        const int root = next % g_size;
        const size_t mn = lay.cols[next];
        const size_t c0 = (size_t)next * lay.b;   // first row of block row k+1
        const size_t c1 = lay.rows[next];         // first row below block row k+1
        double* const W = dW + (size_t)cur * b * n;

        // Update rows [c0, c1) of block column t: the part that carries the
        // next block row and is therefore needed first.
        auto gemmHead = [&](int t) {
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(c1 - c0),
                                     (int)lay.cols[t], (int)bk, &negone, W + c0 * bk, (int)bk,
                                     W + (size_t)t * lay.b * bk, (int)bk, &one,
                                     dLoc + lay.off[t] + c0, (int)lay.rows[t]));
        };
        // Update the remaining rows [c1, rows[t]) of block column t.
        auto gemmTail = [&](int t) {
            if (lay.rows[t] <= c1) return;
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(lay.rows[t] - c1),
                                     (int)lay.cols[t], (int)bk, &negone, W + c1 * bk, (int)bk,
                                     W + (size_t)t * lay.b * bk, (int)bk, &one,
                                     dLoc + lay.off[t] + c1, (int)lay.rows[t]));
        };

        MPI_Request req;
        if (g_rank == root) {
            // Lookahead: update and factor the next diagonal block first, then
            // broadcast it while the GPU updates the other block rows.
            gemmHead(next);
            CUDA_CHECK(cudaMemcpy2DAsync(hDiag, mn * sizeof(double), dLoc + lay.off[next] + c0,
                                         lay.rows[next] * sizeof(double), mn * sizeof(double), mn,
                                         cudaMemcpyDeviceToHost, streamC));
            CUDA_CHECK(cudaEventRecord(evDiag, streamC));
            for (int t = next + 1; t < nb; ++t) {
                if (t % g_size == g_rank) gemmHead(t);
            }

            CUDA_CHECK(cudaEventSynchronize(evDiag));
            const int bad = factorUpperBlock(hDiag, mn, mn, rowbuf.data());
            memcpy(hBcast, hDiag, mn * mn * sizeof(double));
            hBcast[mn * mn] = (double)bad;
            MPI_Ibcast(hBcast, (int)(mn * mn + 1), MPI_DOUBLE, root, MPI_COMM_WORLD, &req);

            if (bad < 0) {
                CUDA_CHECK(cudaMemcpyAsync(dDiag, hBcast, mn * mn * sizeof(double),
                                           cudaMemcpyHostToDevice, streamC));
                CUDA_CHECK(cudaMemcpy2DAsync(dLoc + lay.off[next] + c0,
                                             lay.rows[next] * sizeof(double), hBcast,
                                             mn * sizeof(double), mn * sizeof(double), mn,
                                             cudaMemcpyHostToDevice, streamC));
                retireBlock(next);
            }
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        } else {
            MPI_Ibcast(hBcast, (int)(mn * mn + 1), MPI_DOUBLE, root, MPI_COMM_WORLD, &req);
            for (int t = next + 1; t < nb; ++t) {
                if (t % g_size == g_rank) gemmHead(t);
            }
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (hBcast[mn * mn] < 0.0) {
                CUDA_CHECK(cudaMemcpyAsync(dDiag, hBcast, mn * mn * sizeof(double),
                                           cudaMemcpyHostToDevice, streamC));
            }
        }

        if (hBcast[mn * mn] >= 0.0) {
            failIndex = (int)(c0 + (size_t)hBcast[mn * mn]);
            break;
        }

        exchangeStart(next);
        for (int t = next + 1; t < nb; ++t) {
            if (t % g_size == g_rank) gemmTail(t);
        }
        exchangeFinish(next);
        cur ^= 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(streamC));
    CUDA_CHECK(cudaStreamSynchronize(streamR));

    const bool success = (failIndex < 0);

    // --- Collect the distributed factor and rebuild row-major lower L ---
    if (success) {
        if (pendingSend >= 0 && g_rank != 0) {
            MPI_Request s;
            MPI_Isend(hostLoc + lay.off[pendingSend],
                      (int)(lay.rows[pendingSend] * lay.cols[pendingSend]), MPI_DOUBLE, 0,
                      pendingSend, MPI_COMM_WORLD, &s);
            resReq.push_back(s);
        }
        if (!resReq.empty()) {
            MPI_Waitall((int)resReq.size(), resReq.data(), MPI_STATUSES_IGNORE);
        }
        resReq.clear();

        if (g_rank == 0) {
            // Column gj of the column-major upper factor is row gj of the
            // row-major lower factor; the upper triangle is already zero.
#pragma omp parallel for schedule(static, 16)
            for (long long gjj = 0; gjj < (long long)n; ++gjj) {
                const size_t gj = (size_t)gjj;
                const int t = (int)(gj / lay.b);
                const size_t jj = gj - (size_t)t * lay.b;
                const double* col = (t % g_size == 0)
                                        ? hostLoc + lay.off[t] + jj * lay.rows[t]
                                        : gBuf.data() + blkSrc[t] + jj * lay.rows[t];
                memcpy(A.data() + gj * n, col, (gj + 1) * sizeof(double));
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // --- Cleanup of GPU resources ---
    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFreeHost(hRecv));
    CUDA_CHECK(cudaFreeHost(hDiag));
    CUDA_CHECK(cudaFreeHost(hBcast));
    CUDA_CHECK(cudaFree(dLoc));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dDiag));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(evDiag));
    CUDA_CHECK(cudaEventDestroy(evTrsm));
    CUDA_CHECK(cudaEventDestroy(evPack));
    CUDA_CHECK(cudaEventDestroy(evW));
    CUDA_CHECK(cudaEventDestroy(evFin));
    CUDA_CHECK(cudaEventDestroy(evRes));
    CUDA_CHECK(cudaStreamDestroy(streamC));
    CUDA_CHECK(cudaStreamDestroy(streamT));
    CUDA_CHECK(cudaStreamDestroy(streamR));

    if (!success) {
        // The result is streamed to rank 0 while the factorization runs, so a
        // breakdown leaves transfers in flight; tear the job down directly.
        if (g_rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failIndex);
            printf("Cholesky decomposition failed\n");
        }
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    CUDA_CHECK(cudaFreeHost(hostLoc));
    MPI_Comm_free(&shmComm);

    int exitCode = 0;
    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
