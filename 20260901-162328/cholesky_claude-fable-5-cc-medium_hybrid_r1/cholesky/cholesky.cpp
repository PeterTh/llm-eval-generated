#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (right-looking blocked
// algorithm). Decomposes positive definite matrix A into L * L^T where L is
// lower triangular.
//
// Parallelization layout:
//  - MPI: block rows of the matrix are distributed 1D block-cyclically over
//    the ranks; each rank drives one GPU (rank % deviceCount).
//  - CUDA/cuBLAS: panel TRSM and trailing-matrix GEMM updates (the O(n^3)
//    bulk of the work), as well as matrix generation and validation GEMMs.
//  - OpenMP: host-side diagonal block factorization, panel packing/unpacking,
//    result assembly and validation comparison.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, \
                    __LINE__);                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

namespace {

constexpr size_t kBlock = 256; // block size for the blocked factorization

int g_rank = 0;
int g_nprocs = 1;
cublasHandle_t g_cublas;

inline size_t numBlocks(size_t n) { return (n + kBlock - 1) / kBlock; }
inline int ownerOf(size_t blk) { return (int)(blk % (size_t)g_nprocs); }
inline size_t blockSize(size_t blk, size_t n, size_t nb) {
    return (blk == nb - 1) ? n - blk * kBlock : kBlock;
}
// Row offset of global block `blk` inside its owner's local storage. All
// blocks except the global last one have kBlock rows, so this is exact.
inline size_t localRowOffset(size_t blk) { return (blk / (size_t)g_nprocs) * kBlock; }

// Add `val` to the diagonal entries of a block row stored on the device.
// rowBlock points at the first row of the block; the diagonal entries of the
// block sit at column colStart + r for local row r.
__global__ void addDiagKernel(double* rowBlock, size_t n, size_t colStart, int bs,
                              double val) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < bs) {
        rowBlock[(size_t)r * n + colStart + r] += val;
    }
}

// Host Cholesky factorization of an m x m row-major block (lower triangle).
// Returns -1 on success or the index of the offending diagonal element.
int hostPotrf(double* A, int m, int ld) {
    for (int j = 0; j < m; ++j) {
        double d = A[(size_t)j * ld + j];
        for (int t = 0; t < j; ++t) {
            d -= A[(size_t)j * ld + t] * A[(size_t)j * ld + t];
        }
        if (d <= 0.0) {
            return j;
        }
        d = sqrt(d);
        A[(size_t)j * ld + j] = d;
        const double inv = 1.0 / d;
        // Clamp to the cores actually available to this rank so that MPI
        // core-binding does not oversubscribe the block factorization.
        const int nthreads = std::min(omp_get_num_procs(), omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(nthreads) if (m - j > 64 && nthreads > 1)
        for (int i = j + 1; i < m; ++i) {
            double s = A[(size_t)i * ld + j];
            for (int t = 0; t < j; ++t) {
                s -= A[(size_t)i * ld + t] * A[(size_t)j * ld + t];
            }
            A[(size_t)i * ld + j] = s * inv;
        }
    }
    return -1;
}

// Number of matrix rows owned by rank r.
size_t localRows(int r, size_t n, size_t nb) {
    size_t rows = 0;
    for (size_t i = (size_t)r; i < nb; i += (size_t)g_nprocs) {
        rows += blockSize(i, n, nb);
    }
    return rows;
}

} // namespace

// Distributed generation of the symmetric positive definite matrix
// A = B * B^T + n * I. Every rank generates the identical random B (cheap,
// O(n^2)) and computes its own block rows of A on its GPU. The local block
// rows are left in dLocal (row-major, leading dimension n).
void generatePositiveDefiniteMatrix(double* dLocal, const size_t n) {
    const size_t nb = numBlocks(n);

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Sequential PRNG stream, identical on every rank (matches the reference
    // generator exactly).
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const double one = 1.0, zero = 0.0;
    for (size_t i = (size_t)g_rank; i < nb; i += (size_t)g_nprocs) {
        const size_t bs = blockSize(i, n, nb);
        double* dRows = dLocal + localRowOffset(i) * n;
        // Row-major: A_rows = B_rows * B^T. In column-major cuBLAS terms
        // (row-major buffers viewed as their transposes):
        // A_rows^T (n x bs) = B (n x n) * B_rows^T (n x bs).
        CUBLAS_CHECK(cublasDgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)bs,
                                 (int)n, &one, dB, (int)n, dB + i * kBlock * n, (int)n,
                                 &zero, dRows, (int)n));
        const int threads = 128;
        addDiagKernel<<<((int)bs + threads - 1) / threads, threads>>>(
            dRows, n, i * kBlock, (int)bs, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

// Distributed right-looking blocked Cholesky factorization of the block rows
// stored in dLocal. Returns true on success; on failure the owner rank prints
// the offending diagonal element (matching the reference implementation).
bool choleskyDecomposition(double* dLocal, const size_t n) {
    const size_t nb = numBlocks(n);

    double *dLkk = nullptr, *dPanel = nullptr;
    CUDA_CHECK(cudaMalloc(&dLkk, kBlock * kBlock * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanel, (n + kBlock) * kBlock * sizeof(double)));

    double *hLkk = nullptr, *hGath = nullptr;
    CUDA_CHECK(cudaMallocHost(&hLkk, kBlock * kBlock * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hGath, (n + kBlock) * kBlock * sizeof(double)));

    std::vector<int> sendCounts(g_nprocs), displs(g_nprocs);
    const double one = 1.0, minusOne = -1.0;
    bool ok = true;

    for (size_t k = 0; k < nb && ok; ++k) {
        const size_t bk = blockSize(k, n, nb);
        const int owner = ownerOf(k);

        // 1) Factorize the diagonal block on the owner's CPU (OpenMP).
        int status = -1;
        if (g_rank == owner) {
            double* dDiag = dLocal + localRowOffset(k) * n + k * kBlock;
            CUDA_CHECK(cudaMemcpy2D(hLkk, bk * sizeof(double), dDiag, n * sizeof(double),
                                    bk * sizeof(double), bk, cudaMemcpyDeviceToHost));
            status = hostPotrf(hLkk, (int)bk, (int)bk);
            if (status >= 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       k * kBlock + (size_t)status);
                fflush(stdout);
            } else {
                CUDA_CHECK(cudaMemcpy2D(dDiag, n * sizeof(double), hLkk,
                                        bk * sizeof(double), bk * sizeof(double), bk,
                                        cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (status >= 0) {
            ok = false;
            break;
        }
        MPI_Bcast(hLkk, (int)(bk * bk), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (k + 1 == nb) {
            break; // last diagonal block: no trailing matrix
        }
        CUDA_CHECK(cudaMemcpy(dLkk, hLkk, bk * bk * sizeof(double), cudaMemcpyHostToDevice));

        // 2) Panel TRSM on the GPU: L_ik = A_ik * L_kk^-T for owned rows i > k.
        // Row-major solve X * L_kk^T = A_ik maps to column-major
        // trsm(LEFT, UPPER, OP_T) on the transposed views.
        for (size_t i = k + 1; i < nb; ++i) {
            if (ownerOf(i) != g_rank) continue;
            const size_t bi = blockSize(i, n, nb);
            double* dAik = dLocal + localRowOffset(i) * n + k * kBlock;
            CUBLAS_CHECK(cublasDtrsm(g_cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, (int)bk, (int)bi,
                                     &one, dLkk, (int)bk, dAik, (int)n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3) Assemble the panel column k in global i order on every device.
        // Owned blocks are copied device-to-device (strided -> contiguous);
        // blocks from other ranks are exchanged via an MPI allgather, with
        // each rank contributing its owned blocks in increasing i order.
        for (int r = 0; r < g_nprocs; ++r) sendCounts[r] = 0;
        // Offset of block i within its owner's allgather segment (srcOff) and
        // within the assembled i-ordered panel (dstOff).
        std::vector<size_t> srcOff(nb - k - 1), dstOff(nb - k - 1);
        size_t panelOff = 0;
        for (size_t i = k + 1; i < nb; ++i) {
            const size_t bi = blockSize(i, n, nb);
            srcOff[i - k - 1] = (size_t)sendCounts[ownerOf(i)];
            dstOff[i - k - 1] = panelOff;
            sendCounts[ownerOf(i)] += (int)(bi * bk);
            panelOff += bi * bk;
        }
        displs[0] = 0;
        for (int r = 1; r < g_nprocs; ++r) displs[r] = displs[r - 1] + sendCounts[r - 1];

        for (size_t i = k + 1; i < nb; ++i) {
            if (ownerOf(i) != g_rank) continue;
            const size_t bi = blockSize(i, n, nb);
            CUDA_CHECK(cudaMemcpy2DAsync(dPanel + dstOff[i - k - 1], bk * sizeof(double),
                                         dLocal + localRowOffset(i) * n + k * kBlock,
                                         n * sizeof(double), bk * sizeof(double), bi,
                                         cudaMemcpyDeviceToDevice));
        }
        if (g_nprocs > 1) {
            for (size_t i = k + 1; i < nb; ++i) {
                if (ownerOf(i) != g_rank) continue;
                const size_t bi = blockSize(i, n, nb);
                CUDA_CHECK(cudaMemcpyAsync(hGath + displs[g_rank] + srcOff[i - k - 1],
                                           dPanel + dstOff[i - k - 1],
                                           bi * bk * sizeof(double),
                                           cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hGath, sendCounts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            for (size_t i = k + 1; i < nb; ++i) {
                const int r = ownerOf(i);
                if (r == g_rank) continue;
                const size_t bi = blockSize(i, n, nb);
                CUDA_CHECK(cudaMemcpyAsync(dPanel + dstOff[i - k - 1],
                                           hGath + displs[r] + srcOff[i - k - 1],
                                           bi * bk * sizeof(double),
                                           cudaMemcpyHostToDevice));
            }
        }

        // 4) Trailing-matrix update on the GPU: for each owned row i > k,
        // A_i,(k+1..i) -= L_ik * Panel(k+1..i)^T. In column-major terms:
        // C^T (w x bi) += -Panel (w x bk) * L_ik^T (bk x bi).
        for (size_t i = k + 1; i < nb; ++i) {
            if (ownerOf(i) != g_rank) continue;
            const size_t bi = blockSize(i, n, nb);
            const size_t w = std::min((i + 1) * kBlock, n) - (k + 1) * kBlock;
            double* dLik = dLocal + localRowOffset(i) * n + k * kBlock;
            double* dC = dLocal + localRowOffset(i) * n + (k + 1) * kBlock;
            CUBLAS_CHECK(cublasDgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)w, (int)bi,
                                     (int)bk, &minusOne, dPanel, (int)bk, dLik, (int)n,
                                     &one, dC, (int)n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFreeHost(hGath));
    CUDA_CHECK(cudaFreeHost(hLkk));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFree(dLkk));
    return ok;
}

// Distributed validation: L is broadcast to all ranks, each rank reconstructs
// its block rows of L * L^T on its GPU and compares them against its saved
// original rows with OpenMP; the error maxima are reduced over MPI.
bool validateCholesky(const std::vector<double>& L, const double* hOrigLocal,
                      const size_t n) {
    const size_t nb = numBlocks(n);

    double* dL = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));

    double* hChunk = nullptr;
    CUDA_CHECK(cudaMallocHost(&hChunk, kBlock * n * sizeof(double)));
    for (size_t i = 0; i < nb; ++i) {
        const size_t bs = blockSize(i, n, nb);
        if (g_rank == 0) {
            memcpy(hChunk, L.data() + i * kBlock * n, bs * n * sizeof(double));
        }
        MPI_Bcast(hChunk, (int)(bs * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dL + i * kBlock * n, hChunk, bs * n * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    double* dRecon = nullptr;
    double* hRecon = nullptr;
    CUDA_CHECK(cudaMalloc(&dRecon, kBlock * n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hRecon, kBlock * n * sizeof(double)));

    double maxError = 0.0;
    double relError = 0.0;
    const double one = 1.0, zero = 0.0;
    for (size_t i = (size_t)g_rank; i < nb; i += (size_t)g_nprocs) {
        const size_t bs = blockSize(i, n, nb);
        // Row-major: recon_rows = L_rows(i) * L^T, i.e. column-major
        // recon^T (n x bs) = L (n x n) * L_rows^T (n x bs).
        CUBLAS_CHECK(cublasDgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)bs,
                                 (int)n, &one, dL, (int)n, dL + i * kBlock * n, (int)n,
                                 &zero, dRecon, (int)n));
        CUDA_CHECK(cudaMemcpy(hRecon, dRecon, bs * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        const double* orig = hOrigLocal + localRowOffset(i) * n;
#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
        for (size_t e = 0; e < bs * n; ++e) {
            const double error = fabs(hRecon[e] - orig[e]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(orig[e]) + 1e-10));
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFreeHost(hRecon));
    CUDA_CHECK(cudaFreeHost(hChunk));
    CUDA_CHECK(cudaFree(dRecon));
    CUDA_CHECK(cudaFree(dL));

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (g_rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_rank == 0) fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(g_rank % deviceCount));
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid parallelization: %d MPI ranks x %d OpenMP threads, %d GPU(s)\n",
               g_nprocs, omp_get_max_threads(), deviceCount);
    }

    const size_t nb = numBlocks(n);
    const size_t nLoc = localRows(g_rank, n, nb);

    // Full matrix only on rank 0 (assembly, output, validation input).
    std::vector<double> A;
    if (g_rank == 0) {
        A.resize(n * n);
    }

    // Local block rows on the GPU.
    double* dLocal = nullptr;
    CUDA_CHECK(cudaMalloc(&dLocal, std::max<size_t>(nLoc * n, 1) * sizeof(double)));

    // Generate positive definite matrix (distributed over ranks/GPUs).
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dLocal, n);

    double* hOrigLocal = nullptr;
    if (validate && nLoc > 0) {
        // Save the original local rows for validation.
        CUDA_CHECK(cudaMallocHost(&hOrigLocal, nLoc * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(hOrigLocal, dLocal, nLoc * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dLocal, n);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(dLocal));
        cublasDestroy(g_cublas);
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the factorized block rows on rank 0 and assemble the full L.
    {
        double* hLocal = nullptr;
        CUDA_CHECK(cudaMallocHost(&hLocal, std::max<size_t>(nLoc * n, 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(hLocal, dLocal, nLoc * n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        std::vector<int> counts(g_nprocs), displs(g_nprocs);
        for (int r = 0; r < g_nprocs; ++r) {
            counts[r] = (int)(localRows(r, n, nb) * n);
        }
        displs[0] = 0;
        for (int r = 1; r < g_nprocs; ++r) displs[r] = displs[r - 1] + counts[r - 1];

        std::vector<double> gathered;
        if (g_rank == 0) gathered.resize(n * n);
        MPI_Gatherv(hLocal, (int)(nLoc * n), MPI_DOUBLE, gathered.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFreeHost(hLocal));

        if (g_rank == 0) {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nb; ++i) {
                const size_t bs = blockSize(i, n, nb);
                const size_t src = (size_t)displs[ownerOf(i)] + localRowOffset(i) * n;
                memcpy(A.data() + i * kBlock * n, gathered.data() + src,
                       bs * n * sizeof(double));
            }
            // Zero out the upper triangular part (as the reference does).
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A[i * n + j] = 0.0;
                }
            }
        }
    }

    // Print results for external validation
    if (printResults && g_rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (g_rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, hOrigLocal, n);

        if (g_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    if (hOrigLocal) CUDA_CHECK(cudaFreeHost(hOrigLocal));
    CUDA_CHECK(cudaFree(dLocal));
    cublasDestroy(g_cublas);
    MPI_Finalize();
    return exitCode;
}
