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
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked (right-looking) Cholesky decomposition.
// The matrix is tiled into BLOCK_SIZE x BLOCK_SIZE blocks. Whole block-columns
// are distributed round-robin across MPI ranks (1D cyclic distribution, as in
// ScaLAPACK-style factorizations). Each rank drives its own GPU via cuBLAS for
// the O(n^3) TRSM/GEMM/SYRK work, and uses OpenMP to fan independent per-block
// GPU work out across multiple CUDA streams (and across CPU threads for host
// side bookkeeping). Data is stored in row-major order throughout, matching
// the original algorithm; cuBLAS (column-major) calls are expressed using the
// standard "transpose trick" so that no physical transposition is ever needed.

static constexpr size_t BLOCK_SIZE = 128;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (call);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(_err));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t _st = (call);                                                     \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__,    \
                    (int)_st);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Simple Cholesky decomposition (sequential, unblocked algorithm).
// Used both as the reference algorithm description and as the leaf-level
// factorization kernel applied to a single diagonal block on the CPU.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

namespace hybrid {

inline size_t numBlocks(size_t n, size_t B) { return (n + B - 1) / B; }
inline size_t blockStart(size_t idx, size_t B) { return idx * B; }
inline size_t blockLen(size_t idx, size_t n, size_t B) {
    const size_t start = idx * B;
    return std::min(B, n - start);
}

struct BlockMat {
    double* dptr = nullptr;
    size_t rows = 0;
    size_t cols = 0;
};

// Per-OpenMP-thread CUDA resources: each host thread gets its own stream and
// cuBLAS handle so independent blocks can be processed concurrently on the GPU.
struct ThreadCudaCtx {
    bool initialized = false;
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
};

ThreadCudaCtx& threadCudaCtx(int deviceId) {
    static thread_local ThreadCudaCtx ctx;
    if (!ctx.initialized) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));
        CUBLAS_CHECK(cublasCreate(&ctx.handle));
        CUBLAS_CHECK(cublasSetStream(ctx.handle, ctx.stream));
        ctx.initialized = true;
    }
    return ctx;
}

void extractBlock(const std::vector<double>& A, size_t n, size_t rowStart, size_t rowLen,
                   size_t colStart, size_t colLen, std::vector<double>& buf) {
    buf.resize(rowLen * colLen);
    for (size_t r = 0; r < rowLen; ++r) {
        std::memcpy(&buf[r * colLen], &A[(rowStart + r) * n + colStart], colLen * sizeof(double));
    }
}

void insertBlock(std::vector<double>& A, size_t n, size_t rowStart, size_t rowLen,
                  size_t colStart, size_t colLen, const std::vector<double>& buf) {
    for (size_t r = 0; r < rowLen; ++r) {
        std::memcpy(&A[(rowStart + r) * n + colStart], &buf[r * colLen], colLen * sizeof(double));
    }
}

// Force CUDA context creation (and pick this rank's GPU) ahead of time, so the
// one-time driver/context initialization cost (which can be hundreds of ms
// and does not overlap well across ranks sharing a GPU) is not misattributed
// to the timed decomposition.
void warmupCuda(MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(0));  // triggers context creation
}

// Distributed, GPU-accelerated Cholesky decomposition.
// Only rank 0's `A` is meaningful as input/output; all ranks must call this
// function collectively. On success, rank 0's `A` holds L (lower triangular,
// upper part zeroed), matching the semantics of choleskyDecomposition().
bool choleskyDecompositionDistributed(std::vector<double>& A, const size_t n, MPI_Comm comm) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // Map ranks to GPUs using node-local rank, so multi-node clusters spread
    // ranks across the GPUs available on each node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    // Each rank only needs enough host threads to fan its own independent
    // GPU work (TRSM/GEMM/SYRK calls) across CUDA streams; oversubscribing
    // with the full node core count per rank hurts multi-rank-per-node runs.
    const int originalMaxThreads = omp_get_max_threads();
    const int coresPerNode = omp_get_num_procs();
    const int threadsPerRank = std::max(1, std::min(16, coresPerNode / std::max(1, localSize)));
    omp_set_num_threads(threadsPerRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(comm, 1);
    }
    const int deviceId = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));

    const size_t B = BLOCK_SIZE;
    const size_t nb = numBlocks(n, B);

    // localCol[j] holds the blocks (i, j) for i = j..nb-1, only populated for
    // block-columns j owned by this rank (owner(j) == j % nprocs).
    std::vector<std::vector<BlockMat>> localCol(nb);
    auto owner = [nprocs](size_t j) -> int { return (int)(j % (size_t)nprocs); };
    auto getBlock = [&localCol](size_t j, size_t i) -> BlockMat& { return localCol[j][i - j]; };

    for (size_t j = 0; j < nb; ++j) {
        if (owner(j) == rank) {
            const size_t cj = blockLen(j, n, B);
            localCol[j].resize(nb - j);
            for (size_t i = j; i < nb; ++i) {
                const size_t ri = blockLen(i, n, B);
                double* d = nullptr;
                CUDA_CHECK(cudaMalloc(&d, ri * cj * sizeof(double)));
                localCol[j][i - j] = BlockMat{d, ri, cj};
            }
        }
    }

    // Per-rank byte offset of each owned block within a single packed
    // send/receive buffer, laid out as (j, i) in increasing order. Computing
    // this the same way on every rank avoids having to communicate a layout.
    auto packedOffsets = [&](int r) {
        std::vector<size_t> off;
        size_t total = 0;
        for (size_t j = 0; j < nb; ++j) {
            if (owner(j) != r) continue;
            const size_t cj = blockLen(j, n, B);
            for (size_t i = j; i < nb; ++i) {
                off.push_back(total);
                total += blockLen(i, n, B) * cj;
            }
        }
        return std::make_pair(off, total);
    };

    // --- Distribute blocks from rank 0 to their owning ranks (one message
    // per destination rank rather than one per block) ---
    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            auto [offsets, total] = packedOffsets(r);
            std::vector<double> buf(total);
            size_t idx = 0;
            for (size_t j = 0; j < nb; ++j) {
                if (owner(j) != r) continue;
                const size_t cj = blockLen(j, n, B);
                for (size_t i = j; i < nb; ++i) {
                    const size_t ri = blockLen(i, n, B);
                    std::vector<double> blk;
                    extractBlock(A, n, blockStart(i, B), ri, blockStart(j, B), cj, blk);
                    std::memcpy(buf.data() + offsets[idx], blk.data(), ri * cj * sizeof(double));
                    ++idx;
                }
            }
            if (r == 0) {
                idx = 0;
                for (size_t j = 0; j < nb; ++j) {
                    if (owner(j) != 0) continue;
                    for (size_t i = j; i < nb; ++i) {
                        BlockMat& b = getBlock(j, i);
                        CUDA_CHECK(cudaMemcpy(b.dptr, buf.data() + offsets[idx],
                                               b.rows * b.cols * sizeof(double), cudaMemcpyHostToDevice));
                        ++idx;
                    }
                }
            } else {
                MPI_Send(buf.data(), (int)total, MPI_DOUBLE, r, 0, comm);
            }
        }
    } else {
        auto [offsets, total] = packedOffsets(rank);
        std::vector<double> buf(total);
        MPI_Recv(buf.data(), (int)total, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
        size_t idx = 0;
        for (size_t j = 0; j < nb; ++j) {
            if (owner(j) != rank) continue;
            for (size_t i = j; i < nb; ++i) {
                BlockMat& b = getBlock(j, i);
                CUDA_CHECK(cudaMemcpy(b.dptr, buf.data() + offsets[idx], b.rows * b.cols * sizeof(double),
                                       cudaMemcpyHostToDevice));
                ++idx;
            }
        }
    }

    // Persistent scratch buffer for the broadcast panel on non-owner ranks,
    // sized for the largest possible panel (n rows x B columns). Reused every
    // iteration to avoid expensive cudaMalloc/cudaFree calls in the hot loop.
    double* panelScratch = nullptr;
    CUDA_CHECK(cudaMalloc(&panelScratch, n * B * sizeof(double)));

    bool failed = false;

    // --- Right-looking blocked factorization ---
    for (size_t k = 0; k < nb && !failed; ++k) {
        const int ownerK = owner(k);
        const size_t rk = blockLen(k, n, B);

        int failFlag = 0;
        if (rank == ownerK) {
            BlockMat& Lkk = getBlock(k, k);
            std::vector<double> hostBlock(rk * rk);
            CUDA_CHECK(cudaMemcpy(hostBlock.data(), Lkk.dptr, rk * rk * sizeof(double),
                                   cudaMemcpyDeviceToHost));
            const bool ok = choleskyDecomposition(hostBlock, rk);
            failFlag = ok ? 0 : 1;
            if (ok) {
                CUDA_CHECK(cudaMemcpy(Lkk.dptr, hostBlock.data(), rk * rk * sizeof(double),
                                       cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&failFlag, 1, MPI_INT, ownerK, comm);
        if (failFlag) {
            failed = true;
            break;
        }

        // Panel: solve L(i,k) = A(i,k) * L(k,k)^-T for all i > k owned by ownerK.
        if (rank == ownerK) {
            BlockMat& Lkk = getBlock(k, k);
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = k + 1; i < nb; ++i) {
                ThreadCudaCtx& ctx = threadCudaCtx(deviceId);
                BlockMat& Bik = getBlock(k, i);
                const double alpha = 1.0;
                CUBLAS_CHECK(cublasDtrsm(ctx.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, (int)rk, (int)Bik.rows,
                                          &alpha, Lkk.dptr, (int)rk, Bik.dptr, (int)rk));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            }
        }

        // Broadcast the whole panel (blocks (k..nb-1, k)) to every rank.
        std::vector<size_t> rows(nb);
        std::vector<size_t> offset(nb, 0);
        size_t totalElems = 0;
        for (size_t i = k; i < nb; ++i) {
            rows[i] = blockLen(i, n, B);
            offset[i] = totalElems;
            totalElems += rows[i] * rk;
        }

        std::vector<double> panelHost(totalElems);
        if (rank == ownerK) {
            for (size_t i = k; i < nb; ++i) {
                BlockMat& b = getBlock(k, i);
                CUDA_CHECK(cudaMemcpy(panelHost.data() + offset[i], b.dptr,
                                       rows[i] * rk * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        MPI_Bcast(panelHost.data(), (int)totalElems, MPI_DOUBLE, ownerK, comm);

        std::vector<double*> panelDev(nb, nullptr);
        if (rank == ownerK) {
            for (size_t i = k; i < nb; ++i) {
                panelDev[i] = getBlock(k, i).dptr;
            }
        } else {
            CUDA_CHECK(cudaMemcpy(panelScratch, panelHost.data(), totalElems * sizeof(double),
                                   cudaMemcpyHostToDevice));
            for (size_t i = k; i < nb; ++i) {
                panelDev[i] = panelScratch + offset[i];
            }
        }

        // Trailing update: for every block-column j > k owned by this rank,
        // A(i,j) -= L(i,k) * L(j,k)^T  for i = j..nb-1 (SYRK on the diagonal,
        // GEMM off-diagonal). Independent columns are fanned out over OpenMP
        // threads, each driving its own CUDA stream.
        std::vector<size_t> myCols;
        for (size_t j = k + 1; j < nb; ++j) {
            if (owner(j) == rank) myCols.push_back(j);
        }

        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < myCols.size(); ++idx) {
            const size_t j = myCols[idx];
            ThreadCudaCtx& ctx = threadCudaCtx(deviceId);
            const size_t cj = blockLen(j, n, B);
            double* Ljk = panelDev[j];
            const double alpha = -1.0, beta = 1.0;
            for (size_t i = j; i < nb; ++i) {
                const size_t ri = blockLen(i, n, B);
                BlockMat& Cij = getBlock(j, i);
                double* Lik = panelDev[i];
                if (i == j) {
                    CUBLAS_CHECK(cublasDsyrk(ctx.handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                              (int)cj, (int)rk, &alpha, Ljk, (int)rk, &beta,
                                              Cij.dptr, (int)cj));
                } else {
                    CUBLAS_CHECK(cublasDgemm(ctx.handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)cj, (int)ri,
                                              (int)rk, &alpha, Ljk, (int)rk, Lik, (int)rk, &beta,
                                              Cij.dptr, (int)cj));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }
    }

    CUDA_CHECK(cudaFree(panelScratch));

    // --- Gather the result back to rank 0 (one message per source rank) ---
    if (!failed) {
        if (rank == 0) {
            for (int r = 0; r < nprocs; ++r) {
                auto [offsets, total] = packedOffsets(r);
                std::vector<double> buf(total);
                if (r == 0) {
                    size_t idx = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        if (owner(j) != 0) continue;
                        for (size_t i = j; i < nb; ++i) {
                            BlockMat& b = getBlock(j, i);
                            CUDA_CHECK(cudaMemcpy(buf.data() + offsets[idx], b.dptr,
                                                   b.rows * b.cols * sizeof(double), cudaMemcpyDeviceToHost));
                            ++idx;
                        }
                    }
                } else {
                    MPI_Recv(buf.data(), (int)total, MPI_DOUBLE, r, 0, comm, MPI_STATUS_IGNORE);
                }
                size_t idx = 0;
                for (size_t j = 0; j < nb; ++j) {
                    if (owner(j) != r) continue;
                    const size_t cj = blockLen(j, n, B);
                    for (size_t i = j; i < nb; ++i) {
                        const size_t ri = blockLen(i, n, B);
                        std::vector<double> blk(buf.data() + offsets[idx],
                                                 buf.data() + offsets[idx] + ri * cj);
                        insertBlock(A, n, blockStart(i, B), ri, blockStart(j, B), cj, blk);
                        ++idx;
                    }
                }
            }
        } else {
            auto [offsets, total] = packedOffsets(rank);
            std::vector<double> buf(total);
            size_t idx = 0;
            for (size_t j = 0; j < nb; ++j) {
                if (owner(j) != rank) continue;
                for (size_t i = j; i < nb; ++i) {
                    BlockMat& b = getBlock(j, i);
                    CUDA_CHECK(cudaMemcpy(buf.data() + offsets[idx], b.dptr,
                                           b.rows * b.cols * sizeof(double), cudaMemcpyDeviceToHost));
                    ++idx;
                }
            }
            MPI_Send(buf.data(), (int)total, MPI_DOUBLE, 0, 0, comm);
        }

        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A[i * n + j] = 0.0;
                }
            }
        }
    }

    for (size_t j = 0; j < nb; ++j) {
        for (auto& b : localCol[j]) {
            CUDA_CHECK(cudaFree(b.dptr));
        }
    }

    omp_set_num_threads(originalMaxThreads);
    return !failed;
}

}  // namespace hybrid

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (kept sequential to preserve the deterministic
    // random stream produced by rand_r)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (element order preserved per (i,j); only independent
    // outer iterations are parallelized, so results are unaffected)
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool exitEarly = false;
    int exitCode = 0;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exitEarly = true;
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = true;
            exitCode = 1;
            break;
        }
    }

    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI ranks=%d, OpenMP + CUDA per rank\n", nprocs);
    }

    // Allocate matrix (only rank 0 needs the full matrix)
    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        A.resize(n * n);

        // Generate positive definite matrix
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);

        if (validate) {
            A_orig = A; // Save original for validation
        }

        printf("Computing Cholesky decomposition...\n");
    }

    // Warm up each rank's CUDA context before timing so one-time driver
    // initialization is not counted as decomposition time.
    hybrid::warmupCuda(MPI_COMM_WORLD);

    // Perform Cholesky decomposition (hybrid MPI + OpenMP + CUDA)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = hybrid::choleskyDecompositionDistributed(A, n, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        MPI_Finalize();
        return 0;
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
