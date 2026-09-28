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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization strategy (right-looking blocked algorithm):
//  - Block columns are distributed cyclically across MPI ranks (one GPU per rank).
//  - The panel (diagonal block + sub-diagonal column) is factored on the host
//    using OpenMP, then broadcast to all ranks.
//  - The O(n^3) trailing-submatrix update (SYRK-like rank-B update) runs on the
//    GPU with a shared-memory tiled CUDA kernel; each rank updates only the
//    block columns it owns.
//  - At the end, the distributed lower triangle is reduced back to rank 0.

static constexpr size_t BLOCK = 256; // panel width (multiple of TILE)
static constexpr int TILE = 64;      // CUDA output tile size (per thread block)
static constexpr int KS = 16;        // CUDA k-chunk size
static constexpr int TDIM = 16;      // thread block is TDIM x TDIM, 4x4 outputs per thread

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),  \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// Trailing submatrix update: A[i][j] -= dot(P[i - r0 + pw], P[j - r0 + pw])
// P is the packed factored panel restricted to the trailing rows (m x pw, ld = pw).
// Only lower-triangle tiles of block columns owned by this rank are updated.
__global__ void trailingUpdateKernel(double* __restrict__ A, size_t n,
                                     const double* __restrict__ P, size_t pw,
                                     size_t r0, size_t m, size_t bs,
                                     int rank, int nprocs) {
    // Tiles strictly above the diagonal contribute nothing
    if (blockIdx.x > blockIdx.y) return;

    const size_t j0 = (size_t)blockIdx.x * TILE;
    const size_t i0 = (size_t)blockIdx.y * TILE;

    // Block-cyclic column ownership (tiles never straddle block-column boundaries
    // since r0 is aligned to BLOCK and BLOCK is a multiple of TILE)
    if ((((r0 + j0) / bs) % (size_t)nprocs) != (size_t)rank) return;

    __shared__ double Pi[TILE][KS + 1];
    __shared__ double Pj[TILE][KS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    double acc[4][4] = {};
    for (size_t kk = 0; kk < pw; kk += KS) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = ty + TDIM * l;
            const size_t c = kk + tx;
            const size_t ri = i0 + row;
            const size_t rj = j0 + row;
            Pi[row][tx] = (ri < m && c < pw) ? P[ri * pw + c] : 0.0;
            Pj[row][tx] = (rj < m && c < pw) ? P[rj * pw + c] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < KS; ++t) {
            double a[4], b[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                a[l] = Pi[ty + TDIM * l][t];
                b[l] = Pj[tx + TDIM * l][t];
            }
#pragma unroll
            for (int ri = 0; ri < 4; ++ri) {
#pragma unroll
                for (int ci = 0; ci < 4; ++ci) {
                    acc[ri][ci] += a[ri] * b[ci];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
        const size_t i = i0 + ty + TDIM * ri;
        if (i >= m) continue;
#pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const size_t j = j0 + tx + TDIM * ci;
            if (j <= i) {
                A[(r0 + i) * n + (r0 + j)] -= acc[ri][ci];
            }
        }
    }
}

// Copy a block column (rows pk..pk+rows, cols pk..pk+pw) between the full
// matrix and a packed buffer (ld = pw), entirely on the GPU so host transfers
// stay contiguous.
__global__ void packBlockColumnKernel(double* __restrict__ A, double* __restrict__ P,
                                      size_t n, size_t pk, size_t pw, size_t rows,
                                      bool toPacked) {
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || c >= pw) return;
    if (toPacked) {
        P[i * pw + c] = A[(pk + i) * n + pk + c];
    } else {
        A[(pk + i) * n + pk + c] = P[i * pw + c];
    }
}

// Move all of this rank's owned block columns between dA and dPacked
static void packOwnedColumns(double* dA, double* dPacked, const size_t n,
                             const int rank, const int nprocs, const bool toPacked) {
    size_t off = 0;
    for (size_t jb = (size_t)rank; jb * BLOCK < n; jb += (size_t)nprocs) {
        const size_t pk = jb * BLOCK;
        const size_t pw = std::min(BLOCK, n - pk);
        const size_t rows = n - pk;
        dim3 block(32, 8);
        dim3 grid((unsigned)((pw + 31) / 32), (unsigned)((rows + 7) / 8));
        packBlockColumnKernel<<<grid, block>>>(dA, dPacked + off, n, pk, pw, rows, toPacked);
        off += rows * pw;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// MPI point-to-point transfers chunked to stay within the int count limit
static void sendDoubles(const double* p, size_t count, int dst, int tag) {
    do {
        const int c = (int)std::min(count, (size_t)1 << 28);
        MPI_Send(p, c, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
        p += c;
        count -= (size_t)c;
    } while (count > 0);
}

static void recvDoubles(double* p, size_t count, int src, int tag) {
    do {
        const int c = (int)std::min(count, (size_t)1 << 28);
        MPI_Recv(p, c, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p += c;
        count -= (size_t)c;
    } while (count > 0);
}

// Unblocked factorization of a panel (mrows x pw, ld = pw, packed, top-left is
// the diagonal block at global index pk). OpenMP-parallel over the rows.
static bool factorPanelHost(double* P, const size_t mrows, const size_t pw, const size_t pk) {
    bool ok = true;
    // Single parallel region for the whole panel: per-column barriers are much
    // cheaper than per-column fork/join. The panel is a small fraction of the
    // total work, so a moderate thread count avoids barrier overhead.
    const int nt = std::min(omp_get_max_threads(), 64);
#pragma omp parallel num_threads(nt) shared(ok)
    {
        for (size_t j = 0; j < pw; ++j) {
#pragma omp single
            {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += P[j * pw + k] * P[j * pw + k];
                }
                const double val = P[j * pw + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           pk + j);
                    ok = false;
                } else {
                    P[j * pw + j] = sqrt(val);
                }
            } // implicit barrier: all threads observe ok and the new diagonal
            if (!ok) break;
            const double djj = P[j * pw + j];
#pragma omp for schedule(static)
            for (size_t i = j + 1; i < mrows; ++i) {
                double s = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    s += P[i * pw + k] * P[j * pw + k];
                }
                P[i * pw + j] = (P[i * pw + j] - s) / djj;
            }
        }
    }
    return ok;
}

// Number of doubles needed to hold rank r's owned block columns
// (full block column from the diagonal block downward, packed with ld = pw)
static size_t ownedPackedSize(const size_t n, const int r, const int nprocs) {
    size_t total = 0;
    for (size_t jb = (size_t)r; jb * BLOCK < n; jb += (size_t)nprocs) {
        const size_t pk = jb * BLOCK;
        const size_t pw = std::min(BLOCK, n - pk);
        total += (n - pk) * pw;
    }
    return total;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, const int rank, const int nprocs) {
    const bool prof = getenv("CHOL_PROF") != nullptr;
    double tBcast = 0, tAlloc = 0, tPanelCopy = 0, tPanelFac = 0, tPanelBc = 0, tKern = 0, tGather = 0;
    double t0 = MPI_Wtime();

    t0 = MPI_Wtime();
    const size_t bytes = n * n * sizeof(double);
    const size_t packedSize = ownedPackedSize(n, rank, nprocs);
    double* dA = nullptr;
    double* dPanel = nullptr;
    double* dPacked = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dPanel, n * BLOCK * sizeof(double)));
    if (nprocs > 1) {
        CUDA_CHECK(cudaMalloc(&dPacked, std::max(packedSize, (size_t)1) * sizeof(double)));
    }
    tAlloc = MPI_Wtime() - t0;

    // Distribute the input: each rank only needs the sub-diagonal part of the
    // block columns it owns (the panels of other ranks arrive via broadcast
    // during the factorization). Rank 0 packs and sends each rank's share.
    t0 = MPI_Wtime();
    std::vector<double> packed;
    if (nprocs == 1) {
        // Single rank owns everything: upload the matrix as-is
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        packed.resize(packedSize);
        if (rank == 0) {
            std::vector<double> sendBuf;
            for (int r = 0; r < nprocs; ++r) {
                std::vector<double>& buf = (r == 0) ? packed : sendBuf;
                if (r != 0) buf.resize(ownedPackedSize(n, r, nprocs));
                size_t off = 0;
                for (size_t jb = (size_t)r; jb * BLOCK < n; jb += (size_t)nprocs) {
                    const size_t pk = jb * BLOCK;
                    const size_t pw = std::min(BLOCK, n - pk);
#pragma omp parallel for schedule(static)
                    for (size_t i = pk; i < n; ++i) {
                        memcpy(&buf[off + (i - pk) * pw], &A[i * n + pk], pw * sizeof(double));
                    }
                    off += (n - pk) * pw;
                }
                if (r != 0 && !buf.empty()) {
                    sendDoubles(buf.data(), buf.size(), r, 0);
                }
            }
        } else if (packedSize > 0) {
            recvDoubles(packed.data(), packedSize, 0, 0);
        }
        // Upload the owned block columns and scatter them into place on the GPU
        if (packedSize > 0) {
            CUDA_CHECK(cudaMemcpy(dPacked, packed.data(), packedSize * sizeof(double),
                                  cudaMemcpyHostToDevice));
            packOwnedColumns(dA, dPacked, n, rank, nprocs, false);
        }
    }
    tBcast = MPI_Wtime() - t0;

    std::vector<double> hPanel(n * BLOCK);

    const size_t nBlocks = (n + BLOCK - 1) / BLOCK;
    bool ok = true;

    for (size_t kb = 0; kb < nBlocks; ++kb) {
        const size_t pk = kb * BLOCK;              // panel start
        const size_t pw = std::min(BLOCK, n - pk); // panel width
        const size_t mrows = n - pk;               // panel height
        const int owner = (int)(kb % (size_t)nprocs);

        int okFlag = 1;
        t0 = MPI_Wtime();
        if (rank == owner) {
            // Fetch the (already fully updated) panel from the GPU and factor it
            CUDA_CHECK(cudaMemcpy2D(hPanel.data(), pw * sizeof(double),
                                    dA + pk * n + pk, n * sizeof(double),
                                    pw * sizeof(double), mrows, cudaMemcpyDeviceToHost));
            double t1 = MPI_Wtime();
            tPanelCopy += t1 - t0;
            okFlag = factorPanelHost(hPanel.data(), mrows, pw, pk) ? 1 : 0;
            tPanelFac += MPI_Wtime() - t1;
        }
        t0 = MPI_Wtime();
        MPI_Bcast(&okFlag, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!okFlag) {
            ok = false;
            break;
        }
        MPI_Bcast(hPanel.data(), (int)(mrows * pw), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        tPanelBc += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        if (rank == owner) {
            // Write the factored panel back into the owner's matrix
            CUDA_CHECK(cudaMemcpy2D(dA + pk * n + pk, n * sizeof(double),
                                    hPanel.data(), pw * sizeof(double),
                                    pw * sizeof(double), mrows, cudaMemcpyHostToDevice));
        }

        const size_t r0 = pk + pw;   // trailing submatrix start
        const size_t m = n - r0;     // trailing submatrix size
        if (m > 0) {
            CUDA_CHECK(cudaMemcpy(dPanel, hPanel.data(), mrows * pw * sizeof(double),
                                  cudaMemcpyHostToDevice));
            const size_t g = (m + TILE - 1) / TILE;
            dim3 grid((unsigned)g, (unsigned)g);
            dim3 block(TDIM, TDIM);
            trailingUpdateKernel<<<grid, block>>>(dA, n, dPanel + pw * pw, pw,
                                                  r0, m, BLOCK, rank, nprocs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        tKern += MPI_Wtime() - t0;
    }

    t0 = MPI_Wtime();
    if (ok) {
        // Collect the distributed result on rank 0: each rank downloads and sends
        // only the block columns it owns; rank 0 unpacks them into A, zeroing the
        // upper triangle in the process.
        if (nprocs == 1) {
            // Single rank: download the matrix and zero the upper triangle
            CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                if (i + 1 < n) {
                    memset(&A[i * n + i + 1], 0, (n - i - 1) * sizeof(double));
                }
            }
        } else {
        if (packedSize > 0) {
            packOwnedColumns(dA, dPacked, n, rank, nprocs, true);
            CUDA_CHECK(cudaMemcpy(packed.data(), dPacked, packedSize * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        if (rank != 0) {
            if (packedSize > 0) {
                sendDoubles(packed.data(), packedSize, 0, 1);
            }
        } else {
            std::vector<double> recvBuf;
            for (int r = 0; r < nprocs; ++r) {
                const double* buf;
                if (r == 0) {
                    buf = packed.data();
                } else {
                    recvBuf.resize(ownedPackedSize(n, r, nprocs));
                    if (!recvBuf.empty()) {
                        recvDoubles(recvBuf.data(), recvBuf.size(), r, 1);
                    }
                    buf = recvBuf.data();
                }
                size_t off = 0;
                for (size_t jb = (size_t)r; jb * BLOCK < n; jb += (size_t)nprocs) {
                    const size_t pk = jb * BLOCK;
                    const size_t pw = std::min(BLOCK, n - pk);
#pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        double* dst = &A[i * n + pk];
                        if (i < pk) {
                            // Entirely above the diagonal
                            memset(dst, 0, pw * sizeof(double));
                        } else {
                            const double* src = &buf[off + (i - pk) * pw];
                            const size_t nkeep = std::min(pw, i - pk + 1);
                            memcpy(dst, src, nkeep * sizeof(double));
                            if (nkeep < pw) {
                                memset(dst + nkeep, 0, (pw - nkeep) * sizeof(double));
                            }
                        }
                    }
                    off += (n - pk) * pw;
                }
            }
        }
        }
    }

    tGather = MPI_Wtime() - t0;
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFree(dPacked));
    if (prof) {
        printf("[rank %d] bcast=%.3f alloc+h2d=%.3f pcopy=%.3f pfac=%.3f pbcast=%.3f kern=%.3f gather=%.3f\n",
               rank, tBcast, tAlloc, tPanelCopy, tPanelFac, tPanelBc, tKern, tGather);
        fflush(stdout);
    }

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

    // Compute A = B * B^T (symmetric: compute the lower triangle and mirror)
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
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

    // Compute L * L^T (symmetric: compute the lower triangle and mirror)
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
            reconstructed[j * n + i] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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

    // Assign one GPU per rank (round-robin among the GPUs visible on this node)
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr)); // establish the CUDA context up front

    // Split the node's cores among the node-local ranks to avoid oversubscription
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / std::max(1, localSize)));
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
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs on node: %d\n",
               nprocs, omp_get_max_threads(), deviceCount);
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix on rank 0 (distributed inside the solver)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);

        if (validate) {
            A_orig = A; // Save original for validation
        }
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, rank, nprocs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
            }
            int rc = valid ? 0 : 1;
            MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return rc;
        }
        int rc = 0;
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return rc;
    } else {
        int rc = 0;
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return rc;
    }
}
