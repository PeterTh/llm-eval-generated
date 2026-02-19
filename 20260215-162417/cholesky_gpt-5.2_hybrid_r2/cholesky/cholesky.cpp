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

namespace {

constexpr int kBlockSize = 32; // block-row distributed, GPU-updated

static inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::fflush(stderr);
        std::abort();
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

__global__ void trsm_right_u_from_lT_inplace(double* __restrict__ dA, int n,
                                            int row_start, int local_rows,
                                            int k, int bk,
                                            const double* __restrict__ dLkk) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= local_rows) return;
    const int global_i = row_start + local_i;
    if (global_i < k + bk) return;

    double* row = dA + (size_t)local_i * n + k;

    // Solve x * (Lkk^T) = row, where Lkk is lower-triangular (bk x bk).
    // This is a right-side triangular solve with an upper-triangular matrix (Lkk^T),
    // which is a forward substitution in the column index.
    for (int c = 0; c < bk; ++c) {
        double sum = row[c];
        for (int t = 0; t < c; ++t) {
            // (Lkk^T)[t,c] = Lkk[c,t]
            sum -= row[t] * dLkk[c * bk + t];
        }
        row[c] = sum / dLkk[c * bk + c];
    }
}

__global__ void update_trailing(double* __restrict__ dA, int n,
                               int row_start, int local_rows,
                               int k, int bk,
                               const double* __restrict__ dPanel, int m_total) {
    // Tiled rank-k update for local rows:
    // A(i,j) -= dot(L(i,k:k+bk-1), L(j,k:k+bk-1)), where L(j,*) is provided via dPanel.
    constexpr int TX = 16;
    constexpr int TY = 8;

    __shared__ double shPanel[TX][kBlockSize];
    __shared__ double shLik[TY][kBlockSize];

    const int start_local_i = ((k + bk) > row_start) ? ((k + bk) - row_start) : 0;

    const int j = blockIdx.x * TX + threadIdx.x; // index in trailing columns
    const int local_i = start_local_i + blockIdx.y * TY + threadIdx.y;

    // Load panel vectors for the TX columns in this tile.
    if (threadIdx.y == 0 && j < m_total) {
        const double* p = dPanel + (size_t)j * bk;
        #pragma unroll
        for (int b = 0; b < kBlockSize; ++b) {
            shPanel[threadIdx.x][b] = (b < bk) ? p[b] : 0.0;
        }
    }

    // Load Lik vectors for the TY rows in this tile.
    if (local_i < local_rows) {
        const double* row = dA + (size_t)local_i * n + k;
        for (int b = threadIdx.x; b < bk; b += TX) {
            shLik[threadIdx.y][b] = row[b];
        }
    }

    __syncthreads();

    if (j >= m_total || local_i >= local_rows) return;

    const int global_j = (k + bk) + j;
    double* out = dA + (size_t)local_i * n + global_j;

    double sum = 0.0;
    #pragma unroll
    for (int b = 0; b < kBlockSize; ++b) {
        if (b < bk) sum += shLik[threadIdx.y][b] * shPanel[threadIdx.x][b];
    }
    *out -= sum;
}

static bool potrf_block_inplace(std::vector<double>& blk, int bk) {
    // Unblocked Cholesky on a small contiguous bk x bk block (row-major), producing lower-triangular.
    for (int i = 0; i < bk; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (int k = 0; k < j; ++k) sum += blk[j * bk + k] * blk[j * bk + k];
                const double val = blk[j * bk + j] - sum;
                if (val <= 0.0) return false;
                blk[j * bk + j] = std::sqrt(val);
            } else {
                for (int k = 0; k < j; ++k) sum += blk[i * bk + k] * blk[j * bk + k];
                blk[i * bk + j] = (blk[i * bk + j] - sum) / blk[j * bk + j];
            }
        }
        for (int j = i + 1; j < bk; ++j) blk[i * bk + j] = 0.0;
    }
    return true;
}

struct Dist {
    int size = 1;
    int rank = 0;
    int n = 0;
    int nb = 0;
    std::vector<int> row_start;
    std::vector<int> row_end;

    int owner_of_row(int r) const {
        for (int p = 0; p < size; ++p) {
            if (r >= row_start[p] && r < row_end[p]) return p;
        }
        return 0;
    }
};

static Dist make_blockrow_distribution(int n, int size, int rank) {
    Dist d;
    d.size = size;
    d.rank = rank;
    d.n = n;
    d.nb = (n + kBlockSize - 1) / kBlockSize;
    d.row_start.resize(size);
    d.row_end.resize(size);

    const int blocks_per_rank = d.nb / size;
    const int rem = d.nb % size;

    int cur_block = 0;
    for (int p = 0; p < size; ++p) {
        const int blocks_p = blocks_per_rank + (p < rem ? 1 : 0);
        const int rs = cur_block * kBlockSize;
        const int re = std::min(n, (cur_block + blocks_p) * kBlockSize);
        d.row_start[p] = rs;
        d.row_end[p] = re;
        cur_block += blocks_p;
    }
    return d;
}

static void generatePositiveDefiniteMatrixDistributed(std::vector<double>& localA,
                                                     std::vector<double>& localA_orig,
                                                     std::vector<double>& B,
                                                     const Dist& dist,
                                                     bool keep_orig) {
    const int n = dist.n;
    const int rank = dist.rank;
    const int rs = dist.row_start[rank];
    const int re = dist.row_end[rank];
    const int local_rows = re - rs;

    if (rank == 0) {
        B.resize((size_t)n * n);
        unsigned int seed = 42;
        for (int i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    } else {
        B.resize((size_t)n * n);
    }

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    localA.assign((size_t)local_rows * n, 0.0);

    // Compute local rows of A = B * B^T (row-major) and add diagonal dominance.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int li = 0; li < local_rows; ++li) {
        for (int j = 0; j < n; ++j) {
            const int gi = rs + li;
            const double* Bi = &B[(size_t)gi * n];
            const double* Bj = &B[(size_t)j * n];
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += Bi[k] * Bj[k];
            localA[(size_t)li * n + j] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int li = 0; li < local_rows; ++li) {
        const int gi = rs + li;
        localA[(size_t)li * n + gi] += n;
    }

    if (keep_orig) localA_orig = localA;
}

static bool validateCholeskyRoot(const std::vector<double>& L, const std::vector<double>& A_orig, int n) {
    // Validate by computing L * L^T and comparing with original matrix.
    std::vector<double> reconstructed((size_t)n * n, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            reconstructed[(size_t)i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
    for (int i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (std::fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n_sz = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_sz = (size_t)std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n = (int)n_sz;

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA)\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("MPI ranks: %d\n", size);
        std::printf("OpenMP threads (max): %d\n", omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int devCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&devCount);
    if (devErr != cudaSuccess || devCount <= 0) {
        if (rank == 0) std::fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    const Dist dist = make_blockrow_distribution(n, size, rank);
    const int rs = dist.row_start[rank];
    const int re = dist.row_end[rank];
    const int local_rows = re - rs;

    std::vector<double> B;
    std::vector<double> localA;
    std::vector<double> localA_orig;

    if (rank == 0) std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrixDistributed(localA, localA_orig, B, dist, validate);

    // Device storage for local rows (some ranks may own no rows if MPI size > #block-rows).
    double* dA = nullptr;
    if (local_rows > 0) {
        CUDA_CHECK(cudaMalloc((void**)&dA, (size_t)local_rows * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA, localA.data(), (size_t)local_rows * n * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Device buffer for diagonal block and the replicated panel.
    double* dLkk = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dLkk, (size_t)kBlockSize * kBlockSize * sizeof(double)));

    double* dPanel = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dPanel, (size_t)n * kBlockSize * sizeof(double)));

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int ok_local = 1;

    // Host buffers reused per-iteration.
    std::vector<double> hDiag((size_t)kBlockSize * kBlockSize);
    std::vector<double> hPanelFull;
    std::vector<double> hPanelSend;

    for (int k = 0; k < n; k += kBlockSize) {
        const int bk = std::min(kBlockSize, n - k);
        const int owner = dist.owner_of_row(k);

        // 1) Factor diagonal block on owner (host) then broadcast Lkk.
        if (rank == owner) {
            const int local_k = k - rs;
            // Copy diagonal block from device to host contiguous buffer.
            CUDA_CHECK(cudaMemcpy2D(hDiag.data(), (size_t)bk * sizeof(double),
                                    dA + (size_t)local_k * n + k, (size_t)n * sizeof(double),
                                    (size_t)bk * sizeof(double), (size_t)bk,
                                    cudaMemcpyDeviceToHost));
            ok_local = potrf_block_inplace(hDiag, bk) ? 1 : 0;
            if (ok_local) {
                // Copy factorized block back to device.
                CUDA_CHECK(cudaMemcpy2D(dA + (size_t)local_k * n + k, (size_t)n * sizeof(double),
                                        hDiag.data(), (size_t)bk * sizeof(double),
                                        (size_t)bk * sizeof(double), (size_t)bk,
                                        cudaMemcpyHostToDevice));
            }
        }

        int ok_all = 0;
        MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok_all) {
            if (rank == 0) std::fprintf(stderr, "Error: Matrix is not positive definite (block starting at %d)\n", k);
            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dLkk));
            CUDA_CHECK(cudaFree(dPanel));
            MPI_Finalize();
            return 1;
        }

        // Broadcast Lkk (packed) from owner.
        if (rank != owner) {
            // ensure buffer size
            hDiag.assign((size_t)bk * bk, 0.0);
        }
        if (rank == owner) {
            // hDiag already bk*bk
        }
        MPI_Bcast(hDiag.data(), bk * bk, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dLkk, hDiag.data(), (size_t)bk * bk * sizeof(double), cudaMemcpyHostToDevice));

        // 2) Compute local panel below diagonal: A(i,k:k+bk-1) = A(i,k:k+bk-1) * inv(Lkk^T)
        {
            if (local_rows > 0) {
                const int threads = 256;
                const int blocks = (local_rows + threads - 1) / threads;
                if (blocks > 0) {
                    trsm_right_u_from_lT_inplace<<<blocks, threads>>>(dA, n, rs, local_rows, k, bk, dLkk);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        }

        // 3) Replicate the panel (rows k+bk..n-1) across ranks for the trailing update.
        const int m_total = n - (k + bk);
        if (m_total <= 0) break;

        const int send_i0 = std::max(k + bk, rs);
        const int send_i1 = re;
        const int send_rows = std::max(0, send_i1 - send_i0);
        const int send_local_off = send_i0 - rs;

        hPanelSend.assign((size_t)send_rows * bk, 0.0);
        if (send_rows > 0) {
            CUDA_CHECK(cudaMemcpy2D(hPanelSend.data(), (size_t)bk * sizeof(double),
                                    dA + (size_t)send_local_off * n + k, (size_t)n * sizeof(double),
                                    (size_t)bk * sizeof(double), (size_t)send_rows,
                                    cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(size), displs(size);
        for (int p = 0; p < size; ++p) {
            const int prs = dist.row_start[p];
            const int pre = dist.row_end[p];
            const int pi0 = std::max(k + bk, prs);
            const int pi1 = pre;
            const int prow = std::max(0, pi1 - pi0);
            counts[p] = prow * bk;
            displs[p] = (pi0 - (k + bk)) * bk;
        }

        hPanelFull.assign((size_t)m_total * bk, 0.0);
        MPI_Allgatherv(hPanelSend.data(), send_rows * bk, MPI_DOUBLE,
                       hPanelFull.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(dPanel, hPanelFull.data(), (size_t)m_total * bk * sizeof(double), cudaMemcpyHostToDevice));

        // 4) Trailing update on GPU for local rows: A(:, k+bk:) -= Lik_local * Panel^T
        {
            const int start_local_i = std::max(0, (k + bk) - rs);
            const int update_rows = std::max(0, local_rows - start_local_i);
            if (update_rows > 0 && m_total > 0) {
                dim3 block(16, 8);
                dim3 grid((m_total + block.x - 1) / block.x,
                          (update_rows + block.y - 1) / block.y);
                update_trailing<<<grid, block>>>(dA, n, rs, local_rows, k, bk, dPanel, m_total);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long duration_ms_max = 0;
    MPI_Reduce(&duration_ms, &duration_ms_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy result back to host and gather to rank 0.
    std::vector<double> localL((size_t)local_rows * n);
    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(localL.data(), dA, (size_t)local_rows * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::vector<int> recvCounts(size), recvDispls(size);
    for (int p = 0; p < size; ++p) {
        const int rows_p = dist.row_end[p] - dist.row_start[p];
        recvCounts[p] = rows_p * n;
        recvDispls[p] = dist.row_start[p] * n;
    }

    std::vector<double> L;
    std::vector<double> A_orig;
    if (rank == 0) {
        L.assign((size_t)n * n, 0.0);
        if (validate) A_orig.assign((size_t)n * n, 0.0);
    }

    MPI_Gatherv(localL.data(), local_rows * n, MPI_DOUBLE,
                rank == 0 ? L.data() : nullptr, recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (validate) {
        MPI_Gatherv(localA_orig.data(), local_rows * n, MPI_DOUBLE,
                    rank == 0 ? A_orig.data() : nullptr, recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (dA) CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dLkk));
    CUDA_CHECK(cudaFree(dPanel));

    if (rank == 0) {
        // Match original semantics: upper-triangular part is zero.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                L[(size_t)i * n + j] = 0.0;
            }
        }

        std::printf("Computation time: %ld ms\n", duration_ms_max);
        const double ms = (duration_ms_max > 0) ? (double)duration_ms_max : 1.0;
        double ops = (double)n * (double)n * (double)n / 3.0;
        double gflops = ops / (ms / 1000.0) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholeskyRoot(L, A_orig, n);
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
