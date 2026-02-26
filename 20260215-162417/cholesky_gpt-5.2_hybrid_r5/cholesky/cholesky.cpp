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

constexpr int DEFAULT_BLOCK_SIZE = 128;

inline void die_rank0(int rank, const char* msg) {
    if (rank == 0) {
        fprintf(stderr, "%s\n", msg);
    }
}

inline void checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

__global__ void trsm_right_upper_from_lower(double* __restrict__ A, int lda,
                                           const double* __restrict__ L, int ldl,
                                           int m, int n) {
    // Solve row-wise: X * (L^T) = B  => X = B * inv(L^T)
    // L is lower-triangular (n x n), row-major. A is (m x n), row-major.
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;

    double* row = A + (size_t)r * (size_t)lda;
    for (int j = 0; j < n; ++j) {
        double v = row[j];
        #pragma unroll
        for (int t = 0; t < 0; ++t) {
            // no-op for compilation; real loop below
        }
        for (int t = 0; t < j; ++t) {
            v -= row[t] * L[(size_t)j * (size_t)ldl + (size_t)t];
        }
        v /= L[(size_t)j * (size_t)ldl + (size_t)j];
        row[j] = v;
    }
}

template <int TILE>
__global__ void gemm_update_c_rowmaj(double* __restrict__ C, int ldc,
                                    const double* __restrict__ A, int lda,
                                    const double* __restrict__ B, int ldb,
                                    int m, int n, int k) {
    // C(m x n) -= A(m x k) * B(n x k)^T, all row-major.
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    double acc = 0.0;
    for (int t0 = 0; t0 < k; t0 += TILE) {
        const int t = t0 + threadIdx.x;
        if (row < m && t < k) {
            As[threadIdx.y][threadIdx.x] = A[(size_t)row * (size_t)lda + (size_t)t];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        const int bt = t0 + threadIdx.y;
        if (col < n && bt < k) {
            Bs[threadIdx.y][threadIdx.x] = B[(size_t)col * (size_t)ldb + (size_t)bt];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int tt = 0; tt < TILE; ++tt) {
            acc += As[threadIdx.y][tt] * Bs[tt][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[(size_t)row * (size_t)ldc + (size_t)col] -= acc;
    }
}

bool chol_unblocked_inplace(double* A, int n) {
    // In-place Cholesky on lower triangle (row-major), like original semantics.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (int k = 0; k < j; ++k) {
                    const double v = A[(size_t)j * (size_t)n + (size_t)k];
                    sum += v * v;
                }
                const double val = A[(size_t)j * (size_t)n + (size_t)j] - sum;
                if (val <= 0.0) return false;
                A[(size_t)j * (size_t)n + (size_t)j] = std::sqrt(val);
            } else {
                for (int k = 0; k < j; ++k) {
                    sum += A[(size_t)i * (size_t)n + (size_t)k] * A[(size_t)j * (size_t)n + (size_t)k];
                }
                A[(size_t)i * (size_t)n + (size_t)j] =
                    (A[(size_t)i * (size_t)n + (size_t)j] - sum) / A[(size_t)j * (size_t)n + (size_t)j];
            }
        }
        for (int j = i + 1; j < n; ++j) {
            A[(size_t)i * (size_t)n + (size_t)j] = 0.0;
        }
    }
    return true;
}

struct LocalBlocks {
    int nb = 0;
    int blockSize = DEFAULT_BLOCK_SIZE;
    std::vector<int> brs;              // global block-row indices owned by this rank (increasing)
    std::vector<int> rows;             // rows per owned block row
    std::vector<size_t> offsets;       // starting local row (in rows) per owned block row
    std::vector<int> br_to_local;      // size nb, -1 or local index
    size_t totalRows = 0;
};

LocalBlocks make_local_blocks(int n, int blockSize, int rank, int nranks) {
    LocalBlocks lb;
    lb.blockSize = blockSize;
    lb.nb = (n + blockSize - 1) / blockSize;
    lb.br_to_local.assign(lb.nb, -1);

    size_t off = 0;
    for (int br = 0; br < lb.nb; ++br) {
        if ((br % nranks) != rank) continue;
        const int r = std::min(blockSize, n - br * blockSize);
        lb.br_to_local[br] = (int)lb.brs.size();
        lb.brs.push_back(br);
        lb.rows.push_back(r);
        lb.offsets.push_back(off);
        off += (size_t)r;
    }
    lb.totalRows = off;
    return lb;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Match original semantics: A = B * B^T + nI, with deterministic seed.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (std::fabs(A_orig[i]) + 1e-10);
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool distributed_cholesky_hybrid(double* localA_d,
                                std::vector<double>& localA_h,
                                const LocalBlocks& lb,
                                int n,
                                int rank,
                                int nranks,
                                MPI_Comm comm) {
    const int nb = lb.nb;
    const int BS = lb.blockSize;

    // Diagonal block host/device buffers (max size BSxBS)
    std::vector<double> diag_h((size_t)BS * (size_t)BS);
    double* diag_d = nullptr;
    checkCuda(cudaMalloc(&diag_d, (size_t)BS * (size_t)BS * sizeof(double)), "cudaMalloc diag_d");

    // Reusable device buffer for the gathered panel (upper bound: (n-BS)*BS doubles)
    double* panel_d = nullptr;
    const size_t panel_cap = (size_t)n * (size_t)BS;
    checkCuda(cudaMalloc(&panel_d, panel_cap * sizeof(double)), "cudaMalloc panel_d");

    // Panel buffers
    std::vector<int> recvCounts(nranks, 0), displs(nranks, 0);

    // Precompute block-row sizes
    std::vector<int> br_rows(nb);
    for (int br = 0; br < nb; ++br) {
        br_rows[br] = std::min(BS, n - br * BS);
    }

    for (int kbr = 0; kbr < nb; ++kbr) {
        const int k0 = kbr * BS;
        const int kb = br_rows[kbr];
        const int owner = kbr % nranks;

        // Factor diagonal block on owner rank (host), but store/follow with GPU compute.
        int ok_local = 1;
        if (rank == owner) {
            const int li = lb.br_to_local[kbr];
            if (li < 0) {
                checkCuda(cudaFree(panel_d), "cudaFree panel_d");
                checkCuda(cudaFree(diag_d), "cudaFree diag_d");
                return false;
            }
            const size_t lrow0 = lb.offsets[li];
            const double* src = localA_d + lrow0 * (size_t)n + (size_t)k0;

            checkCuda(cudaMemcpy2D(diag_h.data(), (size_t)kb * sizeof(double),
                                   src, (size_t)n * sizeof(double),
                                   (size_t)kb * sizeof(double), (size_t)kb,
                                   cudaMemcpyDeviceToHost),
                      "D2H diag");

            const bool diag_ok = chol_unblocked_inplace(diag_h.data(), kb);
            ok_local = diag_ok ? 1 : 0;

            // Copy factored diag back to device (keeps local matrix consistent)
            double* dst = localA_d + lrow0 * (size_t)n + (size_t)k0;
            checkCuda(cudaMemcpy2D(dst, (size_t)n * sizeof(double),
                                   diag_h.data(), (size_t)kb * sizeof(double),
                                   (size_t)kb * sizeof(double), (size_t)kb,
                                   cudaMemcpyHostToDevice),
                      "H2D diag");
        }

        int ok_all = 0;
        MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, comm);
        if (!ok_all) {
            checkCuda(cudaFree(panel_d), "cudaFree panel_d");
            checkCuda(cudaFree(diag_d), "cudaFree diag_d");
            return false;
        }

        // No trailing work after factoring the final diagonal block.
        if (kbr == nb - 1) {
            break;
        }

        // Broadcast diagonal block to all ranks
        MPI_Bcast(diag_h.data(), kb * kb, MPI_DOUBLE, owner, comm);

        // All ranks need diag on device for TRSM
        checkCuda(cudaMemcpy(diag_d, diag_h.data(), (size_t)kb * (size_t)kb * sizeof(double), cudaMemcpyHostToDevice),
                  "H2D diag_d");

        // TRSM: compute L_ik in-place for all local block rows i > kbr
        for (size_t idx = 0; idx < lb.brs.size(); ++idx) {
            const int ibr = lb.brs[idx];
            if (ibr <= kbr) continue;
            const int mi = lb.rows[idx];
            const size_t lrow0 = lb.offsets[idx];
            double* Aik = localA_d + lrow0 * (size_t)n + (size_t)k0;

            const int threads = 128;
            const int blocks = (mi + threads - 1) / threads;
            trsm_right_upper_from_lower<<<blocks, threads>>>(Aik, n, diag_d, kb, mi, kb);
            checkCuda(cudaGetLastError(), "trsm kernel");
        }

        // Pack this rank's (i>k) blocks of L(:,k) into send buffer (row-major, compact per block)
        size_t sendCount = 0;
        for (size_t idx = 0; idx < lb.brs.size(); ++idx) {
            if (lb.brs[idx] <= kbr) continue;
            sendCount += (size_t)lb.rows[idx] * (size_t)kb;
        }

        std::vector<double> send_h(sendCount);
        size_t pos = 0;
        for (size_t idx = 0; idx < lb.brs.size(); ++idx) {
            if (lb.brs[idx] <= kbr) continue;
            const int mi = lb.rows[idx];
            const size_t lrow0 = lb.offsets[idx];
            const double* src = localA_d + lrow0 * (size_t)n + (size_t)k0;
            double* dst = send_h.data() + pos;
            checkCuda(cudaMemcpy2D(dst, (size_t)kb * sizeof(double),
                                   src, (size_t)n * sizeof(double),
                                   (size_t)kb * sizeof(double), (size_t)mi,
                                   cudaMemcpyDeviceToHost),
                      "D2H panel block");
            pos += (size_t)mi * (size_t)kb;
        }

        int sendCount_i = (int)sendCount;
        MPI_Allgather(&sendCount_i, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);
        int totalCount_i = 0;
        for (int r = 0; r < nranks; ++r) {
            displs[r] = totalCount_i;
            totalCount_i += recvCounts[r];
        }

        std::vector<double> panel_h((size_t)totalCount_i);
        MPI_Allgatherv(send_h.data(), sendCount_i, MPI_DOUBLE,
                       panel_h.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                       comm);

        // Build within-owner offsets for global block-row -> offset in that owner's contribution.
        std::vector<size_t> withinOwner(nb, 0);
        std::vector<size_t> pref(nranks, 0);
        for (int br = kbr + 1; br < nb; ++br) {
            const int r = br % nranks;
            withinOwner[br] = pref[r];
            pref[r] += (size_t)br_rows[br] * (size_t)kb;
        }

        // Copy panel to device (reusing a preallocated buffer)
        if ((size_t)totalCount_i > panel_cap) {
            checkCuda(cudaFree(panel_d), "cudaFree panel_d");
            checkCuda(cudaFree(diag_d), "cudaFree diag_d");
            return false;
        }
        checkCuda(cudaMemcpy(panel_d, panel_h.data(), (size_t)totalCount_i * sizeof(double), cudaMemcpyHostToDevice), "H2D panel_d");

        // Trailing update: for each local block-row i>k, update blocks (j=k+1..i)
        constexpr int TILE = 16;
        for (size_t iidx = 0; iidx < lb.brs.size(); ++iidx) {
            const int ibr = lb.brs[iidx];
            if (ibr <= kbr) continue;
            const int mi = lb.rows[iidx];
            const size_t lrow0 = lb.offsets[iidx];

            const double* Aik = localA_d + lrow0 * (size_t)n + (size_t)k0;

            for (int jbr = kbr + 1; jbr <= ibr; ++jbr) {
                const int nj = br_rows[jbr];
                const int j0 = jbr * BS;

                const int ownerJ = jbr % nranks;
                const size_t Boff = (size_t)displs[ownerJ] + withinOwner[jbr];
                const double* Bjk = panel_d + Boff;

                double* Cij = localA_d + lrow0 * (size_t)n + (size_t)j0;

                dim3 block(TILE, TILE);
                dim3 grid((nj + TILE - 1) / TILE, (mi + TILE - 1) / TILE);
                gemm_update_c_rowmaj<TILE><<<grid, block>>>(Cij, n, Aik, n, Bjk, kb, mi, nj, kb);
                checkCuda(cudaGetLastError(), "gemm update kernel");
            }
        }
        // No explicit sync required here; the next iteration's device-to-host copies will synchronize as needed.
    }

    checkCuda(cudaFree(panel_d), "cudaFree panel_d");
    checkCuda(cudaFree(diag_d), "cudaFree diag_d");
    return true;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast args
    unsigned long long n_ull = (unsigned long long)n;
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = (size_t)n_ull;
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    // Unconditionally use CUDA: pick a device per rank.
    int devCount = 0;
    cudaError_t ce = cudaGetDeviceCount(&devCount);
    if (ce != cudaSuccess || devCount <= 0) {
        die_rank0(rank, "Error: no CUDA devices available");
        MPI_Finalize();
        return 1;
    }
    checkCuda(cudaSetDevice(rank % devCount), "cudaSetDevice");

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int blockSize = DEFAULT_BLOCK_SIZE;
    LocalBlocks lb = make_local_blocks((int)n, blockSize, rank, nranks);
    std::vector<double> localA_h(lb.totalRows * n);

    std::vector<double> A_full;
    std::vector<double> A_orig;

    if (rank == 0) {
        A_full.resize(n * n);
        if (validate) A_orig.resize(n * n);

        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) A_orig = A_full;

        // Distribute block rows cyclically (point-to-point, no extra dependencies).
        const int nb = lb.nb;
        for (int br = 0; br < nb; ++br) {
            const int rows = std::min(blockSize, (int)n - br * blockSize);
            const int dest = br % nranks;
            const double* src = A_full.data() + (size_t)br * (size_t)blockSize * n;
            if (dest == 0) {
                const int li = lb.br_to_local[br];
                std::memcpy(localA_h.data() + lb.offsets[li] * n, src, (size_t)rows * n * sizeof(double));
            } else {
                MPI_Send(src, rows * (int)n, MPI_DOUBLE, dest, br, MPI_COMM_WORLD);
            }
        }
    } else {
        for (size_t idx = 0; idx < lb.brs.size(); ++idx) {
            const int br = lb.brs[idx];
            const int rows = lb.rows[idx];
            MPI_Recv(localA_h.data() + lb.offsets[idx] * n, rows * (int)n, MPI_DOUBLE, 0, br, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Copy local rows to GPU
    double* localA_d = nullptr;
    checkCuda(cudaMalloc(&localA_d, localA_h.size() * sizeof(double)), "cudaMalloc localA_d");
    checkCuda(cudaMemcpy(localA_d, localA_h.data(), localA_h.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D localA_d");

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = distributed_cholesky_hybrid(localA_d, localA_h, lb, (int)n, rank, nranks, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int ok_local = success ? 1 : 0;
    int ok_all = 0;
    MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!ok_all) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        cudaFree(localA_d);
        MPI_Finalize();
        return 1;
    }

    long long t_ms = duration;
    long long t_ms_max = 0;
    MPI_Reduce(&t_ms, &t_ms_max, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", t_ms_max);
        double ops = (double)n * (double)n * (double)n / 3.0;
        double gflops = ops / ((double)t_ms_max / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather result to rank 0 for printing/validation.
    checkCuda(cudaMemcpy(localA_h.data(), localA_d, localA_h.size() * sizeof(double), cudaMemcpyDeviceToHost), "D2H localA");
    cudaFree(localA_d);

    if (rank == 0) {
        // Reassemble full matrix
        std::vector<double> L(n * n);
        const int nb = lb.nb;
        for (int br = 0; br < nb; ++br) {
            const int rows = std::min(blockSize, (int)n - br * blockSize);
            const int srcRank = br % nranks;
            double* dst = L.data() + (size_t)br * (size_t)blockSize * n;
            if (srcRank == 0) {
                const int li = lb.br_to_local[br];
                std::memcpy(dst, localA_h.data() + lb.offsets[li] * n, (size_t)rows * n * sizeof(double));
            } else {
                MPI_Recv(dst, rows * (int)n, MPI_DOUBLE, srcRank, br, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Enforce lower-triangular output semantics.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                L[i * n + j] = 0.0;
            }
        }

        if (printResults) {
            print_results(L, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(L, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }

    } else {
        // Send owned blocks to rank 0
        for (size_t idx = 0; idx < lb.brs.size(); ++idx) {
            const int br = lb.brs[idx];
            const int rows = lb.rows[idx];
            MPI_Send(localA_h.data() + lb.offsets[idx] * n, rows * (int)n, MPI_DOUBLE, 0, br, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
