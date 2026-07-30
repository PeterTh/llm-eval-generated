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

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TILE_DIM
#define TILE_DIM 16
#endif

// ---------------------------------------------------------------------------
// CUDA error-checking helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = call;                                               \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: ceiling division and min
// ---------------------------------------------------------------------------
static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
static inline int min_i(int a, int b)  { return a < b ? a : b; }

// ---------------------------------------------------------------------------
// CUDA kernel: diagonal Cholesky factorisation for a single NB×NB block
// Each thread handles one row; the whole block is handled by one thread-block.
// ---------------------------------------------------------------------------
__global__ void diag_cholesky_kernel(double *A, int ldA, int n) {
    int row = threadIdx.x;
    if (row >= n) return;

    for (int col = 0; col < n; ++col) {
        // Dot product of rows `row` and `col` over columns 0 .. col-1
        double sum = 0.0;
        if (row >= col) {
            for (int k = 0; k < col; ++k)
                sum += A[row * ldA + k] * A[col * ldA + k];
        }
        __syncthreads();

        if (row == col) {
            double val = A[col * ldA + col] - sum;
            // Guard against tiny negative values from FP roundoff
            if (val < 0.0 && val > -1.0e-12) val = 0.0;
            A[col * ldA + col] = sqrt(val);
        }
        __syncthreads();

        if (row > col)
            A[row * ldA + col] = (A[row * ldA + col] - sum) / A[col * ldA + col];
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: triangular right-solve  X * L^T = B   (right-side TRSM)
// L is n×n lower-triangular.  B / X are m×n (rows independent).
// Each thread processes one row of X with forward substitution across columns.
// Works in-place (X == B).
// ---------------------------------------------------------------------------
__global__ void trsm_kernel(const double *L, int ldL,
                            const double *B, int ldB,
                            double *X, int ldX,
                            int n, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    while (row < m) {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k)
                sum += X[row * ldX + k] * L[col * ldL + k];
            X[row * ldX + col] =
                (B[row * ldB + col] - sum) / L[col * ldL + col];
        }
        row += gridDim.x * blockDim.x;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: trailing submatrix update  C -= A * B^T
// Each thread computes one element of C.
// If `diag` is set (diagonal block), only the lower triangle is updated.
// ---------------------------------------------------------------------------
__global__ void update_sub_kernel(double *C, int ldC,
                                   const double *A, int ldA,
                                   const double *B, int ldB,
                                   int m, int n, int k, int diag) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;
    if (diag && row < col) return;

    double sum = 0.0;
    for (int t = 0; t < k; ++t)
        sum += A[row * ldA + t] * B[col * ldB + t];

    C[row * ldC + col] -= sum;
}


// ---------------------------------------------------------------------------
// CUDA kernel: pack a column block (strided in source) into a flat buffer
// src[r][c] = src_base[r * ld_src + col_ofs + c]   (r=0..n-1, c=0..bw-1)
// dst[r * bw + c] = src[r][c]
// ---------------------------------------------------------------------------
__global__ void pack_col_block_kernel(const double *src, int ld_src, int col_ofs,
                                       double *dst, int bw, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    for (int c = 0; c < bw; ++c)
        dst[static_cast<size_t>(r) * bw + c] = src[static_cast<size_t>(r) * ld_src + col_ofs + c];
}


// ---------------------------------------------------------------------------
// Set up the CUDA device for this MPI rank (round-robin across visible GPUs)
// ---------------------------------------------------------------------------
static void setup_cuda_device(MPI_Comm comm) {
    int local_rank = 0;
    MPI_Comm shm_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &local_rank);

    int ngpus = 0;
    cudaGetDeviceCount(&ngpus);
    if (ngpus == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int dev = local_rank % ngpus;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(0)); // lazy init / verify

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("Using %d GPU(s) – each MPI rank assigned round-robin\n", ngpus);
        printf("GPU %d: %s  (SM %d.%d, %.1f GB)\n", dev, prop.name,
               prop.major, prop.minor, prop.totalGlobalMem / 1e9);
    }

    MPI_Comm_free(&shm_comm);
}

// ---------------------------------------------------------------------------
// Distributed matrix generation – each rank stores only its column blocks.
// All ranks generate the same B (deterministic seed) and compute
// A = B * B^T + n*I for their local columns.
// ---------------------------------------------------------------------------
static void generate_local_matrix(std::vector<double> &local_mat,
                                   std::vector<double> &B,
                                   std::vector<int> const &col_block_ofs,
                                   int first_block, int local_blocks,
                                   int nb, int n, int local_n) {
    // Generate B once per rank (deterministic, same on all ranks)
    B.resize(static_cast<size_t>(n) * n);
    unsigned int seed = 42;
    for (size_t idx = 0; idx < static_cast<size_t>(n) * n; ++idx)
        B[idx] = (static_cast<double>(rand_r(&seed)) / RAND_MAX) - 0.5;

    local_mat.assign(static_cast<size_t>(n) * local_n, 0.0);

    // Compute A[:, local_cols] = B * B^T for local columns
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int lb = 0; lb < local_blocks; ++lb) {
            int gb   = first_block + lb;
            int cst  = gb * nb;
            int cend = min_i(n, cst + nb);
            int bw   = cend - cst;
            int loc  = col_block_ofs[lb];

            for (int c = 0; c < bw; ++c) {
                int gj = cst + c;
                double sum = 0.0;
                for (int k = 0; k < n; ++k)
                    sum += B[static_cast<size_t>(i) * n + k] *
                           B[static_cast<size_t>(gj) * n + k];
                local_mat[static_cast<size_t>(i) * local_n + loc + c] = sum;
            }
        }
        // Add diagonal dominance (n * I)
        int diag_col_block = i / nb;
        int diag_lb = diag_col_block - first_block;
        if (diag_lb >= 0 && diag_lb < local_blocks) {
            int cst  = diag_col_block * nb;
            int loc  = col_block_ofs[diag_lb];
            int lc   = i - cst;   // local column index within its block
            local_mat[static_cast<size_t>(i) * local_n + loc + lc] += n;
        }
    }
}

// ---------------------------------------------------------------------------
// Determine column-block distribution.
// Returns: first_block, local_blocks,
//          col_block_ofs[local_blocks] = local column offset for each block,
//          local_n = total local columns.
// ---------------------------------------------------------------------------
static void compute_distribution(int num_blocks, int rank, int num_ranks,
                                 int nb, int n,
                                 int *first_block, int *local_blocks,
                                 std::vector<int> *col_block_ofs,
                                 int *local_n) {
    int base   = num_blocks / num_ranks;
    int extra  = num_blocks % num_ranks;
    int f = rank * base + (rank < extra ? rank : extra);
    int l = base + (rank < extra ? 1 : 0);
    *first_block = f;
    *local_blocks = l;

    col_block_ofs->resize(l);
    int offset = 0;
    for (int i = 0; i < l; ++i) {
        (*col_block_ofs)[i] = offset;
        int gb = f + i;
        int bw = min_i(nb, n - gb * nb);
        offset += bw;
    }
    *local_n = offset;
}

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA block Cholesky decomposition.
// ---------------------------------------------------------------------------
static bool hybrid_cholesky(std::vector<double> &local_mat,
                             std::vector<int> const &col_block_ofs,
                             int first_block, int local_blocks,
                             int nb, int n, int local_n,
                             MPI_Comm comm,
                             double *gpu_time, double *comm_time) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    int num_blocks     = ceildiv(n, nb);
    int max_bw         = min_i(nb, n);

    // ---- GPU allocations ---------------------------------------------------
    double *d_local   = nullptr;   // entire local matrix stays on GPU
    double *d_col     = nullptr;   // temporary buffer for received column k
    double *h_col     = nullptr;   // host staging buffer for broadcast

    CUDA_CHECK(cudaMalloc(&d_local, static_cast<size_t>(n) * local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col,   static_cast<size_t>(n) * max_bw * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_col, static_cast<size_t>(n) * max_bw * sizeof(double)));

    // Upload the entire local matrix to GPU once
    CUDA_CHECK(cudaMemcpy(d_local, local_mat.data(),
                          static_cast<size_t>(n) * local_n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- compute block ownership -------------------------------------------
    // For a global column block k we need to know which rank owns it.
    auto block_owner_ = [&](int k) -> int {
        int base  = num_blocks / num_ranks;
        int extra = num_blocks % num_ranks;
        if (k < extra * (base + 1))
            return k / (base + 1);
        else
            return (k - extra) / base;
    };

    *gpu_time = 0.0;
    *comm_time = 0.0;

    // ---- right-looking block Cholesky --------------------------------------
    for (int k = 0; k < num_blocks; ++k) {
        int owner   = block_owner_(k);
        int cst_k   = k * nb;
        int cend_k  = min_i(n, cst_k + nb);
        int bw_k    = cend_k - cst_k;
        int send_sz = n * bw_k;

        double t0, t1;

        // --- Step A: diagonal Cholesky + TRSM on GPU (owner only) ---------
        if (rank == owner) {
            // The entire column k is stored in d_local; its local column offset
            int col_k_ofs = 0; // offset of column block k in d_local (in doubles)
            for (int b = 0; b < local_blocks; ++b) {
                if ((first_block + b) == k) {
                    col_k_ofs = col_block_ofs[b];
                    break;
                }
            }

            double *d_Akk = d_local + static_cast<size_t>(k) * nb * local_n + col_k_ofs;
            int diag_n    = min_i(nb, n - k * nb);

            // Factor the diagonal block
            t0 = omp_get_wtime();
            diag_cholesky_kernel<<<1, diag_n>>>(d_Akk, local_n, diag_n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            t1 = omp_get_wtime();
            *gpu_time += (t1 - t0);

            // Triangular solves for all row-blocks i below diagonal in col k
            for (int i = k + 1; i < num_blocks; ++i) {
                int row_start = i * nb;
                int row_end   = min_i(n, row_start + nb);
                int rows_i    = row_end - row_start;

                double *d_Aik = d_local + static_cast<size_t>(row_start) * local_n + col_k_ofs;

                t0 = omp_get_wtime();
                // Right-side solve X * L^T = B  (X = A_ik, L = A_kk, B = A_ik)
                // X is rows_i x bw_k, L is bw_k x bw_k (n = bw_k, m = rows_i)
                int tpb = 256;
                dim3 trsm_grid(ceildiv(rows_i, tpb), 1);
                dim3 trsm_blk(tpb, 1);
                trsm_kernel<<<trsm_grid, trsm_blk>>>(
                    d_Akk, local_n,
                    d_Aik, local_n,
                    d_Aik, local_n,
                    bw_k, rows_i);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                t1 = omp_get_wtime();
                *gpu_time += (t1 - t0);
            }

            // --- Step B: pack factored column k into flat buffer and broadcast --------
            t0 = omp_get_wtime();
            {
                int tpb = 256;
                dim3 pgrid(ceildiv(n, tpb), 1);
                dim3 pblk(tpb, 1);
                pack_col_block_kernel<<<pgrid, pblk>>>(
                    d_local, local_n, col_k_ofs,
                    d_col, bw_k, n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_col, d_col,
                                      static_cast<size_t>(send_sz) * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
            t1 = omp_get_wtime();
            *gpu_time += (t1 - t0);

            t0 = omp_get_wtime();
            MPI_Bcast(h_col, send_sz, MPI_DOUBLE, owner, comm);
            t1 = omp_get_wtime();
            *comm_time += (t1 - t0);

            // d_col already has the packed column block from the pack kernel above.
        } else {
            // Receive column k from owner
            t0 = omp_get_wtime();
            MPI_Bcast(h_col, send_sz, MPI_DOUBLE, owner, comm);
            t1 = omp_get_wtime();
            *comm_time += (t1 - t0);

            // Upload received data to flat buffer on GPU
            CUDA_CHECK(cudaMemcpy(d_col, h_col,
                                  static_cast<size_t>(send_sz) * sizeof(double),
                              cudaMemcpyHostToDevice));
        }

        // --- Step C: update trailing submatrix on GPU (all ranks) -----------
        // For each local column block j > k, update row blocks i >= j.
        t0 = omp_get_wtime();
        dim3 blk(TILE_DIM, TILE_DIM);

        for (int lb = 0; lb < local_blocks; ++lb) {
            int gb = first_block + lb;
            if (gb <= k) continue;

            int cst_j   = gb * nb;
            int cend_j  = min_i(n, cst_j + nb);
            int bw_j    = cend_j - cst_j;
            int lcol_ofs_j = col_block_ofs[lb];

            for (int i = gb; i < num_blocks; ++i) {
                int rst   = i * nb;
                int rend  = min_i(n, rst + nb);
                int rows_i = rend - rst;

                double *d_C = d_local + static_cast<size_t>(rst) * local_n + lcol_ofs_j;

                // A = L_ik (row block i, column k)
                double *d_A = d_col + static_cast<size_t>(rst) * bw_k;
                // B = L_jk (row block j, column k)
                double *d_B = d_col + static_cast<size_t>(cst_j) * bw_k;

                int diag_flag = (i == gb) ? 1 : 0;

                dim3 grid(ceildiv(bw_j, TILE_DIM), ceildiv(rows_i, TILE_DIM));
                update_sub_kernel<<<grid, blk>>>(
                    d_C, local_n, d_A, bw_k, d_B, bw_k,
                    rows_i, bw_j, bw_k, diag_flag);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t1 = omp_get_wtime();
        *gpu_time += (t1 - t0);

    } // end for k

    // Download the factored matrix back to CPU
    CUDA_CHECK(cudaMemcpy(local_mat.data(), d_local,
                          static_cast<size_t>(n) * local_n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_local));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFreeHost(h_col));

    return true;
}

// ---------------------------------------------------------------------------
// Validation – gather the factored matrix to rank 0 and check L * L^T == A_orig
// ---------------------------------------------------------------------------
static bool validate_result(std::vector<double> const &local_L,
                             std::vector<double> const &A_orig,
                             int nb, int n, int local_n,
                             MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    std::vector<double> L_full;
    double max_error = 0.0, rel_error = 0.0;

    if (rank == 0) {
        L_full.assign(static_cast<size_t>(n) * n, 0.0);
    }

    // Gather column blocks to rank 0 using Gatherv
    // For simplicity, rank 0 sends displacements to itself
    std::vector<int> send_counts(num_ranks, 0);
    std::vector<int> displs(num_ranks, 0);

    // Each rank's local_n is different; we need to send n * local_n doubles
    int send_sz = n * local_n;

    // Gather sizes
    MPI_Gather(&send_sz, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, comm);

    // Compute displacements on rank 0
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            displs[r] = offset;
            offset += send_counts[r];
        }
    }

    // Gather the data
    // Note: the received buffer has to be large enough on rank 0
    // L_full needs n * n entries; the sum of send_counts should equal n * n
    MPI_Gatherv(local_L.data(), send_sz, MPI_DOUBLE,
                rank == 0 ? L_full.data() : nullptr,
                send_counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (rank != 0) return true; // only rank 0 validates

    int num_blocks = ceildiv(n, nb);

    // Now L_full has all columns concatenated in rank order.
    // Reconstruct into a proper n × n matrix.
    // The receive layout: for rank r, the data is n contiguous elements for
    // each of its column blocks.  We need to de-interleave into the correct
    // column positions.
    // We stored column blocks contiguously per rank, so L_full has all rank 0's
    // columns first, then rank 1's, etc.
    // Each rank sends n * local_n doubles row-major: row 0 local cols, row 1 local cols, ...
    // So L_full is: [rank0_cols][rank1_cols]...
    // Where rank_r_cols = n * (local_n for rank r) doubles.
    // To reconstruct: for each rank r, for each column block gb it owns,
    // copy data into the right position in the n×n matrix.

    // Actually, let me just scatter and have each rank validate its own columns.
    // Simpler: just check that L * L^T ≈ A_orig by computing the product
    // on rank 0 using the gathered L.

    // Already have L_full which is column-block-major by rank.
    // Rebuild the standard n×n matrix.
    std::vector<double> L_rebuilt(static_cast<size_t>(n) * n, 0.0);

    int offset = 0;
    int base   = num_blocks / num_ranks;
    int extra  = num_blocks % num_ranks;
    for (int r = 0; r < num_ranks; ++r) {
        int l_blocks = base + (r < extra ? 1 : 0);
        int f_block  = r * base + (r < extra ? r : extra);
        int l_n      = 0;
        for (int lb = 0; lb < l_blocks; ++lb) {
            int gb   = f_block + lb;
            int cst  = gb * nb;
            int cend = min_i(n, cst + nb);
            l_n += cend - cst;
        }

        // Copy from L_full[offset] into the right columns of L_rebuilt
        int col_pos = 0; // position within this rank's data
        for (int lb = 0; lb < l_blocks; ++lb) {
            int gb   = f_block + lb;
            int cst  = gb * nb;
            int cend = min_i(n, cst + nb);
            int bw   = cend - cst;

            for (int i = 0; i < n; ++i) {
                for (int c = 0; c < bw; ++c) {
                    int src_idx = offset + static_cast<size_t>(i) * l_n + col_pos + c;
                    int dst_idx = static_cast<size_t>(i) * n + cst + c;
                    L_rebuilt[dst_idx] = L_full[src_idx];
                }
            }
            col_pos += bw;
        }
        offset += n * l_n;
    }

    // Compute L * L^T and compare with A_orig
    std::vector<double> reconstructed(static_cast<size_t>(n) * n, 0.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= j; ++k) {
                sum += L_rebuilt[static_cast<size_t>(i) * n + k] *
                       L_rebuilt[static_cast<size_t>(j) * n + k];
            }
            reconstructed[static_cast<size_t>(i) * n + j] = sum;
            // Since A_orig is symmetric and L*L^T is symmetric:
            reconstructed[static_cast<size_t>(j) * n + i] = sum;
        }
    }

    max_error = 0.0;
    rel_error = 0.0;
    for (size_t idx = 0; idx < static_cast<size_t>(n) * n; ++idx) {
        double err = fabs(reconstructed[idx] - A_orig[idx]);
        max_error = std::max(max_error, err);
        double rel = err / (fabs(A_orig[idx]) + 1e-10);
        rel_error = std::max(rel_error, rel);
    }

    printf("Max absolute error: %.10e\n", max_error);
    printf("Max relative error: %.10e\n", rel_error);

    if (rel_error > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Print results (rank 0 only)
// ---------------------------------------------------------------------------
static void print_local_results(std::vector<double> const &local_L,
                                 int nb, int n, int local_n,
                                 MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    std::vector<double> L_full;
    if (rank == 0) L_full.assign(static_cast<size_t>(n) * n, 0.0);

    int send_sz = n * local_n;
    std::vector<int> send_counts(num_ranks, 0);
    std::vector<int> displs(num_ranks, 0);
    MPI_Gather(&send_sz, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, comm);
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            displs[r] = offset;
            offset += send_counts[r];
        }
    }
    MPI_Gatherv(local_L.data(), send_sz, MPI_DOUBLE,
                rank == 0 ? L_full.data() : nullptr,
                send_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (rank != 0) return;

    // Rebuild full matrix (same as in validation)
    int num_blocks = ceildiv(n, nb);
    std::vector<double> L_rebuilt(static_cast<size_t>(n) * n, 0.0);
    int offset = 0;
    int base   = num_blocks / num_ranks;
    int extra  = num_blocks % num_ranks;
    for (int r = 0; r < num_ranks; ++r) {
        int l_blocks = base + (r < extra ? 1 : 0);
        int f_block  = r * base + (r < extra ? r : extra);
        int l_n      = 0;
        for (int lb = 0; lb < l_blocks; ++lb) {
            int gb   = f_block + lb;
            int cst  = gb * nb;
            int cend = min_i(n, cst + nb);
            l_n += cend - cst;
        }
        int col_pos = 0;
        for (int lb = 0; lb < l_blocks; ++lb) {
            int gb   = f_block + lb;
            int cst  = gb * nb;
            int cend = min_i(n, cst + nb);
            int bw   = cend - cst;
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < bw; ++c)
                    L_rebuilt[static_cast<size_t>(i) * n + cst + c] =
                        L_full[offset + static_cast<size_t>(i) * l_n + col_pos + c];
            col_pos += bw;
        }
        offset += n * l_n;
    }

    print_results(L_rebuilt, "CholeskyL");
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Parse arguments on rank 0, broadcast
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
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

    // Broadcast arguments
    MPI_Bcast(&n, sizeof(n), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, sizeof(validate), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, sizeof(printResults), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Setup CUDA device (each rank picks a GPU)
    setup_cuda_device(MPI_COMM_WORLD);

    int nb        = BLOCK_SIZE;
    int num_blks  = ceildiv(static_cast<int>(n), nb);

    // Determine distribution
    int first_block, local_blocks, local_n;
    std::vector<int> col_block_ofs;
    compute_distribution(num_blks, rank, num_ranks, nb, static_cast<int>(n),
                         &first_block, &local_blocks, &col_block_ofs, &local_n);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Block size:  %d\n", nb);
        printf("MPI ranks:   %d\n", num_ranks);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Validation:  %s\n", validate ? "enabled" : "disabled");
        printf("Column blocks per rank: %d – %d\n", first_block,
               first_block + local_blocks - 1);
        fflush(stdout);
    }

    // Generate matrix (distributed)
    std::vector<double> B_full;
    std::vector<double> local_mat;
    generate_local_matrix(local_mat, B_full, col_block_ofs,
                          first_block, local_blocks, nb,
                          static_cast<int>(n), local_n);
    B_full.clear();
    B_full.shrink_to_fit();

    // Save original if validating
    std::vector<double> A_orig;
    if (validate) {
        // All ranks participate in the gather; only rank 0 stores the result
        int send_sz = static_cast<int>(n) * local_n;
        std::vector<int> sc(num_ranks, 0), disp(num_ranks, 0);
        MPI_Gather(&send_sz, 1, MPI_INT, sc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> gathered;
        if (rank == 0) {
            int total = 0;
            for (int r = 0; r < num_ranks; ++r) { disp[r] = total; total += sc[r]; }
            gathered.assign(total, 0.0);
        }
        MPI_Gatherv(local_mat.data(), send_sz, MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr,
                    sc.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Rebuild into proper n×n layout from the gather buffer
            A_orig.assign(static_cast<size_t>(n) * n, 0.0);
            int base   = num_blks / num_ranks;
            int extra  = num_blks % num_ranks;
            int offset = 0;
            for (int r = 0; r < num_ranks; ++r) {
                int l_blks = base + (r < extra ? 1 : 0);
                int f_blk  = r * base + (r < extra ? r : extra);
                int ln      = 0;
                for (int lb = 0; lb < l_blks; ++lb) {
                    int gb = f_blk + lb;
                    ln += min_i(nb, static_cast<int>(n) - gb * nb);
                }
                int col_pos = 0;
                for (int lb = 0; lb < l_blks; ++lb) {
                    int gb   = f_blk + lb;
                    int cst  = gb * nb;
                    int cend = min_i(static_cast<int>(n), cst + nb);
                    int bw   = cend - cst;
                    for (int i = 0; i < static_cast<int>(n); ++i)
                        for (int c = 0; c < bw; ++c)
                            A_orig[static_cast<size_t>(i) * n + cst + c] =
                                gathered[offset + static_cast<size_t>(i) * ln + col_pos + c];
                    col_pos += bw;
                }
                offset += static_cast<int>(n) * ln;
            }
        }
    }

    // Wait for all ranks to be ready
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    double gpu_time = 0.0, comm_time = 0.0;
    bool success = hybrid_cholesky(local_mat, col_block_ofs,
                                    first_block, local_blocks,
                                    nb, static_cast<int>(n), local_n,
                                    MPI_COMM_WORLD, &gpu_time, &comm_time);

    MPI_Barrier(MPI_COMM_WORLD);
    auto wall_end = std::chrono::high_resolution_clock::now();
    auto wall_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                        wall_end - wall_start);

    if (!success && rank == 0) {
        printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Compute aggregate times across ranks
    double total_gpu = 0.0, total_comm = 0.0;
    MPI_Reduce(&gpu_time, &total_gpu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &total_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (wall_dur.count() / 1000.0) / 1e9;
        printf("Wall-clock time: %ld ms\n", wall_dur.count());
        printf("GPU compute time (max): %.3f s\n", total_gpu);
        printf("Communication time (max): %.3f s\n", total_comm);
        printf("Performance: %.3f GFLOPS\n", gflops);
        fflush(stdout);
    }

    // Print results for external validation
    if (printResults) {
        print_local_results(local_mat, nb,
                            static_cast<int>(n), local_n,
                            MPI_COMM_WORLD);
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            fflush(stdout);
        }
        bool valid = validate_result(local_mat, A_orig, nb,
                                      static_cast<int>(n), local_n,
                                      MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
