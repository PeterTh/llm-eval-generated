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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Uses blocked (tiled) algorithm with 1D block-cyclic row distribution

#define BS 128  // Tile (block) size
#define SMTILE 16  // Shared memory tile for GEMM/SYRK kernels

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// POTRF: In-place Cholesky of a bs×bs tile. ptr points to tile start, lda is row stride.
__global__ void kernel_potrf(double* __restrict__ ptr, int lda, int bs) {
    int tid = threadIdx.x;
    for (int j = 0; j < bs; j++) {
        // Diagonal element: single thread computes
        if (tid == 0) {
            double s = 0.0;
            for (int k = 0; k < j; k++) {
                double v = ptr[j * lda + k];
                s += v * v;
            }
            ptr[j * lda + j] = sqrt(ptr[j * lda + j] - s);
        }
        __syncthreads();
        // Sub-diagonal elements in parallel
        double diag = ptr[j * lda + j];
        for (int i = j + 1 + tid; i < bs; i += blockDim.x) {
            double s = 0.0;
            for (int k = 0; k < j; k++)
                s += ptr[i * lda + k] * ptr[j * lda + k];
            ptr[i * lda + j] = (ptr[i * lda + j] - s) / diag;
        }
        __syncthreads();
    }
    // Zero upper triangle
    for (int i = tid; i < bs; i += blockDim.x)
        for (int j = i + 1; j < bs; j++)
            ptr[i * lda + j] = 0.0;
}

// TRSM: Solve X * L^T = B in-place. X_ptr is the tile to solve, L_ptr is the diagonal factor.
// M rows in X, K columns (= size of diagonal block). Each thread handles one row.
__global__ void kernel_trsm(double* __restrict__ X_ptr, const double* __restrict__ L_ptr,
                            int lda, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    for (int j = 0; j < K; j++) {
        double s = 0.0;
        for (int m = 0; m < j; m++)
            s += X_ptr[row * lda + m] * L_ptr[j * lda + m];
        X_ptr[row * lda + j] = (X_ptr[row * lda + j] - s) / L_ptr[j * lda + j];
    }
}

// SYRK: C -= A * A^T (lower triangular update only)
// C is N×N, A is N×K. Shared memory tiled.
__global__ void kernel_syrk(double* __restrict__ C_ptr, const double* __restrict__ A_ptr,
                            int lda, int N, int K) {
    __shared__ double sA[SMTILE][SMTILE + 1];
    __shared__ double sB[SMTILE][SMTILE + 1];

    int row = blockIdx.y * SMTILE + threadIdx.y;
    int col = blockIdx.x * SMTILE + threadIdx.x;

    if (blockIdx.x * SMTILE > (blockIdx.y + 1) * SMTILE) return;

    double val = 0.0;
    int numT = (K + SMTILE - 1) / SMTILE;
    for (int t = 0; t < numT; t++) {
        int kA = t * SMTILE + threadIdx.x;
        int kB = t * SMTILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < N && kA < K) ? A_ptr[row * lda + kA] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (col < N && kB < K) ? A_ptr[col * lda + kB] : 0.0;
        __syncthreads();
        for (int k = 0; k < SMTILE; k++)
            val += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N && col <= row)
        C_ptr[row * lda + col] -= val;
}

// GEMM: C -= A * B^T
// C is M×N, A is M×K, B is N×K. Shared memory tiled.
__global__ void kernel_gemm(double* __restrict__ C_ptr,
                            const double* __restrict__ A_ptr,
                            const double* __restrict__ B_ptr,
                            int lda, int M, int N, int K) {
    __shared__ double sA[SMTILE][SMTILE + 1];
    __shared__ double sB[SMTILE][SMTILE + 1];

    int row = blockIdx.y * SMTILE + threadIdx.y;
    int col = blockIdx.x * SMTILE + threadIdx.x;

    double val = 0.0;
    int numT = (K + SMTILE - 1) / SMTILE;
    for (int t = 0; t < numT; t++) {
        int kA = t * SMTILE + threadIdx.x;
        int kB = t * SMTILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A_ptr[row * lda + kA] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (col < N && kB < K) ? B_ptr[col * lda + kB] : 0.0;
        __syncthreads();
        for (int k = 0; k < SMTILE; k++)
            val += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C_ptr[row * lda + col] -= val;
}

// Zero the upper triangle of the full matrix on GPU
__global__ void kernel_zero_upper(double* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int i = idx / n;
    int j = idx % n;
    if (j > i) A[i * n + j] = 0.0;
}

// ---------------------------------------------------------------------------
// Host helper: copy a tile between GPU matrix and contiguous host buffer
// ---------------------------------------------------------------------------
static void tile_d2h(double* h_buf, const double* d_A, int ti, int tj, int n,
                     int bs_r, int bs_c, cudaStream_t stream) {
    const double* src = d_A + (size_t)ti * BS * n + tj * BS;
    CUDA_CHECK(cudaMemcpy2DAsync(h_buf, bs_c * sizeof(double),
                                  src, n * sizeof(double),
                                  bs_c * sizeof(double), bs_r,
                                  cudaMemcpyDeviceToHost, stream));
}

static void tile_h2d(double* d_A, const double* h_buf, int ti, int tj, int n,
                     int bs_r, int bs_c, cudaStream_t stream) {
    double* dst = d_A + (size_t)ti * BS * n + tj * BS;
    CUDA_CHECK(cudaMemcpy2DAsync(dst, n * sizeof(double),
                                  h_buf, bs_c * sizeof(double),
                                  bs_c * sizeof(double), bs_r,
                                  cudaMemcpyHostToDevice, stream));
}

// ---------------------------------------------------------------------------
// Blocked Cholesky with MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
bool choleskyDecomposition(double* d_A, const size_t n, int rank, int nprocs) {
    int NT = (int)((n + BS - 1) / BS);

    // Tile size helper (handles last tile being smaller)
    auto tsize = [&](int t) -> int { return (int)std::min((size_t)BS, n - (size_t)t * BS); };

    // Create CUDA streams for overlapping kernels
    int nstreams = std::min(32, std::max(1, omp_get_max_threads()));
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Host buffers for MPI communication (pinned for async transfers)
    double* h_diag = nullptr;
    double* h_col = nullptr;  // buffer for one column of tiles
    CUDA_CHECK(cudaMallocHost(&h_diag, (size_t)BS * BS * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_col, (size_t)NT * BS * BS * sizeof(double)));

    // Buffers for allgather
    std::vector<int> sendcounts(nprocs), displs(nprocs);

    for (int k = 0; k < NT; k++) {
        int bs_k = tsize(k);
        int owner_k = k % nprocs;

        // ---- 1. POTRF: Factor diagonal tile (k,k) ----
        if (rank == owner_k) {
            double* tile_kk = d_A + (size_t)k * BS * n + k * BS;
            kernel_potrf<<<1, std::min(bs_k, 256), 0, streams[0]>>>(tile_kk, (int)n, bs_k);
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));
            // Copy diagonal tile to host for broadcast
            tile_d2h(h_diag, d_A, k, k, (int)n, bs_k, bs_k, streams[0]);
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        }
        MPI_Bcast(h_diag, bs_k * bs_k, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
        // All ranks copy diagonal tile to GPU
        if (rank != owner_k) {
            tile_h2d(d_A, h_diag, k, k, (int)n, bs_k, bs_k, streams[0]);
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        }

        // ---- 2. TRSM: Solve tiles in column k below diagonal ----
        // Each rank processes tiles it owns (1D row-cyclic)
        int my_tile_count = 0;
        #pragma omp parallel for schedule(dynamic) reduction(+:my_tile_count)
        for (int i = k + 1; i < NT; i++) {
            if (i % nprocs == rank) {
                int bs_i = tsize(i);
                int sid = omp_get_thread_num() % nstreams;
                double* X = d_A + (size_t)i * BS * n + k * BS;
                double* L = d_A + (size_t)k * BS * n + k * BS;
                int nthreads = ((bs_i + 31) / 32) * 32;
                kernel_trsm<<<1, nthreads, 0, streams[sid]>>>(X, L, (int)n, bs_i, bs_k);
                my_tile_count++;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Allgather column k tiles
        // Pack owned tiles into contiguous send buffer
        int send_count = 0;
        for (int i = k + 1; i < NT; i++) {
            if (i % nprocs == rank) {
                int bs_i = tsize(i);
                tile_d2h(h_col + (size_t)send_count * BS * BS, d_A, i, k, (int)n, bs_i, bs_k, streams[0]);
                send_count++;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // Compute sendcounts and displacements for allgatherv
        int my_send = 0;
        for (int i = k + 1; i < NT; i++)
            if (i % nprocs == rank) my_send++;
        my_send *= BS * bs_k;

        // Use Allgatherv
        std::vector<int> all_sendcounts(nprocs);
        MPI_Allgather(&my_send, 1, MPI_INT, all_sendcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for (int r = 1; r < nprocs; r++)
            displs[r] = displs[r - 1] + all_sendcounts[r - 1];
        int total_recv = displs[nprocs - 1] + all_sendcounts[nprocs - 1];

        // Pack send buffer properly (contiguous tiles, each BS×bs_k)
        std::vector<double> sendbuf(my_send);
        int off = 0;
        for (int i = k + 1; i < NT; i++) {
            if (i % nprocs == rank) {
                int bs_i = tsize(i);
                // Copy from h_col which already has the tile
                // h_col was packed sequentially for owned tiles
                int tile_idx_in_col = 0;
                for (int ii = k + 1; ii < i; ii++)
                    if (ii % nprocs == rank) tile_idx_in_col++;
                memcpy(sendbuf.data() + off, h_col + (size_t)tile_idx_in_col * BS * BS,
                       (size_t)bs_i * bs_k * sizeof(double));
                off += bs_i * bs_k;
            }
        }

        std::vector<double> recvbuf(total_recv);
        MPI_Allgatherv(sendbuf.data(), my_send, MPI_DOUBLE,
                       recvbuf.data(), all_sendcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack received tiles to GPU (tiles from other ranks)
        for (int r = 0; r < nprocs; r++) {
            if (r == rank) continue;
            int recv_off = displs[r];
            for (int i = k + 1; i < NT; i++) {
                if (i % nprocs == r) {
                    int bs_i = tsize(i);
                    // Copy tile (i, k) to GPU
                    tile_h2d(d_A, recvbuf.data() + recv_off, i, k, (int)n, bs_i, bs_k, streams[0]);
                    recv_off += bs_i * bs_k;
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        // ---- 3. Trailing submatrix update ----
        // Each rank updates tiles in rows it owns
        // OpenMP distributes tile pairs across threads, each using a different CUDA stream
        // Collect work items for this rank
        struct WorkItem { int i; int j; };
        std::vector<WorkItem> work;
        for (int i = k + 1; i < NT; i++) {
            if (i % nprocs != rank) continue;
            for (int j = k + 1; j <= i; j++) {
                work.push_back({i, j});
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int w = 0; w < (int)work.size(); w++) {
            int i = work[w].i;
            int j = work[w].j;
            int bs_i = tsize(i);
            int bs_j = tsize(j);
            int sid = omp_get_thread_num() % nstreams;

            double* C = d_A + (size_t)i * BS * n + j * BS;
            double* A_ik = d_A + (size_t)i * BS * n + k * BS;

            dim3 block(SMTILE, SMTILE);
            dim3 grid((bs_j + SMTILE - 1) / SMTILE, (bs_i + SMTILE - 1) / SMTILE);

            if (i == j) {
                kernel_syrk<<<grid, block, 0, streams[sid]>>>(C, A_ik, (int)n, bs_i, bs_k);
            } else {
                double* A_jk = d_A + (size_t)j * BS * n + k * BS;
                kernel_gemm<<<grid, block, 0, streams[sid]>>>(C, A_ik, A_jk, (int)n,
                    bs_i, bs_j, bs_k);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // No MPI sync needed for trailing update! (1D row ownership guarantees correctness)
    }

    // Zero upper triangle
    int total = (int)(n * n);
    int nthreads_z = 256;
    int nblocks_z = (total + nthreads_z - 1) / nthreads_z;
    kernel_zero_upper<<<nblocks_z, nthreads_z>>>(d_A, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_diag));
    CUDA_CHECK(cudaFreeHost(h_col));
    for (int i = 0; i < nstreams; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    return true;
}

// Generate a symmetric positive definite matrix (same as original, deterministic)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T using OpenMP
    #pragma omp parallel for schedule(dynamic)
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

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    #pragma omp parallel for schedule(dynamic)
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU to rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

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

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d, OpenMP threads: %d, GPU: %d\n",
               nprocs, omp_get_max_threads(), rank % num_devices);
        printf("Tile size: %d\n", BS);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks generate the same matrix (deterministic seed)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    // Allocate GPU memory and copy matrix
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(d_A));
        MPI_Finalize();
        return 1;
    }

    // Gather result to rank 0
    // Each rank has its owned tile rows correct; need to assemble full matrix
    // Copy GPU result to host first
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

    // Gather: for each tile row, the owner broadcasts its rows to all
    int NT = (int)((n + BS - 1) / BS);
    for (int tr = 0; tr < NT; tr++) {
        int owner = tr % nprocs;
        int row_start = tr * BS;
        int row_count = (int)std::min((size_t)BS, n - (size_t)tr * BS);
        MPI_Bcast(A.data() + (size_t)row_start * n, (int)((size_t)row_count * n),
                  MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
