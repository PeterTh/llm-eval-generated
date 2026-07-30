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

// ============================================================================
// Hybrid MPI + OpenMP + CUDA Cholesky Decomposition
// 1D block column cyclic distribution across MPI ranks
// GPU acceleration for panel factorization and trailing submatrix update
// ============================================================================

// Block size - balances GPU efficiency with parallelism granularity
#define BLOCK_SIZE 64
// Thread block size for 2D kernels (each thread handles multiple elements)
#define TILE_SIZE 16
// Elements per thread in each dimension
#define ELEMS_PER_DIM (BLOCK_SIZE / TILE_SIZE)  // = 4

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ====================== CUDA Kernels ======================

// Kernel: Cholesky factorization of an NB x NB diagonal block (in panel buffer)
// Launched with 1 block, BLOCK_SIZE threads.
// Panel is row-major with stride = panel_stride (typically NB for contiguous storage).
__global__ void diag_cholesky_kernel(double* d_panel, int panel_stride, int NB) {
    __shared__ double L[BLOCK_SIZE * BLOCK_SIZE];

    int tx = threadIdx.x; // column index 0..NB-1

    // Load column tx into shared memory
    for (int r = 0; r < NB; ++r) {
        L[r * NB + tx] = d_panel[r * panel_stride + tx];
    }
    __syncthreads();

    // Sequential column-by-column Cholesky
    for (int j = 0; j < NB; ++j) {
        if (tx == j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
                sum += L[j * NB + k] * L[j * NB + k];
            L[j * NB + j] = sqrt(L[j * NB + j] - sum);
        }
        __syncthreads();

        if (tx > j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k)
                sum += L[tx * NB + k] * L[j * NB + k];
            L[tx * NB + j] = (L[tx * NB + j] - sum) / L[j * NB + j];
        }
        __syncthreads();
    }

    // Write back
    for (int r = 0; r < NB; ++r) {
        d_panel[r * panel_stride + tx] = L[r * NB + tx];
    }
}

// Kernel: TRSM for one column of the off-diagonal panel
// X[i][j] = (A[i][j] - sum_{k<j} X[i][k] * L[j][k]) / L[j][j]  for i >= NB
// panel_stride = NB (contiguous storage in panel buffer)
__global__ void trsm_panel_column_kernel(double* d_panel, int panel_rows, int NB, int col) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + NB;
    if (row >= panel_rows) return;

    double sum = 0.0;
    for (int k = 0; k < col; ++k) {
        sum += d_panel[row * NB + k] * d_panel[col * NB + k];
    }
    d_panel[row * NB + col] = (d_panel[row * NB + col] - sum) / d_panel[col * NB + col];
}

// Kernel: Copy panel from local distributed storage to contiguous panel buffer
// Only called by the owner of the current block column
__global__ void copy_panel_to_buffer_kernel(
    double* d_panel, const double* d_A, int local_width,
    int k_off, int panel_rows, int NB,
    int local_j)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= panel_rows || c >= NB) return;

    d_panel[r * NB + c] = d_A[(k_off + r) * local_width + local_j * NB + c];
}

// Kernel: Copy factorized panel from buffer back to local storage
__global__ void copy_panel_to_local_kernel(
    double* d_A, const double* d_panel, int local_width,
    int k_off, int panel_rows, int NB,
    int local_j)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= panel_rows || c >= NB) return;

    d_A[(k_off + r) * local_width + local_j * NB + c] = d_panel[r * NB + c];
}

// Kernel: Trailing submatrix update via SYRK/GEMM
// A[I][J] -= L[I][K] * L[J][K]^T for one block column J (owned) and all I >= J
// Each thread block handles one (I, J) block pair.
// Each thread processes ELEMS_PER_DIM x ELEMS_PER_DIM output elements.
__global__ void trailing_update_kernel(
    double* d_A, int local_width,
    const double* d_panel, int k_off,
    int J, int num_blocks, int NB,
    int local_j)
{
    int I = blockIdx.x + J;
    if (I >= num_blocks) return;

    int base_row = I * NB;
    int local_col_start = local_j * NB;

    int ti = threadIdx.y * ELEMS_PER_DIM;
    int tj = threadIdx.x * ELEMS_PER_DIM;

    // Accumulator in registers
    double acc[ELEMS_PER_DIM * ELEMS_PER_DIM];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_DIM * ELEMS_PER_DIM; ++e) acc[e] = 0.0;

    // Compute: acc[di,dj] = sum_{t=0}^{NB-1} L[I*NB+di][k_off+t] * L[J*NB+dj][k_off+t]
    for (int t = 0; t < NB; ++t) {
        // Load L[I*NB+ti+di][k_off+t] from panel
        double l_ik[ELEMS_PER_DIM];
        #pragma unroll
        for (int di = 0; di < ELEMS_PER_DIM; ++di) {
            int ii = ti + di;
            int pr = base_row + ii - k_off;
            l_ik[di] = d_panel[pr * NB + t];
        }

        // Load L[J*NB+tj+dj][k_off+t] from panel
        double l_jk[ELEMS_PER_DIM];
        #pragma unroll
        for (int dj = 0; dj < ELEMS_PER_DIM; ++dj) {
            int jj = tj + dj;
            int pr = J * NB + jj - k_off;
            l_jk[dj] = d_panel[pr * NB + t];
        }

        #pragma unroll
        for (int di = 0; di < ELEMS_PER_DIM; ++di) {
            #pragma unroll
            for (int dj = 0; dj < ELEMS_PER_DIM; ++dj) {
                acc[di * ELEMS_PER_DIM + dj] += l_ik[di] * l_jk[dj];
            }
        }
    }

    // Write back: A[I*NB+ti+di][local_col_start+tj+dj] -= acc[di,dj]
    int row_base = base_row + ti;
    int col_base = local_col_start + tj;
    #pragma unroll
    for (int di = 0; di < ELEMS_PER_DIM; ++di) {
        int row = row_base + di;
        #pragma unroll
        for (int dj = 0; dj < ELEMS_PER_DIM; ++dj) {
            int col = col_base + dj;
            if (ti + di < NB && tj + dj < NB) {
                d_A[row * local_width + col] -= acc[di * ELEMS_PER_DIM + dj];
            }
        }
    }
}

// ====================== Host Helper Functions ======================

// Generate a symmetric positive definite matrix (OpenMP-parallelized)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);

    // Generate random matrix B (not easily parallelized with rand_r, keep sequential)
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T with OpenMP parallelization
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += (double)n;
    }
}

// Validate Cholesky result by computing L * L^T and comparing with A_orig (OpenMP-parallelized)
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (only lower-triangular part needed, computed via min(i,j) bound)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= std::min(i, j); ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original using parallel reduction
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError,relError)
    for (size_t i = 0; i < n * n; ++i) {
        double error = fabs(reconstructed[i] - A_orig[i]);
        if (error > maxError) maxError = error;
        double rel = error / (fabs(A_orig[i]) + 1e-10);
        if (rel > relError) relError = rel;
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
    printf("Usage: mpirun -np <n> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ====================== Hybrid Cholesky ======================
// Performs Cholesky decomposition using MPI + CUDA
// Matrix is distributed 1D block-column cyclic across MPI ranks
// Returns the computation time in milliseconds on rank 0

double hybridCholeskyDecomposition(
    double* d_A, int local_width, int N, int num_blocks, int NB,
    double* d_panel, double* h_panel,
    int num_procs, int my_rank, cudaStream_t stream)
{
    double t_start = MPI_Wtime();

    for (int k = 0; k < num_blocks; ++k) {
        int k_off = k * NB;
        int owner = k % num_procs;
        int panel_rows = N - k_off;  // rows in the panel (including diagonal block)

        // ---- Panel factorization (owner only) ----
        if (my_rank == owner) {
            int local_j = k / num_procs;

            // 1. Copy panel from local distributed storage to contiguous buffer
            dim3 copy_grid((panel_rows + TILE_SIZE - 1) / TILE_SIZE,
                           (NB + TILE_SIZE - 1) / TILE_SIZE);
            dim3 copy_block(TILE_SIZE, TILE_SIZE);
            copy_panel_to_buffer_kernel<<<copy_grid, copy_block, 0, stream>>>(
                d_panel, d_A, local_width, k_off, panel_rows, NB, local_j);

            // 2. Factorize the diagonal block (NB x NB)
            diag_cholesky_kernel<<<1, NB, 0, stream>>>(d_panel, NB, NB);

            // 3. TRSM for off-diagonal part of the panel, one column at a time
            int trsm_rows = panel_rows - NB;
            if (trsm_rows > 0) {
                dim3 trsm_grid((trsm_rows + 255) / 256);
                for (int col = 0; col < NB; ++col) {
                    trsm_panel_column_kernel<<<trsm_grid, 256, 0, stream>>>(
                        d_panel, panel_rows, NB, col);
                }
            }

            // 4. Copy factorized panel back to local storage
            copy_panel_to_local_kernel<<<copy_grid, copy_block, 0, stream>>>(
                d_A, d_panel, local_width, k_off, panel_rows, NB, local_j);

            // 5. Copy panel to host for MPI broadcast
            CUDA_CHECK(cudaMemcpyAsync(h_panel, d_panel,
                                       (size_t)panel_rows * NB * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ---- Broadcast panel from owner to all ranks ----
        MPI_Bcast(h_panel, panel_rows * NB, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // ---- Non-owners: copy panel from host to device ----
        if (my_rank != owner) {
            CUDA_CHECK(cudaMemcpyAsync(d_panel, h_panel,
                                       (size_t)panel_rows * NB * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ---- Trailing submatrix update (all ranks update their owned blocks) ----
        dim3 update_block(TILE_SIZE, TILE_SIZE);
        for (int J = k + 1; J < num_blocks; ++J) {
            if (J % num_procs == my_rank) {
                int local_j = J / num_procs;
                int num_I = num_blocks - J;
                dim3 update_grid(num_I);
                trailing_update_kernel<<<update_grid, update_block, 0, stream>>>(
                    d_A, local_width, d_panel, k_off, J, num_blocks, NB, local_j);
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    double t_end = MPI_Wtime();
    return (t_end - t_start) * 1000.0; // return ms
}

// ====================== Main ======================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Set CUDA device based on local rank (assign GPUs round-robin within a node)
    int local_rank = 0;
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shm_comm);
    MPI_Comm_rank(shm_comm, &local_rank);
    MPI_Comm_free(&shm_comm);

    cudaDeviceProp prop;
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    int dev_id = local_rank % (gpu_count > 0 ? gpu_count : 1);
    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

    // Parse arguments (only rank 0 prints; all parse for correctness)
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
            if (my_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (my_rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (my_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // Adjust n to be a multiple of BLOCK_SIZE
    size_t n_aligned = (n / BLOCK_SIZE) * BLOCK_SIZE;
    if (n_aligned < (size_t)BLOCK_SIZE) n_aligned = BLOCK_SIZE;

    const int NB = BLOCK_SIZE;
    int num_blocks = (int)(n_aligned / NB);
    int N = (int)n_aligned;

    // Compute rank's owned block columns
    int num_owned = 0;
    for (int b = 0; b < num_blocks; ++b) {
        if (b % num_procs == my_rank) num_owned++;
    }
    int local_width = num_owned * NB;
    size_t local_size = (size_t)N * local_width;

    // Host-side full matrix (each rank generates independently using same seed)
    std::vector<double> A_full(N * N);
    std::vector<double> A_orig;
    if (validate) {
        A_orig.resize(N * N);
    }

    // Generate positive definite matrix
    if (my_rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA Hybrid)\n");
        printf("Matrix size: %d x %d (block size %d)\n", N, N, NB);
        printf("MPI processes: %d\n", num_procs);
        printf("CUDA device: %s (GPU %d)\n", prop.name, dev_id);
        printf("Owned blocks per rank: %d\n", num_owned);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    fflush(stdout);

    generatePositiveDefiniteMatrix(A_full, N);

    if (validate) {
        A_orig = A_full;
    }

    // Allocate local storage on host and fill from full matrix
    std::vector<double> A_local(local_size, 0.0);
    int local_idx = 0;
    for (int b = 0; b < num_blocks; ++b) {
        if (b % num_procs == my_rank) {
            int col_start = b * NB;
            int local_col_start = local_idx * NB;
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < NB; ++c) {
                    A_local[(size_t)r * local_width + local_col_start + c] =
                        A_full[(size_t)r * N + col_start + c];
                }
            }
            local_idx++;
        }
    }

    // Allocate GPU memory
    double *d_A = nullptr, *d_panel = nullptr;
    double *h_panel = nullptr;
    size_t max_panel_size = (size_t)N * NB;
    cudaStream_t stream;

    CUDA_CHECK(cudaMalloc(&d_A, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel, max_panel_size * sizeof(double)));
    CUDA_CHECK(cudaHostAlloc(&h_panel, max_panel_size * sizeof(double),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Copy local matrix to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_A, A_local.data(), local_size * sizeof(double),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warm-up GPU
    if (my_rank == 0) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Run hybrid Cholesky
    double duration_ms = hybridCholeskyDecomposition(
        d_A, local_width, N, num_blocks, NB,
        d_panel, h_panel, num_procs, my_rank, stream);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A_local.data(), d_A, local_size * sizeof(double),
                           cudaMemcpyDeviceToHost));

    // Gather full L on rank 0 for validation
    std::vector<double> L_full;
    double gather_time_start = MPI_Wtime();

    if (my_rank == 0) {
        L_full.resize((size_t)N * N, 0.0);

        // Copy rank 0's owned blocks
        local_idx = 0;
        for (int b = 0; b < num_blocks; b += num_procs) {
            int col_start = b * NB;
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < NB; ++c) {
                    L_full[(size_t)r * N + col_start + c] =
                        A_local[(size_t)r * local_width + local_idx * NB + c];
                }
            }
            local_idx++;
        }

        // Receive from other ranks
        for (int rank = 1; rank < num_procs; ++rank) {
            int recv_owned = 0;
            for (int b = rank; b < num_blocks; b += num_procs) recv_owned++;
            int recv_size = N * recv_owned * NB;
            std::vector<double> recv_buf(recv_size);
            MPI_Recv(recv_buf.data(), recv_size, MPI_DOUBLE, rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int buf_pos = 0;
            for (int b = rank; b < num_blocks; b += num_procs) {
                int col_start = b * NB;
                for (int r = 0; r < N; ++r) {
                    for (int c = 0; c < NB; ++c) {
                        L_full[(size_t)r * N + col_start + c] = recv_buf[buf_pos++];
                    }
                }
            }
        }
    } else {
        // Send owned blocks to rank 0
        int send_owned = 0;
        for (int b = my_rank; b < num_blocks; b += num_procs) send_owned++;
        int send_size = N * send_owned * NB;
        std::vector<double> send_buf(send_size);
        int buf_pos = 0;
        local_idx = 0;
        for (int b = my_rank; b < num_blocks; b += num_procs) {
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < NB; ++c) {
                    send_buf[buf_pos++] = A_local[(size_t)r * local_width + local_idx * NB + c];
                }
            }
            local_idx++;
        }
        MPI_Send(send_buf.data(), send_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    double gather_time = (MPI_Wtime() - gather_time_start) * 1000.0;

    // Check for computational errors
    bool chol_success = true;

    // Report timing and validation (rank 0 only)
    if (my_rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Gather time: %.3f ms\n", gather_time);

        double ops = (double)N * N * N / 3.0;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(L_full, "CholeskyL");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            chol_success = validateCholesky(L_full, A_orig, N);
            if (chol_success) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_panel));
    CUDA_CHECK(cudaFreeHost(h_panel));

    MPI_Finalize();

    if (my_rank == 0 && validate && !chol_success) return 1;
    return 0;
}
