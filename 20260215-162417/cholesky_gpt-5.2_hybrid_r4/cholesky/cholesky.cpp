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

static inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

static inline int tile_dim(const int t, const int n, const int bs) {
    const int start = t * bs;
    const int rem = n - start;
    return rem > bs ? bs : (rem > 0 ? rem : 0);
}

// ---------------- CUDA trailing update: C(mxn) -= A(mxk) * B(nxk)^T ----------------
static constexpr int CUDA_TILE = 16;

__global__ void gemm_update_tn_kernel(double* __restrict__ C,
                                      const double* __restrict__ A,
                                      const double* __restrict__ B,
                                      int m, int n, int k, int ld) {
    __shared__ double As[CUDA_TILE][CUDA_TILE];
    __shared__ double Bs[CUDA_TILE][CUDA_TILE];

    const int row = blockIdx.y * CUDA_TILE + threadIdx.y;
    const int col = blockIdx.x * CUDA_TILE + threadIdx.x;

    double acc = 0.0;
    for (int p0 = 0; p0 < k; p0 += CUDA_TILE) {
        const int a_col = p0 + threadIdx.x;
        const int b_col = p0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? A[row * ld + a_col] : 0.0;
        // B is (n x k) row-major; for B^T we need B[col, p]
        Bs[threadIdx.y][threadIdx.x] = (col < n && b_col < k) ? B[col * ld + b_col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < CUDA_TILE; ++t) {
            acc = fma(As[threadIdx.y][t], Bs[t][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * ld + col] -= acc;
    }
}

static inline void launch_gemm_update(double* dC, const double* dA, const double* dB,
                                      int m, int n, int k, int ld) {
    dim3 threads(CUDA_TILE, CUDA_TILE);
    dim3 blocks((n + CUDA_TILE - 1) / CUDA_TILE, (m + CUDA_TILE - 1) / CUDA_TILE);
    gemm_update_tn_kernel<<<blocks, threads>>>(dC, dA, dB, m, n, k, ld);
}

// ---------------- CPU panel ops (OpenMP) ----------------
static bool tile_potrf_lower(double* A, int nloc, int ld) {
    for (int j = 0; j < nloc; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            const double v = A[j * ld + k];
            sum += v * v;
        }
        const double val = A[j * ld + j] - sum;
        if (val <= 0.0) {
            return false;
        }
        A[j * ld + j] = std::sqrt(val);

        for (int i = j + 1; i < nloc; ++i) {
            double s = 0.0;
            for (int k = 0; k < j; ++k) {
                s += A[i * ld + k] * A[j * ld + k];
            }
            A[i * ld + j] = (A[i * ld + j] - s) / A[j * ld + j];
        }

        // zero upper triangle within the diagonal tile
        for (int k = j + 1; k < nloc; ++k) {
            A[j * ld + k] = 0.0;
        }
    }
    return true;
}

static void tile_trsm_right_upper_trans(double* Aik, const double* Lkk, int m, int k, int ld) {
    // Aik := Aik * inv(Lkk^T), where Lkk is lower-triangular (k x k)
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < k; ++c) {
            double sum = 0.0;
            for (int t = 0; t < c; ++t) {
                // (Lkk^T)[t,c] = Lkk[c,t]
                sum += Aik[r * ld + t] * Lkk[c * ld + t];
            }
            Aik[r * ld + c] = (Aik[r * ld + c] - sum) / Lkk[c * ld + c];
        }
    }
}

// Generate a symmetric positive definite matrix (kept identical to original)
static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

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

    size_t n = 512;
    int validate = 0;
    int printResults = 0;
    int early_exit = 0;
    int early_code = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                early_exit = 1;
                early_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                early_exit = 1;
                early_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&early_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&early_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (early_exit) {
        MPI_Finalize();
        return early_code;
    }

    unsigned long long n_ull = static_cast<unsigned long long>(n);
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_ull);

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: No CUDA devices available (required).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // Choose a performance-oriented block size; keep within n.
    int bs = 256;
    if (n < static_cast<size_t>(bs)) bs = (n >= 128 ? 128 : static_cast<int>(n));
    if (bs <= 0) {
        if (rank == 0) printf("Matrix size must be positive\n");
        MPI_Finalize();
        return 1;
    }

    const int nn = static_cast<int>(n);
    const int nt = (nn + bs - 1) / bs;

    // Tile distribution: cyclic by tile-column (j % size)
    std::vector<int> tile_counts(size, 0);
    for (int j = 0; j < nt; ++j) {
        tile_counts[j % size] += (nt - j);
    }
    std::vector<int> sendcounts(size, 0), displs(size, 0);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = tile_counts[r] * bs * bs;
        displs[r] = (r == 0) ? 0 : (displs[r - 1] + sendcounts[r - 1]);
    }

    const int local_tiles_count = tile_counts[rank];
    std::vector<double> h_tiles(static_cast<size_t>(local_tiles_count) * bs * bs);
    std::vector<int> local_index(static_cast<size_t>(nt) * nt, -1);
    std::vector<int> local_columns;
    local_columns.reserve((nt + size - 1) / size);
    std::vector<int> col_first(nt, -1);

    // Build local tile ordering: for j in {rank, rank+size, ...}, tiles (i=j..nt-1, j)
    int tpos = 0;
    for (int j = rank; j < nt; j += size) {
        local_columns.push_back(j);
        col_first[j] = tpos;
        for (int i = j; i < nt; ++i) {
            local_index[i * nt + j] = tpos;
            ++tpos;
        }
    }

    std::vector<double> sendbuf;
    std::vector<double> A_full;
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) A_orig = A_full;

        sendbuf.resize(static_cast<size_t>(displs[size - 1] + sendcounts[size - 1]));
        for (int r = 0; r < size; ++r) {
            size_t off = static_cast<size_t>(displs[r]);
            for (int j = r; j < nt; j += size) {
                const int bj = tile_dim(j, nn, bs);
                for (int i = j; i < nt; ++i) {
                    const int bi = tile_dim(i, nn, bs);
                    double* tile = sendbuf.data() + off;
                    for (int rr = 0; rr < bs; ++rr) {
                        const int gr = i * bs + rr;
                        for (int cc = 0; cc < bs; ++cc) {
                            const int gc = j * bs + cc;
                            const double v = (rr < bi && cc < bj) ? A_full[static_cast<size_t>(gr) * n + static_cast<size_t>(gc)] : 0.0;
                            tile[rr * bs + cc] = v;
                        }
                    }
                    off += static_cast<size_t>(bs) * bs;
                }
            }
        }
    }

    // Scatter initial lower-triangular tiles to owners.
    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_DOUBLE,
                 h_tiles.data(), sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Device storage: keep all local tiles resident on GPU.
    double* d_tiles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tiles, static_cast<size_t>(local_tiles_count) * bs * bs * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_tiles, h_tiles.data(), static_cast<size_t>(sendcounts[rank]) * sizeof(double), cudaMemcpyHostToDevice));

    // Temporary device buffer for broadcasted panels (max size nt tiles).
    double* d_panel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel, static_cast<size_t>(nt) * bs * bs * sizeof(double)));

    // Factorization
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    std::vector<double> panel;
    panel.reserve(static_cast<size_t>(nt) * bs * bs);

    int ok_local = 1;

    for (int k = 0; k < nt; ++k) {
        const int owner = k % size;
        const int bk = tile_dim(k, nn, bs);
        const int panel_tiles = nt - k;
        panel.assign(static_cast<size_t>(panel_tiles) * bs * bs, 0.0);

        if (rank == owner) {
            const int start = col_first[k];
            const int count = nt - k;
            // Bring current column k (updated) back to host for POTRF/TRSM.
            CUDA_CHECK(cudaMemcpy(h_tiles.data() + static_cast<size_t>(start) * bs * bs,
                                  d_tiles + static_cast<size_t>(start) * bs * bs,
                                  static_cast<size_t>(count) * bs * bs * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            double* Akk = h_tiles.data() + static_cast<size_t>(start) * bs * bs; // tile (k,k)
            if (!tile_potrf_lower(Akk, bk, bs)) {
                ok_local = 0;
            } else {
                // TRSM for tiles below the diagonal in the panel.
                for (int ii = k + 1; ii < nt; ++ii) {
                    const int bi = tile_dim(ii, nn, bs);
                    const int idx = local_index[ii * nt + k];
                    double* Aik = h_tiles.data() + static_cast<size_t>(idx) * bs * bs;
                    tile_trsm_right_upper_trans(Aik, Akk, bi, bk, bs);
                }
            }

            // Push updated panel column back to device.
            CUDA_CHECK(cudaMemcpy(d_tiles + static_cast<size_t>(start) * bs * bs,
                                  h_tiles.data() + static_cast<size_t>(start) * bs * bs,
                                  static_cast<size_t>(count) * bs * bs * sizeof(double),
                                  cudaMemcpyHostToDevice));

            // Pack panel tiles (i,k) for i=k..nt-1 into the broadcast buffer.
            for (int ii = k; ii < nt; ++ii) {
                const int idx = local_index[ii * nt + k];
                const double* src = h_tiles.data() + static_cast<size_t>(idx) * bs * bs;
                std::memcpy(panel.data() + static_cast<size_t>(ii - k) * bs * bs,
                            src,
                            static_cast<size_t>(bs) * bs * sizeof(double));
            }
        }

        int ok_all = 0;
        MPI_Allreduce(&ok_local, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok_all) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Broadcast the panel to all ranks.
        MPI_Bcast(panel.data(), static_cast<int>(panel.size()), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Upload panel to GPU for trailing updates.
        CUDA_CHECK(cudaMemcpy(d_panel, panel.data(), panel.size() * sizeof(double), cudaMemcpyHostToDevice));

        // Trailing update for locally-owned columns j > k
        for (int jj_idx = 0; jj_idx < (int)local_columns.size(); ++jj_idx) {
            const int j = local_columns[jj_idx];
            if (j <= k) continue;
            const int bj = tile_dim(j, nn, bs);
            const double* dLjk = d_panel + static_cast<size_t>(j - k) * bs * bs;

            for (int i = j; i < nt; ++i) {
                const int bi = tile_dim(i, nn, bs);
                const double* dLik = d_panel + static_cast<size_t>(i - k) * bs * bs;
                const int idx = local_index[i * nt + j];
                double* dC = d_tiles + static_cast<size_t>(idx) * bs * bs;
                launch_gemm_update(dC, dLik, dLjk, bi, bj, bk, bs);
            }
        }

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    if (rank == 0) {
        const long ms = static_cast<long>((t1 - t0) * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double gflops = ops / (std::max(1e-9, (t1 - t0))) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather final tiles back to rank 0, assemble full L, and optionally print/validate.
    CUDA_CHECK(cudaMemcpy(h_tiles.data(), d_tiles, static_cast<size_t>(sendcounts[rank]) * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> recvbuf;
    if (rank == 0) {
        recvbuf.resize(static_cast<size_t>(displs[size - 1] + sendcounts[size - 1]));
    }

    MPI_Gatherv(h_tiles.data(), sendcounts[rank], MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Assemble lower-triangular result into A_full
        std::fill(A_full.begin(), A_full.end(), 0.0);
        for (int r = 0; r < size; ++r) {
            size_t off = static_cast<size_t>(displs[r]);
            for (int j = r; j < nt; j += size) {
                const int bj = tile_dim(j, nn, bs);
                for (int i = j; i < nt; ++i) {
                    const int bi = tile_dim(i, nn, bs);
                    const double* tile = recvbuf.data() + off;
                    for (int rr = 0; rr < bi; ++rr) {
                        const int gr = i * bs + rr;
                        for (int cc = 0; cc < bj; ++cc) {
                            const int gc = j * bs + cc;
                            A_full[static_cast<size_t>(gr) * n + static_cast<size_t>(gc)] = tile[rr * bs + cc];
                        }
                    }
                    off += static_cast<size_t>(bs) * bs;
                }
            }
        }
        // Ensure upper triangle is explicitly zeroed.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A_full[i * n + j] = 0.0;
            }
        }

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
            CUDA_CHECK(cudaFree(d_panel));
            CUDA_CHECK(cudaFree(d_tiles));
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(d_panel));
    CUDA_CHECK(cudaFree(d_tiles));
    MPI_Finalize();
    return 0;
}
