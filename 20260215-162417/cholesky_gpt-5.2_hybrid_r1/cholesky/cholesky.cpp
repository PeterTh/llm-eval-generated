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
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static inline void die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    std::abort();
}

static inline void mpi_check(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "MPI error in %s\n", what);
        std::abort();
    }
}

static inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

static inline void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error in %s\n", what);
        std::abort();
    }
}

static inline void cusolver_check(cusolverStatus_t st, const char* what) {
    if (st != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER error in %s\n", what);
        std::abort();
    }
}

// Hybrid MPI + OpenMP + CUDA blocked Cholesky (left-looking, GPU BLAS)
// - MPI: block-row distribution aligned to block size
// - CUDA: POTRF (cuSOLVER) + TRSM/GEMM (cuBLAS)
// - OpenMP: CPU-side packing/conversion/validation

struct BlockRowDist {
    int world = 1;
    int rank = 0;
    int nb = 256;
    int n = 0;
    int numBlocks = 0;

    std::vector<int> blocksPerRank;
    std::vector<int> blockStartRank; // prefix blocks start

    void init(int n_, int nb_, int world_, int rank_) {
        n = n_;
        nb = nb_;
        world = world_;
        rank = rank_;
        numBlocks = (n + nb - 1) / nb;

        blocksPerRank.assign(world, 0);
        const int base = numBlocks / world;
        const int rem = numBlocks % world;
        for (int r = 0; r < world; ++r) {
            blocksPerRank[r] = base + (r < rem ? 1 : 0);
        }
        blockStartRank.assign(world + 1, 0);
        for (int r = 0; r < world; ++r) blockStartRank[r + 1] = blockStartRank[r] + blocksPerRank[r];
    }

    int owner_of_block(int b) const {
        // contiguous block assignment by rank
        // find r such that blockStartRank[r] <= b < blockStartRank[r+1]
        int lo = 0, hi = world;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (blockStartRank[mid] <= b) lo = mid; else hi = mid;
        }
        if (lo >= world) lo = world - 1;
        while (lo + 1 < world && blockStartRank[lo + 1] <= b) ++lo;
        return lo;
    }

    int local_row_start() const {
        const int firstBlock = blockStartRank[rank];
        return std::min(n, firstBlock * nb);
    }

    int local_row_end() const {
        const int lastBlockExcl = blockStartRank[rank + 1];
        return std::min(n, lastBlockExcl * nb);
    }

    int local_rows() const { return local_row_end() - local_row_start(); }
};

static inline int min_int(int a, int b) { return a < b ? a : b; }

static void rowmajor_local_to_colmajor_local(const double* A_rm_local, int n, int m, double* A_cm) {
    // A_rm_local is m x n row-major; A_cm is m x n column-major (ld=m)
    const int ld = m;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A_cm[i + j * ld] = A_rm_local[i * n + j];
        }
    }
}

static void colmajor_local_to_rowmajor(const double* A_cm, int n, int m, double* A_rm_out) {
    const int ld = m;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A_rm_out[i * n + j] = A_cm[i + j * ld];
        }
    }
}

static bool cholesky_mpi_omp_cuda(double* A_local_cm, int n, const BlockRowDist& dist, double& time_ms) {
    const int rank = dist.rank;
    const int world = dist.world;
    const int nb = dist.nb;
    const int row0 = dist.local_row_start();
    const int mloc = dist.local_rows();
    const int ld = mloc;

    int deviceCount = 0;
    cuda_check(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) die("No CUDA devices found");
    cuda_check(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cublas_check(cublasCreate(&cublas), "cublasCreate");
    cusolver_check(cusolverDnCreate(&cusolver), "cusolverDnCreate");

    // Copy local matrix to device
    double* dA = nullptr;
    cuda_check(cudaMalloc((void**)&dA, sizeof(double) * (size_t)ld * (size_t)n), "cudaMalloc dA");
    cuda_check(cudaMemcpy(dA, A_local_cm, sizeof(double) * (size_t)ld * (size_t)n, cudaMemcpyHostToDevice), "H2D A");

    // Work buffers
    double* dDiag = nullptr;
    cuda_check(cudaMalloc((void**)&dDiag, sizeof(double) * (size_t)nb * (size_t)nb), "cudaMalloc dDiag");
    int* dInfo = nullptr;
    cuda_check(cudaMalloc((void**)&dInfo, sizeof(int)), "cudaMalloc dInfo");

    std::vector<double> hDiag((size_t)nb * (size_t)nb);

    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier start");
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < dist.numBlocks; ++b) {
        const int k0 = b * nb;
        const int bk = min_int(nb, n - k0);
        const int owner = dist.owner_of_block(b);

        // 1) Factor diagonal block on owner rank (in-place on local device storage)
        if (rank == owner) {
            const int loc_k0 = k0 - row0;
            double* dAkk = dA + (size_t)k0 * (size_t)ld + (size_t)loc_k0;

            int lwork = 0;
            cusolver_check(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, bk, dAkk, ld, &lwork),
                           "potrf_bufferSize");
            double* dWork = nullptr;
            cuda_check(cudaMalloc((void**)&dWork, sizeof(double) * (size_t)lwork), "cudaMalloc potrf work");
            cusolver_check(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, bk, dAkk, ld, dWork, lwork, dInfo),
                           "potrf");
            cuda_check(cudaFree(dWork), "cudaFree potrf work");

            int info_h = 0;
            cuda_check(cudaMemcpy(&info_h, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "D2H info");
            if (info_h != 0) {
                fprintf(stderr, "POTRF failed at block %d (info=%d)\n", b, info_h);
                cuda_check(cudaFree(dA), "cudaFree dA");
                cuda_check(cudaFree(dDiag), "cudaFree dDiag");
                cuda_check(cudaFree(dInfo), "cudaFree dInfo");
                cublasDestroy(cublas);
                cusolverDnDestroy(cusolver);
                return false;
            }

            // Pack diag block (column-major contiguous) for MPI broadcast
            for (int j = 0; j < bk; ++j) {
                cuda_check(cudaMemcpy(hDiag.data() + (size_t)j * (size_t)bk,
                                      dAkk + (size_t)j * (size_t)ld,
                                      sizeof(double) * (size_t)bk,
                                      cudaMemcpyDeviceToHost),
                           "D2H diag col");
            }
        }

        mpi_check(MPI_Bcast(hDiag.data(), bk * bk, MPI_DOUBLE, owner, MPI_COMM_WORLD), "MPI_Bcast diag");

        // Copy diag to device (all ranks)
        cuda_check(cudaMemcpy(dDiag, hDiag.data(), sizeof(double) * (size_t)bk * (size_t)bk, cudaMemcpyHostToDevice), "H2D diag");

        // 2) TRSM for rows below diagonal on each rank: A(trail_row0:n, k0:k0+bk) = A * inv(Lkk^T)
        const int trail_row0 = k0 + bk;
        const int trail_rows_global = n - trail_row0;
        if (trail_rows_global > 0) {
            const int my_trail_start = std::max(trail_row0, row0);
            const int my_trail_end = std::min(n, row0 + mloc);
            const int my_trail_rows = std::max(0, my_trail_end - my_trail_start);
            const int loc_i0 = my_trail_start - row0;

            if (my_trail_rows > 0) {
                const double alpha = 1.0;
                cublas_check(cublasDtrsm(cublas,
                                         CUBLAS_SIDE_RIGHT,
                                         CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_T,
                                         CUBLAS_DIAG_NON_UNIT,
                                         my_trail_rows, bk,
                                         &alpha,
                                         dDiag, bk,
                                         dA + (size_t)k0 * (size_t)ld + (size_t)loc_i0, ld),
                             "cublasDtrsm");
            }

            // 3) Gather global panel rows [trail_row0, n) for columns [k0, k0+bk) as row-major (one Allgatherv)
            std::vector<double> myPanel_rm((size_t)my_trail_rows * (size_t)bk);
            if (my_trail_rows > 0) {
                std::vector<double> myPanel_cm((size_t)my_trail_rows * (size_t)bk);
                for (int j = 0; j < bk; ++j) {
                    cuda_check(cudaMemcpy(myPanel_cm.data() + (size_t)j * (size_t)my_trail_rows,
                                          dA + (size_t)(k0 + j) * (size_t)ld + (size_t)loc_i0,
                                          sizeof(double) * (size_t)my_trail_rows,
                                          cudaMemcpyDeviceToHost),
                               "D2H panel part");
                }
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < my_trail_rows; ++i) {
                    for (int j = 0; j < bk; ++j) {
                        myPanel_rm[(size_t)i * (size_t)bk + (size_t)j] = myPanel_cm[(size_t)j * (size_t)my_trail_rows + (size_t)i];
                    }
                }
            }

            std::vector<int> recvcounts(world, 0), displs(world, 0);
            for (int r = 0; r < world; ++r) {
                BlockRowDist dtmp = dist;
                dtmp.rank = r;
                const int r0 = dtmp.local_row_start();
                const int r1 = dtmp.local_row_end();
                const int rs = std::max(trail_row0, r0);
                const int re = std::min(n, r1);
                const int rr = std::max(0, re - rs);
                recvcounts[r] = rr * bk;
            }
            for (int r = 1; r < world; ++r) displs[r] = displs[r - 1] + recvcounts[r - 1];

            std::vector<double> panel_rm((size_t)trail_rows_global * (size_t)bk);
            mpi_check(MPI_Allgatherv(myPanel_rm.data(), my_trail_rows * bk, MPI_DOUBLE,
                                    panel_rm.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                                    MPI_COMM_WORLD),
                      "MPI_Allgatherv panel");

            // Transpose to column-major for BLAS: panel_cm (trail_rows_global x bk, ld=trail_rows_global)
            std::vector<double> panel_cm((size_t)trail_rows_global * (size_t)bk);
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < bk; ++j) {
                for (int i = 0; i < trail_rows_global; ++i) {
                    panel_cm[(size_t)i + (size_t)j * (size_t)trail_rows_global] = panel_rm[(size_t)i * (size_t)bk + (size_t)j];
                }
            }

            double* dPanel = nullptr;
            cuda_check(cudaMalloc((void**)&dPanel, sizeof(double) * (size_t)trail_rows_global * (size_t)bk), "cudaMalloc dPanel");
            cuda_check(cudaMemcpy(dPanel, panel_cm.data(), sizeof(double) * (size_t)trail_rows_global * (size_t)bk, cudaMemcpyHostToDevice), "H2D panel");

            // 4) Update trailing submatrix for my local rows: A(loc_i0:, trail_row0:) -= A(loc_i0:, k0:) * Panel^T
            if (my_trail_rows > 0) {
                const double alpha = -1.0;
                const double beta = 1.0;
                cublas_check(cublasDgemm(cublas,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         my_trail_rows, trail_rows_global, bk,
                                         &alpha,
                                         dA + (size_t)k0 * (size_t)ld + (size_t)loc_i0, ld,
                                         dPanel, trail_rows_global,
                                         &beta,
                                         dA + (size_t)trail_row0 * (size_t)ld + (size_t)loc_i0, ld),
                             "cublasDgemm update");
            }

            cuda_check(cudaFree(dPanel), "cudaFree dPanel");
        }

        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier iter");
    }

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize end");
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier end");
    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy back local matrix
    cuda_check(cudaMemcpy(A_local_cm, dA, sizeof(double) * (size_t)ld * (size_t)n, cudaMemcpyDeviceToHost), "D2H A");

    cuda_check(cudaFree(dA), "cudaFree dA");
    cuda_check(cudaFree(dDiag), "cudaFree dDiag");
    cuda_check(cudaFree(dInfo), "cudaFree dInfo");
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);

    const double local_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    double max_ms = 0.0;
    mpi_check(MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce max time");
    time_ms = (rank == 0) ? max_ms : 0.0;
    return true;
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
    
    // Compute A = B * B^T
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
    mpi_check(MPI_Init(&argc, &argv), "MPI_Init");

    int world = 1;
    int rank = 0;
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI_Comm_size");
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
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
        printf("MPI ranks: %d\n", world);
        printf("OpenMP threads (max): %d\n", omp_get_max_threads());
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int ni = (int)n;
    int nb = 256;
    if (ni < nb) nb = ni;
    if (nb < 1) nb = 1;

    BlockRowDist dist;
    dist.init(ni, nb, world, rank);
    const int row0 = dist.local_row_start();
    const int mloc = dist.local_rows();

    // Rank0 generates the matrix (preserve original semantics)
    std::vector<double> A_full;
    std::vector<double> A_orig;
    if (rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) A_orig = A_full;
    }

    // Scatter rows to ranks (row-major), then convert locally to column-major for CUDA
    std::vector<int> sendcounts(world, 0), displs(world, 0);
    for (int r = 0; r < world; ++r) {
        BlockRowDist dtmp = dist;
        dtmp.rank = r;
        const int rs = dtmp.local_row_start();
        const int re = dtmp.local_row_end();
        sendcounts[r] = (re - rs) * ni;
    }
    for (int r = 1; r < world; ++r) displs[r] = displs[r - 1] + sendcounts[r - 1];

    std::vector<double> A_local_rm((size_t)mloc * (size_t)ni);
    mpi_check(MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                          A_local_rm.data(), mloc * ni, MPI_DOUBLE, 0, MPI_COMM_WORLD),
              "MPI_Scatterv A");

    std::vector<double> A_local_cm((size_t)mloc * (size_t)ni);
    rowmajor_local_to_colmajor_local(A_local_rm.data(), ni, mloc, A_local_cm.data());

    if (rank == 0) {
        printf("Computing Cholesky decomposition (MPI+OpenMP+CUDA)...\n");
    }

    double time_ms = 0.0;
    bool success = cholesky_mpi_omp_cuda(A_local_cm.data(), ni, dist, time_ms);

    int ok = success ? 1 : 0;
    int ok_all = 0;
    mpi_check(MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce ok");
    if (!ok_all) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Convert back to row-major and gather to rank0
    std::vector<double> A_local_out_rm((size_t)mloc * (size_t)ni);
    colmajor_local_to_rowmajor(A_local_cm.data(), ni, mloc, A_local_out_rm.data());

    if (rank == 0) A_full.assign(n * n, 0.0);
    mpi_check(MPI_Gatherv(A_local_out_rm.data(), mloc * ni, MPI_DOUBLE,
                         rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                         0, MPI_COMM_WORLD),
              "MPI_Gatherv L");

    if (rank == 0) {
        // Zero upper triangular part (match original semantics)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ni; ++i) {
            for (int j = i + 1; j < ni; ++j) {
                A_full[(size_t)i * (size_t)ni + (size_t)j] = 0.0;
            }
        }

        printf("Computation time: %.3f ms\n", time_ms);
        double ops = (double)n * (double)n * (double)n / 3.0;
        double gflops = ops / (time_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A_full, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
