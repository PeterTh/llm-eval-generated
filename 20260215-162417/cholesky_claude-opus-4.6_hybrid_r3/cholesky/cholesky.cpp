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

// Hybrid MPI + OpenMP + CUDA tiled Cholesky decomposition
// MPI: 1D cyclic distribution of tile rows across ranks
// OpenMP: parallel tile operations within each rank
// CUDA: tile BLAS operations via cuBLAS/cuSOLVER

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t s = (call); \
    if (s != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

static constexpr size_t TILE_SIZE = 256;

// Generate a symmetric positive definite matrix (deterministic, identical on all ranks)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
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

// Extract tile (ti,tj) from row-major matrix into column-major tile storage
static void extractTile(const double* A, size_t n, double* tile,
                        size_t ti, size_t tj, size_t nb) {
    size_t rs = ti * nb, cs = tj * nb;
    size_t rows = std::min(nb, n - rs);
    size_t cols = std::min(nb, n - cs);
    memset(tile, 0, nb * nb * sizeof(double));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            tile[j * nb + i] = A[(rs + i) * n + (cs + j)];
}

// Insert column-major tile back into row-major matrix
static void insertTile(double* A, size_t n, const double* tile,
                       size_t ti, size_t tj, size_t nb) {
    size_t rs = ti * nb, cs = tj * nb;
    size_t rows = std::min(nb, n - rs);
    size_t cols = std::min(nb, n - cs);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            A[(rs + i) * n + (cs + j)] = tile[j * nb + i];
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // CUDA device setup: assign GPU based on local rank
    int ndev;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    MPI_Comm lcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &lcomm);
    int lrank;
    MPI_Comm_rank(lcomm, &lrank);
    MPI_Comm_free(&lcomm);
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    int gpu_id = lrank % ndev;

    int nthreads = omp_get_max_threads();
    size_t nb = std::min(TILE_SIZE, n);
    size_t nt = (n + nb - 1) / nb;
    size_t te = nb * nb;
    size_t tb = te * sizeof(double);

    // Per-thread CUDA handles and streams
    std::vector<cublasHandle_t> cub(nthreads);
    std::vector<cusolverDnHandle_t> cus(nthreads);
    std::vector<cudaStream_t> str(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        CUDA_CHECK(cudaStreamCreate(&str[t]));
        CUBLAS_CHECK(cublasCreate(&cub[t]));
        CUBLAS_CHECK(cublasSetStream(cub[t], str[t]));
        CUSOLVER_CHECK(cusolverDnCreate(&cus[t]));
        CUSOLVER_CHECK(cusolverDnSetStream(cus[t], str[t]));
    }

    // Per-thread GPU tile buffers
    struct GRes { double *da, *db, *dc, *dw; int *di; };
    std::vector<GRes> gr(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        CUDA_CHECK(cudaMalloc(&gr[t].da, tb));
        CUDA_CHECK(cudaMalloc(&gr[t].db, tb));
        CUDA_CHECK(cudaMalloc(&gr[t].dc, tb));
        CUDA_CHECK(cudaMalloc(&gr[t].di, sizeof(int)));
    }

    // POTRF workspace query and allocation
    int potrf_lw = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cus[0], CUBLAS_FILL_MODE_LOWER, (int)nb, gr[0].da, (int)nb, &potrf_lw));
    for (int t = 0; t < nthreads; ++t)
        CUDA_CHECK(cudaMalloc(&gr[t].dw, std::max(1, potrf_lw) * (int)sizeof(double)));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d, OpenMP threads: %d, Tile size: %zu\n", nprocs, nthreads, nb);
    }

    // Generate matrix (deterministic, all ranks get identical result)
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate && rank == 0) {
        A_orig = A;
    }

    // Extract tiles from row-major matrix into column-major tile storage
    std::vector<std::vector<double>> TT(nt * nt, std::vector<double>(te, 0.0));
    for (size_t ti = 0; ti < nt; ++ti)
        for (size_t tj = 0; tj <= ti; ++tj)
            extractTile(A.data(), n, TT[ti * nt + tj].data(), ti, tj, nb);

    // Free full matrix to save memory
    A.clear();
    A.shrink_to_fit();

    // Tiled Cholesky: MPI distributes tile rows cyclically, OpenMP parallelizes
    // within each rank, CUDA accelerates tile BLAS operations
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    int fail = 0;

    for (size_t k = 0; k < nt && !fail; ++k) {
        int nbk = (int)std::min(nb, n - k * nb);
        int ownk = (int)(k % (size_t)nprocs);

        // Step 1: POTRF on diagonal tile (k,k)
        if (rank == ownk) {
            auto& r = gr[0];
            CUDA_CHECK(cudaMemcpy(r.da, TT[k * nt + k].data(), tb, cudaMemcpyHostToDevice));
            CUSOLVER_CHECK(cusolverDnDpotrf(cus[0], CUBLAS_FILL_MODE_LOWER,
                nbk, r.da, (int)nb, r.dw, potrf_lw, r.di));
            CUDA_CHECK(cudaMemcpy(TT[k * nt + k].data(), r.da, tb, cudaMemcpyDeviceToHost));
            int info;
            CUDA_CHECK(cudaMemcpy(&info, r.di, sizeof(int), cudaMemcpyDeviceToHost));
            if (info != 0) {
                fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", k * nb);
                fail = 1;
            }
        }

        MPI_Bcast(&fail, 1, MPI_INT, ownk, MPI_COMM_WORLD);
        if (fail) break;
        MPI_Bcast(TT[k * nt + k].data(), (int)te, MPI_DOUBLE, ownk, MPI_COMM_WORLD);

        // Step 2: TRSM on tiles (i,k) for i > k, parallelized with OpenMP
        #pragma omp parallel
        {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            int tid = omp_get_thread_num();
            auto& r = gr[tid];

            // Each thread uploads L(k,k) to its own GPU buffer
            CUDA_CHECK(cudaMemcpyAsync(r.da, TT[k * nt + k].data(), tb,
                cudaMemcpyHostToDevice, str[tid]));

            #pragma omp for schedule(dynamic)
            for (size_t i = k + 1; i < nt; ++i) {
                if ((int)(i % (size_t)nprocs) != rank) continue;
                int nbi = (int)std::min(nb, n - i * nb);

                CUDA_CHECK(cudaMemcpyAsync(r.db, TT[i * nt + k].data(), tb,
                    cudaMemcpyHostToDevice, str[tid]));

                // Solve X * L(k,k)^T = A(i,k) for X
                double one = 1.0;
                CUBLAS_CHECK(cublasDtrsm(cub[tid],
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    nbi, nbk, &one, r.da, (int)nb, r.db, (int)nb));

                CUDA_CHECK(cudaMemcpyAsync(TT[i * nt + k].data(), r.db, tb,
                    cudaMemcpyDeviceToHost, str[tid]));
                CUDA_CHECK(cudaStreamSynchronize(str[tid]));
            }
        }

        // Broadcast updated column k tiles from their owners
        for (size_t i = k + 1; i < nt; ++i)
            MPI_Bcast(TT[i * nt + k].data(), (int)te, MPI_DOUBLE,
                      (int)(i % (size_t)nprocs), MPI_COMM_WORLD);

        // Step 3: Update trailing submatrix with OpenMP + CUDA
        #pragma omp parallel
        {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            #pragma omp for schedule(dynamic)
        for (size_t i = k + 1; i < nt; ++i) {
            if ((int)(i % (size_t)nprocs) != rank) continue;

            int tid = omp_get_thread_num();
            auto& r = gr[tid];
            int nbi = (int)std::min(nb, n - i * nb);
            double alpha = -1.0, beta = 1.0;

            // Upload A(i,k) once for this tile row
            CUDA_CHECK(cudaMemcpyAsync(r.da, TT[i * nt + k].data(), tb,
                cudaMemcpyHostToDevice, str[tid]));

            // SYRK: A(i,i) -= A(i,k) * A(i,k)^T
            CUDA_CHECK(cudaMemcpyAsync(r.dc, TT[i * nt + i].data(), tb,
                cudaMemcpyHostToDevice, str[tid]));
            CUBLAS_CHECK(cublasDsyrk(cub[tid], CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                nbi, nbk, &alpha, r.da, (int)nb, &beta, r.dc, (int)nb));
            CUDA_CHECK(cudaMemcpyAsync(TT[i * nt + i].data(), r.dc, tb,
                cudaMemcpyDeviceToHost, str[tid]));

            // GEMM: A(i,j) -= A(i,k) * A(j,k)^T for j in (k, i)
            for (size_t j = k + 1; j < i; ++j) {
                int nbj = (int)std::min(nb, n - j * nb);
                CUDA_CHECK(cudaMemcpyAsync(r.db, TT[j * nt + k].data(), tb,
                    cudaMemcpyHostToDevice, str[tid]));
                CUDA_CHECK(cudaMemcpyAsync(r.dc, TT[i * nt + j].data(), tb,
                    cudaMemcpyHostToDevice, str[tid]));
                CUBLAS_CHECK(cublasDgemm(cub[tid], CUBLAS_OP_N, CUBLAS_OP_T,
                    nbi, nbj, nbk, &alpha, r.da, (int)nb, r.db, (int)nb,
                    &beta, r.dc, (int)nb));
                CUDA_CHECK(cudaMemcpyAsync(TT[i * nt + j].data(), r.dc, tb,
                    cudaMemcpyDeviceToHost, str[tid]));
            }

            CUDA_CHECK(cudaStreamSynchronize(str[tid]));
        }
        } // end omp parallel
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    if (fail) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        for (int t = 0; t < nthreads; ++t) {
            cudaFree(gr[t].da); cudaFree(gr[t].db); cudaFree(gr[t].dc);
            cudaFree(gr[t].dw); cudaFree(gr[t].di);
            cublasDestroy(cub[t]); cusolverDnDestroy(cus[t]); cudaStreamDestroy(str[t]);
        }
        MPI_Finalize();
        return 1;
    }

    // Gather all tiles to rank 0
    for (size_t i = 0; i < nt; ++i) {
        int own = (int)(i % (size_t)nprocs);
        if (own == 0) continue;
        for (size_t j = 0; j <= i; ++j) {
            if (rank == own)
                MPI_Send(TT[i * nt + j].data(), (int)te, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            else if (rank == 0)
                MPI_Recv(TT[i * nt + j].data(), (int)te, MPI_DOUBLE, own, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    int ret = 0;

    if (rank == 0) {
        // Reconstruct full result matrix from column-major tiles
        std::vector<double> result(n * n, 0.0);
        for (size_t ti = 0; ti < nt; ++ti)
            for (size_t tj = 0; tj <= ti; ++tj)
                insertTile(result.data(), n, TT[ti * nt + tj].data(), ti, tj, nb);

        // Zero upper triangular part
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                result[i * n + j] = 0.0;

        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(result, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(result, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    if (validate) {
        MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Cleanup GPU resources
    for (int t = 0; t < nthreads; ++t) {
        cudaFree(gr[t].da); cudaFree(gr[t].db); cudaFree(gr[t].dc);
        cudaFree(gr[t].dw); cudaFree(gr[t].di);
        cublasDestroy(cub[t]); cusolverDnDestroy(cus[t]); cudaStreamDestroy(str[t]);
    }

    MPI_Finalize();
    return ret;
}
