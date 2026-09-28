#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked (right-looking) Cholesky decomposition.
//
// The matrix is partitioned into column-panels of width NB, which are
// distributed cyclically (column j -> rank j % nranks) across MPI ranks.
// Each rank owns its panels in device (GPU) memory. For step k of the
// right-looking algorithm the owner of panel k factorizes the diagonal
// block (cuSOLVER potrf) and solves the panel below it (cuBLAS trsm) on
// its assigned GPU, then broadcasts the panel to every rank via MPI. Each
// rank then applies the trailing update (cuBLAS gemm) to the panels it
// owns; independent updates within a step are dispatched concurrently
// across CUDA streams using OpenMP threads.
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, \
                    (int)st_);                                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define CUSOLVER_CHECK(call)                                                       \
    do {                                                                          \
        cusolverStatus_t st_ = (call);                                            \
        if (st_ != CUSOLVER_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuSOLVER error %s:%d: status %d\n", __FILE__, __LINE__, \
                    (int)st_);                                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

namespace {

constexpr int kNumStreams = 4;

// Block-panel geometry helpers for a column-cyclic distribution with panel width NB.
inline size_t numBlocks(size_t n, size_t NB) { return (n + NB - 1) / NB; }
inline size_t rowStart(size_t j, size_t NB) { return j * NB; }
inline size_t rowCount(size_t j, size_t n, size_t NB) { return n - j * NB; }
inline size_t colWidth(size_t j, size_t n, size_t NB) {
    return std::min(NB, n - j * NB);
}

// Distributed hybrid MPI+OpenMP+CUDA blocked Cholesky decomposition.
// On entry/exit, `A` (row-major, n x n) holds the full matrix only on rank 0;
// on other ranks it is unused. Returns the same success value on every rank.
bool choleskyDecompositionDistributed(std::vector<double>& A, const size_t n, int rank,
                                       int nranks, MPI_Comm comm) {
    if (n == 0) return true;

    // Assign one GPU per rank (round-robin across GPUs visible on the local node).
    MPI_Comm localComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    const int deviceId = deviceCount > 0 ? (localRank % deviceCount) : 0;
    CUDA_CHECK(cudaSetDevice(deviceId));

    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    cudaStream_t streams[kNumStreams];
    cublasHandle_t gemmHandles[kNumStreams];
    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUBLAS_CHECK(cublasCreate(&gemmHandles[s]));
        CUBLAS_CHECK(cublasSetStream(gemmHandles[s], streams[s]));
    }

    const size_t NB = std::min<size_t>(512, n);
    const size_t nb = numBlocks(n, NB);

    // Panels owned by this rank (cyclic distribution: owner(j) = j % nranks).
    std::vector<size_t> owned;
    for (size_t j = 0; j < nb; ++j) {
        if (static_cast<int>(j % static_cast<size_t>(nranks)) == rank) owned.push_back(j);
    }

    std::unordered_map<size_t, double*> deviceCol;
    for (size_t j : owned) {
        const size_t cnt = rowCount(j, n, NB) * colWidth(j, n, NB);
        double* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, cnt * sizeof(double)));
        deviceCol[j] = d;
    }

    // --- Distribute panels from rank 0 (row-major) into column-major device buffers ---
    for (size_t j = 0; j < nb; ++j) {
        const size_t ld = rowCount(j, n, NB);
        const size_t cw = colWidth(j, n, NB);
        const size_t cnt = ld * cw;
        const int owner = static_cast<int>(j % static_cast<size_t>(nranks));
        const size_t rs = rowStart(j, NB);

        if (rank == 0) {
            std::vector<double> buf(cnt);
#pragma omp parallel for schedule(static)
            for (size_t c = 0; c < cw; ++c) {
                for (size_t r = 0; r < ld; ++r) {
                    buf[c * ld + r] = A[(rs + r) * n + (rs + c)];
                }
            }
            if (owner == 0) {
                CUDA_CHECK(cudaMemcpy(deviceCol[j], buf.data(), cnt * sizeof(double),
                                       cudaMemcpyHostToDevice));
            } else {
                MPI_Send(buf.data(), static_cast<int>(cnt), MPI_DOUBLE, owner,
                         static_cast<int>(j), comm);
            }
        } else if (rank == owner) {
            std::vector<double> buf(cnt);
            MPI_Recv(buf.data(), static_cast<int>(cnt), MPI_DOUBLE, 0, static_cast<int>(j), comm,
                     MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(deviceCol[j], buf.data(), cnt * sizeof(double),
                                   cudaMemcpyHostToDevice));
        }
    }

    // Scratch buffers used to broadcast the "current" panel each step.
    const size_t maxCnt = rowCount(0, n, NB) * colWidth(0, n, NB);
    std::vector<double> hostBcast(maxCnt);
    double* devBcast = nullptr;
    CUDA_CHECK(cudaMalloc(&devBcast, maxCnt * sizeof(double)));

    double* devWork = nullptr;
    int curLwork = 0;
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    bool overallSuccess = true;

    for (size_t k = 0; k < nb; ++k) {
        const int ownerK = static_cast<int>(k % static_cast<size_t>(nranks));
        const size_t ldK = rowCount(k, n, NB);
        const size_t cwK = colWidth(k, n, NB);

        int infoFlag = 1;

        if (rank == ownerK) {
            double* dK = deviceCol[k];

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                                        static_cast<int>(cwK), dK,
                                                        static_cast<int>(ldK), &lwork));
            if (lwork > curLwork) {
                if (devWork) CUDA_CHECK(cudaFree(devWork));
                CUDA_CHECK(cudaMalloc(&devWork, sizeof(double) * lwork));
                curLwork = lwork;
            }

            CUSOLVER_CHECK(cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER,
                                             static_cast<int>(cwK), dK, static_cast<int>(ldK),
                                             devWork, curLwork, devInfo));

            int hInfo = 0;
            CUDA_CHECK(cudaMemcpy(&hInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

            if (hInfo != 0) {
                infoFlag = 0;
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       k * NB + static_cast<size_t>(hInfo - 1));
            } else if (ldK > cwK) {
                const double alpha = 1.0;
                CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                         static_cast<int>(ldK - cwK), static_cast<int>(cwK),
                                         &alpha, dK, static_cast<int>(ldK), dK + cwK,
                                         static_cast<int>(ldK)));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        MPI_Bcast(&infoFlag, 1, MPI_INT, ownerK, comm);
        if (!infoFlag) {
            overallSuccess = false;
            break;
        }

        // Broadcast panel k to every rank so trailing updates can be applied locally.
        double* colKDev;
        if (nranks == 1) {
            colKDev = deviceCol[k];
        } else {
            if (rank == ownerK) {
                CUDA_CHECK(cudaMemcpy(hostBcast.data(), deviceCol[k], ldK * cwK * sizeof(double),
                                       cudaMemcpyDeviceToHost));
            }
            MPI_Bcast(hostBcast.data(), static_cast<int>(ldK * cwK), MPI_DOUBLE, ownerK, comm);
            if (rank == ownerK) {
                colKDev = deviceCol[k];
            } else {
                CUDA_CHECK(cudaMemcpy(devBcast, hostBcast.data(), ldK * cwK * sizeof(double),
                                       cudaMemcpyHostToDevice));
                colKDev = devBcast;
            }
        }

        // Trailing update: for every panel j > k owned by this rank,
        // colDev[j] -= L(j:end, k) * L(j, k)^T, applied with a single GEMM per panel.
        std::vector<size_t> ownedAfterK;
        for (size_t j : owned) {
            if (j > k) ownedAfterK.push_back(j);
        }

        const int nthreads = std::max(1, std::min<int>(kNumStreams,
                                                         static_cast<int>(ownedAfterK.size())));
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
        for (size_t idx = 0; idx < ownedAfterK.size(); ++idx) {
            // Each OpenMP worker thread has its own CUDA device context stack,
            // so the device selected on the main thread must be re-applied here.
            cudaSetDevice(deviceId);

            const size_t j = ownedAfterK[idx];
            const int sid = omp_get_thread_num() % kNumStreams;

            const size_t offsetRows = (j - k) * NB;
            const size_t m = rowCount(j, n, NB);
            const size_t cwJ = colWidth(j, n, NB);

            double* aPtr = colKDev + offsetRows;
            double* bPtr = colKDev + offsetRows;
            const double alpha = -1.0;
            const double beta = 1.0;

            CUBLAS_CHECK(cublasDgemm(gemmHandles[sid], CUBLAS_OP_N, CUBLAS_OP_T,
                                      static_cast<int>(m), static_cast<int>(cwJ),
                                      static_cast<int>(cwK), &alpha, aPtr,
                                      static_cast<int>(ldK), bPtr, static_cast<int>(ldK), &beta,
                                      deviceCol[j], static_cast<int>(m)));
        }
        for (int s = 0; s < kNumStreams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
    }

    if (overallSuccess) {
        // --- Gather panels back to rank 0 and reconstruct the row-major result ---
        for (size_t j = 0; j < nb; ++j) {
            const size_t ld = rowCount(j, n, NB);
            const size_t cw = colWidth(j, n, NB);
            const size_t cnt = ld * cw;
            const int owner = static_cast<int>(j % static_cast<size_t>(nranks));
            const size_t rs = rowStart(j, NB);

            std::vector<double> buf;
            if (rank == 0) {
                buf.resize(cnt);
                if (owner == 0) {
                    CUDA_CHECK(cudaMemcpy(buf.data(), deviceCol[j], cnt * sizeof(double),
                                           cudaMemcpyDeviceToHost));
                } else {
                    MPI_Recv(buf.data(), static_cast<int>(cnt), MPI_DOUBLE, owner,
                             static_cast<int>(j), comm, MPI_STATUS_IGNORE);
                }
#pragma omp parallel for schedule(static)
                for (size_t c = 0; c < cw; ++c) {
                    for (size_t r = 0; r < ld; ++r) {
                        A[(rs + r) * n + (rs + c)] = buf[c * ld + r];
                    }
                }
            } else if (rank == owner) {
                buf.resize(cnt);
                CUDA_CHECK(cudaMemcpy(buf.data(), deviceCol[j], cnt * sizeof(double),
                                       cudaMemcpyDeviceToHost));
                MPI_Send(buf.data(), static_cast<int>(cnt), MPI_DOUBLE, 0, static_cast<int>(j),
                         comm);
            }
        }

        if (rank == 0) {
            // Zero out the upper triangular part, matching the reference semantics.
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A[i * n + j] = 0.0;
                }
            }
        }
    }

    // --- Cleanup ---
    for (auto& kv : deviceCol) cudaFree(kv.second);
    if (devWork) cudaFree(devWork);
    cudaFree(devInfo);
    cudaFree(devBcast);
    for (int s = 0; s < kNumStreams; ++s) {
        cublasDestroy(gemmHandles[s]);
        cudaStreamDestroy(streams[s]);
    }
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    return overallSuccess;
}

}  // namespace

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
    // Parallelized over independent output rows; per-element accumulation order
    // (the inner k loop) is unchanged, so results are bit-identical to the
    // sequential reference implementation.
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool showHelp = false;
    bool badArg = false;
    const char* badArgStr = nullptr;

    // Parse command line arguments (identical on every rank, given identical argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            badArg = true;
            badArgStr = argv[i];
            break;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (badArg) {
        if (rank == 0) {
            printf("Unknown option: %s\n", badArgStr);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI (%d ranks) + OpenMP (%d threads) + CUDA\n", nranks,
               omp_get_max_threads());
    }

    // Full matrix is only materialized on rank 0; other ranks receive panels via MPI.
    std::vector<double> A;
    std::vector<double> A_orig;

    if (rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);

        if (validate) {
            A_orig = A;  // Save original for validation
        }
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionDistributed(A, n, rank, nranks, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
