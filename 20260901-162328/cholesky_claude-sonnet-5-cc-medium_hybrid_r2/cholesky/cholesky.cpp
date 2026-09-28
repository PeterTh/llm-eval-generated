#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition.
//
// The n x n matrix is tiled into NB x NB blocks. Lower-triangular blocks
// (i,j) with j<=i are distributed across MPI ranks using 1D row-cyclic
// ownership (owner(i) = i % nprocs), so each rank only ever stores the
// block-rows it owns (real distributed memory, not just distributed
// compute). Per outer step k (0..nblocks-1):
//   1) owner(k) factorizes the small diagonal block A(k,k) on the CPU.
//   2) L(k,k) is broadcast to all ranks.
//   3) Each rank computes L(i,k) = A(i,k) * L(k,k)^-T for its owned rows
//      i>k via cuBLAS DTRSM on its local GPU(s), fanned out with OpenMP.
//   4) The full column panel k is gathered to all ranks (MPI_Ibcast) and
//      staged on every local GPU.
//   5) Each rank updates its owned trailing blocks A(i,j) -= L(i,k)*L(j,k)^T
//      via cuBLAS DGEMM, again fanned out across local GPUs with OpenMP.
//
// cuBLAS is column-major while our blocks are stored row-major; rather than
// transposing data we exploit that a row-major (r x c) buffer read as
// column-major is exactly the transpose of the true matrix, and adjust the
// BLAS operation flags/dimensions accordingly (see comments at call sites).

#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                     \
        cublasStatus_t st__ = (call);                                        \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,         \
                    __FILE__, __LINE__);                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Used to factorize the small NB x NB diagonal blocks of the tiled algorithm.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                // Diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (size_t k = 0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (kept strictly sequential so the benchmark
    // matrix is reproducible/identical regardless of parallel configuration)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T. Each output cell is an independent, purely
    // sequential dot-product, so distributing the (i,j) cells across
    // OpenMP threads does not change any numerical result.
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

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (each cell independent -> safe to parallelize)
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

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max : maxError, relError) schedule(static)
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

// ---------------------------------------------------------------------
// Distributed tiled Cholesky infrastructure
// ---------------------------------------------------------------------

static inline size_t numBlocks(size_t n, size_t NB) { return (n + NB - 1) / NB; }
static inline size_t blockDim_(size_t idx, size_t n, size_t NB) {
    size_t start = idx * NB;
    return std::min(NB, n - start);
}
static inline int ownerOf(size_t i, int nprocs) { return static_cast<int>(i % static_cast<size_t>(nprocs)); }
static inline size_t blockKey(size_t i, size_t j, size_t nblocks) { return i * nblocks + j; }

// Per-rank GPU resources: one cuBLAS handle + persistent device buffers per
// local GPU. OpenMP threads (one per local GPU) drive these concurrently.
struct GpuResources {
    std::vector<int> devices;
    std::vector<cublasHandle_t> handles;
    std::vector<double*> d_panel; // holds the whole column panel for step k
    std::vector<double*> d_Lkk;   // holds broadcast diagonal block L(k,k)
    std::vector<double*> d_taskA; // scratch for DTRSM operand / output
    std::vector<double*> d_taskC; // scratch for DGEMM trailing-update block
    size_t maxPanelElems = 0;
    size_t maxBlockElems = 0;

    void init(int rank, int nprocs, size_t n, size_t NB) {
        int totalGpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&totalGpus));
        if (totalGpus <= 0) {
            fprintf(stderr, "No CUDA-capable device found on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int d = 0; d < totalGpus; ++d) {
            if (d % nprocs == rank) devices.push_back(d);
        }
        if (devices.empty()) {
            devices.push_back(rank % totalGpus);
        }

        maxPanelElems = n * NB; // upper bound on total elements in a column panel
        maxBlockElems = NB * NB;

        const size_t numLocal = devices.size();
        handles.resize(numLocal);
        d_panel.resize(numLocal);
        d_Lkk.resize(numLocal);
        d_taskA.resize(numLocal);
        d_taskC.resize(numLocal);

        for (size_t t = 0; t < numLocal; ++t) {
            CUDA_CHECK(cudaSetDevice(devices[t]));
            CUBLAS_CHECK(cublasCreate(&handles[t]));
            CUDA_CHECK(cudaMalloc(&d_panel[t], maxPanelElems * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Lkk[t], maxBlockElems * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_taskA[t], maxBlockElems * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_taskC[t], maxBlockElems * sizeof(double)));
        }
    }

    size_t numLocalGpus() const { return devices.size(); }

    void destroy() {
        for (size_t t = 0; t < devices.size(); ++t) {
            CUDA_CHECK(cudaSetDevice(devices[t]));
            cudaFree(d_panel[t]);
            cudaFree(d_Lkk[t]);
            cudaFree(d_taskA[t]);
            cudaFree(d_taskC[t]);
            cublasDestroy(handles[t]);
        }
    }
};

// Scatter lower-triangular blocks of `full` (only valid on rank 0) to their
// owning ranks. `blocks` is populated with this rank's owned blocks.
void distributeMatrix(const std::vector<double>& full, size_t n, size_t NB, size_t nblocks,
                       int rank, int nprocs,
                       std::unordered_map<size_t, std::vector<double>>& blocks) {
    for (size_t i = 0; i < nblocks; ++i) {
        const size_t bi = blockDim_(i, n, NB);
        const int owner = ownerOf(i, nprocs);
        for (size_t j = 0; j <= i; ++j) {
            const size_t bj = blockDim_(j, n, NB);
            const int tag = static_cast<int>(blockKey(i, j, nblocks) % 1000000007);

            if (rank == 0 && owner == 0) {
                std::vector<double> block(bi * bj);
                for (size_t r = 0; r < bi; ++r) {
                    std::memcpy(&block[r * bj], &full[(i * NB + r) * n + j * NB], bj * sizeof(double));
                }
                blocks.emplace(blockKey(i, j, nblocks), std::move(block));
            } else if (rank == 0) {
                std::vector<double> block(bi * bj);
                for (size_t r = 0; r < bi; ++r) {
                    std::memcpy(&block[r * bj], &full[(i * NB + r) * n + j * NB], bj * sizeof(double));
                }
                MPI_Send(block.data(), static_cast<int>(bi * bj), MPI_DOUBLE, owner, tag, MPI_COMM_WORLD);
            } else if (rank == owner) {
                std::vector<double> block(bi * bj);
                MPI_Recv(block.data(), static_cast<int>(bi * bj), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                blocks.emplace(blockKey(i, j, nblocks), std::move(block));
            }
        }
    }
}

// Gather lower-triangular owned blocks back into a full n x n matrix on rank 0
// (upper triangle stays zero, matching the reference implementation's output).
void gatherMatrix(std::vector<double>& full, size_t n, size_t NB, size_t nblocks,
                   int rank, int nprocs,
                   std::unordered_map<size_t, std::vector<double>>& blocks) {
    if (rank == 0) {
        full.assign(n * n, 0.0);
    }
    for (size_t i = 0; i < nblocks; ++i) {
        const size_t bi = blockDim_(i, n, NB);
        const int owner = ownerOf(i, nprocs);
        for (size_t j = 0; j <= i; ++j) {
            const size_t bj = blockDim_(j, n, NB);
            const int tag = static_cast<int>(blockKey(i, j, nblocks) % 1000000007);

            if (rank == 0 && owner == 0) {
                const std::vector<double>& block = blocks.at(blockKey(i, j, nblocks));
                for (size_t r = 0; r < bi; ++r) {
                    std::memcpy(&full[(i * NB + r) * n + j * NB], &block[r * bj], bj * sizeof(double));
                }
            } else if (rank == owner) {
                const std::vector<double>& block = blocks.at(blockKey(i, j, nblocks));
                MPI_Send(block.data(), static_cast<int>(bi * bj), MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
            } else if (rank == 0) {
                std::vector<double> block(bi * bj);
                MPI_Recv(block.data(), static_cast<int>(bi * bj), MPI_DOUBLE, owner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (size_t r = 0; r < bi; ++r) {
                    std::memcpy(&full[(i * NB + r) * n + j * NB], &block[r * bj], bj * sizeof(double));
                }
            }
        }
    }
}

// Distributed hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition.
bool choleskyDecompositionDistributed(std::unordered_map<size_t, std::vector<double>>& blocks,
                                       size_t n, size_t NB, int rank, int nprocs,
                                       GpuResources& gpu) {
    const size_t nblocks = numBlocks(n, NB);
    const size_t numLocal = gpu.numLocalGpus();
    const double gemmAlpha = -1.0, gemmBeta = 1.0, trsmAlpha = 1.0;

    std::vector<size_t> myRows;
    for (size_t i = 0; i < nblocks; ++i) {
        if (ownerOf(i, nprocs) == rank) myRows.push_back(i);
    }

    bool globalSuccess = true;

    for (size_t k = 0; k < nblocks; ++k) {
        const size_t bk = blockDim_(k, n, NB);
        const int kOwner = ownerOf(k, nprocs);

        // 1) Factor diagonal block A(k,k) -> L(k,k) on the CPU (small block).
        int successFlag = 1;
        if (rank == kOwner) {
            successFlag = choleskyDecomposition(blocks.at(blockKey(k, k, nblocks)), bk) ? 1 : 0;
        }
        MPI_Bcast(&successFlag, 1, MPI_INT, kOwner, MPI_COMM_WORLD);
        if (!successFlag) {
            globalSuccess = false;
            break;
        }

        // 2) Broadcast L(k,k) to all ranks.
        std::vector<double> LkkHost(bk * bk);
        if (rank == kOwner) {
            LkkHost = blocks.at(blockKey(k, k, nblocks));
        }
        MPI_Bcast(LkkHost.data(), static_cast<int>(bk * bk), MPI_DOUBLE, kOwner, MPI_COMM_WORLD);

        for (size_t t = 0; t < numLocal; ++t) {
            CUDA_CHECK(cudaSetDevice(gpu.devices[t]));
            CUDA_CHECK(cudaMemcpy(gpu.d_Lkk[t], LkkHost.data(), bk * bk * sizeof(double), cudaMemcpyHostToDevice));
        }

        // 3) Panel: L(i,k) = A(i,k) * L(k,k)^-T for owned rows i>k.
        std::vector<size_t> panelRows;
        for (size_t i : myRows) {
            if (i > k) panelRows.push_back(i);
        }

        #pragma omp parallel for num_threads(numLocal > 0 ? numLocal : 1) schedule(dynamic)
        for (size_t idx = 0; idx < panelRows.size(); ++idx) {
            const int t = omp_get_thread_num();
            const size_t i = panelRows[idx];
            const size_t bi = blockDim_(i, n, NB);
            CUDA_CHECK(cudaSetDevice(gpu.devices[t]));

            double* hostBuf = blocks.at(blockKey(i, k, nblocks)).data();
            CUDA_CHECK(cudaMemcpy(gpu.d_taskA[t], hostBuf, bi * bk * sizeof(double), cudaMemcpyHostToDevice));

            // Row-major (bi x bk) buffers are read by cuBLAS as column-major
            // (bk x bi) == transpose. Solving L(k,k) * X = A(i,k)^T (side
            // LEFT, UPPER because the transposed buffer of a lower
            // triangular matrix is upper triangular, TRANS to undo it)
            // yields X = L(i,k)^T stored exactly as our row-major L(i,k).
            CUBLAS_CHECK(cublasDtrsm(gpu.handles[t], CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                      static_cast<int>(bk), static_cast<int>(bi), &trsmAlpha,
                                      gpu.d_Lkk[t], static_cast<int>(bk),
                                      gpu.d_taskA[t], static_cast<int>(bk)));

            CUDA_CHECK(cudaMemcpy(hostBuf, gpu.d_taskA[t], bi * bk * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // 4) Gather the full column panel k (blocks L(i,k), i>k) to every rank.
        std::vector<size_t> panelOffset(nblocks, 0), panelBi(nblocks, 0);
        size_t panelTotal = 0;
        for (size_t i = k + 1; i < nblocks; ++i) {
            const size_t bi = blockDim_(i, n, NB);
            panelOffset[i] = panelTotal;
            panelBi[i] = bi;
            panelTotal += bi * bk;
        }

        std::vector<double> panelHost(panelTotal);
        for (size_t i : myRows) {
            if (i > k) {
                std::memcpy(&panelHost[panelOffset[i]], blocks.at(blockKey(i, k, nblocks)).data(),
                            panelBi[i] * bk * sizeof(double));
            }
        }

        std::vector<MPI_Request> reqs;
        reqs.reserve(nblocks > k + 1 ? nblocks - k - 1 : 0);
        for (size_t i = k + 1; i < nblocks; ++i) {
            const int iOwner = ownerOf(i, nprocs);
            MPI_Request req;
            MPI_Ibcast(&panelHost[panelOffset[i]], static_cast<int>(panelBi[i] * bk), MPI_DOUBLE,
                       iOwner, MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }
        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }

        for (size_t t = 0; t < numLocal; ++t) {
            if (panelTotal == 0) continue;
            CUDA_CHECK(cudaSetDevice(gpu.devices[t]));
            CUDA_CHECK(cudaMemcpy(gpu.d_panel[t], panelHost.data(), panelTotal * sizeof(double), cudaMemcpyHostToDevice));
        }

        // 5) Trailing update: A(i,j) -= L(i,k) * L(j,k)^T for owned rows i>k, j in (k,i].
        struct Task { size_t i, j; };
        std::vector<Task> tasks;
        for (size_t i : myRows) {
            if (i <= k) continue;
            for (size_t j = k + 1; j <= i; ++j) tasks.push_back({i, j});
        }

        #pragma omp parallel for num_threads(numLocal > 0 ? numLocal : 1) schedule(dynamic)
        for (size_t idx = 0; idx < tasks.size(); ++idx) {
            const int t = omp_get_thread_num();
            const size_t i = tasks[idx].i;
            const size_t j = tasks[idx].j;
            const size_t bi = blockDim_(i, n, NB);
            const size_t bj = blockDim_(j, n, NB);
            CUDA_CHECK(cudaSetDevice(gpu.devices[t]));

            double* hostC = blocks.at(blockKey(i, j, nblocks)).data();
            CUDA_CHECK(cudaMemcpy(gpu.d_taskC[t], hostC, bi * bj * sizeof(double), cudaMemcpyHostToDevice));

            const double* d_Lik = gpu.d_panel[t] + panelOffset[i];
            const double* d_Ljk = gpu.d_panel[t] + panelOffset[j];

            // C(i,j) -= L(i,k) * L(j,k)^T, expressed in the transposed
            // (column-major) view of our row-major buffers: see derivation
            // in the design notes above cublasDgemm's parameters.
            CUBLAS_CHECK(cublasDgemm(gpu.handles[t], CUBLAS_OP_T, CUBLAS_OP_N,
                                      static_cast<int>(bj), static_cast<int>(bi), static_cast<int>(bk),
                                      &gemmAlpha, d_Ljk, static_cast<int>(bk),
                                      d_Lik, static_cast<int>(bk),
                                      &gemmBeta, gpu.d_taskC[t], static_cast<int>(bj)));

            CUDA_CHECK(cudaMemcpy(hostC, gpu.d_taskC[t], bi * bj * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    return globalSuccess;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    omp_set_dynamic(0);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank under mpirun)
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
        printf("MPI ranks: %d\n", nprocs);
    }

    const size_t NB = std::max<size_t>(1, std::min<size_t>(256, n));
    const size_t nblocks = numBlocks(n, NB);

    GpuResources gpu;
    gpu.init(rank, nprocs, n, NB);

    std::unordered_map<size_t, std::vector<double>> blocks;

    std::vector<double> Aglobal, A_orig;

    if (rank == 0) {
        Aglobal.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(Aglobal, n);
        if (validate) {
            A_orig = Aglobal; // Save original for validation
        }
    }

    distributeMatrix(Aglobal, n, NB, nblocks, rank, nprocs, blocks);
    if (rank == 0 && !validate && !printResults) {
        Aglobal.clear();
        Aglobal.shrink_to_fit();
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionDistributed(blocks, n, NB, rank, nprocs, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        gpu.destroy();
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;

    if (printResults || validate) {
        gatherMatrix(Aglobal, n, NB, nblocks, rank, nprocs, blocks);

        if (rank == 0 && printResults) {
            print_results(Aglobal, "CholeskyL");
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
                bool valid = validateCholesky(Aglobal, A_orig, n);
                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    gpu.destroy();
    MPI_Finalize();
    return exitCode;
}
