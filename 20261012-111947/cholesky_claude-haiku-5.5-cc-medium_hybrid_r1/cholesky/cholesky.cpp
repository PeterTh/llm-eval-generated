#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Work distribution:
//  - MPI:    the matrix is split into column blocks of width NB, dealt cyclically over the ranks
//            (block jb belongs to rank jb % size). Each rank generates and stores only its blocks.
//  - CUDA:   each rank keeps its blocks resident on one GPU (ranks on a node share the node's
//            devices round-robin) and runs the O(n^3) trailing updates there.
//  - OpenMP: each rank's CPU threads generate its columns, factor the panel it owns, and run
//            the validation.
//
// Algorithm: right-looking blocked Cholesky. For each block column kb:
//   1. the owner copies its panel to the host and factors it (diagonal block serially,
//      rows below it in parallel),
//   2. the factored panel is broadcast to all ranks,
//   3. every rank updates its trailing blocks on the GPU: A -= P * P^T.
// The update of block kb+1 is issued first, so the next panel can be prepared on the host
// while the rest of the update still runs on the GPUs (look-ahead).
// The matrix is padded to a multiple of NB with identity rows and columns, which leaves the
// factor of the original n x n part unchanged.

constexpr size_t NB = 128;   // column block width
constexpr size_t TILE = 64;  // GPU output tile edge; NB must be a multiple of it
constexpr size_t TK = 16;    // GPU k-step; NB must be a multiple of it

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                     \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Page-locked host buffer, so that device transfers run at full PCIe bandwidth
struct PinnedBuffer {
    double* data = nullptr;

    explicit PinnedBuffer(const size_t count) {
        if (count > 0) CUDA_CHECK(cudaMallocHost(&data, count * sizeof(double)));
    }
    ~PinnedBuffer() {
        if (data) cudaFreeHost(data);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// Streams and events of the look-ahead pipeline
struct Pipeline {
    cudaStream_t compute = nullptr;
    cudaStream_t copy = nullptr;
    cudaEvent_t lookAhead = nullptr;                 // after the update of the next block column
    cudaEvent_t panelFree[2] = {nullptr, nullptr};   // after the last use of a panel buffer

    Pipeline() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&lookAhead, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelFree[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelFree[1], cudaEventDisableTiming));
    }
    ~Pipeline() {
        cudaEventDestroy(lookAhead);
        cudaEventDestroy(panelFree[0]);
        cudaEventDestroy(panelFree[1]);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(copy);
    }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
};

struct Layout {
    size_t rank;
    size_t size;
    size_t nbBlocks;  // number of column blocks
    size_t Npad;      // nbBlocks * NB, padded matrix dimension
    size_t blkElems;  // Npad * NB, elements per column block (row-major)
    size_t nLoc;      // column blocks owned by this rank
    size_t maxLoc;    // largest nLoc over all ranks
};

Layout makeLayout(const size_t n, const size_t rank, const size_t size) {
    Layout ly{};
    ly.rank = rank;
    ly.size = size;
    ly.nbBlocks = (n + NB - 1) / NB;
    ly.Npad = ly.nbBlocks * NB;
    ly.blkElems = ly.Npad * NB;
    ly.nLoc = rank < ly.nbBlocks ? (ly.nbBlocks - rank - 1) / size + 1 : 0;
    ly.maxLoc = (ly.nbBlocks + size - 1) / size;
    return ly;
}

// Trailing update of the owned blocks jb >= jbMin, one grid slice (blockIdx.z) per local block
// lbBase + blockIdx.z. For panel rows r in [rowOff, R) and block columns c in [0, NB),
// with rowOff = jb * NB - k0:
//   C[r*NB + c] -= sum_p P[r*NB + p] * P[(rowOff + c)*NB + p]
// C is the block's storage starting at its row k0 and P is the factored panel. Each thread
// accumulates a 4x4 register tile with p in increasing order. Threads map to tile elements with
// stride 16, so that shared-memory reads are free of bank conflicts.
__global__ void trailingUpdateKernel(double* __restrict__ dBlk, const double* __restrict__ P,
                                     const int blkElems, const int k0, const int R,
                                     const int size, const int rank, const int lbBase,
                                     const int jbMin) {
    constexpr int T = static_cast<int>(TILE);
    constexpr int K = static_cast<int>(TK);
    constexpr int NBi = static_cast<int>(NB);
    __shared__ double As[T][K + 1];
    __shared__ double Bs[K][T + 1];

    const int lb = lbBase + blockIdx.z;
    const int jb = lb * size + rank;
    if (jb < jbMin) return;
    const int rowOff = jb * NBi - k0;
    const int rBase = rowOff + blockIdx.y * T;
    if (rBase >= R) return;
    double* C = dBlk + lb * blkElems + k0 * NBi;
    const int cBase = blockIdx.x * T;

    const int threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double acc[4][4] = {};
    for (int kk = 0; kk < NBi; kk += K) {
        for (int e = tid; e < T * K; e += threads) {
            const int row = e / K;
            const int k = e % K;
            const int r = rBase + row;
            As[row][k] = (r < R) ? P[r * NBi + kk + k] : 0.0;
            Bs[k][row] = P[(rowOff + cBase + row) * NBi + kk + k];
        }
        __syncthreads();

        for (int k = 0; k < K; ++k) {
            double a[4];
            double b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[threadIdx.y + 16 * i][k];
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][threadIdx.x + 16 * j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        const int r = rBase + threadIdx.y + 16 * i;
        if (r >= R) continue;
        for (int j = 0; j < 4; ++j) C[r * NBi + cBase + threadIdx.x + 16 * j] -= acc[i][j];
    }
}

// Launches the trailing update for local blocks [lbBase, lbBase + count) with jb >= jbMin.
void launchTrailingUpdate(double* dBlk, const double* dPanel, const Layout& ly, const size_t k0,
                          const size_t R, const size_t lbBase, const size_t count,
                          const size_t jbMin, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(NB / TILE),
                    static_cast<unsigned int>((R - NB + TILE - 1) / TILE),
                    static_cast<unsigned int>(count));
    trailingUpdateKernel<<<grid, dim3(16, 16), 0, stream>>>(
        dBlk, dPanel, static_cast<int>(ly.blkElems), static_cast<int>(k0), static_cast<int>(R),
        static_cast<int>(ly.size), static_cast<int>(ly.rank), static_cast<int>(lbBase),
        static_cast<int>(jbMin));
    CUDA_CHECK(cudaGetLastError());
}

// Computes row i of the lower factor in place from the original row i and the finished rows
// j <= min(i, NB-1) of the panel (row-major, NB columns). The arithmetic matches the
// sequential algorithm. Returns false if the diagonal pivot is not positive; failCol is then
// the offending column of the panel.
bool factorPanelRow(double* P, const size_t i, size_t& failCol) {
    double* Ri = P + i * NB;
    const size_t jmax = std::min(i, NB - 1);
    for (size_t j = 0; j <= jmax; ++j) {
        const double* Rj = P + j * NB;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += Ri[k] * Rj[k];
        }
        if (i == j) {
            const double val = Ri[j] - sum;
            if (val <= 0.0) {
                failCol = j;
                return false;
            }
            Ri[j] = sqrt(val);
        } else {
            Ri[j] = (Ri[j] - sum) / Rj[j];
        }
    }
    return true;
}

// Factors the R x NB panel whose first row is global row k0. Returns false if the matrix is not
// positive definite; failIndex is then the global diagonal index.
bool factorPanel(double* P, const size_t R, const size_t k0, size_t& failIndex) {
    // The rows of the diagonal block depend on each other, so they are factored in order
    for (size_t i = 0; i < NB; ++i) {
        size_t failCol = 0;
        if (!factorPanelRow(P, i, failCol)) {
            failIndex = k0 + failCol;
            return false;
        }
    }

    // Rows below the diagonal block depend only on the finished diagonal block
    #pragma omp parallel for schedule(static)
    for (size_t i = NB; i < R; ++i) {
        size_t unused = 0;
        factorPanelRow(P, i, unused);
    }
    return true;
}

// Random matrix B, identical on every rank (same seed and order as the sequential code)
std::vector<double> makeRandomMatrix(const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    return B;
}

// Generates column block jb of A = B * B^T + n * I into out (Npad x NB, row-major).
// Rows and columns beyond n are the identity.
void generateBlockColumn(const std::vector<double>& B, const size_t n, const size_t Npad,
                         const size_t jb, double* out) {
    const size_t c0 = jb * NB;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Npad; ++i) {
        for (size_t c = 0; c < NB; ++c) {
            const size_t j = c0 + c;
            double v;
            if (i < n && j < n) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += B[i * n + k] * B[j * n + k];
                }
                // Add diagonal dominance to ensure positive definiteness
                v = (i == j) ? sum + static_cast<double>(n) : sum;
            } else {
                v = (i == j) ? 1.0 : 0.0;
            }
            out[i * NB + c] = v;
        }
    }
}

// Distributed blocked Cholesky on the device-resident column blocks. Returns false on every
// rank if the matrix is not positive definite. On success the factor is copied to hL, which
// has the same block layout as dBlk. dPanel holds two panel buffers of blkElems each.
bool choleskyDecomposition(const Layout& ly, double* dBlk, double* dPanel, double* hPanel,
                           double* hL, size_t& failIndex) {
    Pipeline pipe;

    for (size_t kb = 0; kb < ly.nbBlocks; ++kb) {
        const size_t k0 = kb * NB;
        const size_t R = ly.Npad - k0;  // panel height
        const size_t owner = kb % ly.size;
        const size_t buf = kb % 2;
        double* dPanelBuf = dPanel + buf * ly.blkElems;
        const size_t panelBytes = R * NB * sizeof(double);

        // 1. The owner factors its panel on the host. Its block kb is final once the look-ahead
        //    update of step kb-1 has run.
        long long status[2] = {1, 0};  // {ok, failIndex}
        if (ly.rank == owner) {
            double* dCol = dBlk + (kb / ly.size) * ly.blkElems + k0 * NB;
            CUDA_CHECK(cudaStreamWaitEvent(pipe.copy, pipe.lookAhead, 0));
            CUDA_CHECK(cudaMemcpyAsync(hPanel, dCol, panelBytes, cudaMemcpyDeviceToHost, pipe.copy));
            CUDA_CHECK(cudaStreamSynchronize(pipe.copy));
            size_t fail = 0;
            if (factorPanel(hPanel, R, k0, fail)) {
                CUDA_CHECK(cudaMemcpyAsync(dCol, hPanel, panelBytes, cudaMemcpyHostToDevice, pipe.copy));
            } else {
                status[0] = 0;
                status[1] = static_cast<long long>(fail);
            }
        }

        // 2. Every rank learns whether the factorization succeeded and receives the panel
        MPI_Bcast(status, 2, MPI_LONG_LONG, static_cast<int>(owner), MPI_COMM_WORLD);
        if (status[0] == 0) {
            failIndex = static_cast<size_t>(status[1]);
            return false;
        }
        MPI_Bcast(hPanel, static_cast<int>(R * NB), MPI_DOUBLE, static_cast<int>(owner), MPI_COMM_WORLD);

        // 3. Stage the panel on the device; the previous user of this buffer must be finished
        CUDA_CHECK(cudaStreamWaitEvent(pipe.copy, pipe.panelFree[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(dPanelBuf, hPanel, panelBytes, cudaMemcpyHostToDevice, pipe.copy));
        CUDA_CHECK(cudaStreamSynchronize(pipe.copy));

        // 4. Look-ahead update of block kb+1 (needed for the next panel), then the other blocks
        if (kb + 1 < ly.nbBlocks && (kb + 1) % ly.size == ly.rank) {
            launchTrailingUpdate(dBlk, dPanelBuf, ly, k0, R, (kb + 1) / ly.size, 1, kb + 1, pipe.compute);
            CUDA_CHECK(cudaEventRecord(pipe.lookAhead, pipe.compute));
        }
        if (ly.nLoc > 0 && kb + 2 < ly.nbBlocks) {
            launchTrailingUpdate(dBlk, dPanelBuf, ly, k0, R, 0, ly.nLoc, kb + 2, pipe.compute);
        }
        CUDA_CHECK(cudaEventRecord(pipe.panelFree[buf], pipe.compute));
    }

    // Bring the owned factor blocks back to the host
    CUDA_CHECK(cudaStreamSynchronize(pipe.compute));
    for (size_t lb = 0; lb < ly.nLoc; ++lb) {
        CUDA_CHECK(cudaMemcpyAsync(hL + lb * ly.blkElems, dBlk + lb * ly.blkElems,
                                   ly.blkElems * sizeof(double), cudaMemcpyDeviceToHost, pipe.copy));
    }
    CUDA_CHECK(cudaStreamSynchronize(pipe.copy));
    return true;
}

// Collects every rank's blocks on rank 0 (rank-major). Other ranks receive an empty vector.
std::vector<double> gatherBlocks(const double* local, const size_t count, const Layout& ly) {
    std::vector<double> all(ly.rank == 0 ? ly.size * count : 0);
    MPI_Gather(local, static_cast<int>(count), MPI_DOUBLE, all.data(), static_cast<int>(count),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return all;
}

// Unpacks rank-major column blocks into a row-major n x n matrix. With lowerOnly, the strict
// upper triangle stays zero, as in the sequential factor.
std::vector<double> unpackBlocks(const std::vector<double>& packed, const Layout& ly,
                                 const size_t n, const bool lowerOnly) {
    std::vector<double> M(n * n, 0.0);
    const size_t rankSpan = ly.maxLoc * ly.blkElems;
    for (size_t r = 0; r < ly.size; ++r) {
        for (size_t lb = 0; lb < ly.maxLoc; ++lb) {
            const size_t jb = lb * ly.size + r;
            if (jb >= ly.nbBlocks) break;
            const double* blk = packed.data() + r * rankSpan + lb * ly.blkElems;
            for (size_t i = 0; i < n; ++i) {
                for (size_t c = 0; c < NB; ++c) {
                    const size_t col = jb * NB + c;
                    if (col < n && (!lowerOnly || col <= i)) {
                        M[i * n + col] = blk[i * NB + c];
                    }
                }
            }
        }
    }
    return M;
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

int runBenchmark(int argc, char** argv, const size_t rank, const size_t size) {
    const bool root = rank == 0;
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Bind this rank to a GPU: ranks on the same node share its devices round-robin
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, static_cast<int>(rank), MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Split the cores this rank may use between the node's ranks, unless the user sets threads
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Generate this rank's column blocks directly into device memory
    const Layout ly = makeLayout(n, rank, size);
    const size_t blkBytes = ly.blkElems * sizeof(double);

    double* dBlk = nullptr;
    double* dPanel = nullptr;
    if (ly.nLoc > 0) CUDA_CHECK(cudaMalloc(&dBlk, ly.nLoc * blkBytes));
    if (ly.blkElems > 0) CUDA_CHECK(cudaMalloc(&dPanel, 2 * blkBytes));

    PinnedBuffer hPanel(ly.blkElems);
    PinnedBuffer hL(ly.maxLoc * ly.blkElems);
    std::vector<double> stage(ly.blkElems);
    std::vector<double> origLocal(validate ? ly.maxLoc * ly.blkElems : 0);
    {
        const std::vector<double> B = makeRandomMatrix(n);
        for (size_t lb = 0; lb < ly.nLoc; ++lb) {
            generateBlockColumn(B, n, ly.Npad, lb * size + rank, stage.data());
            CUDA_CHECK(cudaMemcpy(dBlk + lb * ly.blkElems, stage.data(), blkBytes, cudaMemcpyHostToDevice));
            if (validate) {
                std::copy(stage.begin(), stage.end(), origLocal.begin() + lb * ly.blkElems);
            }
        }
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    size_t failIndex = 0;
    const bool success = choleskyDecomposition(ly, dBlk, dPanel, hPanel.data, hL.data, failIndex);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(t1 - t0));

    if (dBlk) CUDA_CHECK(cudaFree(dBlk));
    if (dPanel) CUDA_CHECK(cudaFree(dPanel));

    if (!success) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIndex);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (!printResults && !validate) return 0;

    // The factor (and the original matrix, for validation) is collected on rank 0
    const std::vector<double> gathered = gatherBlocks(hL.data, ly.maxLoc * ly.blkElems, ly);
    const std::vector<double> gatheredOrig =
        validate ? gatherBlocks(origLocal.data(), ly.maxLoc * ly.blkElems, ly) : std::vector<double>();
    if (!root) return 0;

    const std::vector<double> L = unpackBlocks(gathered, ly, n, true);

    // Print results for external validation
    if (printResults) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const std::vector<double> A_orig = unpackBlocks(gatheredOrig, ly, n, false);
        bool valid = validateCholesky(L, A_orig, n);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, static_cast<size_t>(rank), static_cast<size_t>(size));

    MPI_Finalize();
    return rc;
}
