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

// Hybrid MPI + OpenMP + CUDA blocked (right-looking) Cholesky decomposition.
//
// The matrix is distributed across MPI ranks using a row-cyclic mapping
// (global row g is owned by rank g % P, stored at local index g / P). Each
// rank keeps its full-width local rows resident on its assigned GPU for the
// lifetime of the run. For each panel step k:
//   1. The (small) diagonal block is assembled redundantly on every rank via
//      MPI_Allgatherv and factorized (cheap, negligible cost vs. the O(n^3)
//      trailing update).
//   2. Each rank performs the triangular solve ("panel scale") for the local
//      rows it owns below the panel, using OpenMP across rows.
//   3. The scaled panel columns are gathered (MPI_Allgatherv) so every rank
//      has the full panel needed to update its own trailing rows.
//   4. The (dominant) trailing matrix update is performed on the GPU with a
//      custom CUDA kernel, operating on the device-resident local matrix.
//
// This reproduces the same mathematical Cholesky factorization as the
// original sequential unblocked algorithm (same recurrence, only grouped
// into blocked partial sums), so results agree within standard floating
// point tolerances used by validation.

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// CUDA kernel performing the trailing-matrix update for the local rows owned
// by this rank that fall below the current panel block:
//   A[gi][c] -= dot(panel[gi], panel[c])   for c in [kEnd, gi]
// localA is the device-resident local matrix (rowCountTotal x n, row-major).
// Row li (local index) corresponds to global row gi = li * P + rank.
__global__ void trailingUpdateKernel(double* localA, int loStart, int rowsToUpdate, int P, int rank,
                                      const double* panelFull, int blockLen, int kEnd, int n) {
    int idx = blockIdx.x;
    if (idx >= rowsToUpdate) return;

    int li = loStart + idx;
    int gi = li * P + rank;
    int numCols = gi - kEnd + 1; // columns kEnd..gi inclusive (lower triangle only)
    if (numCols <= 0) return;

    extern __shared__ double sharedPanelRow[];
    double* myRow = localA + static_cast<size_t>(li) * n;
    const double* myPanel = panelFull + static_cast<size_t>(gi - kEnd) * blockLen;

    for (int t = threadIdx.x; t < blockLen; t += blockDim.x) {
        sharedPanelRow[t] = myPanel[t];
    }
    __syncthreads();

    for (int c = threadIdx.x; c < numCols; c += blockDim.x) {
        const double* colPanel = panelFull + static_cast<size_t>(c) * blockLen;
        double sum = 0.0;
        for (int b = 0; b < blockLen; ++b) {
            sum += sharedPanelRow[b] * colPanel[b];
        }
        myRow[kEnd + c] -= sum;
    }
}

// Number of global rows in [a, b) owned by rank r under row-cyclic mapping.
static inline long countOwnedInRange(long a, long b, long r, long P) {
    if (b <= a) return 0;
    long first = a + ((r - a) % P + P) % P; // first global row >= a with row % P == r
    if (first >= b) return 0;
    return (b - 1 - first) / P + 1;
}

// First global row >= a owned by rank r.
static inline long firstOwnedGe(long a, long r, long P) {
    return a + ((r - a) % P + P) % P;
}

static inline long numLocalRows(long n, long P, long r) {
    return countOwnedInRange(0, n, r, P);
}

// Gather rows [a, b) (width columns starting at colStart) from every rank's
// local storage into a dense (b-a) x width buffer, replicated on all ranks.
static std::vector<double> gatherRowRange(long a, long b, long width, long colStart, long n, long P, long rank,
                                           const std::vector<double>& hostLocalA, MPI_Comm comm) {
    std::vector<double> result(static_cast<size_t>(std::max<long>(0, b - a)) * width);
    if (b <= a || width <= 0) return result;

    long localCount = countOwnedInRange(a, b, rank, P);
    std::vector<double> sendbuf(static_cast<size_t>(localCount) * width);

    long gi = firstOwnedGe(a, rank, P);
    long idx = 0;
    while (gi < b) {
        long li = (gi - rank) / P;
        std::memcpy(sendbuf.data() + static_cast<size_t>(idx) * width, hostLocalA.data() + static_cast<size_t>(li) * n + colStart,
                    static_cast<size_t>(width) * sizeof(double));
        ++idx;
        gi += P;
    }

    std::vector<int> recvcounts(P), displs(P);
    long total = 0;
    for (long r = 0; r < P; ++r) {
        long cnt = countOwnedInRange(a, b, r, P) * width;
        recvcounts[r] = static_cast<int>(cnt);
        displs[r] = static_cast<int>(total);
        total += cnt;
    }

    std::vector<double> recvbuf(static_cast<size_t>(total));
    MPI_Allgatherv(sendbuf.data(), static_cast<int>(localCount * width), MPI_DOUBLE, recvbuf.data(), recvcounts.data(),
                    displs.data(), MPI_DOUBLE, comm);

    for (long r = 0; r < P; ++r) {
        long g = firstOwnedGe(a, r, P);
        long off = displs[r];
        while (g < b) {
            std::memcpy(result.data() + static_cast<size_t>(g - a) * width, recvbuf.data() + off, static_cast<size_t>(width) * sizeof(double));
            off += width;
            g += P;
        }
    }

    return result;
}

// Factorize a small blockLen x blockLen diagonal block in place (same
// recurrence as the unblocked algorithm, restricted to the block).
static bool factorizeDiagBlock(std::vector<double>& D, long blockLen, long kStart, bool verbose) {
    for (long i = 0; i < blockLen; ++i) {
        for (long j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (long k = 0; k < j; ++k) {
                    sum += D[j * blockLen + k] * D[j * blockLen + k];
                }
                const double val = D[j * blockLen + j] - sum;
                if (val <= 0.0) {
                    if (verbose) {
                        printf("Error: Matrix is not positive definite at diagonal element %ld\n", kStart + j);
                    }
                    return false;
                }
                D[j * blockLen + j] = sqrt(val);
            } else {
                for (long k = 0; k < j; ++k) {
                    sum += D[i * blockLen + k] * D[j * blockLen + k];
                }
                D[i * blockLen + j] = (D[i * blockLen + j] - sum) / D[j * blockLen + j];
            }
        }
        for (long j = i + 1; j < blockLen; ++j) {
            D[i * blockLen + j] = 0.0;
        }
    }
    return true;
}

// Distributed hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition.
// hostLocalA holds this rank's owned rows (row-cyclic), full width n.
bool choleskyDecompositionHybrid(std::vector<double>& hostLocalA, const size_t n, int rank, int worldSize,
                                  int gpuDevice) {
    const long nn = static_cast<long>(n);
    const long P = worldSize;
    const long localRowCount = numLocalRows(nn, P, rank);

    const long B = std::max<long>(1, std::min<long>(256, nn));
    const long nb = (nn + B - 1) / B;

    CUDA_CHECK(cudaSetDevice(gpuDevice));

    double* localA_dev = nullptr;
    double* panel_dev = nullptr;
    if (localRowCount > 0) {
        CUDA_CHECK(cudaMalloc(&localA_dev, static_cast<size_t>(localRowCount) * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(localA_dev, hostLocalA.data(), static_cast<size_t>(localRowCount) * n * sizeof(double),
                               cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&panel_dev, static_cast<size_t>(nn) * B * sizeof(double)));

    bool ok = true;

    for (long k = 0; k < nb && ok; ++k) {
        const long kStart = k * B;
        const long kEnd = std::min(kStart + B, nn);
        const long blockLen = kEnd - kStart;

        // Refresh host copy of the still-active columns [kStart, kEnd) for all
        // remaining owned rows (>= kStart) from the GPU (device holds the
        // latest values due to prior trailing updates).
        const long loActiveLocal = (firstOwnedGe(kStart, rank, P) - rank) / P;
        const long activeRows = localRowCount - loActiveLocal;
        if (activeRows > 0) {
            CUDA_CHECK(cudaMemcpy2D(hostLocalA.data() + static_cast<size_t>(loActiveLocal) * n + kStart,
                                     n * sizeof(double),
                                     localA_dev + static_cast<size_t>(loActiveLocal) * n + kStart,
                                     n * sizeof(double), static_cast<size_t>(blockLen) * sizeof(double),
                                     static_cast<size_t>(activeRows), cudaMemcpyDeviceToHost));
        }

        // Assemble the diagonal block redundantly on every rank and factorize.
        std::vector<double> diagBlock =
            gatherRowRange(kStart, kEnd, blockLen, kStart, nn, P, rank, hostLocalA, MPI_COMM_WORLD);
        ok = factorizeDiagBlock(diagBlock, blockLen, kStart, rank == 0);
        if (!ok) break;

        // Write the factorized diagonal block back into this rank's owned
        // rows within [kStart, kEnd).
        {
            long gi = firstOwnedGe(kStart, rank, P);
            while (gi < kEnd) {
                long li = (gi - rank) / P;
                std::memcpy(hostLocalA.data() + static_cast<size_t>(li) * n + kStart,
                            diagBlock.data() + static_cast<size_t>(gi - kStart) * blockLen,
                            static_cast<size_t>(blockLen) * sizeof(double));
                gi += P;
            }
        }

        // Panel scale: triangular solve for owned local rows below the block.
        const long loTrailLocal = (firstOwnedGe(kEnd, rank, P) - rank) / P;
        const long trailCountLocal = localRowCount - loTrailLocal;

#pragma omp parallel for schedule(static)
        for (long li = loTrailLocal; li < localRowCount; ++li) {
            double* row = hostLocalA.data() + static_cast<size_t>(li) * n;
            for (long c = 0; c < blockLen; ++c) {
                double sum = 0.0;
                for (long kk = 0; kk < c; ++kk) {
                    sum += row[kStart + kk] * diagBlock[c * blockLen + kk];
                }
                row[kStart + c] = (row[kStart + c] - sum) / diagBlock[c * blockLen + c];
            }
        }

        if (kEnd >= nn) break; // last block: no trailing matrix left

        // Gather the full scaled panel (rows [kEnd, n)) needed by every rank.
        std::vector<double> panelFull =
            gatherRowRange(kEnd, nn, blockLen, kStart, nn, P, rank, hostLocalA, MPI_COMM_WORLD);

        if (trailCountLocal > 0) {
            CUDA_CHECK(cudaMemcpy(panel_dev, panelFull.data(), panelFull.size() * sizeof(double), cudaMemcpyHostToDevice));

            const int threadsPerBlock = 128;
            const size_t sharedBytes = static_cast<size_t>(blockLen) * sizeof(double);
            trailingUpdateKernel<<<static_cast<int>(trailCountLocal), threadsPerBlock, sharedBytes>>>(
                localA_dev, static_cast<int>(loTrailLocal), static_cast<int>(trailCountLocal), static_cast<int>(P),
                rank, panel_dev, static_cast<int>(blockLen), static_cast<int>(kEnd), static_cast<int>(nn));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    if (localA_dev) CUDA_CHECK(cudaFree(localA_dev));
    if (panel_dev) CUDA_CHECK(cudaFree(panel_dev));

    if (!ok) return false;

    // Zero out the strictly-upper-triangular part of each owned row (matches
    // the original algorithm's final output layout).
#pragma omp parallel for schedule(static)
    for (long li = 0; li < localRowCount; ++li) {
        const long gi = li * P + rank;
        double* row = hostLocalA.data() + static_cast<size_t>(li) * n;
        for (long j = gi + 1; j < nn; ++j) {
            row[j] = 0.0;
        }
    }

    return true;
}

// Generate the local rows (row-cyclic ownership) of a symmetric positive
// definite matrix A = B * B^T + n*I, using the same deterministic B matrix
// (identical seed/order on every rank) as the original sequential algorithm,
// so results are bit-identical to the unblocked reference implementation.
void generatePositiveDefiniteMatrixLocal(std::vector<double>& localA, const size_t n, int rank, int worldSize) {
    const long nn = static_cast<long>(n);
    const long P = worldSize;
    const long localRowCount = numLocalRows(nn, P, rank);

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    localA.assign(static_cast<size_t>(localRowCount) * n, 0.0);

#pragma omp parallel for schedule(static)
    for (long li = 0; li < localRowCount; ++li) {
        const long gi = li * P + rank;
        double* row = localA.data() + static_cast<size_t>(li) * n;
        for (long j = 0; j < nn; ++j) {
            double sum = 0.0;
            for (long k = 0; k < nn; ++k) {
                sum += B[gi * n + k] * B[j * n + k];
            }
            row[j] = sum;
        }
        row[gi] += static_cast<double>(nn);
    }
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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    // Pick a GPU for this rank: prefer the node-local rank (so multiple ranks
    // on the same node spread across that node's GPUs), falling back to the
    // global rank modulo device count for single-node runs.
    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);
    if (numGpus <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int localRank = -1;
    const char* localRankEnvs[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                                   "PMI_LOCAL_RANK"};
    for (const char* envName : localRankEnvs) {
        const char* val = std::getenv(envName);
        if (val) {
            localRank = atoi(val);
            break;
        }
    }
    if (localRank < 0) localRank = rank;
    int gpuDevice = localRank % numGpus;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA hybrid)\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n", worldSize, omp_get_max_threads(), numGpus);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> localA;
    generatePositiveDefiniteMatrixLocal(localA, n, rank, worldSize);

    std::vector<double> localA_orig;
    if (validate) {
        localA_orig = localA;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionHybrid(localA, n, rank, worldSize, gpuDevice);

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
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int returnCode = 0;

    if (printResults || validate) {
        std::vector<double> fullA =
            gatherRowRange(0, static_cast<long>(n), static_cast<long>(n), 0, static_cast<long>(n), worldSize, rank,
                            localA, MPI_COMM_WORLD);

        if (printResults && rank == 0) {
            print_results(fullA, "CholeskyL");
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");

            const long nn = static_cast<long>(n);
            const long localRowCount = numLocalRows(nn, worldSize, rank);
            double maxError = 0.0;
            double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
            for (long li = 0; li < localRowCount; ++li) {
                const long gi = li * worldSize + rank;
                const double* origRow = localA_orig.data() + static_cast<size_t>(li) * n;
                for (long j = 0; j < nn; ++j) {
                    double sum = 0.0;
                    for (long k = 0; k < nn; ++k) {
                        sum += fullA[static_cast<size_t>(gi) * n + k] * fullA[static_cast<size_t>(j) * n + k];
                    }
                    const double error = fabs(sum - origRow[j]);
                    maxError = std::max(maxError, error);
                    const double rel = error / (fabs(origRow[j]) + 1e-10);
                    relError = std::max(relError, rel);
                }
            }

            double globalMaxError = 0.0, globalRelError = 0.0;
            MPI_Allreduce(&maxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&relError, &globalRelError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Max absolute error: %.10e\n", globalMaxError);
                printf("Max relative error: %.10e\n", globalRelError);
                if (globalRelError > 1e-6) {
                    printf("Validation failed: relative error too large\n");
                    printf("Validation: FAILED\n");
                    returnCode = 1;
                } else {
                    printf("Validation: PASSED\n");
                }
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
