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

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define CHOLESKY_HAVE_MPI_EXT 1
#endif
#endif

#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// The matrix is distributed over MPI ranks by block-cyclic column blocks.
// Each rank keeps only the columns it owns on the GPU.  The panel is factored
// by its owner rank on the GPU and broadcast to every rank; the trailing
// submatrix update, which contains almost all of the work, is done with a CUDA
// kernel on every rank concurrently.  OpenMP parallelizes the host-side matrix
// generation and validation.

namespace {

constexpr int kBlockSize = 128;      // Cholesky block (panel) size
constexpr int kColsPerThreadBlock = 4;
constexpr int kRowsPerThreadBlock = 256;
constexpr int kThreads = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

__global__ void trailingUpdateKernel(
    double* __restrict__ A,
    const double* __restrict__ P,
    long long n,
    long long localCapacity,
    int nb,
    int rank,
    int numRanks,
    int numBlocks,
    long long start,
    int w,
    long long rows) {
    const int tile = blockIdx.x;
    const int tid = threadIdx.x;
    const long long lc0 = (long long)tile * kColsPerThreadBlock;

    long long gj[kColsPerThreadBlock];
    long long lo[kColsPerThreadBlock];
    bool valid[kColsPerThreadBlock];
    long long minj = n;

#pragma unroll
    for (int c = 0; c < kColsPerThreadBlock; ++c) {
        valid[c] = false;
        const long long lc = lc0 + c;
        if (lc < localCapacity) {
            const int lb = (int)(lc / nb);
            const int within = (int)(lc - (long long)lb * nb);
            const int gb = lb * numRanks + rank;
            if (gb < numBlocks) {
                const long long bstart = (long long)gb * nb;
                const int bw = (int)min((long long)nb, n - bstart);
                if (within < bw) {
                    const long long j = bstart + within;
                    if (j >= start + w) {
                        valid[c] = true;
                        gj[c] = j;
                        lo[c] = lc;
                        if (j < minj) minj = j;
                    }
                }
            }
        }
    }

    bool any = false;
#pragma unroll
    for (int c = 0; c < kColsPerThreadBlock; ++c) any |= valid[c];
    if (!any) return;

    extern __shared__ double sh[];  // [kColsPerThreadBlock][w]
    for (int idx = tid; idx < kColsPerThreadBlock * w; idx += blockDim.x) {
        const int c = idx / w;
        const int p = idx - c * w;
        sh[idx] = valid[c] ? P[(long long)p * rows + (gj[c] - start)] : 0.0;
    }
    __syncthreads();

    const long long rowBase =
        start + (long long)blockIdx.y * kRowsPerThreadBlock;
    const long long rowEnd = min(n, rowBase + kRowsPerThreadBlock);

    for (long long i = rowBase + tid; i < rowEnd; i += blockDim.x) {
        if (i < minj) continue;

        double acc[kColsPerThreadBlock];
#pragma unroll
        for (int c = 0; c < kColsPerThreadBlock; ++c) acc[c] = 0.0;

        const long long pi = i - start;
        for (int p = 0; p < w; ++p) {
            const double av = P[(long long)p * rows + pi];
#pragma unroll
            for (int c = 0; c < kColsPerThreadBlock; ++c) {
                if (valid[c] && i >= gj[c]) acc[c] += av * sh[c * w + p];
            }
        }

#pragma unroll
        for (int c = 0; c < kColsPerThreadBlock; ++c) {
            if (valid[c] && i >= gj[c]) A[lo[c] * n + i] -= acc[c];
        }
    }
}

__global__ void factorDiagBlockKernel(double* __restrict__ P,
                                     long long rows, int w, long long start,
                                     int* __restrict__ fail,
                                     int* __restrict__ failJ) {
    const int tid = threadIdx.x;
    for (int jj = 0; jj < w; ++jj) {
        if (tid == 0) {
            double sum = 0.0;
            for (int pp = 0; pp < jj; ++pp) {
                const double v = P[(long long)pp * rows + jj];
                sum += v * v;
            }
            const double val = P[(long long)jj * rows + jj] - sum;
            if (val <= 0.0) {
                *fail = 1;
                atomicMin(failJ, (int)(start + jj));
                P[(long long)jj * rows + jj] = 1.0;
            } else {
                P[(long long)jj * rows + jj] = sqrt(val);
            }
        }
        __syncthreads();

        const double diag = P[(long long)jj * rows + jj];
        for (int ii = jj + 1 + tid; ii < w; ii += blockDim.x) {
            double sum = 0.0;
            for (int pp = 0; pp < jj; ++pp) {
                sum += P[(long long)pp * rows + ii] *
                       P[(long long)pp * rows + jj];
            }
            P[(long long)jj * rows + ii] =
                (P[(long long)jj * rows + ii] - sum) / diag;
        }
        __syncthreads();
    }
}

__global__ void panelSolveKernel(double* __restrict__ P, long long rows,
                                 int w) {
    const long long ii =
        (long long)w + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ii >= rows) return;

    for (int jj = 0; jj < w; ++jj) {
        double acc = P[(long long)jj * rows + ii];
        for (int pp = 0; pp < jj; ++pp) {
            acc -= P[(long long)pp * rows + ii] *
                   P[(long long)pp * rows + jj];
        }
        P[(long long)jj * rows + ii] =
            acc / P[(long long)jj * rows + jj];
    }
}

void factorPanelGpu(double* d_panel, long long rows, int w, long long start,
                    int* d_fail, int* d_failJ) {
    factorDiagBlockKernel<<<1, 128, 0, 0>>>(d_panel, rows, w, start, d_fail,
                                            d_failJ);

    const long long offDiagRows = rows - w;
    if (offDiagRows > 0) {
        const int blocks = (int)((offDiagRows + kThreads - 1) / kThreads);
        panelSolveKernel<<<blocks, kThreads, 0, 0>>>(d_panel, rows, w);
    }
}

// Generate the columns owned by this rank of exactly the same positive
// definite matrix as the original benchmark (A = B * B^T + n * I).
void generateDistributed(std::vector<double>& localA, size_t n, int rank,
                         int numRanks, size_t nb, size_t localCapacity,
                         size_t numBlocks) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (long long lc = 0; lc < (long long)localCapacity; ++lc) {
        const size_t lb = (size_t)lc / nb;
        const size_t within = (size_t)lc - lb * nb;
        const size_t gb = lb * (size_t)numRanks + (size_t)rank;
        if (gb >= numBlocks) continue;
        const size_t bstart = gb * nb;
        const size_t bw = std::min(nb, n - bstart);
        if (within >= bw) continue;

        const size_t j = bstart + within;
        const double* Bj = &B[j * n];
        double* col = &localA[(size_t)lc * n];
        for (size_t i = 0; i < n; ++i) {
            const double* Bi = &B[i * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            if (i == j) sum += (double)n;
            col[i] = sum;
        }
    }
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[(size_t)i * n + k] * L[j * n + k];
            }
            reconstructed[(size_t)i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError) \
    reduction(max : relError)
    for (long long i = 0; i < (long long)(n * n); ++i) {
        const double error = fabs(reconstructed[(size_t)i] - A_orig[(size_t)i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[(size_t)i]) + 1e-10);
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

// Gather the distributed lower-triangular factor into a row-major matrix on
// rank 0.  Entries above the diagonal are zero, matching the original code.
void gatherFactor(const double* d_local, size_t n, int rank, int numRanks,
                  size_t nb, size_t localCapacity, size_t numBlocks,
                  std::vector<double>& fullRowMajor) {
    const size_t localElems = n * localCapacity;
    std::vector<double> h_local(localElems);
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_local,
                          localElems * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> gathered;
    if (rank == 0) gathered.resize((size_t)numRanks * localElems);

    MPI_Gather(h_local.data(), (int)localElems, MPI_DOUBLE,
               rank == 0 ? gathered.data() : nullptr, (int)localElems,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    fullRowMajor.assign(n * n, 0.0);
    for (int r = 0; r < numRanks; ++r) {
        const double* src = &gathered[(size_t)r * localElems];
        for (size_t lb = 0; lb < (numBlocks + (size_t)numRanks - 1) /
                                       (size_t)numRanks;
             ++lb) {
            const size_t gb = lb * (size_t)numRanks + (size_t)r;
            if (gb >= numBlocks) continue;
            const size_t bstart = gb * nb;
            const size_t bw = std::min(nb, n - bstart);
            for (size_t within = 0; within < bw; ++within) {
                const size_t lc = lb * nb + within;
                const size_t j = bstart + within;
                for (size_t i = j; i < n; ++i) {
                    fullRowMajor[i * n + j] = src[lc * n + i];
                }
            }
        }
    }
}

// Gather the full distributed original matrix to rank 0 (used for validation).
void gatherOriginal(const std::vector<double>& localOrig, size_t n, int rank,
                    int numRanks, size_t nb, size_t localCapacity,
                    size_t numBlocks, std::vector<double>& fullRowMajor) {
    const size_t localElems = n * localCapacity;
    std::vector<double> gathered;
    if (rank == 0) gathered.resize((size_t)numRanks * localElems);

    MPI_Gather(localOrig.data(), (int)localElems, MPI_DOUBLE,
               rank == 0 ? gathered.data() : nullptr, (int)localElems,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    fullRowMajor.assign(n * n, 0.0);
    for (int r = 0; r < numRanks; ++r) {
        const double* src = &gathered[(size_t)r * localElems];
        for (size_t lb = 0; lb < (numBlocks + (size_t)numRanks - 1) /
                                       (size_t)numRanks;
             ++lb) {
            const size_t gb = lb * (size_t)numRanks + (size_t)r;
            if (gb >= numBlocks) continue;
            const size_t bstart = gb * nb;
            const size_t bw = std::min(nb, n - bstart);
            for (size_t within = 0; within < bw; ++within) {
                const size_t lc = lb * nb + within;
                const size_t j = bstart + within;
                for (size_t i = 0; i < n; ++i) {
                    fullRowMajor[i * n + j] = src[lc * n + i];
                }
            }
        }
    }
}

bool choleskyParallel(double* d_A, size_t n, int rank, int numRanks,
                      size_t nb, size_t localCapacity, size_t numBlocks,
                      bool cudaAwareMpi) {
    double* d_panel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel, sizeof(double) * n * nb));

    int* d_fail = nullptr;
    int* d_failJ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failJ, sizeof(int)));

    double* h_panel = new double[n * nb];

    const size_t sharedBytes =
        (size_t)kColsPerThreadBlock * nb * sizeof(double);

    bool ok = true;
    for (size_t kb = 0; kb < numBlocks; ++kb) {
        const size_t start = kb * nb;
        const int w = (int)std::min(nb, n - start);
        const long long rows = (long long)(n - start);
        const int owner = (int)(kb % (size_t)numRanks);
        const size_t lb = kb / (size_t)numRanks;
        const size_t localOff = lb * nb;

        int localFail = 0;
        long long localFailJ = -1;

        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy2D(
                d_panel, (size_t)rows * sizeof(double),
                d_A + localOff * n + start, n * sizeof(double),
                (size_t)rows * sizeof(double), (size_t)w,
                cudaMemcpyDeviceToDevice));

            CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));
            const int initFailJ = (int)n;
            CUDA_CHECK(cudaMemcpy(d_failJ, &initFailJ, sizeof(int),
                                  cudaMemcpyHostToDevice));

            factorPanelGpu(d_panel, rows, w, (long long)start, d_fail, d_failJ);

            int hFail = 0;
            int hFailJ = 0;
            CUDA_CHECK(cudaMemcpy(&hFail, d_fail, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&hFailJ, d_failJ, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (hFail) {
                localFail = 1;
                localFailJ = hFailJ;
            }

            if (!cudaAwareMpi) {
                CUDA_CHECK(cudaMemcpy2D(
                    h_panel, (size_t)rows * sizeof(double), d_panel,
                    (size_t)rows * sizeof(double),
                    (size_t)rows * sizeof(double), (size_t)w,
                    cudaMemcpyDeviceToHost));
            }

            // Keep the owner's distributed columns for the final gather.
            CUDA_CHECK(cudaMemcpy2D(
                d_A + localOff * n + start, n * sizeof(double), d_panel,
                (size_t)rows * sizeof(double), (size_t)rows * sizeof(double),
                (size_t)w, cudaMemcpyDeviceToDevice));
        }

        int globalFail = 0;
        MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        long long globalFailJ = -1;
        MPI_Allreduce(&localFailJ, &globalFailJ, 1, MPI_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD);
        if (globalFail) {
            if (rank == 0) {
                printf(
                    "Error: Matrix is not positive definite at diagonal "
                    "element %lld\n",
                    globalFailJ);
            }
            ok = false;
            break;
        }

        if (cudaAwareMpi) {
            MPI_Bcast(d_panel, (int)(rows * w), MPI_DOUBLE, owner,
                      MPI_COMM_WORLD);
        } else {
            MPI_Bcast(h_panel, (int)(rows * w), MPI_DOUBLE, owner,
                      MPI_COMM_WORLD);
        }

        if (start + (size_t)w >= n) continue;

        if (!cudaAwareMpi && rank != owner) {
            CUDA_CHECK(cudaMemcpyAsync(d_panel, h_panel,
                                       (size_t)rows * w * sizeof(double),
                                       cudaMemcpyHostToDevice, 0));
        }

        const int gridx =
            (int)((localCapacity + kColsPerThreadBlock - 1) /
                  kColsPerThreadBlock);
        const int gridy =
            (int)((n - start + kRowsPerThreadBlock - 1) / kRowsPerThreadBlock);
        if (gridx > 0 && gridy > 0) {
            const dim3 grid(gridx, gridy);
            trailingUpdateKernel<<<grid, kThreads, sharedBytes, 0>>>(
                d_A, d_panel, (long long)n, (long long)localCapacity, (int)nb,
                rank, numRanks, (int)numBlocks, (long long)start, w, rows);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    delete[] h_panel;
    CUDA_CHECK(cudaFree(d_failJ));
    CUDA_CHECK(cudaFree(d_fail));
    CUDA_CHECK(cudaFree(d_panel));
    return ok;
}

// MPI launchers often bind each rank to a single core.  For the hybrid
// code we want each rank to have a fair share of the node's CPUs for OpenMP,
// so replace that binding with a balanced interleaved affinity mask.  Returns
// the number of CPUs assigned to the rank, or 0 if the affinity could not be
// changed.
int applyCpuAffinity(int localRank, int localSize) {
#ifdef __linux__
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus <= 0 || localSize <= 0) return 0;
    if (ncpus > CPU_SETSIZE) ncpus = CPU_SETSIZE;

    cpu_set_t set;
    CPU_ZERO(&set);
    for (long c = localRank; c < ncpus; c += localSize) {
        CPU_SET((int)c, &set);
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) return 0;
#ifdef SYS_set_mempolicy
    // The launcher may also have restricted the NUMA memory policy to the
    // cores it originally assigned.  Return to the default policy so all
    // OpenMP threads see equally local memory.
    syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 0UL);
#endif
    return (int)std::max(1L, (ncpus + (long)localSize - 1) / (long)localSize);
#else
    (void)localRank;
    (void)localSize;
    return 0;
#endif
}

void configureCpuAffinity(int localRank, int localSize) {
    const int threads = applyCpuAffinity(localRank, localSize);
    if (threads > 0) {
        omp_set_num_threads(threads);
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

}  // namespace

int main(int argc, char** argv) {
    // OpenMPI exports the node-local rank before the program starts.  Fix the
    // affinity early so the OpenMP runtime never caches the launcher's
    // single-core binding.
    int preLocalRank = 0;
    int preLocalSize = 1;
    if (const char* v = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
        preLocalRank = atoi(v);
    }
    if (const char* v = getenv("OMPI_COMM_WORLD_LOCAL_SIZE")) {
        preLocalSize = atoi(v);
    }
    const int preThreads = applyCpuAffinity(preLocalRank, preLocalSize);
    if (preThreads > 0) {
        // Force the OpenMP runtime to size its thread pool while the full
        // node affinity mask is still in effect.
        omp_set_num_threads(preThreads);
#pragma omp parallel
        {
        }
    }

    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    if (n == 0) {
        if (rank == 0) {
            printf("Error: matrix size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Give each MPI rank on a node its own GPU when possible.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
        MPI_Comm_free(&nodeComm);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA-capable device found\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    bool cudaAwareMpi = false;
#ifdef CHOLESKY_HAVE_MPI_EXT
    cudaAwareMpi = (MPIX_Query_cuda_support() == 1);
#endif

    // Change the affinity before the OpenMP runtime caches the process
    // topology, then pin the thread count to this rank's fair share.
    configureCpuAffinity(localRank, localSize);
    omp_set_dynamic(0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t nb = (size_t)kBlockSize;
    const size_t numBlocks = (n + nb - 1) / nb;
    const size_t blocksPerRank =
        (numBlocks + (size_t)numRanks - 1) / (size_t)numRanks;
    const size_t localCapacity = blocksPerRank * nb;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> localA(n * localCapacity, 0.0);
    generateDistributed(localA, n, rank, numRanks, nb, localCapacity,
                        numBlocks);

    std::vector<double> localOrig;
    if (validate) {
        localOrig = localA;
    }

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * n * localCapacity));
    CUDA_CHECK(cudaMemcpy(d_A, localA.data(),
                          sizeof(double) * n * localCapacity,
                          cudaMemcpyHostToDevice));

    if (!validate) {
        localA.clear();
        localA.shrink_to_fit();
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyParallel(d_A, n, rank, numRanks, nb,
                                          localCapacity, numBlocks,
                                          cudaAwareMpi);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        CUDA_CHECK(cudaFree(d_A));
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if (printResults || validate) {
        gatherFactor(d_A, n, rank, numRanks, nb, localCapacity, numBlocks,
                     fullL);
    }

    if (printResults && rank == 0) {
        print_results(fullL, "CholeskyL");
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        std::vector<double> fullOrig;
        gatherOriginal(localOrig, n, rank, numRanks, nb, localCapacity,
                       numBlocks, fullOrig);

        if (rank == 0) {
            bool valid = validateCholesky(fullL, fullOrig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }

            CUDA_CHECK(cudaFree(d_A));
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    MPI_Finalize();
    return 0;
}
