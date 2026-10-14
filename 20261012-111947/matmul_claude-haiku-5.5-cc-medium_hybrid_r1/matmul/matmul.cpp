#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Tiling parameters
constexpr size_t CPU_KC = 256;   // k-block for the CPU kernel
constexpr size_t CPU_JC = 512;   // j-block for the CPU kernel
constexpr size_t CPU_MR = 4;     // C rows updated together by the CPU micro-kernel

constexpr int GPU_THREADS = 16;  // threads per block dimension
constexpr int GPU_TM = 8;        // C rows per thread
constexpr int GPU_TN = 8;        // C columns per thread
constexpr int GPU_BM = GPU_THREADS * GPU_TM;  // C rows per thread block
constexpr int GPU_BN = GPU_THREADS * GPU_TN;  // C columns per thread block
constexpr int GPU_BK = 16;       // k-tile depth

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// First row owned by rank r when N rows are split into contiguous blocks
size_t rowStart(const size_t N, const int nranks, const int r) {
    return static_cast<size_t>(r) * (N / nranks) + std::min<size_t>(r, N % nranks);
}

// Fill local rows [rowBegin, rowEnd) of a matrix; mat holds only those rows
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// CPU kernel: C = A * B for `rows` rows, row-major. Every C element is accumulated over k in
// increasing order as `sum += a * b` (no fused multiply-add, see -ffp-contract=off in CMakeLists.txt),
// the same sequence as the GPU kernel, so results match bit-for-bit.
void cpuMatrixMultiplyRows(const double* __restrict__ A, const double* __restrict__ B,
                           double* __restrict__ C, const size_t rows, const size_t N) {
    std::fill(C, C + rows * N, 0.0);
    for (size_t jb = 0; jb < N; jb += CPU_JC) {
        const size_t jn = std::min(CPU_JC, N - jb);
        for (size_t kb = 0; kb < N; kb += CPU_KC) {
            const size_t kn = std::min(CPU_KC, N - kb);
            size_t i = 0;
            for (; i + CPU_MR <= rows; i += CPU_MR) {
                double* __restrict__ c0 = C + (i + 0) * N + jb;
                double* __restrict__ c1 = C + (i + 1) * N + jb;
                double* __restrict__ c2 = C + (i + 2) * N + jb;
                double* __restrict__ c3 = C + (i + 3) * N + jb;
                const double* a = A + i * N;
                for (size_t k = kb; k < kb + kn; ++k) {
                    const double a0 = a[k];
                    const double a1 = a[N + k];
                    const double a2 = a[2 * N + k];
                    const double a3 = a[3 * N + k];
                    const double* __restrict__ b = B + k * N + jb;
                    for (size_t j = 0; j < jn; ++j) {
                        const double bj = b[j];
                        c0[j] += a0 * bj;
                        c1[j] += a1 * bj;
                        c2[j] += a2 * bj;
                        c3[j] += a3 * bj;
                    }
                }
            }
            for (; i < rows; ++i) {
                double* __restrict__ c = C + i * N + jb;
                const double* a = A + i * N;
                for (size_t k = kb; k < kb + kn; ++k) {
                    const double a0 = a[k];
                    const double* __restrict__ b = B + k * N + jb;
                    for (size_t j = 0; j < jn; ++j) {
                        c[j] += a0 * b[j];
                    }
                }
            }
        }
    }
}

// GPU kernel: C[M x N] = A[M x N] * B[N x N], row-major. Shared-memory tiling with 8x8 register
// blocking per thread. The k loop runs in increasing order for every element, as on the CPU.
// Products and sums are rounded separately (__dmul_rn/__dadd_rn) to match `sum += a * b`.
__global__ void __launch_bounds__(GPU_THREADS * GPU_THREADS)
gemmTileKernel(const double* __restrict__ A, const double* __restrict__ B,
               double* __restrict__ C, const size_t M, const size_t N) {
    __shared__ double As[GPU_BK][GPU_BM + 1];  // +1 pads away bank conflicts on the transposed store
    __shared__ double Bs[GPU_BK][GPU_BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * GPU_THREADS + tx;
    const int nthreads = GPU_THREADS * GPU_THREADS;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * GPU_BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * GPU_BN;

    double acc[GPU_TM][GPU_TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += GPU_BK) {
        for (int t = tid; t < GPU_BM * GPU_BK; t += nthreads) {
            const int r = t / GPU_BK;
            const int c = t % GPU_BK;
            const size_t gr = rowBase + r;
            const size_t gk = k0 + c;
            As[c][r] = (gr < M && gk < N) ? A[gr * N + gk] : 0.0;
        }
        for (int t = tid; t < GPU_BK * GPU_BN; t += nthreads) {
            const int r = t / GPU_BN;
            const int c = t % GPU_BN;
            const size_t gk = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

        const size_t kmax = (N - k0 < GPU_BK) ? N - k0 : GPU_BK;
        for (size_t kk = 0; kk < kmax; ++kk) {
            double a[GPU_TM];
            double b[GPU_TN];
            // Thread (tx, ty) owns rows ty + 16x and columns tx + 16y, so warps read conflict-free
            for (int x = 0; x < GPU_TM; ++x) a[x] = As[kk][ty + x * GPU_THREADS];
            for (int y = 0; y < GPU_TN; ++y) b[y] = Bs[kk][tx + y * GPU_THREADS];
            for (int x = 0; x < GPU_TM; ++x) {
                for (int y = 0; y < GPU_TN; ++y) {
                    acc[x][y] = __dadd_rn(acc[x][y], __dmul_rn(a[x], b[y]));
                }
            }
        }
        __syncthreads();
    }

    for (int x = 0; x < GPU_TM; ++x) {
        const size_t gr = rowBase + ty + x * GPU_THREADS;
        if (gr >= M) continue;
        for (int y = 0; y < GPU_TN; ++y) {
            const size_t gc = colBase + tx + y * GPU_THREADS;
            if (gc < N) C[gr * N + gc] = acc[x][y];
        }
    }
}

// Shared row-chunk scheduler for one rank. GPU feeders and CPU threads each claim the next
// block of rows, so faster workers naturally process more rows.
struct RowScheduler {
    RowScheduler(size_t begin, size_t end, size_t gpuGrab, size_t cpuGrab)
        : next(begin), end(end), gpuGrab(gpuGrab), cpuGrab(cpuGrab) {}

    std::atomic<size_t> next;
    const size_t end;
    const size_t gpuGrab;
    const size_t cpuGrab;
};

// Data for one rank: local rows of A and C (row 0 == global row rowBegin), full B
struct RankWork {
    const double* A;
    const double* B;
    double* C;
    size_t N;
    size_t rowBegin;
};

// Rows per chunk: GPU chunks are large enough to fill the device, CPU chunks are small for balance
size_t gpuGrabRows(const size_t localRows, const size_t nGpu) {
    return std::clamp<size_t>((localRows / (4 * nGpu) + GPU_BM - 1) / GPU_BM * GPU_BM, GPU_BM, 4 * GPU_BM);
}

size_t cpuGrabRows(const size_t localRows, const size_t cores) {
    return std::clamp<size_t>(localRows / (4 * cores) / 4 * 4, 4, 64);
}

// Device state for one GPU. B stays resident on the device; A and C chunks are streamed through.
struct GpuState {
    int device;
    double* dA;
    double* dB;
    double* dC;
};

// Allocates device buffers for chunks of up to `grab` rows and uploads B (one thread per GPU)
std::vector<GpuState> setupGpus(const std::vector<int>& devices, const RankWork& w, const size_t grab) {
    std::vector<GpuState> gpus(devices.size());
    const size_t N = w.N;
    #pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < gpus.size(); ++i) {
        GpuState& g = gpus[i];
        g.device = devices[i];
        CUDA_CHECK(cudaSetDevice(g.device));
        CUDA_CHECK(cudaMalloc(&g.dA, grab * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.dC, grab * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(g.dB, w.B, N * N * sizeof(double), cudaMemcpyHostToDevice));
    }
    return gpus;
}

void releaseGpus(const std::vector<GpuState>& gpus) {
    for (const GpuState& g : gpus) {
        CUDA_CHECK(cudaSetDevice(g.device));
        CUDA_CHECK(cudaFree(g.dA));
        CUDA_CHECK(cudaFree(g.dB));
        CUDA_CHECK(cudaFree(g.dC));
    }
}

// OpenMP thread driving one GPU: uploads each claimed chunk of A, multiplies, downloads C
void gpuWorker(const GpuState& g, const RankWork& w, RowScheduler& sched) {
    CUDA_CHECK(cudaSetDevice(g.device));

    const size_t N = w.N;
    const dim3 block(GPU_THREADS, GPU_THREADS);
    for (;;) {
        const size_t r0 = sched.next.fetch_add(sched.gpuGrab);
        if (r0 >= sched.end) break;
        const size_t rows = std::min(sched.gpuGrab, sched.end - r0);
        const size_t local = r0 - w.rowBegin;

        CUDA_CHECK(cudaMemcpy(g.dA, w.A + local * N, rows * N * sizeof(double), cudaMemcpyHostToDevice));
        const dim3 grid((N + GPU_BN - 1) / GPU_BN, (rows + GPU_BM - 1) / GPU_BM);
        gemmTileKernel<<<grid, block>>>(g.dA, g.dB, g.dC, rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(w.C + local * N, g.dC, rows * N * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

// OpenMP thread running the CPU kernel on the row chunks it claims
void cpuWorker(const RankWork& w, RowScheduler& sched) {
    for (;;) {
        const size_t r0 = sched.next.fetch_add(sched.cpuGrab);
        if (r0 >= sched.end) break;
        const size_t rows = std::min(sched.cpuGrab, sched.end - r0);
        const size_t local = r0 - w.rowBegin;
        cpuMatrixMultiplyRows(w.A + local * w.N, w.B, w.C + local * w.N, rows, w.N);
    }
}

// Computes this rank's rows of C: one OpenMP thread per GPU drives it, the remaining threads run the CPU kernel
void computeLocalRows(const RankWork& w, const size_t rowEnd,
                      const std::vector<GpuState>& gpus, const int cores) {
    const size_t localRows = rowEnd - w.rowBegin;
    if (localRows == 0 || w.N == 0) return;

    const int nGpu = static_cast<int>(gpus.size());
    const int nTeam = std::max(cores, nGpu);
    RowScheduler sched(w.rowBegin, rowEnd, gpuGrabRows(localRows, nGpu),
                       cpuGrabRows(localRows, cores));

    #pragma omp parallel num_threads(nTeam)
    {
        const int tid = omp_get_thread_num();
        if (tid < nGpu) {
            gpuWorker(gpus[tid], w, sched);
        } else {
            cpuWorker(w, sched);
        }
    }
}

// Simple validation: compute a single element and compare.
// A is regenerated from its defining formula, since each rank only stores its own rows.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = rank == 0;

    // Ranks on the same node share its GPUs and CPU cores
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = -1;
    for (int i = 1; i < argc && exitCode < 0; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitCode = 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    // Every rank binds to GPUs: devices are split round-robin among the ranks of this node
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        if (root) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    for (int d = 0; d < ndev; ++d) {
        if (d % localSize == localRank) devices.push_back(d);
    }
    if (devices.empty()) devices.push_back(localRank % ndev);

    // Each rank gets an equal share of this node's cores (unless OMP_NUM_THREADS is set)
    int cores = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        cores = std::max(1, omp_get_num_procs() / localSize);
    }
    omp_set_num_threads(cores);

    if (N * N > static_cast<size_t>(INT_MAX)) {
        if (root) fprintf(stderr, "Matrix size too large for MPI result gathering\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks, %d OpenMP threads/rank, %zu GPU(s) on rank 0\n",
               nranks, cores, devices.size());
    }

    // Each rank owns a contiguous block of rows of A and C; B is needed in full
    const size_t rowBegin = rowStart(N, nranks, rank);
    const size_t rowEnd = rowStart(N, nranks, rank + 1);
    const size_t localRows = rowEnd - rowBegin;

    std::vector<double> B(N * N);
    std::vector<double> A(localRows * N);
    std::vector<double> C;       // full result, only on rank 0
    std::vector<double> localC;  // this rank's rows of C on other ranks
    if (root) {
        C.resize(N * N);
    } else {
        localC.resize(localRows * N);
    }
    double* Cloc = root ? C.data() : localC.data();

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initMatrixRows(B.data(), N, 0, N);
    initMatrixRows(A.data(), N, rowBegin, rowEnd);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    const RankWork work{A.data(), B.data(), Cloc, N, rowBegin};

    // GPU setup (context creation, device buffers, upload of B) is not part of the timed computation
    std::vector<GpuState> gpus = setupGpus(devices, work, gpuGrabRows(localRows, devices.size()));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    computeLocalRows(work, rowEnd, gpus, cores);

    // Gather the row blocks of C on rank 0 (rank 0 computed its block in place)
    if (root) {
        std::vector<int> counts(nranks);
        std::vector<int> displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            displs[r] = static_cast<int>(rowStart(N, nranks, r) * N);
            counts[r] = static_cast<int>((rowStart(N, nranks, r + 1) - rowStart(N, nranks, r)) * N);
        }
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(Cloc, static_cast<int>(localRows * N), MPI_DOUBLE, nullptr, nullptr, nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    releaseGpus(gpus);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return root && exitCode > 0 ? exitCode : 0;
}
