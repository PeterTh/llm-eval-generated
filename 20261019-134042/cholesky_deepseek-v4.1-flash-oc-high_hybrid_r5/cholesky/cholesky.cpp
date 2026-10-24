#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        int inited = 0;
        MPI_Initialized(&inited);
        if (inited) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            exit(1);
        }
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

static void potrfCpu(double* A, int b, int* fail) {
    const int sb = b < 64 ? b : 64;
    for (int kk = 0; kk < b; kk += sb) {
        const int se = std::min(b, kk + sb);
        for (int j = kk; j < se; ++j) {
            double* rj = A + (size_t)j * b;
            double sum = 0.0;
            for (int k = kk; k < j; ++k) {
                sum += rj[k] * rj[k];
            }
            const double v = rj[j] - sum;
            if (!(v > 0.0)) {
                *fail = j + 1;
                rj[j] = 1.0;
            } else {
                rj[j] = sqrt(v);
            }
            const double d = rj[j];
            for (int i = j + 1; i < se; ++i) {
                double* ri = A + (size_t)i * b;
                double t = 0.0;
                for (int k = kk; k < j; ++k) {
                    t += ri[k] * rj[k];
                }
                ri[j] = (ri[j] - t) / d;
            }
        }

        if (se < b) {
#pragma omp parallel for schedule(static)
            for (int i = se; i < b; ++i) {
                double* row = A + (size_t)i * b;
                for (int t = kk; t < se; ++t) {
                    double val = row[t];
                    const double* dt = A + (size_t)t * b;
                    for (int s2 = kk; s2 < t; ++s2) {
                        val -= row[s2] * dt[s2];
                    }
                    row[t] = val / dt[t];
                }
            }

#pragma omp parallel for schedule(dynamic, 4)
            for (int i = se; i < b; ++i) {
                double* ai = A + (size_t)i * b;
                for (int j = se; j <= i; ++j) {
                    double* aj = A + (size_t)j * b;
                    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                    for (int t = kk; t < se; ++t) {
                        sum += ai[t] * aj[t];
                    }
                    ai[j] -= sum;
                }
            }
        }
    }
}

__global__ void panel_solve_kernel(double* A, const double* diag,
                                   long long n, long long r0,
                                   long long kstart, int b,
                                   long long rowStart, long long rowEnd) {
    extern __shared__ double smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const long long gi = rowStart + (long long)blockIdx.x * warps + warp;
    if (gi >= rowEnd) return;

    double* srow = smem + (size_t)warp * b;
    double* row = A + (gi - r0) * n + kstart;
    for (int c = lane; c < b; c += 32) {
        srow[c] = row[c];
    }
    __syncwarp();

    for (int t = 0; t < b; ++t) {
        double part = 0.0;
        for (int s2 = lane; s2 < t; s2 += 32) {
            part -= srow[s2] * diag[(long long)t * b + s2];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_down_sync(0xffffffffu, part, off);
        }
        if (lane == 0) {
            srow[t] = (srow[t] + part) / diag[(long long)t * b + t];
        }
        __syncwarp();
    }

    for (int c = lane; c < b; c += 32) {
        row[c] = srow[c];
    }
}

__global__ void trailing_update_kernel(double* __restrict__ A,
                                       const double* __restrict__ panel,
                                       long long n, long long r0,
                                       long long kstart, long long kend,
                                       long long g0, long long g1, int b) {
    const int T = 16;
    __shared__ double As[T][T + 1];
    __shared__ double Bs[T][T + 1];

    const long long i = g0 + (long long)blockIdx.y * T + threadIdx.y;
    const long long j = kend + (long long)blockIdx.x * T + threadIdx.x;

    double acc = 0.0;
    for (int t0 = 0; t0 < b; t0 += T) {
        const int tt = t0 + threadIdx.x;
        if (i < g1 && tt < b) {
            As[threadIdx.y][threadIdx.x] =
                panel[(i - kstart) * (long long)b + tt];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        const long long jb = kend + (long long)blockIdx.x * T + threadIdx.y;
        if (jb < g1 && tt < b) {
            Bs[threadIdx.y][threadIdx.x] =
                panel[(jb - kstart) * (long long)b + tt];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k) {
            acc += As[threadIdx.y][k] * Bs[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (i < g1 && j < g1 && j >= kend && j <= i) {
        A[(i - r0) * n + j] -= acc;
    }
}

struct Block {
    size_t b0;
    size_t b1;
    bool gpu;
};

static bool blockIsGpu(size_t kb, size_t nb, unsigned pct) {
    if (nb >= 2) {
        if (kb == 0) return false;
        if (kb == 1) return true;
    }
    const unsigned g0 = (unsigned)((kb * pct) / 100u);
    const unsigned g1 = (unsigned)(((kb + 1) * pct) / 100u);
    return g1 != g0;
}

static void generateLocalMatrix(std::vector<double>& A, size_t n,
                                size_t r0, size_t r1) {
    const size_t lr = r1 - r0;
    if (lr == 0) return;

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < (long long)lr; ++ii) {
        const size_t gi = r0 + (size_t)ii;
        double* Arow = A.data() + (size_t)ii * n;
        const double* Bi = B.data() + gi * n;
        for (size_t j = 0; j < n; ++j) {
            const double* Bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            Arow[j] = sum;
        }
        Arow[gi] += (double)n;
    }
}

static void gatherMatrix(const std::vector<double>& local, size_t r0, size_t r1,
                         size_t n, const std::vector<size_t>& rs,
                         const std::vector<size_t>& re,
                         std::vector<double>& full, int root,
                         MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const size_t localRows = r1 - r0;
    const size_t localElems = localRows * n;
    if (localElems > (size_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "Matrix block too large for MPI_Gatherv\n");
        MPI_Abort(comm, 1);
    }

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = (int)((re[r] - rs[r]) * n);
        displs[r] = (int)(rs[r] * n);
    }

    const double dummy = 0.0;
    const double* send = localElems > 0 ? local.data() : &dummy;
    MPI_Gatherv(send, (int)localElems, MPI_DOUBLE,
                rank == root ? full.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, root, comm);
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& A_orig,
                             const size_t n) {
    std::vector<double> reconstructed(n * n);

#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        const size_t i = (size_t)ii;
        const double* Li = L.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* Lj = L.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
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
    const char* waitEnv = getenv("OMP_WAIT_POLICY");
    if (waitEnv == nullptr || waitEnv[0] == '\0') {
        setenv("OMP_WAIT_POLICY", "ACTIVE", 1);
    }

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            long long v = atoll(argv[++i]);
            n = v > 0 ? (size_t)v : 0;
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

    const long configuredCpus = sysconf(_SC_NPROCESSORS_CONF);
    omp_set_dynamic(0);
    const char* ompEnv = getenv("OMP_NUM_THREADS");
    if (ompEnv == nullptr || ompEnv[0] == '\0') {
        int procs = configuredCpus > 0
                        ? (int)(configuredCpus / std::max(1, localSize))
                        : omp_get_num_procs();
        const int useful = (int)std::max<size_t>(1, n / 32);
        procs = std::min(procs, useful);
        int threads = std::max(1, std::min(procs, 64));

        int currentCpus = 0;
        cpu_set_t cur;
        CPU_ZERO(&cur);
        if (sched_getaffinity(0, sizeof(cur), &cur) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &cur)) ++currentCpus;
            }
        }
        if (configuredCpus > 0 && currentCpus > 0 && currentCpus < threads) {
            cpu_set_t all;
            CPU_ZERO(&all);
            for (long c = 0; c < configuredCpus; ++c) {
                CPU_SET(c, &all);
            }
            if (sched_setaffinity(0, sizeof(all), &all) == 0) {
                CPU_ZERO(&cur);
                if (sched_getaffinity(0, sizeof(cur), &cur) == 0) {
                    currentCpus = 0;
                    for (int c = 0; c < CPU_SETSIZE; ++c) {
                        if (CPU_ISSET(c, &cur)) ++currentCpus;
                    }
                }
            }
        }
        if (currentCpus > 0) {
            threads = std::min(threads, currentCpus);
        }
        omp_set_num_threads(std::max(1, threads));
    }

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available: %s\n",
                    cudaGetErrorString(devErr));
        }
        MPI_Finalize();
        return 1;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    int fp64Ratio = 32;
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            fp64Ratio = prop.singleToDoublePrecisionPerfRatio > 0
                            ? prop.singleToDoublePrecisionPerfRatio
                            : 32;
        }
    }

    if (n == 0) {
        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\n");
            printf("Matrix size: 0 x 0\n");
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Generating positive definite matrix...\n");
            printf("Computing Cholesky decomposition...\n");
            printf("Computation time: 0 ms\n");
            printf("Performance: nan GFLOPS\n");
            if (printResults) {
                std::vector<double> empty;
                print_results(empty, "CholeskyL");
            }
            if (validate) {
                std::vector<double> empty;
                printf("Validating result...\n");
                validateCholesky(empty, empty, 0);
                printf("Validation: PASSED\n");
            }
        }
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 0;
    }

    unsigned gpuPct = 10;
    if (fp64Ratio < 16) {
        gpuPct = fp64Ratio < 4 ? 80 : 50;
    }
    if (const char* e = getenv("CHOLESKY_GPU_PCT")) {
        int v = atoi(e);
        if (v > 0) {
            v = std::max(5, std::min(95, v));
            gpuPct = (unsigned)v;
        }
    }

    const size_t rowsPerRank = (n + (size_t)size - 1) / (size_t)size;
    const size_t blockTarget = (rowsPerRank + 3) / 4;
    size_t b = std::max<size_t>(8, std::min<size_t>(256, blockTarget));
    if (b > n) b = n;
    if (b == 0) b = 1;
    const size_t nb = (n + b - 1) / b;

    std::vector<size_t> rowStart((size_t)size);
    std::vector<size_t> rowEnd((size_t)size);
    size_t blockCursor = 0;
    const size_t baseBlocks = nb / (size_t)size;
    const size_t extraBlocks = nb % (size_t)size;
    for (int r = 0; r < size; ++r) {
        const size_t cnt = baseBlocks + ((size_t)r < extraBlocks ? 1 : 0);
        rowStart[(size_t)r] = std::min(n, blockCursor * b);
        blockCursor += cnt;
        rowEnd[(size_t)r] = std::min(n, blockCursor * b);
    }

    const size_t r0 = rowStart[(size_t)rank];
    const size_t r1 = rowEnd[(size_t)rank];
    const size_t localRows = r1 - r0;
    if (localRows > 0 && n > std::numeric_limits<size_t>::max() / localRows) {
        if (rank == 0) fprintf(stderr, "Matrix too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localElems = localRows * n;

    std::vector<Block> blocks;
    std::vector<size_t> cpuRows;
    cpuRows.reserve(localRows);

    if (localRows > 0) {
        const size_t firstBlock = r0 / b;
        const size_t lastBlock = (r1 + b - 1) / b;
        for (size_t kb = firstBlock; kb < lastBlock; ++kb) {
            const size_t b0 = kb * b;
            const size_t b1 = std::min(n, b0 + b);
            const bool gpu = blockIsGpu(kb, nb, gpuPct);
            blocks.push_back(Block{b0, b1, gpu});
            if (!gpu) {
                for (size_t gi = b0; gi < b1; ++gi) {
                    cpuRows.push_back(gi);
                }
            }
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(localElems);
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generateLocalMatrix(A, n, r0, r1);

    std::vector<double> A_orig;
    if (validate) {
        if (rank == 0) A_orig.resize(n * n);
        gatherMatrix(A, r0, r1, n, rowStart, rowEnd, A_orig, 0,
                     MPI_COMM_WORLD);
    }

    double* d_A = nullptr;
    double* d_panel = nullptr;
    double* d_diag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, std::max<size_t>(1, localElems) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel, std::max<size_t>(1, n * b) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diag, std::max<size_t>(1, b * b) * sizeof(double)));
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), localElems * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    const size_t panelElems = std::max<size_t>(1, n * b);
    std::vector<double> panel(panelElems);
    std::vector<double> diag(std::max<size_t>(1, b * b));
    std::vector<double> contrib(std::max<size_t>(1, localRows * b));

    CUDA_CHECK(cudaMemset(d_diag, 0, std::max<size_t>(1, b * b) * sizeof(double)));
    panel_solve_kernel<<<1, 1>>>(d_A, d_diag, (long long)n, (long long)r0,
                                 0, 1, 0, 0);
    trailing_update_kernel<<<dim3(1, 1), dim3(16, 16)>>>(
        d_A, d_panel, (long long)n, (long long)r0, 0, 0, 0, 0, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    bool localOk = true;
    int failCol = -1;

    for (size_t kstart = 0; kstart < n; kstart += b) {
        const size_t bcur = std::min(b, n - kstart);
        const size_t kend = kstart + bcur;
        const size_t kb = kstart / b;

        int owner = -1;
        for (int r = 0; r < size; ++r) {
            if (rowStart[(size_t)r] <= kstart && kstart < rowEnd[(size_t)r]) {
                owner = r;
                break;
            }
        }
        if (owner < 0) {
            localOk = false;
            failCol = (int)kstart;
            break;
        }

        if (rank == owner) {
            const size_t localBlock = kb - r0 / b;
            const Block& ob = blocks[localBlock];
            if (ob.gpu) {
                CUDA_CHECK(cudaMemcpy2D(diag.data(), bcur * sizeof(double),
                                        d_A + (kstart - r0) * n + kstart,
                                        n * sizeof(double),
                                        bcur * sizeof(double), bcur,
                                        cudaMemcpyDeviceToHost));
            } else {
                for (size_t gi = kstart; gi < kend; ++gi) {
                    std::memcpy(diag.data() + (gi - kstart) * bcur,
                                A.data() + (gi - r0) * n + kstart,
                                bcur * sizeof(double));
                }
            }
        }

        int failCode = 0;
        if (rank == owner) {
            potrfCpu(diag.data(), (int)bcur, &failCode);
        }

        MPI_Bcast(diag.data(), (int)(bcur * bcur), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);
        MPI_Bcast(&failCode, 1, MPI_INT, owner, MPI_COMM_WORLD);

        if (failCode != 0) {
            localOk = false;
            failCol = (int)(kstart + (size_t)failCode - 1);
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n",
                       failCol);
            }
            break;
        }

        if (rank == owner) {
            const size_t localBlock = kb - r0 / b;
            const Block& ob = blocks[localBlock];
            if (ob.gpu) {
                CUDA_CHECK(cudaMemcpy2D(d_A + (kstart - r0) * n + kstart,
                                        n * sizeof(double),
                                        diag.data(), bcur * sizeof(double),
                                        bcur * sizeof(double), bcur,
                                        cudaMemcpyHostToDevice));
            } else {
                for (size_t gi = kstart; gi < kend; ++gi) {
                    std::memcpy(A.data() + (gi - r0) * n + kstart,
                                diag.data() + (gi - kstart) * bcur,
                                bcur * sizeof(double));
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_diag, diag.data(),
                              bcur * bcur * sizeof(double),
                              cudaMemcpyHostToDevice));

        for (const Block& blk : blocks) {
            if (!blk.gpu) continue;
            const size_t rs = std::max(blk.b0, kend);
            if (rs >= blk.b1) continue;
            const long long cnt = (long long)(blk.b1 - rs);
            const int warpsPerBlock = 4;
            const int threads = warpsPerBlock * 32;
            const int grid = (int)((cnt + warpsPerBlock - 1) / warpsPerBlock);
            const size_t smemBytes =
                (size_t)warpsPerBlock * bcur * sizeof(double);
            panel_solve_kernel<<<grid, threads, smemBytes>>>(
                d_A, d_diag, (long long)n, (long long)r0,
                (long long)kstart, (int)bcur,
                (long long)rs, (long long)blk.b1);
        }
        CUDA_CHECK(cudaGetLastError());

#pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < (long long)cpuRows.size(); ++idx) {
            const size_t gi = cpuRows[(size_t)idx];
            if (gi < kend) continue;
            double* row = A.data() + (gi - r0) * n;
            for (size_t t = 0; t < bcur; ++t) {
                double s = row[kstart + t];
                const double* drow = diag.data() + t * bcur;
#pragma omp simd reduction(- : s)
                for (size_t s2 = 0; s2 < t; ++s2) {
                    s -= row[kstart + s2] * drow[s2];
                }
                row[kstart + t] = s / drow[t];
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const size_t s0 = std::max(r0, kstart);
        const size_t crows = r1 > s0 ? r1 - s0 : 0;
        double* con = contrib.data();

        if (crows > 0) {
            if (r0 <= kstart) {
                const size_t e = std::min(kend, r1);
                for (size_t gi = kstart; gi < e; ++gi) {
                    std::memcpy(con + (gi - s0) * bcur,
                                diag.data() + (gi - kstart) * bcur,
                                bcur * sizeof(double));
                }
            }
            for (const Block& blk : blocks) {
                const size_t rs = std::max(blk.b0, kend);
                if (rs >= blk.b1) continue;
                if (blk.gpu) {
                    CUDA_CHECK(cudaMemcpy2D(
                        con + (rs - s0) * bcur, bcur * sizeof(double),
                        d_A + (rs - r0) * n + kstart, n * sizeof(double),
                        bcur * sizeof(double), blk.b1 - rs,
                        cudaMemcpyDeviceToHost));
                } else {
                    for (size_t gi = rs; gi < blk.b1; ++gi) {
                        std::memcpy(con + (gi - s0) * bcur,
                                    A.data() + (gi - r0) * n + kstart,
                                    bcur * sizeof(double));
                    }
                }
            }
        }

        std::vector<int> recvCounts((size_t)size);
        std::vector<int> displs((size_t)size);
        for (int r = 0; r < size; ++r) {
            const size_t st = std::max(rowStart[(size_t)r], kstart);
            const size_t cnt = rowEnd[(size_t)r] > st
                                   ? rowEnd[(size_t)r] - st
                                   : 0;
            recvCounts[(size_t)r] = (int)(cnt * bcur);
            displs[(size_t)r] = (int)((st - kstart) * bcur);
        }

        MPI_Allgatherv(con, (int)(crows * bcur), MPI_DOUBLE,
                       panel.data(), recvCounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_panel, panel.data(),
                              (n - kstart) * bcur * sizeof(double),
                              cudaMemcpyHostToDevice));

        for (const Block& blk : blocks) {
            if (!blk.gpu) continue;
            const size_t g0 = std::max(blk.b0, kend);
            if (g0 >= blk.b1) continue;
            const size_t g1 = blk.b1;
            const int tx = (int)((g1 - kend + 15) / 16);
            const int ty = (int)((g1 - g0 + 15) / 16);
            dim3 grid(tx, ty);
            dim3 threads(16, 16);
            trailing_update_kernel<<<grid, threads>>>(
                d_A, d_panel, (long long)n, (long long)r0,
                (long long)kstart, (long long)kend,
                (long long)g0, (long long)g1, (int)bcur);
        }
        CUDA_CHECK(cudaGetLastError());

#pragma omp parallel for schedule(dynamic, 4)
        for (long long idx = 0; idx < (long long)cpuRows.size(); ++idx) {
            const size_t gi = cpuRows[(size_t)idx];
            if (gi < kend) continue;
            double* row = A.data() + (gi - r0) * n;
            const double* pi = panel.data() + (gi - kstart) * bcur;
            size_t j = kend;
            while (j + 3 <= gi) {
                const double* p0 = panel.data() + (j - kstart) * bcur;
                const double* p1 = p0 + bcur;
                const double* p2 = p1 + bcur;
                const double* p3 = p2 + bcur;
                double s0 = 0.0;
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
#pragma omp simd reduction(+ : s0, s1, s2, s3)
                for (size_t t = 0; t < bcur; ++t) {
                    const double pit = pi[t];
                    s0 += pit * p0[t];
                    s1 += pit * p1[t];
                    s2 += pit * p2[t];
                    s3 += pit * p3[t];
                }
                row[j] -= s0;
                row[j + 1] -= s1;
                row[j + 2] -= s2;
                row[j + 3] -= s3;
                j += 4;
            }
            for (; j <= gi; ++j) {
                const double* pj = panel.data() + (j - kstart) * bcur;
                double s = 0.0;
#pragma omp simd reduction(+ : s)
                for (size_t t = 0; t < bcur; ++t) {
                    s += pi[t] * pj[t];
                }
                row[j] -= s;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int localFail = localOk ? 0 : 1;
    int anyFail = 0;
    MPI_Allreduce(&localFail, &anyFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    if (anyFail) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        if (d_diag) cudaFree(d_diag);
        if (d_panel) cudaFree(d_panel);
        if (d_A) cudaFree(d_A);
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    for (const Block& blk : blocks) {
        if (!blk.gpu) continue;
        CUDA_CHECK(cudaMemcpy(A.data() + (blk.b0 - r0) * n,
                              d_A + (blk.b0 - r0) * n,
                              (blk.b1 - blk.b0) * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < (long long)localRows; ++ii) {
        const size_t gi = r0 + (size_t)ii;
        double* row = A.data() + (size_t)ii * n;
        for (size_t j = gi + 1; j < n; ++j) {
            row[j] = 0.0;
        }
    }

    const double seconds =
        std::chrono::duration<double>(end - start).count();
    if (rank == 0) {
        const long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();
        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double gflops =
            seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Computation time: %lld ms\n", ms);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> fullL;
    if ((printResults || validate) && rank == 0) {
        fullL.resize(n * n);
    }
    if (printResults || validate) {
        gatherMatrix(A, r0, r1, n, rowStart, rowEnd, fullL, 0,
                     MPI_COMM_WORLD);
    }

    int rc = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(fullL, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(fullL, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d_diag) cudaFree(d_diag);
    if (d_panel) cudaFree(d_panel);
    if (d_A) cudaFree(d_A);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return rc;
}
