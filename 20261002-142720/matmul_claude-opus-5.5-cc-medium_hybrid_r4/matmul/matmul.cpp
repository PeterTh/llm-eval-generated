// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//   * MPI:    C is split into contiguous row slabs, one per rank. Each rank
//             generates its own rows of A and the full B locally (the matrix
//             initialization is a deterministic function of (i, j)), so no
//             input data has to be communicated. The C slabs are gathered on
//             rank 0 at the end.
//   * OpenMP: host-side initialization is multithreaded; inside a rank, one
//             OpenMP thread drives each GPU assigned to that rank, the rank's
//             slab being split again into per-GPU sub-slabs.
//   * CUDA:   a shared-memory / register tiled FP64 GEMM kernel computes each
//             sub-slab. Every C element is accumulated in strictly increasing
//             k order starting from 0.0 with separately rounded multiply and
//             add, reproducing the floating-point operation sequence of the
//             sequential reference build bit for bit (GCC -O3 -march=native
//             vectorizes the k loop two-wide without contraction and only
//             contracts the scalar tail term k = N-1 for odd N into an FMA).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowEnd) of an NxN matrix into mat (row-major,
// mat[0] corresponds to row rowBegin).
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat.data(), N, 0, N);
}

// ---------------------------------------------------------------------------
// CUDA GEMM kernel: C[M x N] = A[M x N] * B[N x N]  (A has M rows, K == N)
// ---------------------------------------------------------------------------
constexpr int BM = 64;           // block tile rows
constexpr int BN = 64;           // block tile cols
constexpr int BK = 16;           // k tile depth
constexpr int TX = 16;           // threads in x
constexpr int TY = 16;           // threads in y
constexpr int TM = BM / TY;      // rows per thread (4)
constexpr int TN = BN / TX;      // cols per thread (4)
constexpr int NTHREADS = TX * TY;

__global__ void __launch_bounds__(NTHREADS)
gemmKernel(const double* __restrict__ A, const double* __restrict__ B,
           double* __restrict__ C, const size_t M, const size_t N) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    // Load mapping: each thread loads BM*BK/NTHREADS = 4 elements of A and B.
    // A tile (BM x BK): thread loads column aK, rows aR + 16*l
    const int aK = tid % BK;
    const int aR = tid / BK;            // 0..15
    // B tile (BK x BN): thread loads row bK + 4*l, column bC
    const int bC = tid % BN;
    const int bK = tid / BN;            // 0..3

    // The tiled loop covers k < K; for odd N the last term is added with an
    // FMA after the loop (zero padding beyond K leaves the sums unchanged).
    const size_t K = N - (N & 1);
    double aReg[4], bReg[4];

    auto loadGlobal = [&](size_t k0) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const size_t r = rowBase + aR + 16 * l;
            const size_t k = k0 + aK;
            aReg[l] = (r < M && k < K) ? A[r * N + k] : 0.0;
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const size_t k = k0 + bK + 4 * l;
            const size_t c = colBase + bC;
            bReg[l] = (k < K && c < N) ? B[k * N + c] : 0.0;
        }
    };
    auto storeShared = [&](int buf) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) As[buf][aK][aR + 16 * l] = aReg[l];
        #pragma unroll
        for (int l = 0; l < 4; ++l) Bs[buf][bK + 4 * l][bC] = bReg[l];
    };

    const size_t numTiles = (K + BK - 1) / BK;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (size_t t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][kk][ty + TY * i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][kk][tx + TX * j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }

        if (t + 1 < numTiles) storeShared(cur ^ 1);
        __syncthreads();
    }

    if (N & 1) {
        const size_t k = N - 1;
        double a[TM], b[TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            const size_t r = rowBase + ty + TY * i;
            a[i] = (r < M) ? A[r * N + k] : 0.0;
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t c = colBase + tx + TX * j;
            b[j] = (c < N) ? B[k * N + c] : 0.0;
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t r = rowBase + ty + TY * i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t c = colBase + tx + TX * j;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}

// Per-GPU working state
struct GpuWork {
    int device = 0;
    size_t rowBegin = 0;   // relative to the rank's slab
    size_t rows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
};

// Compute rows of C held by this rank, using all GPUs assigned to it.
void matrixMultiplyLocal(const double* A, const double* B, double* C, const size_t N,
                         std::vector<GpuWork>& gpus) {
    const int nGpus = static_cast<int>(gpus.size());
    #pragma omp parallel for num_threads(nGpus) schedule(static, 1)
    for (int g = 0; g < nGpus; ++g) {
        GpuWork& w = gpus[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        if (w.rows == 0) continue;
        const size_t slabBytes = w.rows * N * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(w.dB, B, N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, w.stream));
        CUDA_CHECK(cudaMemcpyAsync(w.dA, A + w.rowBegin * N, slabBytes,
                                   cudaMemcpyHostToDevice, w.stream));
        const dim3 block(TX, TY);
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((w.rows + BM - 1) / BM));
        gemmKernel<<<grid, block, 0, w.stream>>>(w.dA, w.dB, w.dC, w.rows, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C + w.rowBegin * N, w.dC, slabBytes,
                                   cudaMemcpyDeviceToHost, w.stream));
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row decomposition of C across ranks
    std::vector<int> rowCounts(nRanks), rowDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        const size_t b = N * r / nRanks;
        const size_t e = N * (r + 1) / nRanks;
        rowDispls[r] = static_cast<int>(b);
        rowCounts[r] = static_cast<int>(e - b);
    }
    const size_t myRowBegin = rowDispls[rank];
    const size_t myRows = rowCounts[rank];

    // GPU assignment: ranks on a node share that node's GPUs round-robin; if a
    // node has more GPUs than ranks, each rank drives several GPUs.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (localSize >= nDevices) {
        myDevices.push_back(localRank % nDevices);
    } else {
        for (int d = localRank; d < nDevices; d += localSize) myDevices.push_back(d);
    }
    // Never use more GPUs than there are 64-row tiles to hand out
    const size_t maxUseful = std::max<size_t>(1, (myRows + BM - 1) / BM);
    if (myDevices.size() > maxUseful) myDevices.resize(maxUseful);

    // Allocate matrices. Rank 0 holds full A, B and C (needed for output and
    // validation); other ranks hold their slab of A and C plus the full B.
    std::vector<double> A(root ? N * N : myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(root ? N * N : myRows * N);

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    if (root) {
        initMatrix(A, N);
    } else {
        initMatrixRows(A.data(), N, myRowBegin, myRowBegin + myRows);
    }
    initMatrix(B, N);
    const double* myA = A.data() + (root ? myRowBegin * N : 0);
    double* myC = C.data() + (root ? myRowBegin * N : 0);

    // Set up GPUs (context creation, allocation, pinning) outside the timed region
    std::vector<GpuWork> gpus(myDevices.size());
    {
        const size_t nG = gpus.size();
        for (size_t g = 0; g < nG; ++g) {
            // Split this rank's rows into BM-aligned chunks per GPU
            const size_t tiles = (myRows + BM - 1) / BM;
            const size_t tb = tiles * g / nG, te = tiles * (g + 1) / nG;
            const size_t rb = std::min(myRows, tb * BM), re = std::min(myRows, te * BM);
            GpuWork& w = gpus[g];
            w.device = myDevices[g];
            w.rowBegin = rb;
            w.rows = re - rb;
            CUDA_CHECK(cudaSetDevice(w.device));
            CUDA_CHECK(cudaFree(nullptr));
            CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
            if (w.rows > 0) {
                CUDA_CHECK(cudaMalloc(&w.dA, w.rows * N * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&w.dB, N * N * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&w.dC, w.rows * N * sizeof(double)));
            }
        }
    }
    const bool pinB = N > 0;
    const bool pinSlab = myRows > 0;
    if (pinB) CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterPortable));
    if (pinSlab) {
        CUDA_CHECK(cudaHostRegister(const_cast<double*>(myA), myRows * N * sizeof(double),
                                    cudaHostRegisterPortable));
        CUDA_CHECK(cudaHostRegister(myC, myRows * N * sizeof(double), cudaHostRegisterPortable));
    }

    // MPI datatype for one matrix row (keeps gather counts small for large N)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N > 0 ? N : 1), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(myA, B.data(), myC, N, gpus);
    if (root) {
        MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), rowCounts.data(), rowDispls.data(),
                    rowType, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(myC, static_cast<int>(myRows), rowType, nullptr, nullptr, nullptr,
                    rowType, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(maxDuration);

    // Release GPU / pinning resources
    if (pinSlab) {
        CUDA_CHECK(cudaHostUnregister(const_cast<double*>(myA)));
        CUDA_CHECK(cudaHostUnregister(myC));
    }
    if (pinB) CUDA_CHECK(cudaHostUnregister(B.data()));
    for (GpuWork& w : gpus) {
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaFree(w.dA));
        CUDA_CHECK(cudaFree(w.dB));
        CUDA_CHECK(cudaFree(w.dC));
        CUDA_CHECK(cudaStreamDestroy(w.stream));
    }
    MPI_Type_free(&rowType);

    int exitCode = 0;
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
            bool valid = validateResult(A, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
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
