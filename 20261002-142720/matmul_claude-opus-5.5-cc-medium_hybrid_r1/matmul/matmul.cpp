// Hybrid MPI + OpenMP + CUDA matrix multiplication benchmark.
//
// Decomposition:
//  - MPI:    rows of C (and A) are block-distributed over ranks; every rank holds the
//            full B. Inputs are generated locally (deterministic), the result is
//            gathered on rank 0.
//  - OpenMP: host-side initialization / validation, and one host thread per GPU when a
//            rank owns several GPUs (e.g. a single rank per node).
//  - CUDA:   register-tiled DGEMM kernel. Each C element is accumulated sequentially
//            over k = 0..N-1 in the same order and with the same rounding as the
//            reference build of the original loop (separately rounded multiply and
//            add; GCC fuses only the final term into an FMA when N is odd), so the
//            results are bitwise identical to the serial reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [rowBegin, rowEnd) of an NxN matrix into mat (row-major, local rows)
void initMatrixRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = mat + (i - rowBegin) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------------
// CUDA DGEMM kernel: C[M x N] = A[M x N(=K)] * B[N x N], all row-major.
// 64x64 output tile per block, 256 threads, 4x4 outputs per thread, BK = 16.
// ---------------------------------------------------------------------------------
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TX = 16;
constexpr int TY = 16;
constexpr int RM = BM / TY; // 4
constexpr int RN = BN / TX; // 4

__global__ void __launch_bounds__(TX * TY)
dgemmKernel(const double* __restrict__ A, const double* __restrict__ B,
            double* __restrict__ C, const int M, const int N) {
    __shared__ double As[2][BK][BM + 1]; // transposed A tile: As[k][m]
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; ++i)
        #pragma unroll
        for (int j = 0; j < RN; ++j) acc[i][j] = 0.0;

    // Load mapping: A tile 64x16 -> each thread loads 4 elements (k fastest across 16 threads)
    // B tile 16x64 -> each thread loads 4 elements (n fastest across 64 threads)
    const int aK = tid % BK;   // 0..15
    const int aM = tid / BK;   // 0..15, +16*l
    const int bN = tid % BN;   // 0..63
    const int bK = tid / BN;   // 0..3, +4*l

    double ra[4], rb[4];

    auto loadGlobal = [&](int k0) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int r = rowBase + aM + 16 * l;
            const int k = k0 + aK;
            ra[l] = (r < M && k < N) ? A[(size_t)r * N + k] : 0.0;
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int k = k0 + bK + 4 * l;
            const int c = colBase + bN;
            rb[l] = (k < N && c < N) ? B[(size_t)k * N + c] : 0.0;
        }
    };
    auto storeShared = [&](int buf) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) As[buf][aK][aM + 16 * l] = ra[l];
        #pragma unroll
        for (int l = 0; l < 4; ++l) Bs[buf][bK + 4 * l][bN] = rb[l];
    };

    const int numTiles = (N + BK - 1) / BK;
    // For odd N the reference fuses the last term (k = N-1) into an FMA
    const bool fuseLast = (N & 1) != 0;
    const int kUnfused = fuseLast ? N - 1 : N;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        const bool hasNext = (t + 1) < numTiles;
        if (hasNext) loadGlobal((t + 1) * BK);

        // Accumulate (unfused) over real k values < kUnfused only
        const int kLimit = max(0, min(BK, kUnfused - t * BK));
        if (kLimit == BK) {
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                double a[RM], b[RN];
                #pragma unroll
                for (int i = 0; i < RM; ++i) a[i] = As[cur][kk][ty + TY * i];
                #pragma unroll
                for (int j = 0; j < RN; ++j) b[j] = Bs[cur][kk][tx + TX * j];
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
            }
        } else {
            for (int kk = 0; kk < kLimit; ++kk) {
                double a[RM], b[RN];
                #pragma unroll
                for (int i = 0; i < RM; ++i) a[i] = As[cur][kk][ty + TY * i];
                #pragma unroll
                for (int j = 0; j < RN; ++j) b[j] = Bs[cur][kk][tx + TX * j];
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
            }
        }

        if (hasNext) storeShared(cur ^ 1);
        __syncthreads();
    }

    if (fuseLast) {
        // The last tile is still resident in shared memory (no further stores)
        const int buf = (numTiles - 1) & 1;
        const int kk = (N - 1) - (numTiles - 1) * BK;
        #pragma unroll
        for (int i = 0; i < RM; ++i)
            #pragma unroll
            for (int j = 0; j < RN; ++j)
                acc[i][j] = __fma_rn(As[buf][kk][ty + TY * i], Bs[buf][kk][tx + TX * j], acc[i][j]);
    }

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int r = rowBase + ty + TY * i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int c = colBase + tx + TX * j;
            if (c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

// Device-side state for one GPU handling rows [rowBegin, rowBegin + rows) of the
// rank-local block.
struct DeviceWork {
    int device = 0;
    size_t rowOffset = 0; // offset within rank-local rows
    size_t rows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    static constexpr int NSTREAMS = 4;
    cudaStream_t streams[NSTREAMS];
    cudaEvent_t bReady;
};

// Compute local C rows with all GPUs assigned to this rank (one OpenMP thread per GPU).
// Pipeline: B upload, then row chunks of (A upload -> kernel -> C download) on
// several streams so transfers overlap computation.
void matrixMultiplyLocal(std::vector<DeviceWork>& work, const double* hA, const double* hB,
                         double* hC, const size_t N) {
    #pragma omp parallel for num_threads(static_cast<int>(work.size())) schedule(static, 1)
    for (size_t w = 0; w < work.size(); ++w) {
        DeviceWork& dw = work[w];
        CUDA_CHECK(cudaSetDevice(dw.device));
        if (dw.rows == 0) continue;

        CUDA_CHECK(cudaMemcpyAsync(dw.dB, hB, N * N * sizeof(double), cudaMemcpyHostToDevice,
                                   dw.streams[0]));
        CUDA_CHECK(cudaEventRecord(dw.bReady, dw.streams[0]));
        for (int s = 1; s < DeviceWork::NSTREAMS; ++s)
            CUDA_CHECK(cudaStreamWaitEvent(dw.streams[s], dw.bReady, 0));

        // Chunk rows (multiple of BM) across streams
        const size_t minChunk = BM * 4;
        size_t chunk = (dw.rows + DeviceWork::NSTREAMS - 1) / DeviceWork::NSTREAMS;
        chunk = std::max(minChunk, ((chunk + BM - 1) / BM) * BM);

        int s = 0;
        for (size_t r0 = 0; r0 < dw.rows; r0 += chunk, s = (s + 1) % DeviceWork::NSTREAMS) {
            const size_t nr = std::min(chunk, dw.rows - r0);
            const size_t off = (dw.rowOffset + r0) * N;
            cudaStream_t st = dw.streams[s];
            CUDA_CHECK(cudaMemcpyAsync(dw.dA + r0 * N, hA + off, nr * N * sizeof(double),
                                       cudaMemcpyHostToDevice, st));
            dim3 block(TX, TY);
            dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                      static_cast<unsigned>((nr + BM - 1) / BM));
            dgemmKernel<<<grid, block, 0, st>>>(dw.dA + r0 * N, dw.dB, dw.dC + r0 * N,
                                                static_cast<int>(nr), static_cast<int>(N));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hC + off, dw.dC + r0 * N, nr * N * sizeof(double),
                                       cudaMemcpyDeviceToHost, st));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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

    // GPU assignment: ranks on the same node share the node's GPUs round-robin; if a
    // rank owns several GPUs it drives them with one OpenMP thread each.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (numDevices >= localSize) {
        for (int d = localRank; d < numDevices; d += localSize) myDevices.push_back(d);
    } else {
        myDevices.push_back(localRank % numDevices);
    }

    // Row distribution over ranks (balanced, contiguous blocks)
    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t begin = N * r / nranks;
        const size_t end = N * (r + 1) / nranks;
        rowDispls[r] = static_cast<int>(begin);
        rowCounts[r] = static_cast<int>(end - begin);
    }
    const size_t myRowBegin = rowDispls[rank];
    const size_t myRows = rowCounts[rank];

    // Allocate matrices. Rank 0 holds full A and C (needed for output/validation);
    // other ranks hold only their row blocks. Every rank holds full B.
    std::vector<double> A(root ? N * N : myRows * N);
    std::vector<double> B(N * N);
    std::vector<double> C(root ? N * N : myRows * N);

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    if (root) initMatrixRows(A.data(), N, 0, N);
    else      initMatrixRows(A.data(), N, myRowBegin, myRowBegin + myRows);
    initMatrixRows(B.data(), N, 0, N);

    const double* hA = A.data() + (root ? myRowBegin * N : 0);
    double* hC = C.data() + (root ? myRowBegin * N : 0);

    // Pin host buffers for fast async transfers
    if (myRows > 0) {
        CUDA_CHECK(cudaHostRegister(const_cast<double*>(hA), myRows * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(hC, myRows * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterDefault));
    }

    // Split this rank's rows over its GPUs and allocate device memory (untimed setup)
    std::vector<DeviceWork> work(myDevices.size());
    for (size_t w = 0; w < work.size(); ++w) {
        DeviceWork& dw = work[w];
        dw.device = myDevices[w];
        const size_t b = myRows * w / work.size();
        const size_t e = myRows * (w + 1) / work.size();
        dw.rowOffset = b;
        dw.rows = e - b;
        CUDA_CHECK(cudaSetDevice(dw.device));
        if (dw.rows == 0) continue;
        CUDA_CHECK(cudaMalloc(&dw.dA, dw.rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dw.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dw.dC, dw.rows * N * sizeof(double)));
        for (int s = 0; s < DeviceWork::NSTREAMS; ++s)
            CUDA_CHECK(cudaStreamCreateWithFlags(&dw.streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&dw.bReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaFuncSetCacheConfig(dgemmKernel, cudaFuncCachePreferShared));
        // Warm up context / kernel module
        dgemmKernel<<<1, dim3(TX, TY), 0, dw.streams[0]>>>(dw.dA, dw.dB, dw.dC, 0, static_cast<int>(N));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Row datatype so gather counts are in rows (avoids int overflow for large N)
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(work, hA, B.data(), hC, N);
    if (nranks > 1) {
        if (root) {
            MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), rowCounts.data(), rowDispls.data(),
                        rowType, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(C.data(), static_cast<int>(myRows), rowType, nullptr, nullptr, nullptr,
                        rowType, 0, MPI_COMM_WORLD);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MPI_Type_free(&rowType);
    for (auto& dw : work) {
        if (dw.rows == 0) continue;
        CUDA_CHECK(cudaSetDevice(dw.device));
        for (int s = 0; s < DeviceWork::NSTREAMS; ++s) CUDA_CHECK(cudaStreamDestroy(dw.streams[s]));
        CUDA_CHECK(cudaEventDestroy(dw.bReady));
        CUDA_CHECK(cudaFree(dw.dA));
        CUDA_CHECK(cudaFree(dw.dB));
        CUDA_CHECK(cudaFree(dw.dC));
    }
    if (myRows > 0) {
        CUDA_CHECK(cudaHostUnregister(const_cast<double*>(hA)));
        CUDA_CHECK(cudaHostUnregister(hC));
        CUDA_CHECK(cudaHostUnregister(B.data()));
    }

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
