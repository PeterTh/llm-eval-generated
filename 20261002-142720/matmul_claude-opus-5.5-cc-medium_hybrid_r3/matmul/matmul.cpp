#include <algorithm>
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

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Device-side initialization of rows [rowBegin, rowBegin + rows) of an NxN matrix
__global__ void initMatrixKernel(double* __restrict__ mat, const size_t N,
                                 const size_t rowBegin, const size_t rows) {
    const size_t total = rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t i = idx / N;
        const size_t j = idx - i * N;
        mat[idx] = getPseudoRndValue(N, rowBegin + i, j);
    }
}

// Tiled FP64 GEMM: C[rows x N] = A[rows x N] * B[N x N].
// Each thread computes a TM x TN micro-tile; every output element accumulates
// over k in ascending order (same summation order as the reference loop).
constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TX = 16, TY = 16;          // thread block dims
constexpr int TM = BM / TY, TN = BN / TX; // 4 x 4 per thread

__global__ void __launch_bounds__(TX * TY)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[2][BK][BM + 1];
    __shared__ double Bs[2][BK][BN];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const size_t rowBase = (size_t)blockIdx.y * BM;
    const size_t colBase = (size_t)blockIdx.x * BN;

    double acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] = 0.0;

    // Load mapping: 256 threads load 64x16 (A) and 16x64 (B) = 4 elements each
    constexpr int LOADS = (BM * BK) / (TX * TY);
    double ra[LOADS], rb[LOADS];

    auto loadGlobal = [&](size_t k0) {
        #pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int e = tid + l * TX * TY;
            // A tile: BM rows x BK cols, k fastest for coalescing
            const int ar = e / BK, ac = e % BK;
            const size_t gr = rowBase + ar, gk = k0 + ac;
            ra[l] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
            // B tile: BK rows x BN cols, column fastest
            const int br = e / BN, bc = e % BN;
            const size_t gkb = k0 + br, gc = colBase + bc;
            rb[l] = (gkb < N && gc < N) ? B[gkb * N + gc] : 0.0;
        }
    };
    auto storeShared = [&](int buf) {
        #pragma unroll
        for (int l = 0; l < LOADS; ++l) {
            const int e = tid + l * TX * TY;
            As[buf][e % BK][e / BK] = ra[l];
            Bs[buf][e / BN][e % BN] = rb[l];
        }
    };

    const size_t numTiles = (N + BK - 1) / BK;
    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    for (size_t t = 0; t < numTiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < numTiles) loadGlobal((t + 1) * BK);

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[cur][k][ty + m * TY];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = Bs[cur][k][tx + n * TX];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n) acc[m][n] = fma(a[m], b[n], acc[m][n]);
        }

        if (t + 1 < numTiles) storeShared(cur ^ 1);
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const size_t r = rowBase + ty + m * TY;
        if (r >= rows) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const size_t c = colBase + tx + n * TX;
            if (c < N) C[r * N + c] = acc[m][n];
        }
    }
}

// A chunk of consecutive rows of C; chunks are the unit of GPU work and of the
// pipelined MPI transfer to the root rank.
struct Chunk {
    size_t rowBegin = 0;     // global row index
    size_t rows = 0;
    size_t localOffset = 0;  // row offset inside the owning GPU's buffers
    cudaEvent_t done = nullptr;
};

// Per-GPU work context
struct GpuWork {
    int device = 0;
    size_t rows = 0;         // total rows over all chunks of this GPU
    std::vector<int> chunks; // indices into the rank's chunk list
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
};

// Deterministic split of a rank's row block into pipeline chunks
// (must be computable by the root for every rank)
std::vector<Chunk> makeChunks(const size_t rowBegin, const size_t rows) {
    constexpr size_t maxChunks = 16;
    size_t k = rows / (4 * BM);
    k = std::max<size_t>(1, std::min(k, maxChunks));
    std::vector<Chunk> chunks;
    if (rows == 0) return chunks;
    for (size_t c = 0; c < k; ++c) {
        Chunk ch;
        ch.rowBegin = rowBegin + rows * c / k;
        ch.rows = rows * (c + 1) / k - rows * c / k;
        chunks.push_back(ch);
    }
    return chunks;
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

    // Node-local rank -> GPU assignment (GPUs on a node are shared round-robin
    // among the ranks on that node; each rank drives its GPUs with OpenMP threads)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (localSize >= numDevices) {
        myDevices.push_back(localRank % numDevices);
    } else {
        for (int d = localRank; d < numDevices; d += localSize) myDevices.push_back(d);
    }
    const int numMyGpus = static_cast<int>(myDevices.size());

    // Row-block decomposition of C (and A) over ranks
    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t b = N * r / nranks, e = N * (r + 1) / nranks;
        rowCounts[r] = static_cast<int>(e - b);
        rowDispls[r] = static_cast<int>(b);
    }
    const size_t myRowBegin = rowDispls[rank];
    const size_t myRows = rowCounts[rank];

    // Split this rank's rows into pipeline chunks, distributed round-robin over its GPUs
    std::vector<Chunk> chunks = makeChunks(myRowBegin, myRows);
    const int numChunks = static_cast<int>(chunks.size());
    std::vector<GpuWork> work(numMyGpus);
    for (int g = 0; g < numMyGpus; ++g) work[g].device = myDevices[g];
    for (int c = 0; c < numChunks; ++c) {
        GpuWork& w = work[c % numMyGpus];
        chunks[c].localOffset = w.rows;
        w.rows += chunks[c].rows;
        w.chunks.push_back(c);
    }

    // Allocate matrices (full matrices only on root; others hold their C block)
    std::vector<double> A, B, C;
    double* localC = nullptr;
    if (root) {
        C.resize(N * N);
        localC = C.data();
        // Pin C so device->host copies and MPI receives run at full bandwidth
        if (N > 0) CUDA_CHECK(cudaHostRegister(C.data(), N * N * sizeof(double), cudaHostRegisterPortable));
    } else if (myRows > 0) {
        CUDA_CHECK(cudaMallocHost(&localC, myRows * N * sizeof(double)));
    }

    // Initialize matrices (on the GPUs; host copies on root only for validation)
    if (root) printf("Initializing matrices...\n");
    if (root && validate) {
        A.resize(N * N);
        B.resize(N * N);
        initMatrix(A, N);
        initMatrix(B, N);
    }

    #pragma omp parallel num_threads(numMyGpus)
    {
        GpuWork& w = work[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
        const size_t rowsAlloc = w.rows > 0 ? w.rows : 1;
        CUDA_CHECK(cudaMalloc(&w.dA, rowsAlloc * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.dC, rowsAlloc * N * sizeof(double)));
        constexpr int initThreads = 256;
        constexpr int initBlocks = 1024;
        for (const int c : w.chunks) {
            Chunk& ch = chunks[c];
            CUDA_CHECK(cudaEventCreateWithFlags(&ch.done, cudaEventDisableTiming));
            initMatrixKernel<<<initBlocks, initThreads, 0, w.stream>>>(
                w.dA + ch.localOffset * N, N, ch.rowBegin, ch.rows);
        }
        initMatrixKernel<<<initBlocks, initThreads, 0, w.stream>>>(w.dB, N, 0, N);
        CUDA_CHECK(cudaGetLastError());
        // Warm-up launch so the timed run excludes one-time module loading costs
        matmulKernel<<<dim3(1, 1), dim3(TX, TY), 0, w.stream>>>(w.dA, w.dB, w.dC, 0, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    }

    // Row type so that counts/displacements stay small for very large N
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Root pre-posts receives for every chunk of every other rank directly into C
    std::vector<MPI_Request> requests;
    if (root) {
        for (int r = 1; r < nranks; ++r) {
            const std::vector<Chunk> remote = makeChunks(rowDispls[r], rowCounts[r]);
            for (size_t c = 0; c < remote.size(); ++c) {
                requests.emplace_back();
                MPI_Irecv(C.data() + remote[c].rowBegin * N, static_cast<int>(remote[c].rows),
                          rowType, r, static_cast<int>(c), MPI_COMM_WORLD, &requests.back());
            }
        }
    }

    // Each OpenMP thread drives one GPU: enqueue all chunk GEMMs and copies asynchronously
    #pragma omp parallel num_threads(numMyGpus)
    {
        GpuWork& w = work[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(w.device));
        for (const int c : w.chunks) {
            const Chunk& ch = chunks[c];
            const dim3 block(TX, TY);
            const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                            static_cast<unsigned>((ch.rows + BM - 1) / BM));
            matmulKernel<<<grid, block, 0, w.stream>>>(w.dA + ch.localOffset * N, w.dB,
                                                       w.dC + ch.localOffset * N, ch.rows, N);
            CUDA_CHECK(cudaGetLastError());
            double* dst = localC + (ch.rowBegin - (root ? 0 : myRowBegin)) * N;
            CUDA_CHECK(cudaMemcpyAsync(dst, w.dC + ch.localOffset * N, ch.rows * N * sizeof(double),
                                       cudaMemcpyDeviceToHost, w.stream));
            CUDA_CHECK(cudaEventRecord(ch.done, w.stream));
        }
    }

    // Pipelined gather: non-root ranks ship each chunk as soon as it reaches the host
    if (!root) {
        for (int c = 0; c < numChunks; ++c) {
            CUDA_CHECK(cudaEventSynchronize(chunks[c].done));
            requests.emplace_back();
            MPI_Isend(localC + (chunks[c].rowBegin - myRowBegin) * N,
                      static_cast<int>(chunks[c].rows), rowType, 0, c, MPI_COMM_WORLD,
                      &requests.back());
        }
    }
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    for (int c = 0; c < numChunks; ++c) CUDA_CHECK(cudaEventSynchronize(chunks[c].done));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Release resources
    MPI_Type_free(&rowType);
    #pragma omp parallel num_threads(numMyGpus)
    {
        GpuWork& w = work[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(w.device));
        for (const int c : w.chunks) CUDA_CHECK(cudaEventDestroy(chunks[c].done));
        CUDA_CHECK(cudaFree(w.dA));
        CUDA_CHECK(cudaFree(w.dB));
        CUDA_CHECK(cudaFree(w.dC));
        CUDA_CHECK(cudaStreamDestroy(w.stream));
    }
    if (root && N > 0) CUDA_CHECK(cudaHostUnregister(C.data()));
    if (!root && localC) CUDA_CHECK(cudaFreeHost(localC));

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
