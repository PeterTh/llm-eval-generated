#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
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

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// GPU tiling parameters
constexpr size_t TILE = 64;                 // C tile edge computed by one thread block
constexpr size_t TK = 16;                   // k-depth staged in shared memory per step
constexpr size_t REG = 4;                   // C elements per thread along each dimension
constexpr size_t THREADS_DIM = TILE / REG;  // 16 -> 256 threads per block

// Row chunk streamed through the GPU per pipeline stage (rows of the rank-local block)
constexpr size_t GPU_CHUNK_ROWS = 128;

// Tiled FP64 GEMM. Every C element is accumulated in a single register over
// k = 0..N-1 in increasing order using FMA.
__global__ void __launch_bounds__(THREADS_DIM * THREADS_DIM)
matmulTiledKernel(const double* __restrict__ A, const double* __restrict__ B,
                  double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[TK][TILE];  // As[k][m]: transposed A tile
    __shared__ double Bs[TK][TILE];  // Bs[k][n]

    const size_t tid = threadIdx.x;
    const size_t tx = tid % THREADS_DIM;
    const size_t ty = tid / THREADS_DIM;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE;

    double acc[REG][REG];
#pragma unroll
    for (size_t i = 0; i < REG; ++i) {
#pragma unroll
        for (size_t j = 0; j < REG; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (size_t k0 = 0; k0 < N; k0 += TK) {
        for (size_t l = tid; l < TILE * TK; l += THREADS_DIM * THREADS_DIM) {
            const size_t m = l / TK;
            const size_t kk = l % TK;
            const size_t gr = rowBase + m;
            const size_t gk = k0 + kk;
            As[kk][m] = (gr < rows && gk < N) ? A[gr * N + gk] : 0.0;
        }
        for (size_t l = tid; l < TK * TILE; l += THREADS_DIM * THREADS_DIM) {
            const size_t kk = l / TILE;
            const size_t n = l % TILE;
            const size_t gk = k0 + kk;
            const size_t gc = colBase + n;
            Bs[kk][n] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (size_t kk = 0; kk < TK; ++kk) {
            double a[REG];
            double b[REG];
#pragma unroll
            for (size_t i = 0; i < REG; ++i) a[i] = As[kk][ty * REG + i];
#pragma unroll
            for (size_t j = 0; j < REG; ++j) b[j] = Bs[kk][tx * REG + j];
#pragma unroll
            for (size_t i = 0; i < REG; ++i) {
#pragma unroll
                for (size_t j = 0; j < REG; ++j) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < REG; ++i) {
        const size_t gr = rowBase + ty * REG + i;
#pragma unroll
        for (size_t j = 0; j < REG; ++j) {
            const size_t gc = colBase + tx * REG + j;
            if (gr < rows && gc < N) {
                C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

// Computes C_h = A_h * B_d for the `rows` local rows. Row chunks are pipelined over two
// CUDA streams with double-buffered device memory, so the upload of one chunk, its kernel,
// and the download of an earlier chunk overlap. Each stream keeps its own chunks in order.
void gpuMultiply(const double* A_h, double* C_h, const double* B_d, const size_t rows,
                 const size_t N) {
    constexpr int NSTREAMS = 2;
    const size_t chunkRows = std::min(GPU_CHUNK_ROWS, std::max<size_t>(rows, 1));

    cudaStream_t streams[NSTREAMS];
    double* dA[NSTREAMS];
    double* dC[NSTREAMS];
    for (int s = 0; s < NSTREAMS; ++s) {
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
        CHECK_CUDA(cudaMalloc(&dA[s], chunkRows * N * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dC[s], chunkRows * N * sizeof(double)));
    }

    size_t chunk = 0;
    for (size_t begin = 0; begin < rows; begin += GPU_CHUNK_ROWS, ++chunk) {
        const size_t m = std::min(GPU_CHUNK_ROWS, rows - begin);
        const int s = static_cast<int>(chunk % NSTREAMS);

        CHECK_CUDA(cudaMemcpyAsync(dA[s], A_h + begin * N, m * N * sizeof(double),
                                   cudaMemcpyHostToDevice, streams[s]));

        const dim3 block(static_cast<unsigned int>(THREADS_DIM * THREADS_DIM));
        const dim3 grid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                        static_cast<unsigned int>((m + TILE - 1) / TILE));
        matmulTiledKernel<<<grid, block, 0, streams[s]>>>(dA[s], B_d, dC[s], m, N);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpyAsync(C_h + begin * N, dC[s], m * N * sizeof(double),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }

    for (int s = 0; s < NSTREAMS; ++s) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
        CHECK_CUDA(cudaFree(dA[s]));
        CHECK_CUDA(cudaFree(dC[s]));
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
    }
}

// Fills `rowCount` rows starting at global row `rowBegin` of an N-column matrix (OpenMP)
void initRows(double* mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (size_t li = 0; li < rowCount; ++li) {
        const size_t i = rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Block distribution of N rows over `size` ranks
void rowRange(const size_t N, const int size, const int rank, size_t& begin, size_t& count) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    count = base + (r < rem ? 1 : 0);
    begin = r * base + std::min(r, rem);
}

// Simple validation: compute a single element and compare
bool validateResult(const double* B, const std::vector<double>& C, const size_t N) {
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
    printf("Launch with mpirun/srun, e.g.: mpirun -np <ranks> %s -n <num>\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    int status = -1;  // -1: continue running, otherwise exit code

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const int n = atoi(argv[++i]);
            if (n <= 0 || n > INT_MAX) {
                if (root) printf("Invalid matrix size: %s\n", argv[i]);
                status = 1;
                break;
            }
            N = static_cast<size_t>(n);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            status = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }
    if (status >= 0) {
        MPI_Finalize();
        return status;
    }

    // Bind each rank to a GPU on its node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no usable CUDA device (%s)\n", rank,
                devErr == cudaSuccess ? "none found" : cudaGetErrorString(devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_CUDA(cudaSetDevice(localRank % deviceCount));

    // Divide host cores among the ranks sharing this node (used for OpenMP host work)
    const int hostThreads = std::max(1, omp_get_max_threads() / localSize);
    omp_set_num_threads(hostThreads);

    size_t rowBegin = 0;
    size_t rowCount = 0;
    rowRange(N, size, rank, rowBegin, rowCount);

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI (%d ranks) + OpenMP (%d threads/rank) + CUDA (1 GPU/rank)\n",
               size, hostThreads);
    }

    // Allocate matrices (pinned host memory for asynchronous transfers)
    const size_t localElems = std::max<size_t>(rowCount, 1) * N;
    double* A_h = nullptr;
    double* B_h = nullptr;
    double* C_h = nullptr;
    CHECK_CUDA(cudaMallocHost(&A_h, localElems * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&B_h, N * N * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&C_h, localElems * sizeof(double)));

    double* dB = nullptr;
    CHECK_CUDA(cudaMalloc(&dB, N * N * sizeof(double)));

    // Initialize matrices (each rank generates its rows of A and all of B)
    if (root) printf("Initializing matrices...\n");
    initRows(A_h, N, rowBegin, rowCount);
    initRows(B_h, N, 0, N);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(dB, B_h, N * N * sizeof(double), cudaMemcpyHostToDevice));
    gpuMultiply(A_h, C_h, dB, rowCount, N);

    auto end = std::chrono::high_resolution_clock::now();
    const double localSec = std::chrono::duration<double>(end - start).count();
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxSec));
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full result on rank 0 only when it is needed for output or validation
    const bool needFull = printResults || validate;
    std::vector<double> C;
    if (needFull) {
        if (root) C.resize(N * N);

        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            size_t b = 0;
            size_t c = 0;
            rowRange(N, size, r, b, c);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(b);
        }
        MPI_Gatherv(C_h, static_cast<int>(rowCount), rowType,
                    root ? C.data() : nullptr, counts.data(), displs.data(), rowType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(C, "MatrixC");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(B_h, C, N) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFreeHost(A_h));
    CHECK_CUDA(cudaFreeHost(B_h));
    CHECK_CUDA(cudaFreeHost(C_h));

    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
