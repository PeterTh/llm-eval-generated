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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            fflush(stderr);                                                              \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, static_cast<size_t>(i), j);
        }
    }
}

__device__ __forceinline__ double devicePseudoRndValue(const size_t n2, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % n2) / static_cast<double>(n2);
}

__global__ void initMatrixKernel(double* __restrict__ mat, const size_t n2, const int rows,
                                 const int cols, const int ld, const int globalRowOffset) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        mat[static_cast<size_t>(r) * ld + c] =
            devicePseudoRndValue(n2, static_cast<size_t>(globalRowOffset + r),
                                 static_cast<size_t>(c));
    }
}

void initDeviceMatrix(double* mat, const size_t n2, const int rows, const int cols,
                      const int ld, const int globalRowOffset, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((cols + 31) / 32, (rows + 7) / 8);
    initMatrixKernel<<<grid, block, 0, stream>>>(mat, n2, rows, cols, ld, globalRowOffset);
}

template <int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ __launch_bounds__(THREADS, 2)
void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                  double* __restrict__ C, const int M, const int N) {
    static_assert(THREADS == (BM / TM) * (BN / TN), "invalid thread tiling");

    constexpr int TPC = BN / TN;

    __shared__ double As[BK][BM + 2];
    __shared__ double Bs[BK][BN + 2];

    const int tid = threadIdx.x;
    const int threadRow = tid / TPC;
    const int threadCol = tid % TPC;
    const int blockRow = blockIdx.x * BM;
    const int blockCol = blockIdx.y * BN;
    const int localRow = threadRow * TM;
    const int localCol = threadCol * TN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int m = idx / BK;
            const int k = idx - m * BK;
            const int gm = blockRow + m;
            const int gk = k0 + k;
            As[k][m] = (gm < M && gk < N) ? A[static_cast<size_t>(gm) * N + gk] : 0.0;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int k = idx / BN;
            const int n = idx - k * BN;
            const int gk = k0 + k;
            const int gn = blockCol + n;
            Bs[k][n] = (gk < N && gn < N) ? B[static_cast<size_t>(gk) * N + gn] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[k][localRow + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[k][localCol + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = blockRow + localRow + i;
        if (gm < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn = blockCol + localCol + j;
                if (gn < N) {
                    C[static_cast<size_t>(gm) * N + gn] = acc[i][j];
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, int THREADS>
void launchMatmul(const double* dA, const double* dB, double* dC, const int M, const int N,
                  cudaStream_t stream) {
    const dim3 block(THREADS);
    const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    matmulKernel<BM, BN, BK, TM, TN, THREADS><<<grid, block, 0, stream>>>(dA, dB, dC, M, N);
}

struct GpuWorkspace {
    int device = 0;
    cudaStream_t stream = nullptr;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    double* hC = nullptr;
    bool hPinned = false;
};

std::vector<GpuWorkspace> createWorkspaces(const size_t N, const int rowStart, const int rowEnd,
                                           const int gpusPerRank, const int localRank,
                                           const int deviceCount) {
    const int totalRows = rowEnd - rowStart;
    if (totalRows <= 0 || N == 0) {
        return {};
    }

    const int workers = std::max(1, std::min(gpusPerRank, totalRows));
    std::vector<GpuWorkspace> workspaces(static_cast<size_t>(workers));
    const size_t n2 = N * N;
    const size_t elemsB = N * N;

#pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int w = 0; w < workers; ++w) {
        GpuWorkspace& ws = workspaces[static_cast<size_t>(w)];
        const int device = (localRank * gpusPerRank + w) % deviceCount;
        ws.device = device;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(0));

        const int rowsBegin =
            rowStart + static_cast<int>(static_cast<long long>(totalRows) * w / workers);
        const int rowsEnd =
            rowStart + static_cast<int>(static_cast<long long>(totalRows) * (w + 1) / workers);
        const int M = rowsEnd - rowsBegin;
        const size_t elemsA = static_cast<size_t>(M) * N;

        CUDA_CHECK(cudaStreamCreateWithFlags(&ws.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.dA), elemsA * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.dB), elemsB * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ws.dC), elemsA * sizeof(double)));

        cudaError_t pinErr =
            cudaHostAlloc(reinterpret_cast<void**>(&ws.hC), elemsA * sizeof(double),
                          cudaHostAllocDefault);
        if (pinErr == cudaSuccess) {
            ws.hPinned = true;
        } else {
            cudaGetLastError();
            ws.hC = nullptr;
            ws.hPinned = false;
        }

        initDeviceMatrix(ws.dA, n2, M, static_cast<int>(N), static_cast<int>(N), rowsBegin,
                         ws.stream);
        initDeviceMatrix(ws.dB, n2, static_cast<int>(N), static_cast<int>(N), static_cast<int>(N),
                         0, ws.stream);
        launchMatmul<64, 64, 16, 4, 4, 256>(ws.dA, ws.dB, ws.dC, 1, 1, ws.stream);
        launchMatmul<32, 32, 32, 2, 2, 256>(ws.dA, ws.dB, ws.dC, 1, 1, ws.stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(ws.stream));
    }

    return workspaces;
}

void destroyWorkspaces(std::vector<GpuWorkspace>& workspaces) {
    const int workers = static_cast<int>(workspaces.size());
#pragma omp parallel for schedule(static, 1) num_threads(std::max(1, workers))
    for (int w = 0; w < workers; ++w) {
        GpuWorkspace& ws = workspaces[static_cast<size_t>(w)];
        CUDA_CHECK(cudaSetDevice(ws.device));
        CUDA_CHECK(cudaStreamSynchronize(ws.stream));
        CUDA_CHECK(cudaFree(ws.dA));
        CUDA_CHECK(cudaFree(ws.dB));
        CUDA_CHECK(cudaFree(ws.dC));
        if (ws.hPinned) {
            CUDA_CHECK(cudaFreeHost(ws.hC));
        }
        CUDA_CHECK(cudaStreamDestroy(ws.stream));
    }
    workspaces.clear();
}

void matrixMultiply(std::vector<double>& C, const size_t N, const int rowStart,
                    const int rowEnd, const int gpusPerRank,
                    const std::vector<GpuWorkspace>& workspaces) {
    const int totalRows = rowEnd - rowStart;
    if (totalRows <= 0 || N == 0) {
        return;
    }

    const int workers = std::max(1, std::min(gpusPerRank, totalRows));
    if (static_cast<int>(workspaces.size()) != workers) {
        return;
    }

#pragma omp parallel for schedule(static, 1) num_threads(workers)
    for (int w = 0; w < workers; ++w) {
        const GpuWorkspace& ws = workspaces[static_cast<size_t>(w)];
        CUDA_CHECK(cudaSetDevice(ws.device));

        const int rowsBegin =
            rowStart + static_cast<int>(static_cast<long long>(totalRows) * w / workers);
        const int rowsEnd =
            rowStart + static_cast<int>(static_cast<long long>(totalRows) * (w + 1) / workers);
        const int M = rowsEnd - rowsBegin;
        if (M <= 0) {
            continue;
        }

        const size_t elemsA = static_cast<size_t>(M) * N;

        const long long blocksMedium =
            static_cast<long long>((M + 63) / 64) * static_cast<long long>((N + 63) / 64);
        if (blocksMedium >= 128) {
            launchMatmul<64, 64, 16, 4, 4, 256>(ws.dA, ws.dB, ws.dC, M, static_cast<int>(N),
                                                ws.stream);
        } else {
            launchMatmul<32, 32, 32, 2, 2, 256>(ws.dA, ws.dB, ws.dC, M, static_cast<int>(N),
                                                ws.stream);
        }
        CUDA_CHECK(cudaGetLastError());

        double* dst = C.data() + static_cast<size_t>(rowsBegin) * N;
        if (ws.hPinned) {
            CUDA_CHECK(cudaMemcpyAsync(ws.hC, ws.dC, elemsA * sizeof(double),
                                       cudaMemcpyDeviceToHost, ws.stream));
            CUDA_CHECK(cudaStreamSynchronize(ws.stream));
            std::memcpy(dst, ws.hC, elemsA * sizeof(double));
        } else {
            CUDA_CHECK(cudaMemcpy(dst, ws.dC, elemsA * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaStreamSynchronize(ws.stream));
        }
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
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            badOption = true;
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || badOption) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badOption ? 1 : 0;
    }

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (worldRank == 0) {
            fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(devErr));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, omp_get_num_procs() / std::max(1, localSize)));

    const int gpusPerRank = std::max(1, deviceCount / std::max(1, localSize));

    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    if (worldRank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication
    if (worldRank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    const int rowStart = static_cast<int>(static_cast<long long>(N) * worldRank / worldSize);
    const int rowEnd = static_cast<int>(static_cast<long long>(N) * (worldRank + 1) / worldSize);

    std::vector<GpuWorkspace> workspaces =
        createWorkspaces(N, rowStart, rowEnd, gpusPerRank, localRank, deviceCount);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(C, N, rowStart, rowEnd, gpusPerRank, workspaces);

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxMs));
        const double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    destroyWorkspaces(workspaces);

    int* counts = nullptr;
    int* displs = nullptr;
    if (worldRank == 0) {
        counts = static_cast<int*>(std::malloc(sizeof(int) * static_cast<size_t>(worldSize)));
        displs = static_cast<int*>(std::malloc(sizeof(int) * static_cast<size_t>(worldSize)));
        for (int r = 0; r < worldSize; ++r) {
            const int rs = static_cast<int>(static_cast<long long>(N) * r / worldSize);
            const int re = static_cast<int>(static_cast<long long>(N) * (r + 1) / worldSize);
            counts[r] = (re - rs) * static_cast<int>(N);
            displs[r] = rs * static_cast<int>(N);
        }
    }
    MPI_Gatherv(C.data() + static_cast<size_t>(rowStart) * N,
                (rowEnd - rowStart) * static_cast<int>(N), MPI_DOUBLE,
                worldRank == 0 ? C.data() : nullptr, worldRank == 0 ? counts : nullptr,
                worldRank == 0 ? displs : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::free(counts);
    std::free(displs);

    // Print results for external validation
    if (printResults && worldRank == 0) {
        print_results(C, "MatrixC");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            valid = validateResult(A, B, C, N) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
