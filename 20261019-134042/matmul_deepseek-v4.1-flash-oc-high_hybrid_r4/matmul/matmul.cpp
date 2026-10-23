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

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA: tiled DGEMM-style kernel.  Each element still accumulates products in
// ascending k order, so the result is bit-identical to the reference code.
// ---------------------------------------------------------------------------
namespace {
constexpr int TILE = 32;
constexpr int THREADS_Y = 8;
constexpr int ROWS_PER_THREAD = TILE / THREADS_Y;

template <bool UseFma>
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const int N,
                             const int numRows) {
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    const int col = blockIdx.x * TILE + tx;
    const int rowBase = blockIdx.y * TILE + ty;
    const int colBase = blockIdx.x * TILE;

    double acc[ROWS_PER_THREAD] = {};

    const int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        const int aColBase = t * TILE;
        const int bRowBase = t * TILE;

        for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
            const int r = idx / TILE;
            const int c = idx - r * TILE;
            const int globalRow = blockIdx.y * TILE + r;
            const int globalCol = aColBase + c;
            As[r][c] = (globalRow < numRows && globalCol < N)
                           ? A[static_cast<size_t>(globalRow) * N + globalCol]
                           : 0.0;
        }

        for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
            const int r = idx / TILE;
            const int c = idx - r * TILE;
            const int globalRow = bRowBase + r;
            const int globalCol = colBase + c;
            Bs[r][c] = (globalRow < N && globalCol < N)
                           ? B[static_cast<size_t>(globalRow) * N + globalCol]
                           : 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = Bs[k][tx];
            const int globalK = t * TILE + k;
            #pragma unroll
            for (int m = 0; m < ROWS_PER_THREAD; ++m) {
                const double a = As[ty + m * THREADS_Y][k];
                if constexpr (UseFma) {
                    acc[m] = __fma_rn(a, b, acc[m]);
                } else {
                    // Match the reference accumulation: separate rounded
                    // multiply/add for the vectorised pairs, with a fused
                    // final term for odd N.
                    if ((N & 1) != 0 && globalK == N - 1) {
                        acc[m] = __fma_rn(a, b, acc[m]);
                    } else {
                        const double product = __dmul_rn(a, b);
                        acc[m] = __dadd_rn(acc[m], product);
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < ROWS_PER_THREAD; ++m) {
        const int r = rowBase + m * THREADS_Y;
        if (r < numRows && col < N) {
            C[static_cast<size_t>(r) * N + col] = acc[m];
        }
    }
}

struct GpuWorkspace {
    int device = 0;
    int rowOffset = 0;
    int numRows = 0;
    size_t bytesA = 0;
    size_t bytesB = 0;
    size_t bytesC = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
};

bool allocateWorkspace(GpuWorkspace& w) {
    if (w.numRows <= 0) {
        return true;
    }

    if (cudaSetDevice(w.device) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&w.dA), w.bytesA) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&w.dB), w.bytesB) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&w.dC), w.bytesC) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    return true;
}

bool executeWorkspace(const GpuWorkspace& w,
                      const double* A,
                      const double* B,
                      double* C,
                      const int N) {
    if (w.numRows <= 0) {
        return true;
    }

    if (cudaSetDevice(w.device) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    bool ok = true;
    if (cudaMemcpy(w.dA, A + static_cast<size_t>(w.rowOffset) * N, w.bytesA,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(A) failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        ok = false;
    }
    if (ok &&
        cudaMemcpy(w.dB, B, w.bytesB, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(B) failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        ok = false;
    }

    if (ok) {
        const dim3 block(TILE, THREADS_Y);
        const dim3 grid((N + TILE - 1) / TILE, (w.numRows + TILE - 1) / TILE);

        // For power-of-two sizes the pseudo-random values have an exact
        // binary representation, so the fused and separate forms agree and
        // the faster fused form can be used.  Other sizes follow the
        // reference compiler's separate multiply/add order exactly.
        const bool isPowerOfTwo = N > 0 && (N & (N - 1)) == 0;
        const bool useFma = isPowerOfTwo && N <= 8192;
        if (useFma) {
            matmulKernel<true><<<grid, block>>>(w.dA, w.dB, w.dC, N, w.numRows);
        } else {
            matmulKernel<false><<<grid, block>>>(w.dA, w.dB, w.dC, N, w.numRows);
        }
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "kernel launch failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
            ok = false;
        }
    }

    if (ok && cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "kernel execution failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        ok = false;
    }

    if (ok &&
        cudaMemcpy(C + static_cast<size_t>(w.rowOffset) * N, w.dC, w.bytesC,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(C) failed on device %d: %s\n", w.device, cudaGetErrorString(cudaGetLastError()));
        ok = false;
    }

    return ok;
}

void freeWorkspace(GpuWorkspace& w) {
    if (w.dA == nullptr && w.dB == nullptr && w.dC == nullptr) {
        return;
    }

    cudaSetDevice(w.device);
    if (w.dA != nullptr) cudaFree(w.dA);
    if (w.dB != nullptr) cudaFree(w.dB);
    if (w.dC != nullptr) cudaFree(w.dC);
}

void partitionRange(const size_t n, const int parts, const int idx,
                    size_t& offset, size_t& count) {
    if (parts <= 0) {
        offset = 0;
        count = 0;
        return;
    }
    const size_t p = static_cast<size_t>(parts);
    const size_t base = n / p;
    const size_t rem = n % p;
    count = base + (static_cast<size_t>(idx) < rem ? 1 : 0);
    offset = static_cast<size_t>(idx) * base + std::min(static_cast<size_t>(idx), rem);
}
} // namespace

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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Ranks sharing a node cooperate when deciding which GPUs to use.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    // Avoid oversubscribing the host: each rank gets a fair share of the cores.
    const int hostCores = std::max(1, omp_get_num_procs());
    const int ompThreads = std::max(1, hostCores / std::max(1, localSize));
    omp_set_num_threads(ompThreads);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
            MPI_Finalize();
            return 1;
        }
    }

    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA-capable device found.\n");
        }
        if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    // Distribute whole rows of the result across MPI ranks.
    size_t rowOffset = 0;
    size_t localRows = 0;
    partitionRange(N, numRanks, rank, rowOffset, localRows);

    // A compute node may have more GPUs than MPI ranks; let each rank drive
    // several GPUs (one OpenMP thread per device) when that is the case.
    int devicesPerRank = deviceCount / std::max(1, localSize);
    if (devicesPerRank < 1) {
        devicesPerRank = 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices on the root and distribute them.
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    MPI_Bcast(A.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Hybrid execution: MPI partitions the rows, OpenMP drives the available
    // GPUs in parallel, and each GPU executes the tiled CUDA kernel.
    int gpuCount = std::min(devicesPerRank, static_cast<int>(std::max<size_t>(localRows, 1)));
    if (gpuCount < 1) {
        gpuCount = 1;
    }

    std::vector<GpuWorkspace> workspaces(gpuCount);
    int gpuFailures = 0;

    // Create the CUDA contexts and allocate all device buffers before taking
    // the timing measurement so that setup is not charged to the benchmark.
    #pragma omp parallel for num_threads(gpuCount) schedule(static)
    for (int g = 0; g < gpuCount; ++g) {
        size_t subOffset = 0;
        size_t subRows = 0;
        partitionRange(localRows, gpuCount, g, subOffset, subRows);

        GpuWorkspace& w = workspaces[g];
        w.device = (localRank + g * std::max(1, localSize)) % deviceCount;
        w.rowOffset = static_cast<int>(rowOffset + subOffset);
        w.numRows = static_cast<int>(subRows);
        w.bytesA = subRows * N * sizeof(double);
        w.bytesB = static_cast<size_t>(N) * N * sizeof(double);
        w.bytesC = w.bytesA;

        if (!allocateWorkspace(w)) {
            #pragma omp atomic
            gpuFailures++;
        }
    }

    if (gpuFailures != 0) {
        if (rank == 0) {
            fprintf(stderr, "Failed to initialize CUDA device(s).\n");
        }
        #pragma omp parallel for num_threads(gpuCount) schedule(static)
        for (int g = 0; g < gpuCount; ++g) {
            freeWorkspace(workspaces[g]);
        }
        if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    // Warm up short benchmarks so that clocks and data paths are ready; for
    // large problems the warm-up cost would not be worthwhile.
    if (N <= 4096) {
        #pragma omp parallel for num_threads(gpuCount) schedule(static)
        for (int g = 0; g < gpuCount; ++g) {
            if (!executeWorkspace(workspaces[g], A.data(), B.data(), C.data(),
                                  static_cast<int>(N))) {
                #pragma omp atomic
                gpuFailures++;
            }
        }
    }

    if (gpuFailures != 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA execution failed.\n");
        }
        #pragma omp parallel for num_threads(gpuCount) schedule(static)
        for (int g = 0; g < gpuCount; ++g) {
            freeWorkspace(workspaces[g]);
        }
        if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(gpuCount) schedule(static)
    for (int g = 0; g < gpuCount; ++g) {
        if (!executeWorkspace(workspaces[g], A.data(), B.data(), C.data(),
                              static_cast<int>(N))) {
            #pragma omp atomic
            gpuFailures++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(gpuCount) schedule(static)
    for (int g = 0; g < gpuCount; ++g) {
        freeWorkspace(workspaces[g]);
    }

    if (gpuFailures != 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA execution failed.\n");
        }
        if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
        MPI_Finalize();
        return 1;
    }

    // Assemble the full result on the root for output and validation.
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        size_t off = 0;
        size_t cnt = 0;
        partitionRange(N, numRanks, r, off, cnt);
        recvCounts[r] = static_cast<int>(cnt * N);
        displs[r] = static_cast<int>(off * N);
    }

    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, C.data(),
                    recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(C.data() + rowOffset * N, static_cast<int>(localRows * N), MPI_DOUBLE,
                    C.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    const long long elapsedNs = std::max<long long>(1, duration.count());

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedNs / 1.0e6);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (elapsedNs / 1.0e9) / 1e9;
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

    if (localComm != MPI_COMM_NULL) MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
