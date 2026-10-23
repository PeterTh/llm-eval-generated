#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;
constexpr int ITEMS_A = (TILE_M * TILE_K) / THREADS_PER_BLOCK;
constexpr int ITEMS_B = (TILE_K * TILE_N) / THREADS_PER_BLOCK;

__global__ void initBKernel(double* __restrict__ B,
                            const unsigned long long nn,
                            const int N) {
    const unsigned long long idx =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nn) {
        return;
    }
    const unsigned long long i = idx / static_cast<unsigned long long>(N);
    const unsigned long long j = idx % static_cast<unsigned long long>(N);
    const unsigned long long value = ((i + 1ULL) * (i + j + 1ULL) * 1299709ULL) % nn;
    B[idx] = static_cast<double>(value) / static_cast<double>(nn);
}

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const int M, const int N, const int K) {
    __shared__ double As[TILE_M][TILE_K];
    __shared__ double Bs[TILE_K][TILE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const int row0 = blockIdx.y * TILE_M + ty * 4;
    const int col0 = blockIdx.x * TILE_N + tx * 4;

    double acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        #pragma unroll
        for (int e = 0; e < ITEMS_A; ++e) {
            const int idx = tid + e * THREADS_PER_BLOCK;
            const int ar = idx / TILE_K;
            const int ak = idx % TILE_K;
            const int gr = blockIdx.y * TILE_M + ar;
            const int gk = k0 + ak;
            As[ar][ak] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0;
        }

        #pragma unroll
        for (int e = 0; e < ITEMS_B; ++e) {
            const int idx = tid + e * THREADS_PER_BLOCK;
            const int bk = idx / TILE_N;
            const int bn = idx % TILE_N;
            const int gk = k0 + bk;
            const int gc = blockIdx.x * TILE_N + bn;
            Bs[bk][bn] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            const double a0 = As[ty * 4 + 0][kk];
            const double a1 = As[ty * 4 + 1][kk];
            const double a2 = As[ty * 4 + 2][kk];
            const double a3 = As[ty * 4 + 3][kk];

            const double b0 = Bs[kk][tx * 4 + 0];
            const double b1 = Bs[kk][tx * 4 + 1];
            const double b2 = Bs[kk][tx * 4 + 2];
            const double b3 = Bs[kk][tx * 4 + 3];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;
            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;
            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = col0 + j;
                if (c < N) {
                    C[static_cast<size_t>(r) * N + c] = acc[i][j];
                }
            }
        }
    }
}

bool gpuMultiply(const int device,
                 const double* hA, double* hC,
                 const size_t rows, const size_t N,
                 bool& outOfMemory) {
    outOfMemory = false;
    if (rows == 0 || N == 0) {
        return true;
    }
    if (N > static_cast<size_t>(INT_MAX) || rows > static_cast<size_t>(INT_MAX)) {
        return false;
    }

    if (cudaSetDevice(device) != cudaSuccess) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }

    const size_t aBytes = rows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    bool ok = true;

    if (cudaMalloc(reinterpret_cast<void**>(&dA), aBytes) != cudaSuccess) {
        outOfMemory = true;
        ok = false;
    }
    if (ok && cudaMalloc(reinterpret_cast<void**>(&dB), bBytes) != cudaSuccess) {
        outOfMemory = true;
        ok = false;
    }
    if (ok && cudaMalloc(reinterpret_cast<void**>(&dC), aBytes) != cudaSuccess) {
        outOfMemory = true;
        ok = false;
    }

    if (ok) {
        ok = cudaMemcpyAsync(dA, hA, aBytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }

    if (ok) {
        const unsigned long long nn = static_cast<unsigned long long>(N) * N;
        const unsigned int initBlocks =
            static_cast<unsigned int>((nn + 255ULL) / 256ULL);
        initBKernel<<<initBlocks, 256, 0, stream>>>(dB, nn, static_cast<int>(N));
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (ok) {
        const unsigned int gridX = static_cast<unsigned int>((N + TILE_N - 1) / TILE_N);
        const unsigned int gridY = static_cast<unsigned int>((rows + TILE_M - 1) / TILE_M);
        matmulKernel<<<dim3(gridX, gridY), dim3(THREADS_X, THREADS_Y), 0, stream>>>(
            dA, dB, dC, static_cast<int>(rows), static_cast<int>(N), static_cast<int>(N));
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (ok) {
        ok = cudaMemcpyAsync(hC, dC, aBytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    }
    if (ok) {
        ok = cudaStreamSynchronize(stream) == cudaSuccess;
    }

    if (dA != nullptr) {
        cudaFree(dA);
    }
    if (dB != nullptr) {
        cudaFree(dB);
    }
    if (dC != nullptr) {
        cudaFree(dC);
    }
    cudaStreamDestroy(stream);

    return ok;
}

void initMatrixRegion(std::vector<double>& mat, const size_t N,
                      const size_t rowStart, const size_t rows,
                      const int nthreads) {
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (size_t i = 0; i < rows; ++i) {
        const size_t gi = rowStart + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, gi, j);
        }
    }
}

void initMatrixFull(std::vector<double>& mat, const size_t N, const int nthreads) {
    #pragma omp parallel for schedule(static) collapse(2) num_threads(nthreads)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

bool validateResultBlock(const std::vector<double>& A, const std::vector<double>& B,
                         const std::vector<double>& C, const size_t N,
                         const size_t rowStart, const size_t rows) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool valid = true;

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            if (i < rowStart || i >= rowStart + rows) {
                continue;
            }

            const size_t li = i - rowStart;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[li * N + k] * B[k * N + j];
            }

            const double actual = C[li * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = false;
            }
        }
    }

    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

struct HostPin {
    void* pointer = nullptr;
    size_t bytes = 0;

    void pin(void* ptr, const size_t size) {
        if (ptr != nullptr && size > 0 &&
            cudaHostRegister(ptr, size, cudaHostRegisterDefault) == cudaSuccess) {
            pointer = ptr;
            bytes = size;
        }
    }

    ~HostPin() {
        if (pointer != nullptr) {
            cudaHostUnregister(pointer);
        }
    }
};

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    if (localSize < 1) {
        localSize = 1;
    }

    int deviceCount = 0;
    const cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            printf("Error: CUDA device is required but none is available (%s)\n",
                   cudaGetErrorString(cudaStatus));
        }
        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return 1;
    }

    std::vector<int> devices;
    for (int d = localRank; d < deviceCount; d += localSize) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        devices.push_back(localRank % deviceCount);
    }

    const int maxThreads = omp_get_max_threads();
    const int procsPerRank = std::max(1, omp_get_num_procs() / localSize);
    int nthreads = std::max(1, std::min(maxThreads, procsPerRank));
    if (nthreads < static_cast<int>(devices.size())) {
        nthreads = static_cast<int>(devices.size());
    }

    const int localDeviceCount = static_cast<int>(devices.size());
    std::vector<int> deviceCounts(static_cast<size_t>(size), 0);
    MPI_Allgather(&localDeviceCount, 1, MPI_INT, deviceCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<size_t> capacities(static_cast<size_t>(size), 1);
    size_t totalCapacity = 0;
    for (int r = 0; r < size; ++r) {
        capacities[static_cast<size_t>(r)] =
            static_cast<size_t>(std::max(1, deviceCounts[static_cast<size_t>(r)]));
        totalCapacity += capacities[static_cast<size_t>(r)];
    }

    std::vector<size_t> rowStarts(static_cast<size_t>(size), 0);
    std::vector<size_t> rowCounts(static_cast<size_t>(size), 0);
    size_t cumulative = 0;
    for (int r = 0; r < size; ++r) {
        const size_t previous = cumulative;
        cumulative += capacities[static_cast<size_t>(r)];
        rowStarts[static_cast<size_t>(r)] = N * previous / totalCapacity;
        rowCounts[static_cast<size_t>(r)] =
            N * cumulative / totalCapacity - rowStarts[static_cast<size_t>(r)];
    }

    const size_t rowStart = rowStarts[static_cast<size_t>(rank)];
    const size_t rows = rowCounts[static_cast<size_t>(rank)];

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t nn = N * N;
    std::vector<double> A(rows * N);
    std::vector<double> B(nn);
    std::vector<double> C(rows * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrixRegion(A, N, rowStart, rows, nthreads);
    initMatrixFull(B, N, nthreads);

    HostPin pinnedA;
    HostPin pinnedC;
    pinnedA.pin(A.data(), A.size() * sizeof(double));
    pinnedC.pin(C.data(), C.size() * sizeof(double));

    for (int device : devices) {
        double warmA = 1.0;
        double warmC = 0.0;
        bool warmOom = false;
        gpuMultiply(device, &warmA, &warmC, 1, 1, warmOom);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    std::atomic<int> gpuError{0};
    std::atomic<int> gpuOutOfMemory{0};

    const int gpuThreads = static_cast<int>(devices.size());

    #pragma omp parallel num_threads(gpuThreads)
    {
        const int tid = omp_get_thread_num();
        if (rows > 0) {
            const size_t begin = rows * static_cast<size_t>(tid) /
                                 static_cast<size_t>(gpuThreads);
            const size_t end = rows * static_cast<size_t>(tid + 1) /
                               static_cast<size_t>(gpuThreads);
            if (end > begin) {
                bool oom = false;
                if (!gpuMultiply(devices[static_cast<size_t>(tid)],
                                 A.data() + begin * N, C.data() + begin * N,
                                 end - begin, N, oom)) {
                    if (oom) {
                        gpuOutOfMemory.store(1, std::memory_order_relaxed);
                    }
                    gpuError.store(1, std::memory_order_relaxed);
                }
            }
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    long long localMs =
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int localOk = gpuError.load(std::memory_order_relaxed) ? 0 : 1;
    int globalOk = 1;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!globalOk) {
        const int localOom = gpuOutOfMemory.load(std::memory_order_relaxed);
        int globalOom = 0;
        MPI_Allreduce(&localOom, &globalOom, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            if (globalOom) {
                printf("CUDA computation failed: out of device memory\n");
            } else {
                printf("CUDA computation failed\n");
            }
        }
        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);
        const double seconds = static_cast<double>(maxMs) / 1000.0;
        const double gflops = (seconds > 0.0)
                                  ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                                     static_cast<double>(N)) / seconds / 1e9
                                  : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        if (N == 0) {
            if (rank == 0) {
                std::vector<double> empty;
                print_results(empty, "MatrixC");
            }
        } else {
            std::vector<int> counts(static_cast<size_t>(size), 0);
            std::vector<int> displacements(static_cast<size_t>(size), 0);
            bool overflow = false;
            long long total = 0;
            for (int r = 0; r < size; ++r) {
                const size_t chunk = rowCounts[static_cast<size_t>(r)] * N;
                if (chunk > static_cast<size_t>(INT_MAX)) {
                    overflow = true;
                    break;
                }
                counts[static_cast<size_t>(r)] = static_cast<int>(chunk);
                if (total + static_cast<long long>(chunk) > static_cast<long long>(INT_MAX)) {
                    overflow = true;
                    break;
                }
                displacements[static_cast<size_t>(r)] = static_cast<int>(total);
                total += static_cast<long long>(chunk);
            }

            if (overflow) {
                if (rank == 0) {
                    printf("Error: matrix too large for MPI_Gatherv\n");
                }
                MPI_Comm_free(&nodeComm);
                MPI_Finalize();
                return 1;
            }

            std::vector<double> Cfull;
            if (rank == 0) {
                Cfull.resize(static_cast<size_t>(total));
            }

            MPI_Gatherv(counts[static_cast<size_t>(rank)] > 0 ? C.data() : nullptr,
                        counts[static_cast<size_t>(rank)], MPI_DOUBLE,
                        rank == 0 ? (Cfull.empty() ? nullptr : Cfull.data()) : nullptr,
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                print_results(Cfull, "MatrixC");
            }
        }
    }

    // Validation
    if (validate) {
        int localValid = 1;
        if (N > 0) {
            localValid = validateResultBlock(A, B, C, N, rowStart, rows) ? 1 : 0;
        }

        int globalValid = 1;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return globalValid ? 0 : 1;
    }

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return 0;
}
