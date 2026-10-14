#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Tiling parameters for the GPU kernel
constexpr int BM = 64;          // C rows per thread block
constexpr int BN = 64;          // C columns per thread block
constexpr int BK = 16;          // k-depth staged in shared memory per step
constexpr int TM = 4;           // C rows per thread
constexpr int TN = 4;           // C columns per thread
constexpr int BLOCK_DIM = 16;   // BM / TM == BN / TN == BLOCK_DIM
constexpr int THREADS = BLOCK_DIM * BLOCK_DIM;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous partition of [0, N) into `parts` blocks: block `part` starts here
constexpr size_t partBegin(size_t part, size_t parts, size_t N) noexcept {
    return part * N / parts;
}

// Initialize rows [rowBegin, rowEnd) of an N-column matrix; mat holds only those rows
void initRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - rowBegin) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled, register-blocked matrix multiply: C[rows x N] = A[rows x N] * B[N x N]
// Each C element accumulates over k in increasing order (one fused multiply-add per step)
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t rows, const size_t N) {
    __shared__ double As[BK][BM + 1];  // transposed A tile, padded against bank conflicts
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.y * BLOCK_DIM + threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        const int kLen = (N - k0 < BK) ? static_cast<int>(N - k0) : BK;

        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int m = idx / BK;
            const int k = idx % BK;
            const size_t row = rowBase + m;
            As[k][m] = (row < rows && k < kLen) ? A[row * N + k0 + k] : 0.0;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int k = idx / BN;
            const int n = idx % BN;
            const size_t col = colBase + n;
            Bs[k][n] = (k < kLen && col < N) ? B[(k0 + k) * N + col] : 0.0;
        }
        __syncthreads();

        for (int k = 0; k < kLen; ++k) {
            double a[TM];
            double b[TN];
            for (int i = 0; i < TM; ++i) a[i] = As[k][threadIdx.y * TM + i];
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][threadIdx.x * TN + j];
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __fma_rn(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const size_t row = rowBase + threadIdx.y * TM + i;
        if (row >= rows) continue;
        for (int j = 0; j < TN; ++j) {
            const size_t col = colBase + threadIdx.x * TN + j;
            if (col < N) C[row * N + col] = acc[i][j];
        }
    }
}

// Compute rows [0, rows) of C on one GPU
void computeOnDevice(const int device, const double* A, const double* B, double* C,
                     const size_t N, const size_t rows) {
    CUDA_CHECK(cudaSetDevice(device));

    const size_t aBytes = rows * N * sizeof(double);
    const size_t bBytes = N * N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, aBytes));
    CUDA_CHECK(cudaMalloc(&dB, bBytes));
    CUDA_CHECK(cudaMalloc(&dC, aBytes));

    CUDA_CHECK(cudaMemcpy(dA, A, aBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bBytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((rows + BM - 1) / BM));
    matmulKernel<<<grid, block>>>(dA, dB, dC, rows, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, dC, aBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// Split this rank's rows across its GPUs, one OpenMP thread driving each GPU
void computeLocalRows(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N, const size_t rows,
                      const std::vector<int>& devices) {
    if (rows == 0 || N == 0) return;

    const int numDevices = static_cast<int>(std::min(devices.size(), rows));

    #pragma omp parallel for schedule(static, 1) num_threads(numDevices)
    for (int t = 0; t < numDevices; ++t) {
        const size_t begin = partBegin(t, numDevices, rows);
        const size_t end = partBegin(t + 1, numDevices, rows);
        computeOnDevice(devices[t], A.data() + begin * N, B.data(), C.data() + begin * N, N,
                        end - begin);
    }
}

// Pick the GPUs this rank drives. Ranks on a node share its GPUs round-robin
std::vector<int> selectDevices(const int localRank, const int localSize) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count <= 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", localRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> devices;
    if (localSize <= count) {
        for (int d = localRank; d < count; d += localSize) {
            devices.push_back(d);
        }
    } else {
        devices.push_back(localRank % count);
    }
    return devices;
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

int runBenchmark(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, omp_get_max_threads());
    }

    // Bind each rank to GPUs on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    const std::vector<int> devices = selectDevices(localRank, localSize);
    for (int d : devices) {
        // Create the CUDA context up front so it stays out of the timed region
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(0));
    }

    // Share GPU counts so each rank's block of rows is proportional to its GPUs
    std::vector<int> gpuCounts(size);
    const int myGpus = static_cast<int>(devices.size());
    MPI_Allgather(&myGpus, 1, MPI_INT, gpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<size_t> rowStart(size + 1, 0);
    size_t totalGpus = 0;
    for (int r = 0; r < size; ++r) totalGpus += gpuCounts[r];
    size_t gpusBefore = 0;
    for (int r = 0; r < size; ++r) {
        rowStart[r] = gpusBefore * N / totalGpus;
        gpusBefore += gpuCounts[r];
    }
    rowStart[size] = N;

    // This rank's block of rows of A and C
    const size_t rowBegin = rowStart[rank];
    const size_t rowEnd = rowStart[rank + 1];
    const size_t rows = rowEnd - rowBegin;

    // Allocate matrices
    const bool needFullC = printResults || validate;
    std::vector<double> A(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C((root && needFullC) ? N * N : rows * N);

    // Initialize matrices
    if (root) printf("Initializing matrices...\n");
    initRows(A, N, rowBegin, rowEnd);
    initRows(B, N, 0, N);

    // Perform matrix multiplication
    if (root) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    computeLocalRows(A, B, C, N, rows, devices);

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxMs));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C on root for output and validation
    if (needFullC) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            displs[r] = static_cast<int>(rowStart[r]);
            counts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        }

        if (root) {
            MPI_Gatherv(MPI_IN_PLACE, 0, rowType, C.data(), counts.data(), displs.data(),
                        rowType, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(C.data(), static_cast<int>(rows), rowType, nullptr, counts.data(),
                        displs.data(), rowType, 0, MPI_COMM_WORLD);
        }
        MPI_Type_free(&rowType);
    }

    // Print results for external validation
    if (root && printResults) {
        print_results(C, "MatrixC");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            std::vector<double> fullA(N * N);
            initRows(fullA, N, 0, N);
            valid = validateResult(fullA, B, C, N) ? 1 : 0;

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
