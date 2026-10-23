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

namespace {

constexpr int kTile = 32;
constexpr int kThreads = 16;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Distribute 'total' items over 'parts' contiguous blocks.
void splitRows(const size_t total, const int parts,
               std::vector<size_t>& starts, std::vector<size_t>& counts) {
    starts.resize(static_cast<size_t>(parts));
    counts.resize(static_cast<size_t>(parts));
    const size_t base = parts > 0 ? total / static_cast<size_t>(parts) : 0;
    const size_t rem = parts > 0 ? total % static_cast<size_t>(parts) : 0;
    for (int p = 0; p < parts; ++p) {
        const size_t sp = static_cast<size_t>(p);
        counts[sp] = base + (sp < rem ? 1 : 0);
        starts[sp] = sp * base + std::min(sp, rem);
    }
}

// Distribute 'total' rows over processes proportionally to the number of
// GPUs owned by each process, keeping every GPU's share balanced.
void splitRowsWeighted(const size_t total, const std::vector<int>& weights,
                       std::vector<size_t>& starts, std::vector<size_t>& counts) {
    const size_t parts = weights.size();
    starts.resize(parts);
    counts.resize(parts);

    size_t totalUnits = 0;
    for (const int w : weights) totalUnits += static_cast<size_t>(std::max(w, 0));
    if (totalUnits == 0) {
        splitRows(total, static_cast<int>(parts), starts, counts);
        return;
    }

    const size_t base = total / totalUnits;
    const size_t rem = total % totalUnits;
    size_t unit = 0;
    size_t offset = 0;
    for (size_t p = 0; p < parts; ++p) {
        size_t count = 0;
        for (int u = 0; u < weights[p]; ++u) {
            count += base + (unit < rem ? 1 : 0);
            ++unit;
        }
        starts[p] = offset;
        counts[p] = count;
        offset += count;
    }
}

void initMatrix(std::vector<double>& mat, const size_t N,
                const size_t rowStart, const size_t rowCount) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rowCount; ++r) {
        const size_t i = rowStart + r;
        double* row = mat.data() + r * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Shared-memory tiled double-precision GEMM.  The k-loop is traversed in
// exactly increasing order so the result is bit-for-bit identical to the
// reference implementation.  When products are exactly representable
// (power-of-two N up to 8192) the faster fused multiply-add path is used;
// otherwise rounded multiply and add must be kept separate.
template <bool UseFma>
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const int N, const int rows) {
    __shared__ double As[kTile][kTile];
    __shared__ double Bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreads + tx;

    const int blockRow = blockIdx.y * kTile;
    const int blockCol = blockIdx.x * kTile;

    const int row0 = blockRow + ty;
    const int row1 = row0 + kThreads;
    const int col0 = blockCol + tx;
    const int col1 = col0 + kThreads;

    double sum00 = 0.0;
    double sum01 = 0.0;
    double sum10 = 0.0;
    double sum11 = 0.0;

    for (int t = 0; t < N; t += kTile) {
#pragma unroll
        for (int idx = tid; idx < kTile * kTile; idx += kThreads * kThreads) {
            const int r = idx / kTile;
            const int c = idx % kTile;

            const int ar = blockRow + r;
            const int ac = t + c;
            As[r][c] = (ar < rows && ac < N)
                           ? A[static_cast<size_t>(ar) * N + ac]
                           : 0.0;

            const int br = t + r;
            const int bc = blockCol + c;
            Bs[r][c] = (br < N && bc < N)
                           ? B[static_cast<size_t>(br) * N + bc]
                           : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const double a0 = As[ty][kk];
            const double a1 = As[ty + kThreads][kk];
            const double b0 = Bs[kk][tx];
            const double b1 = Bs[kk][tx + kThreads];
            if constexpr (UseFma) {
                sum00 = __fma_rn(a0, b0, sum00);
                sum01 = __fma_rn(a0, b1, sum01);
                sum10 = __fma_rn(a1, b0, sum10);
                sum11 = __fma_rn(a1, b1, sum11);
            } else {
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
            }
        }
        __syncthreads();
    }

    if (row0 < rows) {
        if (col0 < N) C[static_cast<size_t>(row0) * N + col0] = sum00;
        if (col1 < N) C[static_cast<size_t>(row0) * N + col1] = sum01;
    }
    if (row1 < rows) {
        if (col0 < N) C[static_cast<size_t>(row1) * N + col0] = sum10;
        if (col1 < N) C[static_cast<size_t>(row1) * N + col1] = sum11;
    }
}

// Simple validation: compute a few elements independently and compare
bool validateResult(const std::vector<double>& B, const std::vector<double>& C,
                    const size_t N) {
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

struct DeviceWork {
    int device = 0;
    size_t subStart = 0;
    size_t subRows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
};

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    // Determine which GPUs belong to this process.  When there are at least as
    // many processes as devices, each process owns one device; otherwise the
    // node-local processes split the devices among themselves and OpenMP
    // threads drive the individual GPUs concurrently.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int deviceCount = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA device initialization failed: %s\n",
                    cudaGetErrorString(devErr));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int firstDevice = 0;
    int localDevices = 1;
    if (nodeSize >= deviceCount) {
        firstDevice = nodeRank % deviceCount;
        localDevices = 1;
    } else {
        const int baseDevices = deviceCount / nodeSize;
        const int remainder = deviceCount % nodeSize;
        localDevices = baseDevices + (nodeRank < remainder ? 1 : 0);
        firstDevice = nodeRank * baseDevices + std::min(nodeRank, remainder);
        if (localDevices <= 0) {
            firstDevice = nodeRank % deviceCount;
            localDevices = 1;
        }
    }

    if (omp_get_max_threads() < localDevices) {
        omp_set_num_threads(localDevices);
    }

    const bool useFma = (N & (N - 1)) == 0 && N <= 8192;

    // Split the output rows over MPI processes, weighted by how many GPUs
    // each process owns so that all GPUs in the allocation do equal work.
    std::vector<int> allLocalDevices(static_cast<size_t>(numProcs));
    MPI_Allgather(&localDevices, 1, MPI_INT, allLocalDevices.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    std::vector<size_t> rankStarts;
    std::vector<size_t> rankCounts;
    splitRowsWeighted(N, allLocalDevices, rankStarts, rankCounts);
    const size_t rowStart = rankStarts[static_cast<size_t>(rank)];
    const size_t rows = rankCounts[static_cast<size_t>(rank)];

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI (%d ranks) + OpenMP (%d threads) + CUDA (%d devices)\n",
               numProcs, omp_get_max_threads(), deviceCount);
    }

    // Each rank only stores its own rows of A and C.  B is required in full,
    // and is re-generated deterministically locally so no broadcast is needed.
    std::vector<double> localA(rows * N);
    std::vector<double> B(N * N);
    std::vector<double> localC(rows * N);
    std::vector<double> C;

    if (rank == 0) {
        printf("Initializing matrices...\n");
        C.resize(N * N);
    }

    initMatrix(localA, N, rowStart, rows);
    initMatrix(B, N, 0, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Split this rank's rows over the node-local GPUs.
    std::vector<DeviceWork> work(static_cast<size_t>(localDevices));
    {
        std::vector<size_t> subStarts;
        std::vector<size_t> subCounts;
        splitRows(rows, localDevices, subStarts, subCounts);
        for (int d = 0; d < localDevices; ++d) {
            DeviceWork& w = work[static_cast<size_t>(d)];
            w.device = firstDevice + d;
            w.subStart = subStarts[static_cast<size_t>(d)];
            w.subRows = subCounts[static_cast<size_t>(d)];
        }
    }

    // Stage A and B on the devices before timing (as in the original code,
    // only the actual multiplication is timed).
#pragma omp parallel for schedule(static, 1) num_threads(localDevices)
    for (int d = 0; d < localDevices; ++d) {
        DeviceWork& w = work[static_cast<size_t>(d)];
        if (w.subRows == 0) continue;
        cudaSetDevice(w.device);
        cudaMalloc(&w.dA, w.subRows * N * sizeof(double));
        cudaMalloc(&w.dB, N * N * sizeof(double));
        cudaMalloc(&w.dC, w.subRows * N * sizeof(double));
        cudaMemcpy(w.dA, localA.data() + w.subStart * N,
                   w.subRows * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(w.dB, B.data(), N * N * sizeof(double),
                   cudaMemcpyHostToDevice);
        // Warm up the context and load the kernel module before timing.
        matmulKernel<true><<<1, 1>>>(w.dA, w.dB, w.dC, 0, 0);
        matmulKernel<false><<<1, 1>>>(w.dA, w.dB, w.dC, 0, 0);
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = omp_get_wtime();

    // Every OpenMP thread drives its own GPU; with several GPUs per process
    // the row blocks are multiplied concurrently.
#pragma omp parallel for schedule(static, 1) num_threads(localDevices)
    for (int d = 0; d < localDevices; ++d) {
        DeviceWork& w = work[static_cast<size_t>(d)];
        if (w.subRows == 0) continue;
        cudaSetDevice(w.device);

        const dim3 block(kThreads, kThreads);
        const dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile),
                        static_cast<unsigned>((w.subRows + kTile - 1) / kTile));

        if (useFma) {
            matmulKernel<true><<<grid, block>>>(w.dA, w.dB, w.dC,
                                                static_cast<int>(N),
                                                static_cast<int>(w.subRows));
        } else {
            matmulKernel<false><<<grid, block>>>(w.dA, w.dB, w.dC,
                                                 static_cast<int>(N),
                                                 static_cast<int>(w.subRows));
        }
        cudaDeviceSynchronize();
    }

    const double t1 = omp_get_wtime();
    double localSeconds = t1 - t0;
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Bring the computed rows back and release the device buffers.
#pragma omp parallel for schedule(static, 1) num_threads(localDevices)
    for (int d = 0; d < localDevices; ++d) {
        DeviceWork& w = work[static_cast<size_t>(d)];
        if (w.subRows == 0) continue;
        cudaSetDevice(w.device);
        cudaMemcpy(localC.data() + w.subStart * N, w.dC,
                   w.subRows * N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(w.dA);
        cudaFree(w.dB);
        cudaFree(w.dC);
    }

    // Assemble the full result matrix on rank 0, where output is produced.
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<int> gatherCounts;
    std::vector<int> gatherDispls;
    int sendRows = static_cast<int>(rows);
    if (rank == 0) {
        gatherCounts.resize(static_cast<size_t>(numProcs));
        gatherDispls.resize(static_cast<size_t>(numProcs));
        for (int r = 0; r < numProcs; ++r) {
            gatherCounts[static_cast<size_t>(r)] =
                static_cast<int>(rankCounts[static_cast<size_t>(r)]);
            gatherDispls[static_cast<size_t>(r)] =
                static_cast<int>(rankStarts[static_cast<size_t>(r)]);
        }
    }

    MPI_Gatherv(sendRows > 0 ? localC.data() : nullptr, sendRows, rowType,
                rank == 0 ? C.data() : nullptr,
                rank == 0 ? gatherCounts.data() : nullptr,
                rank == 0 ? gatherDispls.data() : nullptr,
                rowType, 0, MPI_COMM_WORLD);

    MPI_Type_free(&rowType);
    MPI_Comm_free(&nodeComm);

    int exitCode = 0;

    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(std::llround(maxSeconds * 1000.0));
        printf("Computation time: %lld ms\n", milliseconds);

        const double gflops = maxSeconds > 0.0
                                  ? (2.0 * static_cast<double>(N) * N * N) /
                                        maxSeconds / 1e9
                                  : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
