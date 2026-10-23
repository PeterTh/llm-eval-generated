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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a block of rows. dst points at the first row to be written, while
// globalRowBegin identifies the row of the logical matrix stored first in dst.
void initMatrixRows(double* dst, const size_t N, const size_t numRows,
                    const size_t globalRowBegin) {
    if (numRows == 0 || N == 0) {
        return;
    }

#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(numRows); ++r) {
        const size_t globalRow = globalRowBegin + static_cast<size_t>(r);
        double* row = dst + static_cast<size_t>(r) * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, globalRow, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixRows(mat.data(), N, N, 0);
}

// Tiled double-precision GEMM. Each block computes a 32x32 tile of C and each
// thread accumulates a 2x2 sub-tile. A and B are gathered into shared memory
// so that every value loaded from global memory is reused many times.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const long long N,
                                     const long long M) {
    constexpr int TILE = 32;
    __shared__ double As[TILE][TILE + 1];
    __shared__ double Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int blockRow = blockIdx.y * TILE;
    const int blockCol = blockIdx.x * TILE;
    const int row0 = blockRow + ty * 2;
    const int col0 = blockCol + tx * 2;

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    for (int t = 0; t < N; t += TILE) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int idx = tid + e * (TILE * TILE / 4);
            const int r = idx / TILE;
            const int c = idx - r * TILE;

            const int gr = blockRow + r;
            const int gc = t + c;
            As[r][c] = (gr < M && gc < N)
                           ? A[static_cast<size_t>(gr) * N + gc]
                           : 0.0;

            const int br = t + r;
            const int bc = blockCol + c;
            Bs[r][c] = (br < N && bc < N)
                           ? B[static_cast<size_t>(br) * N + bc]
                           : 0.0;
        }

        __syncthreads();

#pragma unroll 8
        for (int kk = 0; kk < TILE; ++kk) {
            const double a0 = As[ty * 2][kk];
            const double a1 = As[ty * 2 + 1][kk];
            const double b0 = Bs[kk][tx * 2];
            const double b1 = Bs[kk][tx * 2 + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N) {
        C[static_cast<size_t>(row0) * N + col0] = acc00;
    }
    if (row0 < M && col0 + 1 < N) {
        C[static_cast<size_t>(row0) * N + col0 + 1] = acc01;
    }
    if (row0 + 1 < M && col0 < N) {
        C[static_cast<size_t>(row0 + 1) * N + col0] = acc10;
    }
    if (row0 + 1 < M && col0 + 1 < N) {
        C[static_cast<size_t>(row0 + 1) * N + col0 + 1] = acc11;
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const double* A, const double* B,
                    const double* C, const size_t N) {
    if (N == 0) {
        return true;
    }

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

struct DeviceWork {
    int device = -1;
    size_t beginRow = 0;
    size_t rows = 0;
    double* hA = nullptr;
    double* hC = nullptr;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
};

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Discover the ranks that share this node, then reuse the shared-memory
    // communicator to hand out the local GPUs deterministically.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 1;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 1) {
        if (worldRank == 0) {
            fprintf(stderr, "No CUDA-capable device found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Avoid oversubscribing the CPUs when several MPI ranks share a node.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        int threads = omp_get_max_threads() / localSize;
        if (threads < 1) {
            threads = 1;
        }
        omp_set_num_threads(threads);
    }

    // Give every local rank a contiguous set of GPUs. If there are fewer GPUs
    // than ranks, ranks share GPUs in round-robin fashion.
    int deviceAssignCount = 1;
    int deviceAssignStart = 0;
    if (deviceCount >= localSize) {
        const int q = deviceCount / localSize;
        const int r = deviceCount % localSize;
        deviceAssignCount = q + (localRank < r ? 1 : 0);
        deviceAssignStart = localRank * q + std::min(localRank, r);
    } else {
        deviceAssignCount = 1;
        deviceAssignStart = localRank % deviceCount;
    }

    // Rows of C are distributed across GPUs, not merely across ranks, so that
    // nodes with several GPUs per rank keep every accelerator busy. Each rank
    // contributes one weight unit per GPU it owns.
    int localWeight = deviceAssignCount;
    std::vector<int> weights(static_cast<size_t>(worldSize), 1);
    MPI_Allgather(&localWeight, 1, MPI_INT, weights.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    size_t totalWeight = 0;
    size_t unitsBefore = 0;
    for (int r = 0; r < worldSize; ++r) {
        const size_t w = static_cast<size_t>(weights[static_cast<size_t>(r)]);
        if (r < worldRank) {
            unitsBefore += w;
        }
        totalWeight += w;
    }
    if (totalWeight == 0) {
        totalWeight = 1;
    }

    const size_t baseRows = N / totalWeight;
    const size_t extraRows = N % totalWeight;
    auto unitToRow = [baseRows, extraRows](size_t unit) -> size_t {
        return unit * baseRows + std::min(unit, extraRows);
    };

    const size_t rankBegin = unitToRow(unitsBefore);
    const size_t rankEnd = unitToRow(unitsBefore + static_cast<size_t>(localWeight));
    const size_t rankRows = rankEnd - rankBegin;

    std::vector<DeviceWork> works;
    works.reserve(static_cast<size_t>(deviceAssignCount));

    const size_t matrixElements = N * N;
    double* hB = nullptr;
    if (matrixElements > 0) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&hB),
                                matrixElements * sizeof(double),
                                cudaHostAllocDefault),
                  "cudaHostAlloc(B)");
        initMatrixRows(hB, N, N, 0);
    }

    // Split this rank's rows evenly over its local GPUs.
    const size_t rowsPerDevice = deviceAssignCount > 0
                                     ? rankRows / static_cast<size_t>(deviceAssignCount)
                                     : 0;
    const size_t extraDeviceRows = deviceAssignCount > 0
                                       ? rankRows % static_cast<size_t>(deviceAssignCount)
                                       : 0;
    size_t rowOffset = 0;
    for (int d = 0; d < deviceAssignCount; ++d) {
        DeviceWork work;
        work.device = deviceAssignStart + d;
        work.beginRow = rankBegin + rowOffset;
        work.rows = rowsPerDevice + (static_cast<size_t>(d) < extraDeviceRows ? 1 : 0);
        rowOffset += work.rows;

        if (work.rows == 0) {
            continue;
        }

        cudaCheck(cudaSetDevice(work.device), "cudaSetDevice");
        const size_t localElements = work.rows * N;
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&work.hA),
                                localElements * sizeof(double),
                                cudaHostAllocDefault),
                  "cudaHostAlloc(A)");
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&work.hC),
                                localElements * sizeof(double),
                                cudaHostAllocDefault),
                  "cudaHostAlloc(C)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.dA),
                             localElements * sizeof(double)),
                  "cudaMalloc(A)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.dB),
                             matrixElements * sizeof(double)),
                  "cudaMalloc(B)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.dC),
                             localElements * sizeof(double)),
                  "cudaMalloc(C)");

        initMatrixRows(work.hA, N, work.rows, work.beginRow);
        works.push_back(work);
    }

    if (worldRank == 0) {
        printf("Initializing matrices...\n");
        printf("Computing matrix multiplication...\n");
    }

    // Touch the kernel once per device so that module loading and first-launch
    // overhead are not charged to the measured region.
    if (!works.empty() && N > 0) {
        const int numWorks = static_cast<int>(works.size());
#pragma omp parallel for schedule(static) num_threads(numWorks)
        for (int w = 0; w < numWorks; ++w) {
            DeviceWork& work = works[static_cast<size_t>(w)];
            cudaCheck(cudaSetDevice(work.device), "cudaSetDevice");
            const dim3 block(16, 16);
            const dim3 grid(1, 1);
            matrixMultiplyKernel<<<grid, block>>>(
                work.dA, work.dB, work.dC,
                static_cast<long long>(N),
                static_cast<long long>(work.rows));
            cudaCheck(cudaGetLastError(), "warmup matrixMultiplyKernel");
            cudaCheck(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");
        }
    }

    // Compute. Every GPU owned by every rank runs independently; OpenMP turns
    // the per-GPU launches into a single parallel team and MPI provides the
    // global row decomposition.
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    if (!works.empty() && N > 0) {
        const int numWorks = static_cast<int>(works.size());
#pragma omp parallel for schedule(static) num_threads(numWorks)
        for (int w = 0; w < numWorks; ++w) {
            DeviceWork& work = works[static_cast<size_t>(w)];
            cudaCheck(cudaSetDevice(work.device), "cudaSetDevice");

            const size_t localElements = work.rows * N;
            const size_t localBytes = localElements * sizeof(double);

            cudaCheck(cudaMemcpy(work.dA, work.hA, localBytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(A)");
            cudaCheck(cudaMemcpy(work.dB, hB, matrixElements * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(B)");

            const dim3 block(16, 16);
            const dim3 grid(static_cast<unsigned>((N + 31) / 32),
                            static_cast<unsigned>((work.rows + 31) / 32));
            matrixMultiplyKernel<<<grid, block>>>(
                work.dA, work.dB, work.dC,
                static_cast<long long>(N),
                static_cast<long long>(work.rows));
            cudaCheck(cudaGetLastError(), "matrixMultiplyKernel");

            cudaCheck(cudaMemcpy(work.hC, work.dC, localBytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(C)");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds =
        std::chrono::duration<double>(end - start).count();

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate GFLOPS
        const double gflops = (2.0 * static_cast<double>(N) * N * N) /
                              (seconds > 0.0 ? seconds : 1e-9) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the distributed C rows on rank 0 for optional output/validation.
    const bool needResults = validate || printResults;
    std::vector<double> localC;
    std::vector<double> fullC;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (needResults) {
        localC.resize(rankRows * N);
        for (const DeviceWork& work : works) {
            const size_t offset = (work.beginRow - rankBegin) * N;
            const size_t bytes = work.rows * N * sizeof(double);
            std::memcpy(localC.data() + offset, work.hC, bytes);
        }

        if (worldRank == 0) {
            fullC.resize(matrixElements);
            recvCounts.resize(static_cast<size_t>(worldSize));
            displs.resize(static_cast<size_t>(worldSize));
            for (int r = 0; r < worldSize; ++r) {
                size_t before = 0;
                for (int q = 0; q < r; ++q) {
                    before += static_cast<size_t>(weights[static_cast<size_t>(q)]);
                }
                const size_t begin = unitToRow(before);
                const size_t rowCount =
                    unitToRow(before + static_cast<size_t>(weights[static_cast<size_t>(r)])) - begin;
                recvCounts[static_cast<size_t>(r)] =
                    static_cast<int>(rowCount * N);
                displs[static_cast<size_t>(r)] =
                    static_cast<int>(begin * N);
            }
        }

        MPI_Gatherv(localC.data(),
                    static_cast<int>(rankRows * N), MPI_DOUBLE,
                    worldRank == 0 ? fullC.data() : nullptr,
                    worldRank == 0 ? recvCounts.data() : nullptr,
                    worldRank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (worldRank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(fullC, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA(matrixElements);
            initMatrix(fullA, N);
            const bool valid =
                validateResult(fullA.data(), hB, fullC.data(), N);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    for (const DeviceWork& work : works) {
        cudaSetDevice(work.device);
        cudaFree(work.dA);
        cudaFree(work.dB);
        cudaFree(work.dC);
        cudaFreeHost(work.hA);
        cudaFreeHost(work.hC);
    }
    if (hB != nullptr) {
        cudaFreeHost(hB);
    }

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
