#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Generate pseudo-random values for matrix initialization (host and device)
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Tiling parameters for the GEMM kernel
constexpr int THREADS = 16;                  // threads per block dimension
constexpr int TILE = 64;                     // C sub-block computed per thread block
constexpr int REG = TILE / THREADS;          // C elements per thread per dimension
constexpr int BK = 16;                       // K-depth staged in shared memory per step

// Fill `rows` rows of an N-column matrix, starting at global row `rowOffset`
__global__ void initMatrixKernel(double* __restrict__ mat, const size_t rowOffset, const size_t rows, const size_t N) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * N) {
        return;
    }
    mat[idx] = getPseudoRndValue(N, rowOffset + idx / N, idx % N);
}

// C[M x N] = A[M x N] * B[N x N] (row-major). Every C element accumulates over k in
// ascending order. The reference loop `sum += A*B` is compiled with each product rounded
// before the add, except that the last term of an odd-length dot product is fused; this
// kernel reproduces that rounding so the results are bit-identical to the reference.
__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t M, const size_t N) {
    __shared__ double As[TILE][BK + 1];
    __shared__ double Bs[BK][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE;

    double acc[REG][REG] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        for (int e = tid; e < TILE * BK; e += THREADS * THREADS) {
            const int r = e / BK;
            const int c = e % BK;
            const size_t gr = rowBase + r;
            const size_t gc = k0 + c;
            As[r][c] = (gr < M && gc < N) ? A[gr * N + gc] : 0.0;
        }
        for (int e = tid; e < BK * TILE; e += THREADS * THREADS) {
            const int r = e / TILE;
            const int c = e % TILE;
            const size_t gr = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

        const size_t kmax = (N - k0 < static_cast<size_t>(BK)) ? (N - k0) : static_cast<size_t>(BK);
        for (size_t kk = 0; kk < kmax; ++kk) {
            const bool fuseTerm = (N % 2 == 1) && (k0 + kk + 1 == N);
            double a[REG];
            double b[REG];
            for (int r = 0; r < REG; ++r) {
                a[r] = As[ty + r * THREADS][kk];
            }
            for (int c = 0; c < REG; ++c) {
                b[c] = Bs[kk][tx + c * THREADS];
            }
            for (int r = 0; r < REG; ++r) {
                for (int c = 0; c < REG; ++c) {
                    acc[r][c] = fuseTerm ? fma(a[r], b[c], acc[r][c])
                                         : __dadd_rn(acc[r][c], __dmul_rn(a[r], b[c]));
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < REG; ++r) {
        const size_t gr = rowBase + ty + r * THREADS;
        if (gr >= M) {
            continue;
        }
        for (int c = 0; c < REG; ++c) {
            const size_t gc = colBase + tx + c * THREADS;
            if (gc < N) {
                C[gr * N + gc] = acc[r][c];
            }
        }
    }
}

// Split `total` items into `parts` contiguous blocks; returns block `idx`
void splitRange(const size_t total, const int parts, const int idx, size_t& begin, size_t& end) {
    begin = total * static_cast<size_t>(idx) / parts;
    end = total * static_cast<size_t>(idx + 1) / parts;
}

// One GPU's share of this rank's rows of A and C; B is replicated on each GPU
struct GpuBlock {
    int device = 0;
    size_t row0 = 0;   // first global row of this block
    size_t rows = 0;   // number of rows in this block
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
};

// Simple validation: compute a 5x5 set of elements directly and compare.
// Each element is computed by the rank that owns its row; the results are then combined
// so that every rank sees the full sample. Failure messages are printed by rank 0.
bool validateResult(const double* localC, const size_t row0, const size_t rows, const size_t N, const int rank) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    constexpr int NUM_CHECKS = 5 * 5;

    double expected[NUM_CHECKS] = {};
    double actual[NUM_CHECKS] = {};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            if (i < row0 || i >= row0 + rows) {
                continue;
            }

            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            expected[pi * 5 + pj] = sum;
            actual[pi * 5 + pj] = localC[(i - row0) * N + j];
        }
    }

    // Each sample is owned by exactly one rank, so the sum recovers its value exactly
    MPI_Allreduce(MPI_IN_PLACE, expected, NUM_CHECKS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, actual, NUM_CHECKS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            const double e = expected[pi * 5 + pj];
            const double a = actual[pi * 5 + pj];
            const double relError = std::abs((a - e) / (e + 1e-10));

            if (relError > 1e-6) {
                if (rank == 0) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, e, a, relError);
                }
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

int runBenchmark(int argc, char** argv, const int rank, const int numRanks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select the GPUs for this rank: ranks on a node share the node's devices round-robin
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> devices;
    if (localSize <= deviceCount) {
        for (int d = localRank; d < deviceCount; d += localSize) {
            devices.push_back(d);
        }
    } else {
        devices.push_back(localRank % deviceCount);
    }
    const int numDevices = static_cast<int>(devices.size());

    // Row block of C owned by this rank, subdivided among its GPUs
    size_t rankRow0 = 0;
    size_t rankRow1 = 0;
    splitRange(N, numRanks, rank, rankRow0, rankRow1);
    const size_t rankRows = rankRow1 - rankRow0;

    std::vector<GpuBlock> blocks(numDevices);
    for (int t = 0; t < numDevices; ++t) {
        size_t b = 0;
        size_t e = 0;
        splitRange(rankRows, numDevices, t, b, e);
        blocks[t].device = devices[t];
        blocks[t].row0 = rankRow0 + b;
        blocks[t].rows = e - b;
    }

    // Host-side result storage for this rank's rows (pinned for fast device-to-host copies)
    double* hostC = nullptr;
    if (rankRows * N > 0) {
        CUDA_CHECK(cudaMallocHost(&hostC, rankRows * N * sizeof(double)));
    }

    // Allocate device memory and generate A and B directly on each GPU (one OpenMP thread per GPU)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    #pragma omp parallel for num_threads(numDevices) schedule(static, 1)
    for (int t = 0; t < numDevices; ++t) {
        GpuBlock& g = blocks[t];
        CUDA_CHECK(cudaSetDevice(g.device));
        if (g.rows > 0 && N > 0) {
            CUDA_CHECK(cudaMalloc(&g.A, g.rows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&g.C, g.rows * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&g.B, N * N * sizeof(double)));

            const unsigned int blkA = static_cast<unsigned int>((g.rows * N + 255) / 256);
            const unsigned int blkB = static_cast<unsigned int>((N * N + 255) / 256);
            initMatrixKernel<<<blkA, 256>>>(g.A, g.row0, g.rows, N);
            CUDA_CHECK(cudaGetLastError());
            initMatrixKernel<<<blkB, 256>>>(g.B, 0, N, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Perform matrix multiplication (kernels + copy of results back to host), timed across all ranks
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(numDevices) schedule(static, 1)
    for (int t = 0; t < numDevices; ++t) {
        const GpuBlock& g = blocks[t];
        if (g.rows == 0 || N == 0) {
            continue;
        }
        CUDA_CHECK(cudaSetDevice(g.device));

        const dim3 block(THREADS, THREADS);
        const dim3 grid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                        static_cast<unsigned int>((g.rows + TILE - 1) / TILE));
        matrixMultiplyKernel<<<grid, block>>>(g.A, g.B, g.C, g.rows, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hostC + (g.row0 - rankRow0) * N, g.C, g.rows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedSec = std::chrono::duration<double>(end - start).count();

    // The distributed multiply finishes when the slowest rank does
    double maxElapsedSec = 0.0;
    MPI_Reduce(&elapsedSec, &maxElapsedSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Release device memory
    for (const GpuBlock& g : blocks) {
        if (g.rows > 0 && N > 0) {
            CUDA_CHECK(cudaSetDevice(g.device));
            CUDA_CHECK(cudaFree(g.A));
            CUDA_CHECK(cudaFree(g.B));
            CUDA_CHECK(cudaFree(g.C));
        }
    }

    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxElapsedSec));
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / maxElapsedSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full C matrix on rank 0 for printing
    if (printResults) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<double> fullC;
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            fullC.resize(N * N);
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                size_t b = 0;
                size_t e = 0;
                splitRange(N, numRanks, r, b, e);
                counts[r] = static_cast<int>(e - b);
                displs[r] = static_cast<int>(b);
            }
        }
        MPI_Gatherv(hostC, static_cast<int>(rankRows), rowType,
                    fullC.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);

        if (rank == 0) {
            print_results(fullC, "MatrixC");
        }
    }

    // Validation
    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(hostC, rankRow0, rankRows, N, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        status = valid ? 0 : 1;
    }

    if (hostC != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostC));
    }
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int status = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return status;
}
