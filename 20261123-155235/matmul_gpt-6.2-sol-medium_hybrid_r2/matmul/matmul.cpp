#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Range {
    int first;
    int count;
};

struct Block {
    Range rows;
    Range cols;

    size_t elements() const {
        return static_cast<size_t>(rows.count) * cols.count;
    }
};

Range split(int n, int parts, int part) {
    const int extra = n % parts;
    return {part * (n / parts) + std::min(part, extra), n / parts + (part < extra)};
}

Block blockFor(int n, const int dims[2], int rank) {
    return {split(n, dims[0], rank / dims[1]), split(n, dims[1], rank % dims[1])};
}

[[noreturn]] void fail(int rank, const char* message) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(cudaError_t status, int rank, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

void initializeInputs(double* a, double* b, int n, Block block) {
    const size_t elements = static_cast<size_t>(n) *
                            (static_cast<size_t>(block.rows.count) + block.cols.count);
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), (elements + 65535) / 65536));

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static) nowait
        for (int row = 0; row < block.rows.count; ++row) {
            for (int j = 0; j < n; ++j) {
                a[static_cast<size_t>(row) * n + j] = getPseudoRndValue(n, block.rows.first + row, j);
            }
        }

        #pragma omp for schedule(static) nowait
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < block.cols.count; ++j) {
                b[static_cast<size_t>(k) * block.cols.count + j] =
                    getPseudoRndValue(n, k, block.cols.first + j);
            }
        }
    }
}

constexpr int ThreadsX = 32;
constexpr int ThreadsY = 8;
constexpr int TileK = 16;
constexpr int LargeTile = 64;
constexpr int SmallTile = 32;

template <bool Fused, int RowsPerThread, int ColsPerThread>
__device__ __forceinline__ void accumulate(
    const double (&a)[RowsPerThread * ThreadsY][TileK + 1],
    const double (&b)[TileK][ColsPerThread * ThreadsX],
    double (&sums)[RowsPerThread][ColsPerThread], int k) {
    double right[ColsPerThread];
    #pragma unroll
    for (int j = 0; j < ColsPerThread; ++j) {
        right[j] = b[k][threadIdx.x + j * ThreadsX];
    }

    #pragma unroll
    for (int i = 0; i < RowsPerThread; ++i) {
        const double left = a[threadIdx.y + i * ThreadsY][k];
        #pragma unroll
        for (int j = 0; j < ColsPerThread; ++j) {
            if constexpr (Fused) {
                sums[i][j] = __fma_rn(left, right[j], sums[i][j]);
            } else {
                sums[i][j] = __dadd_rn(sums[i][j], __dmul_rn(left, right[j]));
            }
        }
    }
}

template <int TileSize>
__global__ void matrixMultiply(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c, int rows, int cols, int n, bool fuseTail) {
    constexpr int RowsPerThread = TileSize / ThreadsY;
    constexpr int ColsPerThread = TileSize / ThreadsX;
    __shared__ double aTile[TileSize][TileK + 1];
    __shared__ double bTile[TileK][TileSize];
    double sums[RowsPerThread][ColsPerThread] = {};

    const int rowBase = blockIdx.y * TileSize;
    const int colBase = blockIdx.x * TileSize;
    const int thread = threadIdx.y * ThreadsX + threadIdx.x;

    for (int base = 0; base < n;) {
        const int steps = min(TileK, n - base);

        #pragma unroll
        for (int index = thread; index < TileSize * TileK; index += ThreadsX * ThreadsY) {
            const int i = index / TileK;
            const int k = index % TileK;
            if (rowBase + i < rows && k < steps) {
                aTile[i][k] = a[static_cast<size_t>(rowBase + i) * n + base + k];
            } else {
                aTile[i][k] = 0.0;
            }
        }

        #pragma unroll
        for (int index = thread; index < TileK * TileSize; index += ThreadsX * ThreadsY) {
            const int k = index / TileSize;
            const int j = index % TileSize;
            if (k < steps && colBase + j < cols) {
                bTile[k][j] = b[static_cast<size_t>(base + k) * cols + colBase + j];
            } else {
                bTile[k][j] = 0.0;
            }
        }
        __syncthreads();

        if (steps == TileK) {
            #pragma unroll
            for (int k = 0; k < TileK; ++k) {
                accumulate<false>(aTile, bTile, sums, k);
            }
        } else {
            const bool fuse = fuseTail && (steps % 2 != 0);
            for (int k = 0; k < steps - fuse; ++k) {
                accumulate<false>(aTile, bTile, sums, k);
            }
            if (fuse) {
                accumulate<true>(aTile, bTile, sums, steps - 1);
            }
        }
        __syncthreads();
        base += steps;
    }

    #pragma unroll
    for (int i = 0; i < RowsPerThread; ++i) {
        const int row = rowBase + threadIdx.y + i * ThreadsY;
        if (row < rows) {
            #pragma unroll
            for (int j = 0; j < ColsPerThread; ++j) {
                const int col = colBase + threadIdx.x + j * ThreadsX;
                if (col < cols) {
                    c[static_cast<size_t>(row) * cols + col] = sums[i][j];
                }
            }
        }
    }
}

void gatherResults(const double* deviceC, std::vector<double>& result, int n,
                   const int dims[2], Block block, int rank, int ranks) {
    if (rank == 0) {
        result.resize(static_cast<size_t>(n) * n);
        std::vector<MPI_Request> requests;
        requests.reserve(ranks - 1);
        for (int source = 1; source < ranks; ++source) {
            const Block remote = blockFor(n, dims, source);
            if (remote.elements() == 0) {
                continue;
            }

            MPI_Datatype rectangle;
            MPI_Type_vector(remote.rows.count, remote.cols.count, n, MPI_DOUBLE, &rectangle);
            MPI_Type_commit(&rectangle);
            double* destination = result.data() + static_cast<size_t>(remote.rows.first) * n
                                + remote.cols.first;
            requests.emplace_back();
            MPI_Irecv(destination, 1, rectangle, source, 0, MPI_COMM_WORLD, &requests.back());
            MPI_Type_free(&rectangle);
        }
        if (block.elements() != 0) {
            const size_t width = static_cast<size_t>(block.cols.count) * sizeof(double);
            checkCuda(cudaMemcpy2D(result.data(), static_cast<size_t>(n) * sizeof(double),
                                   deviceC, width, width, block.rows.count, cudaMemcpyDeviceToHost),
                      rank, "Copying root result block");
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    } else if (block.elements() != 0) {
        std::unique_ptr<double[]> local(new double[block.elements()]);
        checkCuda(cudaMemcpy(local.get(), deviceC, block.elements() * sizeof(double),
                             cudaMemcpyDeviceToHost), rank, "Copying result block");

        // A single datatype also supports blocks larger than MPI's int count.
        MPI_Datatype rectangle;
        MPI_Type_vector(block.rows.count, block.cols.count, block.cols.count, MPI_DOUBLE, &rectangle);
        MPI_Type_commit(&rectangle);
        MPI_Send(local.get(), 1, rectangle, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rectangle);
    }
}

constexpr int CheckWidth = 5;
using CheckValues = std::array<double, CheckWidth * CheckWidth>;

CheckValues collectChecks(const double* deviceC, Block block, int rank) {
    CheckValues local{};
    CheckValues checks{};
    const int rows = std::min(block.rows.count, CheckWidth - block.rows.first);
    const int cols = std::min(block.cols.count, CheckWidth - block.cols.first);
    if (rows > 0 && cols > 0) {
        const size_t pitch = static_cast<size_t>(block.cols.count) * sizeof(double);
        checkCuda(cudaMemcpy2D(local.data() + block.rows.first * CheckWidth + block.cols.first,
                               CheckWidth * sizeof(double), deviceC, pitch,
                               cols * sizeof(double), rows, cudaMemcpyDeviceToHost),
                  rank, "Copying validation samples");
    }
    MPI_Reduce(local.data(), checks.data(), checks.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return checks;
}

bool validateResult(const CheckValues& checks, const std::vector<double>& result, int n) {
    if (n == 0) {
        return true;
    }
    for (int pi = 0; pi < CheckWidth; ++pi) {
        for (int pj = 0; pj < CheckWidth; ++pj) {
            const int i = pi % n;
            const int j = pj % n;
            double expected = 0.0;
            for (int k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            }

            const double actual = result.empty() ? checks[i * CheckWidth + j]
                                                 : result[static_cast<size_t>(i) * n + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
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

int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long value = strtol(argument, &end, 10);
            if (errno || end == argument || *end || value < 0 || value > INT_MAX ||
                (value > 0 && static_cast<size_t>(value) > SIZE_MAX / sizeof(double) / value)) {
                if (rank == 0) {
                    printf("Invalid matrix size: %s\n", argument);
                    printUsage(argv[0]);
                }
                return 1;
            }
            n = static_cast<int>(value);
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

    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    const Block block = blockFor(n, dims, rank);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    int tileSize = LargeTile;
    auto kernel = matrixMultiply<LargeTile>;
    if (block.elements() != 0) {
        int devices;
        checkCuda(cudaGetDeviceCount(&devices), rank, "Finding CUDA devices");
        if (devices == 0) {
            fail(rank, "No visible CUDA device");
        }
        const int device = localRank % devices;
        checkCuda(cudaSetDevice(device), rank, "Selecting CUDA device");

        int multiprocessors;
        checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                  rank, "Querying CUDA multiprocessors");
        const size_t largeTileCount = static_cast<size_t>((block.rows.count - 1) / LargeTile + 1)
                                    * ((block.cols.count - 1) / LargeTile + 1);
        if (largeTileCount < static_cast<size_t>(2 * multiprocessors)) {
            tileSize = SmallTile;
            kernel = matrixMultiply<SmallTile>;
        }
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, kernel), rank, "Loading CUDA kernel");

        const size_t aSize = static_cast<size_t>(block.rows.count) * n;
        const size_t bSize = static_cast<size_t>(n) * block.cols.count;
        checkCuda(cudaMalloc(&deviceA, aSize * sizeof(double)), rank, "Allocating device input A");
        checkCuda(cudaMalloc(&deviceB, bSize * sizeof(double)), rank, "Allocating device input B");
        checkCuda(cudaMalloc(&deviceC, block.elements() * sizeof(double)), rank, "Allocating device output");

        double* a;
        double* b;
        checkCuda(cudaMallocHost(&a, aSize * sizeof(double)), rank, "Allocating host input A");
        checkCuda(cudaMallocHost(&b, bSize * sizeof(double)), rank, "Allocating host input B");
        initializeInputs(a, b, n, block);
        checkCuda(cudaMemcpy(deviceA, a, aSize * sizeof(double), cudaMemcpyHostToDevice),
                  rank, "Copying input A");
        checkCuda(cudaMemcpy(deviceB, b, bSize * sizeof(double), cudaMemcpyHostToDevice),
                  rank, "Copying input B");
        checkCuda(cudaFreeHost(a), rank, "Freeing host input A");
        checkCuda(cudaFreeHost(b), rank, "Freeing host input B");
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (block.elements() != 0) {
        const dim3 threads(ThreadsX, ThreadsY);
        int rows = block.rows.count;
        int cols = block.cols.count;
        const dim3 grid((cols - 1) / tileSize + 1, (rows - 1) / tileSize + 1);

        // The original GCC x86 loop rounds vectorized products separately and
        // contracts the scalar remainder when the host supports FMA.
        #if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__) && \
            (defined(__x86_64__) || defined(__i386__))
        bool fuseTail = true;
        #else
        bool fuseTail = false;
        #endif
        void* arguments[] = {&deviceA, &deviceB, &deviceC, &rows, &cols, &n, &fuseTail};
        checkCuda(cudaLaunchKernel(kernel, grid, threads, arguments), rank, "Launching matrix multiplication");
        checkCuda(cudaDeviceSynchronize(), rank, "Computing matrix multiplication");
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double gflops = elapsed > 0.0 ? (2.0 * n * n * n) / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (printResults) {
        gatherResults(deviceC, result, n, dims, block, rank, ranks);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }

    int valid = 1;
    if (validate) {
        CheckValues checks{};
        if (!printResults) {
            checks = collectChecks(deviceC, block, rank);
        }
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(checks, result, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (block.elements() != 0) {
        checkCuda(cudaFree(deviceA), rank, "Freeing device input A");
        checkCuda(cudaFree(deviceB), rank, "Freeing device input B");
        checkCuda(cudaFree(deviceC), rank, "Freeing device output");
    }
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        fail(rank, "MPI does not provide MPI_THREAD_FUNNELED");
    }

    int status;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail(rank, error.what());
    }
    MPI_Finalize();
    return status;
}
