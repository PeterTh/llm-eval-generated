#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

struct Slice {
    int first;
    int count;
};

Slice partition(int n, int parts, int index) {
    const int remainder = n % parts;
    return {index * (n / parts) + std::min(index, remainder),
            n / parts + (index < remainder)};
}

struct Tile {
    Slice rows;
    Slice cols;
};

struct RankGrid {
    int ranks;
    int rowGroups;

    Tile tile(int n, int rank) const {
        // Use nearly square tiles even when the rank count has no close factors.
        const int base = ranks / rowGroups;
        const int extra = ranks % rowGroups;
        const int wideRanks = (base + 1) * extra;
        const bool wide = rank < wideRanks;
        const int columns = base + wide;
        const int column = (wide ? rank : rank - wideRanks) % columns;
        const int firstRank = rank - column;
        // Give each row group a number of rows proportional to its rank count.
        const int firstRow = static_cast<long long>(n) * firstRank / ranks;
        const int endRow = static_cast<long long>(n) * (firstRank + columns) / ranks;
        return {{firstRow, endRow - firstRow}, partition(n, columns, column)};
    }
};

double getPseudoRndValue(size_t n, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

[[noreturn]] void abortBenchmark(int rank, const char* operation, const char* error) {
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, error);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, int rank, const char* operation) {
    if (status != cudaSuccess) {
        abortBenchmark(rank, operation, cudaGetErrorString(status));
    }
}

void checkBlas(cublasStatus_t status, int rank, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char error[48];
        snprintf(error, sizeof(error), "cuBLAS error %d", static_cast<int>(status));
        abortBenchmark(rank, operation, error);
    }
}

void multiplyExact(cublasHandle_t blas, const double* a, const double* b, double* c,
                   int n, int rows, int cols, int rank) {
    const double alpha = 1.0;
    const double beta = 0.0;
    // Row-major C = A * B is column-major C^T = B^T * A^T.
    checkBlas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, n,
                         &alpha, b, cols, a, n, &beta, c, cols), rank, "cuBLAS DGEMM");
}

struct DeviceBuffer {
    double* data = nullptr;

    void allocate(size_t elements, int rank) {
        checkCuda(cudaMalloc(&data, elements * sizeof(double)), rank, "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

void initMatrix(double* data, int n, Slice rows, Slice cols) {
    const size_t elements = static_cast<size_t>(rows.count) * cols.count;
    // Small slices cost less to initialize than to start a large OpenMP team.
    const int maxThreads = std::min(omp_get_max_threads(), rows.count);
    const int threads = std::min<size_t>(maxThreads, 1 + (elements - 1) / 16384);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int row = 0; row < rows.count; ++row) {
        const size_t i = rows.first + row;
        double* out = data + static_cast<size_t>(row) * cols.count;
        for (int col = 0; col < cols.count; ++col) {
            out[col] = getPseudoRndValue(n, i, cols.first + col);
        }
    }
}

constexpr int tileRows = 32;
constexpr int tileCols = 64;
constexpr int tileDepth = 16;
constexpr int threadsX = 16;
constexpr int threadsY = 8;

// A thread accumulates a 4 x 4 register tile. Shared-memory tiles reuse each
// global load across many outputs while visiting each dot product in CPU order.
template <bool exactProducts>
__global__ void multiplyOrdered(const double* __restrict__ a, const double* __restrict__ b,
                                double* __restrict__ c, int n, int rows, int cols,
                                unsigned int columnTiles) {
    __shared__ double aTile[tileDepth][tileRows + 1]; // Pad the transposed stores.
    __shared__ double bTile[tileDepth][tileCols];

    const int rowBase = (blockIdx.x / columnTiles) * tileRows;
    const int colBase = (blockIdx.x % columnTiles) * tileCols;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int lane = y * threadsX + x;
    const int evenN = n & ~1;
    double sums[tileRows / threadsY][tileCols / threadsX] = {};

    for (size_t base = 0; base < static_cast<size_t>(evenN); base += tileDepth) {
        for (int index = lane; index < tileRows * tileDepth; index += threadsX * threadsY) {
            const int row = rowBase + index / tileDepth;
            const size_t k = base + index % tileDepth;
            aTile[index % tileDepth][index / tileDepth] =
                row < rows && k < evenN ? a[static_cast<size_t>(row) * n + k] : 0.0;
        }
        for (int index = lane; index < tileDepth * tileCols; index += threadsX * threadsY) {
            const size_t k = base + index / tileCols;
            const int col = colBase + index % tileCols;
            bTile[index / tileCols][index % tileCols] =
                k < evenN && col < cols ? b[k * cols + col] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tileDepth; ++k) {
            double bValues[tileCols / threadsX];
            #pragma unroll
            for (int j = 0; j < tileCols / threadsX; ++j) {
                bValues[j] = bTile[k][x + j * threadsX];
            }
            #pragma unroll
            for (int i = 0; i < tileRows / threadsY; ++i) {
                const double aValue = aTile[k][y + i * threadsY];
                #pragma unroll
                for (int j = 0; j < tileCols / threadsX; ++j) {
                    if constexpr (exactProducts) {
                        sums[i][j] = __fma_rn(aValue, bValues[j], sums[i][j]);
                    } else {
                        sums[i][j] = __dadd_rn(sums[i][j], __dmul_rn(aValue, bValues[j]));
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < tileRows / threadsY; ++i) {
        const int row = rowBase + y + i * threadsY;
        #pragma unroll
        for (int j = 0; j < tileCols / threadsX; ++j) {
            const int col = colBase + x + j * threadsX;
            if (row < rows && col < cols) {
                double value = sums[i][j];
                // The vectorized serial build contracts the final odd term.
                if (n != evenN) {
                    value = __fma_rn(a[static_cast<size_t>(row) * n + evenN],
                                     b[static_cast<size_t>(evenN) * cols + col], value);
                }
                c[static_cast<size_t>(row) * cols + col] = value;
            }
        }
    }
}

constexpr int validationWidth = 5;

bool validateResult(const double* samples, int n) {
    if (n == 0) {
        return true;
    }
    double expected[validationWidth * validationWidth] = {};
    const int width = std::min(n, validationWidth);
    #pragma omp parallel for collapse(2) if(n >= 1024) schedule(static) num_threads(std::min(omp_get_max_threads(), width * width))
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            }
            expected[i * validationWidth + j] = sum;
        }
    }
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            const double reference = expected[i * validationWidth + j];
            const double actual = samples[i * validationWidth + j];
            const double relError = std::abs((actual - reference) / (reference + 1e-10));
            if (!(relError <= 1e-6)) {
                printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, reference, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void collectResults(const DeviceBuffer& c, int n, Slice rows, Slice cols, const RankGrid& grid,
                    int rank, std::vector<double>& result) {
    constexpr int tag = 0;
    if (rank != 0) {
        const size_t elements = static_cast<size_t>(rows.count) * cols.count;
        if (elements == 0) {
            return;
        }
        std::unique_ptr<double[]> tile(new double[elements]);
        checkCuda(cudaMemcpy(tile.get(), c.data, elements * sizeof(double), cudaMemcpyDeviceToHost),
                  rank, "copy result to host");
        // Bound each message count by the MPI int count limit.
        for (int row = 0; row < rows.count;) {
            const int batch = std::min(rows.count - row, INT_MAX / cols.count);
            MPI_Send(tile.get() + static_cast<size_t>(row) * cols.count, batch * cols.count,
                     MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
            row += batch;
        }
        return;
    }

    result.resize(static_cast<size_t>(n) * n);
    std::vector<MPI_Request> receives;
    for (int source = 1; source < grid.ranks; ++source) {
        const auto [sourceRows, sourceCols] = grid.tile(n, source);
        for (int row = 0; sourceCols.count != 0 && row < sourceRows.count;) {
            const int batch = std::min(sourceRows.count - row, INT_MAX / sourceCols.count);
            MPI_Datatype block;
            MPI_Type_vector(batch, sourceCols.count, n, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            double* dest = result.data() + static_cast<size_t>(sourceRows.first + row) * n + sourceCols.first;
            receives.emplace_back();
            MPI_Irecv(dest, 1, block, source, tag, MPI_COMM_WORLD, &receives.back());
            MPI_Type_free(&block);
            row += batch;
        }
    }
    if (rows.count != 0 && cols.count != 0) {
        const size_t rowBytes = static_cast<size_t>(cols.count) * sizeof(double);
        double* dest = result.data() + static_cast<size_t>(rows.first) * n + cols.first;
        checkCuda(cudaMemcpy2D(dest, static_cast<size_t>(n) * sizeof(double), c.data,
                               rowBytes, rowBytes, rows.count, cudaMemcpyDeviceToHost),
                  rank, "copy root result to host");
    }
    for (MPI_Request& receive : receives) {
        MPI_Wait(&receive, MPI_STATUS_IGNORE);
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end;
            errno = 0;
            const long parsed = strtol(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX ||
                (parsed > 0 && static_cast<size_t>(parsed) >
                    std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid matrix size: %s\n", value);
                }
                return 1;
            }
            n = static_cast<int>(parsed);
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

    const RankGrid grid{ranks, static_cast<int>(std::sqrt(ranks) + 0.5)};
    const auto [rows, cols] = grid.tile(n, rank);
    const bool active = rows.count != 0 && cols.count != 0;

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
        fflush(stdout);
    }

    DeviceBuffer a, b, c;
    cublasHandle_t blas = nullptr;
    const bool powerOfTwo = n > 0 && (n & (n - 1)) == 0;
    // For powers of two up to 1024 all products and partial sums are exactly
    // representable in binary64, allowing cuBLAS to reorder the dot products.
    const bool exact = powerOfTwo && n <= 1024;
    // With up to 52 product bits, an FMA gives the same rounded sum as a
    // separate multiply and add even when the accumulation itself rounds.
    const auto orderedKernel = powerOfTwo && n <= 8192 ? multiplyOrdered<true> : multiplyOrdered<false>;
    if (active) {
        int devices;
        checkCuda(cudaGetDeviceCount(&devices), rank, "query CUDA devices");
        if (devices == 0) {
            abortBenchmark(rank, "select CUDA device", "no visible CUDA devices");
        }
        checkCuda(cudaSetDevice(localRank % devices), rank, "select CUDA device");
        const size_t aCount = static_cast<size_t>(rows.count) * n;
        const size_t bCount = static_cast<size_t>(n) * cols.count;
        a.allocate(aCount, rank);
        b.allocate(bCount, rank);
        c.allocate(static_cast<size_t>(rows.count) * cols.count, rank);
        std::unique_ptr<double[]> staging(new double[std::max(aCount, bCount)]);
        initMatrix(staging.get(), n, rows, {0, n});
        checkCuda(cudaMemcpy(a.data, staging.get(), aCount * sizeof(double), cudaMemcpyHostToDevice),
                  rank, "copy A to device");
        initMatrix(staging.get(), n, {0, n}, cols);
        checkCuda(cudaMemcpy(b.data, staging.get(), bCount * sizeof(double), cudaMemcpyHostToDevice),
                  rank, "copy B to device");
        if (exact) {
            checkBlas(cublasCreate(&blas), rank, "create cuBLAS handle");
            // Load cuBLAS's selected kernel before timing the multiply.
            multiplyExact(blas, a.data, b.data, c.data, n, rows.count, cols.count, rank);
            checkCuda(cudaDeviceSynchronize(), rank, "warm up cuBLAS");
        } else {
            cudaFuncAttributes attributes;
            checkCuda(cudaFuncGetAttributes(&attributes, orderedKernel), rank, "load CUDA matmul kernel");
        }
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        if (exact) {
            multiplyExact(blas, a.data, b.data, c.data, n, rows.count, cols.count, rank);
        } else {
            const unsigned int rowTiles = 1 + (rows.count - 1) / tileRows;
            const unsigned int columnTiles = 1 + (cols.count - 1) / tileCols;
            const size_t blocks = static_cast<size_t>(rowTiles) * columnTiles;
            if (blocks > INT_MAX) {
                abortBenchmark(rank, "launch CUDA matmul", "output tile exceeds CUDA grid limit");
            }
            orderedKernel<<<static_cast<unsigned int>(blocks), dim3(threadsX, threadsY)>>>(
                a.data, b.data, c.data, n, rows.count, cols.count, columnTiles);
            checkCuda(cudaGetLastError(), rank, "launch CUDA matmul");
        }
        checkCuda(cudaDeviceSynchronize(), rank, "finish CUDA matmul");
    }
    const double elapsed = active ? MPI_Wtime() - start : 0.0;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (blas != nullptr) {
        checkBlas(cublasDestroy(blas), rank, "destroy cuBLAS handle");
    }
    if (rank == 0) {
        const double gflops = seconds > 0.0 ? (2.0 * n * n * n) / seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", seconds * 1e3);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (printResults) {
        collectResults(c, n, rows, cols, grid, rank, result);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }

    int valid = 1;
    if (validate) {
        const int width = std::min(n, validationWidth);
        double samples[validationWidth * validationWidth] = {};
        if (printResults) {
            if (rank == 0) {
                for (int i = 0; i < width; ++i) {
                    memcpy(samples + i * validationWidth, result.data() + static_cast<size_t>(i) * n,
                           width * sizeof(double));
                }
            }
        } else {
            double localSamples[validationWidth * validationWidth] = {};
            const int sampleRows = std::min(rows.count, width - rows.first);
            const int sampleCols = std::min(cols.count, width - cols.first);
            if (sampleRows > 0 && sampleCols > 0) {
                checkCuda(cudaMemcpy2D(localSamples + rows.first * validationWidth + cols.first,
                                       validationWidth * sizeof(double), c.data, cols.count * sizeof(double),
                                       sampleCols * sizeof(double), sampleRows, cudaMemcpyDeviceToHost),
                          rank, "copy validation samples to host");
            }
            MPI_Reduce(localSamples, samples, validationWidth * validationWidth, MPI_DOUBLE,
                       MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(samples, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortBenchmark(rank, "initialize MPI", "MPI_THREAD_FUNNELED is unavailable");
    }

    int status;
    try {
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::bad_alloc&) {
        abortBenchmark(rank, "allocate host memory", "out of memory");
    }
    MPI_Finalize();
    return status;
}
