#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static void fail(const char *operation, const char *reason) {
    std::fprintf(stderr, "%s failed: %s\n", operation, reason);
    std::abort();
}

static void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        fail(operation, cudaGetErrorString(status));
}

__host__ __device__ static double value(size_t n, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

struct Span {
    int first, count;
};

static Span partition(int total, int parts, int index) {
    const int base = total / parts, extra = total % parts;
    return {index * base + std::min(index, extra), base + (index < extra)};
}

__global__ void initialize(double *matrix, int n, Span rows, Span cols) {
    const size_t count = size_t(rows.count) * cols.count;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < count;
         p += size_t(gridDim.x) * blockDim.x) {
        matrix[p] = value(n, rows.first + p / cols.count, cols.first + p % cols.count);
    }
}

static double *allocate(size_t count) {
    double *ptr;
    check(cudaMalloc(&ptr, count * sizeof(double)), "cudaMalloc");
    return ptr;
}

static double *input(int n, Span rows, Span cols) {
    const size_t count = size_t(rows.count) * cols.count;
    double *ptr = allocate(count);
    initialize<<<std::min<size_t>(65535, (count + 255) / 256), 256>>>(ptr, n, rows, cols);
    check(cudaGetLastError(), "matrix initialization");
    return ptr;
}

// Each thread retains a square of output values in registers and consumes
// shared input panels in increasing k order.
template <int block, bool exactProducts>
__global__ void multiply(const double *a, const double *b, double *c, int m, int n, int width) {
    constexpr int extent = block / 16;
    __shared__ double lhs[16][block + 1], rhs[16][block];
    const int x = threadIdx.x, y = threadIdx.y, lane = 16 * y + x;
    const size_t col = size_t(blockIdx.x) * block, row = size_t(blockIdx.y) * block;
    double acc[extent][extent] = {};
    const int paired = n - (n % 2);
    for (size_t k = 0; k < paired; k += 16) {
#pragma unroll
        for (int t = 0; t < extent; ++t) {
            const int p = lane + t * 256;
            lhs[p % 16][p / 16] =
                row + p / 16 < m && k + p % 16 < paired ? a[(row + p / 16) * n + k + p % 16] : 0.0;
            rhs[p / block][p % block] = k + p / block < paired && col + p % block < width
                                            ? b[(k + p / block) * width + col + p % block]
                                            : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 16; ++t) {
#pragma unroll
            for (int i = 0; i < extent; ++i) {
#pragma unroll
                for (int j = 0; j < extent; ++j) {
                    double aij = lhs[t][y + 16 * i], bij = rhs[t][x + 16 * j];
                    if constexpr (exactProducts)
                        acc[i][j] = __fma_rn(aij, bij, acc[i][j]);
                    else
                        acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(aij, bij));
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < extent; ++i) {
        for (int j = 0; j < extent; ++j) {
            const size_t r = row + y + 16 * i, s = col + x + 16 * j;
            if (r < m && s < width) {
                double sum = acc[i][j];
                // The native vectorized reference contracts its odd scalar tail.
                if (n % 2)
                    sum = __fma_rn(a[r * n + paired], b[size_t(paired) * width + s], sum);
                c[r * width + s] = sum;
            }
        }
    }
}

void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

struct DeviceTile {
    int device;
    Span rows;
    double *a, *b, *c;
};

template <int block> static void launch(const DeviceTile &tile, int n, int width, bool exactProducts) {
    dim3 grid((unsigned(width) + block - 1) / block, (unsigned(tile.rows.count) + block - 1) / block);
    if (exactProducts)
        multiply<block, true><<<grid, dim3(16, 16)>>>(tile.a, tile.b, tile.c, tile.rows.count, n, width);
    else
        multiply<block, false>
            <<<grid, dim3(16, 16)>>>(tile.a, tile.b, tile.c, tile.rows.count, n, width);
}

// Each sender holds a packed rectangle. Receive into its final row-major
// location, splitting large messages to stay within MPI's int count limit.
static void report(const std::vector<double> &packed, int n, int rank, int ranks, const int dims[2]) {
    std::vector<double> result(rank == 0 ? size_t(n) * n : 0);
    const int end = rank == 0 ? ranks : rank + 1;
    for (int source = rank; source < end; ++source) {
        Span r = partition(n, dims[0], source / dims[1]);
        Span c = partition(n, dims[1], source % dims[1]);
        if (c.count > 0) {
            for (int offset = 0; offset < r.count;) {
                int rows = std::min(r.count - offset, INT_MAX / c.count);
                size_t pos = size_t(offset) * c.count;
                if (rank != 0) {
                    MPI_Send(packed.data() + pos, rows * c.count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                } else {
                    double *dest = result.data() + size_t(r.first + offset) * n + c.first;
                    if (source == 0) {
                        for (int i = 0; i < rows; ++i)
                            std::copy_n(packed.data() + pos + size_t(i) * c.count, c.count,
                                        dest + size_t(i) * n);
                    } else {
                        MPI_Datatype rect;
                        MPI_Type_vector(rows, c.count, n, MPI_DOUBLE, &rect);
                        MPI_Type_commit(&rect);
                        MPI_Recv(dest, 1, rect, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Type_free(&rect);
                    }
                }
                offset += rows;
            }
        }
    }
    if (rank == 0)
        print_results(result, "MatrixC");
}

int main(int argc, char **argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char *end;
            errno = 0;
            long parsed = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || parsed < 0 || parsed > INT_MAX ||
                double(parsed) * parsed > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (!rank)
                    fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<int>(parsed);
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "-r")) {
            printResults = true;
        } else {
            bool help = !strcmp(argv[i], "-h");
            if (!rank) {
                if (!help)
                    printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localSize, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localSize);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices), "CUDA device discovery");
    if (devices == 0)
        fail("CUDA device discovery", "no visible GPU");

    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    Span rows = partition(n, dims[0], rank / dims[1]);
    Span cols = partition(n, dims[1], rank % dims[1]);
    const int firstDevice = localRank % devices;
    const int workers = std::min(rows.count, 1 + (devices - 1 - firstDevice) / localSize);
    std::vector<DeviceTile> tiles(cols.count ? workers : 0);
    // For powers of two up to 8192 the input products fit in the FP64
    // significand, so fusing the multiply and add preserves every result bit.
    const bool exactProducts = n > 0 && (n & (n - 1)) == 0 && n <= 8192;
    const int threads = std::max(1, std::min(workers, omp_get_max_threads()));
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
#pragma omp parallel for num_threads(threads)
    for (int w = 0; w < int(tiles.size()); ++w) {
        auto &t = tiles[w];
        t.device = firstDevice + w * localSize;
        check(cudaSetDevice(t.device), "select GPU");
        t.rows = partition(rows.count, workers, w);
        t.rows.first += rows.first;
        t.a = input(n, t.rows, {0, n});
        t.b = t.rows.count == n && cols.count == n ? t.a : input(n, {0, n}, cols);
        t.c = allocate(size_t(t.rows.count) * cols.count);
        check(cudaDeviceSynchronize(), "initialize inputs");
    }

    if (!rank)
        printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#pragma omp parallel for num_threads(threads)
    for (int w = 0; w < int(tiles.size()); ++w) {
        auto &t = tiles[w];
        check(cudaSetDevice(t.device), "select GPU");
        if (size_t(t.rows.count) * cols.count <= 512 * 1024)
            launch<32>(t, n, cols.count, exactProducts);
        else
            launch<64>(t, n, cols.count, exactProducts);
        check(cudaGetLastError(), "launch multiplication");
    }
#pragma omp parallel for num_threads(threads)
    for (int w = 0; w < int(tiles.size()); ++w) {
        check(cudaSetDevice(tiles[w].device), "select GPU");
        check(cudaDeviceSynchronize(), "complete multiplication");
    }
    double localTime = MPI_Wtime() - start, seconds;
    MPI_Reduce(&localTime, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * n * n * n / seconds / 1e9 : 0.0);
    }

    std::vector<double> packed(printResults ? size_t(rows.count) * cols.count : 0);
    int valid = 1;
#pragma omp parallel for num_threads(threads) reduction(& : valid)
    for (int w = 0; w < int(tiles.size()); ++w) {
        auto &t = tiles[w];
        check(cudaSetDevice(t.device), "select GPU");
        if (printResults)
            check(cudaMemcpy(packed.data() + size_t(t.rows.first - rows.first) * cols.count, t.c,
                             size_t(t.rows.count) * cols.count * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy output tile");
        if (validate) {
            int checks = std::min({5 - cols.first, cols.count, n});
            for (int i = t.rows.first; i < std::min(5, t.rows.first + t.rows.count) && checks > 0; ++i) {
                double sample[5];
                check(cudaMemcpy(sample, t.c + size_t(i - t.rows.first) * cols.count,
                                 checks * sizeof(double), cudaMemcpyDeviceToHost),
                      "copy validation samples");
                for (int j = 0; j < checks; ++j) {
                    double expected = 0;
                    for (int k = 0; k < n; ++k)
                        expected += value(n, i, k) * value(n, k, cols.first + j);
                    if (!(std::abs((sample[j] - expected) / (expected + 1e-10)) <= 1e-6)) {
                        fprintf(stderr, "Validation failed at (%d, %d): expected %.10f, got %.10f\n", i,
                                cols.first + j, expected, sample[j]);
                        valid = 0;
                    }
                }
            }
        }
        if (t.a != t.b)
            check(cudaFree(t.b), "free right input");
        check(cudaFree(t.a), "free left input");
        check(cudaFree(t.c), "free output");
    }
    if (printResults)
        report(packed, n, rank, ranks, dims);
    if (validate) {
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank)
            printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
