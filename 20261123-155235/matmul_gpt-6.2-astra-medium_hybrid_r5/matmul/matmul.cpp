#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    int start, count;
};

Range partition(int n, int parts, int index) {
    const int base = n / parts;
    const int extra = n % parts;
    return {base * index + std::min(index, extra), base + (index < extra)};
}

struct Tile {
    Range rows, cols;
    size_t elements() const { return size_t(rows.count) * cols.count; }
};

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %d\n", int(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double* deviceBuffer(size_t count) {
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, count * sizeof(double)));
    return static_cast<double*>(ptr);
}

void initPanel(double* dst, int n, Range rows, Range cols) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows.count; ++i) {
        for (int j = 0; j < cols.count; ++j) {
            dst[size_t(i) * cols.count + j] = getPseudoRndValue(n, rows.start + i, cols.start + j);
        }
    }
}

bool validateTile(const double* device, int n, Tile tile) {
    const int rows = std::min(tile.rows.count, 5 - tile.rows.start);
    const int cols = std::min(tile.cols.count, 5 - tile.cols.start);
    if (rows <= 0 || cols <= 0) return true;
    double sample[5][5];
    check(cudaMemcpy2D(sample, sizeof(sample[0]), device, size_t(tile.cols.count) * sizeof(double),
                       cols * sizeof(double), rows, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double expected = 0;
            for (int k = 0; k < n; ++k) {
                expected += getPseudoRndValue(n, tile.rows.start + i, k)
                          * getPseudoRndValue(n, k, tile.cols.start + j);
            }
            const double error = std::abs((sample[i][j] - expected) / (expected + 1e-10));
            if (!(error <= 1e-6)) {
                std::printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                            tile.rows.start + i, tile.cols.start + j, expected, sample[i][j], error);
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

int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            const char* end = value + strlen(value);
            auto parsed = std::from_chars(value, end, n);
            if (parsed.ec != std::errc{} || parsed.ptr != end || n < 0 ||
                (n > 0 && size_t(n) > SIZE_MAX / sizeof(double) / n)) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(localRank % devices));

    int grid[2] = {};
    MPI_Dims_create(ranks, 2, grid);
    auto tileFor = [&](int r) -> Tile {
        return {partition(n, grid[0], r / grid[1]), partition(n, grid[1], r % grid[1])};
    };
    const Tile tile = tileFor(rank);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }

    double *a = nullptr, *b = nullptr, *c = nullptr;
    cublasHandle_t handle = nullptr;
    if (tile.elements()) {
        check(cublasCreate(&handle));
        const size_t aSize = size_t(tile.rows.count) * n;
        const size_t bSize = size_t(tile.cols.count) * n;
        b = deviceBuffer(bSize);
        c = deviceBuffer(tile.elements());
        auto panel = std::make_unique_for_overwrite<double[]>(std::max(aSize, bSize));
        initPanel(panel.get(), n, {0, n}, tile.cols);
        check(cudaMemcpy(b, panel.get(), bSize * sizeof(double), cudaMemcpyHostToDevice));
        // The benchmark initializes A and B identically. A full-width B panel
        // already contains every row of A needed by a row-slab decomposition.
        if (tile.cols.count == n) {
            a = b + size_t(tile.rows.start) * n;
        } else {
            a = deviceBuffer(aSize);
            initPanel(panel.get(), n, tile.rows, {0, n});
            check(cudaMemcpy(a, panel.get(), aSize * sizeof(double), cudaMemcpyHostToDevice));
        }
        check(cudaDeviceSynchronize());
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (tile.elements()) {
        // Column-major cuBLAS sees the row-major buffers as C^T = B^T A^T.
        const double one = 1, zero = 0;
        check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, tile.cols.count, tile.rows.count,
                          n, &one, b, tile.cols.count, a, n, &zero, c, tile.cols.count));
        check(cudaDeviceSynchronize());
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", 1000 * seconds);
        printf("Performance: %.3f GFLOPS\n", 2.0 * n * n * n / seconds / 1e9);
    }

    if (printResults && n == 0 && rank == 0) print_results(std::vector<double>{}, "MatrixC");
    if (printResults && tile.elements()) {
        if (rank == 0) {
            std::vector<double> result(size_t(n) * n);
            check(cudaMemcpy2D(result.data(), size_t(n) * sizeof(double), c,
                               size_t(tile.cols.count) * sizeof(double),
                               size_t(tile.cols.count) * sizeof(double), tile.rows.count,
                               cudaMemcpyDeviceToHost));
            for (int r = 1; r < ranks; ++r) {
                const Tile other = tileFor(r);
                if (!other.elements()) continue;
                MPI_Datatype block;
                MPI_Type_vector(other.rows.count, other.cols.count, n, MPI_DOUBLE, &block);
                MPI_Type_commit(&block);
                MPI_Recv(result.data() + size_t(other.rows.start) * n + other.cols.start,
                         1, block, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&block);
            }
            print_results(result, "MatrixC");
        } else {
            auto packed = std::make_unique_for_overwrite<double[]>(tile.elements());
            check(cudaMemcpy(packed.get(), c, tile.elements() * sizeof(double), cudaMemcpyDeviceToHost));
            MPI_Datatype row;
            MPI_Type_contiguous(tile.cols.count, MPI_DOUBLE, &row);
            MPI_Type_commit(&row);
            MPI_Send(packed.get(), tile.rows.count, row, 0, 0, MPI_COMM_WORLD);
            MPI_Type_free(&row);
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = validateTile(c, n, tile);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (handle) {
        if (tile.cols.count != n) check(cudaFree(a));
        check(cudaFree(b));
        check(cudaFree(c));
        check(cublasDestroy(handle));
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, ex.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
