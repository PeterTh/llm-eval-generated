#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void fail(const char* operation, const char* detail) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char detail[40];
        snprintf(detail, sizeof(detail), "cuBLAS error %d", static_cast<int>(status));
        fail(operation, detail);
    }
}

struct Slice {
    int start, count;
};

Slice partition(int n, int parts, int index) {
    const int base = n / parts;
    const int extra = n % parts;
    return {index * base + std::min(index, extra), base + (index < extra)};
}

struct Tile {
    Slice rows, cols;
};

Tile tileFor(int n, const int dims[2], int rank) {
    return {partition(n, dims[0], rank / dims[1]),
            partition(n, dims[1], rank % dims[1])};
}

double value(size_t n, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

void initialize(int n, Tile tile, double* a, double* b) {
    const size_t work = size_t(n) * (size_t(tile.rows.count) + tile.cols.count);
    const int threads = std::min<size_t>(omp_get_max_threads(), 1 + work / 16384);
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < tile.rows.count; ++i)
            for (int k = 0; k < n; ++k)
                a[size_t(i) * n + k] = value(n, tile.rows.start + i, k);

        #pragma omp for schedule(static)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < tile.cols.count; ++j)
                b[size_t(k) * tile.cols.count + j] = value(n, k, tile.cols.start + j);
    }
}

// Receive directly into the global row-major layout. Derived datatypes also
// keep the MPI message count representable for tiles larger than INT_MAX.
void gather(int n, int rank, int ranks, const int dims[2], Tile tile,
            const std::vector<double>& local, std::vector<double>& result) {
    if (rank == 0) result.resize(size_t(n) * n);
    if (rank == 0) {
        for (int src = 0; src < ranks; ++src) {
            Tile t = tileFor(n, dims, src);
            if (!t.rows.count || !t.cols.count) continue;
            double* dst = result.data() + size_t(t.rows.start) * n + t.cols.start;
            if (src == 0) {
                for (int i = 0; i < t.rows.count; ++i)
                    std::copy_n(local.data() + size_t(i) * t.cols.count, t.cols.count,
                                dst + size_t(i) * n);
            } else {
                MPI_Datatype layout;
                MPI_Type_vector(t.rows.count, t.cols.count, n, MPI_DOUBLE, &layout);
                MPI_Type_commit(&layout);
                MPI_Recv(dst, 1, layout, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&layout);
            }
        }
    } else if (!local.empty()) {
        MPI_Datatype layout;
        MPI_Type_vector(tile.rows.count, tile.cols.count, tile.cols.count, MPI_DOUBLE, &layout);
        MPI_Type_commit(&layout);
        MPI_Send(local.data(), 1, layout, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&layout);
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

int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            const char* end = arg + strlen(arg);
            auto parsed = std::from_chars(arg, end, n);
            if (parsed.ec != std::errc{} || parsed.ptr != end || n < 0 ||
                size_t(n) * n > SIZE_MAX / sizeof(double)) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices), "count CUDA devices");
    if (!devices) fail("select CUDA device", "no visible GPUs");
    check(cudaSetDevice(localRank % devices), "select CUDA device");

    int dims[2] = {};
    MPI_Dims_create(ranks, 2, dims);
    const Tile tile = tileFor(n, dims, rank);
    const int rows = tile.rows.count, cols = tile.cols.count;
    cublasHandle_t handle{};
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    if (rows && cols) {
        const size_t na = size_t(rows) * n, nb = size_t(n) * cols;
        std::unique_ptr<double[]> a(new double[na]), b(new double[nb]);
        initialize(n, tile, a.get(), b.get());
        check(cublasCreate(&handle), "initialize cuBLAS");
        check(cudaMalloc(&da, na * sizeof(double)), "allocate A");
        check(cudaMalloc(&db, nb * sizeof(double)), "allocate B");
        check(cudaMalloc(&dc, size_t(rows) * cols * sizeof(double)), "allocate C");
        check(cudaMemcpy(da, a.get(), na * sizeof(double), cudaMemcpyHostToDevice), "upload A");
        check(cudaMemcpy(db, b.get(), nb * sizeof(double), cudaMemcpyHostToDevice), "upload B");
        check(cudaDeviceSynchronize(), "finish initialization");
    }

    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows && cols) {
        // Row-major C = A B is column-major C^T = B^T A^T.
        const double one = 1.0, zero = 0.0;
        check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, n,
                          &one, db, cols, da, n, &zero, dc, cols), "multiply matrices");
        check(cudaDeviceSynchronize(), "finish matrix multiplication");
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2e-9 * n * n * n / seconds : 0);
    }

    const int checkRows = std::clamp(5 - tile.rows.start, 0, rows);
    const int checkCols = std::clamp(5 - tile.cols.start, 0, cols);
    const int copyRows = printResults ? rows : (validate ? checkRows : 0);
    const int copyCols = printResults ? cols : (validate ? checkCols : 0);
    std::vector<double> local(size_t(copyRows) * copyCols);
    if (!local.empty()) {
        const size_t pitch = size_t(copyCols) * sizeof(double);
        check(cudaMemcpy2D(local.data(), pitch, dc, size_t(cols) * sizeof(double),
                           pitch, copyRows, cudaMemcpyDeviceToHost), "download results");
    }
    if (handle) {
        check(cudaFree(dc), "free C");
        check(cudaFree(db), "free B");
        check(cudaFree(da), "free A");
        check(cublasDestroy(handle), "destroy cuBLAS handle");
    }
    if (printResults) {
        std::vector<double> result;
        gather(n, rank, ranks, dims, tile, local, result);
        if (!rank) print_results(result, "MatrixC");
    }
    if (validate) {
        if (!rank) printf("Validating result...\n");
        int valid = 1;
        for (int i = 0; i < checkRows; ++i) {
            for (int j = 0; j < checkCols; ++j) {
                const int row = tile.rows.start + i, col = tile.cols.start + j;
                double expected = 0;
                for (int k = 0; k < n; ++k)
                    expected += value(n, row, k) * value(n, k, col);
                const double actual = local[size_t(i) * copyCols + j];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!(error <= 1e-6)) {
                    printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                           row, col, expected, actual, error);
                    valid = 0;
                }
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("initialize MPI", "MPI_THREAD_FUNNELED required");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail("matmul", error.what());
    }
    MPI_Finalize();
    return status;
}
