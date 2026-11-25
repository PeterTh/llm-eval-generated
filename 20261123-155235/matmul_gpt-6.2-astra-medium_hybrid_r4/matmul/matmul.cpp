#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

void fail(const char* operation, const char* reason) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char reason[64];
        std::snprintf(reason, sizeof(reason), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, reason);
    }
}

struct Range {
    int first, count;
};

Range partition(int n, int parts, int index) {
    int base = n / parts, extra = n % parts;
    return {index * base + std::min(index, extra), base + (index < extra)};
}

double value(size_t n, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

// Generate the required panel directly from global indices.
double* uploadPanel(int n, Range rows, Range cols) {
    const size_t count = size_t(rows.count) * cols.count;
    auto host = std::make_unique_for_overwrite<double[]>(count);
    const int threads = std::min<size_t>(std::min(rows.count, omp_get_max_threads()),
                                         (count + 16383) / 16384);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < rows.count; ++i)
        for (int j = 0; j < cols.count; ++j)
            host[size_t(i) * cols.count + j] = value(n, rows.first + i, cols.first + j);

    double* device;
    check(cudaMalloc(&device, count * sizeof(double)), "allocate input panel");
    check(cudaMemcpy(device, host.get(), count * sizeof(double), cudaMemcpyHostToDevice),
          "upload input panel");
    return device;
}

bool validateBlock(int n, Range rows, Range cols, const double* device) {
    const int height = std::min(rows.count, 5 - rows.first);
    const int width = std::min(cols.count, 5 - cols.first);
    if (height <= 0 || width <= 0) return true;

    double samples[5][5];
    check(cudaMemcpy2D(samples, sizeof(samples[0]), device, size_t(cols.count) * sizeof(double),
                       width * sizeof(double), height, cudaMemcpyDeviceToHost), "read validation samples");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double expected = 0.0;
            for (int k = 0; k < n; ++k)
                expected += value(n, rows.first + i, k) * value(n, k, cols.first + j);
            double error = std::abs((samples[i][j] - expected) / (expected + 1e-10));
            if (!(error <= 1e-6)) {
                std::printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                            rows.first + i, cols.first + j, expected, samples[i][j], error);
                return false;
            }
        }
    }
    return true;
}

void outputMatrix(int n, int rank, int size, const int dims[2], Range rows, Range cols,
                  const double* device) {
    if (rank == 0) {
        std::vector<double> result(size_t(n) * n);
        if (n == 0) {
            print_results(result, "MatrixC");
            return;
        }
        check(cudaMemcpy2D(result.data(), size_t(n) * sizeof(double), device,
                           size_t(cols.count) * sizeof(double), size_t(cols.count) * sizeof(double),
                           rows.count, cudaMemcpyDeviceToHost), "download root result");
        for (int peer = 1; peer < size; ++peer) {
            Range r = partition(n, dims[0], peer / dims[1]);
            Range c = partition(n, dims[1], peer % dims[1]);
            if (!r.count || !c.count) continue;
            MPI_Datatype block;
            MPI_Type_vector(r.count, c.count, n, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data() + size_t(r.first) * n + c.first, 1, block, peer, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
        print_results(result, "MatrixC");
    } else if (rows.count && cols.count) {
        std::vector<double> local(size_t(rows.count) * cols.count);
        check(cudaMemcpy(local.data(), device, local.size() * sizeof(double), cudaMemcpyDeviceToHost),
              "download result block");
        MPI_Datatype row;
        MPI_Type_contiguous(cols.count, MPI_DOUBLE, &row);
        MPI_Type_commit(&row);
        MPI_Send(local.data(), rows.count, row, 0, 0, MPI_COMM_WORLD);
        MPI_Type_free(&row);
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

int benchmark(int argc, char** argv, int rank, int size) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            const char* end = arg + strlen(arg);
            auto parsed = std::from_chars(arg, end, n);
            if (parsed.ec != std::errc{} || parsed.ptr != end || n < 0 ||
                (n && size_t(n) > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
                if (!rank) std::fprintf(stderr, "Invalid matrix size: %s\n", arg);
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

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int dims[2] = {};
    MPI_Dims_create(size, 2, dims);
    Range rows = partition(n, dims[0], rank / dims[1]);
    Range cols = partition(n, dims[1], rank % dims[1]);
    bool active = rows.count && cols.count;
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %d x %d\n", n, n);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }

    cublasHandle_t handle{};
    double *a = nullptr, *b = nullptr, *c = nullptr;
    if (active) {
        int devices;
        check(cudaGetDeviceCount(&devices), "enumerate CUDA devices");
        if (!devices) fail("select CUDA device", "no device visible");
        check(cudaSetDevice(localRank % devices), "select CUDA device");
        check(cublasCreate(&handle), "initialize cuBLAS");
        a = uploadPanel(n, rows, {0, n});
        // Both operands use the same generator, so identical panels can share storage.
        b = (rows.count == n && cols.count == n) ? a : uploadPanel(n, {0, n}, cols);
        check(cudaMalloc(&c, size_t(rows.count) * cols.count * sizeof(double)), "allocate result");
        check(cudaDeviceSynchronize(), "complete initialization");
    }
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (active) {
        // Column-major cuBLAS sees each row-major panel transposed: C^T = B^T A^T.
        const double one = 1.0, zero = 0.0;
        check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols.count, rows.count, n,
                          &one, b, cols.count, a, n, &zero, c, cols.count), "multiply panels");
        check(cudaDeviceSynchronize(), "complete matrix multiplication");
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * n * n * n / duration / 1e9);
    }
    if (printResults) outputMatrix(n, rank, size, dims, rows, cols, c);

    int valid = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        int localValid = validateBlock(n, rows, cols, c);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (active) {
        check(cudaFree(c), "free result");
        if (b != a) check(cudaFree(b), "free right panel");
        check(cudaFree(a), "free left panel");
        check(cublasDestroy(handle), "destroy cuBLAS handle");
    }
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("initialize MPI", "insufficient thread support");
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        status = benchmark(argc, argv, rank, size);
    } catch (const std::exception& e) {
        fail("benchmark", e.what());
    }
    MPI_Finalize();
    return status;
}
