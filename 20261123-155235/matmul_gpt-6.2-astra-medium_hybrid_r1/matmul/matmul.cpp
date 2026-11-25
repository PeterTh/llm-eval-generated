#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {
[[noreturn]] void fail(const char* operation, const char* reason) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, message);
    }
}

double value(size_t n, size_t i, size_t j) {
    return ((i + 1) * (i + j + 1) * 1299709 % (n * n)) / static_cast<double>(n * n);
}

struct Tile {
    int row, col, rows, cols;
};

Tile tileFor(int id, const int dims[2], int n) {
    auto boundary = [n](int i, int parts) {
        return static_cast<int>(static_cast<long long>(n) * i / parts);
    };
    int i = id / dims[1], j = id % dims[1];
    int row = boundary(i, dims[0]), col = boundary(j, dims[1]);
    return {row, col, boundary(i + 1, dims[0]) - row, boundary(j + 1, dims[1]) - col};
}

double* allocate(size_t elements) {
    void* ptr;
    check(cudaMalloc(&ptr, elements * sizeof(double)), "cudaMalloc");
    return static_cast<double*>(ptr);
}

struct Worker {
    Tile tile{};
    int device{};
    double *a{}, *b{}, *c{};
    cublasHandle_t handle{};
};

void initPanel(double* dest, size_t n, int row, int col, int rows, int cols) {
    size_t elements = static_cast<size_t>(rows) * cols;
    auto panel = std::make_unique_for_overwrite<double[]>(elements);
    int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + elements / 32768));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            panel[static_cast<size_t>(i) * cols + j] = value(n, row + i, col + j);
    check(cudaMemcpy(dest, panel.get(), elements * sizeof(double), cudaMemcpyHostToDevice),
          "upload input panel");
}

bool validateTile(const Worker& w, int n) {
    Tile t = w.tile;
    int rows = std::min(t.rows, 5 - t.row), cols = std::min(t.cols, 5 - t.col);
    if (rows <= 0 || cols <= 0) return true;
    double sample[5][5];
    check(cudaMemcpy2D(sample, sizeof(sample[0]), w.c, size_t(t.cols) * sizeof(double),
                       cols * sizeof(double), rows, cudaMemcpyDeviceToHost), "read validation sample");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double expected = 0;
            for (int k = 0; k < n; ++k)
                expected += value(n, t.row + i, k) * value(n, k, t.col + j);
            double error = std::abs((sample[i][j] - expected) / (expected + 1e-10));
            if (!(error <= 1e-6)) {
                printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                       t.row + i, t.col + j, expected, sample[i][j], error);
                return false;
            }
        }
    }
    return true;
}

void collectResults(const std::vector<Worker>& workers, const std::vector<int>& offsets,
                    const int dims[2], int n, int rank) {
    std::vector<double> result;
    if (rank == 0) result.resize(size_t(n) * n);
    for (int owner = 0; owner + 1 < static_cast<int>(offsets.size()); ++owner) {
        if (rank != 0 && rank != owner) continue;
        for (int id = offsets[owner]; id < offsets[owner + 1]; ++id) {
            Tile t = tileFor(id, dims, n);
            if (!t.rows || !t.cols) continue;
            size_t width = size_t(t.cols) * sizeof(double);
            if (owner == rank) {
                const Worker& w = workers[id - offsets[owner]];
                check(cudaSetDevice(w.device), "select result device");
                if (rank == 0) {
                    check(cudaMemcpy2D(result.data() + size_t(t.row) * n + t.col,
                                       size_t(n) * sizeof(double), w.c, width, width, t.rows,
                                       cudaMemcpyDeviceToHost), "read result tile");
                } else {
                    std::vector<double> tile(size_t(t.rows) * t.cols);
                    check(cudaMemcpy(tile.data(), w.c, tile.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost), "read result tile");
                    MPI_Datatype row;
                    MPI_Type_contiguous(t.cols, MPI_DOUBLE, &row);
                    MPI_Type_commit(&row);
                    MPI_Send(tile.data(), t.rows, row, 0, 0, MPI_COMM_WORLD);
                    MPI_Type_free(&row);
                }
            } else {
                MPI_Datatype block;
                MPI_Type_vector(t.rows, t.cols, n, MPI_DOUBLE, &block);
                MPI_Type_commit(&block);
                MPI_Recv(result.data() + size_t(t.row) * n + t.col, 1, block, owner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&block);
            }
        }
    }
    if (rank == 0) print_results(result, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            long parsed = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || parsed < 0 || parsed > INT_MAX ||
                (parsed && size_t(parsed) > std::numeric_limits<size_t>::max() / sizeof(double) / size_t(parsed))) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
        printf("Matrix Multiplication Benchmark\nMatrix size: %d x %d\nValidation: %s\n",
               n, n, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (!devices) fail("device selection", "no visible CUDA devices");
    // Stripe the visible GPUs among node-local ranks. A scheduler may expose only one.
    int count = std::max(1, devices / localRanks + (localRank < devices % localRanks));
    std::vector<int> counts(ranks), offsets(ranks + 1);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < ranks; ++i) offsets[i + 1] = offsets[i] + counts[i];
    int dims[2]{};
    MPI_Dims_create(offsets.back(), 2, dims);
    std::vector<Worker> workers(count);
    for (int i = 0; i < count; ++i) {
        Worker& w = workers[i];
        Tile t = w.tile = tileFor(offsets[rank] + i, dims, n);
        if (!t.rows || !t.cols) continue;
        w.device = (localRank + i * localRanks) % devices;
        check(cudaSetDevice(w.device), "select compute device");
        check(cublasCreate(&w.handle), "create cuBLAS handle");
        w.a = allocate(size_t(t.rows) * n);
        w.b = allocate(size_t(n) * t.cols);
        w.c = allocate(size_t(t.rows) * t.cols);
        initPanel(w.a, n, t.row, 0, t.rows, n);
        initPanel(w.b, n, 0, t.col, n, t.cols);
        check(cudaDeviceSynchronize(), "finish panel initialization");
    }
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    constexpr double one = 1.0, zero = 0.0;
    for (const Worker& w : workers) {
        if (!w.handle) continue;
        check(cudaSetDevice(w.device), "select compute device");
        Tile t = w.tile;
        // Row-major C = A B is column-major C^T = B^T A^T in the same storage.
        check(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N, t.cols, t.rows, n,
                          &one, w.b, t.cols, w.a, n, &zero, w.c, t.cols), "DGEMM");
    }
    for (const Worker& w : workers) {
        if (!w.handle) continue;
        check(cudaSetDevice(w.device), "select compute device");
        check(cudaDeviceSynchronize(), "finish DGEMM");
    }
    double elapsed = MPI_Wtime() - start, slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", slowest * 1000);
        printf("Performance: %.3f GFLOPS\n", slowest > 0 ? 2.0 * n * n * n / slowest / 1e9 : 0);
    }
    if (printResults) collectResults(workers, offsets, dims, n, rank);
    int valid = 1;
    if (validate && !rank) printf("Validating result...\n");
    for (const Worker& w : workers) {
        if (!w.handle) continue;
        check(cudaSetDevice(w.device), "select compute device");
        if (validate && !validateTile(w, n)) valid = 0;
        check(cublasDestroy(w.handle), "destroy cuBLAS handle");
        check(cudaFree(w.a), "free A");
        check(cudaFree(w.b), "free B");
        check(cudaFree(w.c), "free C");
    }
    if (validate) {
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support", "MPI_THREAD_FUNNELED required");
    int status;
    try {
        status = run(argc, argv);
    } catch (const std::exception& e) {
        fail("benchmark", e.what());
    }
    MPI_Finalize();
    return status;
}
