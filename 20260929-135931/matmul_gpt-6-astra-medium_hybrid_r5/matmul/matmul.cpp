#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void fail(const char* message) {
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error %d\n", static_cast<int>(status));
        fail("GPU matrix multiplication failed");
    }
}

static size_t boundary(size_t n, int part, int parts) {
    return n / parts * part + std::min(n % parts, static_cast<size_t>(part));
}

struct Worker {
    int device;
    size_t first, rows;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

static void prepare(Worker& w, size_t N, size_t firstColumn, size_t columns, bool printResults) {
    if (!w.rows || !columns) return;
    cudaCheck(cudaSetDevice(w.device));
    cudaCheck(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    blasCheck(cublasCreate(&w.handle));
    blasCheck(cublasSetStream(w.handle, w.stream));
    // Retain full FP64 arithmetic, including on tensor-core equipped devices.
    blasCheck(cublasSetMathMode(w.handle, CUBLAS_DEFAULT_MATH));
    const size_t as = w.rows * N * sizeof(double);
    const size_t bs = N * columns * sizeof(double);
    const size_t cs = w.rows * columns * sizeof(double);
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&w.a), as));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&w.b), bs));
    if (printResults) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&w.c), cs));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.da), as));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.db), bs));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.dc), cs));
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t k = 0; k < N; ++k)
            w.a[i * N + k] = getPseudoRndValue(N, w.first + i, k);
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < columns; ++j)
            w.b[k * columns + j] = getPseudoRndValue(N, k, firstColumn + j);
    cudaCheck(cudaMemcpyAsync(w.da, w.a, as, cudaMemcpyHostToDevice, w.stream));
    cudaCheck(cudaMemcpyAsync(w.db, w.b, bs, cudaMemcpyHostToDevice, w.stream));
    cudaCheck(cudaStreamSynchronize(w.stream));
    cudaCheck(cudaFreeHost(w.a));
    cudaCheck(cudaFreeHost(w.b));
    w.a = w.b = nullptr;
}

static void multiply(Worker& w, size_t N, size_t columns) {
    if (!w.rows || !columns) return;
    cudaCheck(cudaSetDevice(w.device));
    const double alpha = 1.0, beta = 0.0;
    // Row-major C = A B is column-major C^T = B^T A^T.
    blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         static_cast<int>(columns), static_cast<int>(w.rows),
                         static_cast<int>(N), &alpha, w.db, static_cast<int>(columns),
                         w.da, static_cast<int>(N), &beta, w.dc, static_cast<int>(columns)));
    cudaCheck(cudaStreamSynchronize(w.stream));
}

static void release(Worker& w) {
    if (!w.handle) return;
    cudaCheck(cudaSetDevice(w.device));
    blasCheck(cublasDestroy(w.handle));
    cudaCheck(cudaFree(w.da));
    cudaCheck(cudaFree(w.db));
    cudaCheck(cudaFree(w.dc));
    if (w.c) cudaCheck(cudaFreeHost(w.c));
    cudaCheck(cudaStreamDestroy(w.stream));
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || !parsed ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, localRanks, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("The hybrid benchmark requires a CUDA GPU on every MPI rank");
    std::vector<int> assigned;
    // Works with one rank per node, one rank per GPU, or rank-local GPU visibility.
    for (int d = localRank; d < devices; d += localRanks) assigned.push_back(d);
    if (assigned.empty()) assigned.push_back(localRank % devices);
    MPI_Comm_free(&local);

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int rowPart = rank / dims[1], colPart = rank % dims[1];
    const size_t firstRow = boundary(N, rowPart, dims[0]);
    const size_t rows = boundary(N, rowPart + 1, dims[0]) - firstRow;
    const size_t firstColumn = boundary(N, colPart, dims[1]);
    const size_t columns = boundary(N, colPart + 1, dims[1]) - firstColumn;
    const int workerCount = static_cast<int>(assigned.size());
    std::vector<Worker> workers(workerCount);
    omp_set_dynamic(0);
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    #pragma omp parallel for num_threads(workerCount) schedule(static, 1)
    for (int t = 0; t < workerCount; ++t) {
        Worker& w = workers[t];
        w.device = assigned[t];
        w.first = firstRow + boundary(rows, t, workerCount);
        w.rows = boundary(rows, t + 1, workerCount) - boundary(rows, t, workerCount);
        prepare(w, N, firstColumn, columns, printResults);
    }
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workerCount) schedule(static, 1)
    for (int t = 0; t < workerCount; ++t) multiply(workers[t], N, columns);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    std::vector<double> block(printResults ? rows * columns : 0);
    int valid = 1;
    #pragma omp parallel for num_threads(workerCount) schedule(static, 1) reduction(&:valid)
    for (int t = 0; t < workerCount; ++t) {
        Worker& w = workers[t];
        if (!w.rows || !columns) continue;
        cudaCheck(cudaSetDevice(w.device));
        if (printResults) {
            cudaCheck(cudaMemcpyAsync(w.c, w.dc, w.rows * columns * sizeof(double),
                                      cudaMemcpyDeviceToHost, w.stream));
            cudaCheck(cudaStreamSynchronize(w.stream));
            std::copy(w.c, w.c + w.rows * columns,
                      block.data() + (w.first - firstRow) * columns);
        }
        if (validate) {
            for (size_t i = w.first; i < std::min(w.first + w.rows, std::min(N, size_t(5))); ++i) {
                for (size_t j = firstColumn; j < std::min(firstColumn + columns, std::min(N, size_t(5))); ++j) {
                    double actual, expected = 0;
                    const size_t index = (i - w.first) * columns + j - firstColumn;
                    if (printResults) actual = w.c[index];
                    else cudaCheck(cudaMemcpy(&actual, w.dc + index, sizeof(double), cudaMemcpyDeviceToHost));
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(actual) || error > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, error);
                        valid = 0;
                    }
                }
            }
        }
        release(w);
    }

    if (printResults) {
        // Chunked messages avoid the MPI int-count limit. Only output mode needs
        // the full matrix, assembled in the original row-major order on rank zero.
        std::vector<double> C(rank == 0 ? N * N : 0);
        for (int source = 0; source < ranks; ++source) {
            const size_t r0 = boundary(N, source / dims[1], dims[0]);
            const size_t nr = boundary(N, source / dims[1] + 1, dims[0]) - r0;
            const size_t c0 = boundary(N, source % dims[1], dims[1]);
            const size_t nc = boundary(N, source % dims[1] + 1, dims[1]) - c0;
            if (rank != 0 && rank != source) continue;
            std::vector<double> received(rank == 0 && source != 0 ? nr * nc : 0);
            double* data = rank == 0 && source != 0 ? received.data() : block.data();
            if (source != 0) {
                for (size_t offset = 0; offset < nr * nc;) {
                    const int count = static_cast<int>(std::min(nr * nc - offset, size_t(INT_MAX)));
                    if (!rank) MPI_Recv(data + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    else MPI_Send(data + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    offset += count;
                }
            }
            if (!rank && nc) {
                #pragma omp parallel for num_threads(workerCount) schedule(static)
                for (size_t i = 0; i < nr; ++i)
                    std::copy(data + i * nc, data + (i + 1) * nc, C.data() + (r0 + i) * N + c0);
            }
        }
        if (!rank) print_results(C, "MatrixC");
    }
    int allValid = 1;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate && !rank) printf("Validating result...\nValidation: %s\n", allValid ? "PASSED" : "FAILED");
    return allValid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& error) { fail(error.what()); }
    MPI_Finalize();
    return result;
}
