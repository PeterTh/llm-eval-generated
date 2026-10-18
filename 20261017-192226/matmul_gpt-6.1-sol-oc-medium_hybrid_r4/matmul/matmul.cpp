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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Balanced contiguous partitions, including empty blocks when ranks exceed N.
static size_t blockStart(size_t n, int coordinate, int parts) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

// Each rank owns a rectangle of C and the corresponding full-K slabs of A/B.
// Deterministic local generation removes input distribution and avoids storing
// full input matrices on every rank (O(N^2 / sqrt(P)) memory per rank).
struct Matrices {
    double *a = nullptr, *b = nullptr, *c = nullptr;
    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;

    Matrices(size_t m, size_t n, size_t k) {
        if (!m || !n) return;
        checkCuda(cudaMallocHost(&a, m * k * sizeof(double)));
        checkCuda(cudaMallocHost(&b, k * n * sizeof(double)));
        checkCuda(cudaMallocHost(&c, m * n * sizeof(double)));
        checkCuda(cudaMalloc(&da, m * k * sizeof(double)));
        checkCuda(cudaMalloc(&db, k * n * sizeof(double)));
        checkCuda(cudaMalloc(&dc, m * n * sizeof(double)));
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkBlas(cublasCreate(&blas));
        checkBlas(cublasSetStream(blas, stream));
        checkBlas(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }

    ~Matrices() {
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
        if (da) cudaFree(da);
        if (db) cudaFree(db);
        if (dc) cudaFree(dc);
        if (a) cudaFreeHost(a);
        if (b) cudaFreeHost(b);
        if (c) cudaFreeHost(c);
    }
    Matrices(const Matrices&) = delete;
    Matrices& operator=(const Matrices&) = delete;
};

static void initMatrices(Matrices& data, size_t N, size_t row, size_t col,
                         size_t rows, size_t cols) {
    if (!rows || !cols) return;
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < N; ++k)
                data.a[i * N + k] = getPseudoRndValue(N, row + i, k);
        #pragma omp for schedule(static)
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < cols; ++j)
                data.b[k * cols + j] = getPseudoRndValue(N, k, col + j);
    }
}

static void matrixMultiply(Matrices& data, size_t N, size_t rows, size_t cols) {
    if (!rows || !cols) return;
    checkCuda(cudaMemcpyAsync(data.da, data.a, rows * N * sizeof(double),
                              cudaMemcpyHostToDevice, data.stream));
    checkCuda(cudaMemcpyAsync(data.db, data.b, N * cols * sizeof(double),
                              cudaMemcpyHostToDevice, data.stream));
    const double alpha = 1.0, beta = 0.0;
    // Row-major C = A B is column-major C^T = B^T A^T, with no transposes
    // or packing kernels. All arithmetic and accumulation are FP64.
    checkBlas(cublasDgemm(data.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                         int(cols), int(rows), int(N), &alpha,
                         data.db, int(cols), data.da, int(N), &beta,
                         data.dc, int(cols)));
    checkCuda(cudaMemcpyAsync(data.c, data.dc, rows * cols * sizeof(double),
                              cudaMemcpyDeviceToHost, data.stream));
    checkCuda(cudaStreamSynchronize(data.stream));
}

static bool validateResult(const double* C, size_t N, size_t row, size_t col,
                           size_t rows, size_t cols) {
    if (!rows || !cols) return true;
    // Preserve the original 25 checks, and sample every distributed block too.
    const size_t ri[] = {0, 1 % N, 2 % N, 3 % N, 4 % N, row, row + rows - 1};
    const size_t cj[] = {0, 1 % N, 2 % N, 3 % N, 4 % N, col, col + cols - 1};
    int valid = 1;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int pi = 0; pi < 7; ++pi) {
        for (int pj = 0; pj < 7; ++pj) {
            const size_t i = ri[pi], j = cj[pj];
            if (i < row || i >= row + rows || j < col || j >= col + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[(i - row) * cols + j - col];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) valid = 0;
        }
    }
    return valid != 0;
}

static void printDistributedResult(const double* local, size_t N, int rank,
                                    int ranks, const int dims[2]) {
    std::vector<double> result;
    if (rank == 0) result.resize(N * N);
    for (int source = 0; source < ranks; ++source) {
        const size_t row = blockStart(N, source / dims[1], dims[0]);
        const size_t col = blockStart(N, source % dims[1], dims[1]);
        const size_t rows = blockStart(N, source / dims[1] + 1, dims[0]) - row;
        const size_t cols = blockStart(N, source % dims[1] + 1, dims[1]) - col;
        const size_t count = rows * cols;
        if (!count) continue;
        std::vector<double> received;
        if (rank == 0 && source != 0) received.resize(count);
        // Chunk messages to avoid MPI's signed-int count limit.
        for (size_t offset = 0; offset < count;) {
            const int chunk = int(std::min(count - offset, size_t(INT_MAX)));
            if (rank == source && source != 0)
                MPI_Send(local + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (rank == 0 && source != 0)
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
        if (rank == 0) {
            const double* block = source == 0 ? local : received.data();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(block + i * cols, cols, result.data() + (row + i) * N + col);
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || !parsed ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    int exitCode = 0;
    try {
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        int devices = 0;
        checkCuda(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        checkCuda(cudaSetDevice(localRank % devices));
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        const size_t row = blockStart(N, rank / dims[1], dims[0]);
        const size_t col = blockStart(N, rank % dims[1], dims[1]);
        const size_t rows = blockStart(N, rank / dims[1] + 1, dims[0]) - row;
        const size_t cols = blockStart(N, rank % dims[1] + 1, dims[1]) - col;

        if (rank == 0) {
            printf("Matrix Multiplication Benchmark\n");
            printf("Matrix size: %zu x %zu\n", N, N);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI grid: %d x %d; OpenMP threads per rank: %d\n",
                   dims[0], dims[1], omp_get_max_threads());
            printf("Initializing matrices...\n");
        }
        Matrices data(rows, cols, N);
        initMatrices(data, N, row, col, rows, cols);
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        matrixMultiply(data, N, rows, cols);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            const double gflops = (2.0 * N * N * N) / seconds / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        }
        if (printResults) printDistributedResult(data.c, N, rank, ranks, dims);
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            int valid = validateResult(data.c, N, row, col, rows, cols), allValid = 0;
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
            exitCode = allValid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return exitCode;
}
