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

static int worldRank = 0;

[[noreturn]] static void fail(const char* message) {
    fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS status: %d\n", static_cast<int>(status));
        fail("cuBLAS operation failed");
    }
}

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced, contiguous intervals also support more ranks than matrix rows.
static size_t boundary(size_t N, int part, int parts) {
    return N / parts * part + std::min(N % parts, static_cast<size_t>(part));
}

struct Buffer {
    double* host = nullptr;
    double* device = nullptr;
    size_t count;
    explicit Buffer(size_t elements) : count(elements) {
        if (count) {
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(double)));
        }
    }
    ~Buffer() {
        if (device) cudaFree(device);
        if (host) cudaFreeHost(host);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || !parsed ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (!worldRank) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const size_t rowBegin = boundary(N, worldRank / dims[1], dims[0]);
    const size_t rowEnd = boundary(N, worldRank / dims[1] + 1, dims[0]);
    const size_t colBegin = boundary(N, worldRank % dims[1], dims[1]);
    const size_t colEnd = boundary(N, worldRank % dims[1] + 1, dims[1]);
    const size_t rows = rowEnd - rowBegin, cols = colEnd - colBegin;

    // Honor CUDA_VISIBLE_DEVICES, including launchers that expose one GPU per rank.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA device is required on every rank");
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    blasCheck(cublasSetStream(handle, stream));
    blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    if (!worldRank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    const bool active = rows && cols;
    Buffer A(active ? rows * N : 0), B(active ? N * cols : 0), C(rows * cols);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < A.count; ++i)
            A.host[i] = getPseudoRndValue(N, rowBegin + i / N, i % N);
        #pragma omp for schedule(static)
        for (size_t i = 0; i < B.count; ++i)
            B.host[i] = getPseudoRndValue(N, i / cols, colBegin + i % cols);
    }

    if (!worldRank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        cudaCheck(cudaMemcpyAsync(A.device, A.host, A.count * sizeof(double), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(B.device, B.host, B.count * sizeof(double), cudaMemcpyHostToDevice, stream));
        // Row-major C = A B is column-major C^T = B^T A^T.
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(N),
                             &alpha, B.device, static_cast<int>(cols), A.device, static_cast<int>(N),
                             &beta, C.device, static_cast<int>(cols)));
        cudaCheck(cudaMemcpyAsync(C.host, C.device, C.count * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!worldRank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / seconds / 1e9);
    }

    if (printResults) {
        // Gather only on demand. Chunk messages to avoid MPI's int count limit.
        std::vector<double> result, block;
        if (!worldRank) result.resize(N * N);
        for (int source = 0; source < ranks; ++source) {
            const size_t rb = boundary(N, source / dims[1], dims[0]);
            const size_t re = boundary(N, source / dims[1] + 1, dims[0]);
            const size_t cb = boundary(N, source % dims[1], dims[1]);
            const size_t ce = boundary(N, source % dims[1] + 1, dims[1]);
            const size_t count = (re - rb) * (ce - cb);
            if (!count) continue;
            if (!worldRank && source) block.resize(count);
            for (size_t offset = 0; offset < count; ) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                if (worldRank == source && source)
                    MPI_Send(C.host + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (!worldRank && source)
                    MPI_Recv(block.data() + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
            if (!worldRank) {
                const double* data = source ? block.data() : C.host;
                #pragma omp parallel for schedule(static)
                for (size_t i = rb; i < re; ++i)
                    std::copy_n(data + (i - rb) * (ce - cb), ce - cb, result.data() + i * N + cb);
            }
        }
        if (!worldRank) print_results(result, "MatrixC");
    }

    int valid = 1;
    if (validate) {
        if (!worldRank) printf("Validating result...\n");
        #pragma omp parallel for reduction(&:valid) schedule(static)
        for (int point = 0; point < 25; ++point) {
            const size_t i = (point / 5) % N, j = (point % 5) % N;
            if (i >= rowBegin && i < rowEnd && j >= colBegin && j < colEnd) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                const double actual = C.host[(i - rowBegin) * cols + j - colBegin];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || error > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, error);
                    valid = 0;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!worldRank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    blasCheck(cublasDestroy(handle));
    cudaCheck(cudaStreamDestroy(stream));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    MPI_Finalize();
    return status;
}
