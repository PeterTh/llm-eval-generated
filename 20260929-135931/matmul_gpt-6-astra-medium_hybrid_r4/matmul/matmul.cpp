#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

[[noreturn]] void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[80];
        snprintf(message, sizeof(message), "cuBLAS failed (status %d)", int(status));
        fail(message);
    }
}

// Balanced contiguous blocks; ranks beyond an axis length own empty blocks.
size_t boundary(size_t n, int coordinate, int parts) {
    return n / parts * coordinate + std::min(n % parts, size_t(coordinate));
}

struct Matrix {
    double* host = nullptr;
    double* device = nullptr;
    size_t size;

    explicit Matrix(size_t count) : size(count) {
        if (count) {
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(double)));
        }
    }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    ~Matrix() {
        if (device) cudaCheck(cudaFree(device));
        if (host) cudaCheck(cudaFreeHost(host));
    }
};

// Only -r needs a complete matrix. Chunk MPI messages to avoid the int count
// limit, then unpack each rank's rectangular block in global row-major order.
std::vector<double> gatherResult(const Matrix& C, size_t N, int rank,
                                 int ranks, const int dims[2]) {
    std::vector<double> result;
    if (rank == 0) result.resize(N * N);
    constexpr size_t chunk = 1 << 26;
    for (int source = 0; source < ranks; ++source) {
        const size_t row = boundary(N, source / dims[1], dims[0]);
        const size_t col = boundary(N, source % dims[1], dims[1]);
        const size_t rows = boundary(N, source / dims[1] + 1, dims[0]) - row;
        const size_t cols = boundary(N, source % dims[1] + 1, dims[1]) - col;
        const size_t count = rows * cols;
        if (!count) continue;
        if (rank == 0) {
            std::vector<double> received;
            const double* tile = C.host;
            if (source) {
                received.resize(count);
                for (size_t offset = 0; offset < count; offset += chunk)
                    MPI_Recv(received.data() + offset, int(std::min(chunk, count - offset)),
                             MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                tile = received.data();
            }
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(tile + i * cols, cols, result.data() + (row + i) * N + col);
        } else if (rank == source) {
            for (size_t offset = 0; offset < count; offset += chunk)
                MPI_Send(C.host + offset, int(std::min(chunk, count - offset)),
                         MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    return result;
}

bool validateResult(const Matrix& C, size_t N, size_t row, size_t col,
                    size_t rows, size_t cols) {
    int valid = 1;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = size_t(pi) % N;
            const size_t j = size_t(pj) % N;
            if (i < row || i >= row + rows || j < col || j >= col + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C.host[(i - row) * cols + j - col];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                #pragma omp critical
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                valid = 0;
            }
        }
    }
    int globalValid = 0;
    MPI_Allreduce(&valid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalValid != 0;
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
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || end == value || *end || *value == '-' || !parsed ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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

    // Respect scheduler GPU visibility. With multiple visible GPUs, local ranks
    // select distinct devices; launch one MPI rank per GPU for best throughput.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank = 0;
    MPI_Comm_rank(local, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA GPU is required on every participating node.");
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const size_t row = boundary(N, rank / dims[1], dims[0]);
    const size_t col = boundary(N, rank % dims[1], dims[1]);
    const size_t rows = boundary(N, rank / dims[1] + 1, dims[0]) - row;
    const size_t cols = boundary(N, rank % dims[1] + 1, dims[1]) - col;
    const bool active = rows && cols;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    Matrix A(active ? rows * N : 0);
    Matrix B(active ? N * cols : 0);
    Matrix C(rows * cols);
    // Reconstruct only the required input slices, avoiding both full matrix
    // replication and any initialization communication between cluster nodes.
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t index = 0; index < A.size; ++index)
            A.host[index] = getPseudoRndValue(N, row + index / N, index % N);
        #pragma omp for schedule(static)
        for (size_t index = 0; index < B.size; ++index)
            B.host[index] = getPseudoRndValue(N, index / cols, col + index % cols);
    }

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    if (active) {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blasCheck(cublasCreate(&handle));
        blasCheck(cublasSetStream(handle, stream));
        // Full double precision; do not select reduced-precision math modes.
        blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        cudaCheck(cudaMemcpyAsync(A.device, A.host, A.size * sizeof(double), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(B.device, B.host, B.size * sizeof(double), cudaMemcpyHostToDevice, stream));
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A B is column-major C^T = B^T A^T; no transposes
        // or packing kernels are required, including for uneven rank blocks.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             int(cols), int(rows), int(N), &alpha,
                             B.device, int(cols), A.device, int(N), &beta,
                             C.device, int(cols)));
        cudaCheck(cudaMemcpyAsync(C.host, C.device, C.size * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }
    if (active) {
        blasCheck(cublasDestroy(handle));
        cudaCheck(cudaStreamDestroy(stream));
    }
    if (printResults) {
        const auto result = gatherResult(C, N, rank, ranks, dims);
        if (rank == 0) print_results(result, "MatrixC");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(C, N, row, col, rows, cols);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required.");
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    MPI_Finalize();
    return result;
}
