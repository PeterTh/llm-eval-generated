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

// Keep the original unsigned arithmetic and deterministic input values.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

[[noreturn]] void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}

void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[80];
        snprintf(message, sizeof(message), "cuBLAS failed with status %d", static_cast<int>(status));
        fail(message);
    }
}

// Pinned panels allow asynchronous DMA on each GPU's independent stream.
struct HostPanel {
    double* data = nullptr;
    explicit HostPanel(size_t count) {
        if (count) cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&data), count * sizeof(double), cudaHostAllocPortable));
    }
    ~HostPanel() { if (data) cudaFreeHost(data); }
    HostPanel(const HostPanel&) = delete;
    HostPanel& operator=(const HostPanel&) = delete;
};

struct Tile {
    size_t row, rows, col, cols;
};

Tile tileFor(int rank, const int* dims, size_t N) {
    const size_t r = static_cast<size_t>(rank / dims[1]);
    const size_t c = static_cast<size_t>(rank % dims[1]);
    const size_t firstRow = N * r / dims[0];
    const size_t firstCol = N * c / dims[1];
    return {firstRow, N * (r + 1) / dims[0] - firstRow,
            firstCol, N * (c + 1) / dims[1] - firstCol};
}

struct GPUWorker {
    int device;
    size_t firstRow, rows;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

void prepareGPU(GPUWorker& w, size_t N, size_t cols) {
    cudaCheck(cudaSetDevice(w.device));
    cudaCheck(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    blasCheck(cublasCreate(&w.handle));
    blasCheck(cublasSetStream(w.handle, w.stream));
    // Preserve FP64 arithmetic; do not enable reduced precision math modes.
    blasCheck(cublasSetMathMode(w.handle, CUBLAS_DEFAULT_MATH));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.a), w.rows * N * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.b), N * cols * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&w.c), w.rows * cols * sizeof(double)));
    // Load the FP64 kernel before timing; otherwise CUDA's lazy module loading
    // dominates small benchmark runs. Real panel transfers remain timed.
    cudaCheck(cudaMemsetAsync(w.a, 0, sizeof(double), w.stream));
    cudaCheck(cudaMemsetAsync(w.b, 0, sizeof(double), w.stream));
    const double alpha = 1.0, beta = 0.0;
    blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                         &alpha, w.b, 1, w.a, 1, &beta, w.c, 1));
    cudaCheck(cudaStreamSynchronize(w.stream));
}

void multiplyGPU(GPUWorker& w, const HostPanel& A, const HostPanel& B,
                 HostPanel& C, size_t N, size_t cols) {
    cudaCheck(cudaSetDevice(w.device));
    cudaCheck(cudaMemcpyAsync(w.a, A.data + w.firstRow * N, w.rows * N * sizeof(double),
                              cudaMemcpyHostToDevice, w.stream));
    cudaCheck(cudaMemcpyAsync(w.b, B.data, N * cols * sizeof(double),
                              cudaMemcpyHostToDevice, w.stream));
    const double alpha = 1.0, beta = 0.0;
    // Row-major C = A B is column-major C^T = B^T A^T: no transpose or packing pass.
    blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         static_cast<int>(cols), static_cast<int>(w.rows), static_cast<int>(N),
                         &alpha, w.b, static_cast<int>(cols), w.a, static_cast<int>(N),
                         &beta, w.c, static_cast<int>(cols)));
    cudaCheck(cudaMemcpyAsync(C.data + w.firstRow * cols, w.c, w.rows * cols * sizeof(double),
                              cudaMemcpyDeviceToHost, w.stream));
    cudaCheck(cudaStreamSynchronize(w.stream));
}

void releaseGPU(GPUWorker& w) {
    cudaCheck(cudaSetDevice(w.device));
    blasCheck(cublasDestroy(w.handle));
    cudaCheck(cudaFree(w.a));
    cudaCheck(cudaFree(w.b));
    cudaCheck(cudaFree(w.c));
    cudaCheck(cudaStreamDestroy(w.stream));
}

bool validateResult(const HostPanel& A, const HostPanel& B, const HostPanel& C,
                    const Tile& tile, size_t N) {
    int valid = 1;
    // Retain the original 5x5 checks, and also sample every distributed tile.
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int pi = 0; pi < 7; ++pi) {
        for (int pj = 0; pj < 7; ++pj) {
            if (!tile.rows || !tile.cols) continue;
            const size_t i = pi < 5 ? static_cast<size_t>(pi) % N
                                   : tile.row + (pi == 5 ? tile.rows / 2 : tile.rows - 1);
            const size_t j = pj < 5 ? static_cast<size_t>(pj) % N
                                   : tile.col + (pj == 5 ? tile.cols / 2 : tile.cols - 1);
            if (i < tile.row || i >= tile.row + tile.rows ||
                j < tile.col || j >= tile.col + tile.cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A.data[(i - tile.row) * N + k] * B.data[k * tile.cols + j - tile.col];
            const double actual = C.data[(i - tile.row) * tile.cols + j - tile.col];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                #pragma omp critical
                fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, error);
                valid = 0;
            }
        }
    }
    return valid != 0;
}

void gatherResults(const HostPanel& local, size_t N, int rank, int ranks, const int* dims) {
    // Chunk messages to avoid MPI's int count limit. Only -r allocates a full matrix.
    constexpr size_t chunk = 1U << 26;
    const Tile own = tileFor(rank, dims, N);
    if (rank != 0) {
        const size_t count = own.rows * own.cols;
        for (size_t offset = 0; offset < count; offset += chunk)
            MPI_Send(local.data + offset, static_cast<int>(std::min(chunk, count - offset)),
                     MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        return;
    }
    std::vector<double> full(N * N), received;
    for (int source = 0; source < ranks; ++source) {
        const Tile tile = tileFor(source, dims, N);
        const size_t count = tile.rows * tile.cols;
        if (!count) continue;
        const double* panel = local.data;
        if (source != 0) {
            received.resize(count);
            for (size_t offset = 0; offset < count; offset += chunk)
                MPI_Recv(received.data() + offset, static_cast<int>(std::min(chunk, count - offset)),
                         MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            panel = received.data();
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tile.rows; ++i)
            std::copy_n(panel + i * tile.cols, tile.cols, full.data() + (tile.row + i) * N + tile.col);
    }
    print_results(full, "MatrixC");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || end == value || *end || *value == '-' || parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("CUDA GPU required by the hybrid MPI/OpenMP/CUDA benchmark");
    // Honor CUDA_VISIBLE_DEVICES, including per-rank GPU masks supplied by a scheduler.
    std::vector<int> assigned;
    for (int device = localRank; device < devices; device += localRanks) assigned.push_back(device);
    if (assigned.empty()) assigned.push_back(localRank % devices);
    MPI_Comm_free(&node);

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const Tile tile = tileFor(rank, dims, N);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
               N, N, validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; CUDA GPUs assigned per rank; OpenMP host threads: %d\n",
               dims[0], dims[1], omp_get_max_threads());
        printf("Initializing matrices...\n");
    }
    cudaCheck(cudaSetDevice(assigned.front()));
    HostPanel A(tile.rows * N), B(tile.cols * N), C(tile.rows * tile.cols);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < tile.rows; ++i)
            for (size_t k = 0; k < N; ++k)
                A.data[i * N + k] = getPseudoRndValue(N, tile.row + i, k);
        #pragma omp for schedule(static)
        for (size_t k = 0; k < N; ++k)
            for (size_t j = 0; j < tile.cols; ++j)
                B.data[k * tile.cols + j] = getPseudoRndValue(N, k, tile.col + j);
    }

    const int workerCount = tile.cols ? static_cast<int>(std::min(tile.rows, assigned.size())) : 0;
    std::vector<GPUWorker> workers(workerCount);
    for (int w = 0; w < workerCount; ++w) {
        const size_t begin = tile.rows * w / workerCount;
        workers[w].device = assigned[w];
        workers[w].firstRow = begin;
        workers[w].rows = tile.rows * (w + 1) / workerCount - begin;
    }
    if (workerCount) {
        #pragma omp parallel for num_threads(workerCount) schedule(static)
        for (int w = 0; w < workerCount; ++w) prepareGPU(workers[w], N, tile.cols);
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (workerCount) {
        #pragma omp parallel for num_threads(workerCount) schedule(static)
        for (int w = 0; w < workerCount; ++w)
            multiplyGPU(workers[w], A, B, C, N, tile.cols);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed / 1e9);
    }
    if (printResults) gatherResults(C, N, rank, ranks, dims);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(A, B, C, tile, N) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (workerCount) {
        #pragma omp parallel for num_threads(workerCount) schedule(static)
        for (int w = 0; w < workerCount; ++w) releaseGPU(workers[w]);
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    MPI_Finalize();
    return status;
}
