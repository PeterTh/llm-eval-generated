#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) fail("MPI operation failed");
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[128];
        snprintf(message, sizeof(message), "cuBLAS operation failed (status %d)", int(status));
        fail(message);
    }
}

// Preserve the original unsigned-integer arithmetic and double precision.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// Balanced contiguous partitions, including the case of more ranks than rows.
size_t offset(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(size_t(coordinate), N % parts);
}

struct Panel {
    double* a = nullptr;
    double* b = nullptr;
    std::vector<MPI_Request> requests;

    Panel(size_t aCount, size_t bCount) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&a),
                                 std::max(size_t(1), aCount) * sizeof(double)));
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&b),
                                 std::max(size_t(1), bCount) * sizeof(double)));
    }
    ~Panel() {
        cudaFreeHost(a);
        cudaFreeHost(b);
    }
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;
};

// Chunking avoids MPI's signed-int count limit without requiring MPI-4.
void broadcast(double* data, size_t count, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    for (size_t start = 0; start < count;) {
        int chunk = static_cast<int>(std::min(count - start, size_t(INT_MAX)));
        MPI_Request request;
        mpiCheck(MPI_Ibcast(data + start, chunk, MPI_DOUBLE, root, comm, &request));
        requests.push_back(request);
        start += chunk;
    }
}

void preparePanel(Panel& panel, size_t N, size_t first, size_t width,
                  size_t rowStart, size_t rows, size_t colStart, size_t cols,
                  int row, int col, const int* dims, size_t step,
                  MPI_Comm rowComm, MPI_Comm colComm) {
    panel.requests.clear();
    // All blocks are row-major. The input formula permits generating each
    // panel at its broadcast root without storing or scattering full inputs.
    if (col == int(step % dims[1])) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < width; ++k)
                panel.a[i * width + k] = getPseudoRndValue(N, rowStart + i, first + k);
        }
    }
    if (row == int(step % dims[0])) {
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < width; ++k) {
            for (size_t j = 0; j < cols; ++j)
                panel.b[k * cols + j] = getPseudoRndValue(N, first + k, colStart + j);
        }
    }
    broadcast(panel.a, rows * width, int(step % dims[1]), rowComm, panel.requests);
    broadcast(panel.b, width * cols, int(step % dims[0]), colComm, panel.requests);
}

void waitPanel(Panel& panel) {
    // MPI calls remain on the main thread (MPI_THREAD_FUNNELED).
    for (MPI_Request& request : panel.requests)
        mpiCheck(MPI_Wait(&request, MPI_STATUS_IGNORE));
}

bool validateResult(const std::vector<double>& C, size_t N,
                    size_t rowStart, size_t rows, size_t colStart, size_t cols) {
    int valid = 1;
    // Keep the original 25 checks, and check corners of every distributed block.
    std::vector<std::pair<size_t, size_t>> checks;
    for (size_t pi = 0; pi < 5; ++pi)
        for (size_t pj = 0; pj < 5; ++pj) {
            size_t i = pi % N, j = pj % N;
            if (i >= rowStart && i < rowStart + rows &&
                j >= colStart && j < colStart + cols)
                checks.emplace_back(i, j);
        }
    if (rows && cols) {
        checks.emplace_back(rowStart, colStart);
        checks.emplace_back(rowStart + rows - 1, colStart + cols - 1);
    }
    #pragma omp parallel for schedule(static) reduction(&:valid)
    for (size_t p = 0; p < checks.size(); ++p) {
        const auto [i, j] = checks[p];
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k)
            expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
        double actual = C[(i - rowStart) * cols + j - colStart];
        double error = std::abs((actual - expected) / (expected + 1e-10));
        if (!std::isfinite(actual) || error > 1e-6) {
            #pragma omp critical
            fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                    i, j, expected, actual, error);
            valid = 0;
        }
    }
    int allValid = 0;
    mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    return allValid != 0;
}

void transfer(double* data, size_t count, int peer, bool send) {
    for (size_t start = 0; start < count;) {
        int chunk = static_cast<int>(std::min(count - start, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data + start, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data + start, chunk, MPI_DOUBLE, peer, 0,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        start += chunk;
    }
}

void printDistributedResult(std::vector<double>& local, size_t N,
                            const int* dims, int ranks) {
    if (worldRank != 0) {
        transfer(local.data(), local.size(), 0, true);
        return;
    }
    std::vector<double> result(N * N);
    std::vector<double> received;
    for (int rank = 0; rank < ranks; ++rank) {
        int row = rank / dims[1], col = rank % dims[1];
        size_t rowStart = offset(N, row, dims[0]);
        size_t rows = offset(N, row + 1, dims[0]) - rowStart;
        size_t colStart = offset(N, col, dims[1]);
        size_t cols = offset(N, col + 1, dims[1]) - colStart;
        double* block = local.data();
        if (rank != 0) {
            received.resize(rows * cols);
            transfer(received.data(), received.size(), rank, false);
            block = received.data();
        }
        if (cols) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                std::copy_n(block + i * cols, cols,
                            result.data() + (rowStart + i) * N + colStart);
        }
    }
    print_results(result, "MatrixC");
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
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = strtoull(arg, &end, 10);
            if (errno || !*arg || *end || *arg == '-' || parsed == 0 ||
                parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                if (worldRank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int ranks = 0;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(ranks, 2, dims));
    int row = worldRank / dims[1], col = worldRank % dims[1];
    MPI_Comm rowComm, colComm, localComm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                 worldRank, MPI_INFO_NULL, &localComm));
    int localRank = 0, localRanks = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    mpiCheck(MPI_Comm_size(localComm, &localRanks));
    // Respect OMP_NUM_THREADS; otherwise divide available host cores among ranks.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localRanks));
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("CUDA GPU required");
    // Also works when the launcher exposes just one GPU per rank.
    cudaCheck(cudaSetDevice(localRank % devices));

    size_t rowStart = offset(N, row, dims[0]);
    size_t rows = offset(N, row + 1, dims[0]) - rowStart;
    size_t colStart = offset(N, col, dims[1]);
    size_t cols = offset(N, col + 1, dims[1]) - colStart;
    constexpr size_t panelWidth = 1024;
    size_t capacity = std::min(N, panelWidth);
    Panel panels[2] = {{rows * capacity, capacity * cols},
                       {rows * capacity, capacity * cols}};
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceA),
                         std::max(size_t(1), rows * capacity) * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceB),
                         std::max(size_t(1), capacity * cols) * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceC),
                         std::max(size_t(1), rows * cols) * sizeof(double)));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    blasCheck(cublasSetStream(handle, stream));
    // IEEE double precision; no reduced-precision math modes.
    blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI process grid: %d x %d; OpenMP threads per rank: %d\n",
               dims[0], dims[1], omp_get_max_threads());
        printf("Initializing matrices...\n");
        printf("Computing matrix multiplication...\n");
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    preparePanel(panels[0], N, 0, capacity, rowStart, rows, colStart, cols,
                 row, col, dims, 0, rowComm, colComm);
    size_t steps = (N + panelWidth - 1) / panelWidth;
    for (size_t step = 0; step < steps; ++step) {
        size_t first = step * panelWidth;
        size_t width = std::min(panelWidth, N - first);
        Panel& panel = panels[step % 2];
        waitPanel(panel);
        if (rows && cols) {
            cudaCheck(cudaMemcpyAsync(deviceA, panel.a, rows * width * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaMemcpyAsync(deviceB, panel.b, width * cols * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            const double alpha = 1.0, beta = step == 0 ? 0.0 : 1.0;
            // Row-major C=A*B is column-major C^T=B^T*A^T; no transposes needed.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 int(cols), int(rows), int(width), &alpha,
                                 deviceB, int(cols), deviceA, int(width), &beta,
                                 deviceC, int(cols)));
        }
        if (step + 1 < steps) {
            size_t next = first + panelWidth;
            preparePanel(panels[(step + 1) % 2], N, next, std::min(panelWidth, N - next),
                         rowStart, rows, colStart, cols, row, col, dims, step + 1,
                         rowComm, colComm);
            // Enter MPI while the GPU runs, including on MPI implementations
            // without a background progress thread.
            waitPanel(panels[(step + 1) % 2]);
        }
        // Protect both device buffers and pinned host buffers from reuse.
        cudaCheck(cudaStreamSynchronize(stream));
    }
    std::vector<double> C(rows * cols);
    if (!C.empty())
        cudaCheck(cudaMemcpy(C.data(), deviceC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double elapsed = MPI_Wtime() - start, seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }
    if (printResults) printDistributedResult(C, N, dims, ranks);
    bool valid = true;
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        valid = validateResult(C, N, rowStart, rows, colStart, cols);
        if (worldRank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    blasCheck(cublasDestroy(handle));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(deviceA));
    cudaCheck(cudaFree(deviceB));
    cudaCheck(cudaFree(deviceC));
    mpiCheck(MPI_Comm_free(&rowComm));
    mpiCheck(MPI_Comm_free(&colComm));
    mpiCheck(MPI_Comm_free(&localComm));
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support required");
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    mpiCheck(MPI_Finalize());
    return result;
}
