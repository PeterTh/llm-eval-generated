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

static int worldRank;

[[noreturn]] static void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(status, message, &length);
        fail(message);
    }
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[80];
        std::snprintf(message, sizeof(message), "cuBLAS error %d", int(status));
        fail(message);
    }
}

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static size_t boundary(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min(size_t(coordinate), N % parts);
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Chunk collectives and point-to-point transfers to avoid MPI's int count limit.
static void broadcast(double* data, size_t count, int root, MPI_Comm comm,
                      std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count; ) {
        const int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        MPI_Request request;
        mpiCheck(MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &request));
        requests.push_back(request);
        offset += chunk;
    }
}

static void transfer(double* data, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count; ) {
        const int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        if (send) mpiCheck(MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(data + offset, chunk, MPI_DOUBLE, peer, 0,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

struct PanelBuffer {
    double *hostA = nullptr, *hostB = nullptr, *deviceA = nullptr, *deviceB = nullptr;
    cudaEvent_t copied;
    std::vector<MPI_Request> requests;
};

static int run(int argc, char** argv) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end || value == 0 ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() / value /
                                         sizeof(double)) {
                if (!worldRank) std::fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
            N = size_t(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int worldSize;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                MPI_INFO_NULL, &localComm));
    int localRank, deviceCount;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) fail("A CUDA GPU is required on every node.");
    cudaCheck(cudaSetDevice(localRank % deviceCount));
    mpiCheck(MPI_Comm_free(&localComm));

    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(worldSize, 2, dims));
    const int row = worldRank / dims[1], col = worldRank % dims[1];
    MPI_Comm rowComm, colComm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    const size_t rowStart = boundary(N, row, dims[0]);
    const size_t colStart = boundary(N, col, dims[1]);
    const size_t rows = boundary(N, row + 1, dims[0]) - rowStart;
    const size_t cols = boundary(N, col + 1, dims[1]) - colStart;

    // SUMMA: cyclic ownership of K panels across processor rows/columns.
    // Inputs are generated directly at their owners, with the original global indices.
    constexpr size_t panelWidth = 1024;
    const size_t width = std::min(N, panelWidth);
    const size_t steps = (N + width - 1) / width;
    std::vector<std::vector<double>> ownedA(steps), ownedB(steps);
    if (!worldRank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    for (size_t step = 0; step < steps; ++step) {
        const size_t kStart = step * width, kCount = std::min(width, N - kStart);
        if (int(step % dims[1]) == col) {
            auto& a = ownedA[step];
            a.resize(rows * kCount);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < kCount; ++k)
                    a[i * kCount + k] = getPseudoRndValue(N, rowStart + i, kStart + k);
        }
        if (int(step % dims[0]) == row) {
            auto& b = ownedB[step];
            b.resize(kCount * cols);
            #pragma omp parallel for schedule(static)
            for (size_t k = 0; k < kCount; ++k)
                for (size_t j = 0; j < cols; ++j)
                    b[k * cols + j] = getPseudoRndValue(N, kStart + k, colStart + j);
        }
    }

    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    blasCheck(cublasSetStream(handle, stream));
    double* deviceC;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceC), std::max(size_t(1), rows * cols) * sizeof(double)));
    cudaCheck(cudaMemsetAsync(deviceC, 0, rows * cols * sizeof(double), stream));
    PanelBuffer buffers[2];
    for (auto& b : buffers) {
        const size_t bytesA = std::max(size_t(1), rows * width) * sizeof(double);
        const size_t bytesB = std::max(size_t(1), width * cols) * sizeof(double);
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&b.hostA), bytesA));
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&b.hostB), bytesB));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&b.deviceA), bytesA));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&b.deviceB), bytesB));
        cudaCheck(cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming));
    }
    auto beginPanel = [&](size_t step) {
        auto& b = buffers[step % 2];
        // A host buffer can be reused once its previous asynchronous upload finishes.
        if (step >= 2) cudaCheck(cudaEventSynchronize(b.copied));
        const size_t kCount = std::min(width, N - step * width);
        if (int(step % dims[1]) == col && rows)
            std::memcpy(b.hostA, ownedA[step].data(), rows * kCount * sizeof(double));
        if (int(step % dims[0]) == row && cols)
            std::memcpy(b.hostB, ownedB[step].data(), kCount * cols * sizeof(double));
        b.requests.clear();
        broadcast(b.hostA, rows * kCount, int(step % dims[1]), rowComm, b.requests);
        broadcast(b.hostB, kCount * cols, int(step % dims[0]), colComm, b.requests);
    };
    cudaCheck(cudaStreamSynchronize(stream));
    if (!worldRank) std::printf("Computing matrix multiplication...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    beginPanel(0);
    const double alpha = 1.0, beta = 1.0;
    for (size_t step = 0; step < steps; ++step) {
        auto& b = buffers[step % 2];
        if (!b.requests.empty())
            mpiCheck(MPI_Waitall(int(b.requests.size()), b.requests.data(), MPI_STATUSES_IGNORE));
        const size_t kCount = std::min(width, N - step * width);
        cudaCheck(cudaMemcpyAsync(b.deviceA, b.hostA, rows * kCount * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(b.deviceB, b.hostB, kCount * cols * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaEventRecord(b.copied, stream));
        if (rows && cols) {
            // Row-major C = A B is column-major C^T = B^T A^T.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 int(cols), int(rows), int(kCount), &alpha,
                                 b.deviceB, int(cols), b.deviceA, int(kCount),
                                 &beta, deviceC, int(cols)));
        }
        // Nonblocking collectives progress while the current GPU GEMM runs.
        if (step + 1 < steps) beginPanel(step + 1);
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double duration;
    mpiCheck(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!worldRank) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * double(N) * double(N) * double(N) / duration / 1e9);
    }

    std::vector<double> localC;
    if (printResults || validate) {
        localC.resize(rows * cols);
        if (!localC.empty())
            cudaCheck(cudaMemcpy(localC.data(), deviceC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        if (!worldRank) {
            std::vector<double> C(N * N);
            for (int rank = 0; rank < worldSize; ++rank) {
                const int rr = rank / dims[1], cc = rank % dims[1];
                const size_t r0 = boundary(N, rr, dims[0]), c0 = boundary(N, cc, dims[1]);
                const size_t nr = boundary(N, rr + 1, dims[0]) - r0;
                const size_t nc = boundary(N, cc + 1, dims[1]) - c0;
                std::vector<double> received;
                const double* source = localC.data();
                if (rank) {
                    received.resize(nr * nc);
                    transfer(received.data(), received.size(), rank, false);
                    source = received.data();
                }
                if (nc) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < nr; ++i)
                        std::copy_n(source + i * nc, nc, C.data() + (r0 + i) * N + c0);
                }
            }
            print_results(C, "MatrixC");
        } else transfer(localC.data(), localC.size(), 0, true);
    }
    int valid = 1;
    if (validate) {
        if (!worldRank) std::printf("Validating result...\n");
        for (size_t pi = 0; pi < 5; ++pi) {
            const size_t i = pi % N;
            if (i < rowStart || i >= rowStart + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = pj % N;
                if (j < colStart || j >= colStart + cols) continue;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                const double actual = localC[(i - rowStart) * cols + j - colStart];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || error > 1e-6) {
                    std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                i, j, expected, actual, error);
                    valid = 0;
                }
            }
        }
        int allValid;
        mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        valid = allValid;
        if (!worldRank) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    for (auto& b : buffers) {
        cudaCheck(cudaEventDestroy(b.copied));
        cudaCheck(cudaFreeHost(b.hostA));
        cudaCheck(cudaFreeHost(b.hostB));
        cudaCheck(cudaFree(b.deviceA));
        cudaCheck(cudaFree(b.deviceB));
    }
    cudaCheck(cudaFree(deviceC));
    blasCheck(cublasDestroy(handle));
    cudaCheck(cudaStreamDestroy(stream));
    mpiCheck(MPI_Comm_free(&rowComm));
    mpiCheck(MPI_Comm_free(&colComm));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required.");
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    mpiCheck(MPI_Finalize());
    return status;
}
