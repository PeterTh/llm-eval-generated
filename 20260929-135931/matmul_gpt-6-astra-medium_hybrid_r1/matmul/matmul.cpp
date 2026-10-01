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

namespace {
int rank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(status, message, &length);
        fail(message);
    }
}
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}
void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[100];
        std::snprintf(message, sizeof(message), "cuBLAS failed (status %d)", int(status));
        fail(message);
    }
}

// Keep the original size_t arithmetic, including its unsigned wraparound.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct Block {
    size_t row, col, rows, cols;
};
Block blockFor(int id, const int dims[2], size_t N) {
    const size_t r = id / dims[1], c = id % dims[1];
    const size_t row = N * r / dims[0], col = N * c / dims[1];
    return {row, col, N * (r + 1) / dims[0] - row,
                      N * (c + 1) / dims[1] - col};
}

// Two pinned staging buffers bound input memory to O((rows + cols) * panel).
// A common compute stream serializes updates of C; separate transfer streams
// overlap the next panel's H2D copies and CPU initialization with DGEMM.
struct Panel {
    double *a = nullptr, *b = nullptr, *da = nullptr, *db = nullptr;
    cudaStream_t transfer;
    cudaEvent_t ready, done;
    bool used = false;
};

class Multiply {
    Block block;
    size_t N, width;
    Panel panels[2];
    double* dc = nullptr;
    cudaStream_t compute = nullptr;
    cublasHandle_t handle = nullptr;
public:
    Multiply(Block block_, size_t N_) : block(block_), N(N_), width(std::min(N, size_t(1024))) {
        if (!block.rows || !block.cols) return;
        cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        blasCheck(cublasCreate(&handle));
        blasCheck(cublasSetStream(handle, compute));
        blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dc), block.rows * block.cols * sizeof(double)));
        for (auto& p : panels) {
            cudaCheck(cudaStreamCreateWithFlags(&p.transfer, cudaStreamNonBlocking));
            cudaCheck(cudaEventCreateWithFlags(&p.ready, cudaEventDisableTiming));
            cudaCheck(cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming));
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&p.a), block.rows * width * sizeof(double)));
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&p.b), block.cols * width * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&p.da), block.rows * width * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&p.db), block.cols * width * sizeof(double)));
        }
    }

    void run(std::vector<double>& C) {
        if (!block.rows || !block.cols) return;
        for (size_t k0 = 0, step = 0; k0 < N; k0 += width, ++step) {
            Panel& p = panels[step % 2];
            const size_t ksize = std::min(width, N - k0);
            if (p.used) {
                // The CPU may overwrite pinned memory once H2D is complete;
                // device buffers must additionally wait for the previous GEMM.
                cudaCheck(cudaEventSynchronize(p.ready));
                cudaCheck(cudaStreamWaitEvent(p.transfer, p.done, 0));
            }
            #pragma omp parallel
            {
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < block.rows; ++i)
                    for (size_t k = 0; k < ksize; ++k)
                        p.a[i * ksize + k] = getPseudoRndValue(N, block.row + i, k0 + k);
                #pragma omp for schedule(static)
                for (size_t k = 0; k < ksize; ++k)
                    for (size_t j = 0; j < block.cols; ++j)
                        p.b[k * block.cols + j] = getPseudoRndValue(N, k0 + k, block.col + j);
            }
            cudaCheck(cudaMemcpyAsync(p.da, p.a, block.rows * ksize * sizeof(double),
                                      cudaMemcpyHostToDevice, p.transfer));
            cudaCheck(cudaMemcpyAsync(p.db, p.b, block.cols * ksize * sizeof(double),
                                      cudaMemcpyHostToDevice, p.transfer));
            cudaCheck(cudaEventRecord(p.ready, p.transfer));
            cudaCheck(cudaStreamWaitEvent(compute, p.ready, 0));
            const double alpha = 1.0, beta = k0 == 0 ? 0.0 : 1.0;
            // Row-major C = A B is column-major C^T = B^T A^T.
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 int(block.cols), int(block.rows), int(ksize),
                                 &alpha, p.db, int(block.cols), p.da, int(ksize),
                                 &beta, dc, int(block.cols)));
            cudaCheck(cudaEventRecord(p.done, compute));
            p.used = true;
        }
        cudaCheck(cudaStreamSynchronize(compute));
        cudaCheck(cudaMemcpy(C.data(), dc, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }

    ~Multiply() {
        if (!dc) return;
        for (auto& p : panels) {
            cudaCheck(cudaFree(p.da));
            cudaCheck(cudaFree(p.db));
            cudaCheck(cudaFreeHost(p.a));
            cudaCheck(cudaFreeHost(p.b));
            cudaCheck(cudaEventDestroy(p.ready));
            cudaCheck(cudaEventDestroy(p.done));
            cudaCheck(cudaStreamDestroy(p.transfer));
        }
        blasCheck(cublasDestroy(handle));
        cudaCheck(cudaFree(dc));
        cudaCheck(cudaStreamDestroy(compute));
    }
};

bool validateResult(const std::vector<double>& C, Block b, size_t N) {
    int valid = 1;
    // Original 5x5 positions plus corners and center of every distributed block.
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int point = 0; point < 30; ++point) {
        if (!b.rows || !b.cols) continue;
        size_t i, j;
        if (point < 25) {
            i = size_t(point / 5) % N;
            j = size_t(point % 5) % N;
            if (i < b.row || i >= b.row + b.rows || j < b.col || j >= b.col + b.cols) continue;
        } else {
            const int p = point - 25;
            i = b.row + (p == 4 ? b.rows / 2 : (p / 2) * (b.rows - 1));
            j = b.col + (p == 4 ? b.cols / 2 : (p % 2) * (b.cols - 1));
        }
        double expected = 0;
        for (size_t k = 0; k < N; ++k)
            expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
        const double actual = C[(i - b.row) * b.cols + j - b.col];
        const double error = std::abs((actual - expected) / (expected + 1e-10));
        if (!std::isfinite(actual) || error > 1e-6) {
            #pragma omp critical
            std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, error);
            valid = 0;
        }
    }
    int allValid;
    mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    return allValid != 0;
}

// Only -r needs a global matrix. Use bounded messages to avoid MPI int-count
// overflow, and unpack one rank at a time without a second full-size matrix.
void printDistributed(const std::vector<double>& C, size_t N, int ranks, const int dims[2]) {
    constexpr size_t chunk = 1u << 20;
    if (rank != 0) {
        for (size_t off = 0; off < C.size(); off += chunk)
            mpiCheck(MPI_Send(C.data() + off, int(std::min(chunk, C.size() - off)),
                              MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        return;
    }
    std::vector<double> full(N * N), buffer;
    for (int source = 0; source < ranks; ++source) {
        const Block b = blockFor(source, dims, N);
        const size_t count = b.rows * b.cols;
        const double* data = C.data();
        if (source != 0) {
            buffer.resize(count);
            for (size_t off = 0; off < count; off += chunk)
                mpiCheck(MPI_Recv(buffer.data() + off, int(std::min(chunk, count - off)),
                                  MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            data = buffer.data();
        }
        if (!count) continue;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < b.rows; ++i)
            std::copy_n(data + i * b.cols, b.cols, full.data() + (b.row + i) * N + b.col);
    }
    print_results(full, "MatrixC");
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long n = std::strtoull(value, &end, 10);
            if (errno || end == value || *end || *value == '-' || !n || n > INT_MAX ||
                n > std::numeric_limits<size_t>::max() / sizeof(double) / n) {
                if (!rank) std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = size_t(n);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int ranks, localRank, localRanks;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_Comm local;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    mpiCheck(MPI_Comm_rank(local, &localRank));
    mpiCheck(MPI_Comm_size(local, &localRanks));
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA GPU is required on every participating node.");
    // Also works with launcher-provided per-rank CUDA_VISIBLE_DEVICES masks.
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&local));
    // Respect OMP_NUM_THREADS. Otherwise avoid each local rank spawning a
    // machine-sized team; omp_get_num_procs also respects launcher affinity.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / std::min(localRanks, omp_get_num_procs()))));

    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(ranks, 2, dims));
    const Block block = blockFor(rank, dims, N);
    if (!rank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    std::vector<double> C(block.rows * block.cols);
    Multiply multiply(block, N);
    if (!rank) std::printf("Computing matrix multiplication...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    // Includes streamed initialization, transfers, and all GPU work.
    multiply.run(C);
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / std::max(seconds, 1e-12) / 1e9);
    }
    if (printResults) printDistributed(C, N, ranks, dims);
    if (validate) {
        if (!rank) std::printf("Validating result...\n");
        const bool valid = validateResult(C, block, N);
        if (!rank) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required.");
    int result = 1;
    try {
        result = benchmark(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    mpiCheck(MPI_Finalize());
    return result;
}
