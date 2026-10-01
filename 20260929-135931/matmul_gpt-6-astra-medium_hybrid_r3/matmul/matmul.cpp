#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
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
        std::fprintf(stderr, "cuBLAS status: %d\n", static_cast<int>(status));
        fail("cuBLAS operation failed");
    }
}

// Preserve the original unsigned integer arithmetic and double precision inputs.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

// Balanced, contiguous partitions, including empty partitions when ranks > N.
struct Range { size_t begin, size; };
Range partition(size_t n, int parts, int index) {
    const size_t base = n / parts, extra = n % parts;
    return {base * index + std::min(extra, static_cast<size_t>(index)),
            base + (static_cast<size_t>(index) < extra)};
}

struct Buffer {
    double* host = nullptr;
    double* device = nullptr;
    explicit Buffer(size_t count) {
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

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Ordered, chunked messages avoid MPI's int count/displacement limit. Gathering
// is needed only for the original ordered result statistics and byte hash.
void gatherResults(const double* local, size_t N, const int dims[2], int ranks) {
    std::vector<double> full, scratch;
    if (worldRank == 0) {
        full.resize(N * N);
        scratch.resize(partition(N, dims[0], 0).size * partition(N, dims[1], 0).size);
    }
    constexpr size_t chunkLimit = 1u << 26;
    for (int rank = 0; rank < ranks; ++rank) {
        Range rr = partition(N, dims[0], rank / dims[1]);
        Range cc = partition(N, dims[1], rank % dims[1]);
        const size_t count = rr.size * cc.size;
        if (rank != 0 && worldRank == rank) {
            for (size_t offset = 0; offset < count; offset += chunkLimit)
                mpiCheck(MPI_Send(local + offset, static_cast<int>(std::min(chunkLimit, count - offset)),
                                  MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        }
        if (worldRank == 0 && count) {
            const double* source = local;
            if (rank != 0) {
                for (size_t offset = 0; offset < count; offset += chunkLimit)
                    mpiCheck(MPI_Recv(scratch.data() + offset,
                                      static_cast<int>(std::min(chunkLimit, count - offset)),
                                      MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                source = scratch.data();
            }
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rr.size; ++i)
                std::memcpy(full.data() + (rr.begin + i) * N + cc.begin,
                            source + i * cc.size, cc.size * sizeof(double));
        }
    }
    if (worldRank == 0) print_results(full, "MatrixC");
}

bool validateResult(const double* C, size_t N, Range rows, Range cols) {
    int bad = 0;
    // Preserve the original 25 checks and also sample every distributed block,
    // so indexing errors away from the upper-left corner are detected.
    #pragma omp parallel for collapse(2) reduction(|:bad) schedule(static)
    for (int pi = 0; pi < 7; ++pi) {
        for (int pj = 0; pj < 7; ++pj) {
            if (!rows.size || !cols.size) continue;
            const size_t i = pi < 5 ? static_cast<size_t>(pi) % N :
                             rows.begin + (pi == 5 ? 0 : rows.size - 1);
            const size_t j = pj < 5 ? static_cast<size_t>(pj) % N :
                             cols.begin + (pj == 5 ? 0 : cols.size - 1);
            if (i < rows.begin || i >= rows.begin + rows.size ||
                j < cols.begin || j >= cols.begin + cols.size) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[(i - rows.begin) * cols.size + j - cols.begin];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                #pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, error);
                bad = 1;
            }
        }
    }
    int globalBad = 0;
    mpiCheck(MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    return globalBad == 0;
}

int run(int argc, char** argv) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end || parsed == 0 ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (worldRank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int ranks = 0, localRank = 0, localSize = 0;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_Comm shared;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &shared));
    mpiCheck(MPI_Comm_rank(shared, &localRank));
    mpiCheck(MPI_Comm_size(shared, &localSize));
    mpiCheck(MPI_Comm_free(&shared));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("A CUDA GPU is required on every MPI rank");
    // Also supports schedulers exposing one GPU per rank through CUDA_VISIBLE_DEVICES.
    cudaCheck(cudaSetDevice(localRank % devices));
    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(ranks, 2, dims));
    const Range rows = partition(N, dims[0], worldRank / dims[1]);
    const Range cols = partition(N, dims[1], worldRank % dims[1]);
    const bool active = rows.size && cols.size;

    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                    N, N, validate ? "enabled" : "disabled");
        std::printf("MPI process grid: %d x %d\nInitializing matrices...\n", dims[0], dims[1]);
    }
    // 2-D decomposition uses O(N^2/sqrt(P)) input memory per rank instead of
    // replicating an entire operand. Deterministic initialization needs no MPI traffic.
    Buffer A(active ? rows.size * N : 0), B(active ? N * cols.size : 0),
           C(rows.size * cols.size);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < (active ? rows.size : 0); ++i)
            for (size_t k = 0; k < N; ++k)
                A.host[i * N + k] = getPseudoRndValue(N, rows.begin + i, k);
        #pragma omp for schedule(static)
        for (size_t k = 0; k < (active ? N : 0); ++k)
            for (size_t j = 0; j < cols.size; ++j)
                B.host[k * cols.size + j] = getPseudoRndValue(N, k, cols.begin + j);
    }
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    blasCheck(cublasSetStream(handle, stream));
    blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    if (worldRank == 0) std::printf("Computing matrix multiplication...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (active) {
        cudaCheck(cudaMemcpyAsync(A.device, A.host, rows.size * N * sizeof(double), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(B.device, B.host, N * cols.size * sizeof(double), cudaMemcpyHostToDevice, stream));
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A B is column-major C^T = B^T A^T. No transposes or packing.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(cols.size), static_cast<int>(rows.size), static_cast<int>(N),
                             &alpha, B.device, static_cast<int>(cols.size),
                             A.device, static_cast<int>(N), &beta, C.device, static_cast<int>(cols.size)));
        cudaCheck(cudaMemcpyAsync(C.host, C.device, rows.size * cols.size * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / seconds / 1e9);
    }
    if (printResults) gatherResults(C.host, N, dims, ranks);
    bool valid = true;
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        valid = validateResult(C.host, N, rows, cols);
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    blasCheck(cublasDestroy(handle));
    cudaCheck(cudaStreamDestroy(stream));
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fail(error.what());
    }
    mpiCheck(MPI_Finalize());
    return result;
}
