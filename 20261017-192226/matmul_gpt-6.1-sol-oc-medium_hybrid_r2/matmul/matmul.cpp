#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Preserve the original unsigned arithmetic and global matrix coordinates.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(status));
}

// Pinned staging buffers allow asynchronous DMA without driver-side copies.
struct HostBuffer {
    double* data = nullptr;
    explicit HostBuffer(size_t elements) {
        if (elements) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), elements * sizeof(double)));
    }
    ~HostBuffer() { if (data) cudaFreeHost(data); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

struct DeviceWork {
    double *a = nullptr, *b = nullptr, *c = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    ~DeviceWork() {
        if (stream) cudaStreamSynchronize(stream);
        if (handle) cublasDestroy(handle);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (c) cudaFree(c);
        if (stream) cudaStreamDestroy(stream);
    }
};

size_t boundary(size_t n, int part, int parts) {
    return n / parts * part + std::min<size_t>(part, n % parts);
}

// Each OpenMP worker owns one GPU and a disjoint band of the rank's C tile.
// MPI is only called by the main thread (MPI_THREAD_FUNNELED).
void matrixMultiply(const double* a, const double* b, double* c,
                    size_t rows, size_t cols, size_t n,
                    const std::vector<int>& devices) {
    std::exception_ptr failure;
    const int workers = static_cast<int>(devices.size());
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int worker = 0; worker < workers; ++worker) {
        try {
            cudaCheck(cudaSetDevice(devices[worker]));
            const size_t begin = boundary(rows, worker, workers);
            const size_t count = boundary(rows, worker + 1, workers) - begin;
            if (!count || !cols) continue;
            DeviceWork work;
            cudaCheck(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking));
            blasCheck(cublasCreate(&work.handle));
            blasCheck(cublasSetStream(work.handle, work.stream));
            blasCheck(cublasSetMathMode(work.handle, CUBLAS_DEFAULT_MATH));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.a), count * n * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.b), n * cols * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&work.c), count * cols * sizeof(double)));
            cudaCheck(cudaMemcpyAsync(work.a, a + begin * n, count * n * sizeof(double),
                                      cudaMemcpyHostToDevice, work.stream));
            cudaCheck(cudaMemcpyAsync(work.b, b, n * cols * sizeof(double),
                                      cudaMemcpyHostToDevice, work.stream));
            const double alpha = 1.0, beta = 0.0;
            // Row-major C = A B is column-major C^T = B^T A^T; no transpose copies.
            blasCheck(cublasDgemm(work.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(cols), static_cast<int>(count), static_cast<int>(n),
                                 &alpha, work.b, static_cast<int>(cols),
                                 work.a, static_cast<int>(n), &beta, work.c, static_cast<int>(cols)));
            cudaCheck(cudaMemcpyAsync(c + begin * cols, work.c, count * cols * sizeof(double),
                                      cudaMemcpyDeviceToHost, work.stream));
            cudaCheck(cudaStreamSynchronize(work.stream));
        } catch (...) {
            #pragma omp critical
            { if (!failure) failure = std::current_exception(); }
        }
    }
    if (failure) std::rethrow_exception(failure);
}

bool validateResult(const double* c, size_t n, size_t row0, size_t col0,
                    size_t rows, size_t cols) {
    if (!rows || !cols) return true;
    // Include the original 5x5 checks and tile boundaries on every rank.
    for (int pi = 0; pi < 7; ++pi) {
        const size_t i = pi < 5 ? static_cast<size_t>(pi) % n : row0 + (pi == 6 ? rows - 1 : 0);
        if (i < row0 || i >= row0 + rows) continue;
        for (int pj = 0; pj < 7; ++pj) {
            const size_t j = pj < 5 ? static_cast<size_t>(pj) % n : col0 + (pj == 6 ? cols - 1 : 0);
            if (j < col0 || j >= col0 + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = c[(i - row0) * cols + j - col0];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

// Gather only for -r. Chunked vector receives place packed rank tiles directly
// into the root's row-major matrix, without an extra full-size staging matrix.
void printDistributedResult(const double* local, size_t n, int rank, int size, const int dims[2]) {
    std::vector<double> result;
    if (rank == 0) result.resize(n * n);
    for (int source = 0; source < size; ++source) {
        const size_t r0 = boundary(n, source / dims[1], dims[0]);
        const size_t rows = boundary(n, source / dims[1] + 1, dims[0]) - r0;
        const size_t c0 = boundary(n, source % dims[1], dims[1]);
        const size_t cols = boundary(n, source % dims[1] + 1, dims[1]) - c0;
        if (!rows || !cols) continue;
        const size_t chunkRows = std::max<size_t>(1, INT_MAX / cols);
        for (size_t first = 0; first < rows; first += chunkRows) {
            const int count = static_cast<int>(std::min(chunkRows, rows - first));
            if (rank == 0 && source == 0) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < count; ++i)
                    std::copy_n(local + (first + i) * cols, cols,
                                result.data() + (r0 + first + i) * n + c0);
            } else if (rank == 0) {
                MPI_Datatype tile;
                MPI_Type_vector(count, static_cast<int>(cols), static_cast<int>(n), MPI_DOUBLE, &tile);
                MPI_Type_commit(&tile);
                MPI_Recv(result.data() + (r0 + first) * n + c0, 1, tile, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&tile);
            } else if (rank == source) {
                MPI_Send(local + first * cols, static_cast<int>(count * cols), MPI_DOUBLE,
                         0, 0, MPI_COMM_WORLD);
            }
        }
    }
    if (rank == 0) print_results(result, "MatrixC");
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int size) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || *value == '-' || end == value || *end || !parsed || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localSize, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) throw std::runtime_error("MPI rank has no visible CUDA device");
    std::vector<int> devices;
    // Works both with all GPUs visible and launcher-provided per-rank GPU masks.
    for (int d = localRank; d < deviceCount; d += localSize) devices.push_back(d);
    if (devices.empty()) devices.push_back(localRank % deviceCount);
    for (int d : devices) { cudaCheck(cudaSetDevice(d)); cudaCheck(cudaFree(nullptr)); }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    const size_t row0 = boundary(n, rank / dims[1], dims[0]);
    const size_t rows = boundary(n, rank / dims[1] + 1, dims[0]) - row0;
    const size_t col0 = boundary(n, rank % dims[1], dims[1]);
    const size_t cols = boundary(n, rank % dims[1] + 1, dims[1]) - col0;
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI process grid: %d x %d; OpenMP GPU workers on rank 0: %zu\n",
               dims[0], dims[1], devices.size());
        printf("Initializing matrices...\n");
    }
    HostBuffer a(rows && cols ? rows * n : 0), b(rows && cols ? n * cols : 0), c(rows * cols);
    if (rows && cols) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t k = 0; k < n; ++k) a.data[i * n + k] = getPseudoRndValue(n, row0 + i, k);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < n; ++k)
            for (size_t j = 0; j < cols; ++j) b.data[k * cols + j] = getPseudoRndValue(n, k, col0 + j);
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(a.data, b.data, c.data, rows, cols, n, devices);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", 2.0 * n * n * n / seconds / 1e9);
    }
    if (printResults) printDistributedResult(c.data, n, rank, size, dims);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(c.data, n, row0, col0, rows, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        status = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
