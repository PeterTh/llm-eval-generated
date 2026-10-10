#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

namespace {

int worldRank = 0;

void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
        std::abort();
    }
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Keep the original unsigned arithmetic and double-precision initialization.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Block {
    size_t row, col, rows, cols;
};

// A balanced two-dimensional grid reduces input replication to O(N^2/sqrt(P))
// per GPU. Remainders, including grids larger than N, need no padding.
Block blockFor(int worker, const int dims[2], size_t N) {
    const size_t r = static_cast<size_t>(worker / dims[1]);
    const size_t c = static_cast<size_t>(worker % dims[1]);
    const size_t nr = static_cast<size_t>(dims[0]);
    const size_t nc = static_cast<size_t>(dims[1]);
    return {r * (N / nr) + std::min(r, N % nr),
            c * (N / nc) + std::min(c, N % nc),
            N / nr + (r < N % nr), N / nc + (c < N % nc)};
}

struct GpuTile {
    int device;
    Block block;
    double *A = nullptr, *B = nullptr, *C = nullptr;
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    std::string error;

    GpuTile(int deviceId, Block ownedBlock) : device(deviceId), block(ownedBlock) {}
    GpuTile(const GpuTile&) = delete;
    GpuTile& operator=(const GpuTile&) = delete;

    bool empty() const { return block.rows == 0 || block.cols == 0; }

    void allocate(size_t N) {
        if (empty()) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        blasCheck(cublasCreate(&handle), "cublasCreate");
        blasCheck(cublasSetStream(handle, stream), "cublasSetStream");
        const size_t aBytes = block.rows * N * sizeof(double);
        const size_t bBytes = N * block.cols * sizeof(double);
        const size_t cBytes = block.rows * block.cols * sizeof(double);
        // Pinned staging buffers permit concurrent transfers on every GPU.
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&A), aBytes), "cudaMallocHost(A)");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&B), bBytes), "cudaMallocHost(B)");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&C), cBytes), "cudaMallocHost(C)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceA), aBytes), "cudaMalloc(A)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceB), bBytes), "cudaMalloc(B)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceC), cBytes), "cudaMalloc(C)");
    }

    void upload(size_t N) {
        if (empty()) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaMemcpyAsync(deviceA, A, block.rows * N * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "upload A");
        cudaCheck(cudaMemcpyAsync(deviceB, B, N * block.cols * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "upload B");
        // Initialize cuBLAS's lazily loaded DGEMM machinery outside the timed
        // multiplication. This touches only one output element, not the full job.
        const double alpha = 1.0, beta = 0.0;
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha,
                              deviceB, static_cast<int>(block.cols),
                              deviceA, static_cast<int>(N), &beta,
                              deviceC, static_cast<int>(block.cols)), "initialize DGEMM");
        cudaCheck(cudaStreamSynchronize(stream), "input synchronization");
        // Inputs are no longer needed on the CPU: validation regenerates them.
        cudaCheck(cudaFreeHost(A), "cudaFreeHost(A)");
        A = nullptr;
        cudaCheck(cudaFreeHost(B), "cudaFreeHost(B)");
        B = nullptr;
    }

    void multiply(size_t N) {
        if (empty()) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T; no transpose or
        // packing kernel is necessary. DGEMM retains full FP64 precision.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(block.cols), static_cast<int>(block.rows),
                              static_cast<int>(N), &alpha,
                              deviceB, static_cast<int>(block.cols),
                              deviceA, static_cast<int>(N), &beta,
                              deviceC, static_cast<int>(block.cols)), "cublasDgemm");
        cudaCheck(cudaMemcpyAsync(C, deviceC, block.rows * block.cols * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "download C");
        cudaCheck(cudaStreamSynchronize(stream), "multiplication synchronization");
    }

    ~GpuTile() {
        if (empty()) return;
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (handle) cublasDestroy(handle);
        if (deviceA) cudaFree(deviceA);
        if (deviceB) cudaFree(deviceB);
        if (deviceC) cudaFree(deviceC);
        if (A) cudaFreeHost(A);
        if (B) cudaFreeHost(B);
        if (C) cudaFreeHost(C);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Only the main thread calls MPI (MPI_THREAD_FUNNELED). Worker exceptions are
// collected after joining the OpenMP team, before entering any MPI collective.
template <typename Operation>
void onGpus(std::vector<std::unique_ptr<GpuTile>>& tiles, Operation operation) {
    const int count = static_cast<int>(tiles.size());
    #pragma omp parallel for schedule(static) num_threads(count)
    for (int t = 0; t < count; ++t) {
        try {
            operation(*tiles[t]);
        } catch (const std::exception& e) {
            tiles[t]->error = e.what();
        }
    }
    for (const auto& tile : tiles) {
        if (!tile->error.empty()) {
            throw std::runtime_error("GPU " + std::to_string(tile->device) + ": " + tile->error);
        }
    }
}

void initMatrix(double* matrix, size_t rows, size_t cols,
                size_t rowBegin, size_t colBegin, size_t N) {
    // Avoid starting a large CPU team for very small tiles.
    const int threads = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(omp_get_max_threads()), std::max<size_t>(1, rows * cols / 32768)));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i * cols + j] = getPseudoRndValue(N, rowBegin + i, colBegin + j);
        }
    }
}

bool validateResult(const std::vector<std::unique_ptr<GpuTile>>& tiles, size_t N) {
    int failed = 0;
    // These are exactly the original 5x5 check positions, including repeated
    // positions for N < 5. Each check is performed by its output tile's owner.
    #pragma omp parallel for reduction(|:failed) schedule(static) num_threads(std::min(25, omp_get_max_threads()))
    for (int p = 0; p < 25; ++p) {
        const size_t i = static_cast<size_t>(p / 5) % N;
        const size_t j = static_cast<size_t>(p % 5) % N;
        for (const auto& tile : tiles) {
            const Block& b = tile->block;
            if (i < b.row || i >= b.row + b.rows || j < b.col || j >= b.col + b.cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = tile->C[(i - b.row) * b.cols + j - b.col];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                failed = 1;
            }
        }
    }
    int anyFailed = 0;
    mpiCheck(MPI_Allreduce(&failed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
             "validation reduction");
    return anyFailed == 0;
}

// Derived datatypes place each tile directly into the original row-major C.
// A count of one also avoids MPI's INT_MAX limit on element counts.
MPI_Datatype tileType(const Block& b, size_t stride) {
    MPI_Datatype type;
    mpiCheck(MPI_Type_vector(static_cast<int>(b.rows), static_cast<int>(b.cols),
                             static_cast<int>(stride), MPI_DOUBLE, &type), "MPI_Type_vector");
    mpiCheck(MPI_Type_commit(&type), "MPI_Type_commit");
    return type;
}

void printDistributedResults(const std::vector<std::unique_ptr<GpuTile>>& tiles,
                             const std::vector<int>& workerCounts, const int dims[2], size_t N) {
    constexpr int resultTag = 0;
    if (worldRank != 0) {
        for (const auto& tile : tiles) {
            if (tile->empty()) continue;
            MPI_Datatype type = tileType(tile->block, tile->block.cols);
            mpiCheck(MPI_Send(tile->C, 1, type, 0, resultTag, MPI_COMM_WORLD), "send result tile");
            mpiCheck(MPI_Type_free(&type), "MPI_Type_free");
        }
        return;
    }

    std::vector<double> C(N * N);
    for (const auto& tile : tiles) {
        const Block& b = tile->block;
        if (tile->empty()) continue;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < b.rows; ++i) {
            std::memcpy(C.data() + (b.row + i) * N + b.col,
                        tile->C + i * b.cols, b.cols * sizeof(double));
        }
    }
    int worker = workerCounts[0];
    for (size_t rank = 1; rank < workerCounts.size(); ++rank) {
        for (int t = 0; t < workerCounts[rank]; ++t, ++worker) {
            const Block b = blockFor(worker, dims, N);
            if (b.rows == 0 || b.cols == 0) continue;
            MPI_Datatype type = tileType(b, N);
            mpiCheck(MPI_Recv(C.data() + b.row * N + b.col, 1, type, static_cast<int>(rank),
                              resultTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE), "receive result tile");
            mpiCheck(MPI_Type_free(&type), "MPI_Type_free");
        }
    }
    print_results(C, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int worldSize) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                if (worldRank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
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

    MPI_Comm nodeComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                 MPI_INFO_NULL, &nodeComm), "node communicator");
    int localRank = 0, localSize = 1;
    mpiCheck(MPI_Comm_rank(nodeComm, &localRank), "local rank");
    mpiCheck(MPI_Comm_size(nodeComm, &localSize), "local size");
    mpiCheck(MPI_Comm_free(&nodeComm), "MPI_Comm_free");
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) throw std::runtime_error("a CUDA GPU is required on every MPI rank");

    // Use all visible GPUs even with one MPI rank per node. With more ranks,
    // assign disjoint device sets when possible. CUDA_VISIBLE_DEVICES is honored.
    std::vector<int> devices;
    if (localSize <= deviceCount) {
        for (int device = localRank; device < deviceCount; device += localSize) {
            devices.push_back(device);
        }
    } else {
        devices.push_back(localRank % deviceCount);
    }
    const int localWorkers = static_cast<int>(devices.size());
    std::vector<int> workerCounts(worldSize);
    mpiCheck(MPI_Allgather(&localWorkers, 1, MPI_INT, workerCounts.data(), 1, MPI_INT,
                           MPI_COMM_WORLD), "GPU worker counts");
    int firstWorker = 0, totalWorkers = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        if (workerCounts[rank] > INT_MAX - totalWorkers) {
            throw std::runtime_error("too many GPU workers for the MPI process grid");
        }
        if (rank < worldRank) firstWorker += workerCounts[rank];
        totalWorkers += workerCounts[rank];
    }
    int dims[2] = {0, 0};
    mpiCheck(MPI_Dims_create(totalWorkers, 2, dims), "GPU process grid");

    if (worldRank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<std::unique_ptr<GpuTile>> tiles;
    for (int t = 0; t < localWorkers; ++t) {
        tiles.emplace_back(std::make_unique<GpuTile>(devices[t], blockFor(firstWorker + t, dims, N)));
    }
    onGpus(tiles, [N](GpuTile& tile) { tile.allocate(N); });
    // The original matrices are deterministic. Regenerate just the required
    // rows of A and columns of B locally, avoiding an input broadcast entirely.
    for (auto& tile : tiles) {
        if (tile->empty()) continue;
        const Block& b = tile->block;
        initMatrix(tile->A, b.rows, N, b.row, 0, N);
        initMatrix(tile->B, N, b.cols, 0, b.col, N);
    }
    onGpus(tiles, [N](GpuTile& tile) { tile.upload(N); });

    if (worldRank == 0) printf("Computing matrix multiplication...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "benchmark barrier");
    const double start = MPI_Wtime();
    onGpus(tiles, [N](GpuTile& tile) { tile.multiply(N); });
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "benchmark timing reduction");
    if (worldRank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double gflops = seconds > 0.0 ? 2.0 * N * N * N / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) printDistributedResults(tiles, workerCounts, dims, N);
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        const bool valid = validateResult(tiles, N);
        if (worldRank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    const int status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI error handler");
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "world rank");
    int worldSize = 1;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "world size");
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        }
        result = runBenchmark(argc, argv, worldSize);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (MPI_Finalize() != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI_Finalize failed\n", worldRank);
        return 1;
    }
    return result;
}
