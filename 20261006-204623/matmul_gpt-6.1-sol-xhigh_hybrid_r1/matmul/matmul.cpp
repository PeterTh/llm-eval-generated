#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Keep the original size_t arithmetic, including its wraparound behavior.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
    }
}

void checkBlas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
    }
}

void checkMpi(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string("MPI: ") + std::string(message, length));
    }
}

// Match physical GPU identities across node-local ranks. Ordinals alone are
// insufficient when CUDA_VISIBLE_DEVICES gives each rank a different subset
// or ordering. Each visible accelerator gets exactly one worker on its node.
std::vector<int> assignDevices(MPI_Comm node, int localRank, int localRanks, int devices) {
    static_assert(sizeof(cudaUUID_t) == 16);
    std::vector<cudaUUID_t> localIds(devices);
    for (int device = 0; device < devices; ++device) {
        cudaDeviceProp properties;
        checkCuda(cudaGetDeviceProperties(&properties, device));
        localIds[device] = properties.uuid;
    }
    const int localBytes = devices * static_cast<int>(sizeof(cudaUUID_t));
    std::vector<int> counts(localRanks), offsets(localRanks);
    checkMpi(MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, node));
    int bytes = 0;
    for (int i = 0; i < localRanks; ++i) {
        if (counts[i] > INT_MAX - bytes) throw std::runtime_error("Too many node-local GPUs");
        offsets[i] = bytes;
        bytes += counts[i];
    }
    std::vector<cudaUUID_t> allIds(bytes / sizeof(cudaUUID_t));
    checkMpi(MPI_Allgatherv(localIds.data(), localBytes, MPI_BYTE, allIds.data(),
                            counts.data(), offsets.data(), MPI_BYTE, node));
    const auto equal = [](const cudaUUID_t& a, const cudaUUID_t& b) {
        return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
    };
    std::vector<cudaUUID_t> assignedIds;
    std::vector<int> loads(localRanks, 0), owned;
    for (const auto& id : allIds) {
        if (std::any_of(assignedIds.begin(), assignedIds.end(),
                        [&](const cudaUUID_t& previous) { return equal(id, previous); })) continue;
        assignedIds.push_back(id);
        int owner = -1;
        for (int candidate = 0; candidate < localRanks; ++candidate) {
            const size_t first = offsets[candidate] / sizeof(cudaUUID_t);
            const size_t last = first + counts[candidate] / sizeof(cudaUUID_t);
            const bool visible = std::any_of(allIds.begin() + first, allIds.begin() + last,
                                             [&](const cudaUUID_t& other) { return equal(id, other); });
            if (visible && (owner < 0 || loads[candidate] < loads[owner] ||
                            (loads[candidate] == loads[owner] && counts[candidate] < counts[owner]))) {
                owner = candidate;
            }
        }
        ++loads[owner];
        if (owner == localRank) {
            const auto found = std::find_if(localIds.begin(), localIds.end(),
                                            [&](const cudaUUID_t& local) { return equal(id, local); });
            owned.push_back(static_cast<int>(found - localIds.begin()));
        }
    }
    return owned;
}

struct Block {
    size_t row, col, rows, cols;
    bool empty() const { return rows == 0 || cols == 0; }
};

// A balanced 2D grid minimizes replicated input storage per accelerator.
Block getBlock(int id, const int* grid, size_t N) {
    const auto begin = [N](int coordinate, int parts) {
        return (N / parts) * coordinate + std::min(N % parts, static_cast<size_t>(coordinate));
    };
    const int row = id / grid[1];
    const int col = id % grid[1];
    const size_t firstRow = begin(row, grid[0]);
    const size_t firstCol = begin(col, grid[1]);
    return {firstRow, firstCol, begin(row + 1, grid[0]) - firstRow,
            begin(col + 1, grid[1]) - firstCol};
}

struct GpuWorker {
    int device = 0;
    Block block{};
    double *A = nullptr, *B = nullptr, *C = nullptr;
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    std::string error;

    GpuWorker() = default;
    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;

    ~GpuWorker() {
        if (!handle && !stream && !A && !B && !C && !deviceA && !deviceB && !deviceC) return;
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

    void allocate(size_t N, bool needHostResult) {
        if (block.empty()) return;
        const size_t bytesA = block.rows * N * sizeof(double);
        const size_t bytesB = N * block.cols * sizeof(double);
        const size_t bytesC = block.rows * block.cols * sizeof(double);
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checkBlas(cublasCreate(&handle));
        checkBlas(cublasSetStream(handle, stream));
        // DGEMM retains double precision without enabling reduced precision math.
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytesA));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytesB));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytesC));
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&A), bytesA));
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&B), bytesB));
        if (needHostResult) checkCuda(cudaMallocHost(reinterpret_cast<void**>(&C), bytesC));
    }

    void upload(size_t N) {
        if (block.empty()) return;
        checkCuda(cudaMemcpyAsync(deviceA, A, block.rows * N * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
        checkCuda(cudaMemcpyAsync(deviceB, B, N * block.cols * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
        checkCuda(cudaStreamSynchronize(stream));
    }

    void multiply(size_t N) {
        if (block.empty()) return;
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T: no transpose copies.
        checkBlas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(block.cols), static_cast<int>(block.rows),
                             static_cast<int>(N), &alpha, deviceB, static_cast<int>(block.cols),
                             deviceA, static_cast<int>(N), &beta, deviceC,
                             static_cast<int>(block.cols)));
        checkCuda(cudaStreamSynchronize(stream));
    }

    void download() {
        if (block.empty()) return;
        checkCuda(cudaMemcpyAsync(C, deviceC, block.rows * block.cols * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
    }
};

// Only the main thread calls MPI (MPI_THREAD_FUNNELED). Propagate GPU errors out
// of the OpenMP region so it can abort the whole communicator safely.
template <typename Function>
void forEachGpu(std::vector<GpuWorker>& workers, Function function) {
    const int count = static_cast<int>(workers.size());
    if (count == 0) return;
    #pragma omp parallel for num_threads(count) schedule(static, 1)
    for (int i = 0; i < count; ++i) {
        try {
            checkCuda(cudaSetDevice(workers[i].device));
            function(workers[i]);
        } catch (const std::exception& error) {
            workers[i].error = error.what();
        }
    }
    for (const auto& worker : workers) {
        if (!worker.error.empty()) {
            throw std::runtime_error("GPU " + std::to_string(worker.device) + ": " + worker.error);
        }
    }
}

void initMatrices(std::vector<GpuWorker>& workers, size_t N) {
    size_t elements = 0;
    for (const auto& worker : workers) {
        if (!worker.block.empty()) elements += (worker.block.rows + worker.block.cols) * N;
    }
    // Avoid launching a large CPU team for very small inputs; honor OMP_NUM_THREADS.
    const int threads = static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
                                                 std::max(size_t{1}, elements / 65536)));
    #pragma omp parallel num_threads(threads)
    {
        // All threads visit blocks in the same order and share each row loop.
        for (auto& worker : workers) {
            if (worker.block.empty()) continue;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < worker.block.rows; ++i) {
                for (size_t k = 0; k < N; ++k) {
                    worker.A[i * N + k] = getPseudoRndValue(N, worker.block.row + i, k);
                }
            }
            #pragma omp for schedule(static)
            for (size_t k = 0; k < N; ++k) {
                for (size_t j = 0; j < worker.block.cols; ++j) {
                    worker.B[k * worker.block.cols + j] = getPseudoRndValue(N, k, worker.block.col + j);
                }
            }
        }
    }
}

// Validate samples in every distributed block, including the original leading
// 5x5 samples. Host input slices avoid recomputing the pseudo-random sequence.
bool validateResult(const GpuWorker& worker, size_t N) {
    if (worker.block.empty()) return true;
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % worker.block.rows;
            const size_t j = pj % worker.block.cols;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += worker.A[i * N + k] * worker.B[k * worker.block.cols + j];
            }
            const double actual = worker.C[i * worker.block.cols + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        worker.block.row + i, worker.block.col + j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

// Assemble only for -r. MPI subarray types place blocks directly into the
// original row-major order without a packed receive buffer or int-sized counts.
std::vector<double> gatherResult(const std::vector<GpuWorker>& workers, size_t N,
                                 const int* grid, const std::vector<int>& counts, int rank) {
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
        std::vector<MPI_Request> requests;
        int id = 0;
        for (size_t owner = 0; owner < counts.size(); ++owner) {
            for (int local = 0; local < counts[owner]; ++local, ++id) {
                const Block block = getBlock(id, grid, N);
                if (block.empty()) continue;
                if (owner == 0) {
                    const double* source = workers[local].C;
                    const int copyThreads = static_cast<int>(std::min(
                        static_cast<size_t>(omp_get_max_threads()),
                        std::max(size_t{1}, block.rows * block.cols / 262144)));
                    #pragma omp parallel for num_threads(copyThreads) schedule(static)
                    for (size_t i = 0; i < block.rows; ++i) {
                        std::memcpy(C.data() + (block.row + i) * N + block.col,
                                    source + i * block.cols, block.cols * sizeof(double));
                    }
                } else {
                    const int sizes[] = {static_cast<int>(N), static_cast<int>(N)};
                    const int subsizes[] = {static_cast<int>(block.rows), static_cast<int>(block.cols)};
                    const int starts[] = {static_cast<int>(block.row), static_cast<int>(block.col)};
                    MPI_Datatype type;
                    checkMpi(MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                                      MPI_ORDER_C, MPI_DOUBLE, &type));
                    checkMpi(MPI_Type_commit(&type));
                    MPI_Request request;
                    checkMpi(MPI_Irecv(C.data(), 1, type, static_cast<int>(owner), 0,
                                      MPI_COMM_WORLD, &request));
                    requests.push_back(request);
                    checkMpi(MPI_Type_free(&type));
                }
            }
        }
        if (!requests.empty()) {
            checkMpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        }
    } else {
        for (const auto& worker : workers) {
            if (worker.block.empty()) continue;
            MPI_Datatype type;
            checkMpi(MPI_Type_vector(static_cast<int>(worker.block.rows),
                                     static_cast<int>(worker.block.cols),
                                     static_cast<int>(worker.block.cols), MPI_DOUBLE, &type));
            checkMpi(MPI_Type_commit(&type));
            checkMpi(MPI_Send(worker.C, 1, type, 0, 0, MPI_COMM_WORLD));
            checkMpi(MPI_Type_free(&type));
        }
    }
    return C;
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
            char* end = nullptr;
            errno = 0;
            const char* argument = argv[++i];
            const unsigned long long value = std::strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || end == argument || *end != '\0' ||
                value == 0 || value > INT_MAX ||
                value > std::numeric_limits<size_t>::max() / sizeof(double) / value) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            N = static_cast<size_t>(value);
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

    MPI_Comm node;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 0;
    checkMpi(MPI_Comm_rank(node, &localRank));
    checkMpi(MPI_Comm_size(node, &localRanks));

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("At least one CUDA device is required on every rank");
    const auto ownedDevices = assignDevices(node, localRank, localRanks, devices);
    checkMpi(MPI_Comm_free(&node));
    const int workerCount = static_cast<int>(ownedDevices.size());
    std::vector<int> counts(ranks);
    checkMpi(MPI_Allgather(&workerCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    int totalWorkers = 0, firstWorker = 0;
    for (int i = 0; i < ranks; ++i) {
        if (counts[i] > INT_MAX - totalWorkers) throw std::runtime_error("Too many GPU workers");
        if (i < rank) firstWorker += counts[i];
        totalWorkers += counts[i];
    }
    int grid[2] = {0, 0};
    checkMpi(MPI_Dims_create(totalWorkers, 2, grid));

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA workers: %d (%d x %d blocks)\n",
               ranks, totalWorkers, grid[0], grid[1]);
        printf("Initializing matrices...\n");
    }
    std::vector<GpuWorker> workers(workerCount);
    for (int i = 0; i < workerCount; ++i) {
        workers[i].device = ownedDevices[i];
        workers[i].block = getBlock(firstWorker + i, grid, N);
    }
    forEachGpu(workers, [N, validate, printResults](GpuWorker& worker) {
        worker.allocate(N, validate || printResults);
    });
    initMatrices(workers, N);
    forEachGpu(workers, [N](GpuWorker& worker) { worker.upload(N); });

    if (rank == 0) printf("Computing matrix multiplication...\n");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    forEachGpu(workers, [N](GpuWorker& worker) { worker.multiply(N); });
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    checkMpi(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", seconds * 1000.0);
        const double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (validate || printResults) {
        forEachGpu(workers, [](GpuWorker& worker) { worker.download(); });
    }
    if (printResults) {
        const auto C = gatherResult(workers, N, grid, counts, rank);
        if (rank == 0) print_results(C, "MatrixC");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = 1;
        #pragma omp parallel for num_threads(std::max(1, workerCount)) reduction(&:valid)
        for (int i = 0; i < workerCount; ++i) {
            valid &= validateResult(workers[i], N) ? 1 : 0;
        }
        int allValid = 0;
        checkMpi(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    MPI_Finalize();
    return result;
}
