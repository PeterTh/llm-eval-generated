#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
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

int worldRank = 0;

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

void checkMPI(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fail(message);
    }
}

void checkCUDA(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

void checkBLAS(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[96];
        std::snprintf(message, sizeof(message), "cuBLAS failed with status %d", int(status));
        fail(message);
    }
}

// Keep the original unsigned arithmetic and deterministic initialization.
constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

int workerThreads(size_t elements) {
    // Small panels do not benefit from waking a large OpenMP team.
    const size_t useful = std::max(size_t{1}, (elements + 16383) / 16384);
    return static_cast<int>(std::min(size_t(omp_get_max_threads()), useful));
}

struct Block {
    int start;
    int size;
};

Block partition(int N, int coordinate, int parts) {
    return {coordinate * (N / parts) + std::min(coordinate, N % parts),
            N / parts + (coordinate < N % parts)};
}

struct Buffer {
    double* data = nullptr;
    bool host;

    Buffer(size_t elements, bool pinnedHost) : host(pinnedHost) {
        const size_t bytes = std::max(elements, size_t{1}) * sizeof(double);
        if (host) checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&data), bytes));
        else checkCUDA(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
    }
    ~Buffer() {
        if (host) cudaFreeHost(data);
        else cudaFree(data);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Accelerator {
    cudaStream_t upload = nullptr;
    cudaStream_t compute = nullptr;
    cublasHandle_t blas = nullptr;
    Buffer C;

    explicit Accelerator(size_t elements) : C(elements, false) {
        checkCUDA(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking));
        checkCUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        checkBLAS(cublasCreate(&blas));
        checkBLAS(cublasSetStream(blas, compute));
        // Dgemm uses double precision throughout, including accumulation.
        checkBLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }
    ~Accelerator() {
        cublasDestroy(blas);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(upload);
    }
};

struct Panel {
    Buffer hostA, hostB, deviceA, deviceB;
    cudaEvent_t uploaded = nullptr;
    cudaEvent_t consumed = nullptr;
    bool used = false;
    std::vector<MPI_Request> requests;

    Panel(int rows, int columns, int width)
        : hostA(size_t(rows) * width, true), hostB(size_t(width) * columns, true),
          deviceA(size_t(rows) * width, false), deviceB(size_t(width) * columns, false) {
        checkCUDA(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
        checkCUDA(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    }
    ~Panel() {
        cudaEventDestroy(consumed);
        cudaEventDestroy(uploaded);
    }
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;
};

void initializePanel(Panel& panel, int N, Block rows, Block columns, int kStart,
                     int width, bool ownsA, bool ownsB) {
    const int aRows = ownsA ? rows.size : 0;
    const int bRows = ownsB ? width : 0;
    const size_t elements = size_t(aRows) * width + size_t(bRows) * columns.size;
    // Only the broadcast owners generate data. No rank holds a full input matrix.
    #pragma omp parallel for schedule(static) num_threads(workerThreads(elements))
    for (int row = 0; row < aRows + bRows; ++row) {
        if (row < aRows) {
            for (int k = 0; k < width; ++k) {
                panel.hostA.data[size_t(row) * width + k] =
                    getPseudoRndValue(size_t(N), size_t(rows.start) + row,
                                     size_t(kStart) + k);
            }
        } else {
            const int k = row - aRows;
            for (int j = 0; j < columns.size; ++j) {
                panel.hostB.data[size_t(k) * columns.size + j] =
                    getPseudoRndValue(size_t(N), size_t(kStart) + k,
                                     size_t(columns.start) + j);
            }
        }
    }
}

void broadcast(double* data, size_t elements, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    // MPI-3 uses int counts; chunk panels without truncating large matrices.
    for (size_t offset = 0; offset < elements;) {
        const int count = static_cast<int>(std::min(elements - offset, size_t(INT_MAX)));
        MPI_Request request = MPI_REQUEST_NULL;
        checkMPI(MPI_Ibcast(data + offset, count, MPI_DOUBLE, root, comm, &request));
        requests.push_back(request);
        offset += size_t(count);
    }
}

void broadcastPanel(Panel& panel, Block rows, Block columns, int width,
                    int aOwner, int bOwner, MPI_Comm rowComm, MPI_Comm columnComm) {
    panel.requests.clear();
    broadcast(panel.hostA.data, size_t(rows.size) * width, aOwner, rowComm, panel.requests);
    broadcast(panel.hostB.data, size_t(width) * columns.size, bOwner, columnComm, panel.requests);
}

void multiplyPanel(Panel& panel, Accelerator& gpu, Block rows, Block columns,
                   int width, bool first) {
    if (!panel.requests.empty()) {
        checkMPI(MPI_Waitall(static_cast<int>(panel.requests.size()),
                             panel.requests.data(), MPI_STATUSES_IGNORE));
    }
    if (rows.size && columns.size) {
        checkCUDA(cudaMemcpyAsync(panel.deviceA.data, panel.hostA.data,
                                  size_t(rows.size) * width * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.upload));
        checkCUDA(cudaMemcpyAsync(panel.deviceB.data, panel.hostB.data,
                                  size_t(width) * columns.size * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.upload));
    }
    checkCUDA(cudaEventRecord(panel.uploaded, gpu.upload));
    checkCUDA(cudaStreamWaitEvent(gpu.compute, panel.uploaded, 0));
    if (rows.size && columns.size) {
        const double alpha = 1.0;
        const double beta = first ? 0.0 : 1.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T; no transposes or copies.
        checkBLAS(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             columns.size, rows.size, width, &alpha,
                             panel.deviceB.data, columns.size,
                             panel.deviceA.data, width, &beta,
                             gpu.C.data, columns.size));
    }
    checkCUDA(cudaEventRecord(panel.consumed, gpu.compute));
    panel.used = true;
}

bool validateResult(const std::vector<double>& C, int N, Block rows, Block columns) {
    int failed = 0;
    // Preserve the original 5x5 checks and also sample every distributed tile.
    #pragma omp parallel for schedule(static) reduction(|:failed) num_threads(workerThreads(size_t(N) * 34))
    for (int point = 0; point < 34; ++point) {
        int i, j;
        if (point < 25) {
            i = (point / 5) % N;
            j = (point % 5) % N;
        } else {
            if (!rows.size || !columns.size) continue;
            const int r[] = {0, rows.size / 2, rows.size - 1};
            const int c[] = {0, columns.size / 2, columns.size - 1};
            i = rows.start + r[(point - 25) / 3];
            j = columns.start + c[(point - 25) % 3];
        }
        if (i < rows.start || i >= rows.start + rows.size ||
            j < columns.start || j >= columns.start + columns.size) continue;
        double expected = 0.0;
        for (int k = 0; k < N; ++k) {
            expected += getPseudoRndValue(size_t(N), size_t(i), size_t(k)) *
                        getPseudoRndValue(size_t(N), size_t(k), size_t(j));
        }
        const double actual = C[size_t(i - rows.start) * columns.size + j - columns.start];
        const double error = std::abs(actual - expected) / (std::abs(expected) + 1e-10);
        if (!std::isfinite(actual) || error > 1e-6) {
            failed = 1;
            #pragma omp critical
            std::printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, error);
        }
    }
    int globalFailed = 0;
    checkMPI(MPI_Allreduce(&failed, &globalFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    return globalFailed == 0;
}

void printDistributedResults(const std::vector<double>& local, int N, int ranks,
                             const int* dims, Block rows, Block columns) {
    // Gather only for -r. Derived types place each tile directly in row-major order
    // and allow more than INT_MAX elements without a packed root-side copy.
    if (worldRank == 0) {
        std::vector<double> result(size_t(N) * N);
        std::vector<MPI_Request> requests;
        requests.reserve(size_t(ranks - 1));
        for (int rank = 1; rank < ranks; ++rank) {
            const Block r = partition(N, rank / dims[1], dims[0]);
            const Block c = partition(N, rank % dims[1], dims[1]);
            if (!r.size || !c.size) continue;
            const int sizes[] = {N, N};
            const int subsizes[] = {r.size, c.size};
            const int starts[] = {r.start, c.start};
            MPI_Datatype tile;
            checkMPI(MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                              MPI_ORDER_C, MPI_DOUBLE, &tile));
            checkMPI(MPI_Type_commit(&tile));
            MPI_Request request;
            checkMPI(MPI_Irecv(result.data(), 1, tile, rank, 0, MPI_COMM_WORLD, &request));
            requests.push_back(request);
            checkMPI(MPI_Type_free(&tile));
        }
        #pragma omp parallel for schedule(static) num_threads(workerThreads(local.size()))
        for (int i = 0; i < rows.size; ++i) {
            std::copy_n(local.data() + size_t(i) * columns.size, columns.size,
                        result.data() + size_t(rows.start + i) * N + columns.start);
        }
        if (!requests.empty()) {
            checkMPI(MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                                 MPI_STATUSES_IGNORE));
        }
        print_results(result, "MatrixC");
    } else if (rows.size && columns.size) {
        MPI_Datatype tile;
        checkMPI(MPI_Type_vector(rows.size, columns.size, columns.size, MPI_DOUBLE, &tile));
        checkMPI(MPI_Type_commit(&tile));
        checkMPI(MPI_Send(local.data(), 1, tile, 0, 0, MPI_COMM_WORLD));
        checkMPI(MPI_Type_free(&tile));
    }
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int runBenchmark(int N, bool validate, bool printResults, int ranks) {
    MPI_Comm sharedComm;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                 MPI_INFO_NULL, &sharedComm));
    int localRank = 0, localRanks = 0;
    checkMPI(MPI_Comm_rank(sharedComm, &localRank));
    checkMPI(MPI_Comm_size(sharedComm, &localRanks));
    int devices = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&devices);
    if (deviceStatus == cudaErrorNoDevice || (deviceStatus == cudaSuccess && !devices)) {
        fail("MPI/OpenMP/CUDA execution requires a visible CUDA device on every rank.");
    }
    checkCUDA(deviceStatus);
    // One MPI rank per GPU is the intended cluster layout. This also respects
    // schedulers that expose only a single, different GPU to each process.
    checkCUDA(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localRanks));
    }
    checkMPI(MPI_Comm_free(&sharedComm));

    int dims[] = {0, 0};
    checkMPI(MPI_Dims_create(ranks, 2, dims));
    const int row = worldRank / dims[1];
    const int column = worldRank % dims[1];
    const Block rows = partition(N, row, dims[0]);
    const Block columns = partition(N, column, dims[1]);
    MPI_Comm rowComm, columnComm;
    checkMPI(MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm));
    checkMPI(MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm));
    checkMPI(MPI_Comm_set_errhandler(rowComm, MPI_ERRORS_RETURN));
    checkMPI(MPI_Comm_set_errhandler(columnComm, MPI_ERRORS_RETURN));

    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %d x %d\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing matrices...\n");
    }
    const int panelWidth = std::min(N, 2048);
    const int steps = (N - 1) / panelWidth + 1;
    Accelerator gpu(size_t(rows.size) * columns.size);
    Panel panels[] = {{rows.size, columns.size, panelWidth},
                      {rows.size, columns.size, panelWidth}};
    initializePanel(panels[0], N, rows, columns, 0, panelWidth, column == 0, row == 0);
    if (worldRank == 0) std::printf("Computing matrix multiplication...\n");
    checkMPI(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    broadcastPanel(panels[0], rows, columns, panelWidth, 0, 0, rowComm, columnComm);

    // Pipelined 2-D SUMMA: broadcast A along process rows and B along columns.
    // While the GPU consumes a panel, OpenMP prepares and MPI exchanges the next.
    for (int step = 0; step < steps; ++step) {
        const int kStart = step * panelWidth;
        const int width = std::min(panelWidth, N - kStart);
        multiplyPanel(panels[step % 2], gpu, rows, columns, width, step == 0);
        if (step + 1 < steps) {
            Panel& next = panels[(step + 1) % 2];
            // Protect both the host transfer source and device GEMM operands.
            if (next.used) checkCUDA(cudaEventSynchronize(next.consumed));
            const int nextStart = kStart + panelWidth;
            const int nextWidth = std::min(panelWidth, N - nextStart);
            const int aOwner = (step + 1) % dims[1];
            const int bOwner = (step + 1) % dims[0];
            initializePanel(next, N, rows, columns, nextStart, nextWidth,
                            column == aOwner, row == bOwner);
            broadcastPanel(next, rows, columns, nextWidth, aOwner, bOwner, rowComm, columnComm);
        }
    }
    checkCUDA(cudaStreamSynchronize(gpu.compute));
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    checkMPI(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        std::printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double gflops = 2.0 * double(N) * double(N) * double(N) /
                              std::max(duration, std::numeric_limits<double>::min()) / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> localResult;
    if (validate || printResults) {
        localResult.resize(size_t(rows.size) * columns.size);
        if (!localResult.empty()) {
            checkCUDA(cudaMemcpy(localResult.data(), gpu.C.data,
                                  localResult.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
    if (printResults) printDistributedResults(localResult, N, ranks, dims, rows, columns);
    bool valid = true;
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        valid = validateResult(localResult, N, rows, columns);
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    checkMPI(MPI_Comm_free(&rowComm));
    checkMPI(MPI_Comm_free(&columnComm));
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    checkMPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI must support MPI_THREAD_FUNNELED.");
    int ranks = 1;
    checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int N = 512;
    // status: 0 = run, 1 = error, 2 = help. Parse once for consistent rank behavior.
    int options[] = {0, 0, 0};
    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long parsed = std::strtoull(value, &end, 10);
                if (errno || end == value || *end || *value == '-' || !parsed || parsed > INT_MAX ||
                    parsed > static_cast<unsigned long long>(PTRDIFF_MAX) / sizeof(double) / parsed) {
                    std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                    options[2] = 1;
                    break;
                }
                N = static_cast<int>(parsed);
            } else if (std::strcmp(argv[i], "-v") == 0) {
                options[0] = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                options[1] = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                options[2] = 2;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                options[2] = 1;
                break;
            }
        }
        if (options[2]) printUsage(argv[0]);
    }
    checkMPI(MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD));
    checkMPI(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    int status = options[2] == 1 ? 1 : 0;
    if (options[2] == 0) {
        try {
            status = runBenchmark(N, options[0] != 0, options[1] != 0, ranks);
        } catch (const std::exception& error) {
            fail(error.what());
        }
    }
    checkMPI(MPI_Finalize());
    return status;
}
