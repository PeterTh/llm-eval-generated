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
int worldRank = 0;

[[noreturn]] void fail(const char* operation, const char* reason) {
    std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fail(operation, message);
    }
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status));
}

void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[64];
        std::snprintf(message, sizeof(message), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, message);
    }
}

// Retain the original size_t arithmetic, including its unsigned wraparound.
constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) /
           static_cast<double>(n * n);
}

struct HostBuffer {
    double* data = nullptr;
    explicit HostBuffer(size_t count) {
        if (count) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data),
                                            count * sizeof(double)), "allocate pinned memory");
    }
    ~HostBuffer() { if (data) cudaFreeHost(data); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

struct DeviceBuffer {
    double* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                                        count * sizeof(double)), "allocate device memory");
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Grid {
    MPI_Comm cart = MPI_COMM_NULL, row = MPI_COMM_NULL, col = MPI_COMM_NULL;
    int dims[2] = {0, 0}, coords[2] = {0, 0};
    int rows, cols, rowStart, colStart, size;

    static int extent(int n, int parts, int coordinate) {
        return n / parts + (coordinate < n % parts);
    }
    static int offset(int n, int parts, int coordinate) {
        return coordinate * (n / parts) + std::min(coordinate, n % parts);
    }

    explicit Grid(int n) {
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
        mpiCheck(MPI_Dims_create(size, 2, dims), "MPI_Dims_create");
        int periods[2] = {0, 0};
        // No reordering: cartesian rank zero remains the output rank.
        mpiCheck(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart), "MPI_Cart_create");
        mpiCheck(MPI_Cart_coords(cart, worldRank, 2, coords), "MPI_Cart_coords");
        mpiCheck(MPI_Comm_split(cart, coords[0], coords[1], &row), "create row communicator");
        mpiCheck(MPI_Comm_split(cart, coords[1], coords[0], &col), "create column communicator");
        rows = extent(n, dims[0], coords[0]);
        cols = extent(n, dims[1], coords[1]);
        rowStart = offset(n, dims[0], coords[0]);
        colStart = offset(n, dims[1], coords[1]);
    }
    ~Grid() {
        MPI_Comm_free(&row);
        MPI_Comm_free(&col);
        MPI_Comm_free(&cart);
    }
};

struct Panel {
    HostBuffer a, b;
    DeviceBuffer deviceA, deviceB;
    cudaEvent_t copied, computed;
    bool used = false;
    std::vector<MPI_Request> requests;

    Panel(size_t aCount, size_t bCount)
        : a(aCount), b(bCount), deviceA(aCount), deviceB(bCount) {
        cudaCheck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "create copy event");
        cudaCheck(cudaEventCreateWithFlags(&computed, cudaEventDisableTiming), "create compute event");
    }
    ~Panel() {
        cudaEventDestroy(copied);
        cudaEventDestroy(computed);
    }
};

// Chunk broadcasts rather than narrowing potentially large matrix counts to int.
void broadcast(double* data, size_t count, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    size_t offset = 0;
    do {
        int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        mpiCheck(MPI_Ibcast(data ? data + offset : nullptr, chunk, MPI_DOUBLE, root, comm,
                           &request), "MPI_Ibcast");
        requests.push_back(request);
        offset += chunk;
    } while (offset < count);
}

void copyHost(const double* source, double* destination, size_t count) {
    #pragma omp parallel for schedule(static) if(count >= 262144)
    for (size_t i = 0; i < count; ++i) destination[i] = source[i];
}

// A panels belong cyclically to process columns; B panels to process rows.
// Only owners initialize each input element. Panel storage is column-major
// for cuBLAS, with a compact leading dimension for the final short B panel.
void initMatrices(std::vector<double>& a, std::vector<double>& b, int n, int width,
                  const Grid& grid) {
    const size_t aStride = static_cast<size_t>(grid.rows) * width;
    const size_t bStride = static_cast<size_t>(grid.cols) * width;
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t index = 0; index < a.size(); ++index) {
            size_t panel = (index / aStride) * grid.dims[1] + grid.coords[1];
            size_t element = index % aStride;
            size_t k = panel * width + element / grid.rows;
            a[index] = k < static_cast<size_t>(n)
                ? getPseudoRndValue(n, grid.rowStart + element % grid.rows, k) : 0.0;
        }
        #pragma omp for schedule(static)
        for (size_t index = 0; index < b.size(); ++index) {
            size_t panel = (index / bStride) * grid.dims[0] + grid.coords[0];
            size_t element = index % bStride;
            size_t kStart = panel * width;
            size_t kCount = std::min(static_cast<size_t>(width), static_cast<size_t>(n) - kStart);
            b[index] = element < kCount * grid.cols
                ? getPseudoRndValue(n, kStart + element % kCount,
                                    grid.colStart + element / kCount) : 0.0;
        }
    }
}

double matrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                      DeviceBuffer& c, int n, int width, const Grid& grid) {
    const size_t aStride = static_cast<size_t>(grid.rows) * width;
    const size_t bStride = static_cast<size_t>(grid.cols) * width;
    Panel panels[2] = {{aStride, bStride}, {aStride, bStride}};
    cudaStream_t transfer, compute;
    cublasHandle_t handle;
    cudaCheck(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "create transfer stream");
    cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "create compute stream");
    blasCheck(cublasCreate(&handle), "create cuBLAS handle");
    blasCheck(cublasSetStream(handle, compute), "set cuBLAS stream");
    const double alpha = 1.0, zero = 0.0, one = 1.0;

    // Initialize the CUDA/cuBLAS execution path outside the timed benchmark.
    if (grid.rows && grid.cols) {
        cudaCheck(cudaMemsetAsync(panels[0].deviceA.data, 0, aStride * sizeof(double), compute),
                  "initialize warmup A");
        cudaCheck(cudaMemsetAsync(panels[0].deviceB.data, 0, bStride * sizeof(double), compute),
                  "initialize warmup B");
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, grid.rows, grid.cols, width,
                             &alpha, panels[0].deviceA.data, grid.rows,
                             panels[0].deviceB.data, width, &zero, c.data, grid.rows), "warmup DGEMM");
        cudaCheck(cudaStreamSynchronize(compute), "finish warmup");
    }

    auto prepare = [&](size_t step, Panel& panel) {
        // The previous H2D copy must finish before the pinned buffers change.
        if (panel.used) cudaCheck(cudaEventSynchronize(panel.copied), "reuse host panel");
        int k = std::min(static_cast<size_t>(width), static_cast<size_t>(n) - step * width);
        size_t aCount = static_cast<size_t>(grid.rows) * k;
        size_t bCount = static_cast<size_t>(grid.cols) * k;
        int aOwner = static_cast<int>(step % grid.dims[1]);
        int bOwner = static_cast<int>(step % grid.dims[0]);
        if (grid.coords[1] == aOwner && aCount)
            copyHost(a.data() + (step / grid.dims[1]) * aStride, panel.a.data, aCount);
        if (grid.coords[0] == bOwner && bCount)
            copyHost(b.data() + (step / grid.dims[0]) * bStride, panel.b.data, bCount);
        panel.requests.clear();
        broadcast(panel.a.data, aCount, aOwner, grid.row, panel.requests);
        broadcast(panel.b.data, bCount, bOwner, grid.col, panel.requests);
    };

    mpiCheck(MPI_Barrier(grid.cart), "benchmark barrier");
    double start = MPI_Wtime();
    const size_t steps = (static_cast<size_t>(n) + width - 1) / width;
    prepare(0, panels[0]);
    for (size_t step = 0; step < steps; ++step) {
        Panel& panel = panels[step % 2];
        mpiCheck(MPI_Waitall(static_cast<int>(panel.requests.size()), panel.requests.data(),
                             MPI_STATUSES_IGNORE), "wait for input panels");
        int k = std::min(static_cast<size_t>(width), static_cast<size_t>(n) - step * width);
        // Protect device buffers independently of the host staging buffers.
        if (panel.used)
            cudaCheck(cudaStreamWaitEvent(transfer, panel.computed, 0), "reuse device panel");
        if (grid.rows)
            cudaCheck(cudaMemcpyAsync(panel.deviceA.data, panel.a.data,
                                      static_cast<size_t>(grid.rows) * k * sizeof(double),
                                      cudaMemcpyHostToDevice, transfer), "upload A panel");
        if (grid.cols)
            cudaCheck(cudaMemcpyAsync(panel.deviceB.data, panel.b.data,
                                      static_cast<size_t>(grid.cols) * k * sizeof(double),
                                      cudaMemcpyHostToDevice, transfer), "upload B panel");
        cudaCheck(cudaEventRecord(panel.copied, transfer), "record upload completion");
        cudaCheck(cudaStreamWaitEvent(compute, panel.copied, 0), "wait for uploads");
        if (grid.rows && grid.cols)
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, grid.rows, grid.cols, k,
                                 &alpha, panel.deviceA.data, grid.rows, panel.deviceB.data, k,
                                 step == 0 ? &zero : &one, c.data, grid.rows), "DGEMM");
        cudaCheck(cudaEventRecord(panel.computed, compute), "record compute completion");
        panel.used = true;
        // MPI progresses on the main thread while CUDA executes this panel.
        if (step + 1 < steps) prepare(step + 1, panels[(step + 1) % 2]);
    }
    cudaCheck(cudaStreamSynchronize(compute), "finish matrix multiplication");
    double elapsed = MPI_Wtime() - start, maximum = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart), "reduce elapsed time");
    blasCheck(cublasDestroy(handle), "destroy cuBLAS handle");
    cudaCheck(cudaStreamDestroy(transfer), "destroy transfer stream");
    cudaCheck(cudaStreamDestroy(compute), "destroy compute stream");
    return maximum;
}

// Preserve the original top-left checks, extended to every distributed block
// and its final element. Only these samples leave the GPU for -v.
bool validateResult(const DeviceBuffer& c, int n, const Grid& grid) {
    int valid = 1;
    if (grid.rows && grid.cols) {
        int checkRows = std::min(5, grid.rows), checkCols = std::min(5, grid.cols);
        std::vector<double> samples(static_cast<size_t>(checkRows) * checkCols + 1);
        cudaCheck(cudaMemcpy2D(samples.data(), checkRows * sizeof(double), c.data,
                               static_cast<size_t>(grid.rows) * sizeof(double),
                               checkRows * sizeof(double), checkCols, cudaMemcpyDeviceToHost),
                  "read validation samples");
        cudaCheck(cudaMemcpy(&samples.back(), c.data + static_cast<size_t>(grid.rows) * grid.cols - 1,
                             sizeof(double), cudaMemcpyDeviceToHost), "read final block element");
        #pragma omp parallel for schedule(static) reduction(&:valid)
        for (size_t sample = 0; sample < samples.size(); ++sample) {
            size_t i = grid.rowStart + (sample + 1 == samples.size()
                       ? grid.rows - 1 : sample % checkRows);
            size_t j = grid.colStart + (sample + 1 == samples.size()
                       ? grid.cols - 1 : sample / checkRows);
            double expected = 0.0;
            for (size_t k = 0; k < static_cast<size_t>(n); ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            double actual = samples[sample];
            double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                valid = 0;
                #pragma omp critical
                std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                            i, j, expected, actual, error);
            }
        }
    }
    int allValid = 0;
    mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, grid.cart), "reduce validation");
    return allValid != 0;
}

// Receive blocks directly into their final positions, without another N*N
// gather buffer. Derived datatypes also avoid MPI's int element-count limit.
void printResult(const DeviceBuffer& c, int n, const Grid& grid) {
    size_t count = static_cast<size_t>(grid.rows) * grid.cols;
    HostBuffer columnMajor(count);
    if (count)
        cudaCheck(cudaMemcpy(columnMajor.data, c.data, count * sizeof(double), cudaMemcpyDeviceToHost),
                  "download result block");
    std::vector<double> result(worldRank == 0 ? static_cast<size_t>(n) * n : count);
    std::vector<MPI_Request> requests;
    if (worldRank == 0) {
        for (int rank = 1; rank < grid.size; ++rank) {
            int coords[2];
            mpiCheck(MPI_Cart_coords(grid.cart, rank, 2, coords), "locate result block");
            int sizes[2] = {n, n};
            int subsizes[2] = {Grid::extent(n, grid.dims[0], coords[0]),
                              Grid::extent(n, grid.dims[1], coords[1])};
            int starts[2] = {Grid::offset(n, grid.dims[0], coords[0]),
                            Grid::offset(n, grid.dims[1], coords[1])};
            if (!subsizes[0] || !subsizes[1]) continue;
            MPI_Datatype block;
            mpiCheck(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                              MPI_DOUBLE, &block), "create result receive type");
            mpiCheck(MPI_Type_commit(&block), "commit result receive type");
            MPI_Request request;
            mpiCheck(MPI_Irecv(result.data(), 1, block, rank, 0, grid.cart, &request), "receive result block");
            requests.push_back(request);
            mpiCheck(MPI_Type_free(&block), "free result receive type");
        }
    }
    size_t stride = worldRank == 0 ? n : grid.cols;
    // Tile the column-major to row-major transpose for cache reuse.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int rowTile = 0; rowTile < (grid.rows + 31LL) / 32; ++rowTile) {
        for (int colTile = 0; colTile < (grid.cols + 31LL) / 32; ++colTile) {
            for (int i = rowTile * 32; i < std::min(grid.rows, rowTile * 32 + 32); ++i)
                for (int j = colTile * 32; j < std::min(grid.cols, colTile * 32 + 32); ++j)
                    result[static_cast<size_t>(i) * stride + j] =
                        columnMajor.data[static_cast<size_t>(j) * grid.rows + i];
        }
    }
    if (worldRank != 0 && count) {
        MPI_Datatype row, block;
        mpiCheck(MPI_Type_contiguous(grid.cols, MPI_DOUBLE, &row), "create result row type");
        mpiCheck(MPI_Type_contiguous(grid.rows, row, &block), "create result send type");
        mpiCheck(MPI_Type_commit(&block), "commit result send type");
        mpiCheck(MPI_Send(result.data(), 1, block, 0, 0, grid.cart), "send result block");
        mpiCheck(MPI_Type_free(&block), "free result send type");
        mpiCheck(MPI_Type_free(&row), "free result row type");
    }
    if (worldRank == 0) {
        mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                  "finish result collection");
        print_results(result, "MatrixC");
    }
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
    std::printf("Launch one MPI rank per GPU; use OMP_NUM_THREADS for host parallelism.\n");
}

int run(int argc, char** argv) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = std::strtoull(argument, &end, 10);
            if (errno || end == argument || *end || argument[0] == '-' || value == 0 ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() / sizeof(double) / value) {
                if (worldRank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<int>(value);
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

    MPI_Comm local;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                 MPI_INFO_NULL, &local), "create node communicator");
    int localRank, localSize, devices;
    mpiCheck(MPI_Comm_rank(local, &localRank), "get node rank");
    mpiCheck(MPI_Comm_size(local, &localSize), "get node size");
    cudaCheck(cudaGetDeviceCount(&devices), "enumerate CUDA devices");
    if (!devices) fail("select CUDA device", "a CUDA GPU is required on every node");
    cudaCheck(cudaSetDevice(localRank % devices), "select node-local GPU");
    mpiCheck(MPI_Comm_free(&local), "free node communicator");
    // Respect explicit OpenMP settings; avoid spawning hundreds of host
    // threads per GPU when the environment does not specify a team size.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localSize)));

    Grid grid(n);
    int targetWidth = static_cast<int>((static_cast<size_t>(n) + std::max(grid.dims[0], grid.dims[1]) - 1) /
                                       std::max(grid.dims[0], grid.dims[1]));
    int width = static_cast<int>(std::min(static_cast<size_t>(n),
                          std::clamp((static_cast<size_t>(targetWidth) + 255) / 256 * 256,
                                     size_t{256}, size_t{2048})));
    size_t steps = (static_cast<size_t>(n) + width - 1) / width;
    auto ownedPanels = [steps](int coordinate, int parts) -> size_t {
        return steps / parts + (static_cast<size_t>(coordinate) < steps % parts);
    };
    std::vector<double> a(ownedPanels(grid.coords[1], grid.dims[1]) * grid.rows * width);
    std::vector<double> b(ownedPanels(grid.coords[0], grid.dims[0]) * grid.cols * width);
    DeviceBuffer c(static_cast<size_t>(grid.rows) * grid.cols);
    if (worldRank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (%d x %d grid), OpenMP threads per rank: %d\n",
                    grid.size, grid.dims[0], grid.dims[1], omp_get_max_threads());
        std::printf("Initializing matrices...\n");
    }
    initMatrices(a, b, n, width, grid);
    if (worldRank == 0) std::printf("Computing matrix multiplication...\n");
    double seconds = matrixMultiply(a, b, c, n, width, grid);
    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * n * n * n / seconds / 1e9);
    }
    if (printResults) printResult(c, n, grid);
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        bool valid = validateResult(c, n, grid);
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "set MPI error handler");
    if (provided < MPI_THREAD_FUNNELED) fail("MPI initialization", "MPI_THREAD_FUNNELED is required");
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fail("matrix multiplication", error.what());
    }
    mpiCheck(MPI_Finalize(), "MPI_Finalize");
    return result;
}
