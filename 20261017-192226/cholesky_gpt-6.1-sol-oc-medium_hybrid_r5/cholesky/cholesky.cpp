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

// One MPI rank per GPU, with block-cyclic columns stored column-major.
// Only the current panel is replicated during the factorization. All trailing
// columns on a rank are updated by a single large GEMM, rather than tile-sized
// launches. Host-staged collectives work with ordinary and CUDA-aware MPI alike.
constexpr int blockSize = 256;

static void checkCuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
static void checkBlas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", expression, int(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA(call) checkCuda((call), #call)
#define BLAS(call) checkBlas((call), #call)

struct DeviceBuffer {
    double* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&data),
                        std::max(size_t(1), count) * sizeof(double)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Accelerator {
    cudaStream_t stream;
    cublasHandle_t blas;
    Accelerator() {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        CUDA(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "A CUDA GPU is required on every MPI node.\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        CUDA(cudaSetDevice(localRank % devices));
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        BLAS(cublasCreate(&blas));
        BLAS(cublasSetStream(blas, stream));
        // Keep all matrix arithmetic in FP64, including input generation.
        BLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    }
    ~Accelerator() {
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
};

static void broadcast(double* data, size_t count, int root) {
    // MPI's traditional count argument is an int; large matrices need chunks.
    while (count) {
        int chunk = int(std::min(count, size_t(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

static std::vector<int> ownedColumns(int n, int rank, int ranks) {
    std::vector<int> columns;
    for (long long first = static_cast<long long>(rank) * blockSize;
         first < n; first += static_cast<long long>(ranks) * blockSize)
        for (int j = int(first); j < std::min(first + blockSize, (long long)n); ++j)
            columns.push_back(j);
    return columns;
}

__global__ void addDiagonal(double* a, const int* columns, int n, int nc) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nc) a[size_t(j) * n + columns[j]] += n;
}

// Extract the rows of the panel that correspond to locally owned columns.
// This turns a block-cyclic update into one contiguous matrix multiplication.
__global__ void packPanelRows(const double* panel, double* right,
                              const int* columns, int first, int nc,
                              int k, int height, int width) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(nc) * width;
    for (; t < count; t += size_t(blockDim.x) * gridDim.x) {
        int row = int(t % nc), col = int(t / nc);
        right[t] = panel[size_t(col) * height + columns[first + row] - k];
    }
}

static void generatePositiveDefiniteMatrix(DeviceBuffer& a, int n,
                                           const std::vector<int>& columns,
                                           const int* deviceColumns,
                                           Accelerator& gpu, int rank) {
    // Replay the original rand_r stream twice, rather than replicating an
    // n-by-n B on every GPU/host. First distribute each rank's rows of B;
    // then stream B in bounded slabs to form the local columns of B B^T.
    // Setup and factorization both require O(n^2/ranks + n*slab) memory.
    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int nc = int(columns.size());
    const int slab = std::min(n, 1024);
    double* b;
    CUDA(cudaMallocHost(reinterpret_cast<void**>(&b), size_t(n) * slab * sizeof(double)));
    std::vector<double> selected(size_t(n) * nc);
    unsigned int seed = 42;
    for (int first = 0; first < n; first += blockSize) {
        int width = std::min(blockSize, n - first);
        int owner = (first / blockSize) % ranks;
        size_t count = size_t(width) * n;
        if (rank == 0)
            for (size_t i = 0; i < count; ++i)
                b[i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
        size_t localFirst = size_t(first / blockSize / ranks) * blockSize * n;
        if (owner == 0 && rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < width; ++j)
                std::copy_n(b + size_t(j) * n, n,
                            selected.data() + localFirst + size_t(j) * n);
        } else if (owner != 0) {
            for (size_t offset = 0; offset < count; ) {
                int chunk = int(std::min(count - offset, size_t(INT_MAX)));
                if (rank == 0)
                    MPI_Send(b + offset, chunk, MPI_DOUBLE, owner, 1, MPI_COMM_WORLD);
                else if (rank == owner)
                    MPI_Recv(selected.data() + localFirst + offset, chunk, MPI_DOUBLE,
                             0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    }

    DeviceBuffer db(size_t(n) * slab), ds(selected.size());
    CUDA(cudaMemcpyAsync(ds.data, selected.data(), selected.size() * sizeof(double),
                         cudaMemcpyHostToDevice, gpu.stream));
    cudaEvent_t copied;
    CUDA(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    seed = 42;
    const double one = 1.0, zero = 0.0;
    for (int first = 0; first < n; first += slab) {
        int rows = std::min(slab, n - first);
        size_t count = size_t(n) * rows;
        if (first) CUDA(cudaEventSynchronize(copied));
        if (rank == 0)
            for (size_t i = 0; i < count; ++i)
                b[i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
        broadcast(b, count, 0);
        CUDA(cudaMemcpyAsync(db.data, b, count * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.stream));
        CUDA(cudaEventRecord(copied, gpu.stream));
        if (nc)
            BLAS(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, nc, n,
                              &one, db.data, n, ds.data, n, &zero, a.data + first, n));
    }
    if (nc) {
        addDiagonal<<<(nc + 255) / 256, 256, 0, gpu.stream>>>
            (a.data, deviceColumns, n, nc);
        CUDA(cudaGetLastError());
    }
    CUDA(cudaStreamSynchronize(gpu.stream));
    CUDA(cudaEventDestroy(copied));
    CUDA(cudaFreeHost(b));
}

static bool choleskyDecomposition(DeviceBuffer& a, int n,
                                  const std::vector<int>& columns,
                                  const int* deviceColumns, Accelerator& gpu,
                                  int rank, int ranks) {
    const int nc = int(columns.size());
    DeviceBuffer panel(size_t(n) * blockSize);
    DeviceBuffer right(size_t(nc) * blockSize);
    double* hostPanel;
    CUDA(cudaMallocHost(reinterpret_cast<void**>(&hostPanel),
                        size_t(n) * blockSize * sizeof(double)));
    std::vector<double> diagonal(size_t(blockSize) * blockSize);
    cudaEvent_t panelCopied;
    CUDA(cudaEventCreateWithFlags(&panelCopied, cudaEventDisableTiming));
    const double one = 1.0, minusOne = -1.0;

    for (int k = 0; k < n; k += blockSize) {
        // Other ranks can enter the next collective while their previous
        // trailing update is still running. Only the panel owner must await
        // its update; the event protects the reusable pinned staging buffer.
        if (k) CUDA(cudaEventSynchronize(panelCopied));
        int width = std::min(blockSize, n - k), height = n - k;
        int owner = (k / blockSize) % ranks;
        int first = int(std::lower_bound(columns.begin(), columns.end(), k) - columns.begin());
        int valid = 1;
        if (rank == owner) {
            double* tile = a.data + size_t(first) * n + k;
            CUDA(cudaMemcpy2DAsync(diagonal.data(), width * sizeof(double),
                                   tile, size_t(n) * sizeof(double),
                                   width * sizeof(double), width,
                                   cudaMemcpyDeviceToHost, gpu.stream));
            CUDA(cudaStreamSynchronize(gpu.stream));
            // Small diagonal panel on the CPU; its O(b^3) work is dwarfed by
            // the distributed O(n^3) GPU updates. Transpose to row-major so
            // the inner products are contiguous and can use OpenMP SIMD.
            for (int i = 0; i < width; ++i)
                for (int j = 0; j < i; ++j)
                    std::swap(diagonal[size_t(i) * width + j],
                              diagonal[size_t(j) * width + i]);
            for (int j = 0; j < width; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int p = 0; p < j; ++p) {
                    double v = diagonal[size_t(j) * width + p];
                    sum += v * v;
                }
                double d = diagonal[size_t(j) * width + j] - sum;
                if (!(d > 0.0) || !std::isfinite(d)) {
                    printf("Error: Matrix is not positive definite at diagonal element %d\n", k + j);
                    valid = 0;
                    break;
                }
                double pivot = sqrt(d);
                diagonal[size_t(j) * width + j] = pivot;
                for (int i = j + 1; i < width; ++i) {
                    double dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (int p = 0; p < j; ++p)
                        dot += diagonal[size_t(i) * width + p] * diagonal[size_t(j) * width + p];
                    diagonal[size_t(i) * width + j] =
                        (diagonal[size_t(i) * width + j] - dot) / pivot;
                }
            }
            if (valid) {
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < i; ++j)
                        std::swap(diagonal[size_t(i) * width + j],
                                  diagonal[size_t(j) * width + i]);
                CUDA(cudaMemcpy2DAsync(tile, size_t(n) * sizeof(double),
                                       diagonal.data(), width * sizeof(double),
                                       width * sizeof(double), width,
                                       cudaMemcpyHostToDevice, gpu.stream));
                if (height > width)
                    BLAS(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, height - width,
                                      width, &one, tile, n, tile + width, n));
                CUDA(cudaMemcpy2DAsync(hostPanel, size_t(height) * sizeof(double),
                                       tile, size_t(n) * sizeof(double),
                                       size_t(height) * sizeof(double), width,
                                       cudaMemcpyDeviceToHost, gpu.stream));
                CUDA(cudaStreamSynchronize(gpu.stream));
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!valid) {
            CUDA(cudaStreamSynchronize(gpu.stream));
            CUDA(cudaEventDestroy(panelCopied));
            cudaFreeHost(hostPanel);
            return false;
        }
        broadcast(hostPanel, size_t(height) * width, owner);
        CUDA(cudaMemcpyAsync(panel.data, hostPanel, size_t(height) * width * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.stream));
        CUDA(cudaEventRecord(panelCopied, gpu.stream));
        int trailing = int(std::lower_bound(columns.begin(), columns.end(), k + width) - columns.begin());
        int remaining = nc - trailing;
        if (remaining && height > width) {
            int blocks = int(std::min(size_t(65535), (size_t(remaining) * width + 255) / 256));
            packPanelRows<<<blocks, 256, 0, gpu.stream>>>
                (panel.data, right.data, deviceColumns, trailing, remaining, k, height, width);
            CUDA(cudaGetLastError());
            BLAS(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                              height - width, remaining, width, &minusOne,
                              panel.data + width, height, right.data, remaining,
                              &one, a.data + size_t(trailing) * n + k + width, n));
        }
    }
    CUDA(cudaStreamSynchronize(gpu.stream));
    CUDA(cudaEventDestroy(panelCopied));
    CUDA(cudaFreeHost(hostPanel));
    return true;
}

// Gather only for requested validation/output, never in the factorization.
static std::vector<double> gatherMatrix(const DeviceBuffer& a, int n,
                                       const std::vector<int>& columns,
                                       Accelerator& gpu, int rank, int ranks,
                                       bool lowerOnly) {
    std::vector<double> local(size_t(n) * columns.size());
    CUDA(cudaMemcpyAsync(local.data(), a.data, local.size() * sizeof(double),
                         cudaMemcpyDeviceToHost, gpu.stream));
    CUDA(cudaStreamSynchronize(gpu.stream));
    std::vector<double> result;
    if (rank == 0) result.resize(size_t(n) * n);
    for (int source = 0; source < ranks; ++source) {
        auto cols = ownedColumns(n, source, ranks);
        size_t count = size_t(n) * cols.size();
        std::vector<double> received;
        if (rank == 0 && source != 0) received.resize(count);
        double* buffer = source == 0 ? local.data() : received.data();
        for (size_t offset = 0; offset < count; ) {
            int chunk = int(std::min(count - offset, size_t(INT_MAX)));
            if (rank == source && source != 0)
                MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            else if (rank == 0 && source != 0)
                MPI_Recv(buffer + offset, chunk, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < int(cols.size()); ++j)
                for (int i = 0; i < n; ++i)
                    result[size_t(i) * n + cols[j]] = lowerOnly && i < cols[j]
                        ? 0.0 : buffer[size_t(j) * n + i];
        }
    }
    return result;
}

static bool validateCholesky(const std::vector<double>& l,
                             const std::vector<double>& original, int n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k <= j; ++k)
                sum += l[size_t(i) * n + k] * l[size_t(j) * n + k];
            double error = fabs(sum - original[size_t(i) * n + j]);
            double rel = error / (fabs(original[size_t(i) * n + j]) + 1e-10);
            // Reconstruction is symmetric, but inspect both input halves to
            // preserve the original full-matrix validation criterion.
            double upperError = fabs(sum - original[size_t(j) * n + i]);
            rel = std::max(rel, upperError / (fabs(original[size_t(j) * n + i]) + 1e-10));
            error = std::max(error, upperError);
            if (!std::isfinite(sum)) error = rel = std::numeric_limits<double>::infinity();
            maxError = std::max(maxError, error);
            relError = std::max(relError, rel);
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) printf("Validation failed: relative error too large\n");
    return relError <= 1e-6;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            long long parsed = strtoll(argv[++i], &end, 10);
            if (errno || *end || parsed <= 0 || parsed > INT_MAX - blockSize ||
                size_t(parsed) > std::numeric_limits<size_t>::max() / sizeof(double) / size_t(parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(parsed);
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
    Accelerator gpu;
    auto columns = ownedColumns(n, rank, ranks);
    int* deviceColumns;
    CUDA(cudaMalloc(reinterpret_cast<void**>(&deviceColumns),
                    std::max(size_t(1), columns.size()) * sizeof(int)));
    CUDA(cudaMemcpyAsync(deviceColumns, columns.data(), columns.size() * sizeof(int),
                         cudaMemcpyHostToDevice, gpu.stream));
    DeviceBuffer a(size_t(n) * columns.size());
    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA block size: %d\n",
               ranks, omp_get_max_threads(), blockSize);
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(a, n, columns, deviceColumns, gpu, rank);
    std::vector<double> original;
    if (validate) original = gatherMatrix(a, n, columns, gpu, rank, ranks, false);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(a, n, columns, deviceColumns, gpu, rank, ranks);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA(cudaFree(deviceColumns));
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * seconds * 1e9));
    }
    std::vector<double> result;
    if (printResults || validate)
        result = gatherMatrix(a, n, columns, gpu, rank, ranks, true);
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results(result, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(result, original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Finalize();
    return status;
}
