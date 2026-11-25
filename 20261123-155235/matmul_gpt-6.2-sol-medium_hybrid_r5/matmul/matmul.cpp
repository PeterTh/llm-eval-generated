#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int checkWidth = 5;

struct Range {
    int first;
    int count;
};

struct Block {
    Range rows;
    Range cols;
};

struct GpuBlock {
    double* inputA = nullptr;
    double* inputB = nullptr;
    double* output = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t multiply = nullptr;
    int strideB = 0;
};

Range partition(int n, int parts, int index) {
    const int base = n / parts;
    const int extra = n % parts;
    return {base * index + std::min(index, extra), base + (index < extra)};
}

Block blockForRank(int n, const int grid[2], int rank) {
    return {partition(n, grid[0], rank / grid[1]), partition(n, grid[1], rank % grid[1])};
}

size_t elements(int rows, int cols) {
    return static_cast<size_t>(rows) * cols;
}

[[noreturn]] void fail(const char* operation, const char* detail) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(operation, cudaGetErrorString(status));
    }
}

void checkBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char detail[48];
        snprintf(detail, sizeof(detail), "cuBLAS status %d", static_cast<int>(status));
        fail(operation, detail);
    }
}

double* allocateDevice(size_t count, const char* operation) {
    void* buffer;
    checkCuda(cudaMalloc(&buffer, count * sizeof(double)), operation);
    return static_cast<double*>(buffer);
}

double getPseudoRndValue(size_t modulus, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % modulus) / static_cast<double>(modulus);
}

void initRow(double* row, size_t i, Range cols, size_t modulus) {
    const size_t factor = (i + 1) * 1299709;
    const size_t first = i + cols.first + 1;
    if (factor > std::numeric_limits<size_t>::max() / (first + cols.count - 1)) {
        for (int j = 0; j < cols.count; ++j) {
            row[j] = getPseudoRndValue(modulus, i, cols.first + j);
        }
        return;
    }

    // The numerator is an arithmetic progression modulo N squared.
    const size_t step = factor % modulus;
    size_t numerator = (factor * first) % modulus;
    for (int j = 0; j < cols.count; ++j) {
        row[j] = numerator / static_cast<double>(modulus);
        numerator += step;
        numerator -= numerator >= modulus ? modulus : 0;
    }
}

void uploadMatrix(double* device, Range rows, Range cols, size_t modulus,
                  double* staging, int stagingRows) {
    for (int first = 0; first < rows.count;) {
        const int count = std::min(stagingRows, rows.count - first);
        const int threads = static_cast<int>(std::clamp<size_t>(
            elements(count, cols.count) / 8192, 1, std::min(count, omp_get_max_threads())));
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int i = 0; i < count; ++i) {
            initRow(staging + elements(i, cols.count), rows.first + first + i, cols, modulus);
        }
        checkCuda(cudaMemcpy(device + elements(first, cols.count), staging,
                             elements(count, cols.count) * sizeof(double), cudaMemcpyHostToDevice),
                  "upload matrix");
        first += count;
    }
}

GpuBlock prepareBlock(int n, Block block, int localRank) {
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "count CUDA devices");
    if (devices == 0) {
        fail("select CUDA device", "no visible CUDA device");
    }
    // Also works when a launcher exposes one assigned GPU per rank.
    checkCuda(cudaSetDevice(localRank % devices), "select CUDA device");

    GpuBlock gpu;
    checkBlas(cublasCreate(&gpu.blas), "create cuBLAS handle");
    const size_t modulus = elements(n, n);
    const bool sharedInput = block.rows.count == n || block.cols.count == n;
    const Range aRows = sharedInput ? Range{0, n} : block.rows;
    const size_t aSize = elements(aRows.count, n);
    const size_t bSize = sharedInput ? 0 : elements(n, block.cols.count);
    gpu.inputA = allocateDevice(aSize, "allocate GPU input A");
    if (!sharedInput) {
        gpu.inputB = allocateDevice(bSize, "allocate GPU input B");
    }
    gpu.output = allocateDevice(elements(block.rows.count, block.cols.count), "allocate GPU output");

    // Limit the pinned staging buffer to chunks of about 64 MiB.
    constexpr size_t stagingLimit = 64 * 1024 * 1024 / sizeof(double);
    const int stagingRows = static_cast<int>(
        std::min<size_t>(n, std::max<size_t>(1, stagingLimit / n)));
    const int aStagingRows = std::min(stagingRows, aRows.count);
    const size_t bStagingSize = sharedInput ? 0 : elements(stagingRows, block.cols.count);
    const size_t stagingSize = std::max(elements(aStagingRows, n), bStagingSize);
    void* pinned;
    checkCuda(cudaMallocHost(&pinned, stagingSize * sizeof(double)), "allocate pinned input buffer");
    double* staging = static_cast<double*>(pinned);
    uploadMatrix(gpu.inputA, aRows, {0, n}, modulus, staging, aStagingRows);

    if (sharedInput) {
        // Both operands have the same values. In a 1D grid the full input
        // contains both required slices, so the slices share one allocation.
        gpu.strideB = n;
    } else {
        uploadMatrix(gpu.inputB, {0, n}, block.cols, modulus, staging, stagingRows);
        gpu.strideB = block.cols.count;
    }

    checkCuda(cudaFreeHost(staging), "release pinned input buffer");
    return gpu;
}

void captureMultiplication(GpuBlock& gpu, int n, Block block) {
    const double alpha = 1.0;
    const double beta = 0.0;
    const double* a = gpu.inputA;
    const double* b = gpu.inputB;
    if (b == nullptr) {
        b = a + block.cols.first;
        a += elements(block.rows.first, n);
    }
    // Prepare the single GEMM before timing so cuBLAS dispatch and first-use
    // CUDA setup are excluded without running a second multiplication.
    checkCuda(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), "create CUDA stream");
    checkBlas(cublasSetStream(gpu.blas, gpu.stream), "set cuBLAS stream");
    checkCuda(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal),
              "begin GEMM capture");
    // cuBLAS sees the row-major buffers as transposes. Compute C^T = B^T A^T.
    checkBlas(cublasDgemm(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                         block.cols.count, block.rows.count, n,
                         &alpha, b, gpu.strideB, a, n, &beta, gpu.output, block.cols.count),
              "capture GPU multiplication");
    cudaGraph_t graph;
    checkCuda(cudaStreamEndCapture(gpu.stream, &graph), "end GEMM capture");
#if CUDART_VERSION >= 12000
    checkCuda(cudaGraphInstantiate(&gpu.multiply, graph, 0), "instantiate GEMM graph");
#else
    checkCuda(cudaGraphInstantiate(&gpu.multiply, graph, nullptr, nullptr, 0),
              "instantiate GEMM graph");
#endif
    checkCuda(cudaGraphDestroy(graph), "release GEMM graph");
    checkCuda(cudaGraphUpload(gpu.multiply, gpu.stream), "upload GEMM graph");
    checkCuda(cudaStreamSynchronize(gpu.stream), "finish preparing GPU multiplication");
}

void releaseBlock(const GpuBlock& gpu) {
    if (gpu.blas != nullptr) {
        checkCuda(cudaGraphExecDestroy(gpu.multiply), "release executable GEMM graph");
        checkBlas(cublasDestroy(gpu.blas), "release cuBLAS handle");
        checkCuda(cudaStreamDestroy(gpu.stream), "release CUDA stream");
        checkCuda(cudaFree(gpu.inputA), "release GPU input A");
        checkCuda(cudaFree(gpu.inputB), "release GPU input B");
        checkCuda(cudaFree(gpu.output), "release GPU output");
    }
}

void downloadBlock(double* destination, int stride, const double* device, int deviceStride,
                   int rows, int cols) {
    checkCuda(cudaMemcpy2D(destination, stride * sizeof(double),
                           device, deviceStride * sizeof(double),
                           cols * sizeof(double), rows, cudaMemcpyDeviceToHost),
              "download GPU output");
}

std::vector<double> gatherResult(const GpuBlock& gpu, int n, Block local, const int grid[2],
                                 int rank, int ranks) {
    std::vector<double> result;
    std::vector<double> staging;
    if (rank == 0) {
        result.resize(elements(n, n));
        if (gpu.output != nullptr) {
            downloadBlock(result.data(), n, gpu.output, local.cols.count,
                          local.rows.count, local.cols.count);
        }
    }

    // Gather directly into row-major order. Bound each message and temporary buffer
    // independently of the matrix size so MPI's int counts do not limit N squared.
    constexpr int messageElements = 8 * 1024 * 1024;
    const int firstPeer = rank == 0 ? 1 : rank;
    const int lastPeer = rank == 0 ? ranks : rank + 1;
    for (int peer = firstPeer; peer < lastPeer; ++peer) {
        const Block block = blockForRank(n, grid, peer);
        if (block.rows.count == 0 || block.cols.count == 0) {
            continue;
        }
        const int rowsPerMessage = std::max(1, messageElements / block.cols.count);
        for (int first = 0; first < block.rows.count;) {
            const int rows = std::min(rowsPerMessage, block.rows.count - first);
            if (rank == 0) {
                MPI_Datatype tile;
                MPI_Type_vector(rows, block.cols.count, n, MPI_DOUBLE, &tile);
                MPI_Type_commit(&tile);
                double* destination = result.data() + elements(block.rows.first + first, n)
                                      + block.cols.first;
                MPI_Recv(destination, 1, tile, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&tile);
            } else {
                staging.resize(elements(rows, block.cols.count));
                downloadBlock(staging.data(), block.cols.count,
                              gpu.output + elements(first, block.cols.count), block.cols.count,
                              rows, block.cols.count);
                MPI_Send(staging.data(), rows * block.cols.count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            first += rows;
        }
    }
    return result;
}

bool validateResult(const GpuBlock& gpu, const std::vector<double>& result, int n, Block block,
                    int rank, bool fullResult) {
    if (n == 0) {
        return true;
    }
    const int width = std::min(checkWidth, n);
    double samples[checkWidth][checkWidth] = {};
    if (fullResult) {
        if (rank == 0) {
            for (int i = 0; i < width; ++i) {
                std::copy_n(result.data() + elements(i, n), width, samples[i]);
            }
        }
    } else {
        double local[checkWidth][checkWidth] = {};
        const int rows = std::min(block.rows.count, width - block.rows.first);
        const int cols = std::min(block.cols.count, width - block.cols.first);
        if (rows > 0 && cols > 0) {
            downloadBlock(&local[block.rows.first][block.cols.first], checkWidth, gpu.output,
                          block.cols.count, rows, cols);
        }
        MPI_Reduce(local, samples, checkWidth * checkWidth, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0) {
        const size_t modulus = elements(n, n);
        for (int i = 0; i < width && valid; ++i) {
            for (int j = 0; j < width; ++j) {
                double expected = 0.0;
                for (int k = 0; k < n; ++k) {
                    expected += getPseudoRndValue(modulus, i, k) * getPseudoRndValue(modulus, k, j);
                }
                const double actual = samples[i][j];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(error) || error > 1e-6) {
                    printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, error);
                    valid = 0;
                    break;
                }
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (errno == ERANGE || end == value || *end || parsed < 0 ||
                parsed > std::numeric_limits<int>::max() ||
                (parsed > 0 && static_cast<size_t>(parsed) >
                    std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid matrix size: %s\n", value);
                }
                return 1;
            }
            n = static_cast<int>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    int grid[2] = {};
    MPI_Dims_create(ranks, 2, grid);
    const Block block = blockForRank(n, grid, rank);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    GpuBlock gpu;
    if (block.rows.count > 0 && block.cols.count > 0) {
        gpu = prepareBlock(n, block, localRank);
        captureMultiplication(gpu, n, block);
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (gpu.output != nullptr) {
        checkCuda(cudaGraphLaunch(gpu.multiply, gpu.stream), "multiply GPU block");
        checkCuda(cudaStreamSynchronize(gpu.stream), "finish GPU multiplication");
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        const double gflops = duration > 0.0 ? (2.0 * n * n * n) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> result;
    if (printResults) {
        result = gatherResult(gpu, n, block, grid, rank, ranks);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        valid = validateResult(gpu, result, n, block, rank, printResults);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    releaseBlock(gpu);
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fail("initialize MPI", "MPI_THREAD_FUNNELED is unavailable");
    }
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fail("run benchmark", error.what());
    }
    MPI_Finalize();
    return status;
}
