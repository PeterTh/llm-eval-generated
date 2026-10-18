#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// MPI is only called by the main thread (MPI_THREAD_FUNNELED). GPU workers
// terminate the job on a CUDA failure without making MPI calls themselves.
void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, int(status));
        std::abort();
    }
}

// Balanced partitions include the remainder and allow more ranks than rows.
size_t boundary(size_t n, int part, int parts) {
    return (n / parts) * part + std::min(size_t(part), n % parts);
}

// Generate just the input rectangles needed by this GPU. The original size_t
// arithmetic (including its wraparound behavior) is preserved on the device.
__global__ void initBlock(double* data, size_t rows, size_t cols,
                          size_t rowStart, size_t colStart, size_t n) {
    const size_t count = rows * cols;
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += size_t(blockDim.x) * gridDim.x) {
        data[p] = getPseudoRndValue(n, rowStart + p / cols, colStart + p % cols);
    }
}

struct GpuBlock {
    int device = 0;
    size_t row = 0, rows = 0;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

bool validateResult(const double* c, size_t n, size_t row, size_t rows,
                    size_t col, size_t cols) {
    if (!rows || !cols) return true;
    int valid = 1;
    // Retain the original 5x5 checks and also check each distributed block's
    // edges and interior, including rows assigned to different local GPUs.
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int point = 0; point < 50; ++point) {
        const size_t i = point < 25 ? size_t(point / 5) % n
            : row + (rows - 1) * size_t((point - 25) / 5) / 4;
        const size_t j = point < 25 ? size_t(point % 5) % n
            : col + (cols - 1) * size_t((point - 25) % 5) / 4;
        if (i < row || i >= row + rows || j < col || j >= col + cols) continue;
        double expected = 0.0;
        for (size_t k = 0; k < n; ++k)
            expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
        const double actual = c[(i - row) * cols + j - col];
        if (!std::isfinite(actual) ||
            std::abs((actual - expected) / (expected + 1e-10)) > 1e-6) {
            #pragma omp critical
            fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f\n",
                    i, j, expected, actual);
            valid = 0;
        }
    }
    return valid != 0;
}

// Chunk transfers so MPI's int count does not limit the matrix size.
void transfer(double* data, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        if (send) MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, chunk, MPI_DOUBLE, peer, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end || !value ||
                value > INT_MAX || value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = size_t(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, localRanks, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // A single rank uses all visible GPUs; multiple local ranks get disjoint
    // round-robin device sets when there are enough GPUs.
    const int workers = localRank < devices ? (devices - 1 - localRank) / localRanks + 1 : 1;
    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int pr = rank / dims[1], pc = rank % dims[1];
    const size_t row = boundary(N, pr, dims[0]), col = boundary(N, pc, dims[1]);
    const size_t rows = boundary(N, pr + 1, dims[0]) - row;
    const size_t cols = boundary(N, pc + 1, dims[1]) - col;
    double* result = nullptr;
    if (rows && cols)
        cudaCheck(cudaMallocHost(&result, rows * cols * sizeof(double)), "allocate host result");
    std::vector<GpuBlock> blocks(workers);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; local GPU workers: %d\n", dims[0], dims[1], workers);
        printf("Initializing matrices...\n");
    }
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        GpuBlock& g = blocks[w];
        g.device = localRank < devices ? localRank + w * localRanks : localRank % devices;
        g.row = boundary(rows, w, workers);
        g.rows = boundary(rows, w + 1, workers) - g.row;
        if (!g.rows || !cols) continue;
        cudaCheck(cudaSetDevice(g.device), "cudaSetDevice");
        cudaCheck(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "create stream");
        blasCheck(cublasCreate(&g.handle), "create cuBLAS handle");
        blasCheck(cublasSetStream(g.handle, g.stream), "set cuBLAS stream");
        blasCheck(cublasSetMathMode(g.handle, CUBLAS_DEFAULT_MATH), "set double precision math");
        cudaCheck(cudaMalloc(&g.a, g.rows * N * sizeof(double)), "allocate A block");
        cudaCheck(cudaMalloc(&g.b, N * cols * sizeof(double)), "allocate B block");
        cudaCheck(cudaMalloc(&g.c, g.rows * cols * sizeof(double)), "allocate C block");
        const int ablocks = int(std::min(size_t(65535), (g.rows * N + 255) / 256));
        const int bblocks = int(std::min(size_t(65535), (N * cols + 255) / 256));
        initBlock<<<ablocks, 256, 0, g.stream>>>(g.a, g.rows, N, row + g.row, 0, N);
        cudaCheck(cudaGetLastError(), "initialize A");
        initBlock<<<bblocks, 256, 0, g.stream>>>(g.b, N, cols, 0, col, N);
        cudaCheck(cudaGetLastError(), "initialize B");
        cudaCheck(cudaStreamSynchronize(g.stream), "synchronize initialization");
    }
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        GpuBlock& g = blocks[w];
        if (!g.rows || !cols) continue;
        cudaCheck(cudaSetDevice(g.device), "cudaSetDevice");
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A*B is column-major C^T = B^T*A^T.
        blasCheck(cublasDgemm(g.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  int(cols), int(g.rows), int(N), &alpha, g.b, int(cols),
                  g.a, int(N), &beta, g.c, int(cols)), "multiply blocks");
        cudaCheck(cudaMemcpyAsync(result + g.row * cols, g.c,
                  g.rows * cols * sizeof(double), cudaMemcpyDeviceToHost, g.stream), "copy result");
        cudaCheck(cudaStreamSynchronize(g.stream), "synchronize multiplication");
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed / 1e9);
    }
    // Full result assembly is only required for the existing -r interface.
    if (printResults) {
        if (rank != 0) transfer(result, rows * cols, 0, true);
        else {
            std::vector<double> full(N * N), incoming;
            for (int source = 0; source < ranks; ++source) {
                const int sr = source / dims[1], sc = source % dims[1];
                const size_t r0 = boundary(N, sr, dims[0]), c0 = boundary(N, sc, dims[1]);
                const size_t nr = boundary(N, sr + 1, dims[0]) - r0;
                const size_t nc = boundary(N, sc + 1, dims[1]) - c0;
                if (!nr || !nc) continue;
                double* tile = result;
                if (source != 0) {
                    incoming.resize(nr * nc);
                    transfer(incoming.data(), incoming.size(), source, false);
                    tile = incoming.data();
                }
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nr; ++i)
                    std::copy_n(tile + i * nc, nc, full.data() + (r0 + i) * N + c0);
            }
            print_results(full, "MatrixC");
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(result, N, row, rows, col, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        GpuBlock& g = blocks[w];
        if (!g.rows || !cols) continue;
        cudaCheck(cudaSetDevice(g.device), "cudaSetDevice");
        blasCheck(cublasDestroy(g.handle), "destroy cuBLAS handle");
        cudaCheck(cudaFree(g.a), "free A");
        cudaCheck(cudaFree(g.b), "free B");
        cudaCheck(cudaFree(g.c), "free C");
        cudaCheck(cudaStreamDestroy(g.stream), "destroy stream");
    }
    if (result) cudaCheck(cudaFreeHost(result), "free host result");
    MPI_Comm_free(&local);
    MPI_Finalize();
    return valid ? 0 : 1;
}
