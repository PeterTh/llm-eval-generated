#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Quotient/remainder partitioning also handles more ranks than matrix rows.
static size_t beginBlock(size_t n, int coordinate, int parts) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

// A block computes 64x64 outputs. Each thread owns a 4x4 register tile;
// cooperative, coalesced loads reuse each input tile across the entire block.
// Do not split K or use reassociation: output hashes depend on summation order.
__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C,
                               size_t rows, size_t cols, size_t n) {
    __shared__ double as[64][16];
    __shared__ double bs[16][64];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    // Grid-striding avoids CUDA's grid-y limit for very tall local matrices.
    for (size_t tileRow = blockIdx.y; tileRow < (rows + 63) / 64; tileRow += gridDim.y) {
        for (size_t tileCol = blockIdx.x; tileCol < (cols + 63) / 64; tileCol += gridDim.x) {
            const size_t row = tileRow * 64;
            const size_t col = tileCol * 64;
            double sum[4][4] = {};
            for (size_t base = 0; base < n; base += 16) {
                #pragma unroll
                for (int t = tid; t < 1024; t += 256) {
                    const int ar = t / 16, ak = t % 16;
                    const int bk = t / 64, bc = t % 64;
                    as[ar][ak] = row + ar < rows && base + ak < n
                        ? A[(row + ar) * n + base + ak] : 0.0;
                    bs[bk][bc] = base + bk < n && col + bc < cols
                        ? B[(base + bk) * cols + col + bc] : 0.0;
                }
                __syncthreads();
                // Avoid even padded additions so signed zero/rounding match.
                for (int k = 0; k < 16 && base + k < n; ++k) {
                    double a[4], b[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) a[i] = as[ty + i * 16][k];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) b[j] = bs[k][tx + j * 16];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) sum[i][j] += a[i] * b[j];
                }
                __syncthreads();
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (row + ty + i * 16 < rows && col + tx + j * 16 < cols)
                        C[(row + ty + i * 16) * cols + col + tx + j * 16] = sum[i][j];
        }
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool validateResult(const std::vector<double>& C, size_t n,
                           size_t row0, size_t col0, size_t rows, size_t cols) {
    // Include the original 5x5 checks and samples at every distributed tile's
    // boundaries, so validation covers all participating GPUs.
    if (!rows || !cols) return true;
    int valid = 1;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int pi = 0; pi < 7; ++pi) {
        for (int pj = 0; pj < 7; ++pj) {
            const size_t i = pi < 5 ? size_t(pi) % n : row0 + (pi == 5 ? 0 : rows - 1);
            const size_t j = pj < 5 ? size_t(pj) % n : col0 + (pj == 5 ? 0 : cols - 1);
            if (i < row0 || i >= row0 + rows || j < col0 || j >= col0 + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            const double actual = C[(i - row0) * cols + j - col0];
            if (!std::isfinite(actual) || std::abs((actual - expected) / (expected + 1e-10)) > 1e-6) {
                #pragma omp critical
                fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f\n",
                        i, j, expected, actual);
                valid = 0;
            }
        }
    }
    return valid != 0;
}

// Transfer contiguous chunks rather than imposing MPI's int count limit on N*N.
static void transfer(double* data, size_t count, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

static int run(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* argument = argv[++i];
            const unsigned long long value = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || end == argument || *end || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > std::numeric_limits<size_t>::max() / sizeof(double) / value) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(local, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: CUDA GPU required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int pr = rank / dims[1], pc = rank % dims[1];
    const size_t row0 = beginBlock(n, pr, dims[0]);
    const size_t col0 = beginBlock(n, pc, dims[1]);
    const size_t rows = beginBlock(n, pr + 1, dims[0]) - row0;
    const size_t cols = beginBlock(n, pc + 1, dims[1]) - col0;
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI process grid: %d x %d; OpenMP threads per rank: %d\n",
               dims[0], dims[1], omp_get_max_threads());
        printf("Initializing matrices...\n");
    }

    std::vector<double> C(validate || printResults ? rows * cols : 0);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaStream_t stream = nullptr;
    if (rows && cols) {
        // Deterministic generation replaces input broadcasts: each process only
        // stores its A row slab and B column slab (O(N^2/sqrt(P)) per rank).
        // Pinned staging permits asynchronous DMA; it is released before GEMM.
        double *A = nullptr, *B = nullptr;
        CUDA_CHECK(cudaMallocHost(&A, rows * n * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&B, n * cols * sizeof(double)));
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < n; ++k) A[i * n + k] = getPseudoRndValue(n, row0 + i, k);
            #pragma omp for schedule(static)
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < cols; ++j) B[k * cols + j] = getPseudoRndValue(n, k, col0 + j);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&dA, rows * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dB, n * cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dC, rows * cols * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dA, A, rows * n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, B, n * cols * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeHost(A));
        CUDA_CHECK(cudaFreeHost(B));
    }
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows && cols) {
        const dim3 grid(static_cast<unsigned>(std::min((cols + 63) / 64, size_t(65535))),
                        static_cast<unsigned>(std::min((rows + 63) / 64, size_t(65535))));
        multiplyKernel<<<grid, dim3(16, 16), 0, stream>>>(dA, dB, dC, rows, cols, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", 2.0 * double(n) * double(n) * double(n) / seconds / 1e9);
    }
    if (rows && cols) {
        if (validate || printResults)
            CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    if (printResults) {
        if (rank) transfer(C.data(), C.size(), 0, true);
        else {
            std::vector<double> full(n * n), received;
            for (int p = 0; p < ranks; ++p) {
                const size_t r0 = beginBlock(n, p / dims[1], dims[0]);
                const size_t c0 = beginBlock(n, p % dims[1], dims[1]);
                const size_t nr = beginBlock(n, p / dims[1] + 1, dims[0]) - r0;
                const size_t nc = beginBlock(n, p % dims[1] + 1, dims[1]) - c0;
                const double* tile = C.data();
                if (p) {
                    received.resize(nr * nc);
                    transfer(received.data(), received.size(), p, false);
                    tile = received.data();
                }
                if (!nr || !nc) continue;
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nr; ++i)
                    std::copy_n(tile + i * nc, nc, full.data() + (r0 + i) * n + c0);
            }
            print_results(full, "MatrixC");
        }
    }
    int valid = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        const int localValid = validateResult(C, n, row0, col0, rows, cols);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
