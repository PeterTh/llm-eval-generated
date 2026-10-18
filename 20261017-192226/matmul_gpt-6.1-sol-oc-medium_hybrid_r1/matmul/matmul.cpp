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

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        // Workers may report CUDA failures; serialize their MPI error calls.
        #pragma omp critical(cuda_failure)
        {
            fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

// Each 256-thread block produces 64x64 outputs with 4x4 register tiles.
// Explicit round-to-nearest operations retain the serial summation semantics.
__global__ void matrixMultiply(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C,
                               size_t rows, size_t cols, size_t N) {
    __shared__ double a[64][16];
    __shared__ double b[16][64];
    const unsigned tid = threadIdx.x;
    const unsigned ty = tid / 16, tx = tid % 16;
    for (size_t tileRow = blockIdx.y; tileRow < (rows + 63) / 64;
         tileRow += gridDim.y) {
        for (size_t tileCol = blockIdx.x; tileCol < (cols + 63) / 64;
             tileCol += gridDim.x) {
            double sum[4][4] = {};
            for (size_t base = 0; base < N; base += 16) {
                #pragma unroll
                for (unsigned t = tid; t < 1024; t += 256) {
                    const size_t ar = tileRow * 64 + t / 16;
                    const size_t ak = base + t % 16;
                    a[t / 16][t % 16] = ar < rows && ak < N ? A[ar * N + ak] : 0.0;
                    const size_t bk = base + t / 64;
                    const size_t bc = tileCol * 64 + t % 64;
                    b[t / 64][t % 64] = bk < N && bc < cols ? B[bk * cols + bc] : 0.0;
                }
                __syncthreads();
                #pragma unroll
                for (unsigned k = 0; k < 16; ++k) {
                    // Do not add padded zeros: preserve even the final partial tile.
                    if (base + k < N) {
                        #pragma unroll
                        for (unsigned i = 0; i < 4; ++i) {
                            const double av = a[ty + i * 16][k];
                            #pragma unroll
                            for (unsigned j = 0; j < 4; ++j)
                                sum[i][j] = __dadd_rn(sum[i][j],
                                    __dmul_rn(av, b[k][tx + j * 16]));
                        }
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const size_t row = tileRow * 64 + ty + i * 16;
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const size_t col = tileCol * 64 + tx + j * 16;
                    if (row < rows && col < cols) C[row * cols + col] = sum[i][j];
                }
            }
            __syncthreads();
        }
    }
}

// Check every distributed block, including its edges and interior.
bool validateResult(const double* C, size_t N, size_t rowBegin,
                    size_t rows, size_t colBegin, size_t cols) {
    if (!rows || !cols) return true;
    const size_t ri[] = {0, 1 % rows, 2 % rows, 3 % rows, 4 % rows, rows / 2, rows - 1};
    const size_t ci[] = {0, 1 % cols, 2 % cols, 3 % cols, 4 % cols, cols / 2, cols - 1};
    int valid = 1;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int pi = 0; pi < 7; ++pi) {
        for (int pj = 0; pj < 7; ++pj) {
            const size_t i = rowBegin + ri[pi], j = colBegin + ci[pj];
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = C[ri[pi] * cols + ci[pj]];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                #pragma omp critical
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                valid = 0;
            }
        }
    }
    return valid != 0;
}

// Balanced partitions without overflowing N * coordinate.
static size_t boundary(size_t N, int coordinate, int parts) {
    return (N / parts) * coordinate + std::min<size_t>(coordinate, N % parts);
}

struct GpuBlock {
    int device = 0;
    size_t offset = 0, rows = 0;
    double *a = nullptr, *b = nullptr, *c = nullptr;
    cudaStream_t stream = nullptr;
};

// Large blocks are transferred in chunks because MPI counts are signed ints.
static void transfer(double* data, size_t count, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 1);
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
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    try {
        // A near-square process grid gives O(N^2/sqrt(P)) input storage per rank.
        // Inputs are reproducible, so generate the needed panels locally instead
        // of broadcasting whole matrices or funneling initialization through rank 0.
        int dims[2] = {0, 0};
        MPI_Dims_create(ranks, 2, dims);
        const int pr = rank / dims[1], pc = rank % dims[1];
        const size_t rowBegin = boundary(N, pr, dims[0]);
        const size_t rows = boundary(N, pr + 1, dims[0]) - rowBegin;
        const size_t colBegin = boundary(N, pc, dims[1]);
        const size_t cols = boundary(N, pc + 1, dims[1]) - colBegin;

        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank = 0, localRanks = 1;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_size(node, &localRanks);
        int devices = 0;
        cudaCheck(cudaGetDeviceCount(&devices), "enumerating devices");
        if (!devices) {
            fprintf(stderr, "Rank %d requires a CUDA GPU\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> assigned;
        // Use all visible GPUs even with a single MPI rank on a node. Respect
        // scheduler-provided CUDA_VISIBLE_DEVICES; share only if oversubscribed.
        for (int d = localRank; d < devices; d += localRanks) assigned.push_back(d);
        if (assigned.empty()) assigned.push_back(localRank % devices);
        MPI_Comm_free(&node);

        double *A = nullptr, *B = nullptr, *C = nullptr;
        if (rank == 0) printf("Initializing matrices...\n");
        if (rows && cols) {
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&A), rows * N * sizeof(double)), "allocating host A");
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&B), N * cols * sizeof(double)), "allocating host B");
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&C), rows * cols * sizeof(double)), "allocating host C");
            #pragma omp parallel
            {
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < rows; ++i)
                    for (size_t k = 0; k < N; ++k)
                        A[i * N + k] = getPseudoRndValue(N, rowBegin + i, k);
                #pragma omp for schedule(static)
                for (size_t k = 0; k < N; ++k)
                    for (size_t j = 0; j < cols; ++j)
                        B[k * cols + j] = getPseudoRndValue(N, k, colBegin + j);
            }
        }

        const int workers = static_cast<int>(std::min(rows, assigned.size()));
        std::vector<GpuBlock> blocks(workers);
        #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static)
        for (int w = 0; w < workers; ++w) {
            auto& g = blocks[w];
            g.device = assigned[w];
            g.offset = boundary(rows, w, workers);
            g.rows = boundary(rows, w + 1, workers) - g.offset;
            if (!cols) continue;
            cudaCheck(cudaSetDevice(g.device), "selecting device");
            cudaCheck(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "creating stream");
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.a), g.rows * N * sizeof(double)), "allocating device A");
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.b), N * cols * sizeof(double)), "allocating device B");
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&g.c), g.rows * cols * sizeof(double)), "allocating device C");
        }

        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static)
        for (int w = 0; w < workers; ++w) {
            auto& g = blocks[w];
            if (!cols) continue;
            cudaCheck(cudaSetDevice(g.device), "selecting device");
            cudaCheck(cudaMemcpyAsync(g.a, A + g.offset * N, g.rows * N * sizeof(double), cudaMemcpyHostToDevice, g.stream), "copying A");
            cudaCheck(cudaMemcpyAsync(g.b, B, N * cols * sizeof(double), cudaMemcpyHostToDevice, g.stream), "copying B");
            const dim3 grid(static_cast<unsigned>(std::min<size_t>((cols + 63) / 64, 65535)),
                            static_cast<unsigned>(std::min<size_t>((g.rows + 63) / 64, 65535)));
            matrixMultiply<<<grid, 256, 0, g.stream>>>(g.a, g.b, g.c, g.rows, cols, N);
            cudaCheck(cudaGetLastError(), "launching matrix multiplication");
            cudaCheck(cudaMemcpyAsync(C + g.offset * cols, g.c, g.rows * cols * sizeof(double), cudaMemcpyDeviceToHost, g.stream), "copying C");
            cudaCheck(cudaStreamSynchronize(g.stream), "completing matrix multiplication");
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
        }

        // Assemble the full row-major result only when external output needs it.
        if (printResults) {
            if (rank != 0) transfer(C, rows * cols, 0, true);
            else {
                std::vector<double> result(N * N), received;
                for (int source = 0; source < ranks; ++source) {
                    const int sr = source / dims[1], sc = source % dims[1];
                    const size_t rb = boundary(N, sr, dims[0]);
                    const size_t nr = boundary(N, sr + 1, dims[0]) - rb;
                    const size_t cb = boundary(N, sc, dims[1]);
                    const size_t nc = boundary(N, sc + 1, dims[1]) - cb;
                    double* data = C;
                    if (source) {
                        received.resize(nr * nc);
                        data = received.data();
                        transfer(data, nr * nc, source, false);
                    }
                    if (!nc) continue;
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < nr; ++i)
                        std::copy_n(data + i * nc, nc, result.data() + (rb + i) * N + cb);
                }
                print_results(result, "MatrixC");
            }
        }

        int valid = 1, allValid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            valid = validateResult(C, N, rowBegin, rows, colBegin, cols);
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        }

        #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static)
        for (int w = 0; w < workers; ++w) {
            auto& g = blocks[w];
            if (!cols) continue;
            cudaCheck(cudaSetDevice(g.device), "selecting device");
            cudaCheck(cudaFree(g.a), "freeing A");
            cudaCheck(cudaFree(g.b), "freeing B");
            cudaCheck(cudaFree(g.c), "freeing C");
            cudaCheck(cudaStreamDestroy(g.stream), "destroying stream");
        }
        if (A) cudaCheck(cudaFreeHost(A), "freeing host A");
        if (B) cudaCheck(cudaFreeHost(B), "freeing host B");
        if (C) cudaCheck(cudaFreeHost(C), "freeing host C");
        MPI_Finalize();
        return allValid ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
