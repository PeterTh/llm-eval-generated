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

// Keep the original unsigned integer arithmetic, including wraparound.
__host__ __device__ constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::abort();
    }
}
static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::abort();
    }
}

__global__ void initMatrix(double* mat, size_t N, size_t firstRow, size_t count) {
    for (size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         p < count; p += static_cast<size_t>(blockDim.x) * gridDim.x) {
        mat[p] = getPseudoRndValue(N, firstRow + p / N, p % N);
    }
}

// This partition avoids overflowing the product of the extent and rank.
static size_t boundary(size_t extent, int rank, int ranks) {
    return (extent / ranks) * rank + std::min(static_cast<size_t>(rank), extent % ranks);
}

struct DeviceWork {
    int device;
    size_t firstRow, rows;
    double *A = nullptr, *B = nullptr, *C = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

static void prepare(DeviceWork& w, size_t N) {
    cudaCheck(cudaSetDevice(w.device));
    cudaCheck(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    blasCheck(cublasCreate(&w.handle));
    blasCheck(cublasSetStream(w.handle, w.stream));
    blasCheck(cublasSetMathMode(w.handle, CUBLAS_DEFAULT_MATH));
    cudaCheck(cudaMalloc(&w.A, w.rows * N * sizeof(double)));
    cudaCheck(cudaMalloc(&w.B, N * N * sizeof(double)));
    cudaCheck(cudaMalloc(&w.C, w.rows * N * sizeof(double)));
    const unsigned blocksA = static_cast<unsigned>(std::min(size_t(65535), (w.rows * N + 255) / 256));
    const unsigned blocksB = static_cast<unsigned>(std::min(size_t(65535), (N * N + 255) / 256));
    initMatrix<<<blocksA, 256, 0, w.stream>>>(w.A, N, w.firstRow, w.rows * N);
    cudaCheck(cudaGetLastError());
    initMatrix<<<blocksB, 256, 0, w.stream>>>(w.B, N, 0, N * N);
    cudaCheck(cudaGetLastError());
    // Resolve cuBLAS's lazy kernel loading during setup, using a small valid
    // product. The timed multiplication overwrites all of C with beta=0.
    const double alpha = 1.0, beta = 0.0;
    blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          static_cast<int>(std::min(N, size_t(32))),
                          static_cast<int>(std::min(w.rows, size_t(32))),
                          static_cast<int>(std::min(N, size_t(32))),
                          &alpha, w.B, static_cast<int>(N), w.A, static_cast<int>(N),
                          &beta, w.C, static_cast<int>(N)));
    cudaCheck(cudaStreamSynchronize(w.stream));
}

static void matrixMultiply(DeviceWork& w, size_t N) {
    cudaCheck(cudaSetDevice(w.device));
    const double alpha = 1.0, beta = 0.0;
    // Row-major C=A*B is column-major C^T=B^T*A^T. No transposes or copies.
    blasCheck(cublasDgemm(w.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          static_cast<int>(N), static_cast<int>(w.rows), static_cast<int>(N),
                          &alpha, w.B, static_cast<int>(N), w.A, static_cast<int>(N),
                          &beta, w.C, static_cast<int>(N)));
    cudaCheck(cudaStreamSynchronize(w.stream));
}

static void release(DeviceWork& w) {
    cudaCheck(cudaSetDevice(w.device));
    cudaCheck(cudaFree(w.A));
    cudaCheck(cudaFree(w.B));
    cudaCheck(cudaFree(w.C));
    blasCheck(cublasDestroy(w.handle));
    cudaCheck(cudaStreamDestroy(w.stream));
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int benchmark(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || !*value || *end || *value == '-' || parsed == 0 ||
                parsed > INT_MAX || parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Matrix size overflows address space\n");
        return 1;
    }

    // Respect CUDA_VISIBLE_DEVICES; share devices only if there are more local
    // MPI processes than visible GPUs. A single process uses every visible GPU.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, localRanks, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: CUDA GPU required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> assigned;
    for (int d = localRank; d < devices; d += localRanks) assigned.push_back(d);
    if (assigned.empty()) assigned.push_back(localRank % devices);
    MPI_Comm_free(&local);

    const size_t firstRow = boundary(N, rank, ranks);
    const size_t rows = boundary(N, rank + 1, ranks) - firstRow;
    const int workers = static_cast<int>(std::min(rows, assigned.size()));
    std::vector<DeviceWork> work(workers);
    for (int d = 0; d < workers; ++d) {
        work[d].device = assigned[d];
        work[d].firstRow = firstRow + boundary(rows, d, workers);
        work[d].rows = boundary(rows, d + 1, workers) - boundary(rows, d, workers);
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static)
    for (int d = 0; d < workers; ++d) prepare(work[d], N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static)
    for (int d = 0; d < workers; ++d) matrixMultiply(work[d], N);
    double elapsed = MPI_Wtime() - start, seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    std::vector<double> localC(printResults ? rows * N : 0);
    int valid = 1;
    #pragma omp parallel for num_threads(std::max(1, workers)) schedule(static) reduction(&:valid)
    for (int d = 0; d < workers; ++d) {
        auto& w = work[d];
        cudaCheck(cudaSetDevice(w.device));
        if (printResults) {
            cudaCheck(cudaMemcpyAsync(localC.data() + (w.firstRow - firstRow) * N,
                                     w.C, w.rows * N * sizeof(double), cudaMemcpyDeviceToHost, w.stream));
            cudaCheck(cudaStreamSynchronize(w.stream));
        }
        if (validate) {
            // The original 25 sample positions, including repeated positions for N<5.
            for (size_t pi = 0; pi < 5; ++pi) {
                size_t i = pi % N;
                if (i < w.firstRow || i >= w.firstRow + w.rows) continue;
                double samples[5];
                cudaCheck(cudaMemcpyAsync(samples, w.C + (i - w.firstRow) * N,
                                         std::min(N, size_t(5)) * sizeof(double),
                                         cudaMemcpyDeviceToHost, w.stream));
                cudaCheck(cudaStreamSynchronize(w.stream));
                for (size_t pj = 0; pj < 5; ++pj) {
                    size_t j = pj % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    double actual = samples[j];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(actual) || error > 1e-6) {
                        fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                i, j, expected, actual, error);
                        valid = 0;
                    }
                }
            }
        }
        release(w);
    }

    if (printResults) {
        std::vector<double> C(rank == 0 ? N * N : 0);
        // Chunked transfers also support matrices exceeding MPI's int count limit.
        constexpr size_t chunk = 1u << 26;
        if (rank == 0) {
            std::copy(localC.begin(), localC.end(), C.begin());
            for (int source = 1; source < ranks; ++source) {
                size_t offset = boundary(N, source, ranks) * N;
                size_t count = (boundary(N, source + 1, ranks) - boundary(N, source, ranks)) * N;
                for (size_t p = 0; p < count; p += chunk)
                    MPI_Recv(C.data() + offset + p, static_cast<int>(std::min(chunk, count - p)),
                             MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            print_results(C, "MatrixC");
        } else {
            for (size_t p = 0; p < localC.size(); p += chunk)
                MPI_Send(localC.data() + p, static_cast<int>(std::min(chunk, localC.size() - p)),
                         MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    int allValid;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate && rank == 0) {
        printf("Validating result...\nValidation: %s\n", allValid ? "PASSED" : "FAILED");
    }
    return allValid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
