#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 2);
    }
}

// A block computes a 32x32 output tile using only 256 threads.  Each thread
// accumulates four rows, improving instruction-level parallelism while shared
// memory makes every global-memory matrix load useful 32 times.
constexpr int TILE = 32;
constexpr int BLOCK_Y = 8;

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t rows,
                                     size_t n) {
    __shared__ double as[TILE][TILE + 1];
    __shared__ double bs[TILE][TILE + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * TILE + ty;
    double sum[4] = {0.0, 0.0, 0.0, 0.0};

    for (size_t kb = 0; kb < n; kb += TILE) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int laneRow = ty + q * BLOCK_Y;
            const size_t row = row0 + static_cast<size_t>(q * BLOCK_Y);
            const size_t k = kb + tx;
            as[laneRow][tx] = (row < rows && k < n) ? A[row * n + k] : 0.0;
            const size_t bk = kb + laneRow;
            bs[laneRow][tx] = (bk < n && col < n) ? B[bk * n + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double bv = bs[k][tx];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                sum[q] = fma(as[ty + q * BLOCK_Y][k], bv, sum[q]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const size_t row = row0 + static_cast<size_t>(q * BLOCK_Y);
        if (row < rows && col < n) C[row * n + col] = sum[q];
    }
}

static void initLocalMatrices(std::vector<double>& a, std::vector<double>& b,
                              size_t n, size_t firstRow, size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(rows); ++i)
        for (size_t j = 0; j < n; ++j)
            a[static_cast<size_t>(i) * n + j] =
                getPseudoRndValue(n, firstRow + static_cast<size_t>(i), j);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j)
            b[static_cast<size_t>(i) * n + j] =
                getPseudoRndValue(n, static_cast<size_t>(i), j);
}

static bool validateResult(const std::vector<double>& b,
                           const std::vector<double>& c, size_t n) {
    int valid = 1;
#pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (int ii = 0; ii < 5; ++ii) {
        for (int jj = 0; jj < 5; ++jj) {
            const size_t i = static_cast<size_t>(ii) % n;
            const size_t j = static_cast<size_t>(jj) % n;
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k)
                expected += getPseudoRndValue(n, i, k) * b[k * n + j];
            const double relError =
                std::abs((c[i * n + j] - expected) / (expected + 1e-10));
            valid &= (relError <= 1e-6);
        }
    }
    return valid != 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>  Matrix size (default: 512)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false;
    bool argsOk = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argsOk &= end && *end == '\0' && value > 0;
            n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else argsOk = false;
    }
    if (help || !argsOk) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argsOk ? 0 : 1;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::numeric_limits<int>::max()) / n) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    // Bind ranks on each node round-robin to its visible accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA accelerator available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);
    std::vector<double> a(rows * n), b(n * n), localC(rows * n);
    std::vector<double> c;
    if (rank == 0) c.resize(n * n);

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Validation: %s\nInitializing matrices...\n",
                    validate ? "enabled" : "disabled");
    }
    initLocalMatrices(a, b, n, firstRow, rows);

    double *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaCheck(cudaMalloc(&da, std::max<size_t>(1, a.size()) * sizeof(double)), "cudaMalloc A", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&db, b.size() * sizeof(double)), "cudaMalloc B", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dc, std::max<size_t>(1, localC.size()) * sizeof(double)), "cudaMalloc C", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B", MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        const dim3 block(TILE, BLOCK_Y);
        const dim3 grid((n + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
        matrixMultiplyKernel<<<grid, block>>>(da, db, dc, rows, n);
        cudaCheck(cudaGetLastError(), "kernel launch", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(localC.data(), dc, localC.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy C", MPI_COMM_WORLD);
    }
    const double elapsed = MPI_Wtime() - start;

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rr = base + (static_cast<size_t>(r) < extra);
        const size_t off = static_cast<size_t>(r) * base +
                           std::min(static_cast<size_t>(r), extra);
        counts[r] = static_cast<int>(rr * n);
        offsets[r] = static_cast<int>(off * n);
    }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE,
                rank == 0 ? c.data() : nullptr, counts.data(), offsets.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double totalTime = 0.0;
    MPI_Reduce(&elapsed, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cudaFree(dc); cudaFree(db); cudaFree(da);
    int result = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", totalTime * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n",
                    (2.0 * static_cast<double>(n) * n * n) / totalTime / 1e9);
        if (printResults) print_results(c, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateResult(b, c, n) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
