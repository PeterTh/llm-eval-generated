#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr int TILE = 32;

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status, const char* what, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what,
                     cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

// One rank owns a contiguous stripe of rows.  The kernel is deliberately
// independent of the global row offset, so no extra address arithmetic is in
// the inner loop.
__global__ void matmulKernel(const double* __restrict__ a,
                             const double* __restrict__ b,
                             double* __restrict__ c, int rows, int n) {
    __shared__ double as[TILE][TILE];
    __shared__ double bs[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int ak = base + threadIdx.x;
        const int bk = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < rows && ak < n) ? a[row * n + ak] : 0.0;
        bs[threadIdx.y][threadIdx.x] = (bk < n && col < n) ? b[bk * n + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < n) c[row * n + col] = sum;
}

static void initRows(std::vector<double>& matrix, size_t n, size_t firstRow,
                     size_t rows) {
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(rows); ++local) {
        const size_t i = firstRow + static_cast<size_t>(local);
        double* row = matrix.data() + static_cast<size_t>(local) * n;
        for (size_t j = 0; j < n; ++j) row[j] = getPseudoRndValue(n, i, j);
    }
}

static void initFull(std::vector<double>& matrix, size_t n) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j)
            matrix[static_cast<size_t>(i) * n + j] = getPseudoRndValue(n, i, j);
}

static bool validateResult(const std::vector<double>& a, const std::vector<double>& b,
                           const std::vector<double>& c, size_t n) {
    constexpr size_t points[] = {0, 1, 2, 3, 4};
    for (size_t pi : points) for (size_t pj : points) {
        const size_t i = pi % n, j = pj % n;
        double expected = 0.0;
        for (size_t k = 0; k < n; ++k) expected += a[i*n+k] * b[k*n+j];
        const double actual = c[i*n+j];
        const double error = std::abs((actual - expected) / (expected + 1e-10));
        if (error > 1e-6) {
            std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                        i, j, expected, actual, error);
            return false;
        }
    }
    return true;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix size (default: 512)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArgs = true;
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<size_t>(std::sqrt(std::numeric_limits<int>::max())) || badArgs) {
        if (!rank) { if (badArgs) std::printf("Unknown or incomplete option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    const int ni = static_cast<int>(n);
    const size_t firstRow = n * static_cast<size_t>(rank) / ranks;
    const size_t endRow = n * static_cast<size_t>(rank + 1) / ranks;
    const size_t localRows = endRow - firstRow;
    const int localCount = static_cast<int>(localRows * n);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    if (!rank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n,
                    validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d; CUDA + OpenMP hybrid execution\nInitializing matrices...\n", ranks);
    }
    std::vector<double> localA(localRows * n), b(n * n), localC(localRows * n);
    initRows(localA, n, firstRow, localRows);
    initFull(b, n);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    // A job may have more ranks than matrix rows.  CUDA does not guarantee
    // that a zero-byte allocation succeeds, even though that rank has no work.
    const size_t localBytes = (localCount ? static_cast<size_t>(localCount) : 1) * sizeof(double);
    cudaCheck(cudaMalloc(&deviceA, localBytes), "cudaMalloc(A)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&deviceB, n * n * sizeof(double)), "cudaMalloc(B)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&deviceC, localBytes), "cudaMalloc(C)", MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    cudaCheck(cudaMemcpy(deviceA, localA.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy A", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(deviceB, b.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "copy B", MPI_COMM_WORLD);
    const dim3 block(TILE, TILE), grid((ni + TILE - 1) / TILE, (static_cast<int>(localRows) + TILE - 1) / TILE);
    if (localRows) matmulKernel<<<grid, block>>>(deviceA, deviceB, deviceC, static_cast<int>(localRows), ni);
    cudaCheck(cudaGetLastError(), "matmul kernel launch", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(localC.data(), deviceC, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy C", MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    std::vector<double> c;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); c.resize(n*n); }
    MPI_Gather(&localCount, 1, MPI_INT, rank ? nullptr : counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) for (int r = 1; r < ranks; ++r) displs[r] = displs[r-1] + counts[r-1];
    MPI_Gatherv(localC.data(), localCount, MPI_DOUBLE, rank ? nullptr : c.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
    int result = 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0,
                    (2.0 * n * n * n) / seconds / 1e9);
        if (printResults) print_results(c, "MatrixC");
        if (validate) {
            std::printf("Validating result...\n");
            std::vector<double> a(n*n);
            initFull(a, n);
            result = validateResult(a, b, c, n) ? 0 : 1;
            std::printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
