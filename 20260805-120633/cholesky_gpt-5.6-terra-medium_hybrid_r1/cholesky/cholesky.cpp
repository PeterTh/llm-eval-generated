#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// A rank owns a contiguous range of block rows.  All ranks retain a replicated
// matrix so that a small panel can be factored locally after the MPI exchanges.
// The expensive TRSM/SYRK work is nevertheless partitioned, not replicated.
constexpr int kBlockSize = 128;

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void trsmRows(double* a, size_t n, size_t k, size_t width,
                         size_t firstRow, size_t rowCount) {
    const size_t row = firstRow + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= firstRow + rowCount) return;

    for (size_t col = 0; col < width; ++col) {
        double value = a[row * n + k + col];
        for (size_t p = 0; p < col; ++p)
            value -= a[row * n + k + p] * a[(k + col) * n + k + p];
        a[row * n + k + col] = value / a[(k + col) * n + k + col];
    }
}

__global__ void trailingUpdate(double* a, size_t n, size_t k, size_t width,
                               size_t firstRow, size_t rowCount) {
    const size_t localRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t row = firstRow + localRow;
    const size_t firstCol = k + width;
    const size_t col = firstCol + blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= rowCount || col > row) return;

    double value = a[row * n + col];
    for (size_t p = 0; p < width; ++p)
        value -= a[row * n + k + p] * a[col * n + k + p];
    a[row * n + col] = value;
}

static bool factorPanel(std::vector<double>& a, size_t n, size_t k, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        const size_t row = k + i;
        double diagonal = a[row * n + row];
        for (size_t p = 0; p < i; ++p) diagonal -= a[row * n + k + p] * a[row * n + k + p];
        if (diagonal <= 0.0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", row);
            return false;
        }
        a[row * n + row] = std::sqrt(diagonal);
        for (size_t iBelow = i + 1; iBelow < width; ++iBelow) {
            const size_t belowRow = k + iBelow;
            double value = a[belowRow * n + row];
            for (size_t p = 0; p < i; ++p)
                value -= a[belowRow * n + k + p] * a[row * n + k + p];
            a[belowRow * n + row] = value / a[row * n + row];
        }
    }
    return true;
}

static void allGatherRows(std::vector<double>& a, size_t n, size_t firstRow,
                          size_t rows, int ranks, int rank) {
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = firstRow + rows * static_cast<size_t>(r) / ranks;
        const size_t end = firstRow + rows * static_cast<size_t>(r + 1) / ranks;
        const size_t count = (end - begin) * n;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "Matrix partition exceeds MPI count limit\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(begin * n);
    }
    const size_t begin = firstRow + rows * static_cast<size_t>(rank) / ranks;
    MPI_Allgatherv(a.data() + begin * n, counts[rank], MPI_DOUBLE, a.data(),
                   counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

static bool choleskyHybrid(std::vector<double>& a, size_t n, int rank, int ranks) {
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
    if (devices == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // Global MPI ranks are not necessarily local ranks on a multi-node job.
    // Derive the device ordinal within each node to avoid cross-node aliasing.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice", rank);
    MPI_Comm_free(&nodeComm);

    double* deviceA = nullptr;
    const size_t bytes = n * n * sizeof(double);
    cudaCheck(cudaMalloc(&deviceA, bytes), "cudaMalloc", rank);
    cudaCheck(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice),
              "initial host-to-device copy", rank);

    for (size_t k = 0; k < n; k += kBlockSize) {
        const size_t width = std::min(static_cast<size_t>(kBlockSize), n - k);
        int panelOK = 1;
        if (rank == 0 && !factorPanel(a, n, k, width)) panelOK = 0;
        MPI_Bcast(&panelOK, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!panelOK) {
            cudaFree(deviceA);
            return false;
        }
        MPI_Bcast(a.data() + k * n, static_cast<int>(width * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceA + k * n, a.data() + k * n, width * n * sizeof(double),
                             cudaMemcpyHostToDevice), "panel host-to-device copy", rank);

        const size_t trailingFirst = k + width;
        const size_t trailingRows = n - trailingFirst;
        if (trailingRows == 0) break;
        const size_t localFirst = trailingFirst + trailingRows * static_cast<size_t>(rank) / ranks;
        const size_t localEnd = trailingFirst + trailingRows * static_cast<size_t>(rank + 1) / ranks;
        const size_t localRows = localEnd - localFirst;

        if (localRows != 0) {
            trsmRows<<<static_cast<unsigned>((localRows + 255) / 256), 256>>>(deviceA, n, k, width,
                                                                                localFirst, localRows);
            cudaCheck(cudaGetLastError(), "TRSM kernel launch", rank);
            cudaCheck(cudaDeviceSynchronize(), "TRSM kernel", rank);
            cudaCheck(cudaMemcpy(a.data() + localFirst * n, deviceA + localFirst * n,
                                 localRows * n * sizeof(double), cudaMemcpyDeviceToHost),
                      "TRSM device-to-host copy", rank);
        }
        allGatherRows(a, n, trailingFirst, trailingRows, ranks, rank);
        cudaCheck(cudaMemcpy(deviceA + trailingFirst * n, a.data() + trailingFirst * n,
                             trailingRows * n * sizeof(double), cudaMemcpyHostToDevice),
                  "TRSM exchange host-to-device copy", rank);

        if (localRows != 0) {
            const dim3 threads(32, 8);
            const dim3 blocks(static_cast<unsigned>((trailingRows + threads.x - 1) / threads.x),
                              static_cast<unsigned>((localRows + threads.y - 1) / threads.y));
            trailingUpdate<<<blocks, threads>>>(deviceA, n, k, width, localFirst, localRows);
            cudaCheck(cudaGetLastError(), "trailing-update kernel launch", rank);
            cudaCheck(cudaDeviceSynchronize(), "trailing-update kernel", rank);
            cudaCheck(cudaMemcpy(a.data() + localFirst * n, deviceA + localFirst * n,
                                 localRows * n * sizeof(double), cudaMemcpyDeviceToHost),
                      "trailing-update device-to-host copy", rank);
        }
        allGatherRows(a, n, trailingFirst, trailingRows, ranks, rank);
        cudaCheck(cudaMemcpy(deviceA + trailingFirst * n, a.data() + trailingFirst * n,
                             trailingRows * n * sizeof(double), cudaMemcpyHostToDevice),
                  "trailing exchange host-to-device copy", rank);
    }
    cudaCheck(cudaFree(deviceA), "cudaFree", rank);

    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n; ++row)
        for (size_t col = row + 1; col < n; ++col) a[row * n + col] = 0.0;
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t p = 0; p < n; ++p) sum += b[i * n + p] * b[j * n + p];
            a[i * n + j] = sum;
        }
        a[i * n + i] += n;
    }
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, size_t n) {
    double maxError = 0.0, relativeError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relativeError) schedule(static)
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t p = 0; p <= std::min(i, j); ++p) sum += l[i * n + p] * l[j * n + p];
        const double error = std::fabs(sum - original[i * n + j]);
        maxError = std::max(maxError, error);
        relativeError = std::max(relativeError, error / (std::fabs(original[i * n + j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relativeError);
    return relativeError <= 1e-6;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > std::sqrt(static_cast<double>(std::numeric_limits<int>::max()))) {
        if (rank == 0) std::fprintf(stderr, "Invalid matrix size\n");
        MPI_Finalize(); return 1;
    }
    std::vector<double> a(n * n), original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark (MPI ranks: %d, CUDA + OpenMP)\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",
                    ranks, n, n, validate ? "enabled" : "disabled");
        generatePositiveDefiniteMatrix(a, n);
        if (validate) original = a;
    }
    MPI_Bcast(a.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = choleskyHybrid(a, n, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int exitCode = success ? 0 : 1;
    if (rank == 0) {
        if (!success) std::printf("Cholesky decomposition failed\n");
        else {
            const long milliseconds = static_cast<long>(maximumElapsed * 1000.0);
            std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", milliseconds,
                        (static_cast<double>(n) * n * n / 3.0) / maximumElapsed / 1e9);
            if (printResults) print_results(a, "CholeskyL");
            if (validate) { std::printf("Validating result...\n"); exitCode = validateCholesky(a, original, n) ? 0 : 1; std::printf("Validation: %s\n", exitCode == 0 ? "PASSED" : "FAILED"); }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
